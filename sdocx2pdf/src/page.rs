use std::{
    collections::{BTreeSet, HashMap, hash_map::Entry},
    io::Read,
    sync::Arc,
};

use anyhow::Context;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use itertools::{Either, Itertools};
use krilla::{
    color::rgb,
    geom::Transform,
    num::NormalizedF32,
    paint::{Fill, Stroke},
    surface::Surface,
};
use log::{error, info, warn};
use num::ToPrimitive;
use ordered_float::OrderedFloat;
use sdocx::{
    DocObject, MediaStorage, euclid,
    page::{PdfPage as EmbeddedPdfPage, TemplateType, object::InlineObject},
};
use thiserror::Error;

use crate::{
    BasicSplitMode, pdf,
    shape::{NoStyleError, PathDrawingError, ShapeDrawingCtx},
    tool::{EventGroup, Tool},
};

/// Raw SDOCX page units convert to PDF points at an exact 1:2 ratio. Verified against real Samsung
/// Notes "Export to PDF" ground truth: a page declaring raw 1200x1697 has an actual exported
/// /MediaBox of exactly 600x848. This is close to, but is not, an A4-calibrated scale
/// (210mm/297mm land near 1200/1697 raw units by coincidence); mapping the page width to A4 width
/// instead produced pages about 0.4% smaller than the real exporter's, which is enough to visibly
/// misalign thin content once rendered.
const PT_PER_SDOCX_UNIT: f64 = 0.5;

/// Stores loaded PDFs for embeds.
pub struct EmbedMap<'e> {
    /// Maps file names to loaded PDFs.
    pdfs: HashMap<&'e str, krilla::pdf::Pdf>,
}

impl<'e> EmbedMap<'e> {
    pub fn new(document: &'e sdocx::Document, media: &mut MediaStorage) -> anyhow::Result<Self> {
        let mut pdfs = HashMap::new();

        for embed in document
            .pages()
            .iter()
            .flat_map(|page| page.embedded_pdf_pages())
        {
            let emb_pdf_name = embed.file().name();

            let pdf = &*match pdfs.entry(emb_pdf_name) {
                Entry::Occupied(occ) => occ.into_mut(),
                Entry::Vacant(vac) => {
                    let mut pdf_bytes = Vec::new();

                    media
                        .open_file(emb_pdf_name)
                        .with_context(|| format!("failed to open embedded PDF '{emb_pdf_name}'"))?
                        .read_to_end(&mut pdf_bytes)
                        .with_context(|| format!("failed to read embedded PDF '{emb_pdf_name}'"))?;

                    let pdf = krilla::pdf::Pdf::new(pdf_bytes).map_err(|e| {
                        anyhow::anyhow!("failed to parse embedded PDF '{emb_pdf_name}': {e:?}")
                    })?;

                    match pdf
                        .metadata()
                        .title
                        .as_ref()
                        .and_then(|v| str::from_utf8(v).ok())
                    {
                        Some(title) if !title.is_empty() => {
                            info!("Loaded embedded PDF '{emb_pdf_name}' ('{title}')")
                        }
                        _ => info!("Loaded embedded PDF '{emb_pdf_name}'"),
                    }

                    vac.insert(pdf)
                }
            };

            // Ensure that the PDF for this embed actually has the requisite page.
            anyhow::ensure!(
                (embed.page_index() as usize) < pdf.pages().len(),
                "PDF '{emb_pdf_name}' is loaded, but does not have page {}",
                embed.page_index()
            );
        }

        Ok(Self { pdfs })
    }

    fn get_pdf(&self, embed: &'e EmbeddedPdfPage) -> &krilla::pdf::Pdf {
        self.pdfs
            .get(embed.file().name())
            .expect("embed wasn't loaded")
    }

    fn get_crop_box(&self, embed: &'e EmbeddedPdfPage) -> pdf::Box2d {
        let rect = self
            .get_pdf(embed)
            .pages()
            .get(embed.page_index() as usize)
            // Unwrapping is fine here because `new` checks that every embed's page exists.
            .unwrap()
            .crop_box();

        pdf::Box2d::new((rect.x0, rect.y0).into(), (rect.x1, rect.y1).into())
    }

    pub fn into_documents(self) -> HashMap<&'e str, krilla::pdf::PdfDocument> {
        self.pdfs
            .into_iter()
            .map(|(name, pdf)| (name, krilla::pdf::PdfDocument::new(Arc::new(pdf))))
            .collect()
    }
}

trait HasBoundingBox {
    fn bounding_box(&self) -> sdocx::Box2d<f64>;
}

impl HasBoundingBox for DocObject {
    fn bounding_box(&self) -> sdocx::Box2d<f64> {
        self.object_base().rect
    }
}

impl HasBoundingBox for InlineObject {
    fn bounding_box(&self) -> sdocx::Box2d<f64> {
        self.object.bounding_box()
    }
}

impl HasBoundingBox for EmbeddedPdfPage {
    fn bounding_box(&self) -> sdocx::Box2d<f64> {
        self.rect()
    }
}

/// Takes an iterable of input pages (page box + iterable of embedded PDF page crop boxes and
/// destination boxes) and returns a scale to use across all the pages to scale the SDOCX page
/// sizes to PDF page sizes. Returns `None` iff `pages` is empty.
fn compute_uniform_scale<Pages, Embeds>(
    pages: Pages,
) -> Option<euclid::Scale<f64, sdocx::SdocxSpace, pdf::PdfSpace>>
where
    Pages: IntoIterator<Item = (sdocx::Box2d<f64>, Embeds)>,
    Embeds: IntoIterator<Item = (pdf::Box2d, sdocx::Box2d<f64>)>,
{
    // Terminology:
    //  - Input page: A page from an SDOCX document
    //  - Host page: An input page that embeds one or more pages from a PDF document
    //  - Embed: An SDOCX object that renders a page from a PDF in a rectangle on a host page
    //  - Source page: The PDF page in an embed
    //  - Destination box: The rectangle on the host page inside which the source page is rendered
    //  - Output page: A page in the PDF document we are creating
    //
    // The goal here is to find a single scale factor to use to derive the output page sizes from
    // the input page sizes. The output page sizes are physical (PDF) sizes; the input page sizes
    // are in SDOCX units, so they are not physical. We have to try to bridge that gap by assigning
    // a physical meaning to the SDOCX units in the context of this document.
    //
    // If there are embeds in the document, their source pages' crop boxes are transformed to fit
    // in their destination boxes. This is a mapping from PDF-space to SDOCX-space. If we obtain
    // the size of the output page by applying the inverse of this transformation's scale to the
    // size of the input page, the source page will appear on the output page at the size it has in
    // its source document. This is a desirable property because it means that if the destination
    // boxes are exactly the same sizes as the host pages (which is what happens if the input
    // document was created by importing a PDF to annotate), each page of the output PDF containing
    // a source page will be exactly the same size as that source page. In other words, an SDOCX
    // file created by importing an {A1, A2, A3, A4, A5, A6, US Letter, ...} PDF and annotating it
    // will be converted into another {A1, A2, A3, A4, A5, A6, US Letter, ...} PDF.
    //
    // There are some issues we have to consider when implementing this approach:
    //  1. There may be no embeds in the whole document, in which case we have no direct link
    //     between PDF units and SDOCX units.
    //  2. There may be many embeds in the whole document, in which case applying the logic
    //     described above to all embeds could give many conflicting opinions on what the scale
    //     should be.
    //  3. It might not be sensible to scale everything up so that source pages end up at their
    //     real size in the output. For example, if an embed is used as a graphic, the size of the
    //     destination box will likely have little to do with the size of the source page, so we
    //     shouldn't treat them as related.
    //  4. The input page size -> output page size mapping must preserve aspect ratio, so an embed
    //     with a difference between the aspect ratios of the source page and the destination box
    //     is not helpful. (I haven't actually seen this happen, but it's technically possible
    //     within the format.)
    //
    // We implement the following solutions to these problems:
    //  1. If there are no useful embeds (see (3) and (4) below) in the whole document, use the
    //     same fixed scale that Samsung Notes' own "Export to PDF" uses (`PT_PER_SDOCX_UNIT`), so
    //     every output page has the size the app itself would export it at.
    //  2. Consider all useful embeds (see below) individually. Break ties by earliest appearance,
    //     first by page index and then by index in the array of embeds if comparing two embeds
    //     from the same host page. Of the useful embeds, choose the best per the criteria
    //     described in (3). If we have a clear winner, then we have no problem; if we have more
    //     than one winner, we achieve a stable result by way of the tiebreak, regardless of
    //     whether the candidates come from the same or different host pages.
    //  3. Ignore any embed whose destination box is not comparable in size and shape to its host
    //     page. That is, if the destination box and host page have very different aspect ratios or
    //     areas, ignore the embed. When looking for winners amongst embeds that pass this test and
    //     the one in (4), sort first by smallest difference in area between the destination box
    //     and the host page (for a better size match) and then by most similar aspect ratio
    //     (looking for near-exact matches).
    //  4. Consider only embeds whose source pages and destination boxes have the same aspect ratio
    //     (up to a very small tolerance). This will normally include all embeds, because the
    //     destination box usually has the same aspect ratio as the source page, but it means we
    //     can get a uniform scale by taking either source page width / destination box width or
    //     source page height / destination box height (equivalently, since we know them to be
    //     almost or exactly equal).

    let mut pages = pages.into_iter().peekable();

    // If there are no useful embeds, we fall back to a fixed scale, but only if there are any
    // pages at all.
    let any_pages = pages.peek().is_some();

    fn aspect_ratio_factor(a: f64, b: f64) -> f64 {
        // This is the factor by which one box is wider (relative to its height) than the other. No
        // need for `abs` because we must have `a >= b` or `a <= b`, so the max here will be at
        // least 1.
        (a / b).max(b / a) - 1.0
    }

    let scale_candidates = pages.flat_map(|(ipb, embeds)| {
        // Save an indentation level:
        let input_page_box = ipb;

        let host_aspect_ratio = input_page_box.width() / input_page_box.height();
        let host_area = input_page_box.area();

        let extract_candidates = move |(src_box, dest_box): (pdf::Box2d, sdocx::Box2d<f64>)| {
            let dest_aspect_ratio = dest_box.width() / dest_box.height();
            let dest_area = dest_box.area();

            let dest_host_ar_factor = aspect_ratio_factor(dest_aspect_ratio, host_aspect_ratio);

            if dest_host_ar_factor > 0.1 {
                // Rule 3: One box is more than 10% wider relative to its height than the
                // other, so the destination box and host page disagree on shape.
                return None;
            }

            let dest_host_area_abs_rel_diff = (dest_area - host_area).abs() / host_area;

            if dest_host_area_abs_rel_diff > 0.35 {
                // Rule 3: The magnitude of the relative difference between the areas of the
                // destination box and host page is greater than 35%, so the destination box
                // does not fill the host page well (it is too big or too small).
                return None;
            }

            let src_aspect_ratio = src_box.width() / src_box.height();
            let dest_src_ar_factor = aspect_ratio_factor(dest_aspect_ratio, src_aspect_ratio);

            if dest_src_ar_factor > 0.01 {
                // Rule 4: Dest/source box is more than 1% wider relative to its height than
                // the other.
                return None;
            }

            // Per (4), we can take width/width or height/height and it makes little
            // difference.
            let proposed_scale =
                pdf::Length::new(src_box.height()) / sdocx::Length::new(dest_box.height());

            Some((
                proposed_scale,
                // Sort key, per (3):
                (
                    OrderedFloat(dest_host_area_abs_rel_diff),
                    OrderedFloat(dest_host_ar_factor),
                ),
            ))
        };

        embeds.into_iter().filter_map(extract_candidates)
    });

    // Find the best scale candidate according to (3). `min_by_key` retains an earlier minimum if
    // an equal element is found later, so this respects the tiebreak rules in (2).
    match scale_candidates.min_by_key(|(_ps, key)| *key) {
        Some((scale, _sort_key)) => Some(scale),
        None => {
            // We didn't find a sensible scale by inspecting embeds, so use the fixed scale (if
            // there were any pages at all).
            any_pages.then(|| euclid::Scale::new(PT_PER_SDOCX_UNIT))
        }
    }
}

/// Returns an iterator yielding the vertical positions of the page breaks that should be used to
/// split `page` according to the given splitting options. Each pair of adjacent break positions in
/// the returned iterator forms a single page slice. The returned iterator is guaranteed to yield
/// at least two elements.
fn compute_page_breaks(
    page: &sdocx::page::Page,
    auto_split: bool,
    basic_split_mode: Option<BasicSplitMode>,
) -> impl Iterator<Item = f64> {
    let page_height = page.height as f64;

    if !auto_split || page.embedded_pdf_pages().is_empty() {
        // Auto-splitting wasn't requested, or is impossible because there are no embeds.
        return Either::Left(match basic_split_mode {
            // No basic split mode set, so use the whole page.
            None => Either::Left([0.0, page_height].into_iter()),

            Some(bsm) => {
                let output_page_inv_aspect_ratio = match bsm {
                    // W:H = 1:√2 => H:W = √2:1
                    BasicSplitMode::IsoPortrait => std::f64::consts::SQRT_2,
                    // W:H = √2:1 => H:W = 1:√2
                    BasicSplitMode::IsoLandscape => std::f64::consts::FRAC_1_SQRT_2,
                };

                // Pageless documents are vertical, so the height is arbitrary. Use the width and
                // the ISO 216 aspect ratio to determine how tall the page slices should be.
                let slice_height = (page.width as f64) * output_page_inv_aspect_ratio;

                // Round the slice count up so that if the slice height doesn't perfectly divide
                // the height of the page, we extend the page to add a bit of blank space in the
                // final slice rather than dropping the final slice and potentially some page
                // content.
                let slice_count: u32 = (page_height / slice_height).ceil().to_u32().unwrap();

                // Inclusive range so that we get the page breaks at the top and bottom of the
                // document.
                Either::Right((0..=slice_count).map(move |i| slice_height * (i as f64)))
            }
        });
    }

    // Use a set so we can ignore exact duplicates; use a B-tree set so we can keep the order of
    // the breaks.
    let mut proposed_page_breaks = BTreeSet::from([
        // Always start a new page at the beginning of the document, and finish a page at the end
        // of the document.
        OrderedFloat(0.0),
        OrderedFloat(page_height),
    ]);

    for embed in page.embedded_pdf_pages() {
        let min_y = embed.rect().min.y;
        let max_y = embed.rect().max.y;

        // Add page breaks before and after this embed.
        proposed_page_breaks.extend([OrderedFloat(min_y), OrderedFloat(max_y)]);
    }

    let minimum_page_height = {
        // Find the median height of the pages that would be created by our breaks if we used all
        // of them, and use that median to obtain a minimum allowed page height. We have to do this
        // rather than setting some constant threshold because we don't know the scale of the page
        // height units.
        let mut proposed_page_heights = proposed_page_breaks
            .iter()
            .tuple_windows()
            .map(|(a, b)| b - a)
            // Collect and sort. This is not the most efficient way to find the median, but it's
            // fine for this.
            .collect_vec();
        proposed_page_heights.sort_unstable();

        // Indexing directly is fine here because the vector can't be empty, so the floored result
        // of halving the length is guaranteed to be a valid index.
        let median_page_height = proposed_page_heights[proposed_page_heights.len() / 2].0;

        // Only permit pages that are no shorter than 5% of the median page height.
        median_page_height * 0.05
    };

    Either::Right(
        proposed_page_breaks
            .into_iter()
            .map(OrderedFloat::into_inner)
            .with_position()
            // Manipulate adjacent page breaks to avoid having any pages below the minimum height.
            // We have total freedom with how we place page breaks within the page, but we can't
            // move or ignore the breaks at the beginning and end of the page.
            .coalesce(
                move |earlier @ (earlier_pos, earlier_break), later @ (later_pos, later_break)| {
                    if later_break - earlier_break >= minimum_page_height {
                        // Page height is fine. Keep both breaks.
                        return Err((earlier, later));
                    }

                    use itertools::Position::{First, Last, Middle};

                    // We can't get here if there is only one page, because that page's height
                    // would be the median height, so can't be below the threshold. Therefore,
                    // neither position is Only. The pair can't be (First, First) or (Last, Last),
                    // because we're looking at distinct breaks. It can't be (First, Last) because
                    // that would mean we'd dropped all breaks between the start and end of the
                    // document and then decided (per the above condition) that the document height
                    // is much less than the median page height, which is impossible.
                    match (earlier_pos, later_pos) {
                        (First, Middle) => {
                            // We can't drop the earlier break because that's the break at the
                            // start of the document. The later break isn't the one at the end of
                            // the document, so we can drop it.
                            Ok(earlier)
                        }

                        (Middle, Middle) => {
                            // Take the midpoint of the two breaks, replacing both.
                            Ok((Middle, earlier_break.midpoint(later_break)))
                        }

                        (Middle, Last) => {
                            // We can't drop the later break because it's the one at the end of the
                            // document, so drop the earlier one.
                            Ok(later)
                        }

                        // As described above, we shouldn't get here, but just in case:
                        _wtf => Err((earlier, later)),
                    }
                },
            )
            .map(|(_, page_break)| page_break),
    )
}

pub struct PageCapture<'p> {
    /// The rectangle that was used to capture the parent page's content.
    capture_box: sdocx::Box2d<f64>,

    /// The background colour of the parent page, if there is one.
    background_colour: Option<[u8; 4]>,

    /// The background template (paper style) of the parent page, if there is one.
    template_type: Option<&'p TemplateType>,

    /// The embedded PDF pages captured.
    embeds: Vec<&'p EmbeddedPdfPage>,

    /// The normal objects captured.
    objects: Vec<&'p DocObject>,

    /// The inline objects captured.
    inline_objects: Vec<&'p InlineObject>,
}

pub enum PageContent<'p> {
    Slice(
        PageCapture<'p>,
        euclid::Scale<f64, sdocx::SdocxSpace, pdf::PdfSpace>,
    ),
    WholePage {
        page: &'p sdocx::page::Page,
        inline_objects: Vec<&'p InlineObject>,
        scale: euclid::Scale<f64, sdocx::SdocxSpace, pdf::PdfSpace>,
    },
}

impl<'p> PageContent<'p> {
    fn input_box(&self) -> sdocx::Box2d<f64> {
        match self {
            PageContent::Slice(page_capture, ..) => page_capture.capture_box,
            PageContent::WholePage { page, .. } => {
                sdocx::Box2d::from_size(sdocx::Size2d::new(page.width as f64, page.height as f64))
            }
        }
    }

    fn input_to_output_scale(&self) -> euclid::Scale<f64, sdocx::SdocxSpace, pdf::PdfSpace> {
        match self {
            PageContent::Slice(.., scale) => *scale,
            PageContent::WholePage { scale, .. } => *scale,
        }
    }

    pub fn output_size(&self) -> pdf::Size2d {
        self.input_box().size() * self.input_to_output_scale()
    }

    pub fn background_colour(&self) -> Option<[u8; 4]> {
        match self {
            PageContent::Slice(page_capture, ..) => page_capture.background_colour,
            PageContent::WholePage { page, .. } => page.background_colour(),
        }
    }

    pub fn template_type(&self) -> Option<&'p TemplateType> {
        match self {
            PageContent::Slice(page_capture, ..) => page_capture.template_type,
            PageContent::WholePage { page, .. } => page.template_type(),
        }
    }

    pub fn embeds(&self) -> impl Iterator<Item = &'p EmbeddedPdfPage> {
        match self {
            PageContent::Slice(page_capture, ..) => {
                Either::Left(page_capture.embeds.iter().copied())
            }

            PageContent::WholePage { page, .. } => Either::Right(page.embedded_pdf_pages().iter()),
        }
    }

    pub fn objects(&self) -> impl ExactSizeIterator<Item = &'p DocObject> {
        match self {
            PageContent::Slice(page_capture, ..) => {
                Either::Left(page_capture.objects.iter().copied())
            }

            PageContent::WholePage { page, .. } => Either::Right({
                // Wrapper type so we can return an `ExactSizeIterator`.
                struct It<'p, I: Iterator<Item = &'p DocObject>>(I, usize);

                impl<'p, I: Iterator<Item = &'p DocObject>> Iterator for It<'p, I> {
                    type Item = &'p DocObject;

                    fn next(&mut self) -> Option<Self::Item> {
                        self.0.next()
                    }

                    fn size_hint(&self) -> (usize, Option<usize>) {
                        (self.1, Some(self.1))
                    }
                }

                impl<'p, I: Iterator<Item = &'p DocObject>> ExactSizeIterator for It<'p, I> {}

                let objects = page
                    .layers()
                    .iter()
                    .flat_map(|layer| layer.objects().iter());

                let n = page.layers().iter().map(|l| l.objects().len()).sum();

                It(objects, n)
            }),
        }
    }

    pub fn inline_objects(&self) -> impl Iterator<Item = &'p InlineObject> {
        match self {
            PageContent::Slice(page_capture, ..) => page_capture.inline_objects.iter().copied(),
            PageContent::WholePage { inline_objects, .. } => inline_objects.iter().copied(),
        }
    }
}

fn group_inline_objects_by_page(document: &sdocx::Document) -> Vec<Vec<&InlineObject>> {
    let mut by_page = vec![Vec::new(); document.pages().len()];

    let pageless = matches!(document.page_model(), sdocx::PageModel::Pageless);

    for inline_obj in document.body_text().inline_objects() {
        match inline_obj.object.page_index() {
            Some(i) => by_page[i as usize].push(inline_obj),

            // If no page is specified and this is a pageless document, put the inline object on
            // the first (only) page.
            None if pageless => by_page[0].push(inline_obj),

            None => {
                warn!(
                    "Ignoring inline {} because it does not specify a page, \
                    and this is not a pageless document",
                    <&str>::from(&inline_obj.object),
                );
            }
        }
    }

    by_page
}

pub fn split_document_into_pages<'d>(
    doc: &'d sdocx::Document,
    embed_cache: &EmbedMap<'d>,
    user_auto_split: bool,
    user_basic_split: Option<BasicSplitMode>,
) -> impl Iterator<Item = PageContent<'d>> {
    let is_marked_pageless = matches!(doc.page_model(), sdocx::PageModel::Pageless);

    let page = match doc.pages().iter().exactly_one().ok() {
        // Pageless documents get special treatment, because it's up to us to determine whether to
        // insert page breaks, and if so, where.
        Some(only_page) if is_marked_pageless => only_page,

        // Paged documents (and documents with a page count other than 1 which are incorrectly
        // marked as pageless - shouldn't happen, but just in case...) are broken up for us
        // already, so we use the existing pages.
        _ => {
            let uniform_scale = compute_uniform_scale(doc.pages().iter().map(|page| {
                let page_box = sdocx::Box2d::from_size(sdocx::Size2d::new(
                    page.width as f64,
                    page.height as f64,
                ));

                (
                    page_box,
                    page.embedded_pdf_pages()
                        .iter()
                        .map(|embed| (embed_cache.get_crop_box(embed), embed.rect())),
                )
            }));

            return Either::Left(uniform_scale.into_iter().flat_map(move |uniform_scale| {
                let inline_objects = group_inline_objects_by_page(doc);

                doc.pages()
                    .iter()
                    .zip(inline_objects)
                    .with_position()
                    .filter_map(move |(pos, (page, inline_objects))| {
                        // For paged documents, there is a ghost page in the SDOCX that is not
                        // represented in the raster PDF. We ignore it too.
                        if !is_marked_pageless
                            && pos == itertools::Position::Last
                            && page.is_empty()
                            && inline_objects.is_empty()
                        {
                            return None;
                        }

                        Some(PageContent::WholePage {
                            page,
                            inline_objects,
                            scale: uniform_scale,
                        })
                    })
            }));
        }
    };

    // Compute the lowest point the bounding box of any content object reaches so we can ignore any
    // pages that begin after this point.
    let content_max_y = page
        .embedded_pdf_pages()
        .iter()
        .map(|embed| embed.bounding_box().max.y)
        .chain(
            page.layers()
                .iter()
                .flat_map(|layer| layer.objects())
                .map(|object| object.bounding_box().max.y),
        )
        .chain(
            doc.body_text()
                .inline_objects()
                .iter()
                .map(|iobj| iobj.bounding_box().max.y),
        )
        .max_by(f64::total_cmp)
        // If there's no content, just use the whole page height because it's probably sane.
        .unwrap_or(page.height as f64);

    let page_breaks = compute_page_breaks(page, user_auto_split, user_basic_split);

    let content_slices = Vec::from_iter({
        let page_width = page.width as f64;
        let background_colour = page.background_colour();
        let template_type = page.template_type();

        page_breaks
            .tuple_windows()
            .map_while(move |(min_y, max_y)| {
                if min_y >= content_max_y {
                    // There is no content in this slice or any of the ones that follow.
                    return None;
                }

                let capture_box =
                    sdocx::Box2d::new((0.0, min_y).into(), (page_width, max_y).into());

                // todo: Check whether we include embeds/objects that are barely touching the
                // capture box here. If the object won't be visible on the output page, we
                // shouldn't be including it here. It might throw off the scale calculation and
                // could increase the size of the output PDF and make it more expensive to render
                // by including unnecessary duplicate objects on each page.

                // Find all the content that intersects the capture box.
                Some(PageCapture {
                    capture_box,
                    background_colour,
                    template_type,
                    embeds: page
                        .embedded_pdf_pages()
                        .iter()
                        .filter(|embed| capture_box.intersects(&embed.bounding_box()))
                        .collect(),
                    objects: page
                        .layers()
                        .iter()
                        .flat_map(|layer| layer.objects())
                        .filter(|object| capture_box.intersects(&object.bounding_box()))
                        .collect(),
                    // There's only one input page, so all the inline objects in the document must
                    // be on it.
                    inline_objects: doc
                        .body_text()
                        .inline_objects()
                        .iter()
                        .filter(|iobj| capture_box.intersects(&iobj.bounding_box()))
                        .collect(),
                })
            })
    });

    let uniform_scale = compute_uniform_scale(content_slices.iter().map(|slice| {
        (
            slice.capture_box,
            slice
                .embeds
                .iter()
                .map(|embed| (embed_cache.get_crop_box(embed), embed.rect())),
        )
    }));

    // Unwrapping is fine because we'd only get `None` if there were no pages, but we have at least
    // one content slice because we were guaranteed at least one pair of page breaks.
    let uniform_scale = uniform_scale.unwrap();

    Either::Right(
        content_slices
            .into_iter()
            .map(move |c| PageContent::Slice(c, uniform_scale)),
    )
}

#[derive(Debug, Error)]
pub enum DrawObjectError {
    #[error("failed to draw stroke object")]
    Stroke,

    #[error("failed to draw path for shape")]
    ShapePath(PathDrawingError),

    #[error("no style information for line")]
    NoLineStyle(NoStyleError),

    #[error("objects of type '{0}' are not yet supported")]
    Unsupported(&'static str),
}

impl DrawObjectError {
    pub fn log(&self) {
        match self {
            DrawObjectError::Stroke => error!("Failed to draw stroke object"),

            DrawObjectError::ShapePath(e @ PathDrawingError::NothingDrawn(_)) => {
                warn!("Shape will be invisible in output: {e}");
            }

            DrawObjectError::ShapePath(err) => {
                error!("Failed to draw shape: {err}")
            }

            DrawObjectError::NoLineStyle(_) => {
                error!("Line has no style information")
            }

            DrawObjectError::Unsupported(obj_type_name) => {
                warn!("Ignoring '{obj_type_name}' object (not yet supported)")
            }
        }
    }
}

/// SDOCX page -> PDF page conversion context.
pub struct PageConversionCtx<'s> {
    input_box: sdocx::Box2d<f64>,

    /// The scale from document units to PDF points.
    input_to_output_scale: euclid::Scale<f64, sdocx::SdocxSpace, pdf::PdfSpace>,

    /// PDF drawing surface.
    ///
    /// When the page conversion context is constructed, a transformation matrix is pushed to this
    /// drawing surface that scales the content such that drawing operations can be done in
    /// document space rather than PDF space.
    pub surface: Surface<'s>,
}

impl<'s> PageConversionCtx<'s> {
    pub fn new<'p>(
        page: &PageContent<'p>,
        // page_size: (f32, f32),
        // pt_per_unit: f32,
        mut surface: Surface<'s>,
    ) -> PageConversionCtx<'s> {
        // Apply a transformation here so that drawing operations to the surface can use document
        // units rather than locally scaling things whenever we need to draw them.
        let scale = page.input_to_output_scale().get() as f32;
        let input_origin = page.input_box().min.cast::<f32>();

        surface.push_transform(&Transform::from_row(
            scale,
            0.0,
            0.0,
            scale,
            // If the page is actually a slice of a larger page, the input box origin won't be at
            // zero. Translate the content back to the origin.
            -input_origin.x * scale,
            -input_origin.y * scale,
        ));

        PageConversionCtx {
            input_box: page.input_box(),
            input_to_output_scale: page.input_to_output_scale(),
            surface,
        }
    }

    pub fn fill_background(&mut self, r: u8, g: u8, b: u8, a: u8) {
        self.surface.set_stroke(None);
        self.surface.set_fill(Some(Fill {
            paint: rgb::Color::new(r, g, b).into(),
            opacity: NormalizedF32::new(a as f32 / 255.0).unwrap(),
            ..Fill::default()
        }));

        self.surface.draw_path(&self.boundary_path());
    }

    /// Draws the page's background paper style (as chosen in the app's "Change template" menu) on
    /// top of the background colour `background_bgra`. Only grids are supported.
    pub fn draw_background_template(
        &mut self,
        template_type: Option<&TemplateType>,
        background_bgra: Option<[u8; 4]>,
    ) {
        // Grid line spacing and colour are not recorded anywhere in the document itself
        // (confirmed absent by inspecting the page's own raw fields directly), so, like the
        // background colour, the app must be applying a fixed visual template purely from the
        // template type. Measured directly against a real "Export to PDF" ground truth with a
        // narrow grid: line spacing is exactly 27pt (measured 112-113px at 300 DPI, dominant
        // value, both axes, so a square grid), and the line colour is the page's own background
        // colour darkened by a factor of 0.8 (measured (196,182,171) on a (244,228,215)
        // background, ~0.80 on every channel). The same factor was independently arrived at for
        // ruled-paper lines on different ground truth, so Samsung's renderer most likely applies
        // one fixed darken factor to the background colour for every paper style that draws
        // lines against it.
        //
        // Medium/wide grid spacing (1.5x/2x narrow) is a proportional guess, not measured against
        // a real export. Line and dot paper styles are not implemented at all for the same
        // reason: guessing their appearance without a real example to check against risks being
        // confidently wrong rather than usefully approximate.
        //
        // The 0.5pt line width is an empirical compromise, not a measurement: the ground truth's
        // own rendered line is ~4-5px wide at 300 DPI (~1.0-1.1pt), but matching that width
        // directly scored *worse* on both documents with a real grid, and a swept-width
        // comparison (0.1 to 1.0pt) found 0.5pt to be the best net choice. One of the two
        // documents still has a small residual cost at every width tried, which points at an
        // unresolved alignment issue rather than a pure calibration gap.
        const NARROW_GRID_SPACING_PT: f64 = 27.0;
        const LINE_WIDTH_PT: f64 = 0.5;
        const DARKEN_FACTOR: f32 = 0.8;

        let spacing_pt = match template_type {
            Some(TemplateType::NarrowGrid) => NARROW_GRID_SPACING_PT,
            Some(TemplateType::MediumGrid) => NARROW_GRID_SPACING_PT * 1.5,
            Some(TemplateType::WideGrid) => NARROW_GRID_SPACING_PT * 2.0,
            _ => return,
        };

        // We draw in document units, so convert from points.
        let spacing = (pdf::Length::new(spacing_pt) / self.input_to_output_scale).get();
        let line_width = (pdf::Length::new(LINE_WIDTH_PT) / self.input_to_output_scale).get();

        let [b, g, r, _] = background_bgra.unwrap_or([255, 255, 255, 255]);
        let darken = |c: u8| (f32::from(c) * DARKEN_FACTOR) as u8;

        // Lines are positioned relative to the origin of the whole input page rather than to the
        // input box, so the grid continues seamlessly across the slices of a split page.
        let input_box = self.input_box;
        let mut pb = krilla::geom::PathBuilder::new();

        let mut x = (input_box.min.x / spacing).ceil() * spacing;
        while x < input_box.max.x {
            pb.move_to(x as f32, input_box.min.y as f32);
            pb.line_to(x as f32, input_box.max.y as f32);
            x += spacing;
        }

        let mut y = (input_box.min.y / spacing).ceil() * spacing;
        while y < input_box.max.y {
            pb.move_to(input_box.min.x as f32, y as f32);
            pb.line_to(input_box.max.x as f32, y as f32);
            y += spacing;
        }

        let Some(path) = pb.finish() else {
            return;
        };

        self.surface.set_fill(None);
        self.surface.set_stroke(Some(Stroke {
            paint: rgb::Color::new(darken(r), darken(g), darken(b)).into(),
            width: line_width as f32,
            ..Stroke::default()
        }));

        self.surface.draw_path(&path);
    }

    fn boundary_path(&self) -> krilla::geom::Path {
        let mut rect_pb = krilla::geom::PathBuilder::new();

        let input_box = self.input_box.cast::<f32>();
        let origin = input_box.min;
        let size = input_box.size();

        rect_pb.push_rect(
            krilla::geom::Rect::from_xywh(origin.x, origin.y, size.width, size.height).unwrap(),
        );

        rect_pb.finish().unwrap()
    }

    fn draw_stroke_chunk_events<'e>(
        &mut self,
        stroke_events: impl IntoIterator<Item = EventGroup<'e>>,
        fill_boundary: &krilla::geom::Path,
        tool: Tool,
    ) -> Result<(), ()> {
        tool.draw_events(stroke_events, fill_boundary, &mut self.surface)
    }

    pub fn draw_single_object(
        &mut self,
        object: &sdocx::DocObject,
        pen_width_mul: f32,
        marker_width_mul: f32,
        media: &mut MediaStorage,
    ) -> Result<(), DrawObjectError> {
        match object {
            sdocx::DocObject::Stroke(stroke) => self
                .draw_stroke_chunk_events(
                    [EventGroup::from_stroke(stroke)],
                    &self.boundary_path(),
                    Tool::for_stroke(stroke).with_scaled_width(pen_width_mul, marker_width_mul),
                )
                .map_err(|()| DrawObjectError::Stroke),

            sdocx::DocObject::Line(line) => {
                if line.has_control_points() {
                    warn!("Ignoring line control points");
                }

                self.draw_line(line.start(), line.end(), line.colour_effect(), line.style())
                    .map_err(DrawObjectError::NoLineStyle)
            }

            sdocx::DocObject::Shape(_) | sdocx::DocObject::Image(_) => {
                let shape = match object {
                    sdocx::DocObject::Shape(shape) => shape,
                    sdocx::DocObject::Image(image) => &image.shape,
                    _ => unreachable!(),
                };

                self.draw_shape(shape, media)
                    .map_err(DrawObjectError::ShapePath)
            }

            other => Err(DrawObjectError::Unsupported(<&str>::from(other))),
        }
    }

    pub fn draw_objects<'d>(
        &mut self,
        objects: impl ExactSizeIterator<Item = &'d DocObject>,
        pen_width_mul: f32,
        marker_width_mul: f32,
        multi_progress: &MultiProgress,
        media: &mut MediaStorage,
    ) {
        let objects_bar = multi_progress
            .add(ProgressBar::new(objects.len() as _))
            .with_style(
                ProgressStyle::with_template(
                    "Processing objects [{bar:40}] {percent}% [{pos}/{len}]",
                )
                .unwrap()
                .progress_chars("# "),
            );

        // Consecutive strokes very often use the same tool. To reduce the size of the output PDF,
        // we can process chains of such strokes in one go, loading the necessary graphics state
        // only once and then using it for all the strokes rather than loading the same graphics
        // state for every stroke.
        let chunked_objects = objects
            .inspect(|_| objects_bar.inc(1))
            .chunk_by(|&obj| match obj {
                sdocx::DocObject::Stroke(stroke) => Some(
                    Tool::for_stroke(stroke).with_scaled_width(pen_width_mul, marker_width_mul),
                ),
                _non_stroke => None,
            });

        // Compute the filling boundary once for this layer so we don't have to recompute it for
        // every chunk of events we draw.
        let fill_boundary = self.boundary_path();

        for (opt_stroke_tool, objects) in &chunked_objects {
            if let Some(tool) = opt_stroke_tool {
                // This is a chunk of strokes that all use `tool`.

                // Get the event slice for each stroke.
                let stroke_events = objects.map(|o| {
                    let sdocx::DocObject::Stroke(s) = o else {
                        unreachable!()
                    };
                    EventGroup::from_stroke(s)
                });

                if let Err(()) = self.draw_stroke_chunk_events(stroke_events, &fill_boundary, tool)
                {
                    DrawObjectError::Stroke.log();
                }

                continue;
            }

            // This is a chunk of non-strokes.
            for obj in objects {
                if let Err(err) =
                    self.draw_single_object(obj, pen_width_mul, marker_width_mul, media)
                {
                    err.log();
                }
            }
        }
    }
}

impl Drop for PageConversionCtx<'_> {
    fn drop(&mut self) {
        // Pop the coordinate transformation.
        self.surface.pop();
    }
}
