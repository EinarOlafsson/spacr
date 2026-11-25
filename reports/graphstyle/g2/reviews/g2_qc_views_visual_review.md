> Curated publication copy. Original source: `.codex-audit/g2_qc_views_visual_review.md`; original SHA256: `d184dd1c9e2afe95c623ddac2aac4f2ac45579a7173d5f502bbde084fea09722`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# Native QC views: paired visual review

Reviewed root job1669's actual offscreen captures, individually: all 26 PNGs in
each of [reports/graphstyle/g2/evidence/qc-views-before](../evidence/qc-views-before) and `qc-views-after`
(52 PNGs total). No images were regenerated, transformed or edited for this
review. No Python, tests or jobs were executed by this reviewer.

Before report SHA256:
`f36fc11b313cd71a6e49c44f87585934e582a18176ca75864feab8ecf13a1b9a`.
After report SHA256:
`b6fe3072e38bfb546c8b4ebb5ea81633232fbfe35d5c4ffb9aa4171042d60c08`.
The captured candidate uses shared style `34ec8c07`, Measure preview
`3412f73d`, QC browser `5d253e02`, Align `fb9ab643` and harness `9641ecf7`.
Both trees report G1 HEAD `9510143823`; the candidate implementation is bound
by the reported file hashes, not by claiming those modifications are committed.

## Findings requiring disposition

1. **Overview export figure-level typography remains outside the shared style.**
   Both AFTER ordinary export PNGs have visible, unclipped header/caption text,
   but the actual finalizer reports non-bundled font family and undeclared font
   weights. The creator's `figure.text` header and field captions are not styled
   by the new image-axes adapter. Resolve those actual Text artists through the
   shared font/role settings; preserve their strings, coordinates and image data.
   The dark report also has a contrast warning after the normal print writer
   restores screen appearance. The saved PNG itself is a readable white print
   page with dark text; this warning must not be described as invisible text in
   that PNG. A targeted writer recapture should distinguish saved appearance
   from restored interactive figure diagnostics.
2. **The new Align dark low-confidence key needs a contrast boundary.**
   The first key fill is `#1f2124` on `#0d0e10`, visibly much harder to distinguish
   than the corresponding tile, which has an outline. At both 180 and 400 px,
   its text is readable and its rectangle fits. Add an opaque contrasting
   key-only outline/backing while keeping this exact confidence fill and all
   plotted tile colors/coordinates unchanged. This is not a request to change
   the scientific confidence ramp.

These findings were communicated to root before any production edits. The
native data/selection checks pass, but this note does not certify all outputs
as graph-style compliant while the typography/key findings remain.

## Exact scientific and artifact evidence

The full `.scientific` objects compare byte-for-byte after sorted JSON
serialization, independently of the report's `scientific_identity=true` flag.
`different_cases` is empty. Both reports record unchanged source inputs and
input arrays. Every recorded output hash verifies: 26 PNGs and two PDFs per
tree (56 artifacts). PDFs were hash-checked, not visually reviewed here.

That equality covers source array dtype/shape/data/mask digests, supplied crop
arrays and included/excluded identities, exact rendered thumbnail RGBA,
selected QC channel/mask order/composite RGBA, histogram counts and original
level/gamma values, cell-picker grid/selection/mapping, Align plans and clicked
tile indices, and Overview image arrays plus selection provenance. Only the
owned output-directory prefix is normalized in export provenance by the
previously reviewed harness. It does not normalize scientific settings or
coordinates.

## Complete PNG inspection ledger

Each row below covers **four separately opened images**: before/light,
after/light, before/dark and after/dark at the stated relative path.

| Relative PNG | Actual before/after observations |
| --- | --- |
| `histogram-loaded.png` | Before has unlabelled tall bars/handles. After has the same four populated bins (counts sum30), square data area, readable count/display axes, explicit sqrt display explanation and three handle labels. Grey clipping areas and all black/blue/white triangles are visible in both themes; dark reference lines retain a visible backing. No observed clipping or overlap. |
| `histogram-missing.png` | Before is visually indistinguishable from zero counts. After explicitly says “Channel display histogram unavailable”, has no Pixels key/data series, and retains only the supplied level handles. The count scale is a declared layout placeholder, not evidence of observed counts. |
| `histogram-zero.png` | After has the ordinary histogram caption and Pixels key, no positive bars, and the same three handles. It is visibly distinct from unavailable data. The 256 recorded counts are all zero. |
| `align-180.png` | Before has no key. After has four complete readable English key lines within180px, separated from tiles. All four identifiers remain visible and clicks bind the same plan indices. Dark low-confidence swatch contrast is the finding above. |
| `align-400.png` | Same complete key at400px, with no overlap, clipping or truncated label. Tile geometry is translated within the available plot region to reserve key space; source placements remain exact. Dark low-confidence swatch finding also applies. |
| `measure-grid.png` | Both category counts and kept/excluded totals remain exact. AFTER headings use lighter shared typography; all heading text fits. Three thumbnails, exclusion outline and category grouping remain visible. Different available dark widget width is layout, not a crop change. |
| `qc-field.png` | Same grayscale intensity gradient and two green-cell/purple-nucleus outline pairs, same orientation and locations. No visible pixel recoloring, blur, mask shift or opacity change. |
| `cell-picker.png` | Same white page, occupied Well A01 cell and selected two-column row. AFTER occupied label uses smaller shared typography but remains readable and fits. No changed selection geometry or mapping. |
| `crop-0.png` | Original gradient direction, RGB colors and rounded transparent corners preserved exactly; exact RGBA digest agrees. |
| `crop-1.png` | Horizontal reversal relative to crop0 remains exact, with the same colors/rounding and RGBA digest. |
| `crop-2.png` | Existing rolled-row discontinuity remains at the same location, with the same colors/rounding and RGBA digest. No smoothing or new data processing. |
| `qc-composite.png` | Same tiny source-resolution composite, including mask borders and image intensities; exact RGBA digest agrees in both themes. |
| `overview/overview_0001.png` | Both saved outputs remain ordinary white print pages with full Channels/percentiles/seed and field identity headers and the same enlarged two-channel RGB plane. Text and image fit without clipping. Font/weight findings above remain despite readable appearance. |

## Bounds, metadata and limits

All six AFTER native histogram observations have empty `text_geometry.outside`
and `overlaps`. Reported minimum actual font size is8.25pt. Loaded data have
256 bins/four positive bins/sum30; missing data have `available=false`, zero
input bins and no data marks; measured-zero data have `available=true`,256
input bins and a data mark whose counts sum0. Exact inverse level mapping is
retained within the harness's numerical tolerance.

Their normal finalizers intentionally retain “Display-clipping shade and
draggable triangle overlays require visual review.” This review observes those
overlays in these particular supplied states; it does not remove the diagnostic
or certify every interactive state. Missing-data finalizers additionally retain
the explicit unavailable-data/placeholder warning.

All16 AFTER Align key text rectangles (four entries × two widths × two themes)
are reported inside the canvas and outside tiles, at8pt. These English captures
fit on single lines at180px; they do not exercise localized wrapping. The
separate root-run unit cases cover long localized key text, but no localized
native screenshot or platform acceptance is claimed here. Four Measure heading
bounds fit at9.75pt. Non-histogram widgets correctly report that callable native
finalizer observations are unavailable rather than inventing compliance.

This is tiny synthetic offscreen evidence on the supplied environment. It is
not a tutorial recording, native desktop/platform acceptance, evidence of
real-field scientific performance, or a general language/HiDPI/layout audit.

## Prepared follow-up, not a replacement of the reviewed evidence

Root authorized both concrete fixes after receiving this review. The Overview
creator now takes its two figure-text artists' family and body weight directly
from `resolve_style("print", "image")`, preserving strings, point sizes and
coordinates (QC browser `6bbec6d1`). Align adds an opaque foreground outline
only where the unchanged key fill has shared contrast below3:1 against its
ground; plotted tile colors/geometry are untouched (Align `06660e91`). The
existing four Align tests and single Overview test now check these properties
(tests `bddbd32d`). The harness owner added `--case` and `--theme` selectors
with strict selected-case scientific comparisons (`56fc9c96`). These changes
are prepared, not yet validated by a new run at the time of this note.
