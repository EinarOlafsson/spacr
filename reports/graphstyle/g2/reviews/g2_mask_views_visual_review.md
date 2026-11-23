> Curated publication copy. Original source: `.codex-audit/g2_mask_views_visual_review.md`; original SHA256: `8e7cb6600e8dfcecc1abc7264072d0d2547966599174034655857300951cc90e`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# G2 mask-view native visual review

Reviewed 2026-11-23 by `timelapse_plot_audit`, using the existing job 1667 artifacts. **All 48 PNGs were opened individually:** 24 before and 24 after. No new blocking defect was found in these fixtures. No source changes, Python, tests, renders, or jobs were performed for this review.

Evidence roots are [reports/graphstyle/g2/evidence/mask-views-before/](../evidence/mask-views-before) and [reports/graphstyle/g2/evidence/mask-views-after/](../evidence/mask-views-after); each row below covers its named PNG in **both** `light/` and `dark/` under **both** roots. “Accepted” applies only to the visible captured fixture.

| PNG stem | Individually viewed | Before / after finding | Disposition |
| --- | --- | --- | --- |
| `probability` | Before light, before dark, after light, after dark | Before lacks a count axis and class/threshold key. After has readable axes, two class keys and both threshold values; all caption/key text fits. Dark reference backing is visible. | Accepted |
| `probability_nine_classes` | Before light, before dark, after light, after dark | After adds all nine class keys and retains four occupied groups. Classes 7–9 use diagonal hatching matching their keys; no repeated solid encoding for 1/7, 2/8 or 3/9. No clipped keys. Dark narrow bars and hatch colors remain subtle. | Limited: distinct encodings observed, general perceptual distinguishability not established |
| `otsu` | Before light, before dark, after light, after dark | Before has no quantity labels. After labels intensity/pixels and both supplied cuts (1.3, 6.8); the second cut sits on the right frame because it is the displayed maximum. Frame, counts and external key remain readable. | Accepted |
| `normalization` | Before light, before dark, after light, after dark | After clearly identifies black/white cutoffs (1, 5), counts and intensity. The two cut markers, caption and explanatory line fit without visible collision. | Accepted |
| `overlay` | Before light, before dark, after light, after dark | Same RGB gradient and two green boundaries; after adds a readable external “cells” outline key. Image shifts upward to reserve the caption, with no visible content loss. | Accepted image/chrome observation |
| `masks` | Before light, before dark, after light, after dark | Same red/blue objects on black; after explains categorical object colors below the image. No graph palette or opacity applied to the image. | Accepted image/chrome observation |
| `flows` | Before light, before dark, after light, after dark | Same supplied flow RGB pattern; after adds direction/strength and maximum-combination text outside the image. No visible clipping. | Accepted image/chrome observation; synthetic supplied RGB is not model output validation |
| `cell_probability` | Before light, before dark, after light, after dark | Same dark-to-light magma progression; after identifies probability 0–1 and magma below the image. | Accepted image/chrome observation |
| `annotation_count` | Before light, before dark, after light, after dark | Count 39 stays in the upper-left backed badge. New text is smaller/lighter but readable at the saved 240×240 size; crop and border remain intact. | Accepted for two-digit fixture only |
| `boxes_zoomed` | Before light, before dark, after light, after dark | Old partial label is clipped to a trailing fragment. After shows complete inside/partial names on contrasting backgrounds; wholly offscreen name remains absent. Rectangle positions and visible image are unchanged. | Accepted |
| `recrops_zoomed` | Before light, before dark, after light, after dark | After fits inside/partial names within the visible image and keeps the offscreen name absent. The partial recrop outline/fill still extends into the side margin, as before; this is retained behavior. | Accepted |
| `magnifier_locked_updating` | Before light, before dark, after light, after dark | Same lens, enlarged pixels and dashed outline. New smaller/lighter “Locked · Updating Otsu...” caption remains readable and contained on its backing. | Limited: prepared locked/updating state only |

## Paired data and actual layout evidence

The [after report](../evidence/mask-views-after/report.json) names the immutable before report with SHA-256 `a779618a85b6a9a3314d5bb7c84b7156f27701ee9235a0d9d1739cdd5e9208e3`, reports `baseline_comparison.numerical_identity=true`, and has `differences=[]`. Input arrays also equal their post-capture hashes. The strict records include ordered probability classes (including `[1,2,3,4,5,6,7,8,9]`), counts, edges, thresholds/cutoffs, rendered RGB arrays, raw canvas/crop pixels, box/recrop coordinates and viewport, and magnifier pixels/geometry. This supports retained data within these inputs; full QWidget PNG identity is neither expected nor asserted.

Normal resolved themes are light (`bg=#fafafa`, `fg=#0d0e10`) and dark (`bg=#000000`, `fg=#ffffff`). For all eight after histogram captures, actual observations report Open Sans, 8.25 pt, weight 300, opaque text/frame/key colors, data alpha approximately 0.6, and black reference cores at approximately 0.6. All have `text_geometry.outside=[]`, `overlaps=[]`, `incomplete_checks=[]`, and `diagnostics.problems=[]`, consistent with the viewed output. Actual square plot rectangles, shared between themes:

| Fixture | x, y, width, height in device-independent pixels |
| --- | --- |
| Probability | 80.1015625, 15.484375, 460.765625, 460.765625 |
| Nine classes | 72.96875, 15.484375, 555.03125, 555.03125 |
| Otsu / normalization | 72.96875, 15.484375, 475.03125, 475.03125 |

The nine-class marks and opaque keys record matching brush styles `[1,1,1,1,1,1,12,12,12]`. Combined color/pattern identities distinguish all nine, but bars are only about 6.17 pixels wide; dark-theme class 3/9 and hatched colors are visually subdued. This review does not upgrade that observation to an accessibility claim.

## Scope and provenance limits

Before observations truthfully say the old creators have no native style observation API. Image/overlay records explicitly remain `certified=false`; added metadata is not image certification. Magnifier detector queues were closed before enabling, with `detector_threads_started=false`; this demonstrates actual painting of the prepared state, not inference or a completed Otsu result. These images do not visually cover every resize, localized caption, long class name, twelve-cutoff case, negative-range corner, or float64-to-float32 constant case. Other targeted test evidence remains separate.

Both trees report HEAD `9510143823dcc1b984ecc522058a1e3f15e985d8`; the after tree is the frozen validation snapshot, so its recorded module hashes identify the changes. In particular: style `34ec8c07…`, annotate `aba0346b…`, make_masks `43147a58…`, segmentation_views `942a0a96…`; mask_engine is identical to baseline. After report SHA-256: `0357f155aa4fd1fb39013917e7e926e5987b10cc979e77a8b82d554ba2f99e19`. Harness SHA-256: `e40b997d7eb2a468d10239d06a2498940da1af59065ea4851aa458ae1ecbba06`.

These are native widget snapshots from synthetic inputs. They do not establish fresh tutorial captures, full desktop recording acceptance, or scientific pipeline acceptance. Tutorial refresh ownership and the existing relay remain unchanged.
