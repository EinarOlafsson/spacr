> Curated publication copy. Original source: `.codex-audit/g2_extra_qc_browser_review.md`; original SHA256: `7aaa49757f3f2ca110697b0232dd6f7b26008dc337caa45f6229af9cc952d078`. Original observations retain their recorded historical scope. Only selected artifacts are linked; omitted artifacts are labeled local-only and are never retargeted to a later pass. Unlinked context-relative paths and code blocks retain the original review context, which can include local-only omitted outputs; they do not select a later artifact. The validation index identifies the final selected subsets.

# G2 extra-QC browser viewport review

All nine PNGs in [reports/graphstyle/g2/evidence/extra-qc-browser-v2/](../evidence/extra-qc-browser-v2) were viewed individually after root's job 1674. The capture index covers twelve original HTML files: three baseline light/dark pairs were skipped only after identical SHA256 and byte-comparison proof. The retained exit code is 0 and input postflight reports all frozen inputs unchanged. These are actual Chrome screenshots of unchanged local HTML at 1440 × 1000, not embedded-preview exports or restyled pages.

| Individually viewed screenshot | Actual visible finding |
| --- | --- |
| `before-shared-normal.png` | Dark gray page/cards, pale text and two small opposite-gradient thumbnails. Flagged field appears before accepted field; channel 7, focus variance 1.25 and saturated fraction 0.01 remain readable. No display-range caption yet. |
| `before-shared-all_nan.png` | Dark gray page, one readable nonfinite card with channel 0/focus 0/saturated fraction None. The tiny black thumbnail is barely distinguishable; no fallback explanation appears. |
| `before-shared-mixed_nonfinite.png` | Same baseline theme and readable nonfinite metrics. The tiny mixed thumbnail is not a useful visual explanation on its own; no fallback explanation appears. |
| `after-light-normal.png` | White page/cards, dark text/borders. Both fields and their original metrics remain visible; added caption states display range 647.38–1272.6, 1st–99th percentiles of finite sampled preview values, gamma 1 and stride 1. Text wraps inside each card. |
| `after-dark-normal.png` | Black page/cards, pale text/borders. Both thumbnails and the same measured-range captions are visible, with no clipping or overlap observed in this viewport. |
| `after-light-all_nan.png` | Readable caption explicitly calls 0–1 a fallback because the sampled preview has no finite values and says no percentiles were measured; gamma 1/stride 1 remain visible. The original tiny black image is visible as a small mark. |
| `after-dark-all_nan.png` | Same complete fallback explanation in pale text. The black image blends into the black card; the caption, not image visibility, establishes why the preview is black. |
| `after-light-mixed_nonfinite.png` | Same accurate no-finite-sample fallback explanation, distinct filename and original metric values. The tiny mixed image remains visually minimal; no caption clipping observed. |
| `after-dark-mixed_nonfinite.png` | Complete fallback explanation and original metrics remain readable. Tiny mixed pixels provide little visible information at this size; no claim that the mixed image is uniformly black. |

The header consistently distinguishes preview-only contrast stretching from metrics on raw intensities, maximum projections for volumes, and excluded fields from zero-object detections. Candidate theme CSS now produces visibly distinct light and dark pages, while baseline theme inputs were identical. No concrete caption omission, visible text collision or clipping was found in these nine captured viewports.

The prior exact embedded PNG/numerical comparisons remain separate evidence; these screenshots do not establish new numerical equality. The synthetic thumbnails are only 20 × 16 or 3 × 2 pixels and retain their natural small display size. This review does not certify full-page overflow, larger real galleries, responsive layouts at other widths, native Qt, tutorials, or browser-computed font identity. The reported installed Open Sans font match and the HTML CSS font declaration do not by themselves prove which font Chrome actually rendered.

| Screenshot | SHA256 |
| --- | --- |
| `before-shared-normal.png` | `f77639542ca2713287271e1999ac27b27d4866c233da11e39477f73c0d0f91ce` |
| `before-shared-all_nan.png` | `5b0fd6d9721d51d0613dda8ffc1d8a3927c0be9745ea3e41378108d9c0a33fbd` |
| `before-shared-mixed_nonfinite.png` | `2199252bdc2f990667b80d6b41dd9af0b14c030138fa481f2f44f2c369b1cc75` |
| `after-light-normal.png` | `6c0d725d3f1d73aafd8a4fed3451e1e2522e3fc4e32d49269e2da143fe64f3f1` |
| `after-light-all_nan.png` | `545c9a20054e1f490fe2ee28a1124c1e22fbdb385daa7c2ab998662fe94544ca` |
| `after-light-mixed_nonfinite.png` | `3aceb12275f75509616baa802ae7fb377a21306a896fe3b8c93d3e4a13e47b25` |
| `after-dark-normal.png` | `f8e2b8ab05ca5c2403d8f50444a49e10f793dcd484638daf748ea5f05c131b50` |
| `after-dark-all_nan.png` | `c36875380cf4f20f935370443cabb30709d0551311545b691974b58bc4baabbd` |
| `after-dark-mixed_nonfinite.png` | `3352bd2ce27533ce144906ffd40191cec19d3559e6d17af7301b718602e0f869` |

Only this scratch review note was written. No browser, renderer, Python, tests or jobs were run during review.
