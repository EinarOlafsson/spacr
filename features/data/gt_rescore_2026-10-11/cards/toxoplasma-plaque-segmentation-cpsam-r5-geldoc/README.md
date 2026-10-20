---
license: cc-by-4.0
tags: [cellpose, cellpose-sam, microscopy, toxoplasma, plaque-assay, spacr, alternative]
pipeline_tag: image-segmentation
---

# Toxoplasma Plaque v3

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma Plaque v3

**What it segments.** Toxoplasma gondii plaques in crystal-violet-stained plaque-assay wells.

**Version.** spaCR model-zoo key `toxoplasma_plaque_v3`, training round 5 (Gel Doc) (weights `cpsam_plaque_r5_geldoc`). Status: Alternative, not the default (default 2026-10-10 to 2026-10-11). The default plaque model in spaCR is toxoplasma_plaque_v4.

**Trained on.** 496 curated fields (34 reviewed empty) from published figures, lab NAS wells, Bio-Rad Gel Doc lab plates and staged literature crops.

**Date.** trained 2026-10-08

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc](https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc)
- Stock model it was tuned from: [cpsam](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `plaque_model` to `toxoplasma_plaque_v3`; spaCR downloads and checks the weights on first use.
