---
license: cc-by-4.0
tags: [cellpose, cellpose-sam, microscopy, toxoplasma, plaque-assay, spacr, default]
pipeline_tag: image-segmentation
---

# Toxoplasma Plaque v4

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma Plaque v4

**What it segments.** Toxoplasma gondii plaques in crystal-violet-stained plaque-assay wells.

**Version.** spaCR model-zoo key `toxoplasma_plaque_v4`, training round 6 (weights `cpsam_plaque_r6`). Status: Default plaque model in spaCR.

**Trained on.** 335 curated fields from Gel Doc lab plates, lab plates, published figures and staged literature crops, with 65 for validation.

**Date.** trained 2026-10-10

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6](https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6)
- Stock model it was tuned from: [cpsam_v2](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `plaque_model` to `toxoplasma_plaque_v4`; spaCR downloads and checks the weights on first use.
