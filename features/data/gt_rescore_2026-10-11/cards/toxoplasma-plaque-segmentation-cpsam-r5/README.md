---
license: cc-by-4.0
tags: [cellpose, cellpose-sam, segmentation, microscopy, toxoplasma, plaque-assay, spacr]
pipeline_tag: image-segmentation
---

# Toxoplasma Plaque v2

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma Plaque v2

**What it segments.** Toxoplasma gondii plaques in crystal-violet-stained plaque-assay wells.

**Version.** spaCR model-zoo key `toxoplasma_plaque_v2`, training round 5 (weights `cpsam_plaque_r5`). Status: Earlier default, kept so recorded runs reproduce. The default plaque model in spaCR is toxoplasma_plaque_v4.

**Trained on.** 332 fields from published figures, phone-camera wells and two lab microscope plate sets, grouped by figure and plate.

**Date.** trained 2026-09-19

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5](https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5)
- Stock model it was tuned from: [cpsam_v2](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `plaque_model` to `toxoplasma_plaque_v2`; spaCR downloads and checks the weights on first use.
