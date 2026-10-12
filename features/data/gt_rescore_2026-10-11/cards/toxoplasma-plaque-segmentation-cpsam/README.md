---
license: mit
tags:
- cellpose
- segmentation
- toxoplasma
- plaque-assay
- spacr
library_name: spacr
pipeline_tag: image-segmentation
---

# Toxoplasma Plaque v1

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma Plaque v1

**What it segments.** Toxoplasma gondii plaques in crystal-violet-stained plaque-assay wells.

**Version.** spaCR model-zoo key `toxoplasma_plaque_v1`, training round 3 (weights `cpsam_plaque_r3`). Status: Earlier version, kept so recorded runs reproduce. The default plaque model in spaCR is toxoplasma_plaque_v4.

**Trained on.** 136 wells from two lab microscope plate sets and published-figure crops (reconstructed single split; 48 more were held out).

**Date.** trained 2026-08-03

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-plaque-segmentation-cpsam](https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam)
- Stock model it was tuned from: [cpsam](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `plaque_model` to `toxoplasma_plaque_v1`; spaCR downloads and checks the weights on first use.
