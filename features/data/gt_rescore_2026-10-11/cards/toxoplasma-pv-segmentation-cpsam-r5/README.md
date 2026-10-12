---
license: mit
tags:
- cellpose
- segmentation
- toxoplasma
- microscopy
- spacr
library_name: spacr
pipeline_tag: image-segmentation
---

# Toxoplasma PV v2

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma PV v2

**What it segments.** Toxoplasma gondii parasitophorous vacuoles (PVs) in fluorescence images of infected host cells, with the parasite marked by DsRed or an anti-Toxoplasma stain.

**Version.** spaCR model-zoo key `toxoplasma_pv_v2`, training round 5 (weights `cpsam_v2_toxo_r5`). Status: Earlier version, kept so recorded runs reproduce. The default PV model in spaCR is toxoplasma_pv_v4.

**Trained on.** 556 curated fields from four lab PV screens and published-figure crops, including the 11 anchor wells.

**Date.** trained 2026-09-15

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-pv-segmentation-cpsam-r5](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5)
- Stock model it was tuned from: [cpsam_v2](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `pathogen_model_name` to `toxoplasma_pv_v2`; spaCR downloads and checks the weights on first use.
