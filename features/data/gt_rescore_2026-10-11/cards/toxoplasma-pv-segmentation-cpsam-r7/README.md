---
license: cc-by-4.0
tags: [cellpose, cellpose-sam, segmentation, microscopy, toxoplasma, parasitophorous-vacuole, spacr]
pipeline_tag: image-segmentation
---

# Toxoplasma PV v4

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma PV v4

**What it segments.** Toxoplasma gondii parasitophorous vacuoles (PVs) in fluorescence images of infected host cells, with the parasite marked by DsRed or an anti-Toxoplasma stain.

**Version.** spaCR model-zoo key `toxoplasma_pv_v4`, training round 7 (weights `cpsam_v2_toxo_r7`). Status: Default PV model in spaCR.

**Trained on.** The round-6 fields plus 80 hand-curated fields of a new 384-well plate, split 502 train / 123 validation.

**Date.** trained 2026-09-30

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-pv-segmentation-cpsam-r7](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7)
- Stock model it was tuned from: [cpsam_v2](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `pathogen_model_name` to `toxoplasma_pv_v4`; spaCR downloads and checks the weights on first use.
