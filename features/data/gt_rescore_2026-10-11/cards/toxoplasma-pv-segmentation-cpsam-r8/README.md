---
license: cc-by-4.0
tags: [cellpose, cellpose-sam, microscopy, toxoplasma, parasitophorous-vacuole, spacr, candidate]
pipeline_tag: image-segmentation
---

# Toxoplasma PV v5

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma PV v5

**What it segments.** Toxoplasma gondii parasitophorous vacuoles (PVs) in fluorescence images of infected host cells, with the parasite marked by DsRed or an anti-Toxoplasma stain.

**Version.** spaCR model-zoo key `toxoplasma_pv_v5`, training round 8 (weights `cpsam_v2_toxo_r8`). Status: Alternative, not the default. The default PV model in spaCR is toxoplasma_pv_v4.

**Trained on.** 285 curated fields (lab immunofluorescence and DsRed fields plus published-figure crops), split 202 train / 33 validation / 50 test by acquisition and paper.

**Date.** trained 2026-10-09

> **Leaked hold-out.** PV v5 trained or validated on 29 of the 39 literature hold-out fields and on 3 of the 11 anchor wells. Those fields are left out of its test set on page 1, so its scores come from fields it never saw, but it cannot be compared with v1 to v4 on the literature hold-out or the anchors.

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-pv-segmentation-cpsam-r8](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8)
- Stock model it was tuned from: [cpsam_v2](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `pathogen_model_name` to `toxoplasma_pv_v5`; spaCR downloads and checks the weights on first use.
