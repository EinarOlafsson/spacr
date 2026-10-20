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

# Toxoplasma PV v1

![Page 1: scores against the stock model](cards/page1.png)

![Page 2: training facts](cards/page2.png)

![Page 3: about this model](cards/page3.png)

## About Toxoplasma PV v1

**What it segments.** Toxoplasma gondii parasitophorous vacuoles (PVs) in fluorescence images of infected host cells, with the parasite marked by DsRed or an anti-Toxoplasma stain.

**Version.** spaCR model-zoo key `toxoplasma_pv_v1`, training round 2 (weights `cpsam_v2_toxo_r2`). Status: Earlier version, kept so recorded runs reproduce. The default PV model in spaCR is toxoplasma_pv_v4.

**Trained on.** 229 lab fields: 104 NAS gold wells from round 1 plus 125 newly curated RH and ME49 fields.

**Date.** trained 2026-08-31

**Links**

- This model on Hugging Face: [einarolafsson/toxoplasma-pv-segmentation-cpsam](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam)
- Stock model it was tuned from: [cpsam_v2](https://huggingface.co/mouseland/cellpose-sam/blob/main/cpsam_v2)
- spaCR: [github.com/EinarOlafsson/spacr](https://github.com/EinarOlafsson/spacr)

**Load it in spaCR.** Set `pathogen_model_name` to `toxoplasma_pv_v1`; spaCR downloads and checks the weights on first use.
