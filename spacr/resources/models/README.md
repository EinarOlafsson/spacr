---
license: mit
---

# spaCR model zoo

Pretrained models for spaCR, hosted on HuggingFace. Pass a downloaded weight file to spaCR through the
`custom_model` setting (a path, not a bare name).

| task | version | base | headline result | model | training data |
|---|---|---|---|---|---|
| Cells from Hoechst channel | v1 | cpsam_v2 | F1 0.870 vs stock 0.301 | [cross-channel-cell-from-hoechst-cpsam](https://huggingface.co/einarolafsson/cross-channel-cell-from-hoechst-cpsam) | [dataset](https://huggingface.co/datasets/einarolafsson/cross-channel-cell-from-hoechst) |
| Nuclei from cell-mask channel | v1 | cpsam_v2 | F1 0.888 vs stock 0.201 | [cross-channel-nuclei-from-cellmask-cpsam](https://huggingface.co/einarolafsson/cross-channel-nuclei-from-cellmask-cpsam) | [dataset](https://huggingface.co/datasets/einarolafsson/cross-channel-nuclei-from-cellmask) |
| Plaque segmentation | r3 | cpsam | F1 0.856 in-domain | [toxoplasma-plaque-segmentation-cpsam](https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam) | — |
| Plaque-assay well detection | v3 | yolo11n | mAP50 0.993 | [toxoplasma-plaque-well-detector-yolo11](https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11) | — |
| Toxoplasma PV from cell-mask channel | v1 | cpsam_v2 | F1 0.606 vs stock 0.022 | [toxoplasma-from-cellmask-cpsam](https://huggingface.co/einarolafsson/toxoplasma-from-cellmask-cpsam) | [dataset](https://huggingface.co/datasets/einarolafsson/cross-channel-toxoplasma-from-cellmask) |
| Toxoplasma PV segmentation | r2 | cpsam_v2 | F1 0.864 (11 anchor wells) | [toxoplasma-pv-segmentation-cpsam](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam) | — |
| Toxoplasma PV segmentation | r5 | cpsam_v2 | CV F1 0.817 | [toxoplasma-pv-segmentation-cpsam-r5](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5) | — |
| Toxoplasma PV segmentation | r6 | cpsam_v2 | test F1 0.8602 vs stock 0.7648; CV F1 0.8168 | [toxoplasma-pv-segmentation-cpsam-r6](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6) | [dataset](https://huggingface.co/datasets/einarolafsson/toxoplasma-pv-segmentation-dataset) |

```python
from huggingface_hub import hf_hub_download
w = hf_hub_download('<repo>', '<weights file>')
```
