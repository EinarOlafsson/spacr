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
| Plaque segmentation | Gel Doc r5 candidate | cpsam | held-out F1 0.6419 at min_size=0; not promoted | [model and scorecards](https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc) | field assignments and hashes in model repo |
| Plaque-assay well detection | v3 | yolo11n | mAP50 0.993 | [toxoplasma-plaque-well-detector-yolo11](https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11) | — |
| Plaque-assay well detection | v4 | yolo26n | test mAP50 0.9457 (v3: 0.8838) | [toxoplasma-plaque-well-detector-yolo26](https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo26) | [dataset](https://huggingface.co/datasets/einarolafsson/toxoplasma-plaque-well-detector-dataset) |
| Plaque-assay well detection | Gel Doc v4 candidate | yolo11n | clean validation mAP50–95 0.8842; not promoted | [model and scorecards](https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc) | field assignments and hashes in model repo |
| Toxoplasma PV from cell-mask channel | v1 | cpsam_v2 | F1 0.606 vs stock 0.022 | [toxoplasma-from-cellmask-cpsam](https://huggingface.co/einarolafsson/toxoplasma-from-cellmask-cpsam) | [dataset](https://huggingface.co/datasets/einarolafsson/cross-channel-toxoplasma-from-cellmask) |
| Toxoplasma PV segmentation | r2 | cpsam_v2 | F1 0.864 (11 anchor wells) | [toxoplasma-pv-segmentation-cpsam](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam) | — |
| Toxoplasma PV segmentation | r5 | cpsam_v2 | CV F1 0.817 | [toxoplasma-pv-segmentation-cpsam-r5](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5) | — |
| Toxoplasma PV segmentation | r6 | cpsam_v2 | test F1 0.8602 vs stock 0.7648; CV F1 0.8168 | [toxoplasma-pv-segmentation-cpsam-r6](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6) | [dataset](https://huggingface.co/datasets/einarolafsson/toxoplasma-pv-segmentation-dataset) |
| Toxoplasma PV segmentation | r7 | cpsam_v2 | test F1 0.8536 vs stock 0.7648; CV F1 0.8142 | [toxoplasma-pv-segmentation-cpsam-r7](https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7) | [dataset](https://huggingface.co/datasets/einarolafsson/toxoplasma-pv-segmentation-dataset-r7) |

The Gel Doc candidates use the spaCR catalogue keys `toxoplasma_plaque_v3`
and `toxoplasma_well_detector_v3`. They have separate training lineages from
the existing cpsam_v2 plaque r5 and YOLO26 detector v4. Their model repositories
include CSV/JSON/PNG scorecards, training curves, per-image evaluations,
configuration, split provenance, checksums, SQLite databases and PDF reports.
Existing defaults remain unchanged because neither candidate passed its
promotion rule. PV round 7 (`toxoplasma_pv_v4`) also provides standard
scorecards; its headline averages 10 nonempty anchor fields from the 11-field
split. See each model card for aggregation and validation limitations.

```python
from huggingface_hub import hf_hub_download
w = hf_hub_download('<repo>', '<weights file>')
```
