Toxoplasma PV v1
----------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. 115 pairs (104 train / 11 test), 100 epochs, base cpsam_v2

**Measured.** F1 0.867 against 0.713 for stock cpsam, at IoU 0.5

* F1 0.867 at IoU 0.5 against 0.713 for stock cpsam; AJI 0.808 against 0.426
* accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam>`_, as ``cpsam_v2_toxo_r2``.

SHA-256 ``182d8cf6b32c7b9ef2917c85870d188486e5e119f05e9c5c1f07652f6859f2d0``.

Toxoplasma Plaque v1
--------------------

**Architecture.** Cellpose-SAM (cpsam)

**Trained on.** Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set

**Measured.** F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)

* F1 0.856 in-domain and 0.806 on the literature set (3-fold cross-validated, SD 0.020), against 0.718 for round 1
* round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam>`_, as ``cpsam_plaque_r3``.

SHA-256 ``eeecd2d6cd5cbb4dddee71564d5f460d26bb07ac125e0b494b7502fea4292d5d``.

Toxoplasma Plaque Well Detector v1
----------------------------------

**Architecture.** YOLO11n

**Trained on.** whole-plate and multi-well Toxoplasma plaque-assay images; yolo11n base, 150 epochs, batch 16, imgsz 640

**Measured.** mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987

* mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987
* locates WELLS, not plaques; it is the front half of a two-stage pipeline with toxoplasma_plaque_v1, and the well it finds also gives the diameter that makes areas comparable across microscopes

Published as `einarolafsson/toxoplasma-plaque-well-detector-yolo11 <https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11>`_, as ``yolo_welldetect_v3.pt``.

SHA-256 ``b826058754fb5d4df36c3a7283aac049015cbb044b5ef096c55d19f37172a50c``.
