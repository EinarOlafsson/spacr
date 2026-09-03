.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out against stock
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 115 images, 1 dataset
     - F1 0.867 against 0.713 for stock cpsam, at IoU 0.5
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - Toxoplasma plaque assays; 2 datasets, in-domain and literature; image count not recorded
     - F1 0.856 in-domain and 0.834 on the literature set; no stock cpsam baseline measured
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well plaque-assay images; 1 dataset; image count not recorded
     - mAP50 0.993, mAP50-95 0.886; no stock model detects wells
