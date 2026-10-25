.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 229 images from 2 datasets, 104 round-1 and 125 newly curated
     - F1 0.793 against stock cpsam_v2's 0.402 on the 486 strict hold-out fields at IoU 0.5; AJI 0.558 against 0.052
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.707 against stock cpsam's 0.258 on the 498 strict hold-out fields at IoU 0.5; AJI 0.657 against 0.265
   * - ``toxoplasma_plaque_v2``
       (Cellpose-SAM (cpsam_v2))
     - 488 curated fields across four domains -- 298 wells cropped from published figures, 96 phone-camera wells, 67 PFA and 27 methanol-fixed whole-well microscope scans; 27,582 plaques
     - F1 0.455 against stock cpsam_v2's 0.189 on the 282 strict hold-out fields at IoU 0.5; AJI 0.514 against 0.181
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993 on its own held-out split; on the test set shared with v2 it scores mAP50 0.8838, against v2's 0.9457
   * - ``toxoplasma_well_detector_v2``
       (YOLO26n (ultralytics 8.4.155))
     - plate images and literature figures; 1,070 train / 254 val / 129 test, split by PMC article so no paper is in two sets; training data at einarolafsson/toxoplasma-plaque-well-detector-dataset
     - mAP50 0.9457 and mAP50-95 0.8341 against v1's (yolo_welldetect_v3.pt) 0.8838 and 0.7630 on the SAME test set; stock YOLO has no plaque-well class, so v1 is the baseline
   * - ``toxoplasma_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOST CELL MASK channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5
   * - ``toxoplasma_pv_v2``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 556 curated images accumulated over five rounds
     - F1 0.713 against stock cpsam_v2's 0.364 on the 258 strict hold-out fields at IoU 0.5; AJI 0.451 against 0.049
   * - ``toxoplasma_pv_v3``
       (Cellpose-SAM (cpsam_v2))
     - the 556 curated PV fields of round 5, split 437 train / 108 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset
     - F1 0.750 against stock cpsam_v2's 0.390 on the 269 strict hold-out fields at IoU 0.5; AJI 0.528 against 0.055
   * - ``toxoplasma_pv_v4``
       (Cellpose-SAM (cpsam_v2))
     - round 6's 556 curated fields plus 80 hand-curated fields of a new plate (Anu revision, Replication09182026 plate 1); 502 train / 123 validation / 11 test; training data at einarolafsson/toxoplasma-pv-segmentation-dataset-r7
     - F1 0.761 against stock cpsam_v2's 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040
   * - ``toxoplasma_pv_v5``
       (Cellpose-SAM (cpsam_v2, Cellpose 4.0.9))
     - 285 curated PV fields (lab fields and published-figure crops), 13,814 vacuoles; 202 train / 33 validation / 50 test grouped by acquisition and paper
     - F1 0.733 against stock cpsam_v2's 0.383 on the 481 strict hold-out fields at IoU 0.5; AJI 0.585 against 0.051
   * - ``live_cell_v1``
       (Cellpose-SAM (cpsam_v2))
     - 11,007 transmitted-light fields from 14 public datasets, split by acquisition 6,778 train / 2,030 validation / 2,199 test; training data at einarolafsson/live-cell-segmentation-dataset
     - on the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell
   * - ``nuclei_from_cellmask_v1``
       (Cellpose-SAM (cpsam_v2))
     - nuclei predicted from the HOST CELL MASK channel alone; 453 well-grouped held-out fields, hosts HFF/HeLa/THP1
     - F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5
   * - ``cell_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - the HOST CELL outline predicted from the Hoechst (nuclear) channel alone; 2,578 training fields and 451 held-out test fields, split by well so no well is on both sides
     - F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569
   * - ``toxoplasma_from_hoechst_v1``
       (Cellpose-SAM (cpsam_v2))
     - Toxoplasma PV masks predicted from the HOECHST channel alone; 2567 training and 463 held-out fields, split by well, hosts HFF/HeLa/THP1
     - F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5
   * - ``toxoplasma_plaque_v3``
       (Cellpose-SAM (cpsam, Cellpose 4.0.9))
     - 496 curated plaque fields, including 34 reviewed empty negatives and 71 Gel Doc wells; 100 epochs; fixed physical-plate/source groups
     - F1 0.732 against stock cpsam's 0.128 on the 232 strict hold-out fields at IoU 0.5; AJI 0.628 against 0.173
   * - ``toxoplasma_plaque_v4``
       (Cellpose-SAM (cpsam_v2, Cellpose 4.0.9))
     - 663 curated plaque fields from four domains (published-figure wells, Gel Doc wells, lab plates, staged literature crops), 32,717 plaques; 335 train / 65 validation / 263 test
     - F1 0.765 against stock cpsam_v2's 0.338 on the 329 strict hold-out fields at IoU 0.5; AJI 0.662 against 0.261
   * - ``toxoplasma_plaque_v5``
       (Cellpose-SAM (cpsam_v2, Cellpose 4.2.1.1))
     - 1,507 curated plaque fields of 2026-10-15 (Gel Doc wells, lab plates, published figures, earlier curation rounds); 1,201 train / 153 validation / 153 test
     - F1 0.816 against stock cpsam_v2's 0.227 on the 153 strict hold-out fields at IoU 0.5; AJI 0.737 against 0.178
   * - ``toxoplasma_plaque_v6``
       (Cellpose-SAM (cpsam_v2, Cellpose 4.2.1.1))
     - 1,520 curated plaque fields (Gel Doc wells, lab plates, published figures, earlier curation rounds); 1,196 train / 164 validation / 160 test, grouped per-domain split of 2026-10-24
     - F1 0.768 against stock cpsam_v2's 0.294 on the 160 strict hold-out fields at IoU 0.5; AJI 0.752 against 0.249
   * - ``toxoplasma_well_detector_v3``
       (YOLO11n (fine-tuned from detector v3))
     - 452 reviewed training images; 124 validation images; physical plate and figure groups; 150 epochs; YOLO11n v3 initialization
     - Stock was not evaluated in this run; see the named incumbent comparison on the model card
