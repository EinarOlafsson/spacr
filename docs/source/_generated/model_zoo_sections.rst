Toxoplasma PV v1
----------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. Round 2: 229 training images (round 1's 104 plus 125 newly curated RH and ME49 fields), 100 epochs, base cpsam_v2

**Measured.** F1 0.793 against stock cpsam_v2's 0.402 on the 486 strict hold-out fields at IoU 0.5; AJI 0.558 against 0.052

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7930
     - 0.4016
     - 0.3914
   * - Boundary F1 (2 px)
     - 0.8271
     - 0.2584
     - 0.5687
   * - Precision
     - 0.8604
     - 0.2921
     - 0.5683
   * - Recall
     - 0.7354
     - 0.6427
     - 0.0927
   * - AJI
     - 0.5579
     - 0.0517
     - 0.5062
   * - Dice
     - 0.7941
     - 0.1149
     - 0.6792

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 486 scored fields and 14315 annotated objects.

* F1 0.864 at IoU 0.5 against 0.713 for stock cpsam on the 11 wells round 1 also held out (round 1 scored 0.867); AJI 0.809 against 0.426
* accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry
* the held-out literature scorecard is pending a stock-seeded re-curation; on the current literature set, whose truth leans toward this model's lineage, it ties stock Cellpose-SAM on detection (F1 0.403 against 0.400)
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.793 at IoU 0.5 on n = 486 of 655 fields (n differs between models); common subset unseen by every pv model, n = 163, F1 0.783 (95% bootstrap CI 0.710 to 0.830). Without the 98 paged round-3 fields (n = 65): F1 0.732. Earlier numbers in this row are superseded by this re-score where they differ. Default unchanged.

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam>`_, as ``cpsam_v2_toxo_r2``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma PV v1: cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma PV v1: cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma PV v1: cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam/blob/main/cards/page3.png

SHA-256 ``182d8cf6b32c7b9ef2917c85870d188486e5e119f05e9c5c1f07652f6859f2d0``.

Toxoplasma Plaque v1
--------------------

**Architecture.** Cellpose-SAM (cpsam)

**Trained on.** Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set

**Measured.** F1 0.707 against stock cpsam's 0.258 on the 498 strict hold-out fields at IoU 0.5; AJI 0.657 against 0.265

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7074
     - 0.2581
     - 0.4493
   * - Boundary F1 (2 px)
     - 0.6794
     - 0.3269
     - 0.3525
   * - Precision
     - 0.8298
     - 0.2050
     - 0.6248
   * - Recall
     - 0.6165
     - 0.3484
     - 0.2680
   * - AJI
     - 0.6572
     - 0.2649
     - 0.3923
   * - Dice
     - 0.8191
     - 0.4667
     - 0.3524

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 498 scored fields and 19993 annotated objects.

* F1 0.856 in-domain and 0.806 on the literature set (3-fold cross-validated, SD 0.020), against 0.718 for round 1
* round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay
* PREFER THIS ONE FOR MICROSCOPE-ONLY WORK. On the round-5 test split it scores 0.836 on PFA-fixed wells against round 5's 0.808, at precision 0.93 against 0.81. For mixed sources, or any phone-camera image, use toxoplasma_plaque_v2, which round 3 cannot handle at all (0.249 there)
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.707 at IoU 0.5 on n = 498 of 562 fields (n differs between models); common subset unseen by every plaque model, n = 114, F1 0.402 (95% bootstrap CI 0.300 to 0.504). Without the v3-seeded Gel Doc truth (n = 81): F1 0.472. Earlier numbers in this row are superseded by this re-score where they differ. 263-field set seen counts: v1 54, v2 178, v3 137, v4 0; GRA8 Plate1 (hold-out of v3) is the same physical plate as ReimagedPlate1 (training data of v4). Default unchanged.

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam>`_, as ``cpsam_plaque_r3``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma Plaque v1: cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma Plaque v1: cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma Plaque v1: cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam/blob/main/cards/page3.png

SHA-256 ``eeecd2d6cd5cbb4dddee71564d5f460d26bb07ac125e0b494b7502fea4292d5d``.

Toxoplasma Plaque v2 (round 5)
------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma plaque assays stained with crystal violet, from three microscopes and from published figures. Round 5: 332 training fields grouped by figure and by plate so none straddles the split, 100 epochs, base cpsam_v2, empty wells kept as negatives

**Measured.** F1 0.455 against stock cpsam_v2's 0.189 on the 282 strict hold-out fields at IoU 0.5; AJI 0.514 against 0.181

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.4548
     - 0.1891
     - 0.2657
   * - Boundary F1 (2 px)
     - 0.5679
     - 0.2447
     - 0.3232
   * - Precision
     - 0.5004
     - 0.2007
     - 0.2997
   * - Recall
     - 0.4169
     - 0.1787
     - 0.2381
   * - AJI
     - 0.5145
     - 0.1812
     - 0.3332
   * - Dice
     - 0.6961
     - 0.3141
     - 0.3820

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 282 scored fields and 8130 annotated objects.

* no longer the default: toxoplasma_plaque_v3 replaced it on 2026-10-10 (F1 0.844 against 0.799 on 263 held-out fields; this model scores 0.013 on Gel Doc wells), and toxoplasma_plaque_v4 on 2026-10-11; kept so recorded runs reproduce
* COMPLEMENTS toxoplasma_plaque_v1 rather than replacing it: prefer v1 (round 3) for microscope-only work, where it scores 0.836 against this model's 0.808 on PFA-fixed wells and is far more precise; prefer this one for mixed or unknown sources
* the only plaque model trained on phone-camera wells -- F1 0.415 against round 3's 0.249, though recall there is 0.296, so it still misses most plaques on phone images and is not yet a counting tool
* it did NOT clear the promotion bar of 0.02 literature F1 fixed before the run (it came in at -0.001), so round 3 remains production
* balanced precision/recall (0.81/0.83) where round 3 is lopsided (0.93/0.73): round 3's low recall systematically UNDERCOUNTS, which matters more than F1 for a counting assay
* hallucinates 2 objects across 6 blank-lawn wells where round 3 hallucinates 19
* first plaque model on cpsam_v2; rounds 1-4 used cpsam v1, so base and data changed together and the gap to round 3 is not attributable to the extra curation alone
* training data: https://huggingface.co/datasets/einarolafsson/toxoplasma-plaque-dataset
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.455 at IoU 0.5 on n = 282 of 562 fields (n differs between models); common subset unseen by every plaque model, n = 114, F1 0.360 (95% bootstrap CI 0.249 to 0.472). Without the v3-seeded Gel Doc truth (n = 81): F1 0.406. Earlier numbers in this row are superseded by this re-score where they differ. 263-field set seen counts: v1 54, v2 178, v3 137, v4 0; GRA8 Plate1 (hold-out of v3) is the same physical plate as ReimagedPlate1 (training data of v4). Default unchanged.

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5 <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5>`_, as ``cpsam_plaque_r5``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma Plaque v2 (round 5): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma Plaque v2 (round 5): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma Plaque v2 (round 5): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5/blob/main/cards/page3.png

SHA-256 ``0927023a745ac6a19bae0ec72c89b7b864a4ff8d047a41f3f1e9767e1a4d0600``.

Toxoplasma Plaque Well Detector v1
----------------------------------

**Architecture.** YOLO11n

**Trained on.** whole-plate and multi-well Toxoplasma plaque-assay images; yolo11n base, 150 epochs, batch 16, imgsz 640

**Measured.** mAP50 0.993 on its own held-out split; on the test set shared with v2 it scores mAP50 0.8838, against v2's 0.9457

* the 0.993 is measured on v3's OWN split, which is easier than the set v2 is measured on; on that shared set this model scores mAP50 0.8838 against v2's 0.9457, so v2 is the better detector
* kept because the published plaque corpus was measured with these weights, so results in the paper trace back to this row
* locates WELLS, not plaques; it is the front half of a two-stage pipeline with toxoplasma_plaque_v1, and the well it finds also gives the diameter that makes areas comparable across microscopes

Published as `einarolafsson/toxoplasma-plaque-well-detector-yolo11 <https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11>`_, as ``yolo_welldetect_v3.pt``.

SHA-256 ``b826058754fb5d4df36c3a7283aac049015cbb044b5ef096c55d19f37172a50c``.

Toxoplasma Plaque Well Detector v2
----------------------------------

**Architecture.** YOLO26n (ultralytics 8.4.155)

**Trained on.** whole-plate and multi-well Toxoplasma plaque-assay images plus 939 newly reviewed PMC figures, accepted boxes and confirmed negatives alike; yolo26n base, best validation mAP50-95 at epoch 28

**Measured.** mAP50 0.9457 and mAP50-95 0.8341 against v1's (yolo_welldetect_v3.pt) 0.8838 and 0.7630 on the SAME test set; stock YOLO has no plaque-well class, so v1 is the baseline

* on the shared test set it beats v1 (the v3 weights) on every measure, and cuts false boxes on no-well figures from 152 to 49
* 84 of the 129 test images contain no well at all, which is what the false-box count is measured on
* locates WELLS, not plaques; the front half of a two-stage pipeline with the plaque segmentation model
* the repository publishes this weight as weights/best.pt; spaCR saves it under the name above so two detectors cannot both land as best.pt

Published as `einarolafsson/toxoplasma-plaque-well-detector-yolo26 <https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo26>`_, as ``yolo_welldetect_v4.pt``.

SHA-256 ``f2a1e1110f09b2a1d5ef5545adaba7c57f1158669d0bfc50d8fabe9f86da30c7``.

Toxoplasma from Cell Mask (cross-channel)
-----------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** cross-channel: given the host cell image, predicts where the Toxoplasma parasitophorous vacuoles are, with no parasite stain. 100 epochs, base cpsam_v2, AdamW lr 1e-5, targets are PV-regenerated masks

**Measured.** F1 0.606 against 0.021 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5

* F1 0.606, AJI 0.494, Dice 0.610 at IoU 0.5 against stock cpsam_v2's 0.021/0.008/0.020 -- stock cannot do this task at all
* per host: HeLa 0.711, HFF 0.557, THP1 0.465; THP1 is the weak case
* the held-out split selects the checkpoint, so it is validation data rather than an independent test set
* accuracy falls above IoU 0.8 -- suited to counting, occupancy and area rather than precise morphometry

Published as `einarolafsson/toxoplasma-from-cellmask-cpsam <https://huggingface.co/einarolafsson/toxoplasma-from-cellmask-cpsam>`_, as ``toxoplasma_from_cellmask_pv``.

SHA-256 ``481dfccc1a68cc594aafcb71088efc25b5f5c6a6240e52902c0089759b3149ab``.

Toxoplasma PV v2 (round 5)
--------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen (RH and ME49). Round 5: 556 images, 100 epochs, base cpsam_v2

**Measured.** F1 0.713 against stock cpsam_v2's 0.364 on the 258 strict hold-out fields at IoU 0.5; AJI 0.451 against 0.049

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7130
     - 0.3638
     - 0.3492
   * - Boundary F1 (2 px)
     - 0.7340
     - 0.2301
     - 0.5040
   * - Precision
     - 0.7658
     - 0.2560
     - 0.5098
   * - Recall
     - 0.6670
     - 0.6287
     - 0.0383
   * - AJI
     - 0.4507
     - 0.0488
     - 0.4019
   * - Dice
     - 0.6782
     - 0.1116
     - 0.5666

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 258 scored fields and 5766 annotated objects.

* supersedes toxoplasma_pv_v1 (round 2, 229 images): more than twice the training data and cross-validated rather than single-split
* 5-fold CV over 619 pairs: F1 0.817 (SD 0.036), AJI 0.714, Dice 0.802
* per-dataset variance is real -- F1 ranges ~0.74 to ~0.93 by screen
* accuracy falls above IoU 0.8 -- suited to counting and area rather than precise morphometry
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.713 at IoU 0.5 on n = 258 of 655 fields (n differs between models); common subset unseen by every pv model, n = 163, F1 0.761 (95% bootstrap CI 0.691 to 0.806). Without the 98 paged round-3 fields (n = 65): F1 0.742. Earlier numbers in this row are superseded by this re-score where they differ. Default unchanged.

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r5 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5>`_, as ``cpsam_v2_toxo_r5``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma PV v2 (round 5): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma PV v2 (round 5): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma PV v2 (round 5): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5/blob/main/cards/page3.png

SHA-256 ``17c689e3b117745561e20a885c2a2a998ed360fa97cac8c0446316ae5905c10f``.

Toxoplasma PV v3 (round 6)
--------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen (RH and ME49). Round 6 retrains round 5's data with cellpose 4.2.1.1, 100 epochs, base cpsam_v2

**Measured.** F1 0.750 against stock cpsam_v2's 0.390 on the 269 strict hold-out fields at IoU 0.5; AJI 0.528 against 0.055

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7504
     - 0.3899
     - 0.3606
   * - Boundary F1 (2 px)
     - 0.7826
     - 0.2537
     - 0.5289
   * - Precision
     - 0.7998
     - 0.2794
     - 0.5204
   * - Recall
     - 0.7068
     - 0.6447
     - 0.0621
   * - AJI
     - 0.5278
     - 0.0553
     - 0.4725
   * - Dice
     - 0.7499
     - 0.1236
     - 0.6263

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 269 scored fields and 6443 annotated objects.

* NEWEST IS NOT BEST HERE: round 6 does not beat round 2 on the anchor wells -- 0.8602 against 0.8648 -- and the PV project still promotes round 5
* the shipped weights are the FINAL epoch 100, not a validation-chosen checkpoint: the training records show no checkpoint selection (recorded 2026-10-12)
* 5-fold cross-validation, grouped by source: F1 0.8168 +/- 0.028, AJI 0.7516, Dice 0.8424
* the 11 anchor wells have been held out since round 1, so they are the only fields no PV round has ever trained on
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.750 at IoU 0.5 on n = 269 of 655 fields (n differs between models); common subset unseen by every pv model, n = 163, F1 0.790 (95% bootstrap CI 0.726 to 0.835). Without the 98 paged round-3 fields (n = 65): F1 0.754. Earlier numbers in this row are superseded by this re-score where they differ. Default unchanged.

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r6 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6>`_, as ``cpsam_v2_toxo_r6``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma PV v3 (round 6): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma PV v3 (round 6): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma PV v3 (round 6): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6/blob/main/cards/page3.png

SHA-256 ``146ef269979b1d1ab45c11039b0ab164f68001adaa8f73f1f8f18be6fcfd060e``.

Toxoplasma PV v4 (round 7)
--------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles: round 6's DsRed and anti-Toxoplasma-biotin fields plus the Toxoplasma channel of a new 384-well plate. Round 7, cellpose 4.2.1.1, 100 epochs, base cpsam_v2

**Measured.** F1 0.761 against stock cpsam_v2's 0.377 on the 209 strict hold-out fields at IoU 0.5; AJI 0.582 against 0.040

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7611
     - 0.3771
     - 0.3840
   * - Boundary F1 (2 px)
     - 0.8053
     - 0.2229
     - 0.5825
   * - Precision
     - 0.7687
     - 0.2647
     - 0.5040
   * - Recall
     - 0.7538
     - 0.6557
     - 0.0981
   * - AJI
     - 0.5820
     - 0.0396
     - 0.5424
   * - Dice
     - 0.8348
     - 0.0906
     - 0.7442

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 209 scored fields and 5515 annotated objects.

* Scorecard metrics are means over nonempty fields, not pooled object counts; see aggregation.json.
* NEWEST IS NOT BEST HERE: on 20 held-out fields of the new plate it was trained on, round 7 scores F1 0.748 against 0.899 for round 6 (recall 0.636 against 0.829) -- prefer toxoplasma_pv_v3 on that plate
* on the 11 anchor wells it ties round 6 (0.8536 against 0.8602); 5-fold cross-validation F1 0.8142 +/- 0.015 against round 6's 0.8168
* it traces the vacuoles it finds more tightly than round 6 (AJI 0.827 against 0.786 on the new plate) but finds fewer of them
* the drop on the new plate is under investigation: round 7 also under-fits that plate's own training fields, which points at how those fields entered training rather than at generalisation
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.761 at IoU 0.5 on n = 209 of 655 fields (n differs between models); common subset unseen by every pv model, n = 163, F1 0.781 (95% bootstrap CI 0.725 to 0.817). Without the 98 paged round-3 fields (n = 65): F1 0.757. Earlier numbers in this row are superseded by this re-score where they differ. Default unchanged.

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r7 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7>`_, as ``cpsam_v2_toxo_r7``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma PV v4 (round 7): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma PV v4 (round 7): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma PV v4 (round 7): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/blob/main/cards/page3.png

SHA-256 ``621a475c9bfb6865be7de12c5c4638f4509d734e0c4d39adb15ea541e48961d2``.

Toxoplasma PV v5 (round 8, alternative)
---------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2, Cellpose 4.0.9)

**Trained on.** Toxoplasma parasitophorous vacuoles from lab immunofluorescence and DsRed fields and published-figure crops. Round 8, cellpose 4.0.9, best of 100 epochs (epoch 20), base cpsam_v2

**Measured.** F1 0.733 against stock cpsam_v2's 0.383 on the 481 strict hold-out fields at IoU 0.5; AJI 0.585 against 0.051

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7329
     - 0.3831
     - 0.3497
   * - Boundary F1 (2 px)
     - 0.8198
     - 0.2567
     - 0.5631
   * - Precision
     - 0.6459
     - 0.2749
     - 0.3710
   * - Recall
     - 0.8469
     - 0.6322
     - 0.2146
   * - AJI
     - 0.5846
     - 0.0509
     - 0.5337
   * - Dice
     - 0.8098
     - 0.1119
     - 0.6979

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 481 scored fields and 16213 annotated objects.

* ALTERNATIVE, NOT BETTER: on the 15 test fields round 7 never saw it scores F1 0.536 against toxoplasma_pv_v4's 0.525 -- a tie; prefer toxoplasma_pv_v4
* 35 of the 50 test fields were in round 7's training or validation set, so the all-field comparison (0.792 against 0.851) favours toxoplasma_pv_v4 and is not a fair test
* weak on published-figure crops (F1 0.440 on 14 literature fields), as is every PV model
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.733 at IoU 0.5 on n = 481 of 655 fields (n differs between models); common subset unseen by every pv model, n = 163, F1 0.633 (95% bootstrap CI 0.544 to 0.695). Without the 98 paged round-3 fields (n = 65): F1 0.665. Earlier numbers in this row are superseded by this re-score where they differ. LEAKED: this model trained or validated on 29 of 39 curated PV literature hold-out fields and 3 of 11 anchor wells, so its earlier hold-out scores are not hold-out scores; the re-score uses only the fields it never saw. Default unchanged.

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r8 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8>`_, as ``cpsam_v2_toxo_r8``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma PV v5 (round 8, alternative): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma PV v5 (round 8, alternative): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma PV v5 (round 8, alternative): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/blob/main/cards/page3.png

SHA-256 ``a1d7f1a978cf4925ec60db486a00eea33e87e6fc5a6b0d0f34c6db5f132d5c1c``.

Live cell v1 (phase, brightfield, DIC; alternative)
---------------------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** unstained cells in phase contrast, brightfield and DIC: LIVECell, DeepSea, YeaZ, yeast microstructures, five Cell Tracking Challenge sets, BBBC009, BBBC030, QPI and Revvity. Base cpsam_v2, cellpose 4.2.1.1, lr 1e-5, batch 4; stopped at epoch 37 of 100

**Measured.** on the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell

* an optional alternative: stock Cellpose-SAM (cpsam_v2) stays the default for label-free and live-cell segmentation
* TWO STOCK COMPARISONS, NOT ONE: stock cpsam_v2 trained on LIVECell and YeaZ, so its score there is partly memorisation. On the datasets it never saw (DeepSea, the CTC sets, BBBC009, BBBC030, QPI, Revvity, yeast microstructures) this model scores F1 0.960 against 0.885
* it does NOT replace stock on LIVECell-style Incucyte phase: 0.671 against 0.724 there, and it missed its own pre-registered promotion bar
* by modality at IoU 0.5: brightfield 0.964 (stock 0.912), DIC +0.026 over stock, phase 0.689 (stock 0.735)
* F1 0.865 on a train sample, 0.696 on validation and 0.694 on test; no per-epoch loss was recorded

Published as `einarolafsson/live-cell-segmentation-cpsam <https://huggingface.co/einarolafsson/live-cell-segmentation-cpsam>`_, as ``live_cell_v1``.

SHA-256 ``7ade69377093fe81830ddc7c52ba8618bef1fefe7d1243c01b9c1beed7fcb090``.

Cross-channel nuclei-from-cellmask
----------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** cross-channel: given the cell image, predicts where the nuclei are, with no nuclear stain -- which frees the DAPI/Hoechst channel for another marker. 100 epochs, base cpsam_v2

**Measured.** F1 0.888 against 0.201 for stock cpsam_v2 on 453 well-grouped held-out fields, at IoU 0.5

* F1 0.888, AJI 0.792, Dice 0.877 at IoU 0.5 against stock cpsam_v2's 0.201/0.286/0.449
* per host: HFF 0.932, HeLa 0.860, THP1 0.861
* the held-out split selects the checkpoint, so it is validation data rather than an independent test set
* predicts nuclei from cell morphology -- expect degraded accuracy on unusual or highly confluent morphologies

Published as `einarolafsson/cross-channel-nuclei-from-cellmask-cpsam <https://huggingface.co/einarolafsson/cross-channel-nuclei-from-cellmask-cpsam>`_, as ``nuclei_from_cellmask_best``.

SHA-256 ``2675553a46e97a7bc4bd2bfe3e954954194fe71ca4e94261e752a02bf0b6eb47``.

Cross-channel cell-from-hoechst
-------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Hoechst-stained nuclei paired with curated host-cell masks; fine-tuned from stock cpsam_v2, 100 epochs, best epoch 70

**Measured.** F1 0.870 against stock cpsam_v2's 0.301 on 451 held-out fields at IoU 0.5 -- a delta of 0.569

* the counterpart of nuclei_from_cellmask_v1: that one predicts nuclei from the cell mask, this one predicts the cell from the nucleus
* precision 0.944 against recall 0.806 -- it misses cells rather than inventing them, which is the safer direction for counting
* quote the DELTA over stock (0.569), not the ratio: stock's mAP of 0.0575 is a near-zero denominator that makes any ratio look huge

Published as `einarolafsson/cross-channel-cell-from-hoechst-cpsam <https://huggingface.co/einarolafsson/cross-channel-cell-from-hoechst-cpsam>`_, as ``cell_from_hoechst_best``.

SHA-256 ``d1992433b4f2f291f73738830bb953198165fdc10719e54dae9b8c2bd430e0eb``.

Toxoplasma from Hoechst (cross-channel)
---------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** cross-channel: given the Hoechst/nuclear image, predicts where the Toxoplasma parasitophorous vacuoles are, with no parasite stain. 100 epochs, base cpsam_v2, AdamW lr 1e-05, targets are PV-regenerated masks

**Measured.** F1 0.569 against 0.002 for stock cpsam_v2 on 463 well-grouped held-out fields, at IoU 0.5

* F1 0.569, AJI 0.421, Dice 0.546 at IoU 0.5 against stock cpsam_v2's 0.002/0.006/0.016
* the Hoechst route is harder than the cell-mask route -- compare toxoplasma_from_cellmask_v1
* the held-out split selects the checkpoint, so it is validation data rather than an independent test set
* accuracy falls above IoU 0.8 -- suited to counting, occupancy and area rather than precise morphometry

Published as `einarolafsson/toxoplasma-from-hoechst-cpsam <https://huggingface.co/einarolafsson/toxoplasma-from-hoechst-cpsam>`_, as ``toxoplasma_from_hoechst_pv``.

SHA-256 ``8dc05ebced3550d1a418c13d24d319e0482c742988df29a525520026cb2f0d96``.

Toxoplasma Plaque v3 (Gel Doc r5, alternative)
----------------------------------------------

**Architecture.** Cellpose-SAM (cpsam, Cellpose 4.0.9)

**Trained on.** 496 curated plaque fields, including 34 reviewed empty negatives and 71 Gel Doc wells; 100 epochs; fixed physical-plate/source groups

**Measured.** F1 0.732 against stock cpsam's 0.128 on the 232 strict hold-out fields at IoU 0.5; AJI 0.628 against 0.173

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7322
     - 0.1280
     - 0.6042
   * - Boundary F1 (2 px)
     - 0.6211
     - 0.2157
     - 0.4054
   * - Precision
     - 0.6763
     - 0.0827
     - 0.5936
   * - Recall
     - 0.7981
     - 0.2825
     - 0.5156
   * - AJI
     - 0.6275
     - 0.1729
     - 0.4547
   * - Dice
     - 0.8105
     - 0.3337
     - 0.4768

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 232 scored fields and 8379 annotated objects.

* Recommended alternative for Gel Doc wells: on the 153 test fields of the curated 2026-10-15 set it scores object F1 0.954 on Gel Doc wells against 0.868 for the default toxoplasma_plaque_v5 (0.741 against 0.816 overall).
* Alternative since 2026-10-11, when toxoplasma_plaque_v4 (base cpsam_v2) became the default by the maintainer's choice. This model scores higher: on 263 held-out fields F1 0.844 against v4 0.815, and 0.913 against 0.804 on Gel Doc wells, through spacr.plaque.segment_plaque_image. Kept so recorded runs reproduce.
* Default 2026-10-10 to 2026-10-11: F1 0.844 against toxoplasma_plaque_v2 0.799, and 0.913 against 0.013 on Gel Doc wells (features/data/model_compare_2026-10-10).
* Its own release gate on independent literature crops did not pass at the time (literature F1 about 0.54 for every model).
* Gel Doc score requires min_size=0; min_size=150 gives F1 0.1041.
* Stock not measured; comparison baseline is incumbent r3.
* Distinct from the cpsam_v2 round-5 model already registered as toxoplasma_plaque_v2.
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.732 at IoU 0.5 on n = 232 of 562 fields (n differs between models); common subset unseen by every plaque model, n = 114, F1 0.548 (95% bootstrap CI 0.449 to 0.645). Without the v3-seeded Gel Doc truth (n = 81): F1 0.444. Earlier numbers in this row are superseded by this re-score where they differ. 263-field set seen counts: v1 54, v2 178, v3 137, v4 0; GRA8 Plate1 (hold-out of v3) is the same physical plate as ReimagedPlate1 (training data of v4). Default unchanged.

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc>`_, as ``cpsam_plaque_r5_geldoc``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma Plaque v3 (Gel Doc r5, alternative): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma Plaque v3 (Gel Doc r5, alternative): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma Plaque v3 (Gel Doc r5, alternative): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/blob/main/cards/page3.png

SHA-256 ``8f59850dbe09c3a252728b0a7ca38ae5b52a28f2dbaa7a068174b4b7bfac768c``.

Toxoplasma Plaque v4 (round 6, cpsam_v2)
----------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2, Cellpose 4.0.9)

**Trained on.** Toxoplasma plaques from published-figure wells, Bio-Rad Gel Doc wells, lab plates and staged literature crops; round 6 splits, best of 190 epochs (epoch 140), base cpsam_v2

**Measured.** F1 0.765 against stock cpsam_v2's 0.338 on the 329 strict hold-out fields at IoU 0.5; AJI 0.662 against 0.261

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.7648
     - 0.3380
     - 0.4267
   * - Boundary F1 (2 px)
     - 0.6729
     - 0.3539
     - 0.3190
   * - Precision
     - 0.7344
     - 0.2911
     - 0.4433
   * - Recall
     - 0.7977
     - 0.4030
     - 0.3947
   * - AJI
     - 0.6619
     - 0.2610
     - 0.4010
   * - Dice
     - 0.8388
     - 0.4291
     - 0.4097

Evaluation set: ``gt_rescore_20261011``; version ``strict hold-out, pooled``; 329 scored fields and 13915 annotated objects.

* No longer the default: toxoplasma_plaque_v5 replaced it on 2026-10-17 (object F1 0.816 against 0.725 on the 153 curated test fields); kept selectable so recorded runs reproduce.
* The default plaque model 2026-10-11 to 2026-10-17, by the maintainer's rule that spaCR models build on cpsam_v2. It scores below toxoplasma_plaque_v3 on the same 263 held-out fields, through spacr.plaque.segment_plaque_image: F1 0.815 against 0.844, lower on every source (Gel Doc 0.804 against 0.913); v3 stays selectable.
* Fine-tuned from cpsam_v2 (2026-10-10). It replaces the first round-6 weights, fine-tuned from first-generation cpsam (sha256 00f4a86b...a0c171), which remain in the repository history.
* Above toxoplasma_plaque_v2 overall (0.799) and on Gel Doc wells, where v2 scores 0.013.
* Published-figure literature crops stay hard (F1 0.530).
* Re-scored 2026-10-11 on curated ground truth (spaCR pipeline call, defaults; receipt features/data/model_rescore_curated_gt_2026-10-11/receipt.json): strict hold-out, fields this model never saw, F1 0.765 at IoU 0.5 on n = 329 of 562 fields (n differs between models); common subset unseen by every plaque model, n = 114, F1 0.534 (95% bootstrap CI 0.440 to 0.625). Without the v3-seeded Gel Doc truth (n = 81): F1 0.459. Earlier numbers in this row are superseded by this re-score where they differ. 263-field set seen counts: v1 54, v2 178, v3 137, v4 0; GRA8 Plate1 (hold-out of v3) is the same physical plate as ReimagedPlate1 (training data of v4). Default unchanged.

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6 <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6>`_, as ``cpsam_plaque_r6``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma Plaque v4 (round 6, cpsam_v2): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma Plaque v4 (round 6, cpsam_v2): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma Plaque v4 (round 6, cpsam_v2): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/blob/main/cards/page3.png

SHA-256 ``744aee3f71efa68f6a369a6f91fe5850dc61e35f3dc20d53dda2c04fce17b35f``.

Toxoplasma Plaque v5 (round 7, cpsam_v2, default)
-------------------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2, Cellpose 4.2.1.1)

**Trained on.** Toxoplasma plaques from Gel Doc wells, lab plates, published figures and earlier curation rounds (curated dataset of 2026-10-15); 100 epochs, final epoch kept, base cpsam_v2

**Measured.** F1 0.816 against stock cpsam_v2's 0.227 on the 153 strict hold-out fields at IoU 0.5; AJI 0.737 against 0.178

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - F1
     - 0.8157
     - 0.2268
     - 0.5889
   * - Boundary F1 (2 px)
     - 0.7585
     - 0.2883
     - 0.4702
   * - Precision
     - 0.8674
     - 0.1894
     - 0.6779
   * - Recall
     - 0.7699
     - 0.2824
     - 0.4875
   * - AJI
     - 0.7366
     - 0.1784
     - 0.5582
   * - Dice
     - 0.8962
     - 0.3291
     - 0.5671

Evaluation set: ``plaque_curated_20261015_test``; version ``strict hold-out, pooled``; 153 scored fields and 6540 annotated objects.

* The default plaque model since 2026-10-17, by the maintainer's choice; toxoplasma_plaque_v3 is the recommended alternative for Gel Doc wells and toxoplasma_plaque_v4 stays selectable.
* Scored 2026-10-17 on the 153 test fields of the curated 2026-10-15 dataset (spacr.plaque.segment_plaque_image, defaults): object F1 0.816 (95% bootstrap CI 0.778 to 0.848), v4 0.725, v3 0.741, stock cpsam_v2 0.227. On the 81 test fields neither v3 nor v4 saw: v5 0.803, v4 0.661, v3 0.684. AJI is level with v4 (0.737 against 0.744); Gel Doc wells 0.868 against v4 0.901 and v3 0.954.
* The final epoch 100 ships by the maintainer's rule: an earlier checkpoint replaces it only if better on every metric in every domain. Epoch 40 led on validation object F1 (0.766 against 0.750) but trailed on boundary F1 (0.700 against 0.711), AJI (0.661 against 0.667) and Dice (0.857 against 0.869); on test it scored F1 0.823.
* Training ran 100 epochs; indices 91 to 99 were resumed from epoch 90 after a power cut, with the same schedule and seeds (optimizer moments restarted at LR 2e-8).

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7 <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7>`_, as ``cpsam_plaque_r7``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7/resolve/main/cards/page1.png
   :alt: Published scorecard for Toxoplasma Plaque v5 (round 7, cpsam_v2, default): cards/page1.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7/blob/main/cards/page1.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7/resolve/main/cards/page2.png
   :alt: Published scorecard for Toxoplasma Plaque v5 (round 7, cpsam_v2, default): cards/page2.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7/blob/main/cards/page2.png

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7/resolve/main/cards/page3.png
   :alt: Published scorecard for Toxoplasma Plaque v5 (round 7, cpsam_v2, default): cards/page3.png
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r7/blob/main/cards/page3.png

SHA-256 ``9243917beefbc3d1c8d997d56c1980b811ffef9ee075dac60b099aec715fa8a1``.

Toxoplasma Well Detector v3 (YOLO11 Gel Doc, default)
-----------------------------------------------------

**Architecture.** YOLO11n (fine-tuned from detector v3)

**Trained on.** 452 reviewed training images; 124 validation images; physical plate and figure groups; 150 epochs; YOLO11n v3 initialization

**Measured.** Stock was not evaluated in this run; see the named incumbent comparison on the model card.

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - mAP50
     - 0.9940
     - not measured
     - not measured
   * - mAP50-95
     - 0.8842
     - not measured
     - not measured
   * - Precision
     - 0.9698
     - not measured
     - not measured
   * - Recall
     - 0.9979
     - not measured
     - not measured

Evaluation set: ``detector_v3_validation_figure_clean``; version ``20261007T205603``; 73 scored fields and 191 annotated objects.

* The default well detector since 2026-10-12, by the maintainer's choice. On the 73 clean incumbent-validation images its mAP50-95 is lower than the incumbent's; on the 3 Gel Doc validation plates it is higher.
* Gel Doc plates selected best.pt, so their scores are validation rather than independent test.
* Distinct from the YOLO26 checkpoint also historically called detector v4.
* Locates wells, not plaques; one class plaque_well (0).

Published as `einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc <https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc>`_, as ``yolo_welldetect_v4_geldoc.pt``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc/resolve/main/scorecard.png
   :alt: Published scorecard for Toxoplasma Well Detector v3 (YOLO11 Gel Doc, default)
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc/blob/main/scorecard.csv

SHA-256 ``046861021594256957292cfe9488a84e758077a12dac31fe66b8b4083f8dfa18``.
