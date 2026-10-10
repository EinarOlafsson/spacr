Toxoplasma PV v1
----------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen. Round 2: 229 training images (round 1's 104 plus 125 newly curated RH and ME49 fields), 100 epochs, base cpsam_v2

**Measured.** F1 0.864 against 0.713 for stock cpsam on 11 held-out in-house wells, at IoU 0.5; literature hold-out pending

* F1 0.864 at IoU 0.5 against 0.713 for stock cpsam on the 11 wells round 1 also held out (round 1 scored 0.867); AJI 0.809 against 0.426
* accuracy falls sharply above IoU 0.8 -- suited to counting and area rather than precise morphometry
* the held-out literature scorecard is pending a stock-seeded re-curation; on the current literature set, whose truth leans toward this model's lineage, it ties stock Cellpose-SAM on detection (F1 0.403 against 0.400)

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam>`_, as ``cpsam_v2_toxo_r2``.

SHA-256 ``182d8cf6b32c7b9ef2917c85870d188486e5e119f05e9c5c1f07652f6859f2d0``.

Toxoplasma Plaque v1
--------------------

**Architecture.** Cellpose-SAM (cpsam)

**Trained on.** Toxoplasma gondii plaque assays; round 3, evaluated in-domain (NAS) and against a literature generalisation set

**Measured.** F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)

* F1 0.856 in-domain and 0.806 on the literature set (3-fold cross-validated, SD 0.020), against 0.718 for round 1
* round 3 trades precision (0.939 down to 0.858) for recall (0.631 up to 0.811) on the literature set, which is the right direction for a counting assay
* PREFER THIS ONE FOR MICROSCOPE-ONLY WORK. On the round-5 test split it scores 0.836 on PFA-fixed wells against round 5's 0.808, at precision 0.93 against 0.81. For mixed sources, or any phone-camera image, use toxoplasma_plaque_v2, which round 3 cannot handle at all (0.249 there)

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam>`_, as ``cpsam_plaque_r3``.

SHA-256 ``eeecd2d6cd5cbb4dddee71564d5f460d26bb07ac125e0b494b7502fea4292d5d``.

Toxoplasma Plaque v2 (round 5)
------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma plaque assays stained with crystal violet, from three microscopes and from published figures. Round 5: 332 training fields grouped by figure and by plate so none straddles the split, 100 epochs, base cpsam_v2, empty wells kept as negatives

**Measured.** not scored against stock; on 81 held-out fields it ties round 3 on literature (0.819 vs 0.820) and beats it by 0.166 on phone-camera wells (0.415 vs 0.249)

* COMPLEMENTS toxoplasma_plaque_v1 rather than replacing it: prefer v1 (round 3) for microscope-only work, where it scores 0.836 against this model's 0.808 on PFA-fixed wells and is far more precise; prefer this one for mixed or unknown sources
* the only plaque model trained on phone-camera wells -- F1 0.415 against round 3's 0.249, though recall there is 0.296, so it still misses most plaques on phone images and is not yet a counting tool
* it did NOT clear the promotion bar of 0.02 literature F1 fixed before the run (it came in at -0.001), so round 3 remains production
* balanced precision/recall (0.81/0.83) where round 3 is lopsided (0.93/0.73): round 3's low recall systematically UNDERCOUNTS, which matters more than F1 for a counting assay
* hallucinates 2 objects across 6 blank-lawn wells where round 3 hallucinates 19
* first plaque model on cpsam_v2; rounds 1-4 used cpsam v1, so base and data changed together and the gap to round 3 is not attributable to the extra curation alone
* training data: https://huggingface.co/datasets/einarolafsson/toxoplasma-plaque-dataset

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5 <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5>`_, as ``cpsam_plaque_r5``.

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

**Measured.** F1 0.817 +/- 0.036 by 5-fold cross-validation over 619 pairs; ~0.86 against 0.713 for stock on the 11 in-house held-out wells

* supersedes toxoplasma_pv_v1 (round 2, 229 images): more than twice the training data and cross-validated rather than single-split
* 5-fold CV over 619 pairs: F1 0.817 (SD 0.036), AJI 0.714, Dice 0.802
* per-dataset variance is real -- F1 ranges ~0.74 to ~0.93 by screen
* accuracy falls above IoU 0.8 -- suited to counting and area rather than precise morphometry

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r5 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r5>`_, as ``cpsam_v2_toxo_r5``.

SHA-256 ``17c689e3b117745561e20a885c2a2a998ed360fa97cac8c0446316ae5905c10f``.

Toxoplasma PV v3 (round 6)
--------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles stained with goat anti-Toxoplasma-biotin, and tachyzoites expressing DsRed in the PV lumen (RH and ME49). Round 6 retrains round 5's data with cellpose 4.2.1.1, 100 epochs, base cpsam_v2

**Measured.** F1 0.860 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.803 against 0.505

* NEWEST IS NOT BEST HERE: round 6 does not beat round 2 on the anchor wells -- 0.8602 against 0.8648 -- and the PV project still promotes round 5
* it is the first PV round whose checkpoint was chosen on a held-out validation set (108 fields) instead of on the test wells
* 5-fold cross-validation, grouped by source: F1 0.8168 +/- 0.028, AJI 0.7516, Dice 0.8424
* the 11 anchor wells have been held out since round 1, so they are the only fields no PV round has ever trained on

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r6 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r6>`_, as ``cpsam_v2_toxo_r6``.

SHA-256 ``146ef269979b1d1ab45c11039b0ab164f68001adaa8f73f1f8f18be6fcfd060e``.

Toxoplasma PV v4 (round 7)
--------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** Toxoplasma tachyzoite parasitophorous vacuoles: round 6's DsRed and anti-Toxoplasma-biotin fields plus the Toxoplasma channel of a new 384-well plate. Round 7, cellpose 4.2.1.1, 100 epochs, base cpsam_v2

**Measured.** F1 0.854 against stock cpsam_v2's 0.765 on the 11 anchor wells at IoU 0.5; AJI 0.776 against 0.505

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - f1
     - 0.8536
     - 0.7648
     - 0.0888
   * - aji
     - 0.7755
     - 0.5050
     - 0.2705
   * - dice
     - 0.8905
     - 0.6431
     - 0.2474
   * - precision
     - 0.8493
     - 0.7539
     - 0.0954
   * - recall
     - 0.8580
     - 0.7760
     - 0.0820
   * - pixel_iou
     - 0.8055
     - 0.5239
     - 0.2816
   * - mAP
     - 0.4921
     - 0.3620
     - 0.1301

Evaluation set: ``pv_r7_test``; version ``r7-published-truth``; 10 scored fields and 683 annotated objects.

* Scorecard metrics are means over nonempty fields, not pooled object counts; see aggregation.json.
* NEWEST IS NOT BEST HERE: on 20 held-out fields of the new plate it was trained on, round 7 scores F1 0.748 against 0.899 for round 6 (recall 0.636 against 0.829) -- prefer toxoplasma_pv_v3 on that plate
* on the 11 anchor wells it ties round 6 (0.8536 against 0.8602); 5-fold cross-validation F1 0.8142 +/- 0.015 against round 6's 0.8168
* it traces the vacuoles it finds more tightly than round 6 (AJI 0.827 against 0.786 on the new plate) but finds fewer of them
* the drop on the new plate is under investigation: round 7 also under-fits that plate's own training fields, which points at how those fields entered training rather than at generalisation

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r7 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7>`_, as ``cpsam_v2_toxo_r7``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/resolve/main/scorecard.png
   :alt: Published scorecard for Toxoplasma PV v4 (round 7)
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r7/blob/main/scorecard.csv

SHA-256 ``621a475c9bfb6865be7de12c5c4638f4509d734e0c4d39adb15ea541e48961d2``.

Toxoplasma PV v5 (round 8, alternative)
---------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2, Cellpose 4.0.9)

**Trained on.** Toxoplasma parasitophorous vacuoles from lab immunofluorescence and DsRed fields and published-figure crops. Round 8, cellpose 4.0.9, best of 100 epochs (epoch 20), base cpsam_v2

**Measured.** F1 0.792 against stock cpsam_v2's 0.559 on the 50 held-out fields at IoU 0.5

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - f1
     - 0.7921
     - 0.5586
     - 0.2335
   * - precision
     - 0.7531
     - 0.4635
     - 0.2896
   * - recall
     - 0.8354
     - 0.7028
     - 0.1326

Evaluation set: ``retrain_20261009_pv_test``; version ``2026-10-09``; 50 scored fields and 3530 annotated objects.

* ALTERNATIVE, NOT BETTER: on the 15 test fields round 7 never saw it scores F1 0.536 against toxoplasma_pv_v4's 0.525 -- a tie; prefer toxoplasma_pv_v4
* 35 of the 50 test fields were in round 7's training or validation set, so the all-field comparison (0.792 against 0.851) favours toxoplasma_pv_v4 and is not a fair test
* weak on published-figure crops (F1 0.440 on 14 literature fields), as is every PV model

Published as `einarolafsson/toxoplasma-pv-segmentation-cpsam-r8 <https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8>`_, as ``cpsam_v2_toxo_r8``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/resolve/main/scorecard.png
   :alt: Published scorecard for Toxoplasma PV v5 (round 8, alternative)
   :target: https://huggingface.co/einarolafsson/toxoplasma-pv-segmentation-cpsam-r8/blob/main/scorecard.csv

SHA-256 ``a1d7f1a978cf4925ec60db486a00eea33e87e6fc5a6b0d0f34c6db5f132d5c1c``.

Live cell v1 (phase, brightfield, DIC)
--------------------------------------

**Architecture.** Cellpose-SAM (cpsam_v2)

**Trained on.** unstained cells in phase contrast, brightfield and DIC: LIVECell, DeepSea, YeaZ, yeast microstructures, five Cell Tracking Challenge sets, BBBC009, BBBC030, QPI and Revvity. Base cpsam_v2, cellpose 4.2.1.1, lr 1e-5, batch 4; stopped at epoch 37 of 100

**Measured.** on the datasets stock cpsam_v2 never trained on, F1 0.960 against 0.885 at IoU 0.5; over all 2,199 test fields, 0.694 against 0.738, because stock trained on LIVECell and YeaZ and wins on LIVECell

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

Toxoplasma Plaque v3 (Gel Doc r5 candidate)
-------------------------------------------

**Architecture.** Cellpose-SAM (cpsam, Cellpose 4.0.9)

**Trained on.** 496 curated plaque fields, including 34 reviewed empty negatives and 71 Gel Doc wells; 100 epochs; fixed physical-plate/source groups

**Measured.** Stock was not evaluated in this run; see the named incumbent comparison on the model card.

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - f1
     - 0.6419
     - not measured
     - not measured
   * - precision
     - 0.6943
     - not measured
     - not measured
   * - recall
     - 0.5968
     - not measured
     - not measured
   * - ap
     - 0.4726
     - not measured
     - not measured
   * - true_positives
     - 302
     - not measured
     - not measured
   * - false_positives
     - 133
     - not measured
     - not measured
   * - false_negatives
     - 204
     - not measured
     - not measured

Evaluation set: ``geldoc_heldout_min_size_0``; version ``20261007T205603``; 18 scored fields and 506 annotated objects.

* Candidate, not promoted; independent literature uncertainty gate did not pass.
* Gel Doc score requires min_size=0; min_size=150 gives F1 0.1041.
* Stock not measured; comparison baseline is incumbent r3.
* Distinct from the cpsam_v2 round-5 model already registered as toxoplasma_plaque_v2.

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc>`_, as ``cpsam_plaque_r5_geldoc``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/resolve/main/scorecard.png
   :alt: Published scorecard for Toxoplasma Plaque v3 (Gel Doc r5 candidate)
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r5-geldoc/blob/main/scorecard.csv

SHA-256 ``8f59850dbe09c3a252728b0a7ca38ae5b52a28f2dbaa7a068174b4b7bfac768c``.

Toxoplasma Plaque v4 (round 6, alternative)
-------------------------------------------

**Architecture.** Cellpose-SAM (cpsam, Cellpose 4.0.9)

**Trained on.** Toxoplasma plaques from published-figure wells, Bio-Rad Gel Doc wells, lab plates and staged literature crops; round 6, best of 180 epochs (epoch 130), base cpsam

**Measured.** F1 0.813 against stock cpsam 0.300 on the 263 held-out fields at IoU 0.5

.. list-table:: Published scorecard
   :header-rows: 1

   * - Metric
     - This model
     - Stock
     - Difference
   * - f1
     - 0.8127
     - 0.3003
     - 0.5124
   * - precision
     - 0.8232
     - 0.2235
     - 0.5997
   * - recall
     - 0.8025
     - 0.4576
     - 0.3448

Evaluation set: ``retrain_20261009_plaque_test``; version ``2026-10-09``; 263 scored fields and 12792 annotated objects.

* Alternative, not better: F1 0.813 against toxoplasma_plaque_v3 0.844 on the same held-out fields, lower on every source (Gel Doc 0.826 against 0.917).
* Above the spaCR default toxoplasma_plaque_v2 overall (0.798) and on Gel Doc wells, where v2 scores 0.013.
* Fine-tuned from first-generation cpsam, not cpsam_v2.
* Published-figure literature crops stay hard (F1 0.535).

Published as `einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6 <https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6>`_, as ``cpsam_plaque_r6``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/resolve/main/scorecard.png
   :alt: Published scorecard for Toxoplasma Plaque v4 (round 6, alternative)
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-segmentation-cpsam-r6/blob/main/scorecard.csv

SHA-256 ``00f4a86ba1502fd583231e54bc6538f595f67c86cd4f9ee5c415f6f788a0c171``.

Toxoplasma Well Detector v3 (YOLO11 Gel Doc candidate)
------------------------------------------------------

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
   * - mAP50_95
     - 0.8842
     - not measured
     - not measured
   * - precision
     - 0.9698
     - not measured
     - not measured
   * - recall
     - 0.9979
     - not measured
     - not measured

Evaluation set: ``detector_v3_validation_figure_clean``; version ``20261007T205603``; 73 scored fields and 191 annotated objects.

* Candidate, not promoted: clean incumbent-validation mAP50-95 regresses.
* Gel Doc plates selected best.pt, so their scores are validation rather than independent test.
* Distinct from the YOLO26 checkpoint also historically called detector v4.
* Locates wells, not plaques; one class plaque_well (0).

Published as `einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc <https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc>`_, as ``yolo_welldetect_v4_geldoc.pt``.

.. image:: https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc/resolve/main/scorecard.png
   :alt: Published scorecard for Toxoplasma Well Detector v3 (YOLO11 Gel Doc candidate)
   :target: https://huggingface.co/einarolafsson/toxoplasma-plaque-well-detector-yolo11-v4-geldoc/blob/main/scorecard.csv

SHA-256 ``046861021594256957292cfe9488a84e758077a12dac31fe66b8b4083f8dfa18``.
