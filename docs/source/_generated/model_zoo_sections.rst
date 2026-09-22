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
