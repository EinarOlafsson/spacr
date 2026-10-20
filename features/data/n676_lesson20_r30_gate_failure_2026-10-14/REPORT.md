# Lesson20 turn30: actual smoothness failure

Neither opening qualifies for the approved capture-loss-only near-miss policy. Both record 27.68 distinct fps, above the 27.0 near-miss floor but below the unchanged 27.9 strict bound, with actual GUI delivery stalls. No acceptance approval is produced.

The fresh source is `e315f8d6e84ed0c4d2745e0ccc46472709a316ec`: approved app 39078 plus preserved f7 capture tools. Primary a01 has 7 strict passes and 4 failures; the one retry b01 of those 4 still has 4 failures. The whole lesson therefore remains unaccepted. App and recorder exit 0 in each host report do not override failed clip gates. Root reports the enclosing GPU turn terminal exit 0 and the separate scientific reference exit 0; this report does not independently certify scientific acceptance.

| Attempt | Visual | Distinct fps | Freeze seconds | GUI sampler gaps (ms) | Pointer max / bound (px) |
|---|---|---:|---|---|---:|
| a01 | 02_actual_train_and_apply_tabs | 27.68 | [4, 9] | [458, 648] | 59.7 / 76.8 |
| a01 | 20_actual_preview_result | 28.38 | [20] | [] | 47.5 / 76.8 |
| a01 | 22b_actual_zoomed_filtered_preview | 29.04 | [0, 2] | [] | 36.1 / 76.8 |
| a01 | 23b_actual_zoomed_restored_preview | 29.12 | [3] | [] | 36.2 / 76.8 |
| b01 retry | 20_actual_preview_result | 28.34 | [20] | [] | 47.4 / 76.8 |
| b01 retry | 02_actual_train_and_apply_tabs | 27.68 | [4, 9] | [357, 649] | 66.5 / 76.8 |
| b01 retry | 22b_actual_zoomed_filtered_preview | 29.41 | [0] | [] | 39.1 / 76.8 |
| b01 retry | 23b_actual_zoomed_restored_preview | 29.01 | [3] | [] | 37.5 / 76.8 |

## Evidence and scope

The opening raw 100 ms sampler timestamp pairs independently reproduce the gate gaps: a01 `[4.164,4.622]` and `[8.671,9.319]`; b01 `[4.179,4.536]` and `[8.597,9.246]`. Original distinct-picture counts in seconds 4/9 are 16/14 for a01 and 22/14 for b01. Original gate offset alignment puts 29 of 61 unchanged frames inside app stalls for a01 and 24 of 61 for b01. Pointer detection is 100%, with zero jumps; every reviewed gate hides the top bar and records no adapter bypass. These facts do not make either opening capture-loss-only.

Opening watchdog stacks report a01 GUI stalls 0.39 s at clip 4.55 s and 9.06 s, and b01 stall 0.36 s at 8.96 s. The first a01 snapshot stops in `capture_host_adapter.EventPump.once` / `QApplication.processEvents` during the real Home→MakeMasks click. The later a01 main-thread snapshot traverses `make_masks.open_folded`→`map_barcodes.show_as_page`→`app_screen.sizeHint`. The b01 snapshot traverses `open_folded`→`show_as_page`→`host_pages`→`install_close_marks`→`get_font_scale`. These locate sampled work during actual GUI delays; they do not prove that one sampled function caused the full delay. Background import/render-thread stacks are not misidentified as the GUI stack.

The other 3 failed visuals have no recorded GUI sampler gaps or watchdog dumps. Their unchanged gate rejection comes from modern **minimum rate of stably visible fields**, not a legacy sum estimate. Raw per-widget counter interpolation independently reproduces the reported low paint rates: preview second 20 is 18.7/17.7 paints per second (a/b); zoomed filtered second 0 is 23.5/23.4, and a01 second 2 is 23.2; restored second 3 is 19.2/20.4. These are observed visible-field rate drops. They are not watchdog-proven whole-GUI stalls, and their high whole-clip FPS or summed app paint rates do not remove the failure.

## Concrete next action

Home should inspect cold Home→MakeMasks and folded Cellpose page activation, especially page/layout/close-mark work on the actual GUI path, plus the slower stably visible field during preview/filter/restore. No third blind retake is justified by these repeated failures. Do not hide the opening transition with prewarming/cropping, switch background/alpha policy, weaken thresholds, or relabel failed raw reports as accepted.

Any app repair needs review and owning validation, then a fresh whole lesson 20 on one immutable app base; the seven passing a01 clips cannot be spliced across a changed app base. If a telemetry bug is proposed, it must be demonstrated and validated before an honest replay, while preserving these original gates. Root may preserve blocked 20 and select the next authorized lesson serially; this review launches nothing.

`summary.json` binds both source records, all 15 gate/host/stall/pointer/capture receipts, the 8 failed native video byte hashes, and relevant source files. No video was re-decoded, no gate replay was run, and no raw/source/threshold file was edited. This is an AI technical evidence review, not human/native certification or publication approval.
