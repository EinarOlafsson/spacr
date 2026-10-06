# Obsolete 1f3 tests run failure inventory

Run 37494090176; source `1f3e4a240339ebc88993d72c49b39ab5716e3904`; pre-cancel snapshot has 21 failed jobs, six successful jobs and one running Fast shard. Root cancelled the obsolete run to free the current tests workflow group. This is evidence about old source only.

All 21 failed jobs have a saved full log. Missing logs were downloaded as individual gzip files, and earlier saved logs were reused. This archive carries all 21 failed-job logs in `logs/`. `failed-node-index.json.gz` gives every failed job’s node IDs and raw summary lines.

| Job ID | Job | Failed node IDs in log | Saved log |
| --- | --- | ---: | --- |
| 112391876927 | Minimum dependencies (ubuntu-24.04, py3.9) shard 0 of 3 | 11 | `logs/112391876927.log.gz` |
| 112391877141 | Minimum dependencies (ubuntu-24.04, py3.9) shard 1 of 3 | 44 | `logs/112391877141.log.gz` |
| 112391877151 | Fast / Full suite control (ubuntu-24.04, py3.12) shard 2 of 3 | 40 | `logs/112391877151.log.gz` |
| 112391877159 | Qt (0) / Qt shard 0 of 3 (ubuntu-24.04, py3.12) | 6 | `logs/112391877159.log.gz` |
| 112391877174 | Fast / Full suite control (ubuntu-24.04, py3.12) shard 0 of 3 | 11 | `logs/112391877174.log.gz` |
| 112391877201 | Minimum dependencies (ubuntu-24.04, py3.9) shard 2 of 3 | 41 | `logs/112391877201.log.gz` |
| 112391877231 | Slow / Slow (ubuntu-24.04, py3.12) | 3 | `logs/112391877231.log.gz` |
| 112391877252 | Qt (1) / Qt shard 1 of 3 (ubuntu-24.04, py3.12) | 14 | `logs/112391877252.log.gz` |
| 112391877259 | Qt (2) / Qt shard 2 of 3 (ubuntu-24.04, py3.12) | 13 | `logs/112391877259.log.gz` |
| 112391877324 | Coverage shard 5 of 12 | 12 | `logs/112391877324.log.gz` |
| 112391877369 | Coverage shard 0 of 12 | 4 | `logs/112391877369.log.gz` |
| 112391877399 | Coverage shard 9 of 12 | 5 | `logs/112391877399.log.gz` |
| 112391877404 | Coverage shard 11 of 12 | 7 | `logs/112391877404.log.gz` |
| 112391877412 | Coverage shard 8 of 12 | 32 | `logs/112391877412.log.gz` |
| 112391877458 | Coverage shard 2 of 12 | 5 | `logs/112391877458.log.gz` |
| 112391877520 | Coverage shard 4 of 12 | 25 | `logs/112391877520.log.gz` |
| 112391877527 | Coverage shard 1 of 12 | 4 | `logs/112391877527.log.gz` |
| 112391877543 | Coverage shard 6 of 12 | 11 | `logs/112391877543.log.gz` |
| 112391878394 | Coverage shard 7 of 12 | 27 | `logs/112391878394.log.gz` |
| 112391878542 | Coverage shard 10 of 12 | 5 | `logs/112391878542.log.gz` |
| 112432977583 | Coverage / every module at 90% and none loses coverage | 0 | `logs/112432977583.log.gz` |

The coverage combine job uploaded a report but exited with `coverage shards finished with failure`; its failure does not establish a clean absolute-count ratchet verdict. Release gate failure added only after that snapshot propagates failed jobs.

Source-bound triage:

- Generated/API/runtime/localization/settings-flow/help/notebook/README failures are owner refresh work for the newer source. They dominate Fast and MinDeps logs; no guard threshold or generated file was changed here.
- Theme registry/default failures (`ripple`, `cells`, `none`, six-material pin and blobs default) describe obsolete tests at the old source. Many were repaired later in root; use the fresh 3ca run to decide current state.
- Old direct `QColorDialog.getColor` call in Preferences was corrected later by root. Current focused gravity-filter two nodes, clock handoff and center-pixel puncta node pass under 4 GiB.
- MinDeps shard 2 had one distinct native-volume test assumption: Cellpose 4.0.7 converts one-channel ZYXC input to three channels by zero padding. spaCR passes native one-channel data and correct `do_3D/z_axis/channel_axis`. Test-only correction commit `270c7ff4562ebcefb48ab5f0412c094eda257b21` passed with installed 4.2.1.1 and the official 4.0.7 wheel overlay, plus 47 current native batch tests.
- Current `test_system_panel_opacity.py::test_the_ambient_animation_reaches_all_four_bars_alike` still fails alone at root 1248 (CPU ratio 0.87; RAM 1.38; GPU 1.15); root owns visual source investigation.
- Old `test_a_glassed_dialog_rewrites_its_flags_once` captured a prior `_no_setup_card` import hook, but the focused two-node current replay passes. Treat it as unresolved order-sensitive evidence until fresh CI, without a speculative source change.

Fresh current source run 37511792034 at SHA 3ca is the relevant mandatory verdict; the old run was cancelled only after this inventory. Protected serial 37503577012 was untouched.
