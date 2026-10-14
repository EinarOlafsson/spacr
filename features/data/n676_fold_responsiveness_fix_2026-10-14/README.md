# Held cold-fold responsiveness fix

Review artifact only: local commit `a5dc7353bc1da39795c565dd4d4dd27f3f2ba5af` against parent
`10c1167e109b9acea68a7ada63a9136becda1a60`. No app-source push, Home integration, release or full-CI acceptance
is claimed. Home owns review/rebase and integration; the live source stays frozen.

The patch keeps the existing fold construction, host-body migration, decorated tab
insertion and selection, with genuine event-loop passes after completed steps.
It reuses normal input/navigation guards and preference/language scopes. Same-fold
reentry is guarded; other folds and page switches defer through owner-bound timers,
coalesce latest visibility intent and restore guards on failure. No page prewarming,
hidden transitions, threshold changes or substituted UI. Independent contracts
source review found no remaining blocker; native proof remains separately scoped.

Only `spacr/qt/screens/map_barcodes.py`, the breathing docstring in
`spacr/qt/screens/__init__.py`, and
`tests/qt/test_fold_opening_keeps_the_event_loop_alive.py` change. Recorder files and
`human_input.py` are identical to the parent.

- Final source-pinned tests: **35 passed in 15.07s**, including retained state,
  close/reopen, exception cleanup, borrowed guards, same/different-fold reentry,
  deferred controller completion, latest-close cancellation and owner deletion.
- Real r32 Source clip: strict failure despite 29.29 fps; **536 ms** GUI sampler gap
  and app freeze in second 1. Retry also failed (520 ms).
- Fresh r34 Source clip on a5dc: **29.33 fps**, no reasons/freezes/sampler gaps;
  largest recorded sample interval **268 ms**. All **12/12** main+Apply raw gates
  strictly pass at 28.89–29.80 fps with unchanged 27.9 fps/350 ms checks, no
  near-miss approvals, pointer jumps, exposed top bar or bypass. The original
  13 capture-loss seconds across seven otherwise passing clips remain recorded.
- Actual media/receipts/start-PNG hashes match; 502 captured source files match both
  runs, the clean worktree and immutable a5dc. Both cleanup receipts report exit 0
  and no survivors. Science/English/visual/publication acceptance is separate.

## Exact bindings

| Evidence | SHA256 |
| --- | --- |
| `a5dc7353b-fold-responsiveness.patch` | `df2a2d100d0bcc4bea9a534fd8f1af9d3fba7d84bb8197f80b09c3ecf778ee8f` |
| External `work/tutorial_today_20261014/illumination-performance/final-owning-tests.log` | `9876afb8da3400b999f222003f43f05066e01c7917af8a7b47c48f0d0927c6bc` |
| External `work/lesson43-r34/gate-source-review.json` | `05a34a17cfe5fd4d333fc5b1d0f3ab9450d5b0b98edeb891de3b3116a35091d6` |
| r32 `captures/illumination/05_setting_src.gate.json` | `226795b18e8a56cdac4b42cc3bce9a62eb34c4a5df0cb3910474d3cc4afceef8` |
| r34 `captures/illumination/05_setting_src.gate.json` | `10367563e7ac62663a982258fe8f014bacc73cf699697d0a60f64ad5c5e6fa49` |

External paths are beneath `/media/carruthers/mnt3/claude/n676_stages/`;
raw r32/r34 paths are beneath `host/lesson43-r32-20261014/` and
`host/lesson43-r34-20261014/` respectively. Originals remain unchanged.

The patch is the unmodified output of
`git format-patch --stdout --binary --full-index -1 a5dc7353bc1da39795c565dd4d4dd27f3f2ba5af`.
