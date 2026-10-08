# Animation background: source-bound coverage reconciliation

The complete hosted 683 coverage report preceded the eleven-module repair,
the fungal parent-index optimization, the optional Animation background
control, and the positive-cost fungal guard. This archive combines that full
report with focused reports only where Git source lines are byte-identical.
The final ambient statement and branch sets come from the 18-case guard
coverage run against source `237f649ad`; the Preferences and AppScreen sets
come from the integrated `0d8950bb624` report because those source blobs are
unchanged. The verifier checks both exact fungal source transformations and
maps equivalent lines and arcs; it refuses any target source hash change.

| Module | Current missing lines / branches | Original allowance |
| --- | ---: | ---: |
| `spacr/qt/preferences.py` | 84 / 25 | 89 / 25 |
| `spacr/qt/widgets/ambient.py` | 0 / 0 | 1 / 0 |
| `spacr/qt/screens/app_screen.py` | 53 / 35 | 53 / 35 before this feature; zero new gaps |

All executable statements and branch arcs touching source added since
`5b655f8c7a2` are covered. The original numerical ceilings are unchanged.
The integrated background file passed 7/7 in 15.38 seconds under CUDA-hidden
4 GiB branch coverage. The broader focused cohort passed 86/86 under branch
coverage on the same pre-guard `0d8950bb624` source; its raw log and JSON
are archived separately. Its log begins at 01:28:41.813 UTC, 10.11 seconds
after the last native benchmark log entry, so the runs have no observed
overlap. The guard follow-up passed 18/18 under branch coverage; both new
statements and both branch outcomes are directly executed. These are
functional coverage checks, not native performance measurements. The
original full hosted run is not being called green by this local
reconciliation; a corrected-source hosted verdict remains necessary.

The normal AutoAPI extractor reports one new nested picker callable,
13,214→13,215, with no changed existing callable prose. Static `tr()`
extraction reports six new runtime strings and zero removals. Two Preferences
tip entries change; `receipt.json` records their exact before/after text for
normal workstation catalog regeneration. No generated API or translation
file was edited here.

Run `python features/data/43_ambient_background_union_2026-10-08/verify_union.py
--git --api` from the repository; `--api` repeats the real current AutoAPI
extraction. `verify_manifest.py --git` checks every archived payload byte.
