# N47 bounded Qt style-propagation diagnosis, 2026-10-07

The protected uninterrupted serial Qt run `37503577012` at source
`b77916cc94165439fabac6cc9c8e004ba7f944b5` reached 1,053 completed
original-order test files and started `test_page_is_never_black.py` before the
workflow's six-hour limit cancelled it. It reached 63% with peak file-journal
RSS 5,933.4 MiB, below the existing 10.8 GiB pytest guard, and its host log
reported no OOM kill. This is a timeout, not memory or full-suite acceptance.
The successor protected run `37521372347`, exact source
`45f3cb1ec0ec5df9e10bb6bcf788c5939d2cc1a3`, job `112556704860`,
was still in progress at 2026-10-07 00:31 UTC and was not interrupted.

The hosted file journal shows `test_field_fade.py` took 2,071.7 s and ended
at 4,403.5 MiB RSS. It began `test_page_is_never_black.py` after 1,053 file
completions but has no end record. The local isolated bounded probes on
`5f33859801fbfcb7df43f12251ab2c26b4cd9f26` passed all 30 field-fade
tests in 3.74 s and all 115 page tests in 41.14 s under a 4 GiB hard cap,
CUDA hidden and Qt offscreen. The large hosted slowdown is context dependent.
The saved widget journal sampled zero live widgets at the following setups;
it does not prove there were no native objects or other retained resources.

The 30-test `cProfile` run took 4.37 s. `apply_preferences_to_app` used
0.621 s cumulative over 29 calls; `apply_stylesheet_per_window` used 0.375 s
over 26; `_forget_window_stylesheets` used 0.122 s over 30. The fixture's
`collect_once` used 0.065 s over 32 calls. This points to style propagation
as a measurable fresh-process cost, not to a proven cause of multi-minute
hosted nodes. The complete binary profile is `field-fade.pstats.gz`.

`benchmark.py` and `benchmark-raw.json` contain seven independently capped
4 GiB processes. Each creates a hidden QWidget root with 0, 500, or 1,500
QLineEdit children; one case adds 200,000 unrelated Python dicts. Each applies
five preferences saves. The changed-style cases then invoke the existing
stylesheet teardown, as the test fixture does. Warm medians exclude the first
initialization. The `--unchanged` cases keep the chosen theme and do not run
that teardown:

| Population | Changed save | Test teardown | Unchanged save |
| --- | ---: | ---: | ---: |
| 1 widget | 1.59 ms | 0.20 ms | 0.27 ms |
| 1 widget + 200,000 Python dicts | 1.60 ms | 0.19 ms | — |
| 501 widgets | 32.89 ms | 14.07 ms | 1.86 ms |
| 1,501 widgets | 95.79 ms | 40.87 ms | 4.96 ms |

The current stylesheet signature already avoids global repolishing when
preferences are unchanged. `install_button_roles`, icon scaling and console
zoom still visit current widgets to serve controls created after the last
save. The measurements do not justify skipping those updates. No production
change, forced collection, guard change or acceptance waiver follows from
this receipt. The existing 10.8 GiB memory guard and full uninterrupted
serial requirement remain open.

`manifest.json` identifies the exact profiled source bytes and hashes the
local artifacts. The published `origin/nightly` at archive time has identical
`preferences.py`, `theme.py` and `test_field_fade.py` bytes. Full protected
raw job logs remain in the GitHub Actions artifacts; this compact archive
contains its file RSS journal and host memory log.
