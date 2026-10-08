# Coverage 0 idle/click/eager parity failure

This archive transfers the original hosted failure and bounded negative local
reproductions to the Workstation owner. It does not establish a fix or a green
coverage shard.

- Hosted run `37698599898`, source `6836512a98bea77e946dd941a9b4f8a0e38068b5`,
  Coverage shard 0 job `113058129287`: `test_a_category_built_in_idle_time_equals_one_built_by_a_click`
  failed for `regression` at the row-equality assertion. The differing keys were
  hidden Estimator Tuning, Permutation Test, and Plate & Batch Correction.
  The chained assertion does not identify which adjacent pair differed.
- The original node passed alone (1/1, 14.72 seconds); its whole test file
  passed (12/12, 48.86 seconds). Their raw command output was not retained,
  so these are observations, not archived log-backed claims.
- The bounded predecessor cohort of `test_multi_instance_lock_and_jobs_dock.py`,
  `test_graph_spec_uncovered_paths.py`, and
  `test_idle_time_builds_the_closed_categories.py` passed 37/37 under
  `coverage run --branch`, `-n 2 --dist=loadfile`, 4 GiB, offscreen Qt, and
  CUDA hidden. Its raw log records one JobRunner still busy after the named
  parity node; this has not been shown to cause the hosted mismatch.
- A temporary, removed diagnostic test recorded the regression headings before
  and after `_open_every_waiting_heading()`: the three waiting headings changed
  from 0/0/0 rows to 8/14/10; idle and eager each had 8/14/10. All three
  completed headings had `settingsSectionDiscarded=False` locally.

`_IdlePrebuild._work_left()` scans `rendered_settings_sections()`, whereas
`_rows()` in the failing test scans all `_settings_sections`. The former filters
only `settingsSectionDiscarded`, which marks dormant object-gated sections and
their descendants. This is a source distinction, not a proved cause: the local
diagnostic did not encounter a discarded regression heading. The only
`_build_a_waiting_heading()` call site is the initial panel layout; opening a
top-level waiting heading builds nested sections directly, so newly waiting
children were not demonstrated. The hosted node took about 71 seconds against
7–15 seconds in focused local runs, but no timing cause is established.

All product code and the original row/value/search equality assertions remain
unchanged. The workstation owns further reproduction. Do not mark the hosted
failure resolved from the negative local runs.
