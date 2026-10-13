# Protected 45f failures against current CPU source (2026-10-07)

This bounded recheck starts from the 27 failed node IDs in the protected
`45f3cb1ec0` serial artifact. The isolated replay worktree used source
`6f4cb32b61` plus the independent bridge ownership fix `58aa9f7eae`
(local commit `83a2bd3505`), Python 3.12.13 and PySide6 6.11.2. CUDA was
hidden, Qt was offscreen, and every run used a 4 GiB hard cap. It did not
repeat the original-order serial suite or assert full CI acceptance.

Four old `test_ambient.py` fungal census node IDs no longer collect after
`81c102068b`; the current fungal tests target visible connected growth and
time-dependent branches. The focused current fungal selection passed 22
cases. Of the remaining 23 old node IDs, one popup node now expands to two
cases, giving 24 collected cases. The exact-node replay yielded 22 passes
and two failures. `classification.json` maps each old node explicitly.

One failure was a real current geometry defect:
`test_loaded_schema_rules_do_not_overlap_n637.py`. The default Qt styling
passed alone, but the actual spaCR theme made a 40 px remove button overlap
the following exact-values editor. With two loaded condition boxes, each box
reported a 280 px minimum-size hint yet the resizable scroll content
compressed it to 191 px. Its first row received 33 px, so a 40 px button
extended into the next row. The existing test now activates the spaCR theme;
it failed with the old source (`schema-before.log.gz`) and passes after
`675ff1ec7c` constrains the scroll layout to its computed minimum size.
Four related condition-annotation test files passed 19/19. Branch-aware
focused coverage executed the new source line 886; the change adds no branch.
The source/test blobs and complete bounded receipts are in the JSON files.

The other failure was the GUI-marked real-display Home backdrop census.
Its original >25% chromatic threshold was measured for the older Blobs
animation; the app default is now impulse lens. The offscreen replay measured
15.7% chromatic, but it cannot by itself establish a display defect. Root
separately reproduced 14.6% under Xvfb and is pinning this test to Blobs,
while preserving its threshold and the application's new default. That
separate candidate is not included in `675ff1ec7c`. The serial runner's
test selection was not changed or excluded.

The other 21 historical node IDs passed in the bounded current replay,
including source inventory, translation, tooltip, crash recovery, preview
affinity, distribution smoke, and gravity-density checks. This is evidence
for those exact cases in the chosen local environment, not a statement that
their previous 45f assertions were false or that CI is green. Their 45f
failure details were not available because the six-hour serial run ended
before pytest printed a summary.
