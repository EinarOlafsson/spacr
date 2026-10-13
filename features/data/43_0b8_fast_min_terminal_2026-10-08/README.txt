Complete original 0b Fast/Minimum phase: four success, two profile-only failure

Exact protected required run37741615330, attempt1, source
0b8c2c4120a0ede4de568d484ea3fc9d517f1de3. All six Fast/Minimum jobs genuinely
completed. Fast1/Fast2/Min1/Min2 SUCCESS; Fast0/Min0 FAILURE. Every original
full terminal raw log is compressed losslessly and bound to exact job/source
metadata, byte length and SHA256. No parent cancellation alters these verdicts.
Other required jobs and original-order serial are separate, not accepted here.

Only two unique blocking nodes appear, repeated in Fast0 and Min0:
 test_profile_comes_from_the_actual_workflow[qt-3-16-1200-7]
 test_profile_comes_from_the_actual_workflow[coverage-12-32-600-5]
Both omit the dedicated first-open timing file from their old exclusion count:
Qt actually8 versus7; coverage6 versus5. Existing verified repair8f28299
explicitly pins that file once while preserving every prior exclusion and all
original timing limits/routing guards. Frozen measured and repaired test bytes
are retained; this archive does not claim a later candidate's full acceptance.
No application/source/scientific computation, limit, ratchet or policy change.
No duplicate local replay of these already-proved failures was run.

Min0's additional i18n assertion text is explicitly in existing Py3.9 XFAILURES,
not another blocking node. Existing watchdog thread dumps are listed separately
and do not mean a native fault or failed test. All six logs have no observed
fatal-native/worker-crash marker; this is not proof historical Qt faults are
fixed, zero memory leaks, or full N43/N47 acceptance. Both remain OPEN pending
actual corrected complete required/native serial outcomes.

Counts are sums of each job's raw pytest batch summaries, not deduplicated
node IDs. Do not sum across six jobs as unique tests. receipt.json retains
each summary's original line/text and per-phase total, skips and xfails.
Seven production files are byte-identical at measured/repair reference and
local integration, with current-source verification optional. The run JSON
is a contemporaneous still-active full-run snapshot, not a terminal-run claim.

Portable verification (no dependency on an agent-only commit object):
 tools/run_capped.sh 2G python features/data/43_0b8_fast_min_terminal_2026-10-08/verify_manifest.py --bind-current
After integration/sparse checkout use --git HEAD --bind-current. archive_phase.py
is the writer for receipt/payload selection from original scratch paths;
write_manifest.py owns all manifest hashes. No large coverage archive copied.
