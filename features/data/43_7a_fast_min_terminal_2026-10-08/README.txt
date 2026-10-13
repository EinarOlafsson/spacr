Complete corrected 7a Fast/Minimum phase: all six genuinely successful

Protected required run 37744767240, attempt 1, measured source
7a51b6c921ea0d9b51ca3f5e6d28a68904ce2cab. Fast0/Fast1/Fast2 and
Min0/Min1/Min2 all completed with SUCCESS. All six complete original raw
terminal logs are preserved losslessly, with exact job/head/attempt metadata,
original byte lengths and SHA256. No cancellation or recovery is called green.

Both formerly failing workflow-profile phases now truly pass on 7a. The
corrected test pins exactly one dedicated first-open timing exclusion while
preserving all previous exclusions, selection rules and original timing limits.
There are no blocking failed nodes or native-fault markers in these six logs.
Existing Py3.9 i18n XFAILURES and watchdog dumps remain in the original evidence
and are not reclassified as blocking failures or native crashes.

This is acceptance of these six phases on 7a only. Whole required Qt/coverage
and serial acceptance remain separate. Later candidate 5636 changes a Qt
Shiboken-wrapper test; this archive does not relabel the measured source or
claim that newer complete run passed. Seven production files and the corrected
profile test are byte-identical at 7a and the later reference. --bind-current
checks only the declared unchanged production bytes, without requiring any
agent-only commit object. Frozen profile-test bytes are included for portability.

Counts sum each job's terminal pytest batch outcome summaries, not unique node
IDs. Do not add across six jobs as unique tests. The writer recognizes skipped,
xfailed, xpassed and errors independently of passed. Each original summary's
line/text and count is retained; no ceiling/guard/budget or policy changed.
Full run JSON is a contemporaneous still-active snapshot, not a terminal-run
acceptance claim. No new local pytest replay or production change was needed.

Verification from this archive's integration checkout:
 tools/run_capped.sh 2G python features/data/43_7a_fast_min_terminal_2026-10-08/verify_manifest.py --bind-current
After commit/sparse integration add --git HEAD --bind-current. archive_phase.py
recreates selected payloads/receipt from scratch, write_manifest.py owns hashes.
No full coverage zip, core dump or large native image archive is duplicated.
