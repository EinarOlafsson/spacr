# Cancelled original serial Qt run, 2026-10-08

The original source `55badc57ff25ef6a7e14721561773a412b515318` run `37720605434` / job `113127133917` was cancelled at 05:16:06 UTC after the DNA-rain cache assertion was fixed in source `4133beafcd0ae427795617a4a01e295fd40539b7`. The corrected serial run is `37728434064` / job `113161465305`. Cancelling the obsolete run freed the concurrency slot; it was not a green or failed serial acceptance verdict.

The uploaded partial artifact `11530071440` contains 756 completed file records and the beginning of a 757th. The pytest log reaches 43%. It records two call-phase failures: the distribution smoke measurement/UI integration assertion and the stale DNA-rain two-field cache unpack. The corrected source addresses the latter. The last completed file was `tests/qt/test_every_picture_is_drawn_for_the_screen_it_is_on.py`; the run was in `tests/qt/test_every_popup_has_the_card_and_the_rim.py` when cancelled.

The file journal's largest recorded RSS was 4,221,161,472 bytes (4,026 MiB), with a 4,410,544,128-byte high-water mark (4,206 MiB). These describe only the partial prefix. The native fatal log is empty; no native cause, completed serial memory verdict, or full-suite result follows from this cancellation.

`partial-artifact.zip` is the GitHub artifact unchanged. `terminal-job.log.gz` contains the raw job log losslessly; its uncompressed SHA-256 is in `receipt.json`. `before-run.json`, `terminal-run.json`, `terminal-jobs.json`, and `terminal-artifacts.json` preserve API snapshots. `sha256.json` hashes each payload and the receipt; it deliberately does not hash itself.
