# Live i18n review-scope timeout repair, 2026-10-07

The old `e1f54c80650f60942ba6c98d4e999b36a3fa2846` Tests run
`37559626498`, coverage shard 3 job `112602607261`, timed out while
`test_written_review_scope_matches_current_source_bound_evidence` checked all
nine locales as one test. Its full hosted log is
`hosted-coverage3.log.gz`. Commit
`08b497f9c8445fb5f1dad0738c2e3b3100008a84`
splits the unchanged per-locale assertions into nine parameterized tests and
extracts real live API docstrings and canonical runtime sources once per
worker. Each locale still passes its reviewed runtime/API evidence through
the original validators and checks the two shipped API-symbol translations,
report row, and report claims. No review result, dynamic translation decision,
guard ceiling, generated catalog, or product source was changed.

Validation from the source commit above, Python 3.12.13, CUDA hidden, Qt
offscreen, 4 GiB cap:

```
env CUDA_VISIBLE_DEVICES='' QT_QPA_PLATFORM=offscreen SPACR_DEVICE=cpu MPLBACKEND=Agg COVERAGE_FILE=/mnt/wd4tb/scratch/i18n-canonical-review-20261007/coverage-param.data tools/run_capped.sh 4G /mnt/wd4tb/scratch/item47-memory-codex-20261005/ci-near-parity-venv/bin/python -m coverage run --branch -m pytest -q -p no:randomly --durations=10 tests/test_i18n_coverage_audit.py::test_written_review_scope_matches_current_source_bound_evidence
```

Exit 0: nine passed in 526.07 seconds total. The longest call was Simplified
Chinese at 83.23 seconds; the first live-source fixture setup took 47.31
seconds. Even their sum is below the existing 300-second Fast/Minimum test
timeout and the Coverage workflow's 600-second per-test timeout. The coverage
batch timeout remains 2700 seconds. `ruff check --select E,F` and
`git diff --check` passed. An independent AST comparison confirmed the nine
locale bodies, six assertion expressions, and trailing report contract are
unchanged. `pytest-output.txt` assembles the exact stdout chunks returned by
the command tool; it was not independently tee'd to a file during execution.
The raw coverage SQLite process file is preserved as `coverage-param.data.gz`.

`manifest.json` binds the test, builders, current reviewed evidence trees,
published API payload tree and report to Git object IDs. Those owner artifact
objects match published source `28449ef0c84c4be92cf34be18d7372e3f0d9e2c8`.
Its `spacr/` tree is different in `qt/screens/make_masks.py` and
`qt/widgets/ambient.py`, so this local proof is not an exact-source full CI
verdict for that later commit. The newer hosted Tests run was still pending
when this archive was written.
