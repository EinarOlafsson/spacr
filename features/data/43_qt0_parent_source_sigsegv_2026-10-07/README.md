# N43 Qt shard 0 native crash, 2026-10-07

Hosted ordinary run `37559626498`, job `112602607868`, tested source
`e1f54c80650f60942ba6c98d4e999b36a3fa2846` on Python 3.12.14,
PySide6/Qt 6.11.2, pytest 8.4.2. In Qt shard 0 batch 60/241, xdist worker
`gw1` died with SIGSEGV while running
`tests/qt/test_make_masks_parent_source_race_guards.py::test_parent_invalidation_during_magnifier_teardown_updates_the_report`.
The current Python frame was `pytestqt.wait_signal.wait` → `qtbot.waitUntil`
→ `_load_parent` line 36, reached from test line 90. This was before the
test set `snapshot = None` or deleted `_magnifier`. The worker's last sampled
RSS was 455 MiB, host available memory 14,775 MiB, OOM count zero, and the
pytest guard was 6 GiB. The batch retained 1 failed and 106 passed tests.
The full completed job log is `e1-qt0-failed.log.gz` (uncompressed 3,745,382
bytes; compressed SHA-256
`234be7af0588337945b17190cc86a5fe3e05b7c28996353457d7884ebe517083`).
The ordinary job uploaded memory telemetry artifact `11461405237` but no
native core or C backtrace artifact.

The exact source test file passed 8/8 locally on current source
`28449ef0c84c4be92cf34be18d7372e3f0d9e2c8`, hidden CUDA/offscreen,
under a 4 GiB cgroup using Python 3.12.13 and Qt 6.11.2. The exact failing
CI source and its workflow-derived batch 60 were then replayed under the
same local 4 GiB cap and two xdist workers: 110/110 passed in 25.99 seconds.
`manifest.json` pins all 16 files and the original marker, worker count,
timeouts and runner options; `replay-pytest.log.gz` and `result.json` contain the
bounded replay result. The local Python and pytest versions differ from CI,
and one passing replay cannot exclude a contextual native race.

The local commands, run from their respective isolated worktrees, were:

```sh
CUDA_VISIBLE_DEVICES= QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg PYTHONPATH=/mnt/wd4tb/spacr-worktrees/qt-ci-e1probe-20261007 tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python -m pytest tests/qt/test_make_masks_parent_source_race_guards.py -q --timeout=1200 --timeout-method=thread
CUDA_VISIBLE_DEVICES= QT_QPA_PLATFORM=offscreen tools/run_capped.sh 4G /home/olafsson/anaconda3/envs/spacr/bin/python tools/replay_ci_batch.py --suite qt --shard 0 --batch 60 --output /mnt/wd4tb/scratch/e1-qt0-batch60-replay-20261007 --expect-sha e1f54c80650f60942ba6c98d4e999b36a3fa2846 --disable-plugin randomly
```

Relevant Git blob IDs for hosted source / current source:

| File | e1 | 284 |
| --- | --- | --- |
| `tests/qt/test_make_masks_parent_source_race_guards.py` | `7cd5f8ddfbbef326a82e68ce9127669ec948301a` | same |
| `spacr/qt/widgets/primary_mask_selector.py` | `bb956ef9cd15e29ea6572103476a48fbf54cc124` | same |
| `tests/conftest.py` | `df78195257599cb5b24baf4b94a7c6a1dd7c8930` | same |
| `.github/workflows/_pytest-suite.yml` | `b32f42230bb76812c196c5d31cc44919b5a18c9b` | same |
| `tools/run_pytest_batches.py` | `8b59fd00824a28d51c2cbeb69c2453daab28c31d` | same |
| `spacr/qt/screens/make_masks.py` | `46d8b8f3c95fae227abc2e7416f4e9fe1bc1464f` | `d34fd89abd641a9cb1ab3b75d4c57a9c6368fae8` |

The five-line Make Masks difference is the Draw-mode competing-right-click
guard; the failing parent-source setup and selector are unchanged. No source
fix is justified from this replay. Native crash diagnosis remains open.
