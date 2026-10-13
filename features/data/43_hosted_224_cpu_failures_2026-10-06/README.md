# Exact-224 ordinary CI failure receipt

Ordinary run 37537358238 tested Git source `22413aa4d072fe4eb06303cf82d0c6fbfe677d61`. The pre-cancel snapshot has 19 failed jobs, seven successful jobs and two running jobs. Every one of those 19 full failed logs is retained. After the authorized cancellation, the release gate became the 20th failed job because its blocking jobs had failed; its full log is retained separately. The two remaining Fast jobs were cancelled, not counted as test passes or test failures. Only this superseded ordinary run was cancelled; protected serial runs were untouched.

`triage.json` separates CPU repairs, ongoing fungal visibility, workstation-owned generated artifacts and downstream failures. A job can have more than one category. The coverage-combine job refused failed shard tests before evaluating the absolute-count ratchet, so this run supplies no numerical coverage verdict.

The focused sorting/retry receipt is from integrated source `aee25140049788161edd047e59bc9cda6927e810`: 81 passed in 24.52 s under 4 GiB. `manifest.json` binds exact Git file blobs for both source versions and SHA-256 of every archived byte; `verify_archive.py` checks all payload hashes without network access.
