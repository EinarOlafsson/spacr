# Obsolete e1f Fast shard 0 failure, 2026-10-07

Run `37559626498` at source
`e1f54c80650f60942ba6c98d4e999b36a3fa2846`, job `112602607352`
(`Fast tests`, shard 0), failed four of 121 batches. The complete hosted log
is `fast0.log.gz`: compressed SHA-256
`50e4106b83a97446a71dc59b0cd9d8febd8407801e8b77fd9c3d3ceae6a8a5af`;
uncompressed SHA-256
`d2e48cf56abb4f8997ff4925562b6a5ed51f54252bd53272ceb2853b94105d08`
(6,974,033 bytes). This was read after job completion; no local replay or
test-policy change was made for these five assertions.

| Batch | Failed assertion | Disposition |
| --- | --- | --- |
| 91 | `test_written_review_scope_matches_current_source_bound_evidence` timed out while nine locales shared one test | Test-only parameterization `08b497f9c8`; nine focused branch-coverage cases passed, archived separately under `43_i18n_review_timeout_cpu_2026-10-07` |
| 103 | Swedish and French `test_*_reviewed_runtime_text_is_source_bound_and_gate_clean` | Exact retired-tooltip inventory mismatch remains with the reviewed-translation owner; no catalog/policy edit here |
| 104 | `test_every_added_key_is_declared` omitted four F567 timelapse event video keys | CPU test contract was corrected on a later source checkpoint |
| 111 | `test_the_checkout_size_in_the_readme_matches_the_tree` expected 1710 MB but measured 2797 MB | README clone-size measurement and dated prose remain with the documentation owner |

`run-before-cancel.json` and `jobs-before-cancel.json` are direct GitHub REST
snapshots: run `in_progress` with no conclusion, jobs 17 failure, 10 success,
one in progress. The lone active job was Qt shard 1, id `112602607623`.
These files are a pre-cancellation snapshot, not an assertion that the run
later finished or passed. This agent did not cancel any run. Current-source
Tests run `37571318127` at `28449ef0c8` was pending when this was recorded.
