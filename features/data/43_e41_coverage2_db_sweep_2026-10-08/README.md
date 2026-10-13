The complete e41 Coverage2 log contains four failures in two CPU test files.
Two exposed a real database migration defect: scientific column canonicalisation
renamed the private `_spacr_write_queue_commits.field` column to `fieldID`,
although the writer still reads `field`. The fix excludes reserved `_spacr_`
tables from that migration. The existing scientific column count, schema and
value assertions remain; the updated test also writes and replays a real
ticket after migration.

The other two failures used old resource test doubles. A contained child now
removes stale `_trial_result.json` before launching; the double must create a
fresh result, and proves that the stale one was removed. A parallel sweep now
passes a paced wrapper around the `spawn` context; the double checks both
properties while retaining its trial submission, row and order assertions.
Neither resource production module changed.

All four hosted nodes pass locally. The complete two owning files plus
database-schema and write-queue adjacency pass 89 tests under a 4 GiB cap.
The e41 and final source bytes, full hosted log and raw local receipts are
compressed with original SHA-256 hashes. `python verify.py --current` verifies
the filesystem and current source; `python verify.py --git --current` also
verifies the committed payloads.
