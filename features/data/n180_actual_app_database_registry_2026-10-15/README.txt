N180 actual application database workspace contribution and restore sequencing

The real UMAP screen previously omitted the database provider entirely. After
registering it, the complete roundtrip revealed settings replay erased the
carried database association. Providers now resolve the current built database
control without constructing deferred settings; same-app settings replay before
dependent panels. Other panel ordering, input data, skip handling and plain
settings names remain unchanged.

Actual copied/relocated SQLite roundtrip passes with complete providers and
settings. There are48 distinct passing cases across47+1 runs. Original missing
registration and ordering failures, and one incorrect test expectation failure,
remain intact. Read receipt.json for source hashes and exact limits.
