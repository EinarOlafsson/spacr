First ordinary hosted native-route receipt, 2026-10-07

Exact source6836512a98, run37698599898, Coverage7job113058129395.
Artifact11517807991 contains35 text/JSON files18,454 uncompressed bytes;
33 distinct PID journals contain only session start/end identity records.
Actual systemd handler/storage route is verified, pytest shell core limit
is16GiB, and diagnostic collector/upload both complete successfully.
No native crash appears in the full job log. Sole failure is the older API
inventory13214 vs13199(+15), handled by the catalog owner. Collector correctly
reports zero unfinished identities/no core. This proves hosted initialization
and artifacts, not a crash-to-core recovery or resolved spaCR native fault.
Raw artifact ZIP, full compressed log, exact job/artifact metadata and source
hashes are retained. No core was uploaded. Subsequent fault-context update
is absent from this source and is separately proved.

Run provenance is retained separately. The uncompressed raw log SHA256 is
de252d60694f7d2d651f00b40c2d826949ae81dfbc0f6c6e8f87417726cac680.
At raw-log line6014, 23:06:43.6002252Z, the primary hosted puncta True case
PASSES. This single current pass does not resolve the historical native cause
or accept every Qt case. All Qt/native acceptance remains separately tracked.
