# Rejected native tile/GIL feasibility

No production, quality, cache-budget or workflow changes. Hard native 24 FPS
remains OPEN. This archive adds one installed-binding observation; earlier tile
receipts are historical evidence and were not rerun for this checkpoint.

The CUDA-hidden, offscreen, capped-4-GiB probe uses PySide6 6.11.2 and Python
3.12.13. One synthetic 3840×2160 `drawPath` call lasts 4523.50 ms. A Python
heartbeat makes zero ticks strictly inside the call with 10-ms endpoint margins;
the subsequent 100-ms sleep control permits 76 interior ticks. The thread joins.
The one-second Python switch interval limits incidental boundary scheduling.
This is evidence that this installed binding holds the GIL during rasterization,
not a universal statement about other PySide versions or production FPS. Peak
probe RSS is 77,972 KiB. Script, binary/typesystem identities, environment and
argv are bound in the receipts. The probe does not import the application.

Earlier source ac146d3c passes 48 expanded sequential comparisons when each
tile retains full native dimensions, original coordinates and integer clips.
Translated two-tile drawing changes six seam pixels at clock 3600.33. Earlier
balanced full-coordinate two-thread medians are 59.49→98.08 ms at detail 2 and
86.54→142.90 ms at detail 1; peak RSS is 939,528 KiB. Those JSON receipts are
included as historical evidence, not current-source performance acceptance;
the original renderer and full tile scripts remain in the referenced scratch
directory and are not duplicated here. Current f36b33c9 cache guards reject
clipped/non-owned painters, so independent tiles also bypass mature reuse.

The investigation is complete/rejected. There is no justified new tile or live
benchmark. Root requested renderer experiments stop pending an actual CI target.

`verify_proof.py` verifies filesystem payloads, or `--git REV` verifies committed
Git blobs without requiring this private archive commit to survive cherry-pick.
Only hash verification was run during archival; Qt experiments were not rerun.
