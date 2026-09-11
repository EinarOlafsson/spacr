RUN COMPARE: TWO REAL, UNCHANGED MEASURE SNAPSHOTS

This companion to lesson 50 contains the exact measurements databases from
two completed tutorial runs. It is a consistency demonstration, not evidence
of model accuracy, improved segmentation, or identical feature values.

Extract the archive. Activate an environment with spaCR installed, then run:

  python prepare_run_compare_download.py /absolute/path/to/new-comparison

Choose a NEW destination. The helper refuses to overwrite an existing folder.
It verifies each database against its original full SHA-256 fingerprint before
copying, and builds a new comparison index containing only those two artifacts.
This is an explicit preparation step, NOT an automatic GUI import. No Measure
pipeline is run. The helper changes only project/path and the relocation note;
original run IDs, artifact IDs, times, versions, settings and hashes are kept.
The original paths in source_records.json are historical provenance, not paths
you need to create on your computer. Unset SPACR_ARTIFACTS_DB before preparing.

In spaCR open Home > Data > Run Compare. Browse to the new comparison folder.
Choose the 03:37:56 Measure run as A, the 04:05:56 run as B, then click Compare.
Both were produced on 2026-09-11 by spaCR 1.5.0.5; a newer viewer does not change
that historical version. Show unchanged settings reveals equal recorded values.

Counts: cell 601, nucleus 622, pathogen 373, cytoplasm 601, png_list 601;
one plate, four wells, sixteen fields. The independent tutorial verification
also matched the object identities, not just these totals. The Counts tab's
zero badge is zero changes, not zero objects. A delta is B minus A.
Hits says no regression results were registered: this bundle contains no
regression results, so this is NOT a finding that there were no biological hits.

Both snapshots have matching recorded settings. They are not a parameter-change
experiment. Their database bytes differ; no claim is made that all measured
features are bit-identical. The registries refer to selected database artifacts
only, not complete original projects. Images, masks, models and the full run
journals are not included. Do not use this workspace to claim complete archival
reproducibility. The downloaded-data Measure tutorial explains their source.

Same-run selection produces a real warning. Use two distinct runs. A comparison
warning is something to investigate, not something to dismiss to obtain a result.
The public Python entry points are spacr.run_compare.runs_in and compare_runs.
No AI provider, GPU, network request or scientific analysis is needed here.
