# Item 987c: the V1 mask batches that still write archives

Status: spec, for Codex. Follows item 987b (spacr/_mask_batches.py).

987b builds the V1 normalised batches in memory for plain (Y, X, C) fields on one device or on parallel GPU
workers. Four routes still write `masks/*.npz` and so still break the maintainer's rule that Cellpose batches
never touch the disk:

1. **Timelapse** (`settings['timelapse']`): `<plate>_<well>_<field>_norm_timelapse.npz` per time stack, read by
   the tracking and the movie export. Plan entry = one time stack; `_load_batch` stacks the time points.
2. **Volumetric and native Z/T fields** (more than three axes): `_mask_archive_arrays(native=True)` streams the
   NPZ member into a private mapped file. Plan entry = one field; the mapped reader reads the field's `.npy`
   directly (memory-map), no archive.
3. **plot=True**: `plot_arrays` takes the first archive path; give it the plan batch.
4. **Illumination and PSF sessions**: the archives are staged and published as one complete set and the
   session records a field complete only when its archive is durable. Redefine "complete" as "its masks for every
   role are written" and apply the session correction inside `_load_batch` (the session object is needed in the
   worker for parallel runs, so a serialisable session description is part of the work).

Acceptance for each: on one real plate the merged files and every mask are byte-identical to the archive path,
no `.npz` under `masks/`, resume and rerun behave as today, and old projects with archives still load.
Keep the plan an object a chunked reader (item 983) can serve: only `_batch_fields` and `_load_batch` read pixels.
