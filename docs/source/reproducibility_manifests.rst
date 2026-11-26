Reproducibility manifests
=========================

Every pipeline launched from the spaCR application or ``spacr-run`` creates a
run folder below ``~/.spacr/runs``. Recording happens inside the pipeline
worker, so inspecting and hashing a large plate does not block the desktop
event loop.

Each folder contains:

``settings.json`` and ``settings.csv``
   The complete resolved settings used by the pipeline.

``manifest.json``
   A versioned, atomically written record of the module, timestamps, status,
   settings hash, declared seeds, available random-state metadata, spaCR and
   Git versions, recorded package versions, model/file records, warnings and
   an exception traceback when captured. NumPy state and Torch's initial seed
   are recorded only when those libraries are already loaded; file hashing is
   optional as described below.

``log.txt``
   The tail of the application log at completion.

``outputs/``
   Artifacts explicitly attached by pipeline code.

``environment/``
   ``requirements-lock.txt`` contains ``pip freeze`` output, or package
   name/version entries from installed-distribution metadata if pip cannot
   run. ``conda-explicit.txt`` is added only when a conda environment is
   detected and ``conda list --explicit`` succeeds. ``environment_lock`` names
   the files actually written and the environment digest. When the conda file
   is present, use ``conda create --name rerun --file conda-explicit.txt``
   followed by ``python -m pip install -r requirements-lock.txt``; otherwise
   install the requirements into a compatible Python environment.

   The lists are cached under ``~/.spacr/env_locks/<digest>`` and reused for
   the same digest. That digest includes the interpreter identity and recorded
   package versions; package metadata is cached within the process, so restart
   spaCR after changing installed packages to obtain a new snapshot. Failure
   to write a run's lockfiles is recorded under ``provenance_warnings`` and
   does not by itself stop the pipeline.

File provenance
---------------

Set ``hash_inputs=True`` to record input/output file hashes; hashing is off
by default and the manifest then reports ``input_hashing="skipped"``. Discovery
follows path-valued settings, including nested plate lists. Hashed file records
contain the full SHA-256, size, modification timestamp and referring setting
key. Directory walks skip linked entries, version-control folders, caches and
the run journal. Before/after inventories used to identify changed outputs
are limited to 200,000 files per root; truncation is reported in
``provenance_warnings``.

The aggregate ``input_tree_sha256`` and ``output_tree_sha256`` values hash the
recorded dictionaries, including their paths and metadata. Matching values
mean those records match; they do not prove complete file coverage when
hashing was skipped or provenance was incomplete. Check ``input_hashing``,
``provenance_warnings`` and the per-file records before comparing runs.

Crash and failure behavior
--------------------------

A ``running`` manifest is written before the pipeline starts. Normal context
exit attempts an atomic update with success, failure or cancellation status;
caught exceptions are re-raised after their traceback is captured. Forced
process termination or failure to write the final manifest can leave the
stored status as ``running``. Run History can display it as ``interrupted``
when its recorded owner is gone; this does not rewrite the stored manifest.
Provenance failures are logged, with recorded warnings retained under
``provenance_warnings`` when the manifest can be written.

Settings versions and changed defaults
--------------------------------------

Every settings file spaCR writes starts with ``spacr_settings_version``: the
run folder's ``settings.json`` and ``settings.csv``, the settings saved next to
a run, and the file Export settings writes. The number says which defaults the
settings were resolved under.

A file written before a release changed a default does not hold the changed
key, so it would run with the new default. When a settings file is read by
``spacr-run``, ``spacr repro``, the batch queue or Import settings, each key
whose default changed after the file's version, and that the file does not set,
gets the default it had then. A file without ``spacr_settings_version``, such
as a hand-written one, is read as the oldest version, and the log names every
old default it was given. A key the file sets always wins. To use today's
defaults, set those keys or add the current ``spacr_settings_version`` to the
file.

The changes are kept in one table, ``DEFAULT_CHANGES`` in
``spacr/settings.py``, keyed by the number of the item that changed the
default. A key is always above every earlier one, so an item that changes a
default after a higher-numbered item takes the next free number:

- 931: ``ops_compose_blend`` was ``'average'``; it is now ``'feather'``.
- 932: ``ops_registration_precision`` was ``'float64'`` and
  ``ops_registration_backend`` ``'cpu'``; they are now ``'float32'`` and
  ``'gpu'``.
- 933 (item 930's default switch): ``ops_align_model`` was
  ``'similarity'``; it is now ``'tile_affine'``.
- 934 (item 937): ``aggregation_defaults`` was ``'legacy'``; it is now
  ``'per_feature'``, so the screen pipeline also writes a well table in which
  every feature takes its own default.
- 935 (item 940's default switch): ``ops_placement_seed_export`` was
  ``False``; it is now ``True``.
- 936: ``ops_blas_threads`` was ``1`` and ``ops_consensus_space``
  ``'refit'``; they are now ``2`` and ``'hybrid'``.
- 937 (item 941): ``profile_optional_families`` and
  ``ops_ph_profile_optional_families`` were ``['texture_gabor',
  'colocalisation', 'colocalisation_extra']``, the slow families the
  ``'profiling'`` preset measured; they are now ``None``, so ``'profiling'``
  leaves Gabor, granularity and colocalisation out.
- 938 (item 945): ``seg_mask_vram_gb`` was ``0`` (the
  large-site mask step had no card-memory cap of its own); it is now ``6``.
- 939 (item 942): ``ops_decode_auto_spot_grid`` was ``[]`` and
  ``ops_decode_auto_unreachable`` ``'strictest'`` (item 939's auto mode
  tuned the calling thresholds only and fell back to the strictest setting);
  they are now ``[150.0, 200.0, 250.0, 315.0, 400.0]`` and ``'lowest_fdr'``.
- 940 (item 946): ``ops_decode_preset`` was ``'none'``,
  ``ops_decode_threshold_mode`` ``'manual'`` and ``ops_decode_auto_spot_grid``
  ``[150.0, 200.0, 250.0, 315.0, 400.0]``; they are now ``'rule_a_s2'``,
  ``'auto'`` and ``[]``, so a run decodes with rule (a) + S2 and per-well
  thresholds chosen from decoys.
- 941 (item 953): ``ops_cluster_published_controls`` was ``False`` (a
  published comparator's control rows were left out of its Fisher
  background); it is now ``True``, so the comparator is scored with its own
  control rows in the background, as the publication scored it.
- 942 (item 954): ``ops_bench_null_permutations`` was ``200``; it is now
  ``1000``. The first 200 permutations are the same.
- 943 (item 965): ``seg_cells_from_nucleus_seeds`` was ``False`` and
  ``seg_merge_small_nuclei_frac`` ``0.0``; they are now ``True`` and
  ``0.25``, so the large-site path (``ops_large_sites``) grows one cell
  from every nucleus after merging nucleus pieces smaller than a quarter of
  the site's median nucleus area into the nucleus they touch. Funk's
  phenotype path does not read either key.
- 944 (item 966): ``ops_read_mode`` was ``'log'``,
  ``ops_prefix_threshold_tuning`` ``'none'``, ``ops_prefix_decoy_k`` ``0``
  and ``ops_prefix_fdr_target_pct`` ``0.1``; they are now empty,
  ``'per_well_decoy'``, ``4`` and ``1.0``. The spacer-prefix decode reads an
  empty ``ops_read_mode`` as ``'auto'`` while it tunes, so each well chooses
  the log or raw read and its thresholds from decoy calls at a 1 % decoy FDR.
  The OPS barcode decode reads an empty ``ops_read_mode`` as ``'log'``, so
  Funk-style runs are unchanged.
- 947 (item 967): ``ops_prefix_minority_min_reads`` was ``0`` (off); it is
  now ``3``, so in the spacer-prefix decode a barcode with fewer than 3 reads
  in a cell is left out of the call when another barcode of the cell has more
  reads. Under ``ops_prefix_quality_method`` ``'spacr'`` no call changes; the
  cell table gains ``minority_ignored`` and ``minority_rescued``. The OPS
  barcode decode does not read the key.
- 948 (TASK X): ``ops_ph_tile_batch`` was ``8``, ``ops_sbs_tile_batch``
  ``16`` and ``spatial_tile_batch`` ``16``, the Cellpose tile batches the OPS
  phenotype, sequencing and Spatial Mask Generation paths always ran at; they
  are now ``0``, which uses the shared ``mask_eval_batch_size`` (one tile per
  field in the resident batch when that is 0). An old file's
  ``ops_ph_batch_size`` is renamed to ``ops_ph_tile_batch``, so it still sets
  only the phenotype path. 946 and 947 belong to other items.
- 949 (OME-Zarr defaults): new spatial Mask Generation field output and
  Measure field input and crop output use ``'ome-zarr'``. Older files retain ``mask_image_format='npy'``
  and ``crop_image_format='png'``. New OPS phenotype fields and crops use
  ``ops_ph_image_format='ome-zarr'`` and ``ops_crop_image_format='ome-zarr'``;
  older files retain ``'npz'`` and ``'tiff-tar'`` respectively. This does not
  change ``ops_storage``. New Annotate, Image UMAP and classification source
  settings select OME-Zarr; older files retain ``crop_source='auto'`` and
  ``image_source='load_images'`` when absent. An explicit older ``crop_source``
  alias supplies a missing ``image_source`` before that default is filled.
  Measure still needs retained merged filenames, headers and plane layout;
  unavailable or uncommitted native fields, time-series and ambiguous spatial
  identities use NPY. A committed native field with an incompatible shape,
  dtype or plane layout raises an error. Timelapse keeps the legacy NPY field
  path because the spatial HCS store has no time axis.
  Explicit source selections always win. Native classification datasets keep
  one ``crop_index.csv`` pointing to committed crops instead of exporting an
  image file per object; keep the source project with the dataset when moving it.
- 952 (item 1006): ``ops_flatfield`` was ``'none'``; it is now ``'exact'``, so
  an OPS phenotype run divides every pixel by the plate's flat-field maps
  before measuring and needs them (phases ``ph_flatfield`` and
  ``ph_flatfield_objects``, or a prebuilt ``ops_flatfield_maps`` folder).
  ``ops_flatfield_channel_fit`` was the earlier maps, ``'legacy'`` (the
  polynomial DNA map and pixel-median maps of the other channels); it is now
  the object median of every channel, nuclear for DNA and
  ``ops_flatfield_nuclear_stains``, cytoplasmic for the others. 949-951 belong
  to other items.

- 955 (item 1011): new Mask Generation and OPS runs use ``batch_size='auto_vram'``
  and ``mask_eval_batch_size='auto_vram'``. One shared plan measures GPU memory,
  including resident field arrays, resolves a tile batch up to 32, and splits
  fields evenly. CPU runs use at most 8 tiles and 50 fields per batch. Fields
  are shuffled with ``mask_batch_seed=42`` when ``randomize`` is true; stitched
  sequencing windows, Spatial Mask Generation and timelapse keep acquisition
  order. The earlier Mask shuffle was unseeded. The plan and run journal retain
  the seed, field order, batch sizes and resolved tile batch for resume and replay.
  A declared ``gpu_vram_budget_gb`` (or ``SPACR_GPU_VRAM_GB``) pins the budget
  across cards; without it, free GPU memory can change the resolved sizes.
  A 32-tile phenotype batch can need about 8 GB including resident fields.
  Only the cap, seed and budget controls require Show alpha features; the
  automatic behavior applies with alpha controls hidden. Files stamped before
  955 retain their historical field count and tile defaults: 0 for both keys
  since schema 945, and 50 fields before 945. Explicit values always win;
  pre-948 OPS files still retain their per-path tile batches of 8/16/16.
  Old OPS files that omit ``randomize`` keep acquisition order (False); Mask
  keeps its earlier True default, and an explicit choice always wins.
  Set ``batch_size=0`` or ``'whole'`` for the earlier batch estimate and
  ``mask_eval_batch_size=0`` for one tile per resident field.
- 957 (item 1014): ``ops_decode_fdr_calibration`` and
  ``ops_prefix_fdr_calibration`` were ``'none'``; new files use ``'never'``
  to measure the FDR scale from substitution-built control barcodes. Older
  files keep the entered fixed scale and their original decode tables.
  ``'none'`` keeps the earlier uncalibrated tables and reports no shuffles.
- 958 (item 1008): ``cytoskeleton_object_support`` is now ``True``.
  Measure no longer counts cytoskeleton skeleton pixels outside the object.
  Older Overview settings retain ``False`` and reproduce their whole-image
  fibre counts. New per-cell cytoskeleton measurements use the corrected
  object support and 24-pixel padded crops.

- 959 (item 1013): ``ops_read_neighbour_discount`` is now ``0.25``.
  Files stamped before 959 retain ``1.0`` (off), including files from
  schemas 957 and 958. The recipe-specific prefix bleed switch stays
  off by default until its separate acceptance is complete.
- 960 (OPS default table): ``run_ops`` now fills every OPS default itself, so
  a direct call of the Python function resolves the same values as the
  command line and the GUI (for example ``ops_decode_threshold_mode``
  ``'auto'``, ``randomize`` True). Files stamped before 960 keep
  ``ops_feature_screen_permutations`` 0 and ``ops_feature_screen_ridge`` 0.0;
  the other keys that differed already have rows (940, 944, 947, 955, 957).

- 961 (merge 4, one size filter): ``size_filter_inclusive`` is now ``True``.
  An object whose size equals the minimum or maximum is kept everywhere, as
  Measure and the Make Masks window already did. The legacy Make Masks
  post-Cellpose size filter (``filter=True``) and the plot filter dropped it.
  Files stamped before 961 retain ``False`` and reproduce their masks.

- 962 (merge 9b, one per-well centre-scale routine): ``ops_well_statistics``
  is now ``'two_pass'``. The TVN's per-well centre-scale of its PCA input and
  the embedding profiles' per-well mean and standard deviation share one
  routine (``spacr.ops_tvn.batch_mean_sd``): float64, the mean first and then
  the squared deviations from it, read in bounded chunks, and a feature
  with no spread scaled by 1. Files stamped before 962 retain ``'legacy'``
  (StandardScaler in the in-memory TVN; single-pass float32 sums with a
  variance floor of 1e-12 in the embedding profiles) and reproduce their
  earlier output. Numbers 960 and 961 belong to other merges.

- 964 (OPS decode audit, the pixel fallback): ``ops_pixel_read_gate`` is
  now ``True``. The pixel fallback assigns a nucleus only when its reads
  named no guide, or named the pixel guide alone; a read tie, a read vote
  split between two or more guides, or a lone read guide the pixels
  contradict leaves the nucleus uncalled. Files stamped before 964 retain
  ``False`` and reproduce their calls. The decoy FDR report of each well
  (``decode_fdr_<well>.json``) now counts the read calls and the
  pixel-fallback calls apart, each with its own FDR.

- 966 (item 1023, stricter genotype calls): ``ops_min_top_share`` is now
  ``0.9`` (was ``0.0``), ``ops_footprint`` ``3`` (was ``10``) and
  ``ops_read_neighbour_reach`` ``10`` (was empty, which follows
  ``ops_footprint``). A nucleus is called only when its top guide holds at
  least nine tenths of its weighted read votes, by its reads or by the pixel
  fallback; reads count within 3 pixels of the nucleus; the neighbour
  discount keeps its 10-pixel reach. Files stamped before 966 retain the old
  three values and reproduce their calls. ``decode_fdr_<well>.json`` now
  reports the split-read concordance of each well
  (``split_read_concordance``). Number 965 belongs to another item.

- 967 (item 1024, nucleus pairing, alpha): ``nucleus_fragment_merge`` and
  ``ops_ph_nucleus_fragment_merge`` (``'off'``) and ``cellless_nuclei`` and
  ``ops_ph_cellless_nuclei`` (``'drop'``) are new, with their thresholds.
  Both are off by default, so a run pairs nuclei with cells exactly as
  before; the row pins the off values for files stamped before 967, and
  the four fragment tests that are off unless set
  (``nucleus_fragment_min_contact_px`` 0, ``nucleus_fragment_max_dip``,
  ``nucleus_fragment_max_valley_dip``
  and ``nucleus_fragment_max_area_rel`` None, and their ``ops_ph_`` twins).


- 970 (item 1026, the seam owner centroid): ``ops_cross_field_owner_centroid``
  is now ``'linked'``. A cell imaged in overlapping fields gets one centroid,
  the mean of its whole copies in stitched coordinates, and is owned by the
  field whose centre is nearest that centroid; when that field holds no whole
  copy, the nearest field that does owns it, as before. Files stamped before
  970 retain ``'per_copy'`` (each copy's own centroid against its own field
  centre, so registration error between copies could pick a farther field)
  and reproduce their owners. Numbers 966 to 969 belong to other items.

- 971 (item 1031, the tile origin): ``ops_tile_origin`` is now ``'paste'``.
  The composite pastes each sequencing tile at a whole pixel, so the nuclei
  found on it, and the tile-affine transforms fitted to them, are in that
  rounded frame. The decode now moves nuclei into a tile, and reads back
  into the well, by that whole-pixel origin, and the tile-affine merge
  centres its tile groups on it. Files stamped before 971 retain
  ``'float'``, the stored fraction of a pixel (up to half a pixel off on
  each axis), and reproduce their calls. Numbers 966 to 970 belong to other
  items.

- 969 (item 1025, SBS cell segmentation, alpha): ``ops_sbs_cell_method``
  (``'off'``) is new, with ``ops_sbs_cell_model`` and
  ``ops_sbs_cell_threshold``. Off by default, so a run decodes exactly as
  before; the row pins the off value for files stamped before 969. Number 968
  belongs to another item.

- 972 (OPS storage for reruns, items N6 and N7): ``ops_store_raw_masks``
  (default ``False``) keeps the phenotype nucleus and cell masks before
  pairing as ``nuclei_raw`` and ``cells_raw`` labels, and each sequencing
  segmentation window in ``sbs_masks/<plate>.ome.zarr``, so pairing and
  object numbering can be redone without segmenting.
  ``ops_read_cache_floor`` (default ``100``) is the weakest spot the
  decode field cache keeps, until now a fixed 100. Files stamped before
  972 retain ``False`` and ``100`` and reproduce their outputs. Numbers
  966 to 971 belong to other merges.

- 976 (spanning-children merge off, maintainer decision 2026-11-25):
  ``ops_ph_spanning_merge`` (OPS) and ``spanning_merge`` (Measure, External
  Masks) now default to ``'best_parent'``, both through
  ``spacr.measure._assign_nuclei_to_cells``: a nucleus that lies in several
  cells goes to the cell it overlaps most and no cell is merged. The old
  ``'merge'`` (a nucleus with under 90 % in every cell merged those cells,
  even cells holding their own nuclei; at 90 % or more its pixels in the
  other cells were erased) is treated as a bug and kept only for replay. The
  time-lapse tables use the same kernel and now follow ``'best_parent'``
  too, with no setting of their own. Files stamped before 976 replay what
  they ran:

  - Measure and External Masks, every version below 976 and unstamped:
    ``spanning_merge = 'merge'``.
  - OPS, stamped below 959 or unstamped (the frozen A1 run, commit
    26bc722c15, schema 958): ``ops_ph_spanning_merge = 'best_parent'``,
    which reproduces that pairing exactly.
  - OPS, stamped 959 to 975: ``'merge'`` (OPS pairing merged from commit
    1612c53561, schema 959). A file stamped 959 before that commit landed
    (2026-11-22 to 2026-11-24) also gets ``'merge'``; set the key to
    replay it.

  Numbers 966 to 975 belong to other items.

- 978 (item 1039, phenotype segmentation at full resolution, maintainer
  decision 2026-11-25): ``ops_ph_downsample`` is now the cell factor and
  ``1`` (was ``2`` for nuclei and cells); the new
  ``ops_ph_nucleus_downsample`` is the nucleus factor, ``'same'`` by
  default (it follows ``ops_ph_downsample``), so nuclei are at ``1`` too.
  The nucleus and cell diameters stay empty, so nothing is resized. A set
  diameter is in full-resolution pixels at every factor. Files stamped
  before 978 get ``ops_ph_downsample`` ``2`` and
  ``ops_ph_nucleus_downsample`` ``'same'`` and reproduce their masks.
  Number 977 is reserved for another item.
- 979 (A3-BUG2, one outside region): no default changes. Measure's outside
  ring block (``<object>_channel_<i>_outside_<stat>``) now takes its ring
  from ``profile_outside_rule`` through
  ``spacr.measure._outside_ring_selections``, the function the ``outside``
  profiling family and the OPS phenotype path use, so the same rule gives
  the same outside pixels in Measure and OPS. Before, the block ignored the
  rule: under ``'background'`` Measure's nucleus ring kept the neighbouring
  nuclei and OPS's kept background only. A Measure file stamped before 979
  whose ``profile_outside_rule`` is ``'background'`` (set, or from the 965
  row) is read as ``'legacy'``, which replays the old split (block
  ``'object'``, family ``'background'``). ``ops_ph_profile_outside_rule``
  is unchanged, since OPS never ran the block, and refuses ``'legacy'``.
  Numbers 977 and 978 belong to other items.
- 980 (item 1036 taken back, maintainer decision 2026-11-25):
  ``ops_sbs_cellpose_diameter`` is ``None`` again by default, so the OPS
  sequencing nuclei are segmented at native scale and nothing is resized.
  On a whole well the diameter added 0.2 % objects, mostly bright specks,
  and 9, 11 and 13 pixels gave the same result. Files stamped 973 to 979
  get ``11.0`` and reproduce their objects; files stamped below 973 keep
  ``None``. The setting stays available.
- 981 (item 1044, pixel-fallback ownership): ``ops_pixel_ownership`` is
  ``'read_rule'``. A fallback pixel votes for the nucleus a read at that
  place would go to (``ops_footprint``, the tie and territory rule, the
  cell label under a cell mask; ``ops_pixel_radius`` only caps it), the
  read neighbour discount acts on the pixel tallies, and a pixel call is
  refused when its guide is the read guide of a neighbour within
  ``ops_read_neighbour_reach``, the pixel guide of a read-less neighbour
  with as many pixels, or not the guide the pixels within
  ``ops_pixel_radius`` call (so no pixel call the old rule would not
  make). ``'read_reach'`` keeps the assignment without that
  gate. Files stamped before 981 get ``'nearest_nucleus'``, the nearest
  nucleus within ``ops_pixel_radius``, and decode as before. Pixels are
  assigned at ``ops_pixel_footprint``, ``5`` px (item 1044b); reads keep
  ``ops_footprint``. Files stamped before 981 get ``ops_pixel_footprint``
  empty, so an older file that sets ``'read_rule'`` itself assigns its
  pixels at ``ops_footprint``.

Public API
----------

Use :func:`spacr.run_journal.open_run` around a custom pipeline. Within the
context, :meth:`spacr.run_journal.Run.record_model` records a model path;
:meth:`spacr.run_journal.Run.record_input` and
:meth:`spacr.run_journal.Run.record_output` can add paths absent from settings
when ``hash_inputs=True``. Those two methods are no-ops when hashing is off.

.. code-block:: python

   from spacr.run_journal import open_run

   with open_run("my_assay", settings) as run:
       run.record_model("classifier", settings["model_path"])
       result = run_assay(settings)
       run.record_output(result)

``spacr-repro <run-folder>`` replays supported modules with the recorded
settings. The complete API is generated under :mod:`spacr.run_journal`.

Before replaying, use ``spacr-repro <run-folder> --dry`` to inspect the
recorded paths. Replay creates a new journal but reuses recorded output
destinations, so keep a copy of earlier results if needed. Successful
completion does not automatically verify that the new results match the
original run.

Resolved automatic parameters
-----------------------------

Settings schema 950 adds ``spacr_resolved_parameters`` without changing any
defaults. Spatial Mask Generation stores the exact normalization bounds of
each field and the diameter used for each field and object role. A saved
``None`` explicitly means native scale. The display summary may round a
diameter; replay uses the full saved value.

Load the completed run settings or its ``gen_mask_settings.csv`` or JSON
copy to reuse those values. The GUI retains the record through loading and
collecting settings. A selected field can reuse its original cohort bounds
without rerunning the percentile or diameter estimator. Source pixels,
channels, model weights and inference options must match. Missing fields,
changed settings and different padded input shapes are refused rather than
silently fitted again. To estimate again, load a fresh settings file without
the reserved record and select the existing automatic inputs.
Parallel checkpoints also verify these exact values before reusing completed
masks; missing or different records require a separate output folder.

Resolved PSF sampling and width, illumination and Noise2Void model paths,
and replication linkage distance are saved in their existing scalar settings.
Keep fitted model files available at the recorded paths. Each input folder
keeps its own resolved settings; one folder does not seed another with its
fitted model.
For a run over several folders, load the settings file beside the folder
to replay; the run snapshot holds the currently processed folder.

This replay contract covers the spatial shared Mask Generation helpers.
It does not establish byte identity for frozen OPS, time-lapse, native T/Z,
or calibrated partition workflows, or across model/runtime changes.

