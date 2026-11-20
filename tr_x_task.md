# TASK X translation and arrival-contract handover

Source: ws-offload-codex-work-opscellpose, implementation commit 4fc43e0ab78ec5e937c10964bc1b38bbc64ada0d. Evidence-only commits do not change these strings. Base: ef603c010373e769be01b839d88fcf9f7ce9a045.

Languages: sv, de, es, zh_CN, pt, hi, ko, is, fr. Use the normal explicit AI-review label, with no human or native-speaker signoff. Preserve identifiers, placeholders, Sphinx roles and code literals.

This file inventories all 41 new or changed docstrings/messages, including test and private helper docstrings. Canonical extractor admission was checked against this source. Counts: {"canonical API admitted": 4, "private/nested docstring; NOT a rendered canonical API entry": 14, "test docstring; outside application/API source tree": 22, "runtime ui NOT admitted": 1}.

The four admitted API symbols can receive source-bound arrival contracts and records using their canonical blocks below. Private/nested and test docstrings are not rendered canonical API entries: do not create orphan review records for them. If the master intends to publish them, first arrange explicit source admission and regenerate the task from that new source.

The runtime checkpoint-mismatch ValueError is currently a plain formatted string in spacr/_mask_array_batches.py, and this module is outside the runtime extractor allowlist. Before runtime arrival contracts/records, the source lane must add the existing translation wrapper and extractor admission, then verify the extracted template/placeholder contract. No runtime record is valid against the current unadmitted source. This handover makes no source or catalog changes.

Do not translate archived source text. For changed admitted blocks, supersede the old evidence through the established overlay process; keep archived files intact. Validate exact source/context/SHA, placeholders and markup, and the usual per-record runtime/API checks. Skip docs assembly.

## Exact inventory

### 1. spacr/_mask_array_batches.py:1 — <module>

Kind: docstring. Admission: canonical API admitted.

```text
Give array-based mask inputs the same memory limits as prepared batches.
```

Canonical API symbol: `spacr._mask_array_batches`.

```json
[
  {
    "index": 0,
    "source": "Give array-based mask inputs the same memory limits as prepared batches.",
    "sha256": "43315dba26c73a1227d216eb02606a7bd5004adb6d92bc6cc48611589b3e116d",
    "context": "Give array-based mask inputs the same memory limits as prepared batches."
  }
]
```

### 2. spacr/_mask_array_batches.py:14 — <module>._batch_limit

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Resolve the shared field count before loading the rest of one batch.

    :param settings: shared field, tile and GPU memory settings.
    :param image: the first input, used to estimate the resident bytes.
    :param remaining: records still to process, including this input.
    :param gpu: whether the caller is using a CUDA model.
    :returns: the requested field count limited by available RAM and VRAM.
```

### 3. spacr/_mask_array_batches.py:36 — <module>._batch_plan

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Read a persisted grouping so resumed fields retain their tile count.

    :param path: optional plan path beside the pipeline checkpoints.
    :param settings: shared field and tile settings identifying the grouping.
    :returns: plan path and mutable grouping record.
    :raises ValueError: when a saved plan uses different grouping settings.
```

### 4. spacr/_mask_array_batches.py:59 — <module>._iter_array_batches

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Yield input records from one owned resident batch at a time.

    The loader retains the pipeline-specific preparation and normalisation
    contract. Inputs are packed without changing values; each field is
    restored to its original shape and dtype before public model eval.
    Every role of every field finishes before the next batch is loaded.

    :param records: input identifiers, in processing order.
    :param load: callable returning ``(image or None, metadata)``.
    :param settings: shared field, tile, precision and memory settings.
    :param gpu: whether device staging is enabled.
    :param plan_path: optional persisted grouping beside mask checkpoints.
    :param max_fields: optional upstream staging capacity.
    :yields: ``(record, image or None, metadata, batch field count)``.
```

### 5. spacr/_mask_array_batches.py:147 — <module>._array_batches

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Own an array batch until every role finishes, including on early exit.

    :param records: input identifiers; metadata alone may span many batches.
    :param load: the existing pipeline reader and preparation callable.
    :param settings: the shared mask settings.
    :param gpu: enable the existing GPU-fit check and resident device copy.
    :param plan_path: persist original group sizes for partial-batch resume.
    :param max_fields: cap loading by the capacity of an upstream staging queue.
    :yields: a field iterator; closing it frees the final resident batch.
```

### 6. spacr/ops_engine.py:2124 — <module>._cellpose_model

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
The Cellpose model the objects are segmented with.

    :param settings: read for ``key``. Empty or ``cpsam`` builds
        the default library model, which is how the well was validated;
        any other name is passed to Cellpose as the model to load.
        ``mask_inference_precision`` uses the shared auto/bf16/fp32 policy.
    :param gpu: put it on the card on this machine through
        :func:`spacr.accelerator.cellpose_kwargs`, or keep it on the CPU.
    :param key: the setting naming the model; the phenotype phases pass
        ``ops_ph_nucleus_model`` and ``ops_ph_cell_model``.
    :returns: a ``cellpose.models.CellposeModel``.
```

### 7. spacr/ops_engine.py:2161 — <module>._ops_mask_eval

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Run an OPS image through the shared mask tile policy and governor.

    OPS keeps its existing input preparation and public Cellpose
    normalisation. The mask site prefix enables the same full-tile-batch
    wait as Make Masks before memory pressure can change a tile batch.

    :param model: the original Cellpose model.
    :param image: the unchanged OPS plane or channel-last image.
    :param settings: the shared field, tile, precision and memory settings.
    :param site: the OPS segmentation site, prefixed with ``mask_``.
    :param field_count: inputs in the current resident batch.
    :param extra: the existing OPS public eval options.
    :returns: the original public eval result.
```

### 8. spacr/ops_engine.py:2352 — <module>._objects_windows_clear

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Delete window files, the batch plan and the empty folder for one well.

    :param folder: the window folder for the well.
    :returns: how many mask and done-marker files were deleted.
```

### 9. spacr/ops_engine.py:2462 — <module>._objects.load_window

Kind: docstring. Admission: canonical API admitted.

```text
Compose the existing covered input once, or reuse its saved mask.
```

Canonical API symbol: `spacr.ops_engine._objects.load_window`.

```json
[
  {
    "index": 0,
    "source": "Compose the existing covered input once, or reuse its saved mask.",
    "sha256": "da868682c6352a53b7cf5928134258898c264dc7f7a3277cf3c9a5911bb458d9",
    "context": "Compose the existing covered input once, or reuse its saved mask."
  }
]
```

### 10. spacr/ops_engine.py:4467 — <module>._ph_segment_field

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Nuclei and cells of one field, paired so nucleus k is in cell k.

    :param stack: the ``(H, W, C)`` field.
    :param channels: channel names in the stack.
    :param models: ``(nucleus_model, cell_model)``.
    :param settings: read for ``ops_ph_cell_channel``, ``ops_ph_downsample``,
        ``mask_eval_batch_size`` and the two diameters.
    :param field_count: fields in the current resident batch.
    :returns: ``(nuclei, cells, counts)``.

    NUCLEI ON DAPI, CELLS ON THE CELL STAIN WITH DAPI AS THE SECOND
    CHANNEL, as the Cellpose cytoplasm models are trained and as Funk et
    al. segment cells on phalloidin around their nuclei. Both run on the
    field block-averaged by ``ops_ph_downsample`` and the masks are
    repeated back up: at 2 a nucleus is still about 20 px across, which is
    what the models are trained on, and the cost falls fourfold.
```

### 11. spacr/ops_engine.py:4590 — <module>._ph_object_eval

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
One Cellpose evaluation of an object channel, as int32 labels.

    :param image: the plane or tile.
    :param model: the Cellpose model.
    :param settings: the shared mask settings passed to :func:`_ops_mask_eval`.
    :param batch: fields in the resident batch; the shared setting chooses tiles.
    :param extra: further Cellpose arguments (the diameter).
```

### 12. spacr/ops_engine.py:4622 — <module>._ph_segment_objects

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Pathogens and organelles of one field, as Mask Generation segments them (item 970).

    :param stack: the ``(H, W, C)`` field.
    :param channels: channel names in the stack.
    :param settings: read for the object settings
        (:func:`_ph_object_settings`) and ``mask_eval_batch_size``.
    :param gpu: run Cellpose on the card.
    :param cache: Cellpose models by name (:func:`_ph_object_model`).
    :param tile: ``tile(plane, segment)`` runs a Cellpose ``segment`` over
        ``plane`` piece by piece, as the large-site path does
        (:func:`spacr.ops_large_sites.segment_tiled`); None runs it on the
        whole field, as in the phenotype path. Classical organelle methods
        always see the whole field, since their thresholds are global.
    :param field_count: fields in the current resident batch.
    :returns: ``{class: labels}`` for the classes :func:`_ph_object_roles`
        names, each with the field shape; empty when none is set.
    :raises ValueError: for a channel that is not in ``channels``, or an
        organelle morphology and method Mask Generation does not pair.

    THE MASK GENERATION ROUTES, AT FULL FIELD RESOLUTION. Pathogens are
    segmented by Cellpose on the pathogen channel with
    ``ops_ph_pathogen_model`` and ``ops_ph_pathogen_diameter``, as
    ``pathogen_model`` and ``pathogen_diameter`` are in Mask Generation.
    Organelles take the organelle route from Mask Generation
    (:func:`spacr.object.generate_organelle_masks_sam`):
    ``ops_ph_organelle_morphology`` (spots, network, irregular or ring) and
    ``ops_ph_organelle_method``, a classical method
    (:func:`spacr.object._segment_single_image`, with the other
    ``organelle_*`` settings at the Mask Generation defaults) or ``cellpose``
    with ``ops_ph_organelle_model``. Both then go through
    :func:`spacr.object.merge_split_filter_masks` with their minimum area.
    Nuclei and cells are block-averaged by ``ops_ph_downsample`` first;
    these are not, since a pathogen or an organelle may be only a few
    pixels across.
```

### 13. spacr/ops_engine.py:4837 — <module>._ph_options_marker

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Refuse to mix per-field parts made with different options.

    :param folder: a folder of per-field parts.
    :param options: what this run makes the parts with. Missing historical
        keys are None or False; an explicit shared mask setting therefore
        refuses parts whose original tile count or precision was not saved.
    :raises ValueError: when parts made with other options are already
        there. A field that is not redone keeps its part, so a well would
        otherwise mix corrected and uncorrected fields without notice.

    A folder with parts but no marker is treated as missing every option.
    Shared mask settings must be recorded before those parts can be reused.
```

### 14. spacr/ops_engine.py:5283 — <module>._ph_fields

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Segment and/or measure every placed phenotype field of one well.

    :param db: the measurements database.
    :param plate: the plate name.
    :param well: the well name.
    :param phenotype_files: the phenotype index for this well.
    :param settings: the OPS settings; ``ops_ph_measure_workers`` (else
        ``n_workers``) sets how many fields are measured at once.
    :param gpu: segment on the card.
    :param destination: the run folder.
    :param segment: run ``ph_segment``.
    :param measure: run ``ph_measure``.
    :returns: the report.

    RESUMABLE PER FIELD. Masks for each field go to
    ``<well>/phenotype_masks/site-<n>.npz`` and its cells to
    ``<well>/phenotype_cells/site-<n>.parquet``; a field whose file exists
    is not redone, so a run stopped at field 900 resumes at 901. The well
    table, ``ops_phenotype_cells_<well>.parquet``, is the concatenation and
    is rewritten at the end.

    SEGMENTATION RUNS IN THIS PROCESS AND MEASUREMENT IN WORKERS: Cellpose
    already uses every core (or the card), while spaCR measurement of a
    field is one core for a few minutes on the 2,960 px reference plate
    fields, so measuring fields side by side is what keeps the machine busy.

    Segmentation loads one resident batch using the shared automatic
    ``batch_size``, RAM admission and GPU-fit helpers. All roles finish
    before that batch is freed in ``finally`` and the next batch is read.
    OPS channel preparation and public Cellpose normalisation are unchanged.

    FIELD BY FIELD WHEN ASKED (item 872, B1 and B2). ``ops_ph_pipeline``
    hands each field to the measurement workers as soon as its masks are
    saved, so the card segments field n + 1 while field n is measured, in
    place of segmenting the whole well first. ``ops_ph_stage_dir`` stages
    the maximum projection of each field on local disk with
    ``ops_ph_stage_threads`` reader threads, at most ``ops_ph_stage_ahead``
    fields ahead (:class:`spacr.ops_run_infra.FieldStager`): each raw file
    is then read once, not once to segment and once to measure, and the
    local copy is deleted when the field is done. Staging without the
    pipeline measures each field right after segmenting it. Masks and
    tables are the same either way.

    With ``ops_ph_measure_gpu`` and ``ops_ph_measure_gpu_check`` (the
    default), the first field to measure is measured on the card and on
    the CPU first and the columns beyond relative 1e-4 are reported under
    ``ph_measure_gpu_check`` (item 872, B7).
```

### 15. spacr/ops_engine.py:5498 — <module>._ph_fields.load_field

Kind: docstring. Admission: canonical API admitted.

```text
Read one unchanged OPS field once for every segmentation role.
```

Canonical API symbol: `spacr.ops_engine._ph_fields.load_field`.

```json
[
  {
    "index": 0,
    "source": "Read one unchanged OPS field once for every segmentation role.",
    "sha256": "dcc49d3bc4f1adef0e8af9ed5ee0dc7d3632272483d4a086040beaf243e559ce",
    "context": "Read one unchanged OPS field once for every segmentation role."
  }
]
```

### 16. spacr/ops_engine.py:12945 — <module>.run_spatial_masks.load_spatial

Kind: docstring. Admission: canonical API admitted.

```text
Keep the prepared spatial stain unchanged for shared residency.
```

Canonical API symbol: `spacr.ops_engine.run_spatial_masks.load_spatial`.

```json
[
  {
    "index": 0,
    "source": "Keep the prepared spatial stain unchanged for shared residency.",
    "sha256": "4c004f1b37d1bde183a39016efeb40d3091728bdc989fa039073de619b0c8b8d",
    "context": "Keep the prepared spatial stain unchanged for shared residency."
  }
]
```

### 17. spacr/ops_live.py:893 — <module>._run_spots.load_dapi

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Retain the current OPS crop and its public normalisation.
```

### 18. spacr/ops_live.py:1051 — <module>._run_objects.load_crop

Kind: docstring. Admission: private/nested docstring; NOT a rendered canonical API entry.

```text
Keep the same prepared phenotype crop for its two mask roles.
```

### 19. tests/test_ops_cellpose_shared_batches.py:1 — <module>

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
OPS keeps its inputs while sharing mask batch, precision and governor policies.
```

### 20. tests/test_ops_cellpose_shared_batches.py:15 — <module>.memory

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Admit small CPU inputs without depending on the load on this machine.
```

### 21. tests/test_ops_cellpose_shared_batches.py:23 — <module>.test_mixed_shapes_and_dtypes_keep_every_input_byte_and_one_batch

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Packing must retain NaNs, large uint64 integers and strided inputs.
```

### 22. tests/test_ops_cellpose_shared_batches.py:30 — <module>.test_mixed_shapes_and_dtypes_keep_every_input_byte_and_one_batch.load

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Record exactly which batch has been read.
```

### 23. tests/test_ops_cellpose_shared_batches.py:47 — <module>.test_an_error_frees_the_device_batch_before_any_next_batch

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
The shared resident free runs even when the first role raises.
```

### 24. tests/test_ops_cellpose_shared_batches.py:52 — <module>.test_an_error_frees_the_device_batch_before_any_next_batch.store

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Represent the admitted device copy with a real CPU tensor.
```

### 25. tests/test_ops_cellpose_shared_batches.py:57 — <module>.test_an_error_frees_the_device_batch_before_any_next_batch.load

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Load a field only when its own batch is reached.
```

### 26. tests/test_ops_cellpose_shared_batches.py:71 — <module>.test_partial_resume_keeps_the_original_automatic_tile_count

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
The last unfinished field still runs with its original three-field group.
```

### 27. tests/test_ops_cellpose_shared_batches.py:75 — <module>.test_partial_resume_keeps_the_original_automatic_tile_count.load

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Supply a tiny image with no pipeline-specific metadata.
```

### 28. tests/test_ops_cellpose_shared_batches.py:91 — <module>.test_loading_respects_an_upstream_single_field_staging_slot

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
An automatic batch cannot wait for a second field held behind its own slot.
```

### 29. tests/test_ops_cellpose_shared_batches.py:95 — <module>.test_loading_respects_an_upstream_single_field_staging_slot.load

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Refuse a second read until the preceding field has been consumed.
```

### 30. tests/test_ops_cellpose_shared_batches.py:108 — <module>.test_ops_eval_uses_the_shared_tile_policy_and_mask_wait_site

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
The original image reaches the same governed mask helper with a mask site.
```

### 31. tests/test_ops_cellpose_shared_batches.py:112 — <module>.test_ops_eval_uses_the_shared_tile_policy_and_mask_wait_site.evaluate

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Observe the spaCR helper without changing a Cellpose function.
```

### 32. tests/test_ops_cellpose_shared_batches.py:127 — <module>.test_the_retired_ops_tile_key_moves_only_through_the_shared_schema

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Saved OPS settings migrate once and an explicit shared key wins.
```

### 33. tests/test_gpu_memory_governor.py:375 — <module>.test_the_ops_phenotype_segmentation_calls_the_governor

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
OPS retains governed OOM retry through the shared mask helper.
```

### 34. tests/test_gpu_memory_governor.py:382 — <module>.test_the_ops_phenotype_segmentation_calls_the_governor.spy

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Observe the same shared helper the mask generator uses.
```

### 35. tests/test_gpu_memory_governor.py:573 — <module>.test_each_gpu_path_goes_through_the_governor

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Every route retains its direct helper and the shared helpers reach the governor.
```

### 36. tests/test_ops_objects_resume_958.py:112 — <module>._window_files

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Count mask/marker files while checking that their batch grouping is saved.
```

### 37. tests/test_ops_pathogen_organelle_objects_970.py:193 — <module>.test_the_default_run_writes_what_it_wrote_before

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
The same mask classes are saved with a shared tile/precision resume marker.
```

### 38. tests/test_spatial_submodules.py:615 — <module>.test_cellpose_is_driven_through_the_ops_model

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
The selected spatial model keeps its input and uses shared tile/precision controls.
```

### 39. tests/test_spatial_submodules.py:619 — <module>.test_cellpose_is_driven_through_the_ops_model._Model

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Observe public eval options without changing a Cellpose module.
```

### 40. tests/test_spatial_submodules.py:629 — <module>.test_cellpose_is_driven_through_the_ops_model._Model.eval

Kind: docstring. Admission: test docstring; outside application/API source tree.

```text
Return the same planted labels and record only supported eval keywords.
```

### 41. spacr/_mask_array_batches.py:52 — _batch_plan

Kind: runtime message. Admission: runtime ui NOT admitted.

```text
{path} uses different mask batch settings. Use the saved settings or move the mask checkpoint folder away before starting a new segmentation run.
```
