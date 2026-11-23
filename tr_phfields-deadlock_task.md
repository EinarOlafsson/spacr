# phfields-deadlock English source handoff

Authority: Claude WORK_TASKS.md task 0aa. Base: d8f5386b1e5f6f413e87486c4157107a1267f298.

Home owns review and the nine-language records (sv, de, es, zh_CN, pt, hi, ko, is, fr).
The exact two revised package docstrings, the internal extractor docstring clarification,
their before/current hashes and the runtime template are in
`tr_phfields-deadlock_sources.json`. Preserve the `{site}` placeholder in the runtime message.

The public API has zero births, zero removals and one documentation revision, with unchanged
signature and annotations. Its exact prior/current body, signature and canonical blocks/layout
are in `tests/data/release_contracts/phfields_deadlock_api_change_2026-11-22.json`.
The immutable historical arrival JSON and generated catalogs/pins remain unchanged.
The runtime extractor admits only the `FieldStager.path` method through the existing
declared-owner collector. Its one new template is explicitly listed in
`tests/qt/test_i18n_caption_ratchet.py` under `_AWAITING_CATALOG_REBUILD`; remove that
entry when Home lands the generated and reviewed records (the existing ratchet
requires removal once the caption exists).

The private `_ph_fields` paragraph now explains that staged measure-only runs use staging order
and configured workers. The public `FieldStager.path` documentation names the queue error.
The runtime message tells the caller to request a field scheduled for the run.

## spacr.ops_engine._ph_fields

Source: `spacr/ops_engine.py:5960`

Previous source SHA-256: `82e7f681af7f273ea6ea3a61bec467c47f2deb4a942442ef63adc88605ff9d03`

Current source SHA-256: `e0449da8826ef9599339533bac2b545c582c95af5c78af18dac223935560361e`

```rst
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

RESUMABLE PER FIELD. Masks for each field go to a committed OME-Zarr
plate under ``<well>/phenotype_masks/``, or to ``site-<n>.npz`` with
``ops_ph_image_format='npz'``. Cells go to
``<well>/phenotype_cells/site-<n>.parquet``; a committed field
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
pipeline submits each field after segmenting it. Staged measure-only
runs use the same staging order and configured measurement workers.
Masks and tables are the same either way.

With ``ops_ph_measure_gpu`` and ``ops_ph_measure_gpu_check`` (the
default), the first field to measure is measured on the card and on
the CPU first and the columns beyond relative 1e-4 are reported under
``ph_measure_gpu_check`` (item 872, B7).
```

## spacr.ops_run_infra.FieldStager.path

Source: `spacr/ops_run_infra.py:167`

Previous source SHA-256: `051bae481604e16d511eef324e606c19f1fa4d5047055980b6ca7c7b55b962d6`

Current source SHA-256: `4bd8ec18e347ffe5e4a814c8c6d1e61d26358f7eb59e34b1e303c9e430b8a8aa`

```rst
Wait for one field to be staged.

:param site: the site.
:returns: ``(local path or None, unreadable)``; None when the field
    could not be read, in which case its slot is already free.
:raises ValueError: if the field is not in this staging queue.
```

## tools.build_i18n_catalogs._ngff06_runtime_sources

Source: `tools/build_i18n_catalogs.py:4746`

Previous source SHA-256: `4c590e899ba8839fc7c33820f21b0d4d1818d4efff49892e72c2bb801a2f91fc`

Current source SHA-256: `045f0b7aa9c0e4c926e34515604313d9330d1cb1049b2728046a0f404a076cce`

```rst
Return direct literal captions from the declared runtime owners.

:param tree: parsed source module, without importing backend code.
:param owners: top-level function names or class-qualified method names.
:returns: stripped captions, excluding unrelated and nested definitions.
```

## Queue error

Source: `spacr/ops_run_infra.py:180`

Current source SHA-256: `b8ebf564f31ff627a444d834a779f83d727ee549d20b9ac3d781fb2ff3cd066e`

```rst
Phenotype field {site} is not in the staging queue; request a field scheduled for this run.
```

## Runtime catalog review

Read-only extraction against the untouched task base found exactly one added runtime
caption and no removals (10,597 to 10,598 static sources). The existing report-only
external caption pin records this as an unreviewed arrival. Home owns the source
arrival record, generated catalog rows and nine-language review for the template
above; remove its awaiting entry when those records land. No pin was weakened.
