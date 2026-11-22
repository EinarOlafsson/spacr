# Item 990 follow-up translation handover

Branch: `ws-offload-codex-work-990fix`; assigned pilot `e2febd27aa65a46fc195948ca9af995076947aa4`.

Home owns the nine-language records and catalog rebuild. The companion tr_990fix_sources.json gives exact bodies, hashes, signatures, canonical prose blocks, source locations and final file hashes. No translation or catalog is written by this branch.

## Behavior

Training and fine-tuning accept full-well reference cells. The run supplies all cells to fine-tuning, pooled label rows and labelled scoring, using the same normalizer as prediction. Existing mitotic normalization helpers supply the statistics. An omitted training reference retains the complete-population assumption with the existing warning.

The S2 null now permutes daughter-eligible cells within each field. It replaces the earlier all-rows null. Field indices and eligible partner geometry are cached once; measurement adds nucleus_centroid_y, nucleus_centroid_x and nucleus_area from the nuclear mask while preserving all existing columns and values. Missing geometry still has an explicit skip reason. The run permutation parameter defaults to 500.

S5 retains its gene-set permutation test. The supplied one-sample formula gates judgement at one-sided alpha 0.05 and 80% power, with enrichment configurable and default 3.0. Below the required arrest count it reports UNDERPOWERED with passed unset; the report includes baseline, alternative, enrichment, alpha, power, required_cells and judged_cells. Paper inputs p0=0.0010413 and enrichment=2.7597 require 2,867 cells; the 3.0 default requires 2,307, versus 112 available. These are formula checks, not a new real-data acceptance run.

A checkpoint identity hashes input features and classes, deterministic normalized probe probabilities and saved labelled rows. The card body is excluded and the identity persists across saves. Every save atomically replaces the card before publishing the model and final checksums, and logs completion.

Full-plate jobs declare at least 20 GB RAM. The entry-point docstring records the measured 17.6 GB peak and 2.2–2.6 hours at four threads for 5.53 million cells; a single CP186A well took 22–37 minutes and 2.8 GB. Home updates the external job template. Item 995 is unchanged.

## Public API sources

5 revised public bodies, 55 canonical prose blocks, no new public symbols. The historical 20-symbol arrival, catalogs and global pins are unchanged. Protected adapters compose queue, inherited module, then follow-up; the composition regression checks the separate changed-symbol sets.

### `spacr.ops_cell_cycle.fine_tune`

`spacr/ops_cell_cycle.py:477`; SHA-256 `c9ba07bba5dcc5e9195cb268f515d96c1d5cc465d78865bc3ef26d31be28cf9e`.

```python
base: Mapping[str, Any], cells: pd.DataFrame, labels: Sequence[str], *, seed: int=0, tabular: Optional[str]=None, pooled: Optional[pd.DataFrame]=None, reference_cells: Optional[pd.DataFrame]=None
```

The base model refitted on its own labels plus the screen's, the screen's weighted ×5.

:param base: from :func:`load_cell_cycle_checkpoint` (its ``training`` rows are used), or a
    bundle with ``features``, ``card`` and no ``training`` to fit on the screen's labels
    alone.
:param cells: this plate's labelled cells.
:param labels: their stages.
:param seed: the learner seed.
:param tabular: the learner; the base card's when None.
:param pooled: other plates' labelled rows (the model's features, already robust z, and
    ``stage``) for a fine-tune of the whole screen; None refits on this plate's alone.
:param reference_cells: full wells including unlabelled cells, normalized by the same
    helper as prediction. Omission assumes cells is the full population and warns.
:returns: a bundle for :func:`save_cell_cycle_checkpoint`, its card recording the base;
    ``weights`` are the sample weights, the screen rows (other plates first, this plate
    last) after the base rows.

### `spacr.ops_cell_cycle.run_cell_cycle_phase`

`spacr/ops_cell_cycle.py:1253`; SHA-256 `c19a82331fc7a30619be43324922a3e75c42a0a9e47e7db776da6d3d0c20c623`.

```python
db: str, plate: str, wells: Sequence[str], settings: Mapping[str, Any], destination: str, *, permutations: int=PERMUTATIONS, arrest_enrichment: float=3.0
```

The ``ph_cell_cycle`` phase: stages, probabilities and flags of every cell of a plate.

A full-plate run requires at least 20 GB declared RAM in sched or a Home job
template: six wells and 5.53 million cells took 2.2-2.6 hours at four threads,
with peak RSS 17.6 GB; a 16 GB allocation failed. A single CP186A well took
22-37 minutes and 2.8 GB.

Reads the per-well phenotype cell tables, calls the model (fine-tuned on the user's
labels when given), takes G1 / S / G2 from :func:`_dna_phases`, runs
:func:`self_checks`, collapses failed stages over every plate of the screen in ``db`` and
writes ``ops_cell_cycle`` (one row per cell), ``ops_cell_cycle_checks`` (one row per plate
and check) and ``ops_cell_cycle_report_<plate>.html``. With ``ops_cell_cycle_label_queue``
above 0 it also draws the labelling queue (:func:`build_label_queue`) into
``cell_cycle_labels/<plate>/queue_pending.csv``.

:param db: the measurements database.
:param plate: the plate.
:param wells: its wells.
:param settings: read for every ``ops_cell_cycle_*`` setting, ``nucleus_channel``,
    ``cell_cycle_field_min_cells``, ``ops_design`` and ``ops_control_prefix``.
:param destination: the run folder holding the cell tables.
:param permutations: S2, S3 and S5 control draws, default 500.
:param arrest_enrichment: the S5 power alternative, default threefold.
:returns: the phase report; ``label_queue`` names the queue folder, its size and the
    labels file, or is None.
:raises ValueError: for a bad setting, missing cell tables or no model to start from.

### `spacr.ops_cell_cycle.save_cell_cycle_checkpoint`

`spacr/ops_cell_cycle.py:379`; SHA-256 `3fc53574214a93c0c6074a2a52423901227d9677985f77ae1e12b8686fedc7f8`.

```python
bundle: Mapping[str, Any], folder: str, *, base: Optional[str]=None
```

Save every model and card revision under one stable checkpoint identity.

The directory is ``cell_cycle_tabular_<input>_<identity12>``. Its model,
card and SHA256SUMS are staged completely before publication. Each file
is replaced atomically, with the card before the model and checksums last;
an interrupted publication is refused by the loader rather than using a
stale card with a new model.

:param bundle: fitted model, features, optional training rows and mutable card.
    The canonical checkpoint_id hashes input features, classes, fixed normalized
    probe probabilities and saved labelled rows. It is retained across saves;
    the rest of the card is excluded so updated cards replace the same path.
:param folder: directory containing checkpoints.
:param base: base model SHA-256 recorded for fine-tuning.
:returns: the stable checkpoint directory, also when model bytes change.

### `spacr.ops_cell_cycle.self_checks`

`spacr/ops_cell_cycle.py:737`; SHA-256 `a3c8e20a1ebb772379866233151703a9c6b794f92330a5cd7843a47f7e2f696e`.

```python
cells: pd.DataFrame, stages: pd.DataFrame, dna: pd.DataFrame, fits: Mapping[str, Any], *, card: Mapping[str, Any], seed: int=0, arrest_genes: Optional[Sequence[str]]=ARREST_GENES, control_prefix: str='nontargeting', permutations: int=PERMUTATIONS, labelled: Optional[Mapping[str, Any]]=None, arrest_enrichment: float=3.0
```

Checks S1-S6 of one plate.

:param cells: the plate's cells: ``well``, ``site``, ``nucleus_area``,
    OPS ``nucleus_cy``/``nucleus_cx`` (S2; legacy centroid pairs also accepted)
    and ``gene`` (S5) when present.
:param stages: from :func:`predict_stages`.
:param dna: the frame of :func:`_dna_phases`.
:param fits: its fits per group.
:param card: the model card; its ``reference`` S1 value and S4 φ, when present, tighten
    S1 and S4 as section 5 of the spec says.
:param seed: the permutations' seed.
:param arrest_genes: S5's knockouts with a known mitotic arrest; None skips S5.
:param control_prefix: the non-targeting gene prefix.
:param permutations: permutations of S2, S3 and S5. The S2 null now
    permutes daughter-eligible cells within each field, preserving their
    daughter count; the previous all-rows null is replaced.
:param labelled: ``{"truth", "predicted", "shuffled_macro"}`` for S3; None skips it.
:param arrest_enrichment: S5 alternative relative to observed NT, default threefold;
    one-sided alpha 0.05 and power 0.80 determine its minimum arrest count.
:returns: one row per check: ``check``, ``stages`` (judged), ``value``, ``control``,
    ``threshold``, ``passed`` (True, False or None when skipped) and ``note``; the frame's
    ``attrs["samples"]`` keeps S2's observed share and its permutation shares for the
    report. S5 also reports baseline, alternative, enrichment, alpha, power,
    required_cells, judged_cells and status (UNDERPOWERED, PASS, FAIL or SKIPPED).

### `spacr.ops_cell_cycle.train_cell_cycle_model`

`spacr/ops_cell_cycle.py:259`; SHA-256 `4043b92d89a4f32c7f7a16a4cabf57583df275599d40cf6d742669c5a365ec81`.

```python
cells: pd.DataFrame, labels: Sequence[str], *, input_mode: str='DNA', dna: str='dapi', mt: Optional[str]=None, tabular: str='hgb', seed: int=0, groups: Optional[Sequence[Any]]=None, repeats: int=5, permutations: int=PERMUTATIONS, data_files: Sequence[str]=(), reference_cells: Optional[pd.DataFrame]=None
```

A base cell-cycle model and its card, from labelled cells (the one call that trains it).

:param cells: the labelled cells: ``well``, the field column of ``groups``, the feature
    columns of the cell tables and, for the plate-held-out score of two or more plates,
    ``plate``.
:param labels: one stage of :data:`CLASSES` per cell.
:param input_mode: ``'DNA'`` or ``'DNA+MT'``.
:param dna: the DNA channel name in the feature columns.
:param mt: the microtubule channel name; required for ``'DNA+MT'``.
:param tabular: ``'hgb'`` or ``'xgboost'``.
:param seed: the seed of the folds, the shuffles and the learner.
:param groups: the cross-validation group of every cell (fields); the ``well`` and
    ``site`` pair when None.
:param repeats: repeats of the field-grouped 5-fold cross-validation.
:param permutations: shuffled-label models of the control.
:param data_files: the training data files, recorded with their SHA-256 in the card.
:param reference_cells: full wells including unlabelled cells. Required for enriched
    label samples; omission treats cells as the full population with a warning.
:returns: ``{"model", "features", "card", "training"}``; ``training`` keeps the labelled
    rows (features and labels) that fine-tuning refits on. The card's ``cv`` holds the
    field-grouped repeats, per-class F1 with field-bootstrap intervals, the shuffled-label
    control and, with two or more plates, ``plate_held_out`` (each plate predicted by a
    model trained on the others).
:raises ValueError: for an unknown mode or stage, or 'DNA+MT' without ``mt``.

## Runtime sources

Seven added identities and two retired skip messages. Preserve format fields and machine identifiers.

added: `spacr/ops_cell_cycle.py:427`

Cell-cycle checkpoint saved with current card: {path}

added: `spacr/ops_cell_cycle.py:723`

S5 needs a baseline in [0, 1], enrichment above 1, alpha in (0, 0.5) and power in (0.5, 1)

added: `spacr/ops_cell_cycle.py:861`

UNDERPOWERED: S5 has {judged} arrest cells; needs {needed} for {enrichment}x enrichment at alpha {alpha}, power {power}

added: `spacr/ops_cell_cycle.py:113`

reference_cells has no full population for plate {plate}

added: `spacr/ops_cell_cycle.py:811`

skipped: S2 needs nucleus_area and a centroid pair: nucleus_cy/nucleus_cx, nucleus_centroid_y/nucleus_centroid_x or centroid_row/centroid_col

added: `spacr/ops_cell_cycle.py:810`

skipped: fewer than two daughter-stage calls

added: `spacr/ops_cell_cycle.py:894`

skipped: no non-targeting cells for the S5 baseline

removed: `spacr/ops_cell_cycle.py:674`

skipped: fewer than 100 cells of the arrest set

removed: `spacr/ops_cell_cycle.py:611`

skipped: needs two daughter-stage calls and nucleus centroids

reused: `spacr/ops_cell_cycle.py:136`

reference_cells was not supplied; training rows are used as the full normalization population. For enriched label sets, pass all cells of the reference wells in reference_cells

## Other documentation and validation

The JSON inventories 39 private and test docstrings separately; these create no public API arrivals. It also records native receipts and exact final source hashes.

The labelled-only negative control fails the new normalization regression as required. The bounded synthetic S2 benchmark uses 20,000 cells, 50 fields and 25 identical conditional permutations: every statistic matches the original geometric statistic, with 3.91× speedup including setup. This does not claim full-plate performance.

Final item file: 15 passed hidden (1.89 s), 15 visible (1.98 s). Worker/streaming files: 48 passed hidden (28.30 s), 48 visible (26.80 s). Metadata/source owners: 162 passed (158.95 s). Three actual API source gates pass with zero production imports; all five metadata artifacts reproduce byte-identically. Six source rules: 156 passed and one inherited comment failure (222.18 s), from two comments in byte-identical base file spacr/_mask_batches.py; the pilot owner confirmed this inherited failure at 02:42, fixed it on pilot 2106ce3a4a, and instructed this branch to remain unchanged.

Broad affected files ran with CUDA hidden and with visibility unset plus 2 GB VRAM reserved. Their initial sole failures were in the new source-signature guard; the portable guard and corrected negative fixture are checked in the final item-file reruns. WORK_STATUS and the commit body retain raw and corrected counts.
