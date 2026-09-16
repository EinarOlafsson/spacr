# Notes from `spacr/model_zoo.py`

Prose lifted out of `spacr/model_zoo.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (7 entries)
- [ModelEntry.__post_init__](#modelentry__post_init__) (1 entry)
- [_human_bytes](#_human_bytes) (1 entry)
- [classify_kind](#classify_kind) (1 entry)
- [shared_catalogue](#shared_catalogue) (3 entries)
- [catalogue](#catalogue) (2 entries)
- [resolve](#resolve) (1 entry)
- [fetch](#fetch) (1 entry)
- [FieldBenchmark](#fieldbenchmark) (1 entry)
- [benchmark](#benchmark) (2 entries)
- [group_by_fieldset](#group_by_fieldset) (1 entry)
- [_rank_value](#_rank_value) (1 entry)

## Module level

### lines 310-314

```python
{
```

THE THREE BELOW PUBLISH REAL CHECKSUMS, and live in MODEL repos rather than the dataset repo above -- hence `repo_type`. Being verifiable is the difference between an entry `fetch` installs and one it refuses, so a new entry without a sha256 should be treated as unfinished rather than as following the precedent set by the first entry.

### lines 325-331

```python
"architecture": "Cellpose-SAM (cpsam_v2)",
```

THE README TABLE READS THESE THREE, and nothing else. `trained_on` and `notes` stay full length because the Model Zoo screen and instruction 370's scorecard are where the detail belongs; the README table was carrying all of it and became unreadable. Asked 2026-09-02: "just state the model name and architecture, training dataset (staining + number of images from n datasets), and performance on hold out data compared to stock".

### lines 360-363

```python
"dataset": "crystal violet plaque wells; 184 wells from 3 datasets, "
```

FROM THE TRAINING RECORD, `models/cpsam_seg_r3/model.db` in the plaque_assay_model project: 184 rows in `training_set`, by source nas_patrick 68 + nas_bigbean 27 (95 in-house) and lit_pmc_staged 67 + lit_curate_single 22 (89 literature, counted as one dataset).

### lines 366-373

```python
"versus_stock": "F1 0.856 in-domain; 0.806 on literature "
```

THE LITERATURE FIGURE IS 0.806, NOT 0.834, and this row published the wrong one until 2026-09-02. The project corrected it on 2026-08-09 and its own model.db names the old value `literature_generalisation_SINGLESPLIT_optimistic`: 0.834 came from ONE 19-well split and turned out to be the best of three folds. The cross-validated mean is 0.806 with an SD of 0.020 (per fold 0.795 / 0.789 / 0.834). The in-domain 0.856 is confirmed -- an independent harness reproduced 0.855.

### lines 384-390

```python
"round 3 trades precision (0.939 down to 0.858) for recall "
```

"down to" / "up to" rather than "->" BECAUSE THIS PROSE IS

PUBLISHED. It is printed into the README's model zoo table and from there into all nine localized READMEs, and test_localized_readme_inline_markup_is_balanced_and_tight forbids ">" in those files -- it is looking for HTML that has leaked through a translation model, and an ASCII arrow reads as exactly that. It also translates better as words.

### lines 407-412

```python
"dataset": "whole-plate and multi-well crystal violet images; 562 "
```

FROM `data/detector_v3` and the v3 training record: 441 training images (289 wells + 152 background) and 121 validation (83 + 38). The background half is not padding -- v2 was trained on positives only and fired on histology, chest X-rays, logos and Venn diagrams, so the negatives are the reason v3 is the published model.

### lines 415-419

```python
"versus_stock": "mAP50 0.993, mAP50-95 0.886, precision and recall "
```

No stock model detects wells, so this is the hold-out score and not a comparison. mAP50-95 is 0.886 from the final training epoch in `runs/well_detector_v3/results.csv`; a separate val run in model.db reports 0.892, and the two are the same measurement taken twice rather than a disagreement worth publishing.

## ModelEntry.__post_init__

### lines 566-567

```python
"""Fill in the provenance fields and validate the kind.
```

A blank provenance field reads as "no constraints"; it has to say "unknown" out loud instead. object.__setattr__ because frozen.

## _human_bytes

### lines 728-729  _(unsure)_

```python
if unit == "GB":
```

GB is the display ceiling.  Let its last pass finish naturally so values larger than a terabyte reach the reachable fallback below.

## classify_kind

### lines 831-833

```python
near = {p.parent.name.lower(), p.parent.parent.name.lower()}
```

Only the two nearest folders, not every ancestor: a classifier under ``/data/models/screen1/run/`` is still a classifier, and matching any ancestor called "models" would silently relabel a whole tree.

## shared_catalogue

### lines 1494-1497

```python
if (not force and float(_SHARED_CATALOGUE_CACHE["fetched_at"]) > 0
```

THE STAMP, NOT THE CONTENTS, decides freshness. Keying on `entries` meant an empty answer -- which is what an unreachable or unpublished catalogue gives -- was never cached, so the failure was retried by every caller forever.

### lines 1503-1506

```python
wait = (not _on_the_qt_gui_thread()) if block is None else bool(block)
```

The default is decided here rather than in the signature, because what it should be depends on WHERE the call is: waiting is right in a CLI and in a worker, and is the defect this parameter exists for on the GUI thread.

### lines 1518-1527

```python
_SHARED_CATALOGUE_CACHE["fetched_at"] = now
```

STAMP THE FAILURE, or the cache never suppresses anything. The freshness check below reads `entries`, which stays empty when the fetch fails -- so every caller re-fetched, and with the catalogue not yet published that is one 404 per settings panel built. It was reported as four identical lines in thirty seconds.

DEBUG after the first, too. An unpublished or unreachable catalogue is the expected state for anyone who has not contributed a model, and telling them about it repeatedly at INFO makes spaCR look broken for a feature they are not using.

## catalogue

### lines 1727-1731

```python
for entry in shared_catalogue(block=block):
```

The community catalogue, LAST, so a key declared here or by a local file always wins over the shared one. That ordering is the safety property: a shared catalogue is edited by people other than the user running the code, and it must not be able to redefine a model spaCR ships or one the lab pinned in its own file. It can only ADD.

### lines 1736-1738

```python
entries = [e for e in entries if e.name not in RETIRED_MODEL_NAMES]
```

Retired models are dropped LAST, after every source has contributed, so a retirement holds however the entry arrived -- bundled, local discovery of the shipped file, a lab catalogue, or a plugin.

## resolve

### lines 1797-1798

```python
raise ModelUnreadable(
```

It was written as a path, so "no entry called that" would be the wrong complaint: the user pointed at a file and the file is not there.

## fetch

### lines 2023-2024

```python
try:
```

Every failure path leaves the destination untouched: no partial file at a real model name, ever.

## FieldBenchmark

### lines 2131-2133  _(unsure)_

```python
@dataclass
```

benchmarking — "test on 3 fields"

## benchmark

### line 2384  _(unsure)_

```python
if entry.path:
```

Fail on the file, not on a state-dict key three frames inside torch.

### lines 2414-2415  _(unsure)_

```python
scores = mc._score(masks, names, object_type) if qc else [None] * len(fields)
```

mc._score is spacr.seg_qc.score_masks over the whole set at once, which is what gives the plate-relative flags something to compare against.

## group_by_fieldset

### lines 2499-2501  _(unsure)_

```python
def group_by_fieldset(results: Sequence[BenchmarkResult]
```

ranking — the part that refuses

## _rank_value

### lines 2531-2532

```python
return (-(score if score == score else -1.0), result.seconds,
```

nan sorts last rather than first: a model nobody scored is not the best model.
