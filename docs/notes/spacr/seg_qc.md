# Notes from `spacr/seg_qc.py`

Prose lifted out of `spacr/seg_qc.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (4 entries)
- [score_field](#score_field) (4 entries)
- [_iter_masks](#_iter_masks) (1 entry)
- [score_masks](#score_masks) (1 entry)
- [summarize_qc](#summarize_qc) (1 entry)
- [write_scorecard](#write_scorecard) (1 entry)
- [run_segmentation_qc](#run_segmentation_qc) (1 entry)
- [FlagGuidance](#flagguidance) (1 entry)
- [_flag_findings](#_flag_findings) (2 entries)
- [_axis_step](#_axis_step) (1 entry)
- [qc_roots](#qc_roots) (1 entry)
- [read_scorecard](#read_scorecard) (2 entries)
- [_subhead](#_subhead) (1 entry)
- [_headline](#_headline) (1 entry)
- [read_digest](#read_digest) (1 entry)
- [_iter_masks, 2026-09-19](#_iter_masks-2026-09-19) (1 entry)

## Module level

### lines 242-245

```python
_FLAG_SEVERITY: Dict[str, str] = {
```

How bad each flag is. 'fail' means the field's measurements would be wrong, not merely noisy; 'warn' means look at it before you trust it. The severity classes are semantics and live here; the numbers that decide whether a flag fires are thresholds and live in QC_DEFAULTS.

### lines 1523-1525  _(unsure)_

```python
_WELL_RE = re.compile(r"^([A-Za-z]{1,2})(\d{1,3})$")
```

Where a field is: plate, well, row, column

### lines 1633-1635  _(unsure)_

```python
GRADIENT_RATIO = 2.0
```

Findings: what to tell the user, in the order they should hear it

### lines 1964-1978

```python
CARD_PREFIX = "segmentation_qc_"
```

Reading the verdict back off disk

`run_segmentation_qc` already scored these masks once, at mask time, and wrote `<plate>/qc/segmentation_qc_<object_type>.csv`. Everything below reads that file. Nothing below opens a mask -- opening a plate's worth of masks costs seconds to minutes, and a screen that pays that on every visit is a screen that gets switched off. `score_digest` is the one exception and it only runs when a user asks for it by name.

Freshness is the price of not recomputing, and it is paid explicitly: each card's mtime is compared against the newest file in the mask stack it describes, so a card written before the last re-mask is reported as OUT OF DATE rather than quietly believed.

## score_field

### lines 585-588

```python
edge = np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
```

border-touching objects

Border objects are counted, reported, and then excluded from every size statistic below: a truncated object's area understates its size, exactly as diameter._region_diameters argues.

### lines 613-615

```python
enough = interior_areas.size >= th["min_objects"]
```

robust size statistics

Only with enough interior objects to have a distribution. Below the floor a MAD is one object's opinion, so no size flag may be raised from it.

### lines 644-648

```python
confluent = metrics["foreground_fraction"] >= th["foreground_fraction"]
```

fusion, the way diameter.py argues it

Both halves required: dense enough for fusion to be the explanation, AND the mask under-counting what the pixels support. The distance transform is only run when the first half holds, so a healthy plate never pays for it.

### line 673  _(unsure)_

```python
if n_objects < th["min_objects"]:
```

too few objects to be a field

## _iter_masks

### lines 846-851

```python
if callable(mask):
```

A CALLABLE VALUE IS A THUNK, not a mask. That is what lets a caller whose masks are not one-file-per-field -- the v2 pipeline, whose mask is a channel of a merged stack -- be scored without materialising the plate: a mapping of already loaded arrays would hold all 1536 fields at once, which is exactly what the file path above goes out of its way to avoid.

## score_masks

### line 892, trailing  _(unsure)_

```python
except Exception as exc:
```

truncated or corrupt .npy

## summarize_qc

### lines 937-939

```python
scored = [q for q in field_qcs if FLAG_UNREADABLE not in q.flags]
```

A mask that could not be read contributes no count and no size: folding its zero into the plate medians would let one corrupt file drag the reference every other field is judged against.

## write_scorecard

### lines 1105-1107

```python
metric_names: List[str] = []
```

'n_objects' is already a column of its own; carrying the metric copy too would put the same header in the file twice, which csv.DictReader silently collapses.

## run_segmentation_qc

### lines 1212-1219

```python
if mode == "stop" and summary.get("verdict") == "fail":
```

THE GATE. Raised LAST, after the scorecard and the flags are on disk and the verdict has been printed, so stopping the run costs none of the evidence for why -- a gate that stops before it writes the card leaves the user with a failure and nothing to read.

Only on `fail`. A `warn` plate is one the thresholds are unsure about, and halting a plate on an unsure verdict trains people to turn the gate off, which is worse than not having it.

## FlagGuidance

### lines 1231-1243

```python
@dataclass(frozen=True)
```

Plain language: what each flag means, what causes it, what to do

The scorecard above is precise and unreadable to anyone who has not read this module. "3 plates failed QC" is nearly useless; what changes a decision is "plate2 rows E-H hold 4x the objects of rows A-D, which is usually uneven illumination or a threshold set too low". Everything below exists to get from the first sentence to the second.

One entry per member of FLAGS, checked by tests/test_seg_qc_banner.py: a flag added to the vocabulary without an explanation is a flag a user will read as a nine-letter identifier.

## _flag_findings

### line 1708, trailing  _(unsure)_

```python
if guidance is None:
```

a flag with no entry

### lines 1720-1726

```python
severity = _FLAG_SEVERITY.get(flag, "warn")
```

The severity is the flag's own, with one exception that has to be honoured: `_apply_plate_context` demotes empty and near-empty fields on a sparse plate, because with a plate median of 2 pathogens per field a field with none is the assay. `_severity_of` takes the worst flag, so a field carrying an undemoted 'fail' flag is itself 'fail'; if not one member field is, the flag was demoted, and calling it a failure here would contradict the verdict the card already printed.

## _axis_step

### lines 1787-1788

```python
return None
```

A half with a median of zero is an empty half, not a gradient; the empty-field flag is the honest report of that and already fired.

## qc_roots

### lines 2150-2151  _(unsure)_

```python
parent = os.path.dirname(os.path.normpath(root))
```

`<plate>/merged` and `<plate>/norm_channel_stack` both name a folder INSIDE the plate; the card lives beside them, not in them.

## read_scorecard

### lines 2219-2227

```python
if any("\x00" in str(name) for name in header):
```

THE NUL CHECK HAS TO LOOK AT THE HEADER, and this is why. CPython's csv module used to raise `_csv.Error: line contains NUL`, so a corrupted card was caught by the `except csv.Error` below and reported as unreadable. PYTHON 3.12 STOPPED RAISING: it parses the NUL straight through into a FIELD NAME. The check here only ever looked at `row.values()`, so a NUL in the header became a key it could not see, every real column went missing, every default applied, and a damaged scorecard came back "ok" on exactly the interpreters this project targets.

### lines 2264-2267

```python
if "nul" in str(exc).lower():
```

Python <=3.11 rejects a NUL while iterating the CSV; 3.12+ accepts it and the explicit checks above reject it. Keep one diagnosis on every supported interpreter so callers do not have to parse a version-specific stdlib message.

## _subhead

### lines 2348-2349  _(unsure)_

```python
return (
```

Worth saying out loud: every field passed on its own, and the problem is only visible when they are laid out on the plate.

## _headline

### lines 2373-2375

```python
return digest.findings[0].headline
```

The worst finding, not the counts: "12 of 96 fields failed" is a number, "plate2 rows E-H hold 4x the count of rows A-D" is a decision. `diagnose` has already sorted the worst one to the front.

## read_digest

### lines 2450-2452

```python
stale=bool(masks_mtime and mtime and masks_mtime > mtime + 1.0),
```

Only a mask that is genuinely newer counts. Equal mtimes are a coarse filesystem timestamp on a card written moments after the masks, which is the normal case and not a staleness.

## _iter_masks, 2026-09-19

A mask folder on a macOS external volume holds a `._<field>.npy` AppleDouble sidecar beside every mask, and `np.load(..., allow_pickle=False)` refuses it. Measured on the code before this fix, one good mask plus its sidecar scored as "FAIL - 1 of 2 fields failed (50%): fix the segmentation before running Measure": the sidecar counted as a field and as a failed one, so every such plate got a false FAIL. Dot-files are left out here with an inline `startswith(".")` rather than through `spacr.io._listdir_visible`, because this module is tested to import no torch and `spacr.io` imports it at module level. Reasons in `docs/notes/spacr/io.md` under `_listdir_visible`.
