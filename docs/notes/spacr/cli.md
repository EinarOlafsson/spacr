# Notes from `spacr/cli.py`

Prose lifted out of `spacr/cli.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (13 entries)
- [_absorb_registered_gui_only](#_absorb_registered_gui_only) (1 entry)
- [_load_settings_csv](#_load_settings_csv) (1 entry)
- [_allowed_types](#_allowed_types) (1 entry)
- [coerce_value](#coerce_value) (2 entries)
- [use_agg_if_headless](#use_agg_if_headless) (1 entry)
- [_quiet_progress_bars](#_quiet_progress_bars) (1 entry)
- [render_module_description](#render_module_description) (1 entry)
- [cmd_validate](#cmd_validate) (1 entry)

## Module level

### lines 154-171

```python
_MODULE_LIST: Tuple[Module, ...] = (
```

The mapping below is not invented: every entry is a callable that some existing dispatcher already runs. The sources, in order of authority:

spacr/qt/bridge.py :: resolve_pipeline_entry  — the PySide6 GUI spacr/gui_utils.py :: run_function_gui        — the Tk GUI spacr/validate.py  :: APP_FUNCTIONS           — the pre-flight registry

and the defaults helper for each is the one the pipeline itself calls to canonicalize its settings (grep "from .settings import" in the target module), not the one a GUI screen happens to show.

Every app in spacr.qt.app.APPS is either here or in INTERACTIVE_ONLY below, and tests/test_app_registry_parity.py fails when one is in neither. Three were in neither until that test was written: `invasion` and `replication` (both Toxo assays with a Qt button, a settings panel and a submodules entry point, but no `spacr-run`) and `foreign`, which even had a validate entry. An app that ships with a GUI button and no headless path is an app nobody can run on a cluster, and nothing said so.

### lines 567-572

```python
Module(
```

Hand-written, and it has to be: the seam that publishes an app's other strings cannot derive `requires`, `writes` or `note`, which are the three things `--describe` exists to print. There is no app row to take them from either -- the export folded onto the Measure masthead and its registration went with the tile -- so this entry is the whole of what `spacr-run anndata_export` knows about itself.

### lines 596-597  _(unsure)_

```python
ALIASES: Dict[str, str] = {
```

Friendly spellings. Seeded from spacr.validate.APP_ALIASES so a name that works there works here, plus the function names themselves.

### lines 636-640

```python
"cellpose_all": "cellpose_masks",
```

cellpose_all was "benchmark every Cellpose model on one folder". Cellpose 4 ships exactly one model, so the comparison had a single entrant and the run was `cellpose_masks` under another name -- which already defaults model_name to the same stock 'cpsam'. Kept as an alias, not deleted, so a script or settings CSV that still says cellpose_all keeps running.

### lines 680-682

```python
INTERACTIVE_ONLY: Dict[str, str] = {
```

Apps the GUI offers that have NO headless-runnable callable. Naming them in the error message is kinder than "unknown module": the user did not typo, the thing simply cannot run without a person looking at a screen.

### lines 684-702

```python
"outliers": "For headless use, call "
```

THE SIX THE FOLD TOOK OFF THE COMMAND LINE, added 2026-09-07.

None had a row in `_MODULE_LIST`. They answered to `spacr-run` because `_absorb_registered_gui_only` pulls `cli_note=` out of the Qt registry and that pull only happens in a process that has imported `spacr.qt.app`, which a headless `spacr-run` deliberately does not. `571b6e77c` and `00f166a7f` then folded them into host screens. So `spacr-run outliers` answers "unknown module 'outliers'", which tells the user they mistyped a name that was removed.

WRITTEN HERE, not absorbed -- the same decision as `curate`, `image_scatter` and `pca` above, and for the same reason. FOUR OF THEM STILL HAVE A CATALOG ROW, and the text below is that row's `cli_note` COPIED VERBATIM: `INTERACTIVE_ONLY.setdefault` means a hand-written entry outranks the registry, so a paraphrase here would silently replace the sentence the screen declares. That duplication is guarded `test_the_gui_only_sentence_reaches_spacr_run` fails the moment the two drift, which is how the first version of this block was caught.

### lines 717-718

```python
"import_images": "Image import folded into the foreign-format importer. "
```

The other two have NO catalog row left to copy from, so these are written rather than mirrored, and name where the thing went.

### lines 755-759

```python
"curate": "Curate is hand correction of a mask or a track table -- the "
```

WRITTEN HERE, not absorbed. Curate is folded into Make Masks and has no registry row left to carry a `cli_note=`, so the sentence `_absorb_registered_gui_only` used to pull out of the row has to live in this table -- otherwise `spacr-run curate` stops explaining itself and starts guessing that the user meant `convert`.

### lines 784-789

```python
"image_scatter": "Image Scatter is an interactive plot — the hover "
```

WRITTEN HERE, not absorbed. These three are folded -- Image Scatter and PCA onto Image UMAP, Volcano Explorer onto Regression -- and have no registry row left to carry a `cli_note=`, so the sentence `_absorb_registered_gui_only` used to pull out of the row has to live in this table. Without it `spacr-run pca` stops explaining itself and starts guessing the user meant something else.

### lines 809-813

```python
"hit_list": "Hit List is an interactive ranked table — filtering, "
```

THE OTHER THREE FOLDED ROWS. A folded module keeps its key everywhere a key is written, and `spacr-run <key>` is one of those places: a key the registry no longer holds and this table does not name answers "unknown module" and then guesses at a near spelling, which tells a user who typed the right name that they typed the wrong one.

### lines 948-951

```python
_CSV_COLUMNS: Tuple[Tuple[str, str], ...] = (
```

Column-name pairs a spaCR settings CSV can use. ('Key', 'Value') is what spacr.utils.save_settings writes next to every run and what the GUI's "Export settings" button produces; ('setting_key', 'setting_value') is the documented default of spacr.utils.load_settings.

### lines 1083-1085

```python
_TYPE_OVERRIDES: Dict[str, Tuple[type, ...]] = {
```

expected_types declares a few keys more narrowly than the code that reads them. Mirrors _EXPECTED_TYPE_OVERRIDES in spacr.validate — kept in step with it deliberately, so a value the validator accepts is a value --set can write.

### lines 1092-1097

```python
_APP_TYPE_OVERRIDES: Dict[str, Dict[str, Tuple[type, ...]]] = {
```

Per-module narrowings, for keys whose name two pipelines share. Mirrors _APP_TYPE_OVERRIDES in spacr.validate for the same reason as above, and tests/test_app_registry_parity.py asserts the two are equal so the mirror cannot rot: `masks` is declared bool in expected_types (the mask pipeline's save switch), but spacr.foreign.import_project takes it as their mask folder, so `--set masks=/their/masks` was rejected as "cannot be read as bool".

## _absorb_registered_gui_only

### lines 856-858

```python
pull = getattr(app, "registered_metadata", None) if app else None
```

`getattr(..., None)`: the Qt registry may be half-built when this runs, in which case nothing has registered yet and the push half of the seam delivers every row later.

## _load_settings_csv

### lines 1030-1032

```python
raw = ",".join([str(raw)] + [str(x) for x in overflow])
```

A hand-edited CSV with an unquoted list — `channels,[0, 1, 2]` — splits across columns. Rejoin rather than silently storing the fragment '[0'.

## _allowed_types

### lines 1126-1127

```python
out = tuple(type(None) if t is None else t for t in raw if isinstance(t, type) or t is None)
```

expected_types spells NoneType two ways: type(None) for most keys and a bare None for 'sample' / 'x_lim'. Normalize both.

## coerce_value

### lines 1229-1230

```python
try:
```

'4.0' for an int setting is a float that happens to be whole — accept it rather than making the user retype it.

### lines 1258-1259  _(unsure)_

```python
raise SettingsError(
```

Only reachable with a non-empty ``types``: when nothing is declared every branch above is allowed and the ``str`` one always returns.

## use_agg_if_headless

### line 1370, trailing  _(unsure)_

```python
except Exception:
```

matplotlib is optional for --list / --describe

## _quiet_progress_bars

### lines 1458-1460

```python
os.environ.setdefault("TQDM_DISABLE", "1")
```

tqdm reads TQDM_DISABLE; spaCR's own print_progress already emits whole lines, but the handful of `print(..., end='\r')` sites in io / utils / sim do not, so a redirected log gets one long line from those.

## render_module_description

### line 1529, trailing

```python
except Exception:
```

a broken settings helper must not break --describe

## cmd_validate

### lines 1662-1666

```python
if getattr(args, "hash_inputs", None) is not None:
```

hash-inputs / --no-hash-inputs, applied AFTER the settings file so the flag wins. `None` means neither was given, in which case whatever the settings file says stands -- a settings file written by the GUI already carries the user's preference, and a CLI run of that file should reproduce the GUI run rather than silently differ.
