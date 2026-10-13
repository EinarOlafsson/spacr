# Notes from `spacr/run_journal.py`

Prose lifted out of `spacr/run_journal.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [Run.record_warning](#runrecord_warning) (1 entry)
- [Run._inventory_root](#run_inventory_root) (1 entry)
- [Run._capture_final_provenance](#run_capture_final_provenance) (1 entry)
- [Run._write_manifest](#run_write_manifest) (1 entry)
- [Run._write_settings](#run_write_settings) (2 entries)
- [Run._snapshot_log_tail](#run_snapshot_log_tail) (1 entry)
- [open_run](#open_run) (6 entries)
- [recent_runs](#recent_runs) (2 entries)
- [recent_runs._sort_key](#recent_runs_sort_key) (1 entry)
- [search_runs](#search_runs) (2 entries)
- [journal_totals](#journal_totals) (4 entries)
- [_read_run_record](#_read_run_record) (1 entry)
- [format_run_diff](#format_run_diff) (1 entry)

## Module level

### lines 68-73

```python
from .macro import begin_recording, finish_recording
```

The macro recorder. Imported at the top rather than inside `open_run` because it costs nothing to: `spacr.macro` imports only the standard library, and reaches for spacr.ports / spacr.artifacts / spacr.settings lazily, inside the functions that need them. Both calls swallow every exception on purpose — the emitted script is a record of the run, never a condition of it.

### lines 1628-1638

```python
_NULLISH_STRINGS = frozenset({"", "none", "null"})
```

Provenance diff — "what actually changed between run A and run B?"

Why this is not a plain key-by-key dict diff: spaCR's settings schema moves between releases. Diffing a run recorded on 1.4.3.7 (204 keys) against one recorded on 1.4.8.7 (38 keys) turns up ~196 "differences", of which *zero* are decisions the user made — they are keys that simply did not exist on one side. The signal (a knob the user turned) drowns in schema drift. So the diff buckets keys by presence first and only calls a key "changed" when it exists in BOTH runs.

### lines 1640-1642

```python
_NULLISH_STRINGS = frozenset({"", "none", "null"})
```

Strings that stand in for "unset" once a value has round-tripped through CSV (``None`` is written as an empty cell) or through ``json.dumps(..., default=str)``.

## Run.record_warning

### lines 701-702

```python
if len(self.run_warnings) < 500:
```

Bound the manifest if a library repeats the same warning with field-specific text thousands of times.

## Run._inventory_root

### lines 854-856

```python
return path.parent
```

An output FILE: the run may well write siblings next to it a report beside its figures, a tar beside its manifest -- so the directory is the honest unit here.

## Run._capture_final_provenance

### lines 920-923

```python
signature = _inventory_signature(path)
```

The same narrowing as the baseline, and it MUST match it:

a file inventoried at the start and re-walked as a whole directory at the end would report every sibling as an output this run created.

## Run._write_manifest

### lines 983-988

```python
"input_hashing": "on" if self.hashing_enabled() else "skipped",
```

Stated, not implied. A manifest that simply LACKS hashes is indistinguishable from one whose hashes were computed and matched, and telling those apart is the whole value of the record. Top level rather than under `performance`, because "were the digests taken" is a claim about provenance and not a timing.

## Run._write_settings

### line 1022  _(unsure)_

```python
_atomic_write_text(
```

Machine-friendly JSON (source of truth)

### line 1027  _(unsure)_

```python
with open(self.dir / "settings.csv", "w", newline="") as f:
```

Human-friendly CSV (Key,Value — spacr.utils.load_settings compatible)

## Run._snapshot_log_tail

### lines 1044-1047

```python
LOG.warning("could not copy the last %d log lines into %s (%s)",
```

This is the run's own record of what it printed, and it is the first thing anyone opens when a run went wrong. A folder with no log.txt reads as "nothing was logged" rather than "the copy failed", so say which it was — but do not fail the run over it.

## open_run

### line 1124  _(unsure)_

```python
run._write_manifest()
```

A running manifest makes an interrupted process visible and auditable.

### lines 1127-1129

```python
macro = begin_recording(app_key, run.settings, run_dir=run.dir)
```

Macro recorder, half one: start watching for the run id the pipeline is about to mint. See spacr/macro.py — the script lands next to this manifest when the run closes.

### lines 1139-1147

```python
run.status = ("cancelled" if type(e).__name__ == "PipelineCancelled"
```

A RUN THE USER STOPPED IS NOT A RUN THAT FAILED, and instruction 140 C asks for a folder they can see afterwards. Recorded as "cancelled" so the Runs tab, `recent_runs` and a reviewer reading the manifest can all tell "I pressed Stop" from "this screen broke the model" -- which are different things to do next, and the folder is otherwise identical.

The traceback is still kept: where a long fit was interrupted is exactly what a user asks afterwards.

### lines 1167-1168

```python
LOG.exception("Could not finalize run manifest in %s", run.dir)
```

A manifest failure is never silent, but it also must not mask the original pipeline exception during context-manager unwinding.

### lines 1170-1175

```python
try:
```

Instruction 180: what was OPEN around the run, when anything was. Imported here and not at module scope so a pipeline that never touches the GUI does not import it at all, and inside its own try because a workspace bundle is a convenience -- a run that produced results must not be reported as failed because a panel could not describe itself.

### lines 1181-1182  _(unsure)_

```python
finish_recording(macro, status=run.status, settings=run.settings)
```

Macro recorder, half two: write the Python script that repeats this run — and, when it continues one, the whole chain before it.

## recent_runs

### lines 1244-1257

```python
candidates = [
```

Newest-first BY FOLDER NAME before opening anything. Run folders are named `{YYYY-MM-DD_HHMMSS}_{tag}__{app}`, so the name sorts to the second without touching the disk.

This used to read and parse EVERY manifest in the journal and then keep ten. On a real machine that was 3521 folders and ~0.85 s of json.loads on the GUI thread at startup, to populate a ten-row list and it grows with every run the user has ever done, so the launch gets slower the more the tool is used.

Only `limit` are needed, but the name truncates to the second while `start_utc` does not, so runs inside one second can reorder. Reading a margin past the limit and sorting those precisely keeps the documented ordering while bounding the work.

### lines 1272-1275

```python
LOG.warning("skipping run folder %s: its manifest.json could not "
```

Not fatal — one unreadable folder must not empty Run History — but not silent either. This dropped the run from the list with no trace anywhere, so a run the user can see on disk simply was not there in the app, and nothing said why.

## recent_runs._sort_key

### lines 1287-1288  _(unsure)_

```python
def _sort_key(e):
```

Sort by parsed timestamp (with folder-mtime as tiebreaker for any manifests missing / mangled start_utc).

## search_runs

### lines 1357-1358

```python
else:
```

Falsy values became ``[]`` above and took the list arm, so every remaining JSON scalar is truthy and represents one warning.

### lines 1363-1365

```python
if "warnings" not in manifest:
```

Legacy manifests did not structure warnings. Their bounded log tail is still useful, so surface warning-looking lines without failing a history scan on encoding or permissions.

## journal_totals

### lines 1537-1550

```python
names = _run_dir_names(root)
```

INCREMENTAL. This has to read every manifest -- the answer is an aggregate over all runs, so it cannot be bounded the way recent_runs can -- but a journal is append-only in practice, so it does not have to read them all TWICE.

The docstring below used to say "cheap enough to call on Home-screen construction: one iterdir + a file read per run folder". That was true at fifty runs. At 3521 it was 671 ms on the GUI thread at startup, and it got worse with every run the user ever did.

So the counted folder names are remembered alongside the totals, and only folders not already counted are parsed. A DELETED folder cannot be undone incrementally -- nothing records what it contributed -- so that case falls back to a full recount, which is correct and rare.

### lines 1570-1573

```python
LOG.warning("run folder %s is not counted: its manifest.json "
```

The Home dashboard's run count is this number. Skipping a folder in silence made it quietly short — "you have run 12 masks" when the answer is 13 and one manifest is damaged — and a total that is wrong by an unknown amount is worse than one that says so.

### lines 1581-1584

```python
hashes = m.get("model_hashes") or {}
```

Per-run model record. The manifest stores these under

``model_hashes`` as a {name: "filename:digest"} dict (see Run._write_manifest) — NOT a ``models`` list, which never matched and left models_recorded stuck at 0.

### line 1590

```python
for model in m.get("models", []) or []:
```

Back-compat: also honour a legacy ``models`` list of dicts.

## _read_run_record

### lines 1828-1829  _(unsure)_

```python
rec["errors"].append(f"settings.json unreadable ({e.__class__.__name__})")
```

settings.json exists but is unreadable — try the CSV twin before giving up; they are written together.

## format_run_diff

### line 2093

```python
only_a = diff.get("only_in_a") or []
```

schema drift, summarised (never enumerated)
