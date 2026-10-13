# Notes from `spacr/qt/widgets/model_zoo_picker.py`

Prose lifted out of `spacr/qt/widgets/model_zoo_picker.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_DownloadWorker](#_downloadworker) (3 entries)
- [_DownloadWorker.run](#_downloadworkerrun) (1 entry)
- [ModelZooPicker.__init__](#modelzoopicker__init__) (1 entry)
- [ModelZooPicker.refresh](#modelzoopickerrefresh) (1 entry)
- [ModelZooPicker._local_path](#modelzoopicker_local_path) (1 entry)
- [ModelZooPicker._selection_changed](#modelzoopicker_selection_changed) (1 entry)
- [ModelZooPicker._download_selected](#modelzoopicker_download_selected) (2 entries)
- [ModelZooPicker._on_progress](#modelzoopicker_on_progress) (1 entry)
- [ModelZooPicker._finish_download](#modelzoopicker_finish_download) (1 entry)
- [ModelZooPicker._on_download_failed](#modelzoopicker_on_download_failed) (1 entry)
- [ModelZooPicker._stop_any_download](#modelzoopicker_stop_any_download) (1 entry)

## _DownloadWorker

### line 95, trailing  _(unsure)_

```python
progressed = Signal(int, int)
```

bytes done, bytes total (0 = unknown)

### line 96, trailing  _(unsure)_

```python
finished = Signal(str)
```

the installed path

### line 97, trailing  _(unsure)_

```python
failed = Signal(str)
```

the message to show

## _DownloadWorker.run

### lines 117-119

```python
self.failed.emit(str(exc))
```

The message matters more than the type: a ChecksumMismatch here means the bytes that arrived are not the model, which is the one download outcome a user must never be allowed to miss.

## ModelZooPicker.__init__

### lines 247-250

```python
from ..screens.settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). The folder field carried its own tooltip, which is precisely the shape this post-pass exists to move.

## ModelZooPicker.refresh

### lines 318-320

```python
self.status.setText(f"Could not read the model list: {exc}")
```

A zoo that cannot be listed must not be a dialog that cannot be opened: the user may already have the model and only need to find it on disk -- and the stock row always works.

## ModelZooPicker._local_path

### lines 349-352

```python
return str(entry.path)
```

NOT A FILE, and deliberately not checked as one. Cellpose resolves "cpsam" by name; `_resolve_cellpose_pretrained` returns a stock name unchanged. Requiring a file here would grey out the one row that never needs downloading.

## ModelZooPicker._selection_changed

### lines 397-401

```python
self.status.setText(
```

SAID BEFORE THE CLICK, not after it. fetch refuses an entry it cannot verify, so without this the button is enabled, pressing it fails, and the message explains a policy the user had no way to see. They can still choose to accept it -- that is the dialog below -- but it is a choice, made knowingly.

## ModelZooPicker._download_selected

### line 438, trailing  _(unsure)_

```python
self.progress.setRange(0, 0)
```

until the size is known

### lines 461-464

```python
self._thread = QThread(self)
```

OFF THE GUI THREAD. These files are 1.2 GB; fetched from the button handler the event loop stops for minutes, the bar cannot move, and the compositor offers to force-quit spaCR -- instruction 315's subject, reached through a dialog instead of a screen build.

## ModelZooPicker._on_progress

### lines 499-500  _(unsure)_

```python
self.progress.setRange(0, 0)
```

No content-length: a bar with no end is honest, a percentage invented from an unknown total is not.

## ModelZooPicker._finish_download

### lines 515-519

```python
self.refresh()
```

REFRESH FIRST, THEN SAY WHAT HAPPENED. refresh() re-runs _selection_changed, which rewrites the status line from the selected entry -- so a message set before it is overwritten by the entry's own notes, and the failure the user most needs to see is the one that disappears.

## ModelZooPicker._on_download_failed

### lines 533-535

```python
"""Report a failed download in a dialog as well as on the status line.
```

NAMED, not swallowed. fetch refuses an entry whose checksum does not match, and that refusal is the single most important message this dialog can carry: it means the bytes are not the model.

## ModelZooPicker._stop_any_download

### lines 578-580

```python
thread.wait(10000)
```

A bounded wait: an unbounded one turns "close the dialog" into "hang until the download finishes", which is the same freeze this thread was introduced to remove.


---

# Notes from `spacr/qt/widgets/model_zoo_picker.py`

Prose lifted out of `spacr/qt/widgets/model_zoo_picker.py` by `tools/extract_source_notes.py`.
Ordinary comments move here; tool directives and published attribute documentation stay in the module. The path mirrors the source path, which is how its reasons are found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ModelZooPicker.__init__](#modelzoopicker__init__) (2 entries)
- [ModelZooPicker.refresh](#modelzoopickerrefresh) (2 entries)
- [ModelZooPicker._rebuild](#modelzoopicker_rebuild) (1 entry)
- [ModelZooPicker._install_backend](#modelzoopicker_install_backend) (1 entry)
- [ModelZooPicker._show_card](#modelzoopicker_show_card) (1 entry)

## ModelZooPicker.__init__

### lines 1131-1133

```python
self.table.itemClicked.connect(self._row_clicked)
```

A CLICK offers the install, the same as the Make Masks Mode box. itemClicked fires only for a person, so restoring a selection in code never opens a modal.

### lines 1141-1143

```python
self.card = QTextBrowser(self)
```

The scorecard sits between the list and the controls, at a fixed height: a box that grew and shrank with the selected model would move the Download button under the pointer between clicks.

## ModelZooPicker.refresh

### lines 1371-1374

```python
entries += list(model_zoo.catalogue(remote=True, block=False))
```

The catalogue lists the Cellpose stock models too. Duplicates are collapsed per version label by group_entries, which catches the stock row whose key and name disagree -- name "cpsam", key "cpsam_v2" -- where a name comparison here did not.

### lines 1382-1384

```python
entries = [e for e in entries
```

Installable backends survive the kind filter: they are listed so a user learns they exist, which is the whole point of showing a thing that is not installed.

## ModelZooPicker._rebuild

### lines 1404-1406

```python
self._rebuilding = True
```

Tear the old rows down FIRST. A combo box from the previous refresh is still wired to _version_picked, and setRowCount destroying it can emit currentIndexChanged against groups that no longer exist.

## ModelZooPicker._install_backend

### lines 1665-1666

```python
for group, (stem, pairs) in enumerate(self._groups):
```

Leave the row the user just installed selected, so "install it and use it" is one action rather than install-then-hunt-for-the-row.

## ModelZooPicker._show_card

### lines 1761-1762  _(unsure)_

```python
describe = getattr(entry, "describe", None)
```

The stock model is a SimpleNamespace, not a ModelEntry, so it has no describe(); fall back to what any entry-shaped object has.
