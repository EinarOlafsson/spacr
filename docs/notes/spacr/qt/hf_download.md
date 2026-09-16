# Notes from `spacr/qt/hf_download.py`

Prose lifted out of `spacr/qt/hf_download.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [_DownloadDialog](#_downloaddialog) (1 entry)
- [_DownloadDialog.__init__](#_downloaddialog__init__) (2 entries)
- [_DownloadDialog._on_cancel](#_downloaddialog_on_cancel) (1 entry)
- [_HFDownloadUI.on_progress](#_hfdownloaduion_progress) (1 entry)
- [_HFDownloadUI.on_finished](#_hfdownloaduion_finished) (3 entries)
- [_MeasureExampleWorker.run](#_measureexampleworkerrun) (1 entry)
- [_TarExampleWorker.run](#_tarexampleworkerrun) (3 entries)
- [_ChosenArchivesWorker.run](#_chosenarchivesworkerrun) (1 entry)
- [_MeasureTarWorker.after_extract](#_measuretarworkerafter_extract) (1 entry)
- [download_toxo_mito_demo](#download_toxo_mito_demo) (8 entries)

## Module level

### lines 30-40

```python
from ..example_archives import (                              # noqa: F401
```

THE DATA HALF OF THIS MODULE NOW LIVES IN `spacr.example_archives`, and is imported back here so nothing that already calls one of these names has to change -- including the tests that patch `hf_download._download_one` and `hf_download._list_files` to keep the demo flow off the network. Those still name real attributes of this module, and they are still what the workers below resolve.

It moved because `spacr-download` fetches the same datasets from a cluster login node, and importing this module to reach them would demand PySide6 on a machine with no display to give it. What is left here is Qt: the dialog, the threads, the signals. What left was only ever about the data.

## _DownloadDialog

### lines 155-157

```python
class _DownloadDialog(QDialog):
```

GUI-thread receiver

## _DownloadDialog.__init__

### line 205, trailing  _(unsure)_

```python
self._bar.setTextVisible(False)
```

the caption below says it all

### lines 212-213  _(unsure)_

```python
spacer = QWidget(self)
```

The spacer matches the button, so the caption is centred on the window and not on the gap beside the button.

## _DownloadDialog._on_cancel

### line 229  _(unsure)_

```python
def _on_cancel(self) -> None:
```

the QProgressDialog surface the download flow uses

## _HFDownloadUI.on_progress

### lines 367-368  _(unsure)_

```python
self._dlg.setLabelText(f"{percent}%  ({done}/{total})  {name}")
```

The name last: it is the part that can be long, so a window too narrow for all of it still shows the percentage.

## _HFDownloadUI.on_finished

### lines 381-384

```python
"""Tear the download down and hand the result to the caller.
```

Close the dialog *before* invoking the user callback — the callback may open its own modals (Continue/Stop prompts, etc.), and stacking one modal on top of another confuses Qt into the "app not responding" state on Linux.

### lines 413-414  _(unsure)_

```python
for attr in ("_hf_download_thread", "_hf_download_worker",
```

Drop retained refs on the owner so the QThread + dialog can be garbage-collected once the download flow ends.

### lines 423-426

```python
if ok:
```

Defer the user callback via a 0-ms singleShot so Qt processes any pending events (close event, deleteLater) before the chained pipeline modals appear. This is the specific fix for the "force-quit dialog after download" symptom.

## _MeasureExampleWorker.run

### lines 502-504

```python
target = root / name
```

Sub-paths are preserved: `merged/` is where Measure looks, and flattening the repo would put the arrays where nothing reads them.

## _TarExampleWorker.run

### lines 601-603

```python
part.unlink(missing_ok=True)
```

BETWEEN CHUNKS, so Cancel and application shutdown both take effect within a megabyte rather than after the whole set has arrived.

### lines 624-625  _(unsure)_

```python
target.unlink(missing_ok=True)
```

The archive is not kept: it is a second copy of everything that was just written, and these sets are hundreds of megabytes.

### line 629  _(unsure)_

```python
self.after_extract(self._dest)
```

Whatever this particular set needs doing to it after unpacking.

## _ChosenArchivesWorker.run

### line 687, trailing  _(unsure)_

```python
return
```

it emitted its own outcome

## _MeasureTarWorker.after_extract

### line 766  _(unsure)_

```python
expand_measure_arrays(Path(dest) / "merged")
```

No worker is constructed: see expand_measure_arrays.

## download_toxo_mito_demo

### lines 860-863

```python
dlg.setMinimumWidth(max(
```

WIDE ENOUGH FOR WHAT IT WILL SAY, on top of the wrapping the dialog already does. Widening alone never fixed this -- the longest caption is a FILE NAME and there is no longest file name -- but a window sized from "Preparing…" starts absurdly narrow and jumps on the first update.

### lines 870-872

```python
dlg.setAutoClose(True)
```

AutoClose True so hitting max value closes the dialog and returns control to the event loop — otherwise a stuck modal blocks the main thread and Qt shows the "Application not responding" prompt.

### lines 878-891

```python
worker = (worker_factory or _HFDownloadWorker)(dest)
```

WHICH worker, so a second dataset reuses this function's wiring rather than copying it. The thread affinity, the direct-connected cancel and the deliberate absence of a `deleteLater` below are all load-bearing and were each arrived at from a measured crash; a second copy of them would be a second place for one of them to be dropped. THE DEFAULT STAYS THE PER-FILE WORKER, and the Mask demo asks for the tar at its call site instead.

Switching the default here looked tidier and broke

`tests/qt/test_console_thread_safety.py`, which patches `_list_files` and drives this function to prove the offline failure path stays on the GUI thread. The tar worker does not call `_list_files`, so the patched test went to the network for real and aborted. A shared entry point's default is part of its contract with everything already calling it.

### lines 895-896

```python
ui = _HFDownloadUI(dlg, thread, worker, parent, on_done)
```

``ui`` is constructed here, on the GUI thread, so every connection below is a queued one — see _HFDownloadUI's docstring.

### lines 902-906

```python
dlg.canceled.connect(worker.cancel, Qt.DirectConnection)
```

DirectConnection is mandatory here: the worker's event loop is blocked for the whole of run(), so a queued cancel would not be delivered until after the download it was meant to abort had already finished. cancel() only flips a bool, which is safe to do from the GUI thread.

### lines 908-923

```python
try:
```

AND QUITTING THE APPLICATION CANCELS IT TOO.

Nothing did. A download still running when the window closed left a QThread to be destroyed with its thread alive -- "QThread: Destroyed while thread '' is still running", then abort -- because the finished handler that quits and waits for the thread only runs if the worker EMITS finished, and a worker that is still downloading never does.

DirectConnection for the same reason the cancel above uses it: the worker's event loop is blocked for the whole of run(), so a queued call would be delivered after the shutdown it was meant to survive. cancel() only flips a bool.

The wait is bounded and then given up on: a shutdown that hangs on a slow socket is a worse failure than the one being prevented, and the loop checks its flag between files.

### lines 947-958

```python
thread.start()
```

NOTE the absence of `thread.finished.connect(worker.deleteLater)`. `spacr.qt.bridge.make_thread` documents why, from a measured crash: the worker's affinity is the WORKER thread, so a deferred delete is posted into a loop that is stopping, and it races the GUI thread dropping the object's last Python reference in `on_finished`. Two owners, one object — gdb put it in `QThread -> sendPostedEvents -> ~QObject`. Chaining off `thread.finished` rather than `worker.finished` does not help; that exact variant was measured at 2 crashes in 20 runs. The worker is a Python-constructed PySide6 object, so Python already owns it: the last reference (held by `_HFDownloadUI`) frees it, on the thread that holds it.

### lines 960-961  _(unsure)_

```python
parent._hf_download_thread = thread
```

Retain references on the parent so the QThread + worker + dialog aren't garbage-collected while the download is in flight.
