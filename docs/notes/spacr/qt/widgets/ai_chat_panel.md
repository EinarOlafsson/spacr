# Notes from `spacr/qt/widgets/ai_chat_panel.py`

Prose lifted out of `spacr/qt/widgets/ai_chat_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_MessageBubble.__init__](#_messagebubble__init__) (1 entry)
- [_ProvidersDialog.__init__](#_providersdialog__init__) (1 entry)
- [_ProvidersDialog._build_providers_tab](#_providersdialog_build_providers_tab) (1 entry)
- [_ProvidersDialog._build_settings_tab](#_providersdialog_build_settings_tab) (6 entries)
- [_ProvidersDialog._make_provider_row](#_providersdialog_make_provider_row) (2 entries)
- [AIChatPanel.__init__](#aichatpanel__init__) (2 entries)
- [AIChatPanel._build_ui](#aichatpanel_build_ui) (1 entry)
- [AIChatPanel.refresh_provider_combo](#aichatpanelrefresh_provider_combo) (1 entry)
- [AIChatPanel._set_send_mode](#aichatpanel_set_send_mode) (1 entry)
- [AIChatPanel._start_stream](#aichatpanel_start_stream) (2 entries)
- [AIChatPanel._on_stream_finished](#aichatpanel_on_stream_finished) (3 entries)
- [AIChatPanel.clear_chat](#aichatpanelclear_chat) (1 entry)
- [AIChatPanel.open_error_flow](#aichatpanelopen_error_flow) (1 entry)

## _MessageBubble.__init__

### lines 79-80

```python
self._text.setProperty("i18nSkipText", True)
```

User and provider output must never be translated by a later whole-window language refresh.

## _ProvidersDialog.__init__

### lines 121-126

```python
self.setMinimumHeight(scaled_px(560))
```

`scaled_px`, NOT 560. The width beside it was already scaled and the height was not, so at a doubled font the dialog kept its device-pixel floor while every caption in it grew -- and the intro paragraph was squeezed to 81 px of the 180 it needs. Same defect class as the seven settings columns capped in device pixels; found by adding this dialog to the text-fit sweep.

## _ProvidersDialog._build_providers_tab

### lines 182-196

```python
holder = QScrollArea()
```

THE PAGE SCROLLS, BECAUSE A QTabWidget WILL NOT ASK IT HOW TALL IT IS. The intro paragraph wraps to one line more in German than in English -- 108 px against the 97 it was given -- and the dialog cannot discover that: `heightForWidth` does not propagate through a tab widget, so the wrapped label's real height never reaches the dialog's own sizeHint and the paragraph is squeezed.

A height-for-width size policy on the label was tried first and changed nothing, for exactly that reason.

Scrolling is 350's own rule for this -- "use a visible, accessible fallback rather than silently clipping" -- and it costs nothing where the content already fits: a scroll area with `setWidgetResizable(True)` shows no bar until it needs one, so English is unchanged.

## _ProvidersDialog._build_settings_tab

### lines 233-238

```python
self._speed_combo._spacr_setting_label = speed_label
```

THE HEADING IS THIS FIELD'S LABEL, and `install_api_tooltips` has no way to work that out: it finds labels through QFormLayout and QGrid, and this tab is a plain column of heading-then-control pairs. Without the pointer the helper concludes there is no label and deliberately installs no help at all -- which is why this was the one settings dialog in spaCR with no hover help on its combo or its editor.

### line 244

```python
auto_label = QLabel(
```

Auto-file GitHub issue on error

### lines 263-264

```python
self._route_errors_chk = Toggle(
```

Route errors through AI (on by default) — on a pipeline error the AI explains it first; the raw traceback stays hidden unless asked.

### lines 276-279

```python
self._console_aware_chk = Toggle(
```

Console aware — replaces the three-mode combo that used to sit beside the chat input. On by default: the chat gets switched on after something has already gone wrong, so the question it exists to answer needs the console that is already on screen.

### line 290  _(unsure)_

```python
col.addWidget(Divider())
```

GitHub sign-in — the official CLI owns credential storage

### line 327  _(unsure)_

```python
self._prompt_edit._spacr_setting_label = prompt_label
```

Its heading, for the same reason as the speed combo above.

## _ProvidersDialog._make_provider_row

### line 428  _(unsure)_

```python
header = QHBoxLayout()
```

Header line with status

### line 447  _(unsure)_

```python
install_row = QHBoxLayout()
```

Install command

## AIChatPanel.__init__

### lines 539-540

```python
self._thread: Optional[QThread] = None
```

Keep BOTH thread AND worker references — Qt's signal delivery relies on the worker still being reachable.

### lines 545-547

```python
self._retired: List = []
```

(QThread, StreamWorker) pairs whose stream finished but whose OS thread may still be winding down. Held so Python can't GC a still-running QThread — see _prune_retired().

## AIChatPanel._build_ui

### line 616  _(unsure)_

```python
input_row = QHBoxLayout()
```

Input area

## AIChatPanel.refresh_provider_combo

### lines 655-657

```python
self._btn_send.setEnabled(True)
```

Re-enable — the empty-state branch below disables the button, and a later refresh (the Providers dialog's "Refresh", after the user installed a CLI) has to undo that or Send stays dead.

## AIChatPanel._set_send_mode

### line 708  _(unsure)_

```python
self._btn_send.style().unpolish(self._btn_send)
```

Re-polish so QSS picks up the new objectName

## AIChatPanel._start_stream

### lines 758-761

```python
thread, worker = make_stream_thread(
```

`parent=self` is mandatory: it ties the QThread's C++ lifetime to the panel instead of to our Python refcount, so dropping self._thread in _on_stream_finished can't abort with "QThread: Destroyed while thread is still running".

### line 768  _(unsure)_

```python
self._thread = thread
```

Hold references so nothing is GC'd mid-flight.

## AIChatPanel._on_stream_finished

### lines 803-805

```python
"""Close off the answer, successfully or not.
```

Retire the (thread, worker) pair — keep BOTH Python refs until the OS thread has actually exited, otherwise Python can drop the last reference while QThread.isRunning() is still True.

### line 813  _(unsure)_

```python
self._thread = None
```

Reset streaming state so a fast follow-up send works.

### line 822  _(unsure)_

```python
self._pending_bubble.set_text(tr(
```

Provider returned no chunks — surface an obvious message

## AIChatPanel.clear_chat

### lines 904-907

```python
self._pending_bubble = None
```

Forget the in-flight assistant bubble BEFORE deleting the widgets. Keeping the reference would leave _on_chunk writing into a widget whose C++ half deleteLater() already destroyed, which raises "Internal C++ object already deleted" inside a Qt slot.

## AIChatPanel.open_error_flow

### lines 917-919  _(unsure)_

```python
def open_error_flow(self, traceback_text: str, active_app: str = "") -> None:
```

Public API used by AppScreen's Explain-error
