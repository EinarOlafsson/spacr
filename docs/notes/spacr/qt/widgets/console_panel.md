# Notes from `spacr/qt/widgets/console_panel.py`

Prose lifted out of `spacr/qt/widgets/console_panel.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (9 entries)
- [get_split_state](#get_split_state) (1 entry)
- [_CopyGlyphButton](#_copyglyphbutton) (1 entry)
- [_CopyGlyphButton.__init__](#_copyglyphbutton__init__) (1 entry)
- [_CopyGlyphButton.paintEvent](#_copyglyphbuttonpaintevent) (1 entry)
- [_TopicBar.__init__](#_topicbar__init__) (5 entries)
- [_TopicBar.mouseReleaseEvent](#_topicbarmousereleaseevent) (1 entry)
- [_TopicBar._copy_section](#_topicbar_copy_section) (1 entry)
- [_WorkingDots](#_workingdots) (1 entry)
- [_WorkingDots._render](#_workingdots_render) (1 entry)
- [_StdoutBlock.__init__](#_stdoutblock__init__) (1 entry)
- [_StdoutBlock.append](#_stdoutblockappend) (1 entry)
- [_StdoutBlock._trim_to_cap](#_stdoutblock_trim_to_cap) (1 entry)
- [_Bubble](#_bubble) (2 entries)
- [_Bubble._recalc](#_bubble_recalc) (1 entry)
- [_ChatInput.__init__](#_chatinput__init__) (2 entries)
- [_ChatInput.insertFromMimeData](#_chatinputinsertfrommimedata) (1 entry)
- [ConsolePanel](#consolepanel) (1 entry)
- [ConsolePanel.__init__](#consolepanel__init__) (7 entries)
- [ConsolePanel._build_ui](#consolepanel_build_ui) (18 entries)
- [ConsolePanel._restore_split](#consolepanel_restore_split) (1 entry)
- [ConsolePanel._scroll_to_bottom](#consolepanel_scroll_to_bottom) (1 entry)
- [ConsolePanel.begin_topic](#consolepanelbegin_topic) (2 entries)
- [ConsolePanel.append_stdout](#consolepanelappend_stdout) (1 entry)
- [ConsolePanel._append_notice_on_gui_thread](#consolepanel_append_notice_on_gui_thread) (1 entry)
- [ConsolePanel.append_error](#consolepanelappend_error) (1 entry)
- [ConsolePanel.as_text](#consolepanelas_text) (1 entry)
- [ConsolePanel.raise_section](#consolepanelraise_section) (2 entries)
- [ConsolePanel._scroll_widget_to_top](#consolepanel_scroll_widget_to_top) (1 entry)
- [ConsolePanel._is_raised](#consolepanel_is_raised) (2 entries)
- [ConsolePanel.clear](#consolepanelclear) (2 entries)
- [ConsolePanel.set_ai_active](#consolepanelset_ai_active) (1 entry)
- [ConsolePanel._on_submit](#consolepanel_on_submit) (2 entries)
- [ConsolePanel._send_to_ai](#consolepanel_send_to_ai) (4 entries)
- [ConsolePanel._console_context_for_question](#consolepanel_console_context_for_question) (2 entries)
- [ConsolePanel._start_stream](#consolepanel_start_stream) (1 entry)
- [ConsolePanel._prune_retired](#consolepanel_prune_retired) (1 entry)
- [ConsolePanel.shutdown](#consolepanelshutdown) (2 entries)
- [ConsolePanel._on_stage](#consolepanel_on_stage) (1 entry)
- [ConsolePanel._on_chunk](#consolepanel_on_chunk) (1 entry)
- [ConsolePanel._on_stream_finished](#consolepanel_on_stream_finished) (5 entries)
- [ConsolePanel.ai_explanation_of](#consolepanelai_explanation_of) (2 entries)
- [ConsolePanel.open_error_flow](#consolepanelopen_error_flow) (3 entries)

## Module level

### lines 70-85

```python
_TEXT_ROLES = {
```

Console text colours (per the output type) — we colour the *text*, not the background, so there are no coloured boxes.

Resolved through `active_palette()` on every call rather than captured at import time. They used to be three module-level constants read off `theme.PALETTE`, which is the frozen DARK palette, so the console painted the same dark chrome on every theme. Measured on light: `_StdoutBlock` filled itself `#161719` inside a `#fafafa` page (a black rectangle in a white one), and `_Bubble` inked `#ffffff` text on the `#dbe8fb` bubble the app stylesheet paints — 1.24:1. Now 15.59:1.

`COLOR_OUTPUT` / `COLOR_USER` / `COLOR_ERROR` are still importable — module __getattr__ below serves them live — because they read well at the call sites and existing callers spell them that way.

### line 89, trailing  _(unsure)_

```python
"COLOR_OUTPUT": "accent",
```

spaCR output  → blue

### line 90, trailing  _(unsure)_

```python
"COLOR_USER":   "success",
```

user input    → green

### line 91, trailing  _(unsure)_

```python
"COLOR_ERROR":  "error",
```

errors        → red

### line 133  _(unsure)_

```python
AI_COLOR_CLAUDE = "#DE7356"         # Anthropic terracotta / peach
```

spaCR AI text colour depends on the backing provider.

### line 134, trailing  _(unsure)_

```python
AI_COLOR_CLAUDE = "#DE7356"
```

Anthropic terracotta / peach

### line 135, trailing  _(unsure)_

```python
AI_COLOR_OPENAI = "#74AA9C"
```

OpenAI signature green

### line 136, trailing  _(unsure)_

```python
AI_COLOR_GEMINI = "#74AA9C"
```

Gemini — same green as requested

### lines 152-167

```python
DEFAULT_CHAT_HEIGHT = 120
```

Console / AI-chat split

The console box and the AI chat box sit in a vertical QSplitter so dragging the handle trades height between them — a bigger chat box is a smaller console and vice versa, the same gesture the live-preview card above already uses against the console.

The default deliberately reproduces what the panel looked like BEFORE the splitter existed: the chat input was capped at 120px and the console took everything else. A user who never touches the handle therefore sees no change at all. The console is the busier panel during a run, which is the state the app spends its long minutes in, so it keeps the lion's share by default; the user who mostly talks to the AI between runs drags once and the position is remembered.

## get_split_state

### lines 216-218

```python
if isinstance(raw, (bytes, bytearray, QByteArray)) and len(raw):
```

A settings backend can hand back a str (INI round-trip) or None. Only a real byte blob is restorable; anything else means "no preference", which leaves the caller on the default split rather than on a broken one.

## _CopyGlyphButton

### lines 241-243  _(unsure)_

```python
class _CopyGlyphButton(QAbstractButton):
```

Divider bar with a topic label

## _CopyGlyphButton.__init__

### lines 268-269  _(unsure)_

```python
self._flash = Flash(self)
```

The timing is shared with the figure queue's clear control; see spacr.qt.widgets.flash for why it lives in one place.

## _CopyGlyphButton.paintEvent

### lines 301-302  _(unsure)_

```python
painter.drawRoundedRect(off + 1, 1, side, side, 2, 2)
```

Back square first, then the front one over it — slightly offset, so the pair reads as one sheet on top of another.

## _TopicBar.__init__

### lines 333-337

```python
self.setCursor(Qt.PointingHandCursor)
```

The heading is a control now (instruction 110): click it to bring its section to the top and expand it. A pointing hand says so without a border, and StrongFocus keeps it reachable from the keyboard -- a control only a mouse can reach is one some users cannot reach at all.

### lines 344-345

```python
self._chevron = QLabel("▾")
```

A disclosure chevron, because a toggle with no indicator is a control users find by accident.

### lines 352-354

```python
self._label.setProperty("i18nSkipText", True)
```

Topic history is presentation generated in the language active when it was appended. Do not reinterpret composite module/function text during a later whole-window language switch.

### lines 363-366

```python
self._copy_btn = _CopyGlyphButton(self)
```

Copy this section — header and contents. Sits with the header it belongs to rather than in a toolbar, because "this bit" is a thing a reader points at, and hand-selecting a section out of a long console is exactly the chore this removes.

### lines 368-371

```python
copy_tip = "Copy this section, header and all"
```

Topic bars are built as output arrives, long after the panel's own translation pass, so this tooltip is translated here. The English source is kept on the widget so a later language switch can retranslate it from the original rather than from Swedish.

## _TopicBar.mouseReleaseEvent

### lines 406-409

```python
"""Raise this section on a click inside the bar.
```

The copy button and any trailing widget are children with their own handlers, so a click on them never reaches here -- which is what keeps "copy this section" from also moving the viewport. Release rather than press, so dragging off cancels.

## _TopicBar._copy_section

### line 437, trailing  _(unsure)_

```python
for _ in range(6):
```

bounded walk to the panel

## _WorkingDots

### lines 458-460  _(unsure)_

```python
class _WorkingDots(QLabel):
```

Animated "working" indicator — three dots cycling in AI colour

## _WorkingDots._render

### line 508, trailing  _(unsure)_

```python
pad = " " * (2 - self._n)
```

keep three glyph-slots wide

## _StdoutBlock.__init__

### lines 611-612  _(unsure)_

```python
if text_color is None:
```

Colour the TEXT (not a coloured box): each output type gets its own foreground colour while the block background stays neutral.

## _StdoutBlock.append

### lines 702-705

```python
fmt_cursor = QTextCursor(doc.findBlockByNumber(first_touched))
```

Format only the paragraphs this insert created or extended. Qt does copy the previous block's format into a new one, but not on every insertion path, so it is set rather than assumed — over the new range only.

## _StdoutBlock._trim_to_cap

### line 722, trailing  _(unsure)_

```python
removed = block.length()
```

paragraph text + its separator

## _Bubble

### line 886, trailing  _(unsure)_

```python
_H_PAD = 24
```

inner horizontal padding

### line 887, trailing  _(unsure)_

```python
_V_PAD = 12
```

inner vertical padding

## _Bubble._recalc

### line 940, trailing  _(unsure)_

```python
return
```

setFixedHeight below triggers a resizeEvent → guard

## _ChatInput.__init__

### lines 983-985

```python
self.setMinimumHeight(CHAT_MIN_HEIGHT)
```

The floor stays: one line of text plus the field's own padding. It doubles as the stop the splitter honours when the user drags the divider down onto the chat box.

### lines 987-992

```python
self.setAcceptRichText(False)
```

No ceiling. `setMaximumHeight(120)` used to live here, and it is exactly what made the chat box unresizable — the row was pinned at 120px no matter what the layout offered it. The 120 is now the splitter's DEFAULT size (:data:`DEFAULT_CHAT_HEIGHT`), so an untouched panel looks the same as it always did while the handle can still drag the box taller.

## _ChatInput.insertFromMimeData

### line 1024, trailing

```python
return
```

ignore dropped files; never read them in here

## ConsolePanel

### lines 1048-1049  _(unsure)_

```python
ai_stream_finished = Signal()
```

Fires when an AI stream ends (ok or error) so the AppScreen actions row can flip a Cancel button back to something else.

## ConsolePanel.__init__

### lines 1074-1077

```python
self.setAttribute(Qt.WA_StyledBackground, True)
```

QWidget (unlike QFrame) doesn't paint a QSS background/border/radius unless told to — without this the ConsolePanel's rounded surface box never draws and the console area shows straight through to the black app background. WA_StyledBackground makes the rounded box appear.

### lines 1080-1081  _(unsure)_

```python
self._run_module: str = ""
```

Module + function the current pipeline output is coming from, shown in the "spaCR output — <module> — <function>" banner.

### line 1084, trailing

```python
self._last_entry_kind: str = ""
```

"stdout" | "ai" | "user" | ""

### lines 1099-1101

```python
self._console_sent_lengths: Dict[int, int] = {}
```

Per rendered pipeline block, how much has already accompanied an AI turn. The blocks themselves remain the only history store: context is read from their QTextDocuments at ask time.

### lines 1103-1107

```python
self._retired: List = []
```

Retired stream (thread, worker) pairs — we hold these until thread.finished actually emits so Python doesn't GC the QThread while its OS thread is still winding down (which is what causes `QThread: Destroyed while thread '' is still running / Aborted` on the second consecutive AI request).

### lines 1110-1111  _(unsure)_

```python
self._relay_stdout.connect(self.append_stdout)
```

Wired before anything can append: a log record can arrive the instant this panel is registered as the console target.

### lines 1117-1119

```python
try:
```

Pipe records from the global logger into this console. Every

ConsolePanel subscribes to the same shared signal handler, so log records fanned out across screens all see them.

## ConsolePanel._build_ui

### lines 1137-1139

```python
outer.setContentsMargins(0, 0, 0, 0)
```

The panel itself is transparent (see theme QSS) — the rounded box is the ConsoleBox frame below, so the AI chat input can sit UNDER it as a separate, edge-aligned row rather than inside the box.

### lines 1143-1147

```python
self._split = QSplitter(Qt.Vertical)
```

The console box and the chat row are the two halves of a vertical splitter, so the handle between them trades height: drag it up and the chat box grows while the console shrinks by the same amount. This is the same gesture — and deliberately the same idiom — as the live-preview / console splitter one level up in AppScreen.

### lines 1151-1167

```python
self._split.setHandleWidth(SPACING["sm"])
```

A hairline, matching every other splitter in the app — the theme styles `QSplitter::handle` at 1px and this was the one place that overrode it, so the divider under the console read as a thick bar while the identical gesture beside the settings column read as a line.

This handle used to be 8px because it was standing in for the `outer.setSpacing(SPACING.sm)` gap the layout had before the splitter existed. That made the divider its own spacing, which is what made it heavy. The gap is now the handle's real width, so the console and chat sit closer together; the grab area is Qt's, not the painted width, so it stays draggable. 8px of GAP, not 8px of divider. The handle is the spacing the layout had before the splitter existed, and that spacing is worth keeping — what was wrong was that the whole 8px lit up accent-blue on hover, so resizing showed a thick blue slab. The stylesheet below keeps the width and paints only a 1px line in it.

### lines 1169-1173

```python
try:
```

The splitter is scaffolding, not a surface. AppScreen's

`_clear_page_surfaces` sweep tags every QSplitter by type, but a ConsolePanel used on its own (or in a test) never gets that sweep, and an untagged QWidget takes the blanket window fill — an opaque slab straight across the panel, immune to the page-opacity setting.

### lines 1179-1183

```python
try:
```

The theme fills a hovered handle with the accent colour, which on an 8px handle is an 8px blue bar. Keep the 8px of space and draw the highlight as a 1px line inside it: transparent background, one accent border along the top edge. The grab area is unchanged, so the handle is no harder to hit than it ever was.

### lines 1202-1203  _(unsure)_

```python
self._console_box = QFrame()
```

Console box — a rounded surface frame that wraps ONLY the scrolling output. QFrame paints its QSS background/border/radius natively.

### line 1212  _(unsure)_

```python
self._scroll = QScrollArea()
```

Scroll area of entries

### lines 1217-1218  _(unsure)_

```python
self._scroll.viewport().setStyleSheet("background: transparent;")
```

The viewport paints its own background — make it transparent too so the box's rounded surface shows through at the corners.

### lines 1221-1222

```python
self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
```

Never show a horizontal scrollbar — content that doesn't fit must wrap. This is what prevents the runaway-width crash.

### line 1229  _(unsure)_

```python
self._entries.setSpacing(SPACING["xs"])
```

A little breathing room between console entries.

### lines 1242-1249

```python
from PySide6.QtGui import QKeySequence, QShortcut
```

A VISIBLE AFFORDANCE FOR THE PEOPLE WHO DO NOT KNOW THE SHORTCUT (instruction 232). A long run writes thousands of lines and the one that matters is the last; getting to it must not be a scroll through everything above it.

SHOWN ONLY WHEN IT WOULD DO SOMETHING. A button that is always there and usually inert is furniture, and the user stops seeing it -- which is the state it is most needed in.

### lines 1261-1263

```python
self._end_shortcut = QShortcut(QKeySequence("Ctrl+End"), self)
```

HELD, like every other shortcut and filter in this codebase: Qt keeps a bare pointer, and one only the constructor referenced stops working as soon as the call returns.

### lines 1268-1271

```python
input_row = QWidget()
```

AI chat input — a separate row UNDER the console box (not inside it), borderless wrapper so only the text field's own box shows, edges flush with the console + system boxes. AI on/off toggle + provider selector live in the AppScreen actions row.

### lines 1286-1294

```python
self._split.addWidget(input_row)
```

NO console-context control here. This row is the input and nothing else. It used to carry a three-mode combo (Auto / Include / Off) and a permanent status label; "should the AI see my console" has the same answer for every question a user ever asks, which makes it a preference rather than a mode selector belonging on the screen where the questions are typed. It is now one yes/no in the AI settings ai.settings.get_console_aware, default on. What was actually attached is reported on the message it went with, which is where it is legible, rather than as furniture that is stale between asks.

### lines 1297-1303

```python
self._split.setStretchFactor(0, 1)
```

Only the console stretches when the WINDOW resizes. Giving both halves a stretch factor (as the live-preview splitter does, where both halves are content) would grow the chat box every time the window got taller — a visible change for a user who never touched the handle, which is precisely what this must not do. Stretch 0 on the chat box reproduces today's behaviour exactly: the chat box keeps whatever height it has, the console absorbs the rest.

### lines 1308-1310

```python
self._split.splitterMoved.connect(self._on_split_moved)
```

Written on release AND during the drag; QSettings coalesces in memory, and saving as it moves means a split survives even if the app is killed rather than closed.

### lines 1313-1320

```python
self._font_pt = self._zoomed_font_pt()
```

Console font-size control — its own right-aligned row below the input so the text box stays full width, flush with the console + system boxes. Adjusts every stdout/AI entry live. No per-module font-size spinner any more. The console and the AI chat now follow the global Zoom preference like every other piece of text in the app: a second, module-local control for the same thing meant the console could disagree with the interface around it, and meant setting it once did not carry to the next module.

### lines 1323-1325

```python
self._ai_active: bool = False
```

AppScreen creates + owns the AI toggle/provider menu and calls our setters when they change. Panel-internal state stays here so we always know what to do on Enter.

## ConsolePanel._restore_split

### lines 1355-1358

```python
self._split.setChildrenCollapsible(False)
```

`restoreState` also restores whatever collapsible flag was saved with the blob. Re-assert ours so a stale entry — written by an older build, or by hand — cannot bring back a console that vanishes when the handle is dragged to the top.

## ConsolePanel._scroll_to_bottom

### lines 1443-1445

```python
"""Follow the newest line, unless the user has scrolled away.
```

Only while following. Raising a section (clicking its heading) is a statement that the user is reading there, and a log that scrolls away from what you are reading cannot be read at all.

## ConsolePanel.begin_topic

### lines 1576-1577  _(unsure)_

```python
self._last_entry_kind = ""
```

Same subject: keep the bar, but still break the block so the next append opens one in its own colour.

### line 1583, trailing  _(unsure)_

```python
self._last_entry_kind = ""
```

force next append to open a block

## ConsolePanel.append_stdout

### lines 1617-1619

```python
accent = color_output()
```

Open a "spaCR output — <module> — <function>" banner + a fresh blue-text block. Reused until a different entry type breaks it.

## ConsolePanel._append_notice_on_gui_thread

### lines 1661-1663

```python
core = source.strip()
```

Call sites may add line breaks for console layout. Translation keys deliberately omit incidental leading/trailing whitespace, so retain that framing around the translated semantic template.

## ConsolePanel.append_error

### lines 1729-1730  _(unsure)_

```python
if console_write_in_progress():
```

Same re-entrancy refusal as append_stdout, and for the same reason: this path also builds a _StdoutBlock and fills it.

## ConsolePanel.as_text

### line 1749, trailing  _(unsure)_

```python
last = self._entries.count() - 1
```

trailing stretch

## ConsolePanel.raise_section

### lines 1841-1842  _(unsure)_

```python
folded = False
```

A nested heading the user folded stays folded: expanding the section above it is not a request to unfold what is inside it.

### lines 1851-1852  _(unsure)_

```python
QTimer.singleShot(0, lambda: self._scroll_widget_to_top(bar))
```

After layout, not during: the geometry this scroll needs does not exist until the widgets just shown have been laid out.

## ConsolePanel._scroll_widget_to_top

### line 1865, trailing  _(unsure)_

```python
return
```

section torn down between click and layout

## ConsolePanel._is_raised

### line 1880, trailing  _(unsure)_

```python
return False
```

section torn down between click and query

### lines 1882-1883

```python
return abs(scrollbar.value() - min(top, scrollbar.maximum())) <= 4
```

The same 4 px tolerance the follow-output check uses: a scrollbar dragged by hand does not always land on an exact value.

## ConsolePanel.clear

### line 1917  _(unsure)_

```python
while self._entries.count() > 1:
```

Remove every entry (but keep the trailing stretch)

### lines 1926-1928

```python
self._current_topic_label = None
```

FORGET THE TOPIC MEMO. Without this the first banner after a clear matches the last one before it, is skipped as a repeat, and the output that follows sits under no heading at all.

## ConsolePanel.set_ai_active

### lines 1933-1935  _(unsure)_

```python
def set_ai_active(self, on: bool) -> None:
```

AI toggle + provider — external setters called by AppScreen.

## ConsolePanel._on_submit

### lines 1953-1955  _(unsure)_

```python
def _on_submit(self) -> None:
```

Submit — Enter in the input

### line 1970  _(unsure)_

```python
self._append_user(text)
```

Local note — green "spaCR user" text under its own banner.

## ConsolePanel._send_to_ai

### lines 2002-2003

```python
return
```

Silent no-op: another stream is running. The AppScreen actions row exposes the Cancel button, not us.

### lines 2012-2015

```python
self._ai_error_traceback = ""
```

A QUESTION OF THE USER'S OWN ENDS THE ERROR PAIRING. Whatever comes back next answers this, not the crash -- and attaching it to a bug report as an analysis of the crash would put a confident, unrelated explanation in front of a maintainer.

### line 2017  _(unsure)_

```python
self._append_user(text + f"\n\n[{status}]")
```

User message — green "spaCR user" text.

### lines 2019-2021

```python
ai_color = ai_color_for_provider(self._current_provider_name)
```

AI reply — a "spaCR AI" banner tinted in the provider colour, with a three-dot working indicator that cycles until the stream finishes, followed by the reply text in the same provider colour.

## ConsolePanel._console_context_for_question

### lines 2096-2097  _(unsure)_

```python
label = tr("Console context: {n} chars sent", n=f"{len(context):,}")
```

The counts are interpolated after translation so the catalog keys stay free of digits and a translator can move the number.

### lines 2101-2103

```python
return context, label
```

Returned, not written to a persistent label: the caller stamps it on the message this context went with, where it stays true. A shared label is stale from the moment the next question is typed.

## ConsolePanel._start_stream

### lines 2128-2131

```python
thread, worker = make_stream_thread(
```

Parent the thread to this panel so its C++ lifetime is tied to the panel, not to our Python refcount. Without this the QThread can be GC'd between worker.run returning and thread.finished firing → Qt aborts.

## ConsolePanel._prune_retired

### line 2158  _(unsure)_

```python
pass
```

C++ already deleted — safe to drop

## ConsolePanel.shutdown

### lines 2195-2196  _(unsure)_

```python
try:
```

Defensively belt-and-suspender: also try every provider's cancel_stream() in case the worker itself is somehow lost.

### lines 2210-2211

```python
for pair in list(self._retired):
```

Also drain any retired (post-finished) threads that haven't been fully cleaned up yet.

## ConsolePanel._on_stage

### line 2222  _(unsure)_

```python
"""Ignore a stage change from the streaming worker.
```

Could show a spinner; keeping this quiet for now.

## ConsolePanel._on_chunk

### lines 2239-2240  _(unsure)_

```python
if self._current_stdout is None or self._last_entry_kind != "ai":
```

Stream into the provider-coloured AI block created in _send_to_ai. Guard in case it was cleared (open_error_flow uses its own path).

## ConsolePanel._on_stream_finished

### lines 2250-2254

```python
"""Close the reply block and retire the streaming thread.
```

Retire the current (thread, worker) pair — hold both refs in a list so Python can't GC the QThread before its OS thread has fully exited AND Qt's deleteLater has run. Prune already-dead entries on the way in so the list can't grow unbounded across a long session.

### line 2267  _(unsure)_

```python
if self._working_dots is not None:
```

Stop the cycling working dots — the stream is done.

### lines 2280-2284

```python
if getattr(self, "_ai_error_traceback", ""):
```

Kept for the bug report. Only a reply to an error the console itself raised counts: `_ai_error_traceback` is set by `open_error_flow` and cleared by any ordinary question, so an answer about something else cannot be filed as an analysis of the crash.

### lines 2294-2295  _(unsure)_

```python
if self._current_stdout is not None:
```

Terminate the AI reply block with a newline so pipeline stdout that arrives next visually separates from the reply.

### line 2299  _(unsure)_

```python
self.ai_stream_finished.emit()
```

Notify AppScreen so it can flip Cancel→AI on the toggle button.

## ConsolePanel.ai_explanation_of

### lines 2302-2304  _(unsure)_

```python
def ai_explanation_of(self, traceback_text: str) -> str:
```

Public: Explain-error entry point (called from AppScreen)

### lines 2323-2324

```python
return answer if mine.strip() == (traceback_text or "").strip() else ""
```

Compared on the traceback's own text rather than on identity: the reporter is handed the text again by the screen, not the object.

## ConsolePanel.open_error_flow

### lines 2346-2350

```python
self._ai_error_traceback = traceback_text
```

REMEMBER WHAT THIS TURN IS ABOUT, so the answer can be attached to the bug report for this error and no other. A console can hold several explanations across a session; pairing the reply with the traceback that prompted it is what stops an issue about one crash carrying an analysis of a different one.

### lines 2353-2354

```python
self._ai_messages.append({"role": "user", "content": prompt})
```

The AI always receives the full error; the console only echoes the raw traceback when show_raw is True.

### line 2362  _(unsure)_

```python
ai_color = ai_color_for_provider(self._current_provider_name)
```

AI reply with provider colour + cycling working dots.
