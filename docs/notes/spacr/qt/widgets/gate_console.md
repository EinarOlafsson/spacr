# Notes from `spacr/qt/widgets/gate_console.py`

Prose lifted out of `spacr/qt/widgets/gate_console.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_ChatInput.__init__](#_chatinput__init__) (1 entry)
- [evaluate](#evaluate) (1 entry)
- [GateConsole.__init__](#gateconsole__init__) (3 entries)

## _ChatInput.__init__

### lines 69-71

```python
self.setFixedHeight(
```

Sized in LINES rather than pixels so it follows the font scale. A pixel height picked at one zoom level is the wrong height at every other one.

## evaluate

### lines 158-160

```python
value = eval(text, {"__builtins__": _builtins()}, scope)  # noqa: S307
```

eval, not exec: an expression has a VALUE, which is the thing being asked for. Statements would let the user rebind `df` and then wonder why the plot disagrees with the console.

## GateConsole.__init__

### lines 218-221

```python
self.log.setMinimumHeight(CONSOLE_MIN_HEIGHT)
```

A transcript worth reading. It already stretched, but with nothing holding a floor the entry rows below could squeeze it to a couple of lines -- and a console you have to scroll to read one answer in is the moment you least want to be scrolling.

### lines 223-225

```python
self.setMinimumWidth(CONSOLE_MIN_WIDTH)
```

Width floor on the PANEL, not the log: the entry rows below it are what a narrow column really ruins, since a QLineEdit cannot wrap and simply scrolls.

### lines 249-254

```python
self.chat = _ChatInput(self)
```

Multi-line, because a question in words is often a paragraph and a QLineEdit is one line by construction -- it cannot be made taller, only wider. Enter still SENDS: that habit is already built, and taking it away to gain a newline is a bad trade. The newline lives on Shift+Enter, which is where a chat box usually keeps it.
