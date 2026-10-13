# Notes from `spacr/qt/shutdown.py`

Prose lifted out of `spacr/qt/shutdown.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ask_how_to_quit](#ask_how_to_quit) (1 entry)
- [restart_spacr](#restart_spacr) (2 entries)
- [force_quit_now](#force_quit_now) (2 entries)
- [style_as_danger](#style_as_danger) (2 entries)

## ask_how_to_quit

### lines 93-94

```python
force.setObjectName("DangerButton")
```

Red, because it is the one that loses data. The role alone does not colour it on every style.

## restart_spacr

### lines 139-141

```python
subprocess.Popen(started, start_new_session=True,
```

DETACHED. `start_new_session` puts the child in its own process group, so the signal that takes this process down does not follow it, and it survives the terminal that started us.

### lines 147-149

```python
LOG.error("could not start spaCR again (%s); NOT quitting", exc)
```

THE STATE IS LEFT ON DISK DELIBERATELY. spaCR did not restart, so the user will start it themselves, and when they do they should land back where they were.

## force_quit_now

### lines 172-175

```python
pass
```

A BROKEN SINK MUST NOT BLOCK. This runs when a graceful stop has already failed, so a handler that will not flush cannot be what stops the process leaving -- a force quit that hangs is the original complaint, twice.

### lines 181-182  _(unsure)_

```python
pass
```

Same contract for stdout and stderr: a terminal that has gone takes its flush with it.

## style_as_danger

### lines 325-334

```python
ink = P.get("bg") or "#000000"
```

The ink on a FILLED danger surface is `bg`, not white. theme.py's CONTRAST_RULES carries ("bg", "error", 4.5) with the comment "`bg` is the ink on filled accent/danger surfaces: the selected menu row, a pressed button, DangerButton on hover", and the application sheet's own `#DangerButton:pressed` rule inks with `P["bg"]` for that reason. This helper hard-coded `#ffffff` instead, which is only right on the light theme: `error` is a PALE red on cell and glass, so white ink on the hover fill measured 2.20:1 and 2.04:1 — below AA-large, on the one control that force-quits a run. `bg` measures 6.23:1 (light) through 9.55:1 (glass) and is guaranteed by the contrast rule above.

### lines 336-339

```python
name = button.objectName()
```

Key the rule on whatever the button is already called. Setting a name here would take the caller's: `QuitSpacrButton` became `DangerButton` and every lookup for it stopped finding anything a styling helper must not decide a widget's identity.
