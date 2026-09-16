# Notes from `spacr/qt/setup_screen.py`

Prose lifted out of `spacr/qt/setup_screen.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [questions](#questions) (3 entries)
- [_provider_choices](#_provider_choices) (1 entry)

## questions

### lines 122-126

```python
("language", "Language", prefs.get_language, prefs.set_language,
```

THE NATIVE NAME, not the code. A user choosing their own language is the one person who cannot be expected to recognise its ISO code, and `getattr(prefs, "VALID_LANGUAGES", ("en",))` -- which is what this read before -- found no such attribute and offered English alone, on a screen whose first question is the language.

### lines 129-133

```python
("theme", "Theme", prefs.get_theme_choice, prefs.set_theme_choice,
```

FLIPPED, because `theme_choices()` is (caption, value) while every other list here is (value, caption). Normalised at the source rather than special-cased in the screen: a screen that knows which of its questions is back to front is a screen that gets it wrong the next time a question is added.

### lines 138-160

```python
("spacr_mode", "spaCR mode", prefs.get_performance_level,
```

THE LEVELS, NOT THE POSTURES. This offered `SPACR_MODES`, which is the OLD three-value resource posture -- Extra Performance, Performance, Balanced -- while Preferences offers the five `PERFORMANCE_LEVELS`. So Laptop and Workstation existed, were settable in Preferences, and could not be chosen on the screen whose whole job is choosing them once.

They are not interchangeable. `spacr_mode_for_level` folds five levels onto three postures (laptop -> extra_performance, workstation -> balanced), so writing through `set_spacr_mode` cannot express either end of the scale: picking Balanced here and Workstation in Preferences produced the same posture and two different answers to "what did I choose".

The level is the setting a user picks; the posture is what the cleanup code reads. `set_performance_level` writes both, in that order, which is why it is the one to call.

(The previous defect here was the same shape one layer down: a `getattr(prefs, "VALID_SPACR_MODES")` that found nothing and fell back to a one-item default, so the screen offered Balanced alone. Named directly ever since, so a rename breaks the import instead of silently shortening the list.)

## _provider_choices

### lines 211-213

```python
return [("", "whatever is available")] + found if found else []
```

"whatever is available" first, and it IS the default: a machine with two CLIs today may have one tomorrow, and a pinned name that is gone is worse than no preference.
