# Notes from `spacr/qt/menus.py`

Prose lifted out of `spacr/qt/menus.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [set_menu_role](#set_menu_role) (1 entry)
- [name_the_macos_application_menu](#name_the_macos_application_menu) (1 entry)

## set_menu_role

### lines 67-70

```python
pass
```

A binding that will not take the role. The menu still works, it is simply in the ordinary place rather than moved into the macOS application menu -- not a reason to fail building the action, which is what the caller wanted.

## name_the_macos_application_menu

### lines 146-152

```python
class_name = objc.object_getClassName(info) or b""
```

THE ONE CHECK THAT CANNOT ITSELF FAIL. `object_getClassName` is a plain C call into the runtime -- no message is sent, so no Objective-C exception can be raised, and an uncaught one would abort the process rather than surface here as a Python error. `__NSCFDictionary` is the toll-free-bridged class CFBundle builds its info dictionary as; a frozen or immutable class name means this launch is not one where the key can be written.
