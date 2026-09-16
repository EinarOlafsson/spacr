# Notes from `spacr/qt/widgets/setup_dialog.py`

Prose lifted out of `spacr/qt/widgets/setup_dialog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [SetupDialog](#setupdialog) (1 entry)
- [SetupDialog.__init__](#setupdialog__init__) (1 entry)
- [SetupDialog._build_groups](#setupdialog_build_groups) (1 entry)

## SetupDialog

### lines 35-40

```python
class SetupDialog(QDialog):
```

THE ARGUMENTS ARE DOCUMENTED ON `__init__`, ONCE. They were listed here too, as a NumPy ``Parameters`` section, and AutoAPI runs with ``class_content='both'``: the class docstring and ``__init__``'s are concatenated before Napoleon sees them, the section became a field list, and ``__init__``'s opening prose then ended it mid-way -- "Field list ends without a blank line", which `sphinx-build -W` makes fatal.

## SetupDialog.__init__

### lines 81-83

```python
buttons.button(QDialogButtonBox.Cancel).setText("Not now")
```

"Not now" rather than "Cancel": nothing is being cancelled, and a user who reads Cancel as "undo what I already have" will not press it even when it is the right button.

## SetupDialog._build_groups

### lines 107-109

```python
continue
```

A GROUP WITH NOTHING IN IT IS NOT DRAWN. The provider question removes itself when no CLI is installed, and an empty "The assistant" heading would read as a bug.
