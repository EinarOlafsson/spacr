# Notes from `spacr/qt/widgets/api_help_label.py`

Prose lifted out of `spacr/qt/widgets/api_help_label.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ApiHelpLabel._compose_help](#apihelplabel_compose_help) (1 entry)
- [ApiHelpLabel._refresh_help](#apihelplabel_refresh_help) (2 entries)

## ApiHelpLabel._compose_help

### lines 191-193

```python
return escape(description)
```

Nothing to link to. `format_tooltip` would fall back to the documentation index, which is a link that answers no question the reader asked.

## ApiHelpLabel._refresh_help

### lines 222-223  _(unsure)_

```python
self.setToolTip(html)
```

Kept on the widget as well as in the popup: this string is what the accessibility tree reads out.

### lines 226-227

```python
self.setCursor(Qt.WhatsThisCursor)
```

The cursor is the affordance the dot used to be: it says there is something here to read before the popup appears.
