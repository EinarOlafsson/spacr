# Notes from `spacr/qt/ai/issue_preview.py`

Prose lifted out of `spacr/qt/ai/issue_preview.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [IssuePreviewDialog.__init__](#issuepreviewdialog__init__) (2 entries)
- [IssuePreviewDialog._check_for_diagnosis](#issuepreviewdialog_check_for_diagnosis) (1 entry)
- [IssuePreviewDialog._show_diagnosis](#issuepreviewdialog_show_diagnosis) (1 entry)

## IssuePreviewDialog.__init__

### lines 72-74

```python
self._console = console if console is not None else getattr(
```

The console owns the AI conversation; this dialog borrows it. Taken from the parent when not passed, so the screen that opens this does not have to know about the button.

### lines 111-115

```python
self.diagnose_btn = buttons.addButton(
```

ASK spaCR AI FROM HERE. The report is the moment the user is looking hardest at the error, and it is also the last moment before it goes somewhere public -- a diagnosis is worth having in both directions: it may save the report entirely, and it makes the report better if it does not.

## IssuePreviewDialog._check_for_diagnosis

### lines 233-234

```python
self._end_diagnosing()
```

The stream ended without an answer for this error -- a provider error, or a reply the console did not pair with this traceback.

## IssuePreviewDialog._show_diagnosis

### lines 272-273  _(unsure)_

```python
self._source_body += section
```

Into the SOURCE too, so the strip toggle does not drop it:

`_refresh_body` rebuilds the box from `_source_body`.
