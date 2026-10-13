# Notes from `spacr/qt/widgets/settings_advisor_dialog.py`

Prose lifted out of `spacr/qt/widgets/settings_advisor_dialog.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (1 entry)
- [QuestionsPage.__init__](#questionspage__init__) (1 entry)
- [ProposalPage._render](#proposalpage_render) (1 entry)
- [SettingsAdvisorDialog.show_the_proposal](#settingsadvisordialogshow_the_proposal) (1 entry)

## Module level

### lines 54-56

```python
_SETTINGS_ADVISOR_UI_SOURCES = tuple(dict.fromkeys((
```

Exact presentation strings assembled from the headless question records are not visible to the Qt literal extractor. Export them as one deterministic private inventory for the runtime-catalog builder and its coverage tests.

## QuestionsPage.__init__

### lines 110-112

```python
form.addRow("", _muted(question.why_it_matters, self))
```

WHY IT MATTERS, UNDER THE QUESTION. A user who cannot see what an answer buys cannot answer it well, and the alternative is a number typed to make the dialog go away.

## ProposalPage._render

### lines 223-225

```python
same = _same(was, choice.value)
```

UNCHANGED IS SAID, NOT HIDDEN. A proposal that listed only the differences would read as "everything else is wrong", when most of a tuned panel is usually already right.

## SettingsAdvisorDialog.show_the_proposal

### lines 343-347

```python
self._advice = advise_that_runs(self._reading,
```

THE CHECKED ROUTE (196). A proposal the run would refuse is not a proposal, and this window's whole promise is that these are the settings for the user's data -- so what it shows has been asked of the validators that would stop the run, not just of the canonicaliser that fills defaults.
