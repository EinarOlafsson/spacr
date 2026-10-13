# Notes from `spacr/qt/screens/distributed_jobs.py`

Prose lifted out of `spacr/qt/screens/distributed_jobs.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [ExecutionProfileDialog.__init__](#executionprofiledialog__init__) (1 entry)
- [ExecutionProfileDialog._add_profile_row](#executionprofiledialog_add_profile_row) (2 entries)
- [DistributedJobsScreen.__init__](#distributedjobsscreen__init__) (2 entries)
- [DistributedJobsScreen._reload_profiles](#distributedjobsscreen_reload_profiles) (1 entry)
- [DistributedJobsScreen._edit_selected_profile](#distributedjobsscreen_edit_selected_profile) (1 entry)
- [DistributedJobsScreen._job_detail](#distributedjobsscreen_job_detail) (1 entry)

## ExecutionProfileDialog.__init__

### lines 101-103

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## ExecutionProfileDialog._add_profile_row

### lines 321-325

```python
attach_api_tooltip(
```

Module first, then the setting: the dot beside this label was built with the arguments this way round and reached the module's own page, while the label's help -- given them the other way named the help text as the module and landed on the documentation index. With the dot gone, the help is the only route to the page.

### lines 329-333

```python
field.setToolTip("")
```

``attach_api_tooltip`` deliberately expands the plain field help into structured HTML with an API link.  The generic retargeting pass therefore sees two *different* strings and conservatively keeps both.  This row created the richer label explicitly, so the plain duplicate on the editor is safe to remove here.

## DistributedJobsScreen.__init__

### lines 480-481  _(unsure)_

```python
from ..dnd import install_for
```

Drop anywhere on this screen: the path is resolved through spaCR's project layout, so the plate folder finds what this screen reads.

### lines 484-486

```python
from .settings_model import retarget_field_tooltips
```

Hover help belongs on a setting's NAME, not on the field the user is about to type into (instruction 113). One post-pass rather than a convention every hand-built row has to remember.

## DistributedJobsScreen._reload_profiles

### lines 624-632

```python
def _reload_profiles(self, selected: str = "") -> None:
```

NOTE: a hand-rolled ``dragEnterEvent``/``dropEvent`` pair used to sit here, taking the first local ``.csv``/``.json`` and nothing else. The shared dropzone installed in ``__init__`` (:class:`spacr.qt.dnd_handlers.SubmissionSettingsDropHandler`) takes the same files *and* a plate folder, resolving ``settings/*.csv`` inside it and asking which snapshot was meant when there is more than one. Keeping both would have been keeping one: an installed event filter sees the event before the widget's own handler, so these two could never have run again.

## DistributedJobsScreen._edit_selected_profile

### lines 699-700

```python
try:
```

Persist the replacement first: a disk error must not erase the only usable profile merely because the user renamed it.

## DistributedJobsScreen._job_detail

### lines 1010-1011

```python
for key in (
```

Profiles never contain credentials, but avoid encouraging users to paste arbitrary custom command lines into public bug reports.
