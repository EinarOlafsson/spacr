# templaterun (0ai Batch Runner template runs) English source handoff

Translations are dropped until everything else is done (maintainer, 2026-11-23
15:10). This file and `tr_templaterun_sources.json` list the new English
sources only; every one is wrapped in `tr()` and the English fallback works.
No catalog, translation pin or reviewed translation record is changed.

- `spacr/batch.py`: the template refusals (not a folder, no module
  settings) and the plain-word mismatches between a new folder and the
  template (no images, file-name pattern, channels, z planes, image size,
  pixel type, the template run itself).
- `spacr/qt/screens/batch.py`: the Template run and Jobs sections (titles,
  captions, buttons, tooltips, folder pickers), the status lines, the job list
  suffix, and the editing strip on a module screen ("Editing the Batch Runner
  job ...", "Back to the Batch Runner", "Stop editing the job").
- `spacr/qt/dnd_handlers.py`: the refusals of the two new drop sections.

The exact texts and sha256 are in the JSON (regenerated with
`0ai_scratch/tools/tr_new.py` against origin/nightly).
