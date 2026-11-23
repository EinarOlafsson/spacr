# omezarr-p1b (0aj P1b, P1c) English source handoff

Translations are dropped until everything else is done (maintainer, 2026-11-23
15:10). This file and `tr_omezarr-p1b_sources.json` list the new English
sources only; every one is wrapped in `tr()` and the English fallback works.
No catalog, translation pin or reviewed translation record is changed.

P1b:
- `spacr/image_store.py`: the summary line of the originals step (moved /
  deleted / kept) and the per-field "kept in orig/" line; the reasons a plane
  cannot be compared.
- `spacr/io.py`: the refusal for wells that are not letter + number; the
  plate-split dry-run list, its refusals (taken target, bad plate name, the
  check after the moves), the summary line, and the headless refusal naming
  `split_plates`.
- `spacr/core.py`: the refusal when a stored field does not match its originals.
- `spacr/qt/screens/app_screen.py`: the "Several plates in one folder" popup
  (title and text).

Settings help: changed `keep_original_images`, `save_original_images`; new
`split_plates` (label and tooltip, new key).

The exact texts and sha256 are in the JSON (regenerated with
`0aj_scratch/tools/tr_new.py` against 1b31681377).
