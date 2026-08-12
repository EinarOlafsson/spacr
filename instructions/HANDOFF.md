# Handoff — 2026-08-12

Written for whoever picks this up next, human or agent. It records what is
true right now, what needs the maintainer, and the traps that cost time
today so they cost nobody else any.

Read this, then `instructions/00_INDEX.txt`, then the open instruction you
are taking. **The trailing notes at the end of each instruction file are the
current state; the header often says "not started" when it is 80% done.**

---

## 0. FIRST: what is true as of the last update

- **Everything is pushed.** `nightly` is in sync with origin.
- **CI was triggered** against it: runs `31643731063` (tests) and
  `31643733227` (compat-matrix). **Read them first** --
  `gh run view <id> --log-failed` -- they are the live answer for
  instructions 43, 54 and 82.
- **28 of 29 GitHub issues are closed.** Only **#15** is open (measurements
  hangs, "database is locked"); a diagnostic comment on it records what is
  already defended and the two remaining candidate fixes.
- 23 open instructions.

## 1. What needs the maintainer

| Need | Unblocks | Note |
|---|---|---|
| **The cellpose ceiling decision** | 54, 82 | `>=4.0.7,<5.0` admits 4.2.1.1, and ten tests fail on it on EVERY Python version. Not a Python-range question. `CellposeModel.__init__`'s default `pretrained_model` drifted. Pick a ceiling. |
| A push, whenever work accumulates | 82 | the standing rule is: never push without asking |
| Nothing else | | xvfb is installed, the GPU is free, `gh` is authenticated |

`gh` **is** authenticated (`EinarOlafsson`, scopes `gist, read:org, repo`) and
all five workflows carry `workflow_dispatch`, so CI runs can be triggered
with `gh workflow run tests.yml --ref nightly` and read with
`gh run view <id> --log-failed`. Earlier notes claiming CI is unreachable are
wrong.

Genuine machine limits that remain: **no macOS, no Windows, no `makensis`**
(blocks 44, 45, 53), and publishing to PyPI / conda-forge needs the
maintainer's accounts (59).

---

## 2. State of the tree

- Branch `nightly`, **in sync with origin**.
- `docs/source/_extra/tutorials/**`, the i18n catalogs, and
  `instructions/open/48_*` are **owned by a concurrent codex session**. Do not
  edit them. 48 carries a hands-off note saying so.
- 24 open instructions, 52 done.

---

## 3. Traps. Read this section before touching anything.

**NEVER `git stash` in this repository.** There are pre-existing stashes and
`git stash pop` restores the wrong one, conflicting a dozen unrelated files.
Hit twice today. To test "does my change cause this", use a second checkout
or `git worktree add`, never stash.

**Verify a claim before acting on it.** Two claims from an automated survey
were wrong, and acting on one would have introduced a bug:

- "`organelle_min_size` is applied twice" — true, and harmless: the filter
  is idempotent AND the first pass is load-bearing, because the cytoplasm
  mask is built from the organelle mask between the two calls. Removing the
  "duplicate" would silently have carved sub-threshold debris out of a
  measured area.
- "the anndata regression is not from the merge work" — it was.

**Do not trust a list that claims to be exhaustive.** Two live tooltips were
deleted because `test_every_qt_section_hint_names_a_real_category`'s app list
omitted `classify_merged`, which renders both. The list is fixed; the lesson
is not.

**Some failing tests encode old behaviour on purpose.** Several today pinned
the exact defect being fixed (`settings["location_column"] == "test"`,
`assert len(out) == 3`, the three-column results shape). **Rewrite them to
assert the new contract and say why in the test. Never delete, never skip.**

**Two guards I wrote were too broad and had to be narrowed.** A guard that
fires where nothing could happen teaches people to disable it. Scope a guard
to the path that actually does the dangerous thing.

**A NEW MODULE FILE MUST BE REGISTERED IN `spacr._SUBMODULES`.** Adding
`spacr/foo.py` and nothing else turns every compat-matrix cell red on
`tests/test_smoke.py::test_lazy_loader_matches_files` — "file present but not
in _SUBMODULES". This happened TWICE today, with `object_roles.py` and then
`organelle_types.py`, and a scoped local test run cannot catch it because the
failing test is nowhere near the change. After adding any module, run:

    python -m pytest tests/test_smoke.py::test_lazy_loader_matches_files -q

**CPU/GPU etiquette:** at most 4–8 pytest workers. Qt needs
`xvfb-run -a python -m pytest …` — offscreen is not a substitute for a real
X server, which is what finally answered issue #72.

---

## 4. What was done today, with the numbers

| # | Instruction | State |
|---|---|---|
| 72 | Organelle type drives the settings | **done** — 53 settings → 6 visible |
| 73 | Advanced settings by function | **done** — Cell 21→8, Nucleus 20→7, Pathogen 21→8 |
| 74 | Loading screen covers the preload | **done** — 1305 ms vs 1268 ms baseline, no regression |
| 78 | Splash teal → black | **done** — takes the window's own background |
| 79 | Merge keys, duplicates, aggregation | **done** |
| 80 | Statistics at the declared level | **done** |

### The wrong-numbers defects fixed (this is the part that matters)

1. **Cytoplasm was silently dropped from every merge.** The code asked "does
   this table carry `cell_id`?" and cytoplasm is keyed `object_label`. Zero
   cytoplasm columns, no error.
2. **A nucleus was handed a picture of a different cell.** Crop paths were
   matched on the label alone, across object types. Half wrong; the half that
   was right was right by coincidence.
3. **`level` never touched the statistics** despite a tooltip promising it
   did. Object p = 4×10⁻³⁹ vs well p = 0.25 on the same data.
4. **`na='drop'` deleted a cell for having an *unmeasurable* child**, not a
   missing one. A correlation is NaN whenever a channel is flat.
5. **`_merge_grouped` joined inner unconditionally**, silently conditioning
   every result on infection.
6. **`remove_outliers` trimmed before the statistics**, shrinking SD and
   inflating *t*.
7. **A track-level filter never ran on any default configuration** — its
   discovery fallback was unreachable because the default is a non-empty
   string naming a column the classifier never writes.
8. **`crop_source='on_demand'` silently trained on the pre-cut PNGs.** Two
   vocabularies, no translation, and the error was swallowed unless verbose.
9. **Annotation mode overwrote `location_column` and left it overwritten**,
   so a user could not return to metadata mode by changing the mode
   (issues #91–#93, instruction 83).
10. **The filtered organelle plane was never written back**, so crops showed
    debris the measurements had dropped.

---

## 5. Where each open instruction stands

Percentages are estimates from each file's own trailing notes.

**Close to done**
- **81** GitHub issues — 96%. #72 closed under real X (600 resizes, 169 tests,
  no crash). #15 open. #91–#93 fixed but **not closed — waiting on a push** so
  the fix is real for the reporter.
- **77** Phantom settings and wrong numbers — 88%. Items a/c/d done.
- **37** Classify settings overhaul — 85%. crop_source clash fixed; the class
  selector redesign and the `classes` default drift remain.
- **43** Failing GitHub tests — 80%. **All 65 reported failures pass locally.**
  Needs a push to verify in CI.
- **83** Classify controls and location_column — 75%. Code fixed; closing the
  issues is what is left.

**Partly built**
- **31** Gate editor redesign — 70%. Ten specific items; the file lists them.
- **55** Performance sweep — 65%. xvfb now makes the rest measurable.
- **54** Python 3.9–3.14 — 60%. CI is readable now; this is unblocked.
- **47** Qt suite hangs — 50%. xvfb + `--timeout` turns the hang into a
  traceback.
- **76** More than one organelle — 45%. Steps 1–3 and the queued measure bugs
  done. **The real blocker is the plane budget being full at 7.**
- **60** Coverage sweep — 35%. Nothing blocks it.

**Not started, nothing blocking**
- **49** gate editor xD, **52** gate editor 3D, **69** live views, **75** UMAP
  figures. (69 and 75 had agents mid-flight when this was written — check
  `git log` and `git status` before starting either.)

**Blocked on hardware or accounts**
- **53** installers build and run, **59** conda-forge, **44** / **45**
  installer pages.

**Pinned last, by the maintainer**
- **82** green CI then bump to 1.5.0.5 — gated on the push.
- **58** strip `Co-Authored-By` from history — must be the very last thing.

---

## 6. Conventions worth keeping

- A feature goes into `instructions/open/NN_slug.txt` **before** it is coded,
  quoting the maintainer's own words in a `Requested:` line.
- Commits are authored **Einar Olafsson**, with **no AI attribution**.
- Measure before claiming. Every fix above has a number or a command behind
  it, and the two disproven claims are recorded as disproven rather than
  quietly dropped.
- A bug found in passing gets fixed or written down, never just narrated.
- If a change breaks the legacy Tk GUI, ship it and update Tk to fit.
