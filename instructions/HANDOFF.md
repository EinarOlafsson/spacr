# Handoff — 2026-08-14

Written for whoever picks this up next, human or agent. It records what is
true right now, what needs the maintainer, and the traps that cost time so
they cost nobody else any.

Read this, then `instructions/00_INDEX.txt`, then the open instruction you are
taking. The authoritative status is the current ledger and each instruction's
latest dated record; older sections below are preserved as history.

Current checkpoint: branch `nightly`, 88 done / 2 open after instruction 83's
closure. The complete 69-lesson, 487-scene, 50-voice tutorial release is live
at `https://einarolafsson.github.io/spacr/tutorials/`; its reusable audit skill,
scene sampler, live verifier, and tests are committed. Hosted tutorial media is
at Hugging Face commit `17af9e67b7dbd16c465e7f494091eb830728e161`, and live
mobile playback passed the full 69.6-second default narration, captions,
pause/resume, end, and replay contract. Instruction 83 is now complete: all
all nine API catalogs are current at 6,582 symbols, all nine runtime catalogs
are current at 3,588 entries, and every installer catalog is current at 57
strings. Coverage is regenerated; the completion release selection passed 285
tests, and the post-closure feature sync's Sphinx build exits zero with 659
HTML pages and seven warnings.
The coherent catalog, evidence, builder, and test batch is commit
``a09ebe90``.
The only open records are 58 (history/contributor cleanup, intentionally last)
and 82 (green CI and the 1.5.0.5 release).

The remainder of this handoff contains historical investigation notes. Its old
69-done / 17-open count and hands-off ownership table are superseded.

---

## 0. THE FOUR LESSONS. READ THESE BEFORE YOU TOUCH ANYTHING.

### 0a. Audit before you build — EIGHT files have been wrong about themselves

| file | said | was |
|---|---|---|
| 73 | "not started" | all six items done |
| 43 | failures "need a CI run" | every one reproduced locally in seconds |
| 31 | hexbin / colour map / Walk open | all three shipped |
| 49 | "not started" | the reduction shipped; only the column CHOICE was missing |
| 77 | ten findings open | fixed elsewhere, unrecorded |
| 47 | "hangs at 25%" | does not hang at all — see 3d |
| 69 | "not started" | all five steps shipped |
| 86 | (b) checkpoint/resume "not started" | `checkpoint.py` + `resume.py` fully built and wired |

**Read the code before building from any header.** Trailing notes beat the
header; the code beats both. Twice this week the "missing" thing existed under
a name nobody grepped for.

### 0b. GREEN TESTS DO NOT MEAN THE FEATURE WORKS

Instruction 52 was closed on **97 passing tests** and the maintainer opened the
app and found 3D gating unusable. The geometry was right and fully tested; the
*controls* were unreachable, and **no test pressed one**.

If an instruction is about something a user touches, open the app:

```bash
cd /mnt/firecuda2/codex/repo/spacr && spacr
```

A model-layer test suite is necessary and is not sufficient. Where possible,
put the model in a Qt-free module (see `spacr/umap_search.py`) so the two can
be tested apart and the UI test is about the UI.

### 0c. `git commit -F -` WITHOUT PATHS COMMITS THE INDEX

Six consecutive commits carried **tests only**, because they used
`git commit -q -F - <<'MSG'` with no path arguments while only test files had
been `git add`ed. The branch shipped tests referencing code that was not
there. Local pytest passed the whole time, because pytest reads the working
tree and not HEAD.

**Always pass explicit paths, and always verify:**

```bash
git commit -F - -- path/one.py path/two.py <<'MSG' ... MSG
git show --stat HEAD          # <- the check that would have caught it
```

A green test run is not evidence about what was committed.

### 0d. NEVER `Write` A FILE YOU HAVE NOT READ

`spacr/checkpoint.py` was overwritten with a new module of the same name. It
already existed, with `CheckpointStore`, atomic writes and signature checking.
Restored from git, but nothing warned first. `ls spacr/ | grep <name>` before
creating anything, and prefer `Edit` over `Write` for a path that may exist.

---

## 1. What needs the maintainer

| # | Question | Cost to answer |
|---|---|---|
| **81** | The reporter's `df -T` on the path in issue #15's traceback. If it is a local filesystem the shipped WAL fix covers them; if NFS/CIFS/Lustre, WAL is unsafe there and the fix is different. | one comment |
| **81** | A stack trace, or a repro on a real display, for the remaining native crash. Four inspections and one measurement say it is **not** in any Python path. Guessing would be a change with no evidence. | — |
| **44/45/53** | A macOS host, a Windows host, and `makensis`. The Linux halves can be done without them. | — |
| **59** | conda-forge accounts. | — |
| **93** | Whether a stack whose intensities exceed the 16-bit ceiling should be **refused** rather than silently rescaled. See §4. | a decision |

---

## 2. State of the tree

* Branch `nightly`, 51 commits on 2026-08-13, all pushed.
* Working tree carries four files that are **not ours**: `README.rst`,
  `docs/source/index.rst`, `skill/FACTS.md`, `tests/test_docs_media_budget.py`.
  These belong to the concurrent codex session (48/83). **Do not commit them.**
* `instructions/open/48` and `83` are codex's. So are
  `docs/source/_extra/tutorials/**` and the i18n catalogs.
* `spacr-nightly` at `/home/olafsson/repo/spacr-nightly` is a **stale**
  checkout (last commit 2026-07-26). Line numbers quoted from it will not
  match. The working copy is `/mnt/firecuda2/codex/repo/spacr`.

### The environment

* CI cells are already local conda envs: `spacr12` (3.12, numpy 2.4.6,
  sklearn 1.9, cellpose 4.2.1.1), `spacr13`, `spacr14`. Default `spacr` is
  3.10 / sklearn 1.7.2. Failures that "do not reproduce locally" reproduce in
  `spacr12` in seconds.
* Qt tests need `xvfb-run -a`. `-p no:randomly` for anything order-sensitive.
* Max 16 CPU cores, max 4 concurrent subagents.

---

## 3. Traps

### 3a. A new public module obliges an i18n rebuild

Adding a module to `spacr/__init__.py::_SUBMODULES` turns the docs job red
until `python tools/build_documentation_i18n.py --sources-only` is run (writes
only `en.json`). Forgetting `_SUBMODULES` itself turns **every** compat-matrix
cell red on `test_smoke.py::test_lazy_loader_matches_files`. It has happened
twice.

### 3b. Headless Qt refuses static modals

`QMessageBox.information` / `QInputDialog.getText` raise in tests by design —
`tests/qt/conftest.py` enforces it, because a modal runs its event loop in C++
and hangs the run. `monkeypatch.setattr(QMessageBox, "information",
staticmethod(lambda *a, **k: None))`. Patching `exec` does **not** cover it.

### 3c. `spacr.settings.tooltips` is NOT complete on import

Six pipelines register their keys from their own module via
`register_defaults`, which runs on import of that module. Read cold, `dst` and
`cmap` look undocumented — and a tool that then writes "no description" is not
missing a sentence, it is writing a **wrong** one. See
`tools/build_notebook_settings.py::_load_registrations`.

Also: `register_defaults` **refuses** to let one module redefine another's
tooltip. Adding `'dst'` to the core dict breaks `import spacr.sequencing_qc`.

### 3d. Instruction 47 does not describe a hang

Two full runs, both under a per-test `--timeout=900` that **never fired**:

| cap | reached | ended by |
|---|---|---|
| 2 h | 66% | my `timeout`, EXIT=124 |
| 5 h | 83% | my `timeout`, EXIT=124 |

Output was still being written a minute before each kill. The suite
**decelerates**: 33 %/hour over the first two thirds, **5.7 %/hour** over the
next — about six-fold. Something accumulates between tests (leaked widgets,
live QThreads, unclosed figures).

That also explains the file's own contradiction: at a decaying rate, where a
run *appears* to stall depends on how long you waited.

**Next step is not "find the hanging test".** Log RSS and
`len(QApplication.allWidgets())` per test and find what climbs. `pytest-xdist
--dist loadfile` would make it finish by restarting workers, but would **hide**
the leak — and if those widgets outlive their screens in the app too, it is a
product defect. ~15 failures appeared in run A and are **unidentified**; `-rf`
was passed to run B and it was killed before the summary printed.

### 3e. Two corrections I published and had to retract

Both are recorded because being wrong twice in the same way is the risk.

* **`clear_field_rows` is called.** I reported it as dead code and a live
  duplicate-row bug. The grep excluded `resume.py` itself; it is called from
  `plan_measure_resume` at `resume.py:2147`. **There is no duplicate-row bug.**
* **Grouped splitting exists.** I confirmed a report that spaCR never groups
  train/test splits. True of five sites — and **false as a blanket**:
  `active_learning.py` has `StratifiedGroupKFold` with a `GroupShuffleSplit`
  fallback, and `cv_group_by` already defaulted to `'well'`.

Grep the module you are about to accuse, not just its callers.

---

## 4. Findings filed but not fixed

**93 — the intensity rescale factor is per field and unrecorded.**
`measure._promote_merged_to_uint16` picks `factor = 65535/top` from **that
field's own maximum** when intensities exceed 65535, and runs once per file. A
bright field is divided by more than a dim one, so the same object measures
differently in each. The factor prints only under `verbose` and is never
stored. The float-on-[0,1] path uses a fixed 65535 and is fine; the dangerous
path is rare, which is why it would go unnoticed.

**Not yet investigated — a merge warning the maintainer saw in a real run:**

```
'plateID':  57170 of 65737 objects disagree between cell and pathogen
'rowID':    57170 …   'columnID': 57170 …   'fieldID': 57170 …
```

`object_label` disagreeing is expected — a cell and its pathogen have
different labels. The **identity** columns disagreeing means the merge may be
pairing rows from different fields. This could be serious and nobody has
looked.

---

## 5. Where each open instruction stands

| # | Item | Stage |
|---|---|---|
| **52** | 3D plane-anchored gates | **Controls rebuilt today.** Plane picker, shape dropdown, spin/draw, dragged slab. Geometry (Cylinder/Prism/Box/Composite/thresholds) was already right |
| **95** | Image UMAP, starplast-style | **Model + GPU button built.** The 2D/3D container, the grid-on-black, the clustering walk and removing the figure slider are NOT |
| **94** | Splits group by well | ~40%. Ladder (cell/field/well/plate) built, `none`→`cell` renamed with aliases. Five sites still ungrouped |
| **93** | Per-field intensity factor | Filed, not started |
| **76** | More than one organelle | Not started |
| **47** | Qt suite | Diagnosed (§3d). ~15 failures unnamed |
| **75** | Image UMAP figures | **Superseded by 95** — can be closed |
| **81** | GitHub issues | 25 of 26 closed |
| 44, 45, 53, 59 | Installers, conda-forge | Blocked, §1 |
| 48, 83 | Tutorials, catalogs | **codex — do not touch** |
| **82** | Green CI | **SECOND TO LAST.** Version bump discarded by the maintainer |
| **58** | Strip Claude from history | **LAST.** Includes `git config user.name "Einar Olafsson"` — the repo has none set, so codex commits show `olafsson` |

---

## 6. Standing rules the maintainer has set

* A feature goes into `instructions/open/NN_slug.txt` **before** it is coded,
  quoting the request in a `Requested:` line. Merge overlapping asks into the
  first task; do not file duplicates.
* Print the done/left table whenever an item is finished.
* Commits are authored **Einar Olafsson**, never any AI attribution, and carry
  no `Co-Authored-By` trailer.
* Fix bugs and logic that lead to erroneous or misleading results.
* If a change breaks the legacy Tk GUI, ship it and update Tk to fit.
* Correct the format going forward and migrate old data, rather than
  preserving a bug for compatibility.
* Standing push approval for this run. **82 then 58 are the last two, in that
  order.**

---

## 7. Conventions worth keeping

* **Say what a number cannot say.** A truncated inventory, a skipped hash, a
  refused projection — each is stated rather than passed over. An absent
  fingerprint that reads as an absent difference is a false assurance.
* **Greyed, not removed** (INVARIANTS 6) for a control another mode does not
  read. A control that vanishes takes its value with it.
* **One source of truth.** Two editors of the same setting drift; the
  `_ClusterSettingsDialog` docstring records what it cost last time.
* **Refuse rather than fall back silently** where the fallback would be
  presented as the thing that was asked for — a random split reported as
  grouped, ImageNet statistics given to a run that asked for its own.
* **Measure, then decide.** Every palette, threshold and default that changed
  this week changed on a number recorded in the instruction file.
