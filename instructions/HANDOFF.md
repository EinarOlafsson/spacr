# Handoff — header rewritten 2026-09-02

Written for whoever picks this up next, human or agent. It records what is
true right now, what needs the maintainer, and the traps that cost time so
they cost nobody else any.

Read this, then `instructions/00_INDEX.txt`, then the open instruction you are
taking. The INDEX is generated from the folder and carries a percentage on
every open row; this file carries the things a generator cannot know.

**THERE IS ONE LEDGER AND IT IS `00_INDEX.txt`.** `instructions/PROGRESS.md`
was a second, hand-maintained one; it went stale, listed about 34 closed items
as unfinished, and carried a row for an item 60 that has no file anywhere. It
was retired on 2026-09-04 at the maintainer's decision — see instruction 352
for the measurement and for where item 60's goal now lives. Do not start
another one.

## WHERE THINGS ACTUALLY STAND (2026-09-02)

Branch `nightly`. **349 done / 24 open** (2026-09-04; the count is generated
into `00_INDEX.txt`, so read it there rather than here). The old header said 92/10 and named
a `codex/tutorial-api-final` checkpoint; both were nine months of work out of
date and are in section 8 below with the rest of the history.

Two sessions share this repository. `instructions/open/325` is the channel
between them and the record of who owns what — **read it before touching
anything**, and announce there before editing `setup.py`, `spacr/__init__.py`,
`spacr/schema.py`, `spacr/accelerator.py` or `.github/**`.

WHAT IS RELEASE-BLOCKING, from instruction 331 which splits the list into
before and after the version bump: **288** (green CI, per-module coverage),
**05** and **304** (the bump and Zenodo, both needing the maintainer), **316**
(translations), and **01** (Windows self-update, code-complete and waiting only
on publication).

**314 CAME OFF THAT LIST ON 2026-09-04**, closed by the maintainer — "i never
get that problem any more!" — and NOT by a fix: nothing was ever changed
against the stall. Its 2026-09-03 measurement of a 6,252 ms event-loop freeze
opening Regression stands unretracted in `done/314`, along with Home's
unexplained 2.03 → 4.51 s doubling, which was always a separate regression and
is still unowned. If a module feels slow again, read that file before
measuring anything: eleven causes are already eliminated there.

WHAT NEEDS THE MAINTAINER AND NOTHING ELSE — see section 1 — is now short
enough to list here: the Zenodo toggle, the 1.5.0.5 go-ahead, the measure
settings instruction 337 part 3 needs, three sentences of Spanish, Chinese and
Korean for instruction 306, and the nine hand-written `_ROWS` translations
instruction 316 is waiting on.

## THE MEASUREMENT LESSONS OF 2026-09-02

Four items were advanced in one night and every one of them turned on a
measurement being wrong before it was right. They are here because they cost
hours and will cost them again.

**A HEAD BASELINE, OR THE NUMBER MEANS NOTHING.** Run the same selection twice
— once with the work stashed — and `comm` the two failure lists. Every claim of
"no regressions" made this week rests on that and none of it would survive
without it. It found two failures that were mine and cleared five that were
not.

**A CACHED IMPORT AND THE IMPORT ARE TWO PIECES OF STATE.** `monkeypatch`
restores what it was asked to restore. It does not restore a module-level
`_ZERNIKE_AVAILABLE` filled while a fake package sat in `sys.modules`, and it
does not restore a module object deleted by a reload. Both poisoned the whole
process from one file; instruction 346 has the bisect that found them.

**MEASURE A WIDGET ONLY AFTER THE LAYOUT SETTLES.** One
`app.processEvents()` after `show()` is not enough — widths are still
pre-layout defaults. A clipping sweep run that way reported 38 problems in
German where there are none. Pump until the geometry stops changing.

**AND ASK A WIDGET WHAT IT IS PAINTING, NOT WHAT IT HOLDS.** A control that
elides on purpose reports its full caption from `text()`. Comparing that to
its width reports clipping by construction. `displayed_text()` exists for
this.

**TWO MEASUREMENTS THAT DISAGREE ARE WORTH MORE THAN ONE THAT LOOKS RIGHT.**
Both clipping retractions were caught that way, not by re-reading the code.

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

* Branch `nightly`. SUPERSEDED 2026-09-02: the file list and ownership that
  used to be here were true in August and are not now.
  **`instructions/open/325` sections 1 and 2 are the current answer** to which
  tree is whose and which files are whose, and it stays current because both
  sessions write to it.
* THERE ARE TWO LIVE TREES, one per session:

      Claude   /mnt/firecuda2/Claude/repo/spacr
      Codex    /mnt/firecuda2/codex/repo/spacr

  Confirm with `git worktree list` before your first commit. An older version
  of this section called the Claude tree a stale mirror; that stopped being
  true in August and cost a session an hour of confusion on 2026-09-01.
* `spacr-nightly` at `/home/olafsson/repo/spacr-nightly` IS still stale
  (last commit 2026-07-26). Line numbers quoted from it will not match.

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

### 3f. `pip -e` POINTED AT THE WRONG CHECKOUT, AND A CHECK CAN'T SEE IT

FOUND AND FIXED 2026-08-18. `pip show spacr` reported

    Editable project location: /mnt/firecuda2/Claude/repo/spacr

which is the STALE MIRROR (§2), frozen at commit `90714c9e`, while all work
happens in `/mnt/firecuda2/codex/repo/spacr`. Consequences, both real:

  * `spacr` on PATH launched dead code. The maintainer had the GUI open all
    day against a tree fifteen commits behind, which makes "I still see the
    bug" impossible to interpret.
  * ANY CHECK RUN AS `python /some/other/dir/script.py` VERIFIED THE MIRROR.
    `python script.py` puts the SCRIPT's directory on `sys.path` and never
    adds cwd, so it falls through to site-packages and the editable finder
    answers. Four such checks ran before this was noticed.

Fixed with `pip install -e /mnt/firecuda2/codex/repo/spacr --no-deps
--no-build-isolation`. Verify after any env change:

    cd <live tree>  && python -c "import spacr; print(spacr.__file__)"
    python /tmp/anywhere/check.py        # <- the one that used to lie

WHAT DOES *NOT* SAVE YOU, measured rather than assumed: setting PYTHONPATH is
not the general fix and neither is trusting cwd. On this interpreter
`sys.meta_path` is

    [DistutilsMetaFinder, PynvmlFinder, BuiltinImporter, FrozenImporter,
     PathFinder, _EditableFinder, _EditableFinder]

so `_EditableFinder` sits AFTER `PathFinder` and cwd/PYTHONPATH DO win here --
but that ordering is a setuptools implementation detail, not a guarantee, and
a peer session had a recorded incident from a repo where a `git worktree`
control silently tested current code for this family of reason.

THE RULE THAT SURVIVES BOTH: ASSERT THE RESOLVED PATH INSIDE THE CHECK.

    import spacr; assert "/codex/repo/spacr/" in spacr.__file__, spacr.__file__

A check that prints its own `__file__` cannot lie about which tree it read.
One that trusts its invocation can, and did.

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

> **SUPERSEDED — this table is an August snapshot.** Several of its items are
> closed and fourteen more have been filed since. `instructions/00_INDEX.txt`
> is regenerated from the folder and carries a percentage on every open row;
> read that instead. The table is kept because its one-line characterisations
> of 52, 95, 94 and 47 are still the best short descriptions of what those
> items were about.

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

* **CHECK THE GITHUB ISSUES AT THE START OF EVERY SESSION, AND AGAIN
  PERIODICALLY WITHIN IT.** Set 2026-08-17. This is not an instruction that
  can be finished, so it deliberately has no number in `open/` to be moved to
  `done/` -- it is a recurring check, and its home is here because this file
  is what a session reads first.

      gh issue list --repo EinarOlafsson/spacr --state open

  Read each one, fix what is fixable, and reply on the issue saying what was
  done and in which commit. An issue that is a duplicate, a question, or a
  decision for the maintainer gets said so on the issue rather than left
  open in silence.

  Auto-filed issues carry a traceback fingerprint and the pipeline settings,
  so they are usually reproducible without asking the reporter. TWO THINGS
  THEY ALSO DO, both seen on the first one checked (#108):
    - the title names the CRASH, but the body often reports a DIFFERENT
      problem the user hit first. Read the prose, not just the traceback.
    - paths are redacted to `<PATH>` / `<DB>`, so the shape of a path is
      evidence even when its content is not -- `~<DB>` is a tilde that was
      never expanded.

* A feature goes into `instructions/open/NN_slug.txt` **before** it is coded,
  quoting the request in a `Requested:` line. Merge overlapping asks into the
  first task; do not file duplicates.
* Print the done/left table whenever an item is finished. Refined
  2026-08-17: show it EVERY time an item reaches 100% and every time
  the maintainer adds one, unprompted -- one row per item with a
  percentage, grouped by the instruction that owns it, and an overall
  figure underneath. Keep the rows honest: work a background agent has
  built but not committed is not 100%.
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


## 8. The header this file used to carry (August 2026)

Preserved because this file's own convention is that older sections stay as
history. It describes the tutorial release and the instruction-83 catalog
freeze; its counts are the ones the 2026-09-02 header replaced.

---

# Handoff — 2026-08-15

Written for whoever picks this up next, human or agent. It records what is
true right now, what needs the maintainer, and the traps that cost time so
they cost nobody else any.

Read this, then `instructions/00_INDEX.txt`, then the open instruction you are
taking. The authoritative status is the current ledger and each instruction's
latest dated record; older sections below are preserved as history.

Current checkpoint: branch `codex/tutorial-api-final`, 92 done / 10 open after
this closeout. The complete tutorial release has 73 lessons, 508 purposeful
scenes, eight languages, 50 voices, 3,650 strict-freshness narration tracks,
and 73 4K silent masters. Its reusable audit skill, frame sampler, live
verifier, and tests are committed. Two hundred new audio/timing pairs and four
masters were uploaded to the existing Hugging Face release surface; the
tutorial commits are ``4caa7db1`` and ``6aeb6693``, and the matching main
publication change was merged through PR #105 at
``ea0d96b7d6f545bae8f73c1a7af2460f8457979a``.

Instruction 83 is complete on the current source freeze: all nine API
catalogs are current at 6,655 symbols, all nine runtime catalogs are current
at 3,678 entries, and every installer catalog is current at 57 strings.
Coverage, exact source-bound review evidence, signature/placeholder guards,
and the English manifests were regenerated together. Instruction 108 records
the bounded human review and the explicitly named mechanically checked
remainder. The coherent catalog/evidence/test closeout is commit
``16ee2065``.
Instruction 99 adds first-class CV-model explanations and regression-hit to
candidate-cell investigation with guarded provenance and quantitative evidence.
Instructions 58 and 82 are closed in the immutable pre-rewrite ledger: the
1.5.0.5 release remains canceled, while the approved contributor-history
rewrite and green post-rewrite CI are the final external operations. Their new
SHAs and run IDs are intentionally reported outside the repository because the
history instruction forbids a later commit.

The remainder of this handoff contains historical investigation notes. Its old
69-done / 17-open count and hands-off ownership table are superseded.

---
