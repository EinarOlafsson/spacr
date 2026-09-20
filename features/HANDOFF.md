# Handoff — header rewritten 2026-09-02, paths corrected 2026-09-12

> **PATHS UPDATED 2026-09-12.** `instructions/` became
> `features/new` and `features/future`, and this file -- the one the
> index tells you to read FIRST -- still pointed at the old folder in
> six places, including `00_INDEX.txt` itself. Anyone following it
> landed on a path that does not exist. The counts further down are
> from 2026-09-04 and are NOT corrected here on purpose: the index is
> generated and this file is not, so read counts there.

Written for whoever picks this up next, human or agent. It records what is
true right now, what needs the maintainer, and the traps that cost time so
they cost nobody else any.

Read this, then `features/00_INDEX.txt`, then the open instruction you are
taking. The INDEX is generated from the folder and carries a percentage on
every open row; this file carries the things a generator cannot know.

**THERE IS ONE LEDGER AND IT IS `00_INDEX.txt`.** `instructions/PROGRESS.md (retired, and the folder is now features/)`
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

Two sessions share this repository.
`features/325_two_sessions_one_repo_working_protocol.temp` is the channel
between them and the record of who owns what — **read it before touching
anything**,

> NOT UNDER `open/`, AND NOT A MISFILING. The maintainer took 325 off the
> instructions list on 2026-09-09: it is a standing protocol rather than a
> task that can be finished, so it does not belong in a ledger that counts
> what is done. `build_instruction_index.py` globs the feature folders, so 325 is deliberately invisible to `00_INDEX.txt` and
> `--check` passes without it. Do not "repair" it back into a numbered list. and announce there before editing `setup.py`, `spacr/__init__.py`,
`spacr/schema.py`, `spacr/accelerator.py` or `.github/**`.

NOTHING IS RELEASE-BLOCKING ANY MORE. This paragraph used to name 288, 05,
304, 316 and 01, carried over from instruction 331 which split the old list
into before and after the version bump. That split was abolished on
2026-09-12 when the maintainer replaced the instructions list with
`features/new` and `features/future` and said plainly that NEITHER LIST
BLOCKS A RELEASE. `00_INDEX.txt` has said so since; this file did not, and
since the index tells every reader to come here first, the first thing a new
session was told was the abolished rule. 1.5.0.7 shipped on 2026-09-12 with
several of those five still open, which settles it by demonstration.

What is left of that list is ordinary work: 288 (per-module coverage) and
316 (translations) are open items like any other, 05 and 304 are done, and
01 is code-complete. Section 1 below is the only thing that genuinely gates
on someone other than the session doing the work.

**314 CAME OFF THAT LIST ON 2026-09-04**, closed by the maintainer — "i never
get that problem any more!" — and NOT by a fix: nothing was ever changed
against the stall. Its 2026-09-03 measurement of a 6,252 ms event-loop freeze
opening Regression stands unretracted in `done/314`, along with Home's
unexplained 2.03 → 4.51 s doubling, which was always a separate regression and
is still unowned. If a module feels slow again, read that file before
measuring anything: eleven causes are already eliminated there.

WHAT NEEDS THE MAINTAINER AND NOTHING ELSE — see section 1 — is now short
enough to list here: the measure settings instruction 337 part 3 needs, three
sentences of Spanish, Chinese and Korean for instruction 306, and the nine
hand-written `_ROWS` translations instruction 316 is waiting on.

**THE ZENODO TOGGLE AND THE 1.5.0.5 GO-AHEAD CAME OFF THAT LIST ON
2026-09-14**, and not by being granted — by being checked. Zenodo has been
archiving releases all along (`10.5281/zenodo.22726094`, v1.5.0.7, published
2026-09-12; concept DOI `10.5281/zenodo.21343316`) and conda-forge has had a
live self-updating feedstock since 2026-08-27. Both were listed as blocked on
the maintainer for weeks. **ONE API CALL EACH CLOSED THEM.** Before writing
"blocked on the maintainer" about anything outward-facing, ask the outside
world first — 59 has the full account and the one red test it turned up.

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

## THE MEASUREMENT LESSONS OF 2026-09-10

Four more, and every one cost real time before it paid.

**A RE-IMPORTED MODULE LIVES IN THREE PLACES, NOT TWO.** `sys.modules`, the
module object itself, and — the one nobody restores — **the attribute on the
parent package**. `importlib.import_module` rebinds `spacr.qt.screens` →
`settings_model`, and a fixture that saves and restores `sys.modules` leaves
the two disagreeing. That matters because they are reached by different
syntax: `from .settings_model import X` goes through the ATTRIBUTE while
`monkeypatch.setattr("spacr.qt.screens.settings_model.X", spy)` goes through
`sys.modules`. A spy installed by one is invisible to the other and it fails
SILENTLY — the code under test runs the unpatched original and something
else goes red. Found three times now; `tests/qt/test_zz_a_reimported_module_is_put_back_properly.py`
is the detector. NOTE that `importlib.reload` and `runpy.run_module` are
both SAFE — reload re-executes in place, runpy never touches the package.

**XVFB IS A REAL X SERVER AND IT HAS A GPU PATH.** Two instructions had
written off measurements as needing a physical display. Neither did:
`xvfb-run` gives real windows with real occlusion, and
`gpu_is_available()` returns True under it because Mesa's llvmpipe is a
genuine GLX context. 385's whole remaining search space — "does an X server
stop feeding an obscured native GL surface" — was answerable here in four
minutes. It says nothing about how FAST a card draws; it says everything
about what the X server and the compositor do.

**AND COUNT THE THING THAT IS ACTUALLY DRAWN.** A vispy canvas never
delivers `QEvent.Paint` to the QWidget around it, so counting paint events
on the wrapper reports ZERO for a backdrop that is visibly running. The
first version of that measurement had zero in every column INCLUDING the
baseline, which is the only reason it was caught. Always have a baseline
column; a number with nothing to compare it to is not a measurement.

**REBUILD THE CATALOGS ONCE, AT THE END OF A BATCH OF CODE.** Two new public
docstrings from an unrelated performance fix turned a green API audit red
and moved the symbol count 10,283 → 10,284. A rebuild is tens of minutes per
language; doing it after each change multiplies that by the number of
changes and throws every intermediate result away. Finish the code, then
rebuild, then commit the catalogs. A private name (`_build_the_dialog`) costs
nothing at all — it never enters the manifest.

## THE MEASUREMENT LESSONS OF 2026-09-14

### Adding a protected term retroactively invalidates every reviewed record that translated it

`_PROTECTED_TERMS` is SHARED BETWEEN THE TWO i18n LANES and neither lane's code
says so. The other session protected the twenty UI screen names for a
runtime-lane reason -- English sense expansion was renaming Plate Queue to
"Plate-processing list" before translation. That reached back into hand-written
API evidence and invalidated fourteen records in de, is and pt that had
translated one of those names.

It does not fail softly. `reviewed_api_block_translations` RAISES:

    ValueError: rejected reviewed API target
    'spacr.qt.preferences.get_dashboard_watermark#1'

from `build_documentation_i18n.py:5772`, via `_syntax_preserved` returning
False, about a file nobody edited, in a lane nobody touched. The blast radius
is invisible from the change.

Confirm the cause the way it was confirmed here, rather than reasoning about it:

    git show <before>:tools/build_i18n_catalogs.py | grep -c UI_SCREEN_NAMES   # 0
    git show <after>:tools/build_i18n_catalogs.py  | grep -c UI_SCREEN_NAMES   # 3
    git diff --stat <before> <after> -- tools/build_documentation_i18n.py      # unchanged

The validator had not moved; only the term list had. And the right response was
to fix the fourteen records, not to scope the protection: a German API page
calling a screen "Laufverlauf" sends a reader to something the interface does
not have. The protection did not create those defects, it revealed them.

### Exact English is not always a fallback, and every audit assumed it was

Some rows are made entirely of protected literals: a bare identifier, a caption
whose every word is a product name, a format string of nothing but
placeholders. There is no prose in them to change, so asking a model for a
translation can only damage them. A nine-language rebuild produced:

    'extra_performance'  ->  'extra_performance oder'                  (de)
    'extra_performance'  ->  'extra_performance에 해당되는 글 1건'      (ko)
    'Image UMAP…'        ->  'Image UMAP...'                           (de)
    '[{severity}] {object_type}: {flags}'
                         ->  '({severity}] {object_type}: {flags}'     (de)

The last turns a matched bracket pair into a mismatched one in a user-facing
line. NOT ONE AUDIT COULD SEE ANY OF IT, because every coverage and
exact-English check asks whether the target DIFFERS from the English. All four
differ. All four scored as translated, and the damage RAISED the coverage
number.

`_IDENTITY_TEXT` is this rule written out by hand and its own comment records
the failures it was written for (`viridis` to Korean "virus", `slurm` to German
"mud"). Enumeration cannot keep up: every new identifier-shaped caption
reintroduces the bug until a human happens to notice. The derived half now
lives in `_reviewed_translation`.

ORDERING IS LOAD-BEARING. `{location}: {path}` is also entirely protected AND
carries a deliberate hand-written record in all nine locales, where a reviewer
added the word path/Pfad/chemin/sökväg to make it readable. Ask the identity
question before reading the records and all nine are discarded. It was nearly
filed as a fourth instance of damage.

### A probe that is wrong in the safe direction still lies, and it lies quietly

The identity rule was measured twice wrong before it was measured right. Both
times the probe was fed the wrong string, and both times it produced a
confident, plausible, actionable number:

  * `getattr(module, "UI_SOURCES", {})` against the LOCALE catalogs, which name
    that table `UI`. The `{}` default made `ui.get(s, s) == s` true for every
    row, reporting "0 translated, 304 English" for all nine locales -- a total
    failure of tonight's entire rebuild. Believed, it would have triggered a
    pointless re-run of everything.
  * the predicate applied to `SETTING_LABELS` KEYS (`adjust_cells`) rather than
    to the English source, which is the VALUE (`Adjust cells`). It said 1,033
    rows were affected and 1,920 German labels would be reverted. Believed, the
    fix would have been abandoned as far too dangerous.

The true number was FOUR. Neither probe raised anything; both returned a number.
WHEN A MEASUREMENT SAYS THE WORK FAILED COMPLETELY, OR THAT IT IS FAR MORE
DANGEROUS THAN EXPECTED, CHECK WHAT YOU FED IT BEFORE ACTING ON IT. The same
mistake appeared a third time in a test written the same hour -- the sweep in
`test_a_row_with_nothing_to_translate_is_left_alone.py` looked English values up
as if they were keys, and `seg_qc` collided, reporting a German setting LABEL as
damage to an identifier.

### Wait on the process, not on its wrapper

`systemd-run --user --scope bash -c '...'` gives at least three PIDs: the scope
unit, the shell, and the python process. `kill -0` on the wrapper reported the
rebuild "finished" while it was still translating French, 264 of 304. The log's
own "REBUILD EXIT=" line had not been written yet, which was the giveaway. The
script then ran a SECOND python for the audit, so even the first python exiting
did not mean the script was done.

    pgrep -f 'build_i18n_catalogs' | head      # find the real one
    while kill -0 $PID 2>/dev/null; do sleep 20; done

This is the fourth recorded instance of reading a background job's state from
the wrong handle. See also "Do not read a background job's log until the process
has exited" below. And `pgrep -f <pattern>` still matches the shell that runs
it: a waiter looking for `sweep3.sh` matched itself and reported a stopped sweep
as alive.

### Do not let the tree move under a sweep -- including your own commits

A clean full-suite sweep was running while eight agents edited the same
worktree. It had 13,514 tests green over 12 batches and its remaining 35
batches would have been uninterpretable, because any red could equally have
been a real defect or an agent mid-edit. It was stopped rather than finished.
FIVE HOURS OF CPU FOR A RESULT THAT CANNOT BE TRUSTED IS NOT CHEAPER THAN
STOPPING.

The rebuild that WOULD have moved the tree was run in a separate worktree
instead (`git worktree add -b i18n-rebuild <path> origin/nightly`), which costs
one checkout and removes the conflict entirely. Do that whenever a long
verification and a large mechanical change want the same tree.

## THE MEASUREMENT LESSONS OF 2026-09-13

### Sixty-two ledger files were wrong about their own state, and the ledger had already said so four times

The index header warns about this. So does `69`, whose heading reads
"2026-08-13 - DONE. THE FILE'S 'not started' WAS WRONG, AND THIS IS THE
SEVENTH". So does `119`: "Filed to done/ after re-reading the file rather than
its header, which still said 'not started'." All three diagnoses were correct,
were written down, and were followed by a month in which the header did not
change. **Noticing was never the scarce step.**

`tests/test_a_ledger_file_does_not_contradict_itself.py` now fails a file whose
Status OPENS with a not-started claim while its trailing notes declare
completion. Fifteen files were found by reading; WIDENING THE PATTERN to the
ledger's own `2026-08-13 - DONE` heading form found five more in one run,
including 69 itself. Only one direction is checked, and the docstring measures
why: the reverse sweep returns 31 files and nearly all are healthy.

### Two guards were reading folders that no longer exist

Both passed for four weeks because a missing directory globs to nothing and an
empty set satisfies almost any assertion.

* `test_the_working_folders_do_not_reach_main` checked a prefix its fixture
  never creates.

AND THE SIZE UNDERSELLS THE SOURCE-INSTALL ONE. 15 MB is modest beside the
420 MB of `docs/` the same list excludes. The cost is not the megabytes: it is
that 396 internal ledger files would land in a user's install -- files that
quote the maintainer's requests verbatim, name unfixed defects, and argue about
what not to build. `!/features/` keeps working notes out of a distribution, and
that is what the exclusion is for; the disk saving is incidental.
* `test_duplicate_instruction_numbers_are_ordered_by_filename` called
  `_entries("done")` and compared an empty list to its own sort, while
  `features/new` held 390 rows and eight duplicate ids.

**Assert the subject is non-empty before asserting anything about it.** The new
ledger check does this, and so should anything that locates its subject by
path.

### A green build is not a green audit — demonstrated again

Both catalog lanes rebuilt cleanly (API 10,401 -> 10,478 symbols, runtime
1,040 -> 1,050 settings, +N/-0 in all nine languages). The closing audit then
found Map Barcodes' new prose untranslated in EIGHT of nine languages,
Icelandic worst at 37 blocks of copied English. The repair pass is a separate
run and is the one that makes it true.

### Ratchets: subtract, never bump

Map Barcodes moved six. Each was re-measured by diffing against the baseline
commit, and three digests were moved by recomputing with the new entries
removed and checking the old pin returned byte for byte. That caught what a
total hides: parameters moved +129 while 63 new callables carry 127 — the other
two were `barcode_set` added to two EXISTING functions, a different event
entirely.

*The trap:* my first subtraction failed because I rebuilt the changed lines
field by field and forgot one field. A mismatch then reads as "something
unexplained moved" when what moved was the reconstruction. **Subtract using the
recorded baseline line, never a rebuild of it.**

### `import spacr` here runs code from four days ago

The editable install points correctly at
`/home/carruthers/Documents/repo/spacr`, and that checkout is 2,505 commits
behind `origin/nightly`. Trap 3f tells you to check where the install points;
that check passes. See §3f for the one-line version that also reads the
resolved location's HEAD.

---

### A refactor repairs the half of a pair that goes red, and never sees the half that goes green

Instruction 380 moved the composed stylesheet off the `QApplication` and onto
each top-level window. Commit `94f590e0b`, in that same lane, contains

    -    assert "registered widget QSS: FieldFade" in qapp.styleSheet()
    +    assert "registered widget QSS: FieldFade" in theme_mod.window_stylesheet(qapp)

Two hundred lines earlier in the SAME FILE sits the same assertion with the
opposite polarity — `not in qapp.styleSheet()` — and it was not touched. The
positive one went red when the sheet moved; the negative one became "no marker
is ever found in the empty string" and stayed green. Same file, same afternoon,
same person.

**Every `x in value` a refactor breaks has an `x not in value` somewhere that it
did not.** They are one assertion seen from two sides, and only one of them is
on the screen at the end of the day. When a move turns a test red, GREP THE
SAME FILE FOR THE SAME READ before fixing it.

`tests/test_perf_guard.py` was the other victim and shows the cost: of its three
stylesheet assertions, one failed and two passed vacuously for two days. It was
found by a full-suite sweep, not by anyone looking at the theme.
`tools/which_assertions_went_vacuous.py` is the instrument, and takes
`--accessor module:Class.method` so it is not about stylesheets.

### Name the containers your search recognises, or the count is a property of the grep

`layout` and `measure` were reported as settings that nothing reads. Both are
read, in four modules, through

    resolved = default_settings(settings)

so the settings dictionary is bound to a local called `resolved` by the time a
key comes out of it. A search for `settings[...]`, `settings.get(...)` and
`settings.setdefault(...)` sees none of it. Measured: **four modules, 84 reads,
50 distinct keys** — and the four are `convert`, `align`, `foreign` and
`external_masks`, the pipeline entry points such an audit exists to check.

That is the third instrument in one item to have its own blind spot reported as
a property of the code: `expected_types` cannot see keys registered inside a
function body (item 397), the tooltip dict cannot see keys registered from a
module body (trap 3c), and a grep cannot see keys read out of a renamed local.
In all three the first reading was "these settings do not exist".

### Two probes that disagree is a gift; one probe is believed

The vacuous-assertion sweep above first reported **41 of 55 sites vacuous**. The
true number was 11 and the 41 were the healthy ones: the counter stored
`[full, empty]` and the reporting script unpacked `(empty, full)`. Every row was
inverted, the summary was dramatic, and nothing about it looked wrong.

It was caught because a second probe on the WRITE side said 72,979 characters
were installed and never cleared, against a read probe insisting the value was
always empty. Both could not be true. **A measurement that cannot be checked
against a differently-shaped measurement of the same thing should be reported
with that stated.** The counter is now a named dict, because a key cannot be
unpacked backwards.

## 0. THE FOUR LESSONS. READ THESE BEFORE YOU TOUCH ANYTHING.

### 0a. Audit before you build — SIXTY-NINE files have been wrong about themselves

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

**THERE IS NOW A CHECK, added 2026-09-13, and the count above went from eight
to sixty-nine the day it was written.** (Sixty-two corrected on 2026-09-13 plus the eight already tabled, LESS `69`,
which appears in both: it was wrong about itself in August, was listed here as
an example of the problem, and was STILL wrong this morning.)

`32` is the sharpest of them: its own
heading reads "2026-08-17 - CLOSED. THE WORK LANDED; NOBODY WROTE IT DOWN
HERE", above a Status line that still said "not started" a month later, and a
re-audit on 2026-08-26 passed without touching it either. `69` is in the table
above as an
example from August; it was STILL saying "not started" on 2026-09-13, above its
own heading reading "2026-08-13 - DONE. THE FILE'S 'not started' WAS WRONG, AND
THIS IS THE SEVENTH". `119` closes with "Filed to done/ after re-reading the
file rather than its header, which still said 'not started'" -- and did not
change the header either.

So this lesson has been learned, written down, and re-learned at least three
times, which is the argument that reading is not the fix.
`tests/test_a_ledger_file_does_not_contradict_itself.py` fails any file whose
Status OPENS with a not-started claim while its trailing notes declare
completion. SIXTY-TWO were corrected; five of those were found only by widening
the pattern to the ledger's own `2026-08-13 - DONE` heading form, after
thirteen rounds of reading had missed them.

THE MOST EXPENSIVE SHAPE IS NOT "DONE LABELLED NOT STARTED". It is `70`,
whose header read "under investigation 2026-08-10; findings go here" over a
file containing both the measurement and "CLOSED BY THE MAINTAINER. NO GPU
OFFLOAD, NO NEW DEPENDENCY", written the same day. A mislabelled done item
costs a reader minutes. A header inviting an investigation that has already
produced a decision costs somebody a day, and the decision was the
maintainer's.

FOUR PHRASINGS, AND THAT IS THE FINDING RATHER THAN THE COUNT. This ledger
closes an item in at least four ways, and a sweep keyed on any one of them
finds a fraction of the total:

    a Status corrected off "not started"      caught 20
    a dated `2026-08-13 - DONE` heading       caught  5 more, in one run
    a Status still reading "filed <date>"     caught 18 more
    a dated `-- CLOSED` or `-- SHIPPED`       caught 11 more
    plus `CLOSING.` and `INSTRUCTION n IS COMPLETE`

I learned them one at a time, over five passes, each time believing the
previous sweep had been exhaustive. If you are looking for these, search for
ALL of them at once -- the query is in the check's docstring.

It is deliberately narrow and says so: only an unmistakable completion sentence
counts, it checks one direction (the reverse sweep returns 31 healthy files),
and it shares the index's blind spot on the eleven files at the top of
`features/` that 398 is about.

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
| ~~**81**~~ | ~~The reporter's `df -T` on issue #15~~ — **NOT NEEDED, checked 2026-09-13.** #15 was closed 2026-08-14 and 81's own header records it "fixed across local and shared filesystems in 13c41543", so the answer stopped mattering: both branches are covered. | — |
| ~~**81**~~ | ~~A stack trace for the remaining native crash~~ — **NOT NEEDED, checked 2026-09-13.** That is #72, closed 2026-08-12. The repository has **0 open issues and 86 closed**. | — |
| **44/45/53** | A macOS host, a Windows host, and `makensis`. The Linux halves can be done without them. | — |
| ~~**59**~~ | ~~conda-forge accounts~~ — **NOT NEEDED, checked 2026-09-14.** `conda-forge/spacr-feedstock` has existed since 2026-08-27 with `einarolafsson` as maintainer, and its autotick bot `[bot-automerge]`d v1.5.0.6 and v1.5.0.7 by itself. spaCR is published. | — |
| **93** | Whether a stack whose intensities exceed the 16-bit ceiling should be **refused** rather than silently rescaled. See §4. | a decision |

**RAISED OVERNIGHT 2026-09-15.** Each of these is an item that is finished
except for one question, so an answer converts directly into a closed item.
None is blocking anything else.

| # | Question | Cost to answer |
|---|---|---|
| ~~**288**~~ | ~~Is 100% per-module coverage still the goal?~~ — **ANSWERED AND BUILT 2026-09-19.** "Change the 100% goal to 90%. So the goal is 90% for all modules, and if the module would benefit from more than 90% coverage, implement that, but only in cases where coverage is useful." The gate is now a 90% FLOOR plus the no-regression ratchet — a module's bar is `max(90%, what it already has)` — in `tools/verify_module_coverage.py` (`COVERAGE_FLOOR_PERCENT`, report schema v4), with named reasoned exemptions in `tools/coverage_floor_exemptions.txt` instead of pragmas. **The count was the wrong count:** 175 was how many modules are below ONE HUNDRED percent; below NINETY it is **29 of 566**, recovered from the ratchet's own baseline without a full coverage run. 402 are already at 100%. See 288's last section for the list. | — |
| ~~**293**~~ | ~~Should the Graph Builder's inference defer to the setting?~~ — **ANSWERED AND BUILT 2026-09-15 (b3d9d0a9a).** The answer taken was: defer to the setting when the setting names a chart the data can actually take, and say so when it cannot. `resolved_kind` is now pin → setting → inference, and `describe()` appends the reason whenever the setting was not followed, so a user who set a default and did not get it is told why rather than left to guess. | — |
| ~~**289**~~ | ~~Give the `control_chart` "Control is" list a `settingKey`, or leave it as the one exception?~~ — **ANSWERED AND BUILT 2026-09-15 (0b8d51dec).** The narrow fix was taken: the list got `settingKey` and `settingsAppKey` rather than widening `_is_a_settings_field`, which would have put all 45 screens at risk for one row. | — |
| ~~**368**~~ | ~~Build the nested-helper API pages, or record it as dropped?~~ — **ANSWERED 2026-09-15: neither.** Measured first, which changed the answer: it is 842 helpers and 1,684 translatable blocks across nine locales, several times the whole night's translation batch. The maintainer's answer was "don't work on it now", and it is filed as **411**, now assigned to Codex. 368 points at it. | — |
| ~~**403**~~ | ~~May the search write into the form without a click?~~ — **ANSWERED 2026-09-15: "Live search, Apply by click".** The search re-runs on a settings change like Live preview; the form changes only on Apply, so nothing typed is overwritten. Filed in 403 with the debounce requirement, and it agrees with what `apply_proposal`'s docstring already said. | — |

---

## 2. State of the tree

* Branch `nightly`. SUPERSEDED 2026-09-02: the file list and ownership that
  used to be here were true in August and are not now.
  **`features/325_two_sessions_one_repo_working_protocol.temp`
  sections 1 and 2 are the current answer** to which
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
* **THE THREE PATHS ABOVE ARE MACHINE-SPECIFIC, checked 2026-09-13.** On the
  host `carruthers` none of them exists -- there is no `/mnt/firecuda2` at
  all -- and the live checkout is:

      /home/carruthers/Documents/repo/spacr        (branch nightly)

  with agent worktrees under `/tmp/claude-1000/<repo>/<session>/scratchpad/`.
  The advice in this section is still right and is the part to keep: RUN
  `git worktree list` BEFORE YOUR FIRST COMMIT. The absolute paths are not
  portable between the maintainer's machines and should be read as an example
  of the shape rather than as somewhere to `cd`.

### The environment

* CI cells are already local conda envs: `spacr12` (3.12, numpy 2.4.6,
  sklearn 1.9, cellpose 4.2.1.1), `spacr13`, `spacr14`. Default `spacr` is
  3.10 / sklearn 1.7.2. Failures that "do not reproduce locally" reproduce in
  `spacr12` in seconds.
* Qt tests need `xvfb-run -a`. `-p no:randomly` for anything order-sensitive.
* Max 16 CPU cores, max 4 concurrent subagents.

---

## 3. Traps

### 3.00. THE NAS CAN HANG A WHOLE SESSION, AND `timeout` DOES NOT SAVE YOU

**Set 2026-09-20, after a session stopped responding and the maintainer had
to restart it by hand.** It was waiting on the NAS. Nothing it had built was
lost -- every branch it owned was already pushed -- but the session was gone.

`/nas_mnt` is NFSv3 mounted `hard` (`timeo=600,retrans=2`, automounted with
`x-systemd.idle-timeout=600`). A hard mount does not fail when the server
stops answering. It blocks, with no deadline of its own, in uninterruptible
sleep, and **SIGKILL does not reach a process in that state**.

THE PART THAT SURPRISES PEOPLE: `timeout 30 ls /nas_mnt` hangs just as
badly. After its deadline `timeout` signals the child and then WAITS for it
to die, and the child is never going to die. `timeout -k` is the same story
with an extra signal nobody receives. Every guard that ends in "and then we
wait for it" is no guard at all.

WHAT TO USE INSTEAD. `tools/nas_guard.sh`, which never waits on the process
that touches the mount -- it runs the probe detached, in its own session,
and reads the answer from a file against a deadline:

    tools/nas_guard.sh check /nas_mnt 5     # 0 answers, 2 did not, 3 not there
    tools/nas_guard.sh run 60 ls /nas_mnt/some/plate
    tools/nas_guard.sh paths                # the mounts to treat this way

Giving up abandons one sleeping process. The kernel reaps it if the server
comes back; it does not cost the session. `tests/test_a_stalled_mount_cannot
_hang_a_session.py` holds all of this, including the rule that the script may
not be "simplified" into a `timeout` call.

AND DO NOT SWEEP THE MOUNT. `find`, `grep -r`, `du` and `ls -R` over a path
that includes `/nas_mnt` are the usual way in. Exclude it, or run the sweep
through the guard.

### 3.0. AN EXIT CODE IS A CLAIM ABOUT THE CHECKS, NEVER ABOUT THE CONTENT

**Read this before you run any repair pass, 2026-09-14.** Between them the two
audits check key coverage, source hashes, protected literals, markup, English
residue, a fixed false-friend list and target script. Neither of them reads. A
repair that satisfies every check can still change what a row MEANS, and the
exit code will not move. **So diff the rows against `origin/nightly` and revert
every one the audit did not require.**

**The face that looks wrong the moment anyone reads it: English substituted.**
`--repair-invalid-only --languages zh_CN ko` took 85 invalid rows to 17. Among
the 68 it "repaired" was the UI row `Filter rows — type a gene, a guide,
anything in the table`, which is `'筛选行 — 键入基因、引导 RNA或表格中的任何内容'`
at HEAD (the `UI` table of `spacr/qt/i18n_catalogs/zh_CN.py`) and came back as
the English source verbatim. A following `--repair-untranslated` then
rewrote 3,160 rows and still left 17. Nothing reported any of it; it was
caught by diffing ONE row against HEAD.

**The face that does not, and it is the one that gets committed: fluent
nonsense.** A repair pass rewrote 87 zh_CN rows where CI had named 3, and the
runtime audit exited 0 on all of them:

| key | before | after the "repair" | what the repair says |
|---|---|---|---|
| `pca_whiten` | `白色PCA` | `白色白色` | "white white" |
| `guide_permutations` | `引导 RNA转换` | `导游转换` | "tour guide" |
| `surrogate_model` | `Surrogate 模型` | `超越模型` | "surpass model" |

The last two were reverted and read as "before" today. `pca_whiten` is now
`'Pca 白化'`, which is what the repair should have produced: it drops the
invented literal and keeps the meaning.

THE MECHANISM IS THE PART TO RECORD. The audit had been FAILING the old
`pca_whiten`: the source label is `'Pca whiten'`
(`SETTING_LABELS` in `spacr/qt/i18n_catalogs/en.py`), no uppercase `PCA` in
it, so the old Chinese had INVENTED a protected literal. The repair removed
the invented literal by removing the meaning, and the gate cannot tell those
two apart. Reproducible against the tree as it stands, with
`tools/build_i18n_catalogs.py` imported:

    _syntax_preserved_or_reviewed('Pca whiten', '白色PCA',  'zh_CN')  -> False
    _syntax_preserved_or_reviewed('Pca whiten', '白色白色', 'zh_CN')  -> True
    _looks_degenerate('Pca whiten', '白色白色', 'zh_CN')              -> False

`白色白色` is two repeats and `_looks_degenerate` (`build_i18n_catalogs.py:4879`)
wants four -- `([\u3400-\u9fff]{1,6})\1{3,}` -- before it calls a loop. A row
can get worse in every way a reader cares about and still go green.

**THE COROLLARY, which is not obvious and cost a near-miss: DIFFING THE ROWS IS
NECESSARY AND NOT SUFFICIENT — READ THE ENGLISH BEFORE YOU REVERT.**
`spacr.crops.apply_display_order` in Korean went `RGB의 변환` -> `색상의 순열`,
which reads as a dropped protected literal until you read the source it
translates: `:raises CropError: an order that is not a permutation of rgb.`
(`spacr/crops.py:3538`) — LOWERCASE rgb. The old Korean had invented the
uppercase `RGB`, and 순열 ("permutation") is more accurate than 변환
("conversion") besides. A row that looks degraded may be one that stopped
lying.

Here the GATE was right and the READER was about to be wrong, which is the
reverse of everything above it. Block 5 of that symbol is what `audit()` hands
to `_syntax_preserved` (`tools/build_documentation_i18n.py:6339`):

    _syntax_preserved('an order that is not a permutation of rgb.',
                      'RGB의 변환이 아닌 순서입니다.')  -> False   # HEAD
    _syntax_preserved('an order that is not a permutation of rgb.',
                      '색상의 순열이 아닌 순서입니다.')  -> True    # the "damage"

`_PROTECT_RE` matches a bare `RGB`, so the old row carries a protected literal
its source does not have -- and `build_documentation_i18n.py --audit` says so
today, exit 1:

    ko: 10 API blocks changed protected code/literals
    (spacr.crops.apply_display_order#5, spacr.crops.display_order_indices#3, ...)
    zh_CN: 3 API blocks changed protected code/literals
    (spacr.crops.apply_display_order#5, ...)

zh_CN carries the same invented literal in the same block --
`命令不是RGB的转换。` -- and neither row is new: the Korean has been in the
tree since `dc68f7b0d` and the Chinese since `774b9a7e8`, both 2026-08-14.
Reverting the repair to "restore the literal" would have put a rejected row
back; leaving it alone leaves one there. THE REPAIR WAS RIGHT ON THIS ROW.
Diff every row, then read the English for each one before you decide which way
it moved.

### 3a. A changed public DOCSTRING obliges an i18n rebuild — not a new module

**THE RULE WAS TOO NARROW AND THE SYMBOL COUNT IS BLIND TO THE DIFFERENCE,
corrected 2026-09-14.** This entry used to say "a new public module obliges a
rebuild". A new module is only the loudest case. The obligation is on the
SOURCE TEXT: any change to a public docstring moves its source hash, and the
catalogs then owe a translation for it.

Lived on 2026-09-13: cherry-picking 372 part 14-I edited existing docstrings in
`ops_layout` and `ops_phenotype`. The API symbol count did not move — 10,531
before and after — and three source hashes did. A rebuild triggered on "new
modules landed" would have missed all three, and the docs audit would have
caught it later and further away.

So the trigger is `git diff` touching a docstring, not `git status` showing a
new file. `python tools/build_documentation_i18n.py --sources-only` writes only
`en.json`; the nine locales follow.

**The new-module case is still true and still costly**: Map Barcodes added
`spacr/barcode_search.py` and with it 77 API entries, which took six tests red
across two files until both catalog lanes were rebuilt.

### 3a-bis. Changing what a CHECKER can see invalidates content written while it was blind

Three instances in one night, 2026-09-13/14, and the shape is stable enough to
predict:

| change | what it invalidated |
|---|---|
| 20 UI screen names added to `_PROTECTED_TERMS` | 8 reviewed API records, in 4 locales |
| an identity rule added to `_translate_batches` | 4 machine rows revealed as damage, incl. a mismatched bracket pair |
| the term boundary widened from `\w` to `[A-Za-z0-9_]` | 87 runtime catalog rows, 85 of them zh_CN |

In all three the failure surfaced FAR from the change and read as bad content
rather than a moved goalpost. In all three the right response was to fix the
CONTENT, not to narrow the checker back — the checker had been wrong, and the
content had never been examined.

**THE COROLLARY, which is the part that costs time if you skip it: any change
to a protect pattern, a term list or an identity rule must be verified with
BOTH AUDITS before pushing.**

### 3a-ter. Four states, not two, when a measurement disagrees with you

Both sessions lost time on 2026-09-13/14 by collapsing these into "regression"
or "no regression". They are distinct and the response to each differs:

| state | what it looks like | what to do |
|---|---|---|
| real pattern, right explanation | fix reproduces and survives a revert | fix it |
| **real pattern, WRONG explanation** | the fix works for a reason you did not give | keep looking; the wrong story will mislead the next person |
| **real observation, NO pattern** | two clean measurements agree, and the thing measured moves on its own | measure again on a quiet machine before believing either |
| no observation | you misread the output | re-read before acting |

Lived examples, all from one night:
* The Slow segfault was blamed on `stall_watch.py` from a partial faulthandler
  dump. The crashing thread is the one labelled **`Current thread`**; the rest
  are merely listed. Right area, wrong thread.
* `_lay_out` was "fixed" against two tests that state the opposite contract on
  purpose. Right defect class, wrong remedy.
* 85 zh_CN rows were blamed on the word "guide" being a protected term. It is
  not one — but "guide" WAS the cause, as the SOURCE TRIGGER for an expansion
  the Chinese makes. Right pattern, wrong mechanism, then talked out of the
  right answer entirely.
* A flowview red reproduced cleanly on two trees and was not a regression at
  all — see item 402. `git bisect` even named a first-bad commit, and that
  commit tests GOOD while its parent tests BAD.

**BISECT IS MEANINGLESS ON A NON-MONOTONIC PROPERTY.** It will still return a
commit, confidently. Before trusting one, test the named commit AND its parent
explicitly; if the answer inverts, the property is flaky and the bisect is
noise with a hash attached.

**A RED RECORDED WHILE THE MACHINE WAS BUSY IS A WORKLIST ITEM, NOT A
FINDING.** Re-check it on a quiet box before it becomes anybody's evidence.


    python tools/build_documentation_i18n.py --audit
    QT_QPA_PLATFORM=offscreen python tools/build_i18n_catalogs.py --audit

NOT with `check_reviewed_api_evidence.py` / `check_reviewed_runtime_evidence.py`.
Those validate reviewed EVIDENCE against its sources; the audits validate the
GENERATED CATALOG rows, and the audits are what `docs.yml` fails on. Verifying
the boundary widening with the evidence checkers alone reported 1,296 records
clean while 87 catalog rows were broken.

~~Forgetting `_SUBMODULES` itself turns every compat-matrix cell red.~~
**NO LONGER POSSIBLE, checked 2026-09-13.** `_SUBMODULES` is computed --
`set(_DOCUMENTED_SUBMODULES) | _submodules_on_disk()` -- so it reads the
directory and cannot be forgotten. The comment above it records why: "a module
that existed but could not be reached through the package, so the names are
taken from the directory whenever there is one to read." The failure that
happened twice was fixed by making the list self-updating, which is the right
shape and is worth copying: THE LIST THAT CANNOT GO STALE IS THE ONE NOBODY
MAINTAINS.

`_DOCUMENTED_SUBMODULES` IS STILL HAND-WRITTEN AND HAS DRIFTED 39 MODULES
BEHIND THE PACKAGE -- 195 named against 234 on disk, `barcode_search`,
`accelerator`, `embeddings`, `infection`, `object_distances` and most of the
`ops_*` family among the absentees. The only test on it is a SUBSET assertion,
which a missing entry satisfies, so nothing reports the drift.

THE MECHANISM LOOKS ALARMING AND THE MEASUREMENT SAYS IT IS INERT. In a
PyInstaller bundle there is no directory to scan, so `_submodules_on_disk()`
returns nothing and `_SUBMODULES` collapses to the documented tuple alone --
`__getattr__` then raises AttributeError for all 39. That is why the comment
calls it "the frozen-bundle floor".

Checked by AST rather than by grep, because grep says the opposite: 28 of the
39 appear as `spacr.<name>` in the sources, and EVERY ONE is a Sphinx
cross-reference in a docstring (`:mod:`spacr.object_distances``). Executable
`spacr.<missing>` attribute access in the package: ZERO. Nothing in spaCR
reaches these modules by attribute at run time; they are imported by path,
which a bundle serves from its archive.

So the floor being 39 behind costs nothing today. It would cost something the
first time a frozen build did `import spacr; spacr.bystanders...`, and it is
cheap to prevent: the same union trick `_SUBMODULES` already uses fixed the
failure that "happened twice".

### 3b. Headless Qt refuses static modals

`QMessageBox.information` / `QInputDialog.getText` raise in tests by design —
`tests/qt/conftest.py` enforces it, because a modal runs its event loop in C++
and hangs the run. `monkeypatch.setattr(QMessageBox, "information",
staticmethod(lambda *a, **k: None))`. Patching `exec` does **not** cover it.

### 3c. `spacr.settings.tooltips` is NOT complete on import

Six pipelines register their keys from their own module via
`register_defaults`, which runs on import of that module. A tool that reads the
dict cold and then writes "no description" is not missing a sentence, it is
writing a **wrong** one. See
`tools/build_notebook_settings.py::_load_registrations`.

**THE EXAMPLE CHANGED, re-measured 2026-09-13.** This trap named `dst` and
`cmap`. Both are documented cold now, so anyone checking the trap against them
would conclude it had gone away. It has not: `tooltips` still grows from
37,247 to 37,260 once the pipeline modules are imported, and the thirteen keys
that arrive late are

    collision_max_distance  exclude_starved_wells  min_reads_per_well
    on_error  on_error_attempts  on_error_backoff  position_effect_ratio
    qc_data  starved_read_fraction  sweep_points  sweep_span
    target_grnas_per_well  target_statistic

SAME MECHANISM AS 397, reached from the other side. That item counts rows in
`SETTING_API_TARGETS` naming a key absent from `expected_types` and finds 76
that are real settings declared where `expected_types` cannot see them --
`collision_max_distance` and `exclude_starved_wells` are on both lists. A
setting registered from its own module's body is invisible to any tool that
reads a module-scope table, and that is one fact with two symptoms.

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

**A LIVE INSTANCE ON THE HOST `carruthers`, measured 2026-09-13 01:33.** The
paths below are from the other machine; the hazard is here too, and it is not
a mispointed editable install this time. `pip -e` points where it should --

    Editable project location: /home/carruthers/Documents/repo/spacr

-- and that checkout is 2,505 COMMITS BEHIND `origin/nightly`, its HEAD being
`cd92bc381` from 2026-09-09 21:47. It does NOT contain the Map Barcodes merge
and does NOT contain 391's settings migration.

So `import spacr` from anywhere that is not a worktree -- and therefore
`spacr`, `spacr-qt`, and any script run from another directory -- gets code
from 2026-09-09. That is the same consequence 3f records: the GUI runs against
a tree days behind, and "I still see the bug" becomes impossible to interpret.

NOT UPDATED BY THIS SESSION, ON PURPOSE. Another session is working in that
checkout at exactly that commit, and pulling 2,505 commits under it would be
the worst kind of help. Whoever owns it should decide. Reported rather than
fixed, which is the same call 398 makes about the eleven files.

THE CHECK IS ONE LINE and is worth running before trusting any GUI observation:

    python -c "import spacr; print(spacr.__file__)" && \
      git -C "$(python -c 'import spacr,os;print(os.path.dirname(os.path.dirname(spacr.__file__)))')" log -1 --format='%h %ci'

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

### 3g. `import ultralytics` REWRITES THE USER'S CONFIG, ON THE FIRST IMPORT

**TWICE ON 2026-09-20, BY TWO RUNS THAT BOTH KNEW THE RULE.** Importing
`ultralytics` writes `~/.config/Ultralytics/settings.json` before any model is
loaded, and when the version it finds is older it prints

    Ultralytics settings reset to default values

and does exactly that. The first import of item 424's rerun hit a file dating
from 2026-07-21; there was no backup and whatever non-default entries it held
are gone. Hours later, correcting that very note, a one-line
`python -c "import ultralytics; print(ultralytics.__version__)"` typed to check
a venv rewrote it again.

THE SHAPE, WHICH IS NOT SPECIFIC TO ULTRALYTICS: a scientific package treats
`$HOME` as ITS state directory and writes there at import, not at first use.
So sandboxing after the import, or around "the real run" only, is too late --
the loss happens in the probe you did not think of as a run.

WHAT ACTUALLY WORKS, in this order:

  1. Put it in the tool, not in the habit. `tools/measure_plaque_detector_
     transfer.py` now sets `YOLO_CONFIG_DIR` to `<out>/yolo-config` before
     anything imports ultralytics, prints where it pointed, and records it in
     `detections.json`; `--yolo-config-dir` is how you ask for the real one.
  2. For anything typed at a shell, put the environment in front of the
     command and not in a file you might forget to source:

         env HOME=$SBX XDG_CONFIG_HOME=$SBX/.config \
             XDG_DATA_HOME=$SBX/.local/share YOLO_CONFIG_DIR=$SBX/ultra \
             python -c "..."

  3. Verify by where the file landed, not by intent: after the run,
     `find $OUT -name settings.json` should find it, and
     `stat -c %y ~/.config/Ultralytics/settings.json` should be unchanged.

This is the second config-wipe incident on this machine in two days. The other
one took six entries out of the maintainer's `~/.config/spacr/qt.conf`.

## 4. Findings filed but not fixed

**93 — RESOLVED, corrected 2026-09-13.** This entry described the intensity
rescale factor as "per field and unrecorded", with all three of its complaints
now answered. Re-read the code before using it; it was stale.

* **Not per field.** `spacr/intensity_rescale.py:build_plate_plan` inspects
  every field as one plate set, and raw-valued fields on a plate SHARE
  `65535 / plate_max`. A per-field decision survives only as a recorded
  fallback for a file that could not be inspected.
* **Recorded.** `measurements.db:intensity_rescale` carries
  `rescale_factor REAL NOT NULL`, `rescale_scope` and `plate_intensity_max`;
  the table is in `schema.FIELD_PROVENANCE_TABLES` and `resume.py` knows it.
* **Not verbose-only.** The warning at `measure.py:3148` is commented
  "Deliberately independent of verbose: this conversion changes the unit
  represented by one stored intensity count", and names the table it was
  written to.

The original observation was right and the fix went in without this entry
being updated, which is the same failure the merge entry below had. When a
section-4 finding is closed, close it HERE too -- an open finding that is not
open costs somebody a second investigation.

**RESOLVED 2026-09-13 — a merge warning the maintainer saw in a real run.**
This entry said "this could be serious and nobody has looked" until
2026-09-13. Somebody had looked, and the answer is written in
`spacr/merge_tables.py` beside the code that fixes it.

```
'plateID':  57170 of 65737 objects disagree between cell and pathogen
'rowID':    57170 …   'columnID': 57170 …   'fieldID': 57170 …
```

TWO CAUSES, BOTH FIXED, both documented where they were fixed:

* `object_label` was in `MUST_AGREE` and should not have been. A cell's
  `object_label` is its label in the CELL mask; a pathogen's is its label in
  the PATHOGEN mask. Two labellings of two objects with no reason to coincide,
  so the warning fired on nearly every row of every healthy screen — and a
  warning that fires on the normal case teaches its reader to ignore it.
* The identity columns were compared against ABSENCE. An uninfected cell kept
  on purpose (`keep_uninfected=True`) has no pathogen row, so every
  pathogen-side column is missing for it. `_columns_agree` now treats EITHER
  side missing as "cannot disagree".

THE SHAPE OF THE NUMBERS IS THE EVIDENCE: plateID, rowID, columnID and fieldID
all disagreeing at exactly 57,170 is what "57,170 cells have no pathogen in
them" looks like, not what "the merge is pairing rows from different fields"
looks like — a mispairing would not hit four columns at an identical count.
`merge_tables.py` records the same shape from a smaller run, 172 of 553, where
172 was exactly the number of cells with no pathogen.

Verified behaviourally on 2026-09-13 rather than read off the docstring: the
uninfected-cell case now reports 0 conflicts, a genuine plate1-vs-plate2
disagreement still reports 1, and `object_label` is no longer in `MUST_AGREE`.

The maintainer's specific run was not re-run — that needs their data — but the
defect it displayed is closed and the numbers fit it exactly.

---

## 5. Where each open instruction stands

> **SUPERSEDED — this table is an August snapshot.** Several of its items are
> closed and fourteen more have been filed since. `features/00_INDEX.txt`
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
| 44, 45, 53 | Installers | Blocked, §1 |
| **59** | conda-forge | **DONE — published since 2026-08-27, bot-maintained.** Verified 2026-09-14 |
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

* **ASK WITH THE QUESTION PROMPT, NEVER IN THE CHAT.** Set 2026-09-04: "if you
  have a question ask me with the question prompt dont ask in the chat." A
  question buried in a paragraph of report is a question that gets skimmed
  past, and this repository's whole intake item (357) exists because answers
  that live only in a chat log are answers nobody can act on next week. Use the
  structured question tool, put the recommended option first, and say what each
  choice costs. This applies to every decision that is genuinely the
  maintainer's -- a release, a retirement, wording he must approve -- and not
  to reporting a measurement he did not ask a question about.

* A feature goes into `features/future/NN_slug.txt` **before** it is coded,
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

Read this, then `features/00_INDEX.txt`, then the open instruction you are
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

### A batched sweep is the only instrument that sees a whole class of defect

2026-09-13. The suite was run end to end in 60-file processes, from both ends
of the alphabet, until all 2,790 files were covered. It found eight defects,
and **every one of them passes when its file is run alone**:

    Map Barcodes' findings table sorted read counts as words (10 before 9)
    fractal travel read process-global QApplication.mouseButtons()
    dialog filters called caplog.at_level on the root, not spacr.qt.app
    test_barcode_search named /home/olafsson/, so it skipped everywhere
    the app-registry restore fixture stopped at tests/qt/
    eight animation verdicts described animations that no longer ship
    the i18n protector rejected correct Chinese and Korean
    a one-line wrapper hid 117 status captions from the extractor

A per-file loop -- which is how most of these files are ever run -- reports all
eight green. **The cost is wall clock and nothing else.**

TWO RULES THE SWEEP ITSELF TAUGHT:

* **THE TREE MUST NOT MOVE UNDER IT.** A sweep running while the catalogs were
  being rebuilt recorded 20 reds that were the rebuild's own half-written
  state. Every one green on re-check. `red.txt` is a worklist; a red recorded
  while its file was being edited says nothing.
* **RE-CHECK EVERY RED BEFORE BELIEVING IT.** Of 27 recorded, 8 were real, 1
  was deliberate, and 18 were contamination or already-fixed rows the
  cumulative log kept.

### Three times in one session, a second measurement caught the first

Worth stating as a habit rather than three anecdotes.

* A `[full, empty]` counter unpacked as `(empty, full)` reported 41 of 55
  assertions vacuous. The truth was 11, and the 41 were the healthy ones.
  Caught by a write-side probe contradicting the read-side one.
* A probe that instruments an accessor failed 16 tests *itself*. Caught by
  running the same files without it: 0 failed, 2,007 passed.
* An i18n regex fix appeared to break 9 tests. Two changes were live at once;
  the other one invalidates catalogs by itself. Isolated, the regex was
  innocent -- 4 failed either way, the same four.

  IN ALL THREE THE FIRST NUMBER WAS PLAUSIBLE AND WRONG, and in two of them it
  pointed at the more dramatic conclusion. A measurement that cannot be checked
  against a differently-shaped measurement of the same thing should be reported
  with that said out loud.

### Do not read a background job's log until the process has exited

Item 83 already carries the neighbouring rule -- "NEVER READ AN EXIT CODE
THROUGH A PIPE" -- after two sessions were fooled by `... | tail` reporting
success for a command that failed. This is the same family and I hit it THREE
TIMES in one afternoon:

* A 66-file probe run was reported as "0 failures" at 39% because the log had
  no `FAILED` line YET. It had one four lines later.
* A catalog repair was reported as "all three locales took it" because the
  grep for the failure string found nothing. The run was at 936 of 2,774.
* A waiter keyed on `pgrep -f 'pytest tests/qt/test_a_dialog'` fired
  immediately, because the file list had been re-ordered and the real process
  matched a different first filename.

  THE SHAPE IS ALWAYS THE SAME: absence of a failure line is read as absence
  of failure, when it only means the job has not got there. A grep over a
  growing file answers a question about the PAST, and the question being asked
  is about the FUTURE.

WHAT TO DO INSTEAD, and the PID matters more than the pattern:

    PID=$(ps -eo pid,args | grep '[p]ython tools/thing' | awk 'NR==1{print $1}')
    while kill -0 $PID 2>/dev/null; do sleep 30; done
    # only now is the log complete
    grep -E 'FAILED|Error' "$LOG"

`pgrep -f <pattern>` is the trap twice over: it matches the waiter's own
command line -- which cost an hour earlier the same night, and cost a `pkill`
that killed six of this session's own waiters -- and it matches on a spelling
of the command that may not be the one running.

### A header that claims MORE work than exists is invisible to every gate

Two items in one afternoon, both found by reading rather than by any check:

    377   finished 2026-09-04 with `health_percentage 42 -> 100` measured and
          the commit named; header said OPEN for nine days
    388   all three steps landed and the blocker cleared itself; header said
          OPEN AND ACTIONABLE

Nothing fails when a done thing says it is open. The ledger's consistency guard
checks the other direction -- a "not started" header over a finished tail --
because that one is unambiguous, and its docstring measures why the reverse
cannot be gated: 31 files trip it and nearly all are healthy.

  388 IS THE INTERESTING ONE. Its last blocker was a runtime catalog rebuild
  "that has not landed". The rebuild landed today as a side effect of
  unrelated work -- the `_set_status` wrapper fix carried 117 captions into the
  catalogs and took the two bystander tooltips with them. AN ITEM CAN BE
  UNBLOCKED BY A CHANGE MADE FOR ANOTHER REASON, and nothing tells it. When a
  file says it waits on X, and you have just done X, go and read it.
