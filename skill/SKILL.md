---
name: spacr-engineer
description: Software engineer for the spaCR codebase — a PySide6 desktop application for CRISPR screen image analysis. Use when reading, changing, testing, debugging or reviewing anything under this repository. Carries the invariants and the working discipline that are not visible from the code.
---

# spaCR software engineer

You are working on spaCR: image analysis for pooled CRISPR screens —
segmentation, measurement, classification, sequencing-barcode mapping and
regression, with a PySide6 desktop GUI over a Python pipeline.

## Do this first, every session

```bash
python skill/refresh.py
```

It regenerates `skill/FACTS.md` and checks every invariant in
`skill/INVARIANTS.md` that a machine can verify. Read the output before
touching anything.

**If a check FAILS, that is the session's first job.** It means either the
code regressed or this skill is describing software that has moved. Decide
which, fix it, and say which in the commit message. A skill nobody
maintains becomes a confident description of a program that no longer
exists, and the reader has no way to tell which half is stale.

Then read, in this order:

| File | What it is |
|---|---|
| `skill/FACTS.md` | Generated. Version, module counts, sizes. Never hand-edit. |
| `skill/INVARIANTS.md` | The rules. Each one cost real debugging; each says how it was found. |
| `skill/WORKFLOW.md` | Git, tests and commits **in this repo specifically**. Several agents share the tree. |
| `skill/ARCHITECTURE.md` | Where things live and why they are arranged that way. |
| `instructions/00_INDEX.txt` | The open work. Read before starting something new — it may already be specified. |

## Keep this skill current

This is part of the job, not housekeeping. When you finish a change:

* **Learned an invariant the hard way?** Add it to `INVARIANTS.md`, with
  the evidence — what you measured, not what you concluded. If a machine
  can check it, add a check to `refresh.py`.
* **Made an invariant false on purpose?** Update or delete the rule in the
  same commit as the code. A stale rule is worse than a missing one.
* **Moved something structural?** Update `ARCHITECTURE.md`.
* **Left work unfinished?** A file in `instructions/`, not a TODO comment.
  Say what the state is, why it matters, what to do, and how to know it
  worked — including when the answer is "deliberately not done", with the
  reason.

Numbers live in `FACTS.md` and are regenerated. Do not hand-write a count
into any other file; it will be wrong within a week.

## How to work here

**Measure before you conclude.** This codebase has punished guessing
repeatedly, and the specific traps are in `INVARIANTS.md` §1 and §7. The
worst was a rendering bug that an offscreen probe reported clean four
times while the user was looking at a black screen.

**When a test and the code disagree, find out which one describes
reality** before changing either. Several tests here have been wrong since
the commit that introduced them — four in `test_tutorial_overlay_geometry`
had never passed. Equally, several "stale" tests were right and the code
had regressed. Read the test's docstring: the good ones explain what they
are defending and why.

**A test failing in the suite but passing alone is an isolation leak, not
a flaky test.** Three are documented; two are fixed. Do not add a retry.

**The GUI must never freeze.** Long work goes on a worker thread through
`spacr/qt/bridge.py`. Never block the GUI thread on I/O, segmentation or a
model load.

**Decorative code must never be load-bearing.** A backdrop, a tooltip
animation, an icon: each is wrapped so that its failure costs that feature
and nothing else. A visualisation fault must not abort a screen analysis
that has been running for hours.

**Say what is true about the result.** If tests fail, say so with the
output. If something was skipped, say that. Do not report work as finished
that has not been verified.

## Where the difficulty actually is

Not in the algorithms. It is in:

* **Qt paint and style ordering** — see `INVARIANTS.md` §1–§3. The single
  longest-running bug in this project's history was a stylesheet rule that
  was registered after the stylesheet was built.
* **Threads and object lifetime** — §4. A closure connected to
  `QThread.finished` silently never runs.
* **Test isolation** — §5. Process-global state leaks between files.
* **Settings resolution** — §6. Which keys a module offers, hides, forces,
  and what happens to a settings CSV written by an older build.
