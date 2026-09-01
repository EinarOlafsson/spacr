323 — THE README NEEDS A TABLE OF WHAT EACH HARDWARE CONFIGURATION CAN DO
=========================================================================

ASKED FOR (2026-08-31)
----------------------
"add a table to README that shows what the different supported hardware
configurations can do."

WHY A TABLE AND NOT A SENTENCE
------------------------------
Instruction 319 landed and the README now names every backend -- NVIDIA
(CUDA), AMD (ROCm on Linux, Metal on macOS), Apple Silicon (Metal), Intel
Arc/Xe (XPU). That is the right list and it still leaves the question a
reader actually has unanswered: WILL MY SLOW STEP BE SLOW?

Naming a backend implies everything works on it. Nothing does, quite:

  * segmentation and model training are accelerated on every GPU backend;
  * the cuML reductions -- UMAP, t-SNE, clustering -- are CUDA-ONLY,
    because RAPIDS is. That is not a gap to fill by enabling something;
    there is no non-CUDA cuML to enable.
  * mixed-effects regression needs float64, and MPS REJECTS float64
    outright -- a TypeError, not a slow path. So it is CPU on Metal
    whatever the card.

A user with an Intel iMac should be able to see, without running
anything, that their Radeon takes Cellpose from 444 s to 3.2 s per image
AND that their UMAP will not move. Both facts, in one place.

DERIVE IT. DO NOT TYPE IT.
--------------------------
`spacr/accelerator.py::capabilities()` ALREADY returns
``(task, accelerated, detail)`` for the running machine, and
`spacr/doctor.py` already renders it. That function is the source of
truth and the table must be generated from it, in the way the module grid
now is -- see instruction 321, where a hand-written copy of Home's layout
went stale within a day of Home changing.

The table has one axis `capabilities()` cannot supply on its own: it
answers for THIS machine, and the table is about all of them. So the
generator asks it once per backend with the probe faked -- exactly how
`tests/test_the_accelerator_resolver.py` already tests 19 backends on a
machine that has one. Do not invent a second table of per-task facts;
fake the backend and ask the real function.

SHAPE
-----
FOUR COLUMNS. The mixed-models column was asked for and then removed on
2026-08-31: it is one row of the four that varies for one reason
(float64 on Metal), and a column earns its width by distinguishing more
than one thing.

  Hardware                      Cellpose 4   Torch   UMAP/clustering
  NVIDIA (CUDA)                    GPU        GPU          GPU
  AMD on Linux (ROCm)              GPU        GPU          CPU
  AMD in an Intel Mac (Metal)      GPU        GPU          CPU
  Apple Silicon (Metal)            GPU        GPU          CPU
  Intel Arc/Xe (XPU)               GPU        GPU          CPU
  No GPU                           CPU        CPU          CPU

Check every cell against `capabilities()` under a faked backend before
publishing it -- the rows above are the EXPECTATION, not the answer.

THE LEGEND, under the table:

  green: supported (stable)
  purple: implemented (beta)
  red: not supported

and the CELL TEXT coloured accordingly.

COLOUR CANNOT BE DONE THE OBVIOUS WAY, AND THIS IS MEASURED
-----------------------------------------------------------
RST colour roles do not colour anything on GitHub. Tested:
`.. role:: green` plus ``:green:`GPU``` parses with zero docutils errors
and emits `<span class="green">` with NO inline style. GitHub applies no
custom CSS to a README, so every cell would render in the body colour and
the legend would describe a distinction the reader cannot see. Sphinx
would honour it on the docs site; the README is the surface that matters
here and it would not.

USE COLOURED CIRCLES IN THE CELL, which render identically on GitHub,
PyPI, Sphinx and a plain text editor:

  🟢 GPU     supported (stable)
  🟣 GPU     implemented (beta)
  🔴 CPU     not supported

The legend then names the circle rather than a colour nobody can see, and
carries the same three words.

WHICH CELL IS WHICH COLOUR IS A JUDGEMENT, AND IT MUST BE EVIDENCE-BASED
------------------------------------------------------------------------
Not every "GPU" is the same claim. As of 2026-08-31:

  CUDA          stable. Years of use, and the resolver's tests assert it
                is unchanged by instruction 319.
  Metal on an   implemented AND MEASURED on real hardware -- 444.5 s to
  Intel Mac     3.2 s for one 256x256 Cellpose image. Beta because it is
                one machine and one day old.
  Metal on      implemented, NOT measured -- no Apple Silicon machine has
  Apple Silicon run it. Beta, and honestly so.
  ROCm          implemented, NOT measured -- no ROCm machine available.
  XPU           implemented, NOT measured -- no Arc hardware available.

Do not paint all five green because the code exists. "Implemented" and
"works on hardware somebody ran it on" are different claims, and the
purple tier exists precisely to keep them apart.

THE NO-GPU ROW IS THE TRAP
--------------------------
Its cells say CPU, and colouring them red would say spaCR does not
support running without a GPU. That is FALSE and it is the most damaging
thing this table could claim -- everything works on a CPU, and the
measurements, the regression and every figure are unaffected by any of
this.

So the No-GPU row is GREEN: supported, stable, slower. Say the "slower"
in a sentence beside the table rather than in a colour, because a colour
cannot carry "works fine, takes 139 times longer".

Read the legend as describing the SUPPORT STATE OF THAT COMBINATION, not
the presence of a GPU. Red means "this hardware cannot accelerate this
task" -- which is true of UMAP on Metal, where RAPIDS simply does not
exist for it, and is not true of anything in the No-GPU row.

WATCH
-----
"AMD in an Intel Mac (Metal)" is its own row and must not be merged into
either neighbour. 319's own backend table filed Metal under Apple Silicon
and AMD under ROCm; since ROCm has no macOS build, the configuration that
actually works appeared nowhere, and the fault was not found until
somebody measured a Radeon at 139x. A table that repeats that grouping
tells an Intel-Mac owner their card is unsupported.

Say what CPU means in numbers at least once, next to the table: 444.5 s
against 3.2 s for one 256x256 Cellpose image is the sentence that makes
the table worth reading. It is a measurement from a real machine and
should be labelled as one, with the machine named.

DONE MEANS
----------
* The table is in README.rst, generated rather than typed, and every cell
  agrees with `capabilities()` under that backend's faked probe.
* A test regenerates it and fails if the file disagrees -- the same
  mechanism the module grid uses.
* The nine translated READMEs carry it too. They are generated, so this
  is a regeneration, not nine edits -- but the column headers and any
  prose row are new user-facing strings and belong in the pending
  translation list in instruction 316, NOT machine-drafted.
* No cell claims acceleration the resolver reports as detected-but-not-
  usable. That distinction is 319's, and a table is exactly where it
  would get flattened.
