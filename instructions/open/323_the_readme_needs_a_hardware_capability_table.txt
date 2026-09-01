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
Rows are hardware, columns are tasks. Something like:

  Hardware                        Segment  Train  UMAP/clustering  Mixed models
  NVIDIA (CUDA)                     GPU     GPU        GPU             GPU
  AMD on Linux (ROCm)               GPU     GPU        CPU             GPU
  AMD in an Intel Mac (Metal)       GPU     GPU        CPU             CPU
  Apple Silicon (Metal)             GPU     GPU        CPU             CPU
  Intel Arc/Xe (XPU)                GPU     GPU        CPU             CPU
  No GPU                            CPU     CPU        CPU             CPU

Check every cell against `capabilities()` under a faked backend before
publishing it -- the rows above are the EXPECTATION, not the answer.

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
