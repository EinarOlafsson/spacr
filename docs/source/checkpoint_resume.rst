Checkpoint and resume
=====================

spaCR checkpoints long work only after a safe unit has finished. A checkpoint
never means killing a write halfway through: it means that a later invocation
can prove which fields, trials, or plate jobs settled and continue after them.
The Qt **Stop** button uses the same boundaries: it requests cooperative
cancellation, lets the current safe unit finish, and retains the checkpoint for
the next run. See :doc:`threading_cancellation_audit`.

Supported workflows
-------------------

Mask
   Enable ``resume`` in Advanced settings. Complete mask and merged ``.npy``
   fields are structurally validated. Missing, empty, or truncated arrays are
   regenerated, and new mask arrays are written with temporary-file plus atomic
   replace semantics.

Measure
   Enable ``resume`` in Advanced settings. A field is skipped only when every
   Measure-owned table in ``measurements.db`` is complete. Partial field rows
   are cleared in one transaction before remeasurement, while tables owned by
   conversion, alignment, or other modules are never deleted.

Format Converter
   Enable the Apple-style **Resume** switch. The converter writes
   ``.spacr_conversion.checkpoint.json`` after a whole field is complete.
   Resume reopens each target's TIFF metadata before accepting the field and
   atomically repairs a missing or corrupt target. Source identity and the full
   mapping plan must match.

Image UMAP search
   Open **UMAP settings**, then enable **Resume checkpoint**. Trial metadata is
   written after every trial and embeddings are stored as adjacent NumPy
   artifacts. Grid searches skip completed configurations. Adaptive 2×2
   searches also persist their centre, best score, completed-round count, and
   partial-round corners, so only missing corners run after an interruption.

Batch Runner
   A saved queue is written after every job transition. Loading and resuming it
   leaves successful jobs alone. A Mask, Measure, or Format Converter job that
   was running when the machine stopped is restarted with ``resume=True``, so
   its own verified field boundary is reused. Other jobs restart at the job
   boundary.

Interrupted normalization archives
----------------------------------

Mask also checks the normalized ``masks/*.npz`` archives before segmentation.
It checks ZIP member extents, a readable numeric ``data.npy`` header, enough
declared pixel bytes for the shape and dtype, and one filename per batch
field. This structural check does not inflate every pixel array or perform
a full pixel CRC scan.

Damaged archives are set aside as ``<name>.damaged``. With preprocessing
disabled, the run stops and explains how to recover them. The same quarantine
still prevents a later run from silently omitting those fields. It is resolved
by a valid same-name replacement, or by readable archives covering every
existing field in ``stack/``. Legacy object-valued filename arrays cannot
prove that coverage. Enabling preprocessing lets recoverable fields be rebuilt
from the available source data; valid archives and quarantined evidence are
preserved.

Safety rules
------------

Checkpoint files are atomic JSON documents with a workflow name, format
version, boundary, timestamps, input/settings signature, status, completed
units, and workflow state. spaCR refuses a resume when this signature differs;
start with Resume disabled to create a fresh checkpoint. A corrupt checkpoint
is reported and preserved for diagnosis rather than silently discarded.

The persistence API is :class:`spacr.checkpoint.CheckpointStore`. Conversion
uses :func:`spacr.convert.convert`, UMAP search uses
:func:`spacr.hyperparam.umap_search`, Measure uses
:func:`spacr.resume.plan_measure_resume`, and queues use
:func:`spacr.batch.resume_queue`.
