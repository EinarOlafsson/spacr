Reproducibility manifests
=========================

Every pipeline launched from the Qt application, the classic GUI, or
``spacr-run`` creates a run folder below ``~/.spacr/runs``. Recording happens
inside the pipeline worker, so inspecting and hashing a large plate does not
block the desktop event loop.

Each folder contains:

``settings.json`` and ``settings.csv``
   The complete resolved settings used by the pipeline.

``manifest.json``
   A versioned, atomically written record of the module, timestamps, status,
   settings hash, declared random seeds, Python/NumPy/Torch random-state
   identifiers, spaCR and Git versions, all installed package versions, model
   hashes, input hashes, output hashes, warnings, and an exception traceback.

``log.txt``
   The tail of the application log at completion.

``outputs/``
   Artifacts explicitly attached by pipeline code.

File provenance
---------------

spaCR recursively discovers existing paths in settings, including paths nested
inside plate lists. Every regular input file receives a full SHA-256 digest,
size, modification timestamp, and the setting key that selected it. Files that
are created or modified under those roots during the run are recorded as
outputs. Symlinks, version-control folders, caches, and the run journal itself
are excluded.

The manifest also includes deterministic aggregate ``input_tree_sha256`` and
``output_tree_sha256`` values. These make it cheap to establish whether two
complete sets match while retaining the per-file records needed to locate a
difference.

Crash and failure behavior
--------------------------

A ``running`` manifest is written before the pipeline starts. It is replaced
atomically when the run succeeds or fails. Exceptions are re-raised to the
normal GUI/CLI error handling after their traceback is retained. Problems
reading or hashing provenance are logged and listed under
``provenance_warnings``; they are not silently discarded.

Public API
----------

Use :func:`spacr.run_journal.open_run` around a custom pipeline. Within the
context, :meth:`spacr.run_journal.Run.record_input`,
:meth:`spacr.run_journal.Run.record_model`, and
:meth:`spacr.run_journal.Run.record_output` can add paths that are not present
in settings.

.. code-block:: python

   from spacr.run_journal import open_run

   with open_run("my_assay", settings) as run:
       run.record_model("classifier", settings["model_path"])
       result = run_assay(settings)
       run.record_output(result)

``spacr repro <run-folder>`` replays supported modules with the recorded
settings. The complete API is generated under :mod:`spacr.run_journal`.
