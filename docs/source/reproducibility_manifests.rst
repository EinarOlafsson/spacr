Reproducibility manifests
=========================

Every pipeline launched from the spaCR application or ``spacr-run`` creates a
run folder below ``~/.spacr/runs``. Recording happens inside the pipeline
worker, so inspecting and hashing a large plate does not block the desktop
event loop.

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

``environment/``
   ``requirements-lock.txt``, a ``pip freeze`` of the Python environment the
   run used, and, when spaCR runs inside a conda environment,
   ``conda-explicit.txt``, the output of ``conda list --explicit``. The
   manifest names both files under ``environment_lock`` together with a
   SHA-256 digest of the environment. Rebuild the environment with
   ``conda create --name rerun --file conda-explicit.txt`` followed by
   ``python -m pip install -r requirements-lock.txt``.

   pip and conda are asked once per environment: the lists are kept under
   ``~/.spacr/env_locks/<digest>`` and every later run with the same digest
   copies them. Installing, removing or upgrading any package changes the
   digest. If the lists cannot be written, the run continues and the reason
   is listed under ``provenance_warnings``.

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

``spacr-repro <run-folder>`` replays supported modules with the recorded
settings. The complete API is generated under :mod:`spacr.run_journal`.
