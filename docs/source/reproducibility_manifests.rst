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
   settings hash, declared seeds, available random-state metadata, spaCR and
   Git versions, recorded package versions, model/file records, warnings and
   an exception traceback when captured. NumPy state and Torch's initial seed
   are recorded only when those libraries are already loaded; file hashing is
   optional as described below.

``log.txt``
   The tail of the application log at completion.

``outputs/``
   Artifacts explicitly attached by pipeline code.

``environment/``
   ``requirements-lock.txt`` contains ``pip freeze`` output, or package
   name/version entries from installed-distribution metadata if pip cannot
   run. ``conda-explicit.txt`` is added only when a conda environment is
   detected and ``conda list --explicit`` succeeds. ``environment_lock`` names
   the files actually written and the environment digest. When the conda file
   is present, use ``conda create --name rerun --file conda-explicit.txt``
   followed by ``python -m pip install -r requirements-lock.txt``; otherwise
   install the requirements into a compatible Python environment.

   The lists are cached under ``~/.spacr/env_locks/<digest>`` and reused for
   the same digest. That digest includes the interpreter identity and recorded
   package versions; package metadata is cached within the process, so restart
   spaCR after changing installed packages to obtain a new snapshot. Failure
   to write a run's lockfiles is recorded under ``provenance_warnings`` and
   does not by itself stop the pipeline.

File provenance
---------------

Set ``hash_inputs=True`` to record input/output file hashes; hashing is off
by default and the manifest then reports ``input_hashing="skipped"``. Discovery
follows path-valued settings, including nested plate lists. Hashed file records
contain the full SHA-256, size, modification timestamp and referring setting
key. Directory walks skip linked entries, version-control folders, caches and
the run journal. Before/after inventories used to identify changed outputs
are limited to 200,000 files per root; truncation is reported in
``provenance_warnings``.

The aggregate ``input_tree_sha256`` and ``output_tree_sha256`` values hash the
recorded dictionaries, including their paths and metadata. Matching values
mean those records match; they do not prove complete file coverage when
hashing was skipped or provenance was incomplete. Check ``input_hashing``,
``provenance_warnings`` and the per-file records before comparing runs.

Crash and failure behavior
--------------------------

A ``running`` manifest is written before the pipeline starts. Normal context
exit attempts an atomic update with success, failure or cancellation status;
caught exceptions are re-raised after their traceback is captured. Forced
process termination or failure to write the final manifest can leave the
stored status as ``running``. Run History can display it as ``interrupted``
when its recorded owner is gone; this does not rewrite the stored manifest.
Provenance failures are logged, with recorded warnings retained under
``provenance_warnings`` when the manifest can be written.

Public API
----------

Use :func:`spacr.run_journal.open_run` around a custom pipeline. Within the
context, :meth:`spacr.run_journal.Run.record_model` records a model path;
:meth:`spacr.run_journal.Run.record_input` and
:meth:`spacr.run_journal.Run.record_output` can add paths absent from settings
when ``hash_inputs=True``. Those two methods are no-ops when hashing is off.

.. code-block:: python

   from spacr.run_journal import open_run

   with open_run("my_assay", settings) as run:
       run.record_model("classifier", settings["model_path"])
       result = run_assay(settings)
       run.record_output(result)

``spacr-repro <run-folder>`` replays supported modules with the recorded
settings. The complete API is generated under :mod:`spacr.run_journal`.
