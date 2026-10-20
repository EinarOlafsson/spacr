Python API quickstart
=====================

Use the Python API when a workflow needs to run from a notebook, a reusable
script, a server or a scheduler. The desktop application and the Python API
call the same pipeline functions and use the same setting names.

Install spaCR
-------------

.. code-block:: bash

   python -m pip install spacr

Use the typed workflow configuration
------------------------------------

The top-level API provides ``MaskConfig`` and ``MeasureConfig`` with
``run_mask`` and ``run_measure``. Typed fields cover the commonly set options;
``extra`` accepts any other setting and raises ``ValueError`` if it repeats a
typed field.

.. code-block:: python

   from spacr import MaskConfig

   mask = MaskConfig(
       "/data/screen/plate01",
       cell_channel=0,
       nucleus_channel=1,
       pathogen_channel=2,
       cell_diameter=60,
       nucleus_diameter=20,
       pathogen_diameter=8,
   )

Call ``mask.to_settings()`` when a complete dictionary is needed for a saved
settings file. It expands through the same defaults used by the GUI.

Validate before a long run
--------------------------

Set ``dry_run`` to inspect the input and return a list of problems without
loading a model, using the GPU or writing results.

.. code-block:: python

   from spacr import run_mask

   check = dict(mask.to_settings(), dry_run=True)
   problems = run_mask(check)
   for problem in problems:
       print(problem)

An empty list means that the preflight checks passed. A preflight check cannot
guarantee model quality; inspect the segmentation preview before processing a
full screen.

Generate masks
--------------

.. code-block:: python

   run_mask(mask)

A normal run returns ``None`` and writes masks, overlays, object counts and the
resolved settings below ``src``. Invalid required inputs raise ``ValueError``;
runtime progress and recoverable field failures are written to the spaCR log.

Measure objects and save crops
------------------------------

.. code-block:: python

   from spacr import MeasureConfig, run_measure

   measure = MeasureConfig(
       "/data/screen/plate01/merged",
       cell_mask_dim=4,
       nucleus_mask_dim=5,
       pathogen_mask_dim=6,
       channels=(0, 1, 2, 3),
       crop_mode=("cell",),
       save_png=True,
       png_channel_mapping={"r": 2, "g": 1, "b": 0},
   )

   problems = run_measure(dict(measure.to_settings(), dry_run=True))
   if problems:
       raise RuntimeError("Measure preflight failed:\n" +
                          "\n".join(map(str, problems)))
   run_measure(measure)

Measure writes ``measurements/measurements.db`` and the resolved settings. If
``save_png`` is enabled, it also writes one crop set for each ``crop_mode``.

Run the same contract from a shell
----------------------------------

The headless command suits a scheduler because it validates setting names and
values before importing PyTorch, Cellpose and the other pipeline dependencies.

.. code-block:: bash

   spacr-run --list
   spacr-run --describe mask
   spacr-run validate --module mask --settings mask_settings.csv
   spacr-run mask --settings mask_settings.csv

Use ``spacr-run --list-models`` (or ``--models``) to list registered models,
including models that have not been downloaded. Supply the registered key or
filename in ``custom_model`` for masking or ``model_path`` for classifier
inference. Python and notebook entry points use the same resolver. A missing
registered checkpoint is downloaded through Model Zoo when the workflow runs,
verified against its catalogue checksum, and cached for reuse. Preflight
recognizes registered downloadable selections without downloading the weights.
Existing local paths take precedence; an unknown missing local path or a model
of the wrong kind raises an error rather than selecting another model.
Built-in foundation models retain their backend's normal download behavior.

Boolean values in settings CSV files are case-insensitive. ``TRUE`` and
``FALSE``, including surrounding whitespace, load as Python booleans, so
spreadsheet-exported settings can pass the same preflight checks on a cluster.

Command-line overrides are applied after the file:

.. code-block:: bash

   spacr-run mask --settings mask_settings.csv --set test_mode=true

Unknown settings and values that cannot be converted are refused; an unknown
name is reported with the closest known setting when one exists. Use ``spacr-doctor`` when the problem is the environment rather
than a setting.

Continue from notebooks
-----------------------

The repository's ``Notebooks/`` directory contains complete Mask, Measure,
Classify, barcode and regression examples. Treat the settings helpers and the
:doc:`curated API reference <api/index>` as authoritative for the installed
version; notebooks are worked examples rather than a compatibility contract.

Export measured objects to AnnData
----------------------------------

Follow :doc:`anndata_export` to export the measurement database through the
same entry point used by the desktop application, choose a missing-value
policy, and inspect the resulting object-by-feature matrix.
