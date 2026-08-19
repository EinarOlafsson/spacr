Python API quickstart
=====================

Use the Python API when a workflow needs to run from a notebook, a reusable
script, a server or a scheduler. The desktop application and the Python API
call the same pipeline functions and use the same setting names.

Install the headless package
----------------------------

.. code-block:: bash

   python -m pip install spacr

Add ``[qt]`` only when the same environment also needs the desktop interface.

Use the typed workflow configuration
------------------------------------

The top-level API contains a small, stable interface for the principal
workflows. Typed fields cover the choices most scripts make; ``extra`` accepts
advanced settings while refusing a second value for an existing typed field.

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

The headless command is useful in a scheduler because it validates setting
names and values before importing the heavy pipeline stack.

.. code-block:: bash

   spacr-run --list
   spacr-run --describe mask
   spacr-run validate --module mask --settings mask_settings.csv
   spacr-run mask --settings mask_settings.csv

Command-line overrides are applied after the file:

.. code-block:: bash

   spacr-run mask --settings mask_settings.csv --set test_mode=true

Unknown settings and values that cannot be converted are refused with a
suggestion. Use ``spacr-doctor`` when the problem is the environment rather
than a setting.

Continue from notebooks
-----------------------

The repository's ``Notebooks/`` directory contains complete Mask, Measure,
Classify, barcode and regression examples. Treat the settings helpers and the
:doc:`curated API reference <api/index>` as authoritative for the installed
version; notebooks are worked examples rather than a compatibility contract.
