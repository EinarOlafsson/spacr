|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |Preprint| |DOI|

.. |Docs| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/pages/pages-build-deployment/badge.svg
   :target: https://einarolafsson.github.io/spacr/
   :alt: Documentation
.. |Tutorials| image:: https://img.shields.io/badge/Tutorials-Interactive%20walkthrough-4A9EFF
   :target: https://einarolafsson.github.io/spacr/tutorials/
   :alt: Interactive tutorials
.. |PyPI| image:: https://img.shields.io/pypi/v/spacr
   :target: https://pypi.org/project/spacr/
   :alt: PyPI version
.. |Python| image:: https://img.shields.io/badge/Python-3.9%E2%80%933.14-3776AB?logo=python&logoColor=white
   :target: https://pypi.org/project/spacr/
   :alt: Python 3.9 through 3.14
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg?branch=nightly
   :target: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml
   :alt: Test suite
.. |Qt| image:: https://img.shields.io/badge/GUI-Qt%20%28PySide6%29-41CD52
   :target: https://einarolafsson.github.io/spacr/
   :alt: Qt interface
.. |Source| image:: https://img.shields.io/badge/GitHub-Source-181717?logo=github
   :target: https://github.com/EinarOlafsson/spacr
   :alt: GitHub source
.. |Issues| image:: https://img.shields.io/github/issues/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/issues
   :alt: GitHub issues
.. |License| image:: https://img.shields.io/github/license/EinarOlafsson/spacr
   :target: https://github.com/EinarOlafsson/spacr/blob/main/LICENSE
   :alt: BSD 3-Clause license
.. |Preprint| image:: https://img.shields.io/badge/bioRxiv-2026.07.08.737057-BF2636
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1
   :alt: bioRxiv preprint
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343316-blue
   :target: https://doi.org/10.5281/zenodo.21343316
   :alt: Zenodo DOI
.. |Release| image:: https://img.shields.io/github/v/release/EinarOlafsson/spacr?label=Installers
   :target: https://github.com/EinarOlafsson/spacr/releases/latest
   :alt: Latest installers
.. |Conda| image:: https://anaconda.org/conda-forge/spacr/badges/version.svg
   :target: https://anaconda.org/conda-forge/spacr
   :alt: conda-forge version

.. image:: spacr/resources/icons/logo_spacr_readme.png
   :alt: spaCR
   :width: 920

spaCR
=====

.. spacr-language-picker-begin

Languages: `🌐 English ▾ <docs/i18n/readme/README.md>`_

.. spacr-language-picker-end

**Spatial phenotype analysis of CRISPR screens.**

spaCR segments and measures single cells in high-content microscopy images,
integrates per-object phenotypes with sequencing-derived guide abundance, and
estimates which genes are associated with phenotypic changes. Starting from
plate images and FASTQ reads, it produces per-object measurements, trained
classifiers, per-guide and per-gene effect estimates, and a ranked hit list.

The segmentation, measurement, annotation and classification modules also
run without a sequencing arm.

Images, masks, crops, measurements, annotations, predictions, barcodes and
well identifiers live in one SQLite project.

Runs as a desktop application or headlessly on a workstation, server or
cluster.

Hardware support
~~~~~~~~~~~~~~~~

.. spacr-hardware-begin

.. list-table::
   :header-rows: 1
   :widths: 32 18 18 22

   * - Hardware
     - Cellpose 4
     - Torch
     - UMAP / clustering
   * - NVIDIA (CUDA)
     - 🟢 GPU
     - 🟢 GPU
     - 🟢 GPU
   * - AMD on Linux (ROCm)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - AMD in an Intel Mac (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Apple Silicon (Metal)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - Intel Arc/Xe (XPU)
     - 🟣 GPU
     - 🟣 GPU
     - 🔴 CPU
   * - No GPU
     - 🟢 CPU
     - 🟢 CPU
     - 🟢 CPU

🟢 supported (stable)   🟣 implemented (beta)   🔴 CPU support only

.. spacr-hardware-end


Install spaCR
-------------

Desktop application
~~~~~~~~~~~~~~~~~~~

The installers bundle their own Python. Conda is not required.

.. spacr-installer-links-begin

|InstallerLinux| |InstallerMacOS| |InstallerWindows| |InstallerLegacy|

.. |InstallerWindows| image:: spacr/resources/icons/platforms/windows.png
   :width: 64
   :alt: Download spaCR 1.5.0.4 for Windows 10/11
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Windows-Online-Setup.exe
.. |InstallerMacOS| image:: spacr/resources/icons/platforms/macos.png
   :width: 64
   :alt: Download spaCR 1.5.0.4 for macOS 11+ (Intel and Apple silicon)
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-macOS-Universal-Online.pkg
.. |InstallerLinux| image:: spacr/resources/icons/platforms/linux.png
   :width: 64
   :alt: Download spaCR 1.5.0.4 for 64-bit Linux
   :target: https://github.com/EinarOlafsson/spacr/releases/download/v1.5.0.4/SpaCR-1.5.0.4-Linux-x86_64-Online.run
.. |InstallerLegacy| image:: spacr/resources/icons/platforms/legacy.png
   :width: 64
   :alt: Earlier spaCR installers
   :target: docs/source/installers.rst

.. spacr-installer-links-end

The first three icons download the current release. The spaCR icon opens the
complete installer archive. Installer links and versioned filenames are
updated by the release workflow; earlier installers remain in the same
release archive.

On Linux, make the downloaded file executable and run it:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

On macOS, open the ``.pkg``. The current beta is not notarized; if Gatekeeper
blocks it, choose **System Settings → Privacy & Security → Open Anyway**.

See the `installer guide <docs/source/installer_guide.rst>`_ for update, uninstall,
offline and troubleshooting instructions.

PyPI installation
~~~~~~~~~~~~~~~~~

For the PyPI release, install spaCR with pip inside a Conda environment.
Python 3.12 has the widest choice of optional scientific packages:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"
   spacr

spaCR supports Python **3.9 through 3.14**, except Python 3.14.1, which
torchvision excludes. Linux is recommended for the heaviest CUDA and ROCm
workflows; macOS and Windows are also supported, and both use their GPUs —
macOS through Metal, which covers Apple Silicon and the AMD cards in Intel
Macs, and Windows through CUDA or DirectML.

For a server, cluster or CI runner, omit Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example
``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and
``spacr[czi,nd2,lif]``. See the `installation guide
<docs/source/installer_guide.rst>`_ for the complete extras and Python-version
compatibility table.

Conda-forge installation
~~~~~~~~~~~~~~~~~~~~~~~~

The official conda-forge package installs spaCR and its desktop dependencies
into the active environment:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

Install from source
~~~~~~~~~~~~~~~~~~~

Clone the repository and install it in editable mode, so your working copy
*is* the installed package and edits take effect without reinstalling::

    git clone https://github.com/EinarOlafsson/spacr.git
    cd spacr
    conda create -n spacr python=3.12 -y
    conda activate spacr
    pip install -e .
    spacr

The default branch is ``nightly``. For a specific release::

    git clone --branch v1.5.0.5 https://github.com/EinarOlafsson/spacr.git

To pull later changes, from inside the clone::

    git pull
    pip install -e .

The second line is only needed when dependencies or entry points changed;
Python code is picked up without it. If a command still runs old code after
pulling, ``spacr-doctor`` reports which ``spacr`` is actually on your path,
which is the usual cause.

Install from source (light)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Full clone: 427 MB. Core clone: 76 MB.

::

    curl -fsSL https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/packaging/install_from_source.sh -o install_spacr.sh
    sh install_spacr.sh --branch nightly

Skips ``docs/``, ``tests/``, Cellpose checkpoints, archived figures and the
extended translation catalogs. The result is a normal checkout.

Options: ``--dir``, ``--branch`` (default ``main``), ``--with-tests``,
``--with-docs``, ``--with-translations``, ``--no-install``.

``packaging/source_install_excludes.txt`` lists every skipped path.


Command-line entry points
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr                                      # launch the Qt application
   spacr-doctor                               # diagnose the installation
   spacr-run --list                           # list headless modules
   spacr-run --describe MODULE                # inspect a module contract
   spacr-run MODULE --settings settings.csv   # execute a module
   spacr-run validate --module MODULE \
       --settings settings.csv                # validate before running
   spacr-repro RUN_DIR                        # replay a recorded run

Set ``SPACR_LOG_LEVEL=DEBUG`` when troubleshooting. Rotating logs are written
to ``~/.spacr/logs/spacr.log``.

``spacr-run --list`` lists modules with headless command-line entry points.
GUI-only annotation, curation, comparison and exploration modules are omitted.


Core workflow
-------------

The primary workflow comprises six modules:

- **Mask** segments cells, nuclei, pathogens and organelles with Cellpose.
- **Measure** writes morphology, intensity, texture, spatial and
  colocalization features, together with object crops, to SQLite.
- **Annotate** labels crops in a keyboard-driven grid and supports
  active-learning queues.
- **Classify** trains image or measurement-based models and records held-out
  performance with each checkpoint.
- **Map Barcodes** maps FASTQ reads to wells and gRNAs, with abundance,
  collision and coverage QC.
- **Regression** estimates guide, gene, condition and control effects with
  model families suited to continuous, fractional and count responses.

spaCR modules
-------------

.. spacr-workflow-begin

| |Module_mask|\ |Module_measure|\ |Module_annotate|\ |Module_classify_merged|\ |Module_map_barcodes|\ |Module_regression|
| |Module_foreign|\ |Module_run_compare|\ |Module_experiment_design|\ |Module_power|\ |Module_dose_response|\ |Module_qc_dashboard|
| |Module_make_masks|\ |Module_align|\ |Module_umap|\ |Module_gate_editor|\ |Module_graph_builder|\ |Module_analyze_plaques|
| |Module_recruitment|\ |Module_invasion|\ |Module_replication|

.. |Module_mask| image:: spacr/resources/icons/workflow/mask.png
   :width: 16.0%
   :alt: Open the Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.preprocess_generate_masks
   :align: middle
.. |Module_measure| image:: spacr/resources/icons/workflow/measure.png
   :width: 16.0%
   :alt: Open the Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Module_annotate| image:: spacr/resources/icons/workflow/annotate.png
   :width: 16.0%
   :alt: Open the Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Module_classify_merged| image:: spacr/resources/icons/workflow/classify_merged.png
   :width: 16.0%
   :alt: Open the Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Module_map_barcodes| image:: spacr/resources/icons/workflow/map_barcodes.png
   :width: 16.0%
   :alt: Open the Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Module_regression| image:: spacr/resources/icons/workflow/regression.png
   :width: 16.0%
   :alt: Open the Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Module_foreign| image:: spacr/resources/icons/workflow/apps/foreign.png
   :width: 16.0%
   :alt: Open the Import API
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |Module_run_compare| image:: spacr/resources/icons/workflow/apps/run_compare.png
   :width: 16.0%
   :alt: Open the Run Compare API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |Module_experiment_design| image:: spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 16.0%
   :alt: Open the Experiment Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |Module_power| image:: spacr/resources/icons/workflow/apps/power.png
   :width: 16.0%
   :alt: Open the Power / Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |Module_dose_response| image:: spacr/resources/icons/workflow/apps/dose_response.png
   :width: 16.0%
   :alt: Open the Dose–Response API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle
.. |Module_qc_dashboard| image:: spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 16.0%
   :alt: Open the QC API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |Module_make_masks| image:: spacr/resources/icons/workflow/apps/make_masks.png
   :width: 16.0%
   :alt: Open the Make Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |Module_align| image:: spacr/resources/icons/workflow/apps/align.png
   :width: 16.0%
   :alt: Open the Align & Stitch API
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |Module_umap| image:: spacr/resources/icons/workflow/apps/umap.png
   :width: 16.0%
   :alt: Open the Image UMAP API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html#spacr.core.generate_image_umap
   :align: middle
.. |Module_gate_editor| image:: spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 16.0%
   :alt: Open the Gate Editor API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |Module_graph_builder| image:: spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 16.0%
   :alt: Open the Graph Builder API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |Module_analyze_plaques| image:: spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 16.0%
   :alt: Open the Plaque Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_plaques
   :align: middle
.. |Module_recruitment| image:: spacr/resources/icons/workflow/apps/recruitment.png
   :width: 16.0%
   :alt: Open the Recruitment API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_recruitment
   :align: middle
.. |Module_invasion| image:: spacr/resources/icons/workflow/apps/invasion.png
   :width: 16.0%
   :alt: Open the Invasion Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_invasion
   :align: middle
.. |Module_replication| image:: spacr/resources/icons/workflow/apps/replication.png
   :width: 16.0%
   :alt: Open the Replication Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html#spacr.submodules.analyze_replication
   :align: middle

.. spacr-workflow-end

Every module spaCR ships, in the order the home screen lists them: the six
pipeline modules first, then everything else. Select a tile to open that
module's API page.


Make Masks
~~~~~~~~~~

Make Masks appears under **Tools** for manual correction of segmentation
masks; its masthead opens the Cellpose workflows. Nine tools: **Brush**,
**Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**,
**Zoom** and **Recrop**. Draw makes one filled label from a closed outline,
Divide separates a merged object along a drawn line, Recrop turns one object
in a crowded field into its own field.

See the `feature guide <docs/source/features.rst>`_ for each tool.

Other resources
~~~~~~~~~~~~~~~

- `Interactive tutorials <https://einarolafsson.github.io/spacr/tutorials/>`_
  — 73 guided workflows from installation through hit investigation.
- `Python API quickstart <docs/source/python_api.rst>`_ — run and validate
  pipelines from scripts, notebooks or a cluster.
- `Feature guide <docs/source/features.rst>`_ — capabilities, maturity and
  optional integrations.
- `Curated API reference
  <https://einarolafsson.github.io/spacr/api/index.html>`_ — supported entry
  points by task, with the complete module reference one level deeper.
- `Language & translation guide <docs/source/localization.rst>`_ — interface
  languages, contextual help and scientific-output policy.

Language & translation
~~~~~~~~~~~~~~~~~~~~~~

The interface supports ten languages across navigation and Preferences. AI and
LIVE controls, module descriptions and reviewed contextual help are also
translated. Change the language under **spaCR → Preferences → Language**
without restarting. Logs, paths, database values and measurements are never
translated; scientific output remains canonical English. See the
`contextual-help policy <docs/source/localization.rst#contextual-help>`_.

Animated setting guidance
~~~~~~~~~~~~~~~~~~~~~~~~~

Settings with a visual explanation offer an **Animation** control in their
tooltip. Browse the `setting animation gallery
<https://einarolafsson.github.io/spacr/setting_animations.html>`_ or the
`Setting animation registry
<https://einarolafsson.github.io/spacr/api/spacr/setting_animations/index.html>`_.

Data
----

Reference datasets
~~~~~~~~~~~~~~~~~~

|DataBioStudies| |DataHuggingFace| |DataNCBI| |DataSpaCRPower| |DataBioRxiv|

.. |DataBioStudies| image:: spacr/resources/icons/databanks/biostudies_button.png
   :width: 72
   :alt: Open the BioStudies microscopy dataset
   :target: https://doi.org/10.6019/S-BIAD2135
.. |DataHuggingFace| image:: spacr/resources/icons/databanks/huggingface_button.png
   :width: 72
   :alt: Open the Hugging Face testing dataset
   :target: https://huggingface.co/datasets/einarolafsson/toxo_mito
.. |DataNCBI| image:: spacr/resources/icons/databanks/ncbi_button.png
   :width: 72
   :alt: Open the NCBI sequencing dataset
   :target: https://www.ncbi.nlm.nih.gov/bioproject/?term=PRJNA1261935
.. |DataSpaCRPower| image:: spacr/resources/icons/databanks/spacrpower_button.png
   :width: 72
   :alt: Open spaCRPower
   :target: https://github.com/maomlab/spaCRPower
.. |DataBioRxiv| image:: spacr/resources/icons/databanks/biorxiv_button.png
   :width: 72
   :alt: Open the bioRxiv preprint
   :target: https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1

Model zoo
~~~~~~~~~

spaCR ships a catalogue of trained models and fetches them on demand. Open
**Model Zoo** from the home screen to browse and install them, or name a key
in a settings file -- ``pathogen_model: toxoplasma_pv_v1`` -- and the model is
downloaded and checksum-verified the first time it is needed. Every published
entry carries a SHA-256; an entry without one is refused rather than installed,
because a truncated or substituted checkpoint cannot be told from the real one.

.. spacr-model-zoo-begin

.. list-table::
   :header-rows: 1
   :widths: 24 34 42

   * - Model
     - Training data
     - Hold-out performance
   * - ``toxoplasma_pv_v1``
       (Cellpose-SAM (cpsam_v2))
     - anti-Toxoplasma-biotin and DsRed PV lumen; 115 images, 1 dataset
     - F1 0.867 against 0.713 for stock cpsam, at IoU 0.5
   * - ``toxoplasma_plaque_v1``
       (Cellpose-SAM (cpsam))
     - crystal violet plaque wells; 184 wells from 3 datasets, 95 in-house and 89 literature
     - F1 0.856 in-domain; 0.806 on literature (3-fold cross-validated, SD 0.020)
   * - ``toxoplasma_well_detector_v1``
       (YOLO11n)
     - whole-plate and multi-well crystal violet images; 562 images from 1 dataset, 190 of them with no well in them
     - mAP50 0.993, mAP50-95 0.886, precision and recall both 0.987

.. spacr-model-zoo-end

Every figure above is measured on images the model never saw in training.

**Precision** is how many of the objects a model reported are real; **recall**
is how many of the real objects it found. They fail in opposite directions:
poor precision invents plaques, poor recall misses them.

**F1** is the two combined, and is quoted because each alone is trivially
gamed -- report one unmistakable plaque for near-perfect precision, or every
dark blob for near-perfect recall. Which you would rather lose depends on the
assay, and counting is usually better served by over-calling: the plaque model
was accepted at precision 0.858 with recall 0.811 over an earlier round at
0.939 and 0.631.

**IoU**, intersection over union, is how much a predicted object and the real
one overlap, divided by the area they cover together. It is the ruler the rest
are read against, so a score means nothing without its threshold: "F1 0.867 at
IoU 0.5" counts a vacuole as found when the two outlines agree over half their
combined area.

**mAP50** and **mAP50-95** belong to the detector. The first asks whether the
wells were found; the second repeats it across ten thresholds from 0.5 to
0.95, so it also asks how tightly each box is drawn. The gap between them is
placement, not detection.

**Cross-validated**, with an **SD**, means the score is the mean of three runs
on different splits and the SD is how far they moved apart. One split can be
lucky: this model's literature figure is 0.834 on a single 19-well split and
0.806 across all three.

Models are hosted on their author's own Hugging Face account, so contributing
one does not mean handing write access to anyone else's. ``spacr.model_zoo``'s
``publish_model`` performs the upload and prints the catalogue row to add.


Diagnosing performance
----------------------

Generate a hardware report and attach it to a performance-related issue::

    python tools/spacr_hardware_report.py

Saves to ``~/.spacr/reports`` and prints the path. ``--quick`` skips the
longer benchmarks; ``--out PATH`` sets the location.

Reads no project data. Times imports, numeric libraries, window
construction and animation. Reports processor-architecture emulation (an
x86_64 Python build on Apple Silicon) and NumPy's BLAS implementation.

Command-line reference
----------------------

Every command below is installed by ``pip install spacr``. All of them accept
``--help``.

Launching the application
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr              # the desktop application
   spacr-tutorial     # the interactive tutorial library
   spacr-server       # no first-run setup screen, for unattended launches

``spacr-server`` skips the modal setup screen, which would otherwise block
an unattended job.

``spacr-qt`` and ``spacr-nightly`` are aliases of ``spacr``.

When spaCR will not start
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-doctor       # diagnose the installation and say how to fix it
   safespacr          # the least spaCR that can still change a setting

``spacr-doctor`` prints one line per check, with a command to run for each
failure. It also reports which ``spacr`` is on the path, which is what an old
editable install shadows.

``safespacr`` reads every preference as its default and forces the backdrop,
animations, verbose logging and preloading off. Use it when a saved
preference breaks the launch. It changes nothing permanently.

Running modules headlessly
~~~~~~~~~~~~~~~~~~~~~~~~~~

No Qt, no display — for clusters, servers and CI.

.. code-block:: bash

   spacr-run --list                              # modules with a headless entry
   spacr-run --describe MODULE                   # what a module consumes and produces
   spacr-run validate --module MODULE \
       --settings settings.csv                   # check settings before spending the run
   spacr-run MODULE --settings settings.csv      # execute
   spacr-remote --help                           # submit and monitor SSH, Slurm or cloud jobs

``validate`` reads the same settings the run would and reports what is
missing, contradictory or pointing at nothing.

``spacr-run --list`` shows only modules with a headless entry point;
annotation, curation and exploration are interactive and omitted.

Inspecting a run afterwards
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every run is journalled to ``~/.spacr/runs`` with its settings, hashed
inputs, outputs, warnings, versions and seeds.

.. code-block:: bash

   spacr-repro RUN_DIR        # replay a recorded run from its journal
   spacr-workspace RUN_DIR    # what that run had open: databases, montages, views

Auditing data and installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   spacr-db-audit DB      # SQLite health, integrity, locking, reader/writer probe
   spacr-leakage          # classifier train/test leakage audit
   spacr-plugins          # installed plugin registry and failure diagnostics

Environment
~~~~~~~~~~~

.. code-block:: bash

   SPACR_LOG_LEVEL=DEBUG spacr      # verbose logging for one launch

Rotating logs are written to ``~/.spacr/logs/spacr.log``. Attach that file
to a bug report.


Contributing and support
------------------------

Submit bug reports and focused feature requests through
`GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_.
When reporting a failure, include the spaCR version, operating system, Python
version, module settings and the relevant log excerpt. ``spacr-doctor``
collects most of this information; include the hardware report when reporting
performance problems.

Licensing
~~~~~~~~~

spaCR is released under the `BSD 3-Clause License
<https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_.

If spaCR contributed to published work, a citation is appreciated and is not
a condition of the licence — see `Citing spaCR`_ below.

Tutorials
~~~~~~~~~

The `interactive spaCR tutorial library
<https://einarolafsson.github.io/spacr/tutorials/>`_ contains narrated,
captioned walkthroughs of installation and of each application workflow, in
73 lessons with 50 voices across eight languages.

Citing spaCR
~~~~~~~~~~~~

If spaCR contributes to your research, cite:

Olafsson EB, *et al.* A pooled image-based CRISPR screen identifies
EAF1 as a *T. gondii* modulator of ESCRT subversion.

`bioRxiv preprint <https://www.biorxiv.org/content/10.64898/2026.07.08.737057v1>`_ ·
`software archive <https://doi.org/10.5281/zenodo.21343316>`_

Acknowledgments
~~~~~~~~~~~~~~~

spaCR builds on open scientific software including NumPy, pandas,
scikit-image, scikit-learn, Cellpose, PyTorch and Qt. See the
`translation model attribution <docs/i18n/TRANSLATION_MODELS.md>`_ for the
models used to prepare the multilingual documentation and interface catalogs.
