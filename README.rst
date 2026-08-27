|Docs| |Tutorials| |PyPI| |Conda| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
   :alt: PolyForm Noncommercial license
.. |DOI| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.21343317-blue
   :target: https://doi.org/10.5281/zenodo.21343317
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

Languages: `English <README.rst>`_ · `Svenska <docs/i18n/readme/README.sv.rst>`_ ·
`Deutsch <docs/i18n/readme/README.de.rst>`_ ·
`Español <docs/i18n/readme/README.es.rst>`_ ·
`简体中文 <docs/i18n/readme/README.zh_CN.rst>`_ ·
`Português <docs/i18n/readme/README.pt.rst>`_ ·
`हिन्दी <docs/i18n/readme/README.hi.rst>`_ ·
`한국어 <docs/i18n/readme/README.ko.rst>`_ ·
`Íslenska <docs/i18n/readme/README.is.rst>`_ ·
`Français <docs/i18n/readme/README.fr.rst>`_

**Spatial phenotype analysis of CRISPR screens.**

spaCR segments and measures single cells in high-content microscopy images,
integrates per-object phenotypes with sequencing-derived guide abundance, and
estimates which genes are associated with phenotypic changes. Starting from
plate images and FASTQ reads, it produces per-object measurements, trained
classifiers, per-guide and per-gene effect estimates, and a ranked hit list.

For image-based pooled CRISPR screens, spaCR provides the workflow from image
segmentation through hit prioritization. For high-content microscopy studies
without sequencing-based screens, the segmentation, measurement, annotation
and classification modules can be used independently.

Images, masks, crops, measurements, annotations, predictions, barcodes and
well identifiers live in one SQLite project, so a number in a result can be
traced back to the object it came from.

Run spaCR as a desktop application or headlessly on a workstation, server or
cluster. Both drive the same modules, and CUDA is used automatically where a
module supports it.


Workflow at a glance
--------------------

.. spacr-workflow-begin

|Workflow_mask|\ |Workflow_arrow|\ |Workflow_measure|\ |Workflow_arrow|\ |Workflow_annotate|\ |Workflow_arrow|\ |Workflow_classify_merged|\ |Workflow_arrow|\ |Workflow_map_barcodes|\ |Workflow_arrow|\ |Workflow_regression|

.. |Workflow_mask| image:: spacr/resources/icons/workflow/mask.png
   :width: 14.5%
   :alt: Open the Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |Workflow_measure| image:: spacr/resources/icons/workflow/measure.png
   :width: 14.5%
   :alt: Open the Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |Workflow_annotate| image:: spacr/resources/icons/workflow/annotate.png
   :width: 14.5%
   :alt: Open the Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/annotate/index.html
   :align: middle
.. |Workflow_classify_merged| image:: spacr/resources/icons/workflow/classify_merged.png
   :width: 14.5%
   :alt: Open the Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |Workflow_map_barcodes| image:: spacr/resources/icons/workflow/map_barcodes.png
   :width: 14.5%
   :alt: Open the Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |Workflow_regression| image:: spacr/resources/icons/workflow/regression.png
   :width: 14.5%
   :alt: Open the Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |Workflow_arrow| image:: spacr/resources/icons/workflow/arrow.png
   :width: 2.5%
   :align: middle

**Data**

|App_align|\ |App_convert|\ |App_foreign|\ |App_external_masks|\ |App_queue|

|App_batch|\ |App_distributed_jobs|\ |App_db_browser|\ |App_make_masks|\ |App_data_manager|

|App_project_browser|

**Results & QC**

|App_plate_view|\ |App_umap|\ |App_train_compare|\ |App_run_history|\ |App_report|

|App_run_compare|\ |App_investigate_hit|\ |App_control_chart|

**Explore**

|App_pipeline_graph|\ |App_profiler|\ |App_qc_dashboard|\ |App_lineage|\ |App_layer_viewer|

|App_graph_builder|\ |App_tabulate|\ |App_feature_dict|\ |App_trellis|\ |App_gate_editor|

|App_feature_explorer|\ |App_outliers|

**Assays**

|App_analyze_plaques|\ |App_recruitment|\ |App_invasion|\ |App_replication|

**Design**

|App_experiment_design|\ |App_power|\ |App_dose_response|

.. |App_align| image:: spacr/resources/icons/workflow/apps/align.png
   :width: 19.9%
   :alt: Open the Align & Stitch API
   :target: https://einarolafsson.github.io/spacr/api/spacr/align/index.html
   :align: middle
.. |App_convert| image:: spacr/resources/icons/workflow/apps/convert.png
   :width: 19.9%
   :alt: Open the Format Converter API
   :target: https://einarolafsson.github.io/spacr/api/spacr/convert/index.html
   :align: middle
.. |App_foreign| image:: spacr/resources/icons/workflow/apps/foreign.png
   :width: 19.9%
   :alt: Open the Import Project API
   :target: https://einarolafsson.github.io/spacr/api/spacr/foreign/index.html
   :align: middle
.. |App_external_masks| image:: spacr/resources/icons/workflow/apps/external_masks.png
   :width: 19.9%
   :alt: Open the External Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/external_masks/index.html
   :align: middle
.. |App_queue| image:: spacr/resources/icons/workflow/apps/queue.png
   :width: 19.9%
   :alt: Open the Plate Queue API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/plate_queue/index.html
   :align: middle
.. |App_batch| image:: spacr/resources/icons/workflow/apps/batch.png
   :width: 19.9%
   :alt: Open the Batch Runner API
   :target: https://einarolafsson.github.io/spacr/api/spacr/batch/index.html
   :align: middle
.. |App_distributed_jobs| image:: spacr/resources/icons/workflow/apps/distributed_jobs.png
   :width: 19.9%
   :alt: Open the Distributed Jobs API
   :target: https://einarolafsson.github.io/spacr/api/spacr/remote_execution/index.html
   :align: middle
.. |App_db_browser| image:: spacr/resources/icons/workflow/apps/db_browser.png
   :width: 19.9%
   :alt: Open the Database Browser API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/db_browser/index.html
   :align: middle
.. |App_make_masks| image:: spacr/resources/icons/workflow/apps/make_masks.png
   :width: 19.9%
   :alt: Open the Make Masks API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/make_masks/index.html
   :align: middle
.. |App_data_manager| image:: spacr/resources/icons/workflow/apps/data_manager.png
   :width: 19.9%
   :alt: Open the Data Manager API
   :target: https://einarolafsson.github.io/spacr/api/spacr/data_manager/index.html
   :align: middle
.. |App_project_browser| image:: spacr/resources/icons/workflow/apps/project_browser.png
   :width: 19.9%
   :alt: Open the Project Browser API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/project_browser/index.html
   :align: middle
.. |App_plate_view| image:: spacr/resources/icons/workflow/apps/plate_view.png
   :width: 19.9%
   :alt: Open the Plate Viewer API
   :target: https://einarolafsson.github.io/spacr/api/spacr/plate_qc/index.html
   :align: middle
.. |App_umap| image:: spacr/resources/icons/workflow/apps/umap.png
   :width: 19.9%
   :alt: Open the Image UMAP API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |App_train_compare| image:: spacr/resources/icons/workflow/apps/train_compare.png
   :width: 19.9%
   :alt: Open the Training Runs API
   :target: https://einarolafsson.github.io/spacr/api/spacr/train_compare/index.html
   :align: middle
.. |App_run_history| image:: spacr/resources/icons/workflow/apps/run_history.png
   :width: 19.9%
   :alt: Open the Run History API
   :target: https://einarolafsson.github.io/spacr/api/spacr/run_journal/index.html
   :align: middle
.. |App_report| image:: spacr/resources/icons/workflow/apps/report.png
   :width: 19.9%
   :alt: Open the Report API
   :target: https://einarolafsson.github.io/spacr/api/spacr/report/index.html
   :align: middle
.. |App_run_compare| image:: spacr/resources/icons/workflow/apps/run_compare.png
   :width: 19.9%
   :alt: Open the Run Compare API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/run_compare/index.html
   :align: middle
.. |App_investigate_hit| image:: spacr/resources/icons/workflow/apps/investigate_hit.png
   :width: 19.9%
   :alt: Open the Investigate Hit API
   :target: https://einarolafsson.github.io/spacr/api/spacr/hit_investigation/index.html
   :align: middle
.. |App_control_chart| image:: spacr/resources/icons/workflow/apps/control_chart.png
   :width: 19.9%
   :alt: Open the Control Charts API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/control_chart/index.html
   :align: middle
.. |App_pipeline_graph| image:: spacr/resources/icons/workflow/apps/pipeline_graph.png
   :width: 19.9%
   :alt: Open the Pipeline Graph API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/pipeline_graph/index.html
   :align: middle
.. |App_profiler| image:: spacr/resources/icons/workflow/apps/profiler.png
   :width: 19.9%
   :alt: Open the Prediction Profiler API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/profiler/index.html
   :align: middle
.. |App_qc_dashboard| image:: spacr/resources/icons/workflow/apps/qc_dashboard.png
   :width: 19.9%
   :alt: Open the QC Dashboard API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/qc_dashboard/index.html
   :align: middle
.. |App_lineage| image:: spacr/resources/icons/workflow/apps/lineage.png
   :width: 19.9%
   :alt: Open the Lineage API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/lineage/index.html
   :align: middle
.. |App_layer_viewer| image:: spacr/resources/icons/workflow/apps/layer_viewer.png
   :width: 19.9%
   :alt: Open the Layer Viewer API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/layer_viewer/index.html
   :align: middle
.. |App_graph_builder| image:: spacr/resources/icons/workflow/apps/graph_builder.png
   :width: 19.9%
   :alt: Open the Graph Builder API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/graph_builder/index.html
   :align: middle
.. |App_tabulate| image:: spacr/resources/icons/workflow/apps/tabulate.png
   :width: 19.9%
   :alt: Open the Tabulate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/tabulate/index.html
   :align: middle
.. |App_feature_dict| image:: spacr/resources/icons/workflow/apps/feature_dict.png
   :width: 19.9%
   :alt: Open the Feature Dictionary API
   :target: https://einarolafsson.github.io/spacr/api/spacr/feature_dict/index.html
   :align: middle
.. |App_trellis| image:: spacr/resources/icons/workflow/apps/trellis.png
   :width: 19.9%
   :alt: Open the Small Multiples API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/trellis/index.html
   :align: middle
.. |App_gate_editor| image:: spacr/resources/icons/workflow/apps/gate_editor.png
   :width: 19.9%
   :alt: Open the Gate Editor API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/gate_editor/index.html
   :align: middle
.. |App_feature_explorer| image:: spacr/resources/icons/workflow/apps/feature_explorer.png
   :width: 19.9%
   :alt: Open the Feature Explorer API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/feature_explorer/index.html
   :align: middle
.. |App_outliers| image:: spacr/resources/icons/workflow/apps/outliers.png
   :width: 19.9%
   :alt: Open the Outliers API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/outliers/index.html
   :align: middle
.. |App_analyze_plaques| image:: spacr/resources/icons/workflow/apps/analyze_plaques.png
   :width: 19.9%
   :alt: Open the Plaque Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_recruitment| image:: spacr/resources/icons/workflow/apps/recruitment.png
   :width: 19.9%
   :alt: Open the Recruitment API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_invasion| image:: spacr/resources/icons/workflow/apps/invasion.png
   :width: 19.9%
   :alt: Open the Invasion Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_replication| image:: spacr/resources/icons/workflow/apps/replication.png
   :width: 19.9%
   :alt: Open the Replication Assay API
   :target: https://einarolafsson.github.io/spacr/api/spacr/submodules/index.html
   :align: middle
.. |App_experiment_design| image:: spacr/resources/icons/workflow/apps/experiment_design.png
   :width: 19.9%
   :alt: Open the Experiment Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/experiment_design/index.html
   :align: middle
.. |App_power| image:: spacr/resources/icons/workflow/apps/power.png
   :width: 19.9%
   :alt: Open the Power / Design API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/power/index.html
   :align: middle
.. |App_dose_response| image:: spacr/resources/icons/workflow/apps/dose_response.png
   :width: 19.9%
   :alt: Open the Dose–Response API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/screens/dose_response/index.html
   :align: middle

.. spacr-workflow-end

Select a workflow module to open its API page. The grid contains every other
application in the same categories and order used on the spaCR home screen.


Install spaCR
-------------

Desktop application
~~~~~~~~~~~~~~~~~~~

The desktop installers include a private Python environment, so conda and an
existing Python installation are not required.

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

Conda-forge installation
~~~~~~~~~~~~~~~~~~~~~~~~

The official conda-forge package installs spaCR and its desktop dependencies
into the active environment:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

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
torchvision excludes. Linux is recommended for CUDA workflows; macOS and
Windows are also supported.

For a server, cluster or CI runner, omit Qt:

.. code-block:: bash

   python -m pip install spacr
   spacr-run --list

Optional integrations are installed separately, for example
``spacr[zarr]``, ``spacr[omero]``, ``spacr[napari]`` and
``spacr[czi,nd2,lif]``. See the `installation guide
<docs/source/installer_guide.rst>`_ for the complete extras and Python-version
compatibility table.

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


What you can do
---------------

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

The same project can also design plates, estimate power, correct batch effects,
inspect segmentation quality, explore linked plots and crops, export AnnData,
resume interrupted work and record the settings behind each result.

Modules available from host screens
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Twenty modules are integrated into related host screens rather than displayed
as separate Home tiles. Each opens from its host screen's masthead and uses the
active project. Mask, Measure, Annotate, Classify, Map Barcodes, Regression,
Image UMAP and Make Masks provide these integrated modules. Their help and API
documentation remain available, and modules with pipeline entry points can
still run headlessly. The `feature guide <docs/source/features.rst>`_ lists
each integrated module and its host.

Make Masks
~~~~~~~~~~

Make Masks appears under **Data** and provides manual correction of
segmentation masks. Its masthead also provides access to the Cellpose
workflows. The canvas has nine tools: **Brush**,
**Erase**, **Erase object**, **Wand +**, **Wand −**, **Draw**, **Divide**,
**Zoom** and **Recrop**. Draw creates one filled label from a free-form closed
outline. Divide separates a merged object along a user-defined line while
preserving all other object labels.

Recrop extracts a single-object field from a staged image containing multiple
objects. A bounding box around one object writes the corresponding image and
mask regions as a new field, schedules that field after the current one and
removes the original multi-object field from the curation queue. Recrop changes
the active field rather than editing label pixels.

Running Cellpose-SAM from Make Masks displays two intermediate outputs beside
the mask:
the **cell-probability map** and the **flow field**. A mask is a threshold on
the probability map, and flow-consistency checks can reject objects whose
derived flows differ from the predicted field. Inspect these outputs to
distinguish low cell probability from inconsistent flow when evaluating an
incorrect or incomplete mask.

Objects and settings
~~~~~~~~~~~~~~~~~~~~

spaCR supports cell, nucleus and pathogen objects, a cytoplasm derived from
their masks, and between zero and twenty-six organelle slots. Each organelle
slot has an independent channel, diameter, morphology preset and detection
method.

The settings panel displays controls only when they apply. Organelle slots
above the configured count are hidden, an object with no assigned channel is
excluded from the run, and morphology-specific controls are shown only for the
selected method. The **3D** and **Time** switches define the dimensionality:
``z_stack`` enables volumetric settings, ``timelapse`` enables tracking
settings, and four-dimensional settings appear when both are enabled.

Choose the next page by what you want to do:

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
translated. Change the
language under **spaCR → Preferences → Language** without restarting. Logs,
paths, database values and measurements are never translated; scientific
output remains canonical English. See the `contextual-help policy
<docs/source/localization.rst#contextual-help>`_.

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

Diagnosing performance
----------------------

Generate a hardware report and attach it to a performance-related issue::

    python tools/spacr_hardware_report.py

The command prints a report and saves a copy under ``~/.spacr/reports``; the
last line identifies the saved path. ``--quick`` omits the longer benchmarks,
and ``--out PATH`` selects another output location.

The report does not open a project or read project data. It records import and
numeric-library timing, display scaling, active preferences, main-window and
module-screen construction, and animation performance. The report file is the
only output it creates.

It also identifies processor-architecture emulation, such as an x86_64 Python
build on Apple Silicon, and the BLAS implementation used by NumPy. Either can
substantially affect performance.

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

The current development branch is source-available under the
`PolyForm Noncommercial License 1.0.0
<https://github.com/EinarOlafsson/spacr/blob/main/LICENSE>`_. Commercial use
requires a separate license from the copyright holder. Released versions
through spaCR 1.4.9.9 remain available under the MIT License that accompanied
those releases.

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
`software archive <https://doi.org/10.5281/zenodo.21343317>`_

Acknowledgments
~~~~~~~~~~~~~~~

spaCR builds on open scientific software including NumPy, pandas,
scikit-image, scikit-learn, Cellpose, PyTorch and Qt. See the
`translation model attribution <docs/i18n/TRANSLATION_MODELS.md>`_ for the
models used to prepare the multilingual documentation and interface catalogs.
