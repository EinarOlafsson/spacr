|Docs| |Tutorials| |PyPI| |Python| |Tests| |Qt| |Source| |Issues| |License| |DOI|

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
.. |Tests| image:: https://github.com/EinarOlafsson/spacr/actions/workflows/tests.yml/badge.svg
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
.. |CondaRecipe| image:: https://img.shields.io/badge/conda--forge-recipe-44A833?logo=anaconda
   :target: https://github.com/EinarOlafsson/spacr/tree/main/conda-forge/recipe
   :alt: conda-forge recipe

.. image:: spacr/resources/icons/logo_spacr.png
   :alt: spaCR
   :align: center
   :width: 360

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
links each cell to the gRNA it received, and reports which genes changed the
phenotype. Plate images and FASTQ reads go in; per-object measurements,
trained classifiers, per-guide and per-gene effect sizes, and a ranked hit
list come out.

If you run image-based pooled CRISPR screens, that is the whole path. If you
have high-content microscopy and no screen, the segmentation, measurement,
annotation and classification half runs on its own.

Images, masks, crops, measurements, annotations, predictions, barcodes and
well identifiers live in one SQLite project, so a number in a result can be
traced back to the object it came from.

Run spaCR as a desktop application or headlessly on a workstation, server or
cluster. Both drive the same modules, and CUDA is used automatically where a
module supports it.


Workflow at a glance
--------------------

|WorkflowMask| |WorkflowArrow| |WorkflowMeasure| |WorkflowArrow| |WorkflowAnnotate| |WorkflowArrow| |WorkflowClassify| |WorkflowArrow| |WorkflowBarcodes| |WorkflowArrow| |WorkflowRegression|

.. |WorkflowMask| image:: spacr/resources/icons/workflow/mask.png
   :width: 96
   :alt: Open the Mask API
   :target: https://einarolafsson.github.io/spacr/api/spacr/core/index.html
   :align: middle
.. |WorkflowMeasure| image:: spacr/resources/icons/workflow/measure.png
   :width: 96
   :alt: Open the Measure API
   :target: https://einarolafsson.github.io/spacr/api/spacr/measure/index.html
   :align: middle
.. |WorkflowAnnotate| image:: spacr/resources/icons/workflow/annotate.png
   :width: 96
   :alt: Open the Annotate API
   :target: https://einarolafsson.github.io/spacr/api/spacr/qt/annotate_engine/index.html
   :align: middle
.. |WorkflowClassify| image:: spacr/resources/icons/workflow/classify_merged.png
   :width: 96
   :alt: Open the Classify API
   :target: https://einarolafsson.github.io/spacr/api/spacr/classify/index.html
   :align: middle
.. |WorkflowBarcodes| image:: spacr/resources/icons/workflow/map_barcodes.png
   :width: 96
   :alt: Open the Map Barcodes API
   :target: https://einarolafsson.github.io/spacr/api/spacr/sequencing/index.html
   :align: middle
.. |WorkflowRegression| image:: spacr/resources/icons/workflow/regression.png
   :width: 96
   :alt: Open the Regression API
   :target: https://einarolafsson.github.io/spacr/api/spacr/ml/index.html
   :align: middle
.. |WorkflowArrow| image:: spacr/resources/icons/workflow/arrow.png
   :width: 18
   :align: middle

.. image:: spacr/resources/icons/workflow_home_apps.png
   :alt: All other spaCR applications grouped as they appear on the home screen
   :align: center

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

See the `installer guide <docs/source/installers.rst>`_ for update, uninstall,
offline and troubleshooting instructions.

Python installation
~~~~~~~~~~~~~~~~~~~

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
``spacr[ome-zarr]``, ``spacr[omero]``, ``spacr[napari]`` and
``spacr[czi,nd2,lif]``. See the `installation guide
<docs/source/installers.rst>`_ for the complete extras and Python-version
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
to ``~/.spacr/logs/spacr.log``. The classic Tk interface remains available as
``spacr-legacy`` but is no longer developed.


What you can do
---------------

Most screens follow six modules:

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


Contributing and support
------------------------

Bug reports and focused feature requests are welcome through
`GitHub Issues <https://github.com/EinarOlafsson/spacr/issues>`_.
When reporting a failure, include the spaCR version, operating system, Python
version, module settings and the relevant log excerpt. ``spacr-doctor``
collects most of that for you.

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
