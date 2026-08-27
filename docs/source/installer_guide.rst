.. _installer-guide:

Installer guide
===============

This page covers the current desktop installers, conda-forge and PyPI
installation, updates, removal, offline preparation and the files to check
when installation fails. For older downloadable versions, use the
:doc:`installer archive <installers>`.

Choose an installation
----------------------

Use a desktop installer when you want an application launcher and a private
Python environment. The installer does not modify an existing Python
installation. It downloads a managed Python 3.12 runtime and the exact spaCR
version named by the installer, then checks the environment before replacing
an existing working installation.

Use the official `conda-forge package <https://anaconda.org/conda-forge/spacr>`_
when Conda should install spaCR and resolve its desktop and
dependencies. Use ``pip`` for the PyPI release when spaCR must live in an
existing Python environment, notebook, server or cluster, or when you need a
PyPI extra that is not part of the conda package. Python 3.12 currently offers
the widest selection of optional scientific packages.

Desktop installers
------------------

Download the current installer from the `spaCR README
<https://github.com/EinarOlafsson/spacr#install-spacr>`_. The installers require
an internet connection while they create the private environment.

Windows 10/11
~~~~~~~~~~~~~

Run ``SpaCR-<version>-Windows-Online-Setup.exe``. The default per-user
location is ``%LOCALAPPDATA%\spaCR`` and does not require administrator
access. Automatic hardware acceleration is selected by default. It installs a
CUDA-capable PyTorch build on compatible NVIDIA systems and falls back safely
elsewhere. Clear the component only when you require the smaller CPU-only
installation.

macOS 11 or later
~~~~~~~~~~~~~~~~~

Open ``SpaCR-<version>-macOS-Universal-Online.pkg``. The application is placed
in ``/Applications/spaCR.app``. On first launch, a visible Terminal bootstrap
creates the private runtime under ``~/Library/Application Support/spaCR``.
The current beta is not notarized. If Gatekeeper blocks it, open **System
Settings → Privacy & Security** and choose **Open Anyway** for spaCR.

Linux x86-64
~~~~~~~~~~~~

Make the downloaded installer executable and run it:

.. code-block:: bash

   chmod +x SpaCR-*-Linux-x86_64-Online.run
   ./SpaCR-*-Linux-x86_64-Online.run

The default installation root is ``~/.local/share/spacr``. The launcher is
written to ``~/.local/bin/spacr`` and the desktop entry to
``~/.local/share/applications``. Add ``~/.local/bin`` to ``PATH`` if your shell
does not already include it.

Automatic backend selection is the Linux default. It installs CUDA support
when compatible NVIDIA hardware is available. To require the smaller CPU-only
build instead, run the installer with ``--torch-backend cpu``.

Updating
--------

Download and run the installer for the newer version. Installation is staged
and validated before it replaces the active private environment; a failed
update leaves the previous working environment in place. Project folders and
results are not stored in the installation directory and are not removed by
an update.

Update an environment installed from conda-forge with:

.. code-block:: bash

   conda update conda-forge::spacr

Update an environment installed from PyPI with:

.. code-block:: bash

   python -m pip install --upgrade "spacr[qt]"

For reproducible work, install an exact version instead of following the
latest release. Use the command for the package source already installed in
the environment:

.. code-block:: bash

   conda install conda-forge::spacr=1.5.0.4
   python -m pip install "spacr[qt]==1.5.0.4"

Uninstalling
------------

* **Windows:** open **Settings → Apps → Installed apps → spaCR → Uninstall**,
  or run ``%LOCALAPPDATA%\spaCR\Uninstall.exe``.
* **macOS:** run
  ``/Library/Application Support/spaCR/uninstall-spacr.sh`` in Terminal. This
  removes the application, command launcher and system installer support.
  Remove ``~/Library/Application Support/spaCR`` separately to delete the
  per-user private runtime.
* **Linux:** run ``~/.local/share/spacr/uninstall-spacr.sh``. This removes the
  launcher, desktop entry and private environment.
* **conda-forge:** activate the environment and run ``conda remove spacr``.
* **PyPI:** activate the environment and run
  ``python -m pip uninstall spacr``.

Remove the environment itself if it was created only for spaCR.

Uninstalling does not delete microscopy projects, databases or exported
results. User preferences, run records and logs under ``~/.spacr`` are also
left in place so they can be inspected or reused. Remove that directory
separately only if those records are no longer needed.

Offline installation
--------------------

The small desktop installers are online installers and cannot complete
without network access. For an offline workstation, prepare a wheel directory
on a networked machine with the same operating system, architecture and Python
minor version:

.. code-block:: bash

   python -m pip download --dest spacr-wheelhouse "spacr[qt]==1.5.0.4"

Copy ``spacr-wheelhouse`` to the offline machine, create and activate a Python
environment, then install without contacting a package index:

.. code-block:: bash

   python -m pip install --no-index --find-links spacr-wheelhouse \
       "spacr[qt]==1.5.0.4"

Repeat the download for the required optional extras. GPU-enabled PyTorch
builds may require a separate wheel source, so prepare and test the complete
wheelhouse on a matching connected machine before moving it to an isolated
system.

Conda-forge installation
------------------------

Install the official conda-forge package directly into an activated
environment. It includes spaCR's desktop and core dependencies:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   conda install conda-forge::spacr
   spacr

PyPI installation and extras
----------------------------

The PyPI package supports Python 3.9 through 3.14 except Python 3.14.1. To
install the PyPI release and desktop interface inside a Conda environment:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install "spacr[qt]"

Omit ``qt`` for a headless server. Extras can be combined, for example
``spacr[qt,czi,nd2,lif]``. Common additions are:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Extra
     - Adds
   * - ``qt``
     - The PySide6 desktop interface.
   * - ``czi``, ``nd2``, ``lif``
     - Additional microscopy file readers.
   * - ``napari``
     - Transfer images and masks to napari.
   * - ``anndata``
     - AnnData and ``.h5ad`` export support.
   * - ``omero``
     - OMERO import support.
   * - ``trackastra``, ``btrack``, ``ultrack``
     - Optional tracking backends.
   * - ``boosting``
     - CatBoost and LightGBM classifiers.
   * - ``numpyro``, ``pymc``
     - Optional Bayesian regression backends.
   * - ``rapids``
     - RAPIDS acceleration where compatible CUDA wheels are available.
   * - ``tutorial``
     - Packages used by the interactive tutorial environment.

Troubleshooting
---------------

The desktop installers write ``install.log`` inside their private installation
root. Windows also writes ``nsis-bootstrap-status.txt`` if the wrapper fails
before the Python bootstrap starts. Runtime logs are under
``~/.spacr/logs/spacr.log`` (the equivalent home directory on Windows).

For a Python installation, run:

.. code-block:: bash

   python -m pip check
   python -c "import spacr; print(spacr.__version__)"
   spacr-doctor

Include the installer version, operating system, ``install.log`` and the
output of ``spacr-doctor`` when filing a `GitHub issue
<https://github.com/EinarOlafsson/spacr/issues>`_.
