.. _installer-guide:

Installer guide
===============

This page covers the current desktop installers, conda-forge, PyPI and
container installation, updates, removal, offline preparation and the files
to check when installation fails. For older downloadable versions, use the
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

Use a :ref:`container image <container-images>` when the install itself is the
problem: a cluster node, a cloud instance, a shared machine you cannot change,
or an analysis that has to be re-runnable years from now. The images are for
the CLI and the pipelines; the desktop interface in a container is a Linux-only
extra and is documented as one.

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

Older installations are removed first
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before it installs, every installer -- and **Help → Check for updates** --
looks for spaCR installations an earlier installer made and removes them, so
two versions never sit side by side with one of them on the path. It finds
the Windows online and offline installs, the macOS application and its
per-user environment, the Linux online install and the Debian package,
together with their launchers, shortcuts, menu entries and uninstall
registrations. Environments you created yourself and source checkouts are
listed but never removed, and your project folders and results are never
touched. If a copy cannot be removed -- for example a macOS application in
``/Applications`` that needs an administrator -- the installer says which,
and installs nothing, so you are never left with half an update.

To see what it would find without changing anything, run the finder from a
spaCR checkout: ``python spacr/install_cleanup.py find``.

Recovering an older desktop installation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Desktop builds at version 1.5.0.1 and earlier, plus the Windows 1.5.0.4
build, can try ``python -m pip`` even though their private environment has no
``pip``. If **Help → Check for updates** reports ``No module named pip``, run
the command for the original installation below. It uses the installer's
private ``uv`` executable to update that same environment; it needs neither
administrator access nor a reinstall.

Linux:

.. code-block:: bash

   ~/.local/share/spacr/bootstrap/uv pip install --upgrade --python ~/.local/share/spacr/venv/bin/python spacr

macOS:

.. code-block:: bash

   "$HOME/Library/Application Support/SpaCR/bootstrap/uv" pip install --upgrade --python "$HOME/Library/Application Support/SpaCR/venv/bin/python" spacr

Windows PowerShell:

.. code-block:: powershell

   & "$env:LOCALAPPDATA\SpaCR\bootstrap\uv.exe" pip install --upgrade --python "$env:LOCALAPPDATA\SpaCR\venv\Scripts\python.exe" spacr

These are the original installers' default roots. If a different destination
was selected during installation, replace the root before ``bootstrap`` and
``venv`` with that destination. Installers built from version 1.5.0.5 and
later carry the corrected updater and do not depend on ``python -m pip``.

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

.. _container-images:

Container images
----------------

Two images are published to the GitHub Container Registry as part of every
spaCR release, after that version reaches PyPI. Each one is built and then
checked before it is pushed — it must report the version its tag claims, it
must not be running as root, and it must complete one real pipeline — so an
image that exists is an image that ran. An image that fails a check is not
published, and the release run that built it is red.

Images begin with the first release made after this page described them.
Older versions have no image, and ``docker pull`` will say so rather than
give you something unrelated.

They exist for the headless half of spaCR: the CLI, the pipelines, a cluster
job and a reviewer re-running an analysis a year later. They are not a way to
install the desktop application, which the platform installers above do
better.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Image
     - For
   * - ``ghcr.io/einarolafsson/spacr:<version>``
     - CPU only. Works on any x86-64 host with Docker or Podman, needs no
       driver, and runs the measure and regression half of spaCR at full
       speed.
   * - ``ghcr.io/einarolafsson/spacr:<version>-cuda12.4``
     - CUDA 12.4. Needs an NVIDIA driver of **550 or newer on the host** and
       the NVIDIA Container Toolkit. Without ``--gpus`` it behaves as the CPU
       image.

``:latest`` and ``:cpu`` follow the newest CPU release; ``:cuda`` and
``:cuda12.4`` follow the newest CUDA release. Name an exact version for
anything you intend to reproduce.

Models and data are mounted, never baked in. A cpsam checkpoint is about
1.2 GB and goes stale between releases, so the image ships none: mount a
folder on ``/models`` and it becomes both the folder the Model Zoo downloads
into and the folder Cellpose loads from. SAMCell and DINOCell are not in the
images either, because they pin PyTorch versions that conflict with spaCR's
and with each other; install one inside a running container, or let the Model
Zoo install it into an isolated environment of its own.

Running a pipeline
~~~~~~~~~~~~~~~~~~

``<version>`` below is a spaCR version that has a published image — the
`GHCR package page <https://github.com/EinarOlafsson/spacr/pkgs/container/spacr>`_
lists them. ``:latest`` takes the newest CPU image if you do not need a
particular one.

.. code-block:: bash

   docker pull ghcr.io/einarolafsson/spacr:<version>

   docker run --rm \
       --user "$(id -u):$(id -g)" \
       -v "$PWD/screen:/data" \
       -v "$HOME/.cellpose/models:/models" \
       ghcr.io/einarolafsson/spacr:<version> \
       spacr-run measure --settings /data/settings/measure_settings.csv

On a GPU host, add ``--gpus all`` and use the CUDA tag:

.. code-block:: bash

   docker run --rm --gpus all \
       --user "$(id -u):$(id -g)" \
       -v "$PWD/screen:/data" \
       -v "$HOME/.cellpose/models:/models" \
       ghcr.io/einarolafsson/spacr:<version>-cuda12.4 \
       spacr-run mask --settings /data/settings/gen_mask_settings.csv

Pass ``--user "$(id -u):$(id -g)"``. Every file a container writes to a
mounted folder is owned by the user ID inside the container, so without it
the results belong to a user that does not exist on the host and cannot be
deleted without ``sudo``. The images already run as a non-root user, and the
entrypoint moves the cache directories somewhere writable when the user ID
you pass has no home inside the image.

``spacr-run --list`` prints every module that runs headless, and
``spacr-run --describe <module>`` prints what one needs and what it writes.
``spacr-doctor`` reports what the container found, including whether the GPU
is visible:

.. code-block:: bash

   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:<version>-cuda12.4 spacr-doctor

The desktop interface in a container
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Linux only, and unsupported elsewhere.** The images carry the Qt runtime
libraries, so a Linux host running X11 can pass its display socket in:

.. code-block:: bash

   xhost +SI:localuser:"$(id -un)"
   docker run --rm \
       --user "$(id -u):$(id -g)" \
       -e DISPLAY \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v "$PWD/screen:/data" \
       ghcr.io/einarolafsson/spacr:<version> \
       spacr

Do not pass the host's ``XDG_RUNTIME_DIR`` in. That path does not exist
inside the container, and Qt complains about it on every start; the image
makes its own runtime directory under the container's cache folder instead.

A container rarely has a usable OpenGL context, so the animated backdrop may
not draw. ``safespacr`` starts the same application with the backdrop and GL
switched off, and is the right command when the window is slow or blank:

.. code-block:: bash

   docker run --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
       ghcr.io/einarolafsson/spacr:<version> safespacr

On macOS and Windows this needs a third-party X server and is not tested or
supported. Use the desktop installer on those platforms.

Building the images yourself
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build from the repository root; the Dockerfiles expect the checkout as their
build context:

.. code-block:: bash

   docker build -f packaging/docker/Dockerfile.cpu -t spacr:cpu .
   docker build -f packaging/docker/Dockerfile.cuda -t spacr:cuda .

``--build-arg SPACR_UID=$(id -u) --build-arg SPACR_GID=$(id -g)`` bakes your
own user ID into the image, which is an alternative to passing ``--user`` on
every run. ``--build-arg PYTHON_VERSION=3.11`` selects a different
interpreter. Each image can be checked the way the release workflow checks
it:

.. code-block:: bash

   docker run --rm spacr:cpu spacr --version
   docker run --rm spacr:cpu python3 /opt/spacr/smoke_pipeline.py

The second command runs a real pipeline on a synthetic field and reports one
line per check; it needs no model, no GPU and no network.

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
