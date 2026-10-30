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
when Conda should install spaCR and resolve its desktop and core
dependencies. Use ``pip`` for the PyPI release when spaCR must live in an
existing Python environment, notebook, server or cluster, or when you need a
PyPI extra that is not part of the conda package. Python 3.12 currently offers
the widest selection of optional scientific packages.

Use a :ref:`container image <container-images>` when spaCR cannot be
installed directly on the host, such as a cluster node, a cloud instance or a
shared machine you cannot change, or when an analysis must be rerun later with
the same software versions. The images are for the CLI and the pipelines;
running the desktop interface in a container is supported only on Linux and is
described separately below. On a cluster without Docker, build the same
image as an :ref:`Apptainer or SingularityCE file <apptainer-images>`. For a
workstation that has no network access at all, use an
:ref:`offline installer bundle <offline-bundle>`.

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
CUDA-capable PyTorch build on compatible NVIDIA systems. On other systems it
installs the PyTorch build that ``uv`` selects for the detected hardware, which
is the CPU build when no supported GPU is detected. Clear the component only
when you require the smaller CPU-only installation. The unticked component
**Install without wgpu (use OpenGL/CPU for 3D)** leaves out the wgpu 3D
renderer; see :ref:`wgpu-3d-rendering`. From PowerShell, the same choice is
the ``-WithoutWgpu`` switch of ``install_spacr_windows.ps1``.

macOS 11 or later
~~~~~~~~~~~~~~~~~

Open ``SpaCR-<version>-macOS-Universal-Online.pkg``. The application is placed
in ``/Applications/spaCR.app``. The spaCR setup window creates its private
runtime under ``~/Library/Application Support/spaCR`` and displays progress,
dependency choices and any required consent. Complete those choices in the
window; setup does not require answers in Terminal. When setup finishes,
spaCR opens its startup window and then the main application.
The current beta is not notarized. If Gatekeeper blocks it, open **System
Settings → Privacy & Security** and choose **Open Anyway** for spaCR. The
first-launch window has an unchecked **Install without wgpu (use OpenGL/CPU
for 3D)** box; see :ref:`wgpu-3d-rendering`.

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

To leave out the wgpu 3D renderer, run the installer with ``--without-wgpu``,
or tick **Install without wgpu (use OpenGL/CPU for 3D)** in the installer
window; the box is unchecked by default. On a Linux system whose glibc is
older than 2.28 the installer leaves wgpu out by itself and prints one line
saying so, because wgpu publishes Linux wheels only for glibc 2.28 and newer.
``--dry-run`` shows the decision in its plan, as the ``wgpu 3D rendering``
line. See :ref:`wgpu-3d-rendering`.

Linux ARM64 build requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Linux ARM64 environments build ModernGL's GLContext backend from source. This
requires a C++ compiler and the X11 development headers. The Unix installer's
normal system-dependency step installs these packages on ARM64:

.. list-table::
   :header-rows: 1

   * - Package manager
     - Build packages
   * - APT
     - ``build-essential libx11-dev``
   * - DNF or Zypper
     - ``gcc-c++ libX11-devel``
   * - Pacman
     - ``base-devel libx11``

When using ``--skip-system-deps`` or installing with ``pip`` directly, provide
these build packages beforehand alongside the required desktop runtime
libraries. The managed Python already includes its own Python headers; for a
system Python, provide that interpreter's development headers as well.

Updating
--------

Choose **Help → Check for updates**, review the offered version and choose
**Update**. Supported macOS installations and dedicated private Python
installations prepare the replacement in the background. When it is ready,
choose **Restart** to finish closing the current application and open the
verified replacement through the spaCR startup window. No manual click on the
application icon is needed. Frozen Windows and Debian installations use their
native authorization and replacement workflows.

Older Linux and Windows online layouts without a verified update adapter are
refused before replacement starts. The application stays open and keeps its
data. Use the matching release installer for those installations.

Alternatively, download and run the installer for the requested version from
the `GitHub releases <https://github.com/EinarOlafsson/spacr/releases>`_. This
is also the way to move an older installer to the current desktop update flow.
Use the installer for the same platform and installation family as the copy
being replaced.

Installation is staged and validated before replacement. An unsuccessful
preparation keeps the previous application; activation failures retain or
restore the previous runtime and report any recovery needed. Ordinary updates
keep models, caches, preferences, run history and project outputs. Optional
update cache cleanup starts unselected and requires a separate confirmation.

Save unsaved work before choosing **Restart**. If a screen refuses closure or
a worker is still stopping, spaCR keeps the window open and does not authorize
the replacement. Resolve the unsaved changes or wait for the worker to finish,
then choose **Help → Check for updates** again to retry the update. A refused
restart cancels that update handoff.

Update channel and What's new
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Preferences → Modules → Update channel** chooses what **Help → Check for
updates** offers. *Stable releases* (the default) offers PyPI's latest
release. *Nightly builds* also offers pre-releases and development builds,
which arrive sooner and are tested less; withdrawn (yanked) versions are never
offered. After an update, the next launch opens **What's new in spaCR** with the
release notes of every version between the previous and the running one: the
notes bundled with the build, plus the published GitHub releases when
**Release news** is on and GitHub can be reached. **Help → What's new…** reopens
it at any time.

Proxies and corporate certificates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every download spaCR makes (model weights, updates, plug-ins, pip, uv, conda
and the segmentation-backend installers) honours ``HTTPS_PROXY`` and
``REQUESTS_CA_BUNDLE`` (or ``SSL_CERT_FILE``). To set them inside spaCR
instead, fill in **Preferences → Modules → Proxy** (for example
``http://proxy.example.org:3128``) and **Certificate bundle** (a PEM file from
your IT department). These values override the environment, are
kept in ``~/.spacr/network.json`` so command-line runs use them too, and are
passed on as ``HTTPS_PROXY``, ``HTTP_PROXY``, ``REQUESTS_CA_BUNDLE``,
``SSL_CERT_FILE``, ``CURL_CA_BUNDLE``, ``PIP_CERT`` and ``GIT_SSL_CAINFO``.
``spacr-doctor`` has a *proxy and certificates* row. It checks that the
certificate bundle exists and can be read, and that PyPI answers through the
configured proxy. A password in the proxy address is masked in its output.

macOS online installation: update and reopen
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose **Help → Check for updates**, accept the available update, and confirm
the requested version. spaCR prepares a separate private runtime using the
installation's bundled ``uv`` while the window remains responsive. After
preparation succeeds, **Restart** activates the verified runtime and opens its
startup window and main window. The existing launcher is retained. No remove
prompt, manual icon relaunch or Terminal questionnaire is part of this flow.

If preparation or verification fails, the application stays on the previous
version and shows the failure. If an analysis or screen prevents safe closure,
the replacement is not authorized. Details are retained in
``~/.spacr/logs/update.log``. A frozen application bundle uses its verified
bundle replacement workflow instead of updating a Python environment.

Supported frozen macOS, Windows and Debian installations prepare their
replacement while the current application remains open. Protected installations
request the operating system's graphical administrator authorization before
offering **Restart**. Cancellation keeps the current application open. Activation
waits for safe closure and actual process exit; the new application must confirm
its version, executable and main window before recovery files are retired.

Windows automatic updates require the release's checksummed
``spaCR-VERSION-update.exe``. The ordinary ``spaCR-VERSION-setup.exe`` remains
the interactive installation download and is never substituted by the updater.
Protected macOS updates require an installed helper that advertises the current
native phase protocol. Older helpers, missing update assets and unsupported
legacy Linux or Windows online layouts report the limitation while keeping the
application open. Use the matching ordinary installer from GitHub releases to
bring such an installation forward; shared and Conda environments use their
package manager.

Reinstalling the same version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Download that version's matching desktop installer from GitHub releases and
run it again. Reinstallation repairs the application code and launcher while
preserving user data. Do not uninstall first merely to repair an installation.
Check the version in **Help → About spaCR** after the application reopens; the
version must match the installer you selected.

Existing installations and full installers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The full installers and other installer update paths look for spaCR
installations made by another spaCR installer and remove them, so that only one
installer-made copy remains. They find
the Windows online and offline installs, the macOS application and its
per-user environment, the Linux online install and the Debian package,
together with their launchers, shortcuts, menu entries and uninstall
registrations. Environments you created yourself and source checkouts are
listed but never removed, and your project folders and results are never
touched. If a copy cannot be removed -- for example a macOS application in
``/Applications`` that needs an administrator -- the installer names it
and does not install the new version.

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

   python -m pip install --upgrade spacr

For reproducible work, install an exact version instead of following the
latest release. Use the command for the package source already installed in
the environment, replacing ``VERSION`` with a release that source publishes:

.. code-block:: bash

   conda install conda-forge::spacr=VERSION
   python -m pip install "spacr==VERSION"

``python -m pip index versions spacr`` lists the PyPI releases, and
``conda search -c conda-forge spacr`` lists the conda-forge builds.
conda-forge can trail PyPI by a release or more, so check the source you
install from.

Uninstalling
------------

Open **Preferences → Performance → Uninstall**. In the popup, choose either
**Current version** or **All detected spaCR versions**. The preview identifies
the installation paths and versions. The all-versions choice also finds spaCR
in detected Python and Conda environments; it removes the spaCR package from
those environments and keeps their other packages and files. Editable source
checkouts are shown as retained.

The popup separately selects **Caches**, **Models**, **Logs**, **Settings**,
**History**, **Temporary files**, **Downloads** and **Other spaCR data**.
Every cleanup category starts selected for uninstall. Clear any category to
keep it, inspect the exact paths listed, and confirm the uninstall. Cancel
leaves the application and files unchanged. spaCR closes safely before its
running installation is removed.

Supported protected macOS applications, Windows frozen installations and Debian
packages request graphical administrator authorization during preparation,
while spaCR remains open. The helper must be installed, verified and advertise
the required uninstall protocol. Declining authorization or cancelling the
popup leaves the application and files intact. Removal waits for confirmed safe
closure and actual process exit.

If a selected installation has an older unsupported helper, or a selected data
path cannot be removed with the available permissions, preparation stops before
spaCR closes and nothing is removed. The preview still identifies the affected
paths. The native alternatives below remain available for those installations.

These categories cover identified spaCR-owned locations. Shared external
caches, unverified or linked folders and arbitrary project folders are kept;
the preview marks retained locations. History includes spaCR's run records and
any output copies stored within those records. Keep History if those copies
are still needed. Projects, databases and exports outside the listed spaCR
locations are not removed.

Uninstall cleanup defaults differ from update defaults: an ordinary update
preserves user data, while the uninstall popup selects the displayed cleanup
categories until you opt out. The native and command-line alternatives below
retain their existing explicit purge switches.

Native and command-line alternatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Windows:** open **Settings → Apps → Installed apps → spaCR → Uninstall**,
  or run ``%LOCALAPPDATA%\spaCR\Uninstall.exe``.
* **macOS:** run
  ``/Library/Application Support/spaCR/uninstall-spacr.sh`` in Terminal. This
  removes the application, command launcher and system installer support.
  Add ``--dry-run`` to preview the software paths without removing anything.
  The command launcher is removed only if it is spaCR's exact installed
  symlink; unrelated files and links are kept.
  Remove ``~/Library/Application Support/spaCR`` separately to delete the
  per-user private runtime.
* **Linux:** run ``~/.local/share/spacr/uninstall-spacr.sh``. This removes the
  launcher, desktop entry and private environment.
* **conda-forge:** activate the environment and run ``conda remove spacr``.
* **PyPI:** activate the environment and run
  ``python -m pip uninstall spacr``.

These package-manager commands remove the package from the selected
environment. They do not remove the whole environment. The native and
command-line alternatives keep user preferences, run records and logs unless
their explicit purge option is used.

Clean uninstall (purge)
~~~~~~~~~~~~~~~~~~~~~~~

A purge is opt-in. It also deletes spaCR's caches, backend environments and
user data: ``~/.spacr`` (logs, run records, models, backends, news),
``~/.cache/spacr``, ``~/.config/spacr``, ``~/.local/state/spacr``,
``~/spacr-demos``, ``~/spacr-tutorials``, the macOS preferences file, the
Windows ``HKCU\Software\spacr\qt`` settings, a backends folder named by
``SPACR_BACKENDS_DIR`` and every cache moved from **Preferences → Storage**.
It lists everything first and deletes only after you type ``purge`` (or pass
``--yes``). Shared caches such as Hugging Face's and Torch's own folders, and
your projects, are never touched.

* **Windows:** run ``%LOCALAPPDATA%\spaCR\Uninstall.exe /PURGE``.
* **macOS:** run ``uninstall-spacr.sh --purge`` as above.
  Add ``--dry-run`` to preview both software and data removal. A refused or
  failed purge stops the uninstall before the application is removed.
* **Linux:** run ``~/.local/share/spacr/uninstall-spacr.sh --purge``.
* **Any install, before uninstalling:** ``python -m spacr.install_cleanup
  purge`` (add ``--dry-run`` to list only).

.. _portable-mode:

Portable mode
-------------

Portable mode keeps everything spaCR writes for a user next to the
application instead of in the home folder: preferences, caches, logs, run
records, model downloads, plugins and segmentation backends. Use it to run
spaCR from a USB drive or an external disk, or on a shared computer where
nothing should be left behind.

Turn it on in either of two ways:

* **Marker file.** Create an empty file named ``spacr-portable`` in the
  spaCR install folder, which is the folder that holds spaCR's Python
  environment. On Windows that is ``%LOCALAPPDATA%\spaCR``; on macOS
  ``~/Library/Application Support/spaCR``; on Linux
  ``~/.local/share/spacr``. In a conda or virtual environment, the
  environment folder itself also works.
* **Environment variable.** Set ``SPACR_PORTABLE=1`` to keep data next to
  the application, or ``SPACR_PORTABLE=/path/to/folder`` to choose the
  folder. ``SPACR_PORTABLE=0`` turns portable mode off even when a marker
  file exists.

spaCR then keeps its data in a ``spacr-data`` folder beside the marker (or
inside the chosen folder): ``settings`` holds the preferences as an INI file
on every platform, so nothing is written to the Windows registry or the macOS
preference files, and ``logs``, ``runs``, ``backends``, ``plugins``,
``models`` and ``cache`` hold the rest. **Help → About spaCR** shows the
folder in use while portable mode is on. Restart spaCR after creating or
removing the marker.

Portable mode is one switch for the folder variables that can also be set
one at a time. It fills in any of them that are not already set, and a
variable you set yourself always wins:

``SPACR_HOME``
   Preferences, run records and every other per-user spaCR folder.
``SPACR_LOG_DIR``, ``SPACR_BACKENDS_DIR``, ``SPACR_PLUGIN_HOME``
   Logs, segmentation backend environments and plugins.
``XDG_CACHE_HOME``, ``XDG_STATE_HOME``, ``TORCH_HOME``, ``HF_HOME``, ``MPLCONFIGDIR``, ``CELLPOSE_LOCAL_MODELS_PATH``
   Caches of example data, chained settings, PyTorch, Hugging Face,
   Matplotlib and Cellpose models.

Existing data in ``~/.spacr`` is not moved. Copy it into ``spacr-data`` first
to keep it.

.. _offline-bundle:

Offline installation
--------------------

The small desktop installers are online installers and cannot complete
without network access. For a locked-down microscope PC, build an offline
bundle on a networked machine. It holds everything the installer would
download: the pinned ``uv`` tool, the managed Python 3.12 runtime, every wheel
of the locked environment, Cellpose weights (``cpsam`` by default), optional
Mask test data, ``bundle.json`` and a ``SHA256SUMS`` file. Bundles target
``linux-x86_64``, ``windows-x86_64`` or ``macos-arm64`` and can be built for
another platform than the one building them. The PyTorch wheel line is fixed
when the bundle is built: ``cpu``, or a CUDA line such as ``cu126``.

From a spaCR checkout, on the networked machine:

.. code-block:: bash

   python packaging/offline/build_offline_bundle.py --platform linux-x86_64 \
       --torch-backend cpu --out dist/offline \
       --test-data ~/.cache/spacr/example_data/plate1 --archive

This writes the folder ``spaCR-VERSION-Linux-x86_64-Offline-cpu`` under
``dist/offline`` and, with ``--archive``, the same folder as one ``.tar``.
``--extras`` chooses optional spaCR extras (none by default); the standard
package includes the Qt desktop interface. ``--cellpose-model``
adds Cellpose weights and can be repeated, ``--test-fields`` limits the
number of test fields, and ``--from-source .`` packs spaCR built from the
checkout instead of the PyPI release.

Copy the bundle to the offline machine and run its installer with the bundle
folder. On Linux or macOS:

.. code-block:: bash

   ./install.sh --offline-bundle "$PWD" --check-mask

On Windows, in PowerShell inside the bundle folder:

.. code-block:: powershell

   .\install.ps1 -OfflineBundle . -CheckMask

The installer checks every file against ``SHA256SUMS``, installs Python and
the wheels with no package index, and copies the Cellpose weights without
overwriting existing ones. ``--check-mask`` (``-CheckMask``) then runs Mask on
a scratch copy of the bundled test data and passes when masks and merged
stacks are written; on a CPU this can take an hour for two fields.

What the bundle does not contain: the separate environments of optional
segmentation backends such as Cellpose 3 or StarDist, and Model Zoo models
other than the Cellpose weights you chose; those still need a network. On
Linux the system Qt libraries are not bundled, so a minimal system without
them runs spaCR headless only. ``SHA256SUMS`` detects damaged files, but the
bundle is not signed, so transfer it through a channel you trust.

Install into an existing Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To install spaCR into an existing Python environment offline, prepare a wheel directory
on a networked machine with the same operating system, architecture and Python
minor version, replacing ``VERSION`` with the release to install:

.. code-block:: bash

   python -m pip download --dest spacr-wheelhouse "spacr==VERSION"

Copy ``spacr-wheelhouse`` to the offline machine, create and activate a Python
environment, then install without contacting a package index:

.. code-block:: bash

   python -m pip install --no-index --find-links spacr-wheelhouse \
       "spacr==VERSION"

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

Python 3.15 on Linux
~~~~~~~~~~~~~~~~~~~~

Use the nightly source in a new Linux Conda environment for Python 3.15.
The current PyPI release excludes this interpreter and pip can fall back
to an older spaCR release. Install the Linux build packages listed above
before running this recipe; some native dependencies still build from source.

.. code-block:: bash

   conda create -n spacr15 --override-channels -c conda-forge python=3.15 pip cvxopt tabmat h5py "libsqlite=3.54.1=h0737f62_101" -y
   conda activate spacr15
   python -m pip install --upgrade pip
   python -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   python -m pip install "git+https://github.com/EinarOlafsson/spacr.git@nightly"
   python -m pip install --force-reinstall --only-binary=:all: "scipy>=1.18,<2"
   python -m pip check
   spacr

For a local source checkout, replace the Git URL with ``.``. The selected SQLite build and SciPy wheel
reinstallation prevent a C++ runtime mismatch between Conda libraries and the
PyPI Torch/Qt libraries; complete all Conda installations before these pip
steps. The CPU Torch command does not install a CUDA runtime.

Python 3.15 uses scikit-image 0.27.0rc1 (a release candidate) or newer below
0.28 because that line provides its image-processing wheels. The optional
pyfixest, CatBoost, PyMC and Piper speech packages are not installed by default
on this interpreter because compatible upstream binaries or dependencies are
missing. Existing Python 3.14
requirements and scientific defaults are unchanged. This recipe is qualified
on Linux x86_64; Windows is untested and macOS retains its older Qt requirement.

PyPI installation and extras
----------------------------

The PyPI package supports Python 3.9 through 3.14 except Python 3.14.1.
Choose either a Python virtual environment or a Conda environment for the
PyPI release. In both cases, pip installs spaCR; the conda-forge route above
uses Conda to install spaCR instead.

For a Python virtual environment, first install Python 3.12, then run:

.. code-block:: bash

   python3.12 -m venv spacr-venv

Activate the environment on Linux or macOS with:

.. code-block:: bash

   source spacr-venv/bin/activate

On Windows, create it with ``py -3.12 -m venv spacr-venv`` and activate it in
PowerShell with:

.. code-block:: powershell

   .\spacr-venv\Scripts\Activate.ps1

In the activated environment, install and open spaCR:

.. code-block:: bash

   python -m pip install --upgrade pip
   python -m pip install spacr
   spacr

To install the same PyPI release and desktop interface inside a Conda
environment:

.. code-block:: bash

   conda create -n spacr python=3.12 -y
   conda activate spacr
   python -m pip install --upgrade pip
   python -m pip install spacr

The standard package includes the Qt desktop interface and command-line
pipelines. On a headless server, use ``spacr-run`` without opening the desktop
application. Extras can be combined, for example ``spacr[czi,nd2,lif]``.
Common additions are:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Extra
     - Adds
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

.. _wgpu-3d-rendering:

3D rendering and wgpu
---------------------

spaCR installs `wgpu <https://github.com/pygfx/wgpu-py>`_ by default where it
publishes wheels. wgpu draws the **True 3D** view through Vulkan, Metal or
Direct3D 12. Without wgpu, spaCR draws 3D through OpenGL (VisPy), and through
the CPU when OpenGL is not available either. The **Render backend** list offers
Auto, Vulkan, Metal, Direct3D 12, OpenGL and CPU, showing only the backends
this computer supports. **Auto** prefers wgpu (Vulkan on Linux and Windows,
Metal then available Vulkan on macOS, Direct3D 12 on Windows when Vulkan
is unavailable), then OpenGL, then the CPU.

.. list-table::
   :header-rows: 1
   :widths: 34 30 36

   * - Installation
     - wgpu
     - True 3D view uses
   * - Python 3.10 on Windows x86-64/ARM64, macOS Intel/Apple silicon, or
       Linux x86-64/aarch64 with glibc 2.28 or newer
     - Installed, ``wgpu<=0.31.0``
     - Vulkan, Metal or Direct3D 12 (Auto), else OpenGL, else CPU
   * - Python 3.11 or newer on the same platforms (the desktop installers
       use Python 3.12)
     - Installed, ``wgpu<0.33``
     - Vulkan, Metal or Direct3D 12 (Auto), else OpenGL, else CPU
   * - Python 3.9
     - Not installed
     - OpenGL, else CPU
   * - Other processors (for example 32-bit x86, ppc64le, s390x)
     - Not installed
     - OpenGL, else CPU
   * - Linux with glibc older than 2.28 (installers skip wgpu automatically)
     - Not installed
     - OpenGL, else CPU
   * - Installed without wgpu (installer option, or the commands below)
     - Not installed
     - OpenGL, else CPU

The desktop installers skip wgpu when it is resolved, so it is never
downloaded: they pass ``uv`` an override file containing the single line
``wgpu; sys_platform == "never"``. Because the marker never holds, the
resolver drops wgpu, and with it ``rendercanvas``, which only wgpu needs.

With pip, install spaCR as usual and then remove wgpu:

.. code-block:: bash

   python -m pip install spacr
   python -m pip uninstall -y wgpu rendercanvas

``pip check`` then reports that spaCR's ``wgpu`` requirement is missing; this
is expected, and spaCR draws 3D through OpenGL or the CPU.

On Linux with glibc older than 2.28 (check with ``getconf GNU_LIBC_VERSION``),
``pip install spacr`` finds no wgpu wheel and tries to build wgpu from its
source release, which fails. pip cannot drop a dependency while it resolves,
so install with `uv <https://docs.astral.sh/uv/>`_ and the same override the
installers use, inside the activated environment:

.. code-block:: bash

   python -m pip install uv
   printf '%s\n' 'wgpu; sys_platform == "never"' > no-wgpu.txt
   python -m uv pip install --override no-wgpu.txt spacr

Use the same ``--override no-wgpu.txt`` when upgrading spaCR in that
environment. Add ``--dry-run`` to the last command to see the resolved
packages, without wgpu, before anything is installed.

.. _container-images:

Container images
----------------

Two images are published to the GitHub Container Registry as part of every
spaCR release, after that version reaches PyPI. Each one is built and then
checked before it is pushed: it must report the version its tag claims, it
must not run as root, and it must complete one pipeline run on a synthetic
field. An image that fails a check is not published, and the workflow run that
built it fails.

The public `GHCR package page
<https://github.com/EinarOlafsson/spacr/pkgs/container/spacr>`_ lists available
versions. The commands below name the image tags of the spaCR release this
page documents. Not every spaCR release has a container image; check the
package page before pulling a tag.

The images run spaCR without a display: the CLI, the pipelines, cluster jobs
and reruns of an earlier analysis with a pinned version. Use the platform
installers above to install the desktop application.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Image
     - For
   * - ``ghcr.io/einarolafsson/spacr:1.5.1.7``
     - CPU only. Runs Linux x86-64 containers with Docker or Podman and
       needs no GPU driver.
   * - ``ghcr.io/einarolafsson/spacr:1.5.1.7-cuda12.4``
     - CUDA 12.4. Needs an NVIDIA driver of **550 or newer on the host** and
       the NVIDIA Container Toolkit. Without ``--gpus`` it behaves as the CPU
       image.

``:latest`` and ``:cpu`` follow the newest CPU release; ``:cuda`` and
``:cuda12.4`` follow the newest CUDA release. Name an exact version for
anything you intend to reproduce.

Models and data are mounted at run time; neither is included in the image. A cpsam checkpoint is about
1.2 GB and may be updated between releases, so the image ships none: mount a
folder on ``/models`` and it becomes both the folder the Model Zoo downloads
into and the folder Cellpose loads from. SAMCell and DINOCell are not in the
images either, because they pin PyTorch versions that conflict with spaCR's
and with each other; install one inside a running container, or let the Model
Zoo install it into an isolated environment of its own.

Install Docker and check the image
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Install `Docker Desktop <https://docs.docker.com/get-started/get-docker/>`_
or, on a Linux server, `Docker Engine
<https://docs.docker.com/engine/install/>`_. Configure it to run Linux
containers. These images target x86-64; a native ARM64 image is not provided.

Check that Docker can start the pinned CPU image and list spaCR's headless
pipelines:

.. code-block:: bash

   docker pull ghcr.io/einarolafsson/spacr:1.5.1.7
   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.7 spacr --version
   docker run --rm ghcr.io/einarolafsson/spacr:1.5.1.7 spacr-run --list

For NVIDIA GPU processing, install the host driver and follow the
`NVIDIA Container Toolkit installation guide
<https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html>`_
to configure Docker's NVIDIA runtime. The CUDA image's host-driver requirement
is listed above; the CPU image needs neither the toolkit nor ``--gpus``.

Running a pipeline
~~~~~~~~~~~~~~~~~~

These examples use a Linux Bash shell. Prepare a ``screen`` folder containing
your data and exported settings, and a model-cache folder. The
`bind mounts <https://docs.docker.com/engine/storage/bind-mounts/>`_ make them
available inside the container as ``/data`` and ``/models``. Paths inside
the settings CSV must use these container paths, rather than the host's
paths. Write outputs under ``/data`` so they remain in ``screen`` after
``--rm`` removes the container.

.. code-block:: bash

   mkdir -p screen "$HOME/.cellpose/models"

   docker run --rm \
       --user "$(id -u):$(id -g)" \
       -v "$PWD/screen:/data" \
       -v "$HOME/.cellpose/models:/models" \
       ghcr.io/einarolafsson/spacr:1.5.1.7 \
       spacr-run measure --settings /data/settings/measure_settings.csv

On a GPU host, add ``--gpus all`` and use the CUDA tag:

.. code-block:: bash

   docker run --rm --gpus all \
       --user "$(id -u):$(id -g)" \
       -v "$PWD/screen:/data" \
       -v "$HOME/.cellpose/models:/models" \
       ghcr.io/einarolafsson/spacr:1.5.1.7-cuda12.4 \
       spacr-run mask --settings /data/settings/gen_mask_settings.csv

On Linux, pass ``--user "$(id -u):$(id -g)"`` to match your host user and group.
Files written to a mounted folder use the container's user ID; without this
option, ownership or permissions may differ from your host account. The
images already run as a non-root user, and the
entrypoint moves the cache directories somewhere writable when the user ID
you pass has no home inside the image.

``spacr-run --list`` prints every module that runs headless, and
``spacr-run --describe <module>`` prints what one needs and what it writes.
``spacr-doctor`` reports what the container found, including whether the GPU
is visible:

.. code-block:: bash

   docker run --rm --gpus all ghcr.io/einarolafsson/spacr:1.5.1.7-cuda12.4 spacr-doctor

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
       ghcr.io/einarolafsson/spacr:1.5.1.7 \
       spacr

Do not pass the host's ``XDG_RUNTIME_DIR`` in. That path does not exist
inside the container, and Qt prints a warning about it at every start; the image
makes its own runtime directory under the container's cache folder instead.

A container rarely has a usable OpenGL context, so the animated backdrop may
not draw. ``safespacr`` starts the same application with the backdrop and GL
switched off, and is the right command when the window is slow or blank:

.. code-block:: bash

   docker run --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
       ghcr.io/einarolafsson/spacr:1.5.1.7 safespacr

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

.. _apptainer-images:

Apptainer and SingularityCE on a cluster
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Where a cluster allows Apptainer or SingularityCE but not Docker, repackage
the Docker image as one read-only ``.sif`` file. Build from the repository
root as an ordinary user, with Apptainer 1.2 or newer or SingularityCE 4.0 or
newer:

.. code-block:: bash

   apptainer build --build-arg IMAGE=ghcr.io/einarolafsson/spacr:1.5.1.7 \
       spacr-cpu.sif packaging/apptainer/spacr.def
   apptainer build --build-arg IMAGE=ghcr.io/einarolafsson/spacr:1.5.1.7-cuda12.4 \
       spacr-cuda.sif packaging/apptainer/spacr.def

To start from an image you built yourself, add
``--build-arg BOOTSTRAP=docker-daemon --build-arg IMAGE=spacr:cpu``. The build
runs the image's smoke test. If a login node has no user namespaces, build on
a workstation and copy the ``.sif`` file over. Where ``/etc/subuid`` lists you
but ``newuidmap`` is not installed, add ``--ignore-subuid``.

.. code-block:: bash

   apptainer run spacr-cpu.sif                          # list the modules
   apptainer run spacr-cpu.sif spacr-run mask --settings mask.csv
   apptainer run --nv spacr-cuda.sif spacr-doctor       # check the GPU

``--nv`` binds the host NVIDIA driver; the CUDA image needs driver 550 or
newer. Unlike Docker, the container runs as you, with your real home
mounted: models already in ``~/.cellpose/models`` are found, and run manifests
go to ``~/.spacr/runs/``. ``$HOME``, the current folder and ``/tmp`` are
available by default; add other data folders with ``--bind``. To use a shared
model folder, bind it read-only to ``/models``:
``--bind /shared/cellpose_models:/models:ro``. If a host ``PYTHONPATH`` from
a conda or module setup leaks into the container, add ``--cleanenv``.

``packaging/apptainer/spacr_slurm.sh`` is a Slurm array job that runs Mask
then Measure on one plate per task, from one pair of settings files. Copy it,
set ``SIF``, ``PLATES`` (a file with one plate folder per line),
``MASK_SETTINGS`` and ``MEASURE_SETTINGS`` and the ``#SBATCH`` lines for your
cluster, then submit it:

.. code-block:: bash

   sbatch --array=0-$(( $(wc -l < plates.txt) - 1 )) spacr_slurm.sh

For a CPU partition, remove the ``--gres`` line, replace the ``GPU_FLAG``
line with ``GPU_FLAG=`` and point ``SIF`` at the CPU image.

The desktop interface on a cluster node
---------------------------------------

The GUI needs a display. On a cluster, use one of these:

* **X11 forwarding.** Connect with ``ssh -X user@login-node`` (or ``ssh -Y``
  for trusted forwarding), start an interactive job with X11 forwarding (for
  example ``srun --x11 --pty bash`` on Slurm), then run ``spacr``. This works
  but is slow over long distances.
* **Open OnDemand desktop or VNC.** Start a remote desktop session from your
  cluster's Open OnDemand page, or a VNC server on the node, open a terminal
  inside it and run ``spacr``. This is much faster than forwarding.

Qt's X11 plugin needs ``libxcb-cursor.so.0`` and a few related libraries that
cluster nodes often lack. Without root, install them into your conda or
mamba environment:

.. code-block:: bash

   mamba install -c conda-forge xcb-util-cursor xcb-util-wm xcb-util-image \
       xcb-util-keysyms xcb-util-renderutil xcb-util libxkbcommon

or create the environment from ``environment-linux.yaml``, which already
lists them. ``spacr`` finds them in the active environment by itself, and
names any that are still missing instead of crashing; ``spacr-doctor`` shows
the same check as its ``X11 libraries`` row. With root, ``sudo apt install
libxcb-cursor0`` (Debian/Ubuntu) or ``sudo dnf install xcb-util-cursor``
(RHEL/Rocky) installs them system-wide.

A shared home directory is fine: spaCR keeps its single-window lock and its
crash counter per host, so a session on one node never blocks or counts
against another.

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
