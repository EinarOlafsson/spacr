# spaCR — packaging and distribution

## Recommended: lightweight online installers

The online installers are deliberately small. They do **not** bundle conda,
Python, Qt, PyTorch, CUDA, or the scientific stack. During installation they:

1. download a pinned standalone `uv` bootstrap over TLS;
2. download a private managed CPython 3.12 runtime;
3. install the portable CPU/MPS PyTorch build by default;
4. install `spacr[qt]` in a private environment;
5. run `pip check` and import spaCR, Qt, and PyTorch before activating it; and
6. create the platform's normal application launcher and uninstaller.

No existing system Python is modified. A failed update preserves the previous
working spaCR environment. Windows offers NVIDIA acceleration as an optional
installer component; Linux users can request accelerator auto-detection with
``--torch-backend auto``. The CPU default avoids an accidental multi-gigabyte
CUDA download and works on every supported machine. Apple's standard PyTorch
wheel retains Metal Performance Shaders (MPS) support.

Installer status, help, errors, and progress are localized for every spaCR UI
language: English, Swedish, German, Spanish, Simplified Chinese, Portuguese,
Hindi, Korean, Icelandic, and French. The installer selects the operating
system's UI locale and falls back to English. Set
``SPACR_INSTALL_LANGUAGE`` to one of ``en``, ``sv``, ``de``, ``es``,
``zh_CN``, ``pt``, ``hi``, ``ko``, ``is``, or ``fr`` to override it. Canonical
locale resources live in ``packaging/i18n``; generated shell, PowerShell, and
NSIS resources are embedded by the native builders.

Each bootstrap writes ``install.log`` under its private installation root and
supplies compatible Numba/llvmlite floors. This prevents
SHAP's unbounded transitive requirements from resolving to obsolete source
releases that advertise Python 3.12 compatibility but reject it while building.
On Intel macOS the bootstrap also applies wheel-aware ceilings for
Numba/llvmlite, NumPy, and OpenCV. Modern releases no longer publish all of
their Intel wheels, while the final Intel PyTorch build still requires the
NumPy 1.x ABI. The install validation performs a real NumPy-to-PyTorch tensor
conversion so this incompatibility cannot pass as an import-only success.

| Target | Builder | Release artifact | Default install |
|---|---|---|---|
| Windows 10/11 | `online/build_windows_online.ps1` | `SpaCR-<ver>-Windows-Online-Setup.exe` | `%LOCALAPPDATA%\SpaCR` |
| macOS 11+, Intel/Apple silicon | `online/build_macos_online.sh` | `SpaCR-<ver>-macOS-Universal-Online.pkg` | `/Applications/SpaCR.app` plus a per-user private runtime |
| Linux x86-64 | `online/build_linux_online.sh` | `SpaCR-<ver>-Linux-x86_64-Online.run` | `~/.local/share/spacr` |

`.github/workflows/online-installers.yml` builds all three on native GitHub
runners, performs a real CPU installation and import check on every operating
system, collects the installers under `spacr/application/`, writes SHA-256
hashes, and rewrites the download block in `README.rst` to point at immutable
assets on the matching GitHub release. The version is always read from
`setup.py`.

`packaging/release.py` owns the cross-platform release metadata:

```bash
# Print the current package version
python packaging/release.py version

# Validate and increment the one canonical version
python packaging/release.py bump 1.4.9.9

# After the three native builders have populated dist/online
python packaging/release.py collect --branch main
```

The native installers cannot all be generated on one local operating system.
The GitHub workflow runs each builder on its matching native runner and then
calls the collection command once all three artifacts exist.

## One-click releases

Run **Actions → release SpaCR → Run workflow**, enter the new version, and
leave the target as `main`. `.github/workflows/release.yml` then:

1. validates and commits the version increment;
2. builds and validates the wheel and source distribution;
3. publishes to PyPI using trusted publishing and waits until that immutable
   version is available from the PyPI API;
4. builds Windows, macOS, and Linux installers on native runners;
5. commits the current installers under `spacr/application/` and updates the
   README links; and
6. tags that exact installer commit, creates the GitHub release, and attaches the three
   installers, wheel, source distribution, and SHA-256 manifest.

GitHub displays manual ``workflow_dispatch`` buttons from the default branch,
so merge `release.yml` into `main` once to enable that button permanently.
There is also a branch-native path: changing ``VERSION`` in `setup.py` and
pushing that commit to `main` automatically runs
steps 2-6 for the already-incremented version. Rerunning the same version is
safe: an existing PyPI artifact is not uploaded twice, existing release
assets are replaced, and an existing tag must already point to the exact
release commit. Installers are never generated until PyPI confirms the exact
version, so their online bootstrap can immediately install the package they
name.

One-time repository setup:

1. In GitHub **Settings → Actions → General → Workflow permissions**, allow
   read and write permissions so the workflow can commit the version and
   installers.
2. Create a GitHub environment named `pypi`.
3. On PyPI, add a trusted publisher for owner `EinarOlafsson`, repository
   `spacr`, workflow `release.yml`, environment `pypi`.

No PyPI API token is stored in GitHub. If `main` has branch
protection, allow `github-actions[bot]` to push these two release commits or
replace the direct-push steps with your protected-branch merge policy.

## Conda-forge releases

The prepared recipe and bot configuration live in `../conda-forge`. Unlike
PyPI, conda-forge requires a one-time reviewed pull request to
`conda-forge/staged-recipes`; a source-repository workflow cannot bypass that
review. Follow `../conda-forge/README.md` once. After the recipe is accepted,
the conda-forge bot detects each new PyPI version, tests its update PR, and
automerges passing version-only updates. No Anaconda token is stored here.

Linux installs the small Qt/OpenGL runtime libraries through apt, dnf, zypper,
or pacman when available. macOS packages may be signed by setting
`PRODUCTSIGN_IDENTITY`; public distribution should additionally use an Apple
Developer ID and notarization. Windows uses a per-user NSIS installer and does
not require administrator access.

The macOS package itself performs no network installation in PackageKit's
time-limited ``postinstall`` process. On first launch, SpaCR opens a Terminal
bootstrap that installs the private runtime under
``~/Library/Application Support/SpaCR`` and then relaunches the application.
Progress and dependency errors remain visible, and slow downloads are not
terminated by the package installer's script timeout.

The unversioned bootstrap scripts support dry runs:

```bash
packaging/online/install_spacr_unix.sh --dry-run --platform linux
```

```powershell
.\packaging\online\install_spacr_windows.ps1 -DryRun
```

`SPACR_PACKAGE_SPEC`/`--package-spec` can point a bootstrap at a test wheel or
development build without changing the release builders.

## Legacy self-contained builders

Scripts in this directory turn the spacr Python package into a native
installer/executable for each of the three target platforms:

| Target             | Script                    | Output                   |
|--------------------|---------------------------|--------------------------|
| Windows 10/11 (x64)| `build_windows.ps1`       | `dist/SpaCR-<ver>.exe`   |
| macOS 11+ (arm64/x64)| `build_macos.sh`        | `dist/SpaCR-<ver>.dmg`   |
| Debian/Ubuntu (x64)| `build_debian.sh`         | `dist/spacr_<ver>_amd64.deb` (installable via `sudo apt install ./spacr_<ver>_amd64.deb`) |

**Common contract**

The launcher `spacr_launcher.py` in this directory is the entry point
every installer wraps — it calls `spacr.gui.gui_app()`. So a single
launcher spec drives all three build systems; only the packaging /
metadata / signing differs per platform.

**What each build does under the hood**

* Windows: PyInstaller `--onefile --windowed` bundles Python, spacr,
  cellpose, torch (CPU or CUDA depending on your local env), plus a
  hidden-imports list of the heavy scientific stack (numpy, scipy,
  sklearn, statsmodels, skimage, matplotlib, cv2). The resulting
  `.exe` runs on any Windows 10+ machine.

* macOS: PyInstaller `--windowed` produces a `SpaCR.app` bundle, which
  `hdiutil` then packs into a signed (ad-hoc) `.dmg` you can drag into
  `/Applications`. Requires code-signing for distribution outside your
  own Mac — that step is left explicit at the top of the script.

* Debian: `stdeb` converts the `setup.py` into `debian/` control files,
  then `dpkg-buildpackage` produces a `.deb` that pins the required
  system libs (libgl1, libglib2.0-0, libsm6, libxext6, libxrender1)
  in the `Depends:` field. Install with
  `sudo apt install ./dist/spacr_<ver>_amd64.deb`.

**Cross-building caveat**

You *cannot* cross-build these from a single machine:

  * `.exe` requires Windows (or Wine + a Python-for-Windows install)
  * `.dmg` requires macOS (Apple's `hdiutil` + `codesign`)
  * `.deb` requires a Debian/Ubuntu box (or a `debian:12` docker image)

The scripts assume they run on their native platform; each fails fast with a
clear error if run elsewhere. For normal releases, prefer the online
installers above: frozen artifacts duplicate Python and the scientific stack.
