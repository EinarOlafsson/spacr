# spaCR — packaging and distribution

## Recommended: lightweight online installers

The online installers are deliberately small. They do **not** bundle conda,
Python, Qt, PyTorch, CUDA, or the scientific stack. During installation they:

1. download a pinned standalone `uv` bootstrap over TLS;
2. download a private managed CPython 3.12 runtime;
3. detect the available PyTorch accelerator and choose CUDA/AMD/Intel or CPU;
4. install `spacr[qt,zernike,btrack,czi]` in a private environment;
5. run `pip check` and import spaCR, Qt, and PyTorch before activating it; and
6. create the platform's normal application launcher and uninstaller.

No existing system Python is modified. A failed update preserves the previous
working spaCR environment.

| Target | Builder | Release artifact | Default install |
|---|---|---|---|
| Windows 10/11 | `online/build_windows_online.ps1` | `SpaCR-<ver>-Windows-Online-Setup.exe` | `%LOCALAPPDATA%\SpaCR` |
| macOS 11+, Intel/Apple silicon | `online/build_macos_online.sh` | `SpaCR-<ver>-macOS-Universal-Online.pkg` | `/Applications/SpaCR.app` plus a private runtime |
| Linux x86-64 | `online/build_linux_online.sh` | `SpaCR-<ver>-Linux-x86_64-Online.run` | `~/.local/share/spacr` |

`.github/workflows/online-installers.yml` builds all three on native GitHub
runners. A published GitHub release receives the installers as assets
automatically. A manual workflow run produces downloadable test artifacts.

Linux installs the small Qt/OpenGL runtime libraries through apt, dnf, zypper,
or pacman when available. macOS packages may be signed by setting
`PRODUCTSIGN_IDENTITY`; public distribution should additionally use an Apple
Developer ID and notarization. Windows uses a per-user NSIS installer and does
not require administrator access.

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
