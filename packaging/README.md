# spacr — packaging & distribution

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

The scripts assume they run on their native platform; each fails fast
with a clear error if run elsewhere. A GitHub Actions matrix that runs
all three in parallel (one job per OS) is the recommended CI pattern —
see `.github/workflows/build-installers.yml` (not included; add if you
want CI-driven releases).
