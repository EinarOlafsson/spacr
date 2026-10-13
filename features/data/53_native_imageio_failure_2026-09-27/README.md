# Native artifact launch failure — 2026-09-27

Run [36342374345](https://github.com/EinarOlafsson/spacr/actions/runs/36342374345)
built the real macOS DMG/app and Windows NSIS/onedir artifacts from
`6898d3c264003c61034cf8d2f2895de1f8fbf870` successfully. Fresh artifact-only
jobs installed and launched those binaries with native Cocoa/Windows Qt,
validated installed import origins and resources, constructed Measure and
clicked its real Run action. These retained smoke.json files are unmodified
copies from the two GitHub Actions smoke artifacts.

Both pipelines then failed in `spacr.io → imageio.__init__`, where
`importlib.metadata.version("imageio")` raised `PackageNotFoundError`.
Neither run completed the small analysis, and neither reached uninstall
acceptance. The macOS application log is retained verbatim as additional
context; the definitive pipeline exception is in each smoke receipt.

The shared PyInstaller spec now explicitly adds `copy_metadata("imageio")`
to its data collection, as it already does for spaCR. ImageIO 2.37.4 in the
local declared environment uses that exact import-time distribution lookup.
This repair changes packaging only; it does not bypass the metadata lookup,
replace the installed analysis or alter runtime/API source.

The existing Debian checkout-ownership failure and its exact-directory fix
have a separate receipt. Native rebuild, successful complete analysis and
uninstall preservation remain required on all advertised platforms. The
broader item 53 update/backend/account/consent requirements remain open.

The [upstream ImageIO hook](https://github.com/pyinstaller/pyinstaller-hooks-contrib/blob/master/_pyinstaller_hooks_contrib/stdhooks/hook-imageio.py)
inspected during this repair collects ImageIO resources and lazy plugins,
but does not copy distribution metadata. Explicit `copy_metadata` therefore
addresses a distinct bundle requirement; recursive dependency collection is
unnecessary for the observed failure.

SHA-256 of the original downloaded files:

```text
19a259148f31072f86a99e80bc0453e775d0716a28037e067d682140c8114a3b  macos-smoke.json
eaf862f6f512a4144108076e64f8b055a588bd2c637b514bd2b3c1e9339b3539  windows-smoke.json
684e14b138c3a5f055d5f4b65243ecc5649cf6f49f65ca7844e1e2055d7bd20b  macos-application.log
```

Local repair validation: `tests/test_pyinstaller_spec.py` and
`tests/test_native_installer_workflow.py` passed **7 tests in 0.82 s** under
the 4G cap with private HOME/TMPDIR and hidden accelerators. The regression
contract requires both spaCR and ImageIO metadata to reach `Analysis.datas`.
This checks the packaging specification and workflow; it does not establish
a successful rebuilt native analysis. Log:
`/mnt/wd4tb/scratch/spacr-completion/53-native-feasibility/imageio-tests.log`
(SHA-256 `5fba704ce67a9a77c05e6385d95f0f3ea27cbb3486b5c2b870755a32c83d8c4e`).
