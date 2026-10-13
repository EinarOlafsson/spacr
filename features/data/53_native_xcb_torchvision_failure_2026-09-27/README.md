# Item 53: genuine frozen-runtime failures and bounded packaging repair

Source: `5e7cc6d9eca9342c522ac8c366228f6428a70ee8`, spaCR 1.5.1.0.
[Native artifact run 36345082025](https://github.com/EinarOlafsson/spacr/actions/runs/36345082025)
built all three installer families successfully, then all four fresh-host
smokes failed. This receipt preserves failures rather than treating a build
as runtime acceptance.

| Host | Job | Failure |
| --- | --- | --- |
| Debian 12 | 108695736916 | Qt xcb plugin fails before application startup; exit 134 |
| Ubuntu 22.04 | 108695736927 | Same Qt plugin failure; exit 134 |
| macOS 15 | 108695797476 | Real Measure Run reaches torchvision import, then missing `torchvision::nms`; exit 1 |
| Windows 2025 | 108695797520 | Same real Measure failure; exit 1 |

The downloaded Debian artifact receipts use matrix index 0 for Ubuntu 22.04
and index 1 for Debian 12. Their installation logs confirm those identities.
Full downloaded artifacts and package-file lists remain under
`/mnt/wd4tb/scratch/spacr-completion/53-native-feasibility/ci-36345082025`.
Tracked files retain raw job/build logs, both Linux application logs and both
native scientific-run failure JSONs. `SHA256SUMS` binds those exact bytes.

The Debian build log records unresolved `libxcb-shape.so.0`, including the
dependency of `libQt6XcbQpa.so.6`. Neither packaged files nor fresh-host package
installation supplied it. `libxcb-cursor0` was installed; Qt's generic cursor
warning does not identify the missing shape dependency. The fix adds
`libxcb-shape0` to both builder prerequisites and the installed package's Depends.

The same build records `Hidden import "torchvision._C" not found!`, and its
installed file manifest contains no torchvision `_C` library. The exact
[hooks-contrib 2026.7 hook](https://raw.githubusercontent.com/pyinstaller/pyinstaller-hooks-contrib/v2026.7/_pyinstaller_hooks_contrib/stdhooks/hook-torchvision.py)
only requests that hidden import and retains Python sources. The fix collects
shared libraries from torchvision alone, preserving package-relative paths,
and feeds them to Analysis as binaries so dependency analysis still runs.
Explicit `.so`/`.pyd` patterns are necessary: the
[PyInstaller 6.22.3 helper's defaults](https://raw.githubusercontent.com/pyinstaller/pyinstaller/v6.22.3/PyInstaller/utils/hooks/__init__.py)
omit those non-`lib` Unix names and Windows Python extensions. Collection now
fails immediately if `_C` is missing; it does not install a stub operator.

Local validation: **14 passed in 0.85 seconds**, exit 0, using only
`tests/test_pyinstaller_spec.py` and `tests/test_native_installer_workflow.py`
through `tools/run_capped.sh 4G`, private HOME/TMPDIR and hidden accelerators.
Shell syntax and whitespace checks also passed. Tests exercise platform file
patterns, missing-operator refusal, Analysis wiring, and both Debian dependency
sets without importing Torch or building an application.

The repaired artifacts have **not yet been rebuilt or rerun natively**.
Updater, effective backend selection, account/consent end-to-end evidence and
GPU acceptance remain separate open requirements. No item 53 closure is claimed.
