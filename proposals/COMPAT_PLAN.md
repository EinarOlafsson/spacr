# spaCR compatibility remediation plan
**Scope:** Python 3.9–3.14 × {Linux, macOS, Windows} × {x86-64, ARM64} × {CUDA, ROCm, MPS, CPU}
**Repo:** `/mnt/firecuda2/Claude/repo/spacr` @ `spacr-nightly` (`9046dd4`), version `1.4.9.3`
**Synthesised from:** five independent audits (Python-version, OS/arch, device-backend, dependency/wheel, multiprocessing/Qt)
**Dev machine:** Linux x86-64 + CUDA, CPython 3.10.19. **34 of the 36 matrix cells cannot be executed here.**

---

## 0. Verdict in one paragraph

spaCR's **own Python source is not the problem**. I re-verified this myself: all 155 files under `spacr/` and all 521 `.py` files in the repo parse cleanly under `ast.parse(feature_version=(3,9))` — zero failures. Every version blocker is a *pin* in `setup.py`, and every platform blocker is either a *missing wheel upstream* or a *POSIX assumption baked into a path/encoding/multiprocessing decision*. That is good news: the expensive part (rewriting the language level of a 155-file package) does not exist. The work is metadata, four dependency decisions, one device resolver, one multiprocessing contract, and about a dozen Windows-correctness fixes. The single most valuable change in the whole document is not a code fix at all — it is a CI matrix, because it converts roughly twenty of these findings from "audit prose someone has to trust" into "a red build someone has to fix".

**What is claimable today, honestly: `>=3.10,<3.13`, Linux/macOS/Windows on x86-64, plus macOS arm64, with only one cell ever actually executed.** Not `>=3.9` (never run, and torch's cp39 window closed at 2.8.0). Not `OS Independent`. Not Windows-on-ARM at any price.

---

## 1. The honest matrix, today

**How to read the labels.** `GREEN` = executed and passing (CI or by hand). `AMBER` = static analysis says it installs and there is no known blocker, but nobody has ever run it. `DEGRADED` = installs and runs, silently worse than the user thinks. `RED` = install fails. Every `AMBER`/`RED` below is an **inference from wheel metadata**, which I re-queried from the PyPI JSON API myself rather than trusting the audits; the inference is strong for install-time (wheel presence is a fact) and weak for run-time (nobody has run it).

### 1.1 Install-ability, as pinned today

| | 3.9 | 3.10 | 3.11 | 3.12 | 3.13 | 3.14 |
|---|---|---|---|---|---|---|
| **Linux x86-64** | AMBER | **GREEN** | AMBER | AMBER | RED | RED |
| **Linux ARM64** | RED¹ | RED¹ | RED¹ | RED¹ | RED | RED |
| **macOS x86-64 (Intel)** | DEGRADED² | DEGRADED² | DEGRADED² | DEGRADED² | RED | RED |
| **macOS ARM64 (M1–M4)** | DEGRADED³ | DEGRADED³ | DEGRADED³ | DEGRADED³ | RED | RED |
| **Windows x86-64** | AMBER⁴ | AMBER⁴ | AMBER⁴ | AMBER⁴ | RED | RED |
| **Windows ARM64** | RED⁵ | RED⁵ | RED⁵ | RED⁵ | RED⁵ | RED⁵ |

**One cell of thirty-six has ever been executed.** That is `.github/workflows/tests.yml`: `ubuntu-22.04`, Python 3.10, one job.

¹ **Linux ARM64** — needs a C/C++ toolchain at install time for at least three deps, so it fails on a bare node. PROVED from PyPI: `mahotas 1.4.18` ships **no aarch64 wheel at all** (aarch64 exists only in `1.4.13`, cp36–cp39); `psutil 5.9.8` (`<6.0`) ships manylinux **x86_64 and i686 only**; `aicspylibczi 3.3.1` ships manylinux **x86_64 only** and additionally needs CMake + libCZI headers. On 3.9 specifically, `mahotas 1.4.13`'s cp39 aarch64 wheel exists, so 3.9 is the *least* broken ARM-Linux cell — a fact with no practical value.

² **macOS Intel** — PROVED: the last torch with **any** macOS `x86_64` wheel is **2.2.2**. `torch>=2.0,<3.0` is satisfiable on Intel Mac only by backtracking to 2.2.2 / torchvision 0.17.2, silently, with no warning. Cellpose 4 (`cpsam`) then runs on a torch released two years before it. Also `numpy 1.26.4` has Intel-Mac wheels, so the install *succeeds* — this is the worst kind of failure: quiet.

³ **macOS Apple Silicon** — installs and runs, on the **CPU**, always. PROVED by grep: `torch.backends.mps`, `torch.mps` and the string `'mps'` appear **zero times** in `spacr/`. All 17 `torch.device(...)` sites resolve `cuda ? cuda : cpu`. Worse, spaCR *actively defeats* Cellpose's own working MPS detection by passing an explicit `device=torch.device('cpu')`, which overrides `gpu=` in `CellposeModel.__init__`. Then Cellpose's `use_bfloat16=True` default costs a further **2.9–4.9×** on a CPU without AVX512-BF16 (measured on this box; the M-series cores are also without it).

⁴ **Windows x86-64** — installs. Then meets a wall of POSIX assumptions: `cv2.imwrite`'s return value dropped on the primary crop path (silent data loss on any non-ASCII path), mixed `/`+`\` separators written into `png_list.png_path` (breaks Windows→Linux workflows), spaCR's own console output unencodable in any Windows codepage (kills every batch-queue job), MAX_PATH overflow at ~116 characters of source path, and the spawn re-import tax (measured here: **13.6 s and 1.2 GB per worker**, with `n_jobs` uncapped at the number of files). None of this is fatal at install; all of it is fatal at use.

⁵ **Windows ARM64** — categorically unreachable, and will stay that way. PROVED: **torch has never published a `win_arm64` wheel, for any version.** Neither has `numpy<2.0` (first numpy with win_arm64 is **2.3.0**), `opencv-python-headless` (win32/win_amd64 only), `mahotas`, `pylibCZIrw`, or `aicspylibczi`. `PySide6` *does* ship win_arm64 — which is exactly the trap: the GUI layer installs and the compute layer does not.

### 1.2 Run-ability, given a successful install

Installing is the easy half. Even in the AMBER cells, three findings mean the thing a user actually does may not work:

- **Linux + Qt GUI + Measure can deadlock at "Progress: 0/N" with no traceback.** The Qt path never sets a start method (PROVED: nothing under `spacr/qt/` calls `set_start_method`; only `gui.py:317` and `gui_core.py:1676` do, and neither is on the Qt path), so it inherits `fork` on Linux, and forks out of a QThread while `spacr-prewarm` is importing modules on another thread. Forking while another thread holds a per-module import lock deadlocks the child permanently — reproduced on 3.10 and 3.12. CPython's own `DeprecationWarning: This process is multi-threaded, use of fork()` **cannot** fire here, because a QThread is invisible to Python's `threading` registry. This is the default entry point (`spacr` → Qt), on the primary platform, in the most natural flow.
- **Worker `print()` output never reaches either GUI console, on any platform.** Under `fork` it is destroyed outright (the worker emits a Qt signal into a dead QApplication copy); under `spawn`/`forkserver` it goes to a terminal that a bundled `.app`/`.exe` does not have. Progress ticks appear (they run in the parent's callback thread) and per-field diagnostics do not — which is why every worker-side failure presents as "it just stopped".
- **Abort orphans the pool.** `thread_control.get("run_thread").terminate()` SIGTERMs the intermediate process; the daemonic pool grandchildren survive and keep holding VRAM (reproduced). The Qt GUI has no path to the pool at all — `_on_stop` only calls `requestInterruption()`, and nothing checks it.

---

## 2. The floor and the ceiling

### 2.1 Floor: can 3.9 be supported? — **Yes, technically. No, do it.**

The source is 3.9-clean (verified above). The dependencies are the question, and here I have to **correct one audit**: the claim that torch has no cp39 wheels is true of *current* torch and false in general. PROVED from PyPI:

```
torch 2.8.0   requires_python >=3.9.0
              cp39 wheels: manylinux_2_28_x86_64, manylinux_2_28_aarch64,
                           macosx_11_0_arm64, win_amd64
torch 2.13.0  requires_python >=3.10        (no cp39)
```

So `torch>=2.0,<3.0` on Python 3.9 resolves to **torch ≤ 2.8.0**, not to 2.2.2, and not to nothing. 3.9 is *resolvable*.

**The cost of keeping it:** you freeze torch at 2.8 for 3.9 users, you cap PySide6 at 6.10 (6.11+ requires ≥3.10), `torchcam` backtracks (latest requires ≥3.11), `monai`, `torch-geometric`, `numba`, `llvmlite`, `pingouin` and `gdown` all declare `requires_python >=3.10` at their current releases and must backtrack, and every one of those backtracks is a resolver branch that can fail in a way nobody will ever reproduce. In exchange you support an interpreter that reached end-of-life security-only status in 2025 and that **no spaCR test has ever run on**.

**Decision: drop 3.9. Set the floor at 3.10.** It is what CI runs, what `packaging/build_debian.sh:59` declares (`python3 (>= 3.10)`), what `packaging/build_macos.sh:10` and `build_windows.ps1:4` say, and what current torch requires. The one dissenting artefact is `environment.yaml:23` (`python=3.9.19`), which is stale and should be regenerated or deleted — it also pins `torch==2.4.0`, which current spaCR cannot use.

### 2.2 Ceiling: 3.13 and 3.14

**Today the ceiling is 3.12**, and there are exactly two reasons, both in `setup.py`:

| Blocker | Line | Fact (PROVED from PyPI) |
|---|---|---|
| `numpy>=1.26.4,<2.0` | `setup.py:10` | Only 1.26.4 satisfies it. Its wheels are **cp39–cp312, full stop**. First numpy with cp313 is **2.1.0**; first with cp314 is **2.3.2**. On 3.13 pip falls back to the sdist and attempts a source build of a numpy that predates 3.13's C-API. |
| `mahotas>=1.4.13,<2.0` | `setup.py:18` | mahotas 1.4.18 ships cp36–cp312. **No version of mahotas has ever shipped a cp313 or cp314 wheel.** |

**After the pin work (Wave 3), 3.13 is fully reachable** on Linux x86-64/aarch64, Windows x86-64 and macOS arm64. Everything else in the core has cp313 wheels: torch (2.10.0+), scipy (1.14.1+), pandas, scikit-image, scikit-learn, matplotlib, pillow (10.4.0+), lxml (5.3.0+), psutil (7.1.2+), fastremap, statsmodels, PyWavelets, bottleneck, shap, h5py, PySide6.

**3.14 is reachable too, and closer than the audits concluded**, because of a subtlety worth stating so nobody re-derives the wrong answer: **a naive "does a cp314 wheel exist" scan under-reports**, and I made that mistake myself before catching it. Two wheel kinds are forward-compatible and show up as "no cp314":

- **`abi3` wheels.** `opencv-python-headless` ships `cp37-abi3` for *every* release including 5.0.0.93 — it works on 3.13 and 3.14 unchanged. One audit listed opencv as having "no cp313"; that is a false negative. Same for `PySide6` (`cp310-abi3`, `requires_python <3.15,>=3.10`) — Qt is fine on 3.14.
- **`py3-none-any` wheels.** `cellpose`, `xgboost`, `btrack`, `umap-learn`, `seaborn`, `pingouin`, `transformers`, `tifffile`, `monai`, `captum` are pure Python and version-agnostic.

The **real** 3.14 blockers, after `numpy<3.0`/`pillow<13`/`lxml<7`/`psutil<8`:

| Package | Status | Verdict |
|---|---|---|
| `mahotas` | no cp314 (and no cp313) at any version | → make optional (Wave 3) |
| `pylibCZIrw` | no cp314 at any version incl. 6.1.0 | → make optional (Wave 3) |
| `aicspylibczi` | no cp314; also no linux-aarch64 ever | → **delete**, zero imports in `spacr/` |
| `torchvision` | `requires_python: !=3.14.1,>=3.10` | odd but harmless — 3.14.0 and 3.14.2+ are fine |

Everything else clears: numba 0.63+, llvmlite 0.46+, pandas 2.3.3+, scikit-image 0.26+, scikit-learn 1.7.2+, matplotlib 3.10.5+, h5py 3.15+, imagecodecs, rapidfuzz 3.14+, biopython 1.86+, fastremap 1.17.5+, torch 2.9.0+, tables 3.11+.

**So: `>=3.10,<3.15` is achievable, with 3.13/3.14 core-only** (no Zernike moments, no CZI) unless mahotas and pylibCZIrw ship wheels.

### 2.3 Recommended `requires-python`, by wave

| After | Value | Rationale |
|---|---|---|
| **Wave 0 (today)** | `>=3.10,<3.13` | The only range with evidence. Turns spaCR's worst first-contact experience (a numpy compiler error) into one clear pip sentence. |
| Wave 3 | `>=3.10,<3.14` | numpy/pillow/lxml/psutil unpinned; mahotas + CZI moved to extras. |
| Wave 3 + verified | `>=3.10,<3.15` | Only once a 3.14 CI cell is green. Do not claim it before. |

**Free-threaded builds (3.13t / 3.14t): don't.** numpy 1.26, mahotas, opencv, protobuf, statsmodels and transformers publish no `cp313t`/`cp314t` wheels. Not close.

---

## 3. Corrections to the audits — read before executing

The five audits are strong and mostly agree. These five points differ, and executing the wrong version of them would waste days.

1. **`opencv-python-headless` is NOT a Python-version blocker.** It ships `cp37-abi3` wheels for every platform except win_arm64, at every version 4.11 → 5.0.0.93. It works on 3.13 and 3.14 today. (The numpy-2 `requires_dist` observation still stands and is a *reason to unpin numpy*, not a reason to touch opencv.)
2. **Python 3.9 is not blocked by torch.** torch 2.8.0 has cp39 wheels and `requires_python >=3.9.0`. Drop 3.9 for *policy* reasons (untested, drags six deps backwards), not because it is impossible.
3. **The frozen build already calls `freeze_support()`.** `packaging/spacr_launcher.py:17` does it first thing. The recursive-app-launch risk is therefore *lower* than one audit rated it. The two real packaging bugs stand: the launcher runs `spacr.gui.gui_app()` (**Tk**) while `setup.py:118` maps `spacr` → `spacr.qt:run` (**Qt**) — the installers ship a different application than pip does; and `print(..., file=sys.stderr)` at `spacr_launcher.py:22` is a silent no-op in a windowed build where `sys.stderr is None`. Adding `mp.freeze_support()` to `spacr.qt:run` is still cheap insurance, just not a CRITICAL.
4. **`Pool(initializer=..., initargs=(Value, Lock))` is fine under spawn.** One auditor hypothesised it would raise "Synchronized objects should only be shared through inheritance", tested it, and disproved it. Do not "fix" `spacr/io.py:3351-3354`.
5. **`torch.cuda.empty_cache()`, `torch.cuda.memory_allocated()` and `torch.backends.cudnn.benchmark = True` are all safe no-ops on non-CUDA torch** (executed). Do not add guards around the 13 `empty_cache()` sites — it is churn. The one that *does* need a guard is `deep_spacr.py:2`, and only on ROCm (it maps to MIOpen exhaustive kernel search).

Counts I re-verified myself on this checkout, so the plan is anchored to reality: **155** `.py` files in `spacr/`; **75** files import PySide6, **0** import PyQt6; **17** `torch.device(` sites; **32** `torch.cuda.is_available()` sites; **0** `mps` references; **87** `sqlite3.connect` sites of which **45** pass `timeout=`; **21** `cv2.imwrite` sites; **33** builtin text-mode `open()` calls with no `encoding=`; **0** callers of `reset_mp` outside tests.

---

## 4. Ordered remediation plan

Ordering principle: **cheapest-and-highest-value first**, with one hard dependency edge (Wave 1 must precede Wave 3) and one deliberate early investment (Wave 2, CI) that pays for every wave after it.

Effort estimates are for one experienced developer who knows this codebase.

---

### WAVE 0 — Tell the truth in the metadata
**Effort: 3–4 hours. Fixable blind. Zero runtime risk. Do this first, today.**

| # | Change | Files |
|---|---|---|
| 0.1 | **Delete `setup.py:175-190`** — the module-level `subprocess.run(['pip','install',dep])` loop. It runs on every build and every `pip install .`, breaks PEP 517 isolated builds and every offline install, silently swallows failures behind `except CalledProcessError: pass`, invokes bare `pip` (absent from PATH in many venv layouts), and installs an entire **second, unused Qt binding** — the list is a hand-copy of `cellpose`'s `gui` extra (`pyqtgraph, pyqt6, pyqt6.sip, qtpy, superqt`), duplicated within itself. 75 files import PySide6; 0 import PyQt6. | `setup.py` |
| 0.2 | **Add real project metadata to `pyproject.toml`.** It is currently three lines (build-system only). Add `[project]` with `requires-python = ">=3.10,<3.13"`, name, version, deps — or, if you prefer to keep `setup.py` authoritative for now, at minimum add `python_requires=">=3.10,<3.13"` to the `setup()` call. Without it pip cannot decline early: on 3.13 it resolves, downloads hundreds of MB, and dies inside a numpy source build with a compiler error that reads as "spaCR is broken". | `pyproject.toml`, `setup.py` |
| 0.3 | **Replace the classifiers.** Drop `Operating System :: OS Independent` (untested and now known false for Windows-on-ARM). Add `Operating System :: POSIX :: Linux`, `:: MacOS`, `:: Microsoft :: Windows`, and `Programming Language :: Python :: 3.10 / 3.11 / 3.12`. | `setup.py:168-172` |
| 0.4 | **Fix the three-way self-contradiction.** `environment.yaml:23` pins `python=3.9.19` and `torch==2.4.0` (unsatisfiable with current spaCR); `requirements.txt:14` pins `cellpose>=3.0.6,<4.0` while `setup.py:13` pins `>=4.0,<5.0` **and the code is written entirely against the Cellpose-4 API** (`cpsam`, `pretrained_model=`, `parse_cellpose4_output`) — anyone installing from `requirements.txt` gets a Cellpose whose `CellposeModel`/`assign_device` signatures differ. `requirements.txt` also still carries the PyQt6 block. Regenerate both from `setup.py`, or delete them and point at `pip install -e .`. | `environment.yaml`, `requirements.txt` |
| 0.5 | **Delete `segment-anything`** (`setup.py:14`). PyPI's `segment-anything` has exactly **one** release (1.0, 2023-04-06) with empty author, homepage and summary — Meta never published SAM to PyPI. spaCR imports it in 0 files, and `cellpose` depends on `segment_anything` anyway. It is an unpinned, unattributed name in the supply chain for no benefit. | `setup.py:14` |
| 0.6 | **Delete `aicspylibczi`** (`setup.py:43`). **0 import statements and 0 raw-string references anywhere in `spacr/`.** It is the single blocker forcing a CMake + libCZI source build on ARM Linux, for nothing. | `setup.py:43` |

**Acceptance:** `python -m build` succeeds in a network-isolated sandbox (it currently cannot); `pip install spacr` on 3.13 fails in under two seconds with *"Requires-Python >=3.10,<3.13"* instead of a numpy compiler error.

---

### WAVE 1 — The tripwire: `np.trapz` → `np.trapezoid`
**Effort: 1 hour. Fixable blind. MUST land before Wave 3.**

`np.trapz` was removed in NumPy 2.0. The instant anyone lifts the numpy pin, `import spacr.timelapse` dies at module scope — because the existing fallback is **already dead**:

```python
# spacr/timelapse.py:20-24
try:
    from numpy import trapz
except ImportError:
    from scipy.integrate import trapz   # ALSO removed, in SciPy 1.14
```

`scipy.integrate.trapz` is gone in scipy 1.15/1.16 — both permitted by spaCR's own `scipy>=1.12.0,<2.0`. Only numpy 1.26 is holding this up.

Five call sites plus the import block:
- `spacr/utils.py:4657` — `np.trapz(sorted_precisions, x=sorted_recalls)`
- `spacr/attribution.py:1410` — `float(np.trapz(sc, frac))`
- `spacr/timelapse.py:20-24` (the import), `:1705`, `:1706`, `:1744`

Replace with `np.trapezoid` and delete the try/except entirely. Add a one-line regression test that asserts `import spacr.timelapse` succeeds.

---

### WAVE 2 — The CI matrix
**Effort: 4–6 hours to write, then it works forever. Fixable blind. Highest leverage item in this document.**

Put this *before* the dependency and correctness work, not after. Roughly twenty findings in this plan — the Windows encoding bugs, the path-separator bug, the case-sensitivity bug, the SQLite locking, the MAX_PATH overflow, the spawn tax, every wheel-availability claim — become red builds the day this lands, and stop being things a human has to keep believing. Full YAML in §5.

New cells will be red on day one. That is the point. Mark the not-yet-fixed ones `continue-on-error: true` with a comment naming the wave that clears them, and remove the flag as each wave lands.

---

### WAVE 3 — Unpin the ceiling, and split the extras
**Effort: 2–3 days, most of it validating numpy 2. Fixable blind, must be test-validated.**

Two halves. The pin half is minutes; the numpy-2 half is the real work.

#### 3a. Raise the ceilings

| Pin | Now | To | Why |
|---|---|---|---|
| `numpy` | `>=1.26.4,<2.0` | `>=1.26.4,<3.0` | The 3.13/3.14 blocker. Also drags five deps backwards today: `scipy` stuck at 1.17.1 (1.18 needs numpy≥2), `opencv-python-headless` at 4.11.0.86, `shap` at 0.49.1, `tifffile` at 2026.3.3, `imagecodecs` at 2026.1.14. The numpy-2 ABI is forward-compatible (extensions built against 2.x import under 1.x, not the reverse), so widening costs nothing. |
| `psutil` | `>=5.9.8,<6.0` | `>=5.9.8,<8` | 5.9.8 has **no linux-aarch64 wheel**; psutil 7.x does. Forces a C build on every Graviton/Ampere/Jetson box. |
| `pillow` | `>=10.2.0,<11.0` | `>=10.2,<13` | 10.4.0 tops out at cp313. First cp314 is 11.3.0. |
| `lxml` | `>=5.1.0,<6.0` | `>=5.1,<7` — **or delete** | 5.4.0 tops out at cp313; first cp314 is 6.0.1. **0 import statements in `spacr/`** — deleting is defensible. |
| `tifffile` | `>=2023.4.12` | keep | already open |

**The numpy-2 validation is the actual cost.** NEP 50 promotion rules changed (`np.float32(1) + 1.0` now stays float32), `np.trapz`/`np.float_`/`np.product`/`np.in1d`/`np.NaN` are gone (audits found **only** `np.trapz` present — Wave 1 clears it), and `copy=False` semantics in `np.array` changed. Budget a full test-suite run on 3.11 + numpy 2, then 3.12, then 3.13, and expect to chase a handful of dtype assertions in `measure`/`utils`.

#### 3b. Make the thin-platform blockers optional

The principle: **a user on a thin platform should get a working core.** Three packages block whole matrix cells for features that are individually optional.

**`mahotas` → `extras_require['zernike']`.** Highest value-per-line in the entire audit set. It is imported once, at `spacr/measure.py:12` — *top level*, so `import spacr.measure` fails outright on 3.13/3.14 and on ARM Linux without a C++ toolchain. It feeds exactly one function, `_calculate_zernike` (`measure.py:338`, called at `:379` from `_morphological_measurements`), which **already has an off-switch**: `_morphological_measurements(..., zernike=True)` at `measure.py:463`. Move the import inside the function with the same guarded-import pattern spaCR already uses at `timelapse.py:618-628`:

```python
def _calculate_zernike(...):
    try:
        from mahotas.features import zernike_moments
    except ImportError as exc:
        raise RuntimeError(
            "Zernike moments require mahotas, which has no wheels for "
            "Python 3.13+ or ARM Linux. Install with `pip install spacr[zernike]`, "
            "or set zernike=False to skip these columns."
        ) from exc
```

**CZI/ND2/LIF readers → `extras_require['formats']`.** `pylibCZIrw` is the single hardest 3.14 blocker (no cp314 at any version). `spacr/convert.py:182-186` **already has the right pattern** — a clean `find_spec` table mapping `.nd2/.czi/.lif` to install hints — but `spacr/io.py:1` defeats it by importing `czifile` and `readlif` at module top level, so `import spacr.io` hard-fails without them. Make those two lazy, move `pylibCZIrw`, `czifile`, `nd2reader`, `readlif` into `formats`, and the existing pattern starts working as designed. *(Correction to one audit: `pylibCZIrw` is NOT x86-only — 5.1.1 ships cp39–cp313 for linux x86_64 **and aarch64**, macOS x86_64 **and arm64**, and win_amd64. Its only gaps are cp314 and win_arm64.)*

**Sdist-only deps → optional or dropped.** PROVED: `gpustat`, `gputil`, `matplotlib_venn`, `trackpy`, `ttkthemes` publish **no wheel on any platform at any version**. All pure Python, so they build trivially — but they break `pip install --only-binary=:all:` and every air-gapped install. `gputil`/`gpustat` are NVIDIA-only monitoring and should be an extra (see Wave 4). `ttkthemes`/`customtkinter` are Tk-GUI-only.

**Never-imported deps to review for deletion** (PROVED: 0 import statements in all 155 files): `monai`, `segmentation_models_pytorch`, `torch-geometric`, `wandb`, `gdown`, `lxml`, `aicspylibczi`, `customtkinter`, `ttkthemes`, `ttf_opensans`, `brokenaxes`, `gpustat`, `keyring`, `transformers`, `openai`, `numba`, `protobuf`, `numexpr`, `imagecodecs`.
**Be careful with that list** — several are legitimately indirect and must stay: `tables` (0 imports but 5 `read_hdf`/`to_hdf`/`HDFStore` sites — a pandas engine), `openpyxl` (2 `read_excel`/`to_excel` sites), `numexpr`/`bottleneck` (optional pandas accelerators), `imagecodecs` (backs `czifile`/`tifffile`), `numba` (transitive via `umap-learn`/`shap`), `protobuf` (via `wandb`). The genuinely removable set is `monai`, `segmentation_models_pytorch`, `torch-geometric`, `wandb`, `gdown`, `brokenaxes`, `customtkinter`, `ttkthemes`, `ttf_opensans` — each removal deletes resolver constraints that can force backtracking elsewhere. **Bonus:** `transformers` is declared but never imported, and the comment at `setup.py:73-79` explains `huggingface-hub<1.0` is capped *because of* transformers. Dropping the unused dep frees hugginface-hub to 1.x.

**Fix the double-OpenCV bug while you are here.** `setup.py:37` puts `opencv-python-headless` in base and `setup.py:149` defines `'full': ['opencv-python']`. Both distributions install a top-level `cv2`; `pip install spacr[full]` lands both and which wins depends on install order. `setup.py:136` `'headless': ['opencv-python-headless']` is a no-op repeat of a base dep. Pick one: keep headless in base, delete the `full` extra.

**Proposed final dependency layout** — see §7.

---

### WAVE 4 — One device resolver
**Effort: 2–3 days. Mostly fixable blind; MPS and ROCm claims need a Mac and an AMD box to confirm.**

Today there is **no single place that decides the device** — there are ~24 independent decisions: 17 `torch.device(...)` ternaries (15 of them the byte-identical string `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`), 3 Qt call sites passing `gpu=torch.cuda.is_available(), device=None`, `GradCAM(use_cuda=True)`, `utils.py:5566`'s `torch.cuda.FloatTensor` ternary, and `gui_utils.initialize_cuda`. There is **no user-facing setting** to choose a device (`grep "'device'" spacr/settings.py` → only `n_jobs`) and **no `SPACR_DEVICE` env var**.

Full contract in §6. The four fixes it subsumes, in value order:

1. **Apple Silicon gets its GPU back.** Stop passing an explicit `device=torch.device('cpu')` into Cellpose at `object.py:706-707,1123,1463`, `spacr_cellpose.py:107,110,332,336`, `model_compare.py:1056-1059`, `utils.py:5228-5232` — `device` overrides `gpu` in `CellposeModel.__init__`, and Cellpose 4 is thoroughly MPS-aware (it even carries an explicit MPS workaround in `dynamics.compute_masks`). And flip the three Qt sites (`qt/annotate_engine.py:143-148`, `qt/widgets/live_preview.py:362-371`, `qt/widgets/timelapse_preview.py:321-328`) from `gpu=torch.cuda.is_available()` to `gpu=(backend != 'cpu')`. `spacr_cellpose.py:110` is the sharpest case: `CellposeModel(gpu=True, pretrained_model='cpsam', device=device)` — the author *wanted* GPU and the explicit CPU device silently won.
2. **Stop running bfloat16 on CPUs that can't do it.** `grep -rn use_bfloat16` returns nothing, so Cellpose 4's `use_bfloat16=True` default applies at all 8 `CellposeModel(...)` sites. Measured on this box (Zen 3, `avx512_bf16: False`): conv2d fp32 197 ms vs bf16 582 ms (**2.9×**), matmul fp32 37 ms vs bf16 179 ms (**4.9×**). Pass `use_bfloat16=dc.use_bfloat16`.
3. **`pin_memory` must not raise.** `spacr/io.py:623` `_pin_memory_batch` calls `Tensor.pin_memory()` **directly**, bypassing the DataLoader's own downgrade-with-warning guard — PROVED to raise `RuntimeError: Unexpected error from cudaGetDeviceCount()` on a non-CUDA machine. It is user-reachable (a checkbox at `gui_elements.py:6071`) and it fires on a producer thread whose `except Exception as e: self._error = e` defers the error to somewhere unrelated. `deep_spacr.py:1402` hardcodes `pin_memory=True` in a DataLoader. **The correct pattern already exists eleven lines away**: `deep_spacr.py:170` writes `pin_memory=(device.type == 'cuda')`.
4. **`GradCAM` stops calling `.cuda()` unconditionally.** `utils.py:5640-5647` defaults `use_cuda=True` and calls `model.cuda()`, `x.cuda()`, `one_hot.cuda()` with no availability check. No internal callers (the pipeline uses the correct `GradCAMGenerator`), so it is a public-API landmine rather than a live break — `spacr.utils.GradCAM(model, ['layer'])` dies on every Mac and every CPU box.

**Also in this wave, cheaply:** wrap the unguarded `GPUtil.getGPUs()` at `gui_core.py:1126` — it is inside `update_usage`, which re-arms every 500 ms via `after()`, so on a driver mismatch it throws a `ValueError` into the Tk event loop twice a second (PROVED here: `GPUtil` parses `"Failed to initialize NVML: Driver/library version mismatch"` as an int). `gui_core.py:1188` and the Qt equivalent are already guarded; line 1126 was missed. Move `gputil`/`gpustat` to an extra.

**Dead code to delete rather than fix:** `_get_cellpose_batch_size` (`utils.py:1429-1452`) prints *"CUDA is not available. Please check your installation and GPU"* on every Mac, and has **zero callers** in `spacr/`. `get_cuda_version()` (`utils.py:7253`) shells out to `nvcc` — zero callers, returns `None` under ROCm.

---

### WAVE 5 — One multiprocessing contract
**Effort: 3–4 days. Linux parts verifiable here; Windows/macOS behaviour needs Wave 2's CI.**

Today the start method is decided in four incompatible ways: `gui.py:317` and `gui_core.py:1676` force `spawn` (the latter with `force=True`, *inside* `initiate_root()`, i.e. after Tk is up); `reset_mp` (`utils.py:1274-1289`) would force `fork` on Linux **and Darwin**; and the Qt path — the default entry point — sets nothing and inherits `fork` on Linux.

| # | Change | Files |
|---|---|---|
| 5.1 | **Take an explicit context everywhere.** Replace bare `mp.Pool(...)` / `Manager()` with a module-level `_CTX = mp.get_context("forkserver" if sys.platform != "win32" else "spawn")`. `forkserver` was measured end-to-end on `measure_crop` here and produced correct DB rows. This kills the Qt fork-during-import deadlock, and it is what Python 3.14 will do by default on POSIX anyway — so you are moving *to* the future default, not away from it. | `measure.py:2158`, `object.py:1939`, `sim.py:1107`, `sequencing.py:435,539`, `timelapse.py:7450`, `io.py:523,3354,5352`, `utils.py:3614` |
| 5.2 | **Never `set_start_method(force=True)` in library code.** `gui_core.py:1676` mutates `multiprocessing._default_context` for the whole interpreter, irreversibly — a user who set `forkserver` for their own analysis and then opens the Tk GUI gets spawn for everything afterwards, silently. Use the context object from 5.1 and thread it through `Process`/`Queue`/`Value`/`Pool`/`Manager` (all four are currently imported bare in `gui_core.py`). | `gui_core.py:1676`, `gui.py:317` |
| 5.3 | **Delete `reset_mp`.** Zero callers outside tests. Its Darwin branch forces `fork` on the one platform where CPython changed the default *to* spawn precisely because fork corrupts Objective-C runtime state (Accelerate, the macOS matplotlib backend, Qt). It is also the only thing in the tree that could force `fork` on a 3.14 Linux box and re-introduce the deadlock. Delete `close_file_descriptors` (`utils.py:1301-1310`, `import resource` — cannot even import on Windows) and `close_multiprocessing_processes` (`utils.py:1312-1331`, which iterates **every process on the machine** via `psutil.process_iter` and terminates anything whose cmdline contains `"multiprocessing"`, then closes fds 3..NOFILE **in the calling process**) at the same time. All three are dead, public, and dangerous. | `utils.py:1274-1331`, `tests/test_cov_utils_settings_mp.py` |
| 5.4 | **Cap and floor `n_jobs`.** `measure.py:2158` has no cap at `len(files)`: a 16-core Windows box with default `n_jobs=12` boots 12 interpreters (**measured: 13.6 s and 1.2 GB each — 15 GB**) to measure a 4-field test plate. `object.py:1934` already does `min(n_jobs, n_items)` correctly — copy it. And `sequencing.py:413,517` use `cpu_count() - 3` with no floor, which **raises `ValueError: Number of processes must be at least 1`** on any 2–3 core host (CI runners, small VMs, ARM boards); `measure.resolve_n_jobs` (`measure.py:137`) and `sim.py:1103` already guard with `max(1, ...)`. | `measure.py:2158`, `sequencing.py:413,517` |
| 5.5 | **Give workers a real output channel.** Pass an explicit `mp.Queue` into `_measure_crop_core` (picklable under spawn) and drain it in the parent, or use `multiprocessing.get_logger()` + `QueueHandler`. Never rely on inherited `sys.stdout` in a worker: under fork the worker's `print` calls `Signal.emit` in a dead QApplication copy and the text vanishes; under spawn/forkserver it goes to a terminal a bundled app does not have. This is why every worker failure currently presents as "it just stopped". | `qt/bridge.py:117-121`, `gui_utils.py:330-331`, `measure.py` |
| 5.6 | **Make abort actually abort.** `gui_core.py:1256`'s `terminate()` SIGTERMs the intermediate process; the default handler skips `multiprocessing.util._exit_function` so daemonic pool workers survive (reproduced: 3 orphans still running, still holding VRAM). Use `psutil.Process(pid).children(recursive=True)` → terminate → `wait(3)` → kill, or install a SIGTERM handler in `run_function_gui` that calls `pool.terminate()`. The Qt GUI has no path to the pool at all (`_on_stop` only calls `requestInterruption()`, which nothing checks) — give it one. | `gui_core.py:1256`, `qt/screens/app_screen.py:1198-1210` |
| 5.7 | **Reduce the spawn tax.** Hoist `_measure_crop_core`'s function-local imports (`measure.py:1548-1550`) to module level; add `Pool(initializer=...)` to warm the chain once; and **stop pulling `umap` from the measure path** — `spacr/utils.py:88` → `umap/__init__.py:7` → `umap/parametric_umap.py:13` does `import tensorflow as tf`, which on any machine with TF installed adds ~1600 modules and 400 MB *per worker*, on a project with a standing TF ban. spaCR's own source imports no TF; this arrives through umap and should be cut off with a lazy import. | `measure.py:1548-1550`, `utils.py:88` |
| 5.8 | `mp.freeze_support()` as the first statement of `spacr.qt.run()`. Cheap insurance; `packaging/spacr_launcher.py:17` already does it for the shipped installers. | `qt/__init__.py:26` |
| 5.9 | Document the `__main__` guard. Nothing in the docs or notebooks tells a Windows/macOS user to wrap `measure_crop(...)` / `preprocess_generate_masks(...)` in `if __name__ == "__main__":`. Installed console scripts are safe (pip's wrappers carry the guard, and `spawn._fixup_main_from_name` short-circuits on `*.__main__`) — user scripts are not. | docs, `Notebooks/` |

---

### WAVE 6 — Windows and cross-OS correctness
**Effort: 4–6 days. Writable blind; *verifiable* only on Windows (Wave 2's CI, or a VM).**

Ordered within the wave by damage.

| # | Finding | Fix | Files |
|---|---|---|---|
| 6.1 | **Silent crop loss on non-ASCII paths.** `cv2.imwrite`'s return value is dropped on the primary crop path. OpenCV hands the filename to `fopen()` through the MSVC CRT, which uses the **ANSI codepage** — `C:\Users\Müller\`, a CJK username, an accented plate name all make it return `False` without raising. Worse: `stamp_crop_folder` runs *first* (`measure.py:1457`) so the folder is marked valid-and-empty, and `img_paths.append` (`measure.py:1862`) → `filepaths_to_database` (`utils.py:821-870`) records every path regardless. **The DB claims N crops, zero exist, the run reports success.** | Add `_imwrite`/`_imread` wrappers using `cv2.imencode`/`imdecode` + Python's Unicode-correct `open()`, and route the ~8 user-path call sites through them. `crops.py:1637` already does it right (`if not cv2.imwrite(...): raise`) — that's the convention. | `measure.py:1459`, `spacr_cellpose.py:207,313`, `plot.py:4627`, `crops.py` |
| 6.2 | **Windows→Linux workflows are broken.** `utils.py:1716` hard-codes `f"{source_folder}/data/"` and the following `os.path.join`s add `\`, producing `D:\Screens\plate1/data/single_nucleus/.../plate1_A01\cell_png/x.png` in `png_list.png_path`. Both relocation helpers then fail (reproduced): `io.py:4287`'s `p.split('/data/')` leaves a literal backslash → ENOENT; `agreement.py:730`'s `lstrip("/\\")` leaves the drive letter → `/mnt/data/plate1/D:\Screens\...`. This is *the* normal screening workflow — acquire on the instrument PC, analyse on the cluster — and it breaks Annotate, Agreement and Classify outright. | Store `png_path` **relative to `src`, POSIX-normalised** (`os.path.relpath(p, src).replace(os.sep, '/')`) and resolve at read time. Ship a one-shot migration for existing DBs — spaCR already has the pattern in `rename_columns_in_db`. Related: `utils.py:4178`'s `x.split('/')[-1]` returns the whole path on Windows, and the next line's 4-column split then raises `ValueError: Columns must be same length as key`; use `os.path.basename`. | `utils.py:1716,1742,4178`, `measure.py:1859-1861`, `io.py:4287`, `qt/screens/agreement.py:730` |
| 6.3 | **Every batch-queue job dies at the first arrow it prints.** No Windows codepage encodes spaCR's own output set (tested: `→` fails cp1252 *and* cp437; `┌│└` fail cp1252; `—` and `…` fail cp437). Harmless while stdout is a real console, fatal the moment it is redirected — and `batch.py:1288` hands the child a file descriptor, so `spacr-run` inherits a pipe and falls back to cp1252. Twelve plates queued overnight → twelve `UnicodeEncodeError` tracebacks. | Two independent changes, do both: (a) in `batch.py:subprocess_runner`, pass `env` with `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` — the parent's `encoding='utf-8'` on line 1283 does **not** reach the child; (b) at the top of `cli.py:main()` and both GUI entry points, `sys.stdout.reconfigure(encoding='utf-8', errors='replace')`. `errors='replace'` is the belt: a box-drawing header degrades to `?` instead of killing an eight-hour run. | `batch.py:1283-1288`, `cli.py`, `gui_utils.py:330`, `qt/bridge.py:120` |
| 6.4 | **33 text-mode `open()` calls with no `encoding=`** (my count on this checkout), plus ~29 `Path.write_text`/`read_text`. This matters *specifically because of the domain*: `cp1252` cannot encode `Δku80` — the standard *T. gondii* parental strain — nor `α-tubulin`, `≥`, `β-gal`; `cp932`/`cp936` cannot encode `µm`. A settings CSV with `pathogen: RH Δhxgprt` raises at the *end* of a run, when the journal is sealed. `run_journal.py:270` is a guaranteed round-trip bug on any non-UTF-8 locale (reads with `encoding="utf-8"`, writes with `write_text` in the locale encoding — and the content contains the `→` from 6.3). | Add `encoding="utf-8"` everywhere. Then add `tests/test_encoding_hygiene.py` — the same AST scan asserting zero hits, ~25 lines, catches it forever. **Credit where due:** all 122 `csv.writer` sites already pass `newline=""` correctly; only the encoding half was missed. | `spacrops.py` (10 sites), `run_journal.py:256,270,455`, `pipeline_v2.py`, `resume.py`, `qt/*`, `ml.py:824` |
| 6.5 | **MAX_PATH.** Measured overhead of `_generate_names` + the crop write is **87–146 characters**, so a 116-char source folder overflows. A real case measured: a 107-char OneDrive-for-Business path produces a **246-character** crop path — 14 characters of headroom. Windows reports `ENOENT`, not `ENAMETOOLONG`, so the user sees "No such file or directory" for a folder plainly visible in Explorer. And `packaging/spacr.spec` sets **no manifest** (grep for `longPathAware` → nothing), so the packaged `.exe` is *strictly worse* than `pip install`, which inherits CPython's long-path-aware manifest. | (a) Add `<longPathAware>true</longPathAware>` to the PyInstaller `EXE()` manifest — one line. (b) Startup pre-flight: on Windows, refuse a `src` longer than ~110 chars with a message naming the limit. (c) Longer term shorten the tree — `data/multiple_nucleus/multiple_pathogens/` is 42 characters encoding two booleans; `data/mn_mp/` says the same in 11. | `packaging/spacr.spec:89-101`, `cli.py`, both GUI source-pickers, `utils.py:1716-1743` |
| 6.6 | **SQLite.** 42 of 87 `sqlite3.connect` sites have no `timeout` (default 5 s, which the multiprocess measure writers blow through), and `PRAGMA journal_mode=WAL` is set in exactly one place — `qt/annotate_engine.py:599` — which is *not* the measurements DB. Degrades sharply on Windows because "point spaCR at the plate folder" there very often means `Z:\` or a OneDrive-synced folder, and WAL **does not work at all** on a network filesystem (it needs shared memory). | Central `_connect()` helper: `timeout=30`, `PRAGMA journal_mode=WAL`, `synchronous=NORMAL`, `busy_timeout=30000`, with WAL skipped when the path is on a network volume. Route all 87 sites through it. Add a startup warning when `src` is on a remote mount (Windows `GetDriveTypeW == DRIVE_REMOTE` or a `\\` prefix; POSIX mount type in `nfs/cifs/smbfs/fuse.*`). | new helper in `utils.py`; 87 call sites |
| 6.7 | **Case sensitivity, both directions.** `f.endswith('.tif')` at `spacr_cellpose.py:126,254`, `submodules.py:198,199,316,317,526,966`; `.png` at `deep_spacr.py:1697,1806,2357`; `.npy` at ~15 sites. Older Zeiss/Leica exports write `.TIF` — skipped on **every** platform, reported as "0 images found" in a folder full of images. And constructed-name lookups (`object.py:1729`) succeed on macOS/Windows and fail on Linux for the same data — a pipeline validated on the PI's MacBook fails on the cluster. | `f.lower().endswith(('.tif','.tiff'))` throughout — ~25 sites, purely mechanical. `model_compare.py:1138` and `plot.py:4908` already do it right. | as listed |
| 6.8 | **`os.replace` under AV/indexer contention.** The temp-then-replace pattern is **correct** (`os.replace` overwrites on Windows unlike `os.rename`, and the temp file is correctly created in the destination directory) — needs no redesign. The residual Windows-only gap is `MoveFileEx` failing with `PermissionError` when the *destination* is open without `FILE_SHARE_DELETE`: a real-time AV scanner, an Explorer preview handler, a second spaCR process. | Bounded retry with backoff, Windows only, re-raising after 5 attempts. | `io.py:2490`, `crops.py:1225,1645`, `convert.py:1515` |
| 6.9 | **`RotatingFileHandler` rollover on Windows.** `logger.py:37-42` writes to `Path.home()/log` with `maxBytes=2_000_000`; `doRollover()`'s `os.rename` raises `PermissionError [WinError 32]` when pool children hold the same file open. 2 MB is about one long segmentation run of `print_progress` output, so this is routine. | Per-process log files under `~/.spacr/logs/spacr-{pid}.log` (which `logging_util.py:90` already does correctly), or rotate only in the parent. Also move `logger.py` out of `~` root. | `logger.py:37-42`, `logging_util.py:152`, `qt/verbose_logger.py:93` |
| 6.10 | **`_has_display()` and `ml.py:58` disagree.** `cli.py:836` accepts `WAYLAND_DISPLAY`; `ml.py:58` checks only `DISPLAY`. On a Wayland-only desktop with no XWayland (Fedora/GNOME default, RHEL 9 default) `cli.py` says "display exists" while importing `spacr.ml` forces `Agg` — `plt.show()` silently draws nothing. Both also return `True` unconditionally on Darwin, wrong for macOS over SSH. | One shared helper imported by both: add `WAYLAND_DISPLAY`; on Darwin treat `SSH_CONNECTION`/`SSH_TTY` with no `TERM_PROGRAM` as headless. | `cli.py:834-838`, `ml.py:58` |
| 6.11 | **Unsanitised user strings become directory names.** No filename sanitiser exists anywhere (`grep -niE 'def .*(sanitiz|safe_name|slugify)'` finds only `qt/ai/issue_report.py:96 sanitize_path`, which redacts for bug reports). `sim.py:984` and `plot.py:4459` take free text straight from the GUI. `sim: run 1` → `OSError [WinError 123]` on Windows; a `/` silently creates a nested folder on Linux. | One `safe_component()` helper in `utils.py` (strip `<>:"/\|?*\x00-\x1f`, handle `CON/PRN/AUX/NUL/COM1-9/LPT1-9`, cap length), applied wherever a settings string becomes a path component. | `utils.py`, `sim.py:984`, `plot.py:4459` |
| 6.12 | **`format_path_for_system` (`utils.py:7944`) is wrong on POSIX.** It does `path.replace("\\", "/")` on Linux/macOS — but backslash is a **legal POSIX filename character**, so it corrupts real paths. It also raises `ValueError: Unsupported OS` for anything outside the big three, a crash where a no-op would do. Both callers (`core.py:159`, `measure.py:2014`) only need normalisation. | Replace the whole body with `os.path.normpath(os.path.abspath(os.path.expanduser(path)))` and delete the branching. | `utils.py:7944` |
| 6.13 | Small, visible, cheap: `qt/app.py:724` builds `file://C:\Users\...` (needs `file:///C:/Users/...`; better, `os.startfile`/`open`/`xdg-open`). `qt/notify.py:56` imports `win10toast`, which is **in no dependency list anywhere** (grepped all five), is unmaintained since 2020, and has no ARM64 support — the `except Exception: return False` hides it; use `QSystemTrayIcon.showMessage`, already implemented as `notify_tray`. On Darwin `notify()` returns `True` unconditionally after `osascript` (`check=False`), so on macOS 12+ where the notification is silently suppressed, the working tray fallback is **never called**. `updater.py`'s `run_pip_upgrade` cannot work on Windows (the running `Scripts\spacr.exe` cannot overwrite itself) — show the command instead. `qt/app.py:425,468` should set `setMenuRole` explicitly (`grep setMenuRole` → 0 hits) since Qt's `TextHeuristicRole` relocates anything containing `settings`/`config`/`options`/`quit` into the macOS app menu, and spaCR's menus are full of the word "settings". | as described | `qt/app.py:425,468,724`, `qt/notify.py`, `updater.py` |

**Do not "fix" these — they are already right**, and each was written deliberately: `gui_utils.py:163` `set_cpu_affinity` (Linux-only is correct; `cpu_affinity` genuinely does not exist on macOS), `qt/theme.py:403` `_qss_url`, `qt/screens/db_browser.py:207` `_read_only_uri` (percent-escapes while preserving the drive colon), `qt/dnd.py:214` (`QUrl.toLocalFile()`), all 122 `csv.writer` `newline=""` sites, and the temp-then-`os.replace` pattern in 6.8. One genuine gap in an otherwise-correct site: `gui_core.py:517`'s mouse-wheel binding gives Linux only `Button-4/5`, but **Tk 8.7/9.0 on Linux emits `MouseWheel` instead** — add it to the Linux branch before distros pick up Tk 9.

---

### WAVE 7 — Packaging and architecture
**Effort: 2–3 days. Needs a Mac and a Windows box (or CI runners) to verify.**

| # | Change | Files |
|---|---|---|
| 7.1 | **The installers ship the wrong GUI.** `packaging/spacr_launcher.py:20` runs `spacr.gui.gui_app()` — the legacy **Tk** GUI — while `setup.py:118` maps `spacr` → `spacr.qt:run` (**Qt**). The `.deb`/`.dmg`/`.exe` are a different application from `pip install spacr`. Point the launcher at `spacr.qt:run` (keep the existing `freeze_support()` first). | `packaging/spacr_launcher.py:20` |
| 7.2 | **Frozen builds are mute.** `spacr.spec:99` sets `console=False`, so `sys.stdout`/`sys.stderr` are `None` in a windowed Windows build: `spacr_launcher.py:22`'s `print(..., file=sys.stderr)` produces **no output at all** — a failed import dies with exit code 2 and a blank screen. `cli.py:930`'s `sys.stdout.isatty()` would raise `AttributeError`. | Redirect both streams to `~/.spacr/logs/` at the top of the launcher; guard `cli.py:930` with `getattr(sys.stdout, 'isatty', lambda: False)()`. | `packaging/spacr_launcher.py`, `cli.py:930` |
| 7.3 | **The .dmg is single-architecture and unlabelled.** `spacr.spec` sets no `target_arch`, and `build_macos.sh:63` names the output `spaCR-$VERSION.dmg` with no arch. A .dmg built on an Intel runner runs on Apple Silicon only under Rosetta 2, with x86-64 torch inside — **no MPS at all**, which is the entire GPU story on a Mac. Two mutually incompatible artefacts, one filename. | Build both; name them `-arm64.dmg` / `-x86_64.dmg`. `universal2` is not realistic — torch/numpy/opencv do not ship universal2 wheels. Also replace `build_macos.sh:53`'s `codesign --deep` (deprecated since macOS 11, fails notarization on 13+) with per-binary signing, innermost first. | `packaging/spacr.spec`, `packaging/build_macos.sh:53,63` |
| 7.4 | Add `<longPathAware>` manifest (see 6.5) and set the Windows arch explicitly. | `packaging/spacr.spec` |
| 7.5 | **README honesty section.** State plainly: Windows-on-ARM unsupported (no torch wheel exists); Intel Mac supported at torch ≤2.2.2 or not at all — pick one and say which; ARM Linux supported wheel-only after Wave 3; ROCm requires the pre-install step in §6. | `README.rst` |

---

## 5. The CI matrix I would actually write

**What CI can cover:** every OS × arch × Python cell that has a GitHub-hosted runner, at the level of *install + import + the non-GPU test suite*. That is a real chunk of the matrix and it catches, at minimum: wheel availability (all of §2), the encoding bugs (6.3, 6.4), the path-separator bug (6.2), case sensitivity (6.7), SQLite locking (6.6), MAX_PATH (6.5), the spawn tax and `n_jobs` floor (5.4), and every future regression of the same shape.

**What CI cannot cover:** CUDA (needs a self-hosted NVIDIA runner), ROCm (needs a self-hosted AMD runner — there is no hosted option anywhere), real MPS throughput (`macos-14` runners *do* have Apple GPUs, so MPS *availability* is testable; MPS *performance* is not meaningfully measurable on a shared runner), the packaged `.exe`/`.app`/`.deb` at runtime (buildable in CI, not launchable), and anything needing a real display or real microscope data.

```yaml
name: tests

on:
  push:
    branches: [main, nightly, spacr-nightly, "**"]
  pull_request:
    branches: [main]
  workflow_dispatch:

jobs:

  # ---------------------------------------------------------------
  # 1. Import smoke — the cheap wide net. Installs and imports every
  #    top-level module on every cell we can reach. Catches wheel
  #    availability, top-level import errors, encoding-at-import and
  #    platform-guard bugs in ~4 minutes per cell.
  # ---------------------------------------------------------------
  smoke:
    name: smoke ${{ matrix.os }} py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    continue-on-error: ${{ matrix.experimental }}
    timeout-minutes: 25
    strategy:
      fail-fast: false
      matrix:
        experimental: [false]
        include:
          # --- Linux x86-64: the reference platform ---
          - {os: ubuntu-24.04,    python: "3.10", experimental: false}
          - {os: ubuntu-24.04,    python: "3.11", experimental: false}
          - {os: ubuntu-24.04,    python: "3.12", experimental: false}
          # RED until Wave 3 (numpy<2.0, mahotas)
          - {os: ubuntu-24.04,    python: "3.13", experimental: true}
          # RED until Wave 3 + pylibCZIrw/mahotas move to extras
          - {os: ubuntu-24.04,    python: "3.14", experimental: true}

          # --- Linux ARM64 (Graviton-class; free for public repos) ---
          # RED until Wave 3 (mahotas/psutil/aicspylibczi have no aarch64 wheel)
          - {os: ubuntu-24.04-arm, python: "3.12", experimental: true}

          # --- Windows x86-64 ---
          - {os: windows-2022,    python: "3.11", experimental: false}
          - {os: windows-2022,    python: "3.12", experimental: false}

          # --- macOS Apple Silicon (macos-14+ is arm64) ---
          - {os: macos-14,        python: "3.11", experimental: false}
          - {os: macos-14,        python: "3.12", experimental: false}

          # --- macOS Intel (macos-13 is the last x86-64 runner image) ---
          # DEGRADED: torch backtracks to 2.2.2. Kept non-blocking so the
          # day GitHub retires macos-13 we notice, not so it gates merges.
          - {os: macos-13,        python: "3.12", experimental: true}

          # --- Windows on ARM: documents the gap, never expected green ---
          # torch has NEVER shipped a win_arm64 wheel. This cell exists so
          # that the day it does, CI tells us.
          - {os: windows-11-arm,  python: "3.12", experimental: true}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python }}
          cache: pip

      - name: Install CPU-only torch first (keeps the CUDA wheel off the runner)
        run: |
          python -m pip install --upgrade pip
          python -m pip install --index-url https://download.pytorch.org/whl/cpu \
            "torch>=2.0" "torchvision"

      - name: Install spacr
        run: python -m pip install -e ".[qt,dev]"

      - name: Import every module
        env:
          QT_QPA_PLATFORM: offscreen
          PYTHONUTF8: "0"            # deliberately NOT forced: we want to catch
                                     # encoding bugs the way a real user hits them
        run: |
          python -c "import spacr, spacr.utils, spacr.io, spacr.measure, spacr.core, spacr.plot, spacr.settings, spacr.ml, spacr.timelapse, spacr.deep_spacr, spacr.crops, spacr.sequencing; print('ok', spacr.__file__)"
          python -c "import spacr.qt"
          spacr-run --list

      - name: Report resolved backend
        run: python -c "from spacr.utils import resolve_device; print(resolve_device())"
        # (lands with Wave 4; harmless to add early as `|| true`)

  # ---------------------------------------------------------------
  # 2. Full suite — the deep net, on the cells where it is affordable.
  #    Runs the real tests, including the multiprocessing and path
  #    round-trip tests that only fail off-Linux.
  # ---------------------------------------------------------------
  tests:
    name: tests ${{ matrix.os }} py${{ matrix.python }}
    runs-on: ${{ matrix.os }}
    needs: smoke
    continue-on-error: ${{ matrix.experimental }}
    timeout-minutes: 60
    strategy:
      fail-fast: false
      matrix:
        include:
          - {os: ubuntu-24.04, python: "3.10", experimental: false}
          - {os: ubuntu-24.04, python: "3.12", experimental: false}
          # RED until Wave 6 (encoding, separators, MAX_PATH, sqlite)
          - {os: windows-2022, python: "3.12", experimental: true}
          # RED until Wave 5 (spawn tax / n_jobs cap) + Wave 4 (MPS)
          - {os: macos-14,     python: "3.12", experimental: true}

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "${{ matrix.python }}", cache: pip}

      - name: Linux Qt/offscreen system libs
        if: runner.os == 'Linux'
        run: |
          sudo apt-get update
          sudo apt-get install --no-install-recommends -y \
            libegl1 libxkbcommon0 libdbus-1-3 libpulse0 libx11-xcb1 \
            libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 \
            libxcb-render-util0 libxcb-shape0 libxcb-sync1 libxcb-xfixes0 \
            libxcb-xinerama0 libxcb-xkb1 libxkbcommon-x11-0 libgl1 \
            libglib2.0-0 libnss3 libxcomposite1 libxcursor1 libxdamage1 \
            libxi6 libxtst6 libasound2 libsm6 libxext6 libxrender1 ffmpeg

      - run: |
          python -m pip install --upgrade pip
          python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" "torchvision"
          python -m pip install -e ".[qt,dev]"

      - name: Run tests
        env:
          QT_QPA_PLATFORM: offscreen
          PYTHONPATH: ${{ github.workspace }}
        run: python -m pytest tests/ --ignore=tests/test_gpu_pipeline.py -m "not slow and not heavy" -v --tb=short

  # ---------------------------------------------------------------
  # 3. Hostile-path job — the bugs that only appear with a user whose
  #    name has an umlaut and a plate called "RH Δhxgprt". This is the
  #    cheapest possible guard for findings 6.1, 6.2, 6.4 and 6.11.
  # ---------------------------------------------------------------
  hostile-paths:
    name: hostile paths ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    needs: smoke
    continue-on-error: true          # until Wave 6
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-24.04, windows-2022, macos-14]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12", cache: pip}
      - run: |
          python -m pip install --upgrade pip
          python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch>=2.0" "torchvision"
          python -m pip install -e ".[qt,dev]"
      - name: Run the pipeline under a non-ASCII, deeply-nested path
        env:
          QT_QPA_PLATFORM: offscreen
          # No PYTHONUTF8 / PYTHONIOENCODING: reproduce the cp1252 default.
        run: python -m pytest tests/ -m "hostile_path" -v --tb=short

  # ---------------------------------------------------------------
  # 4. Build the installers on every OS. Catches spec-file breakage
  #    early. Does not (cannot) launch them.
  # ---------------------------------------------------------------
  package:
    name: package ${{ matrix.os }}
    runs-on: ${{ matrix.os }}
    if: github.ref == 'refs/heads/main' || github.event_name == 'workflow_dispatch'
    continue-on-error: true
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-24.04, windows-2022, macos-14, macos-13]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.12"}
      - run: python -m pip install pyinstaller && python -m pip install -e ".[qt]"
      - run: pyinstaller packaging/spacr.spec
      - uses: actions/upload-artifact@v4
        with: {name: "spacr-${{ matrix.os }}", path: dist/}
```

**Runner notes.** `ubuntu-24.04-arm` and `windows-11-arm` are GitHub-hosted ARM labels (free for public repos); confirm the exact labels against GitHub's current runner-images list before merging, since they are the newest and most likely to have been renamed. `macos-14` and `macos-15` are arm64; `macos-13` is the last x86-64 image and will be retired — treat that cell as documentation of the Intel-Mac gap, not as a gate.

**Self-hosted, and unavoidable:**

| Cell | Runner | What only it can prove |
|---|---|---|
| CUDA | self-hosted NVIDIA (this dev box qualifies) | `tests/test_gpu_pipeline.py`, real Cellpose throughput, VRAM behaviour, the `empty_cache` sites under load |
| ROCm | self-hosted AMD (MI-series or Radeon) | that `torch.version.hip` routing works, that `cudnn.benchmark` → MIOpen search is acceptable, that pinned memory works under HIP. **There is no hosted alternative anywhere.** |
| MPS throughput | a physical Mac | the 10–30× MPS-vs-CPU claim, and whether any op in the spaCR path falls back |

Add `pull_request` gating only on the non-experimental cells, so a red Windows job informs without blocking until Wave 6 lands.

---

## 6. Accelerator policy: one `resolve_device()`

**The problem in one sentence: today a user cannot tell which backend ran.** The only signal is `print('Torch CUDA is not available, using CPU')` from three of thirty-two `is_available()` sites, and on a Mac that message is not just unhelpful, it is false — there is nothing to check and nothing to fix.

### The contract

One function in `spacr/utils.py`, called by all ~24 decision sites, returning a frozen dataclass — **not** a bare `torch.device`, because the four things that must agree with the device are exactly the four things currently guessed independently.

```python
@dataclass(frozen=True)
class DeviceChoice:
    torch_device: torch.device
    backend: str            # 'cuda' | 'rocm' | 'mps' | 'cpu'
    name: str               # GPU model, or CPU model
    vram_gb: float | None
    use_bfloat16: bool
    pin_memory: bool
    cellpose_kwargs: dict   # {'gpu':…, 'device':…, 'use_bfloat16':…}
    reason: str             # 'auto' | 'settings' | 'env' | 'explicit'

def resolve_device(prefer=None, *, verbose=True) -> DeviceChoice: ...
```

| Field | Value | Replaces |
|---|---|---|
| `torch_device` | `torch.device(...)` | the 17 ternaries |
| `backend` | ROCm distinguished by `torch.version.hip is not None` | nothing — currently unknowable from spaCR |
| `name` | `torch.cuda.get_device_properties(i).name`, `platform.processor()`, or the Apple chip | `_get_cellpose_batch_size`'s print |
| `vram_gb` | CUDA/ROCm → `total_memory`; MPS → `torch.mps.recommended_max_memory()`; CPU → `None` | `_get_cellpose_batch_size` (dead code) |
| `use_bfloat16` | `True` only for CUDA ≥SM80 / ROCm gfx90a+ / MPS. **`False` on CPU** | the measured 2.9–4.9× CPU penalty |
| `pin_memory` | `backend in ('cuda','rocm')` — pinned memory *does* work under HIP | `io.py:623`, `deep_spacr.py:1402` |
| `cellpose_kwargs` | one dict, spread into all 8 `CellposeModel(...)` sites | the `device`-overrides-`gpu` bug |

**Auto-resolution order:** `cuda`/`rocm` if `torch.cuda.is_available()` → else `mps` if `torch.backends.mps.is_available()` → else `cpu`. **ROCm arrives free through the same branch as CUDA**, which is exactly what the HIP shim intends — that is why AMD support is mostly a documentation and reporting problem, not a code problem.

**Override, three layers, most specific wins:**
1. explicit `prefer=` argument
2. a new `'device'` key in `spacr/settings.py`, surfaced in both GUIs (**currently absent entirely** — there is no way for a user to choose)
3. `SPACR_DEVICE` env var, matching the existing `SPACR_MASK_FORMAT` / `SPACR_LOG_DIR` convention

**An explicit request that cannot be satisfied must raise, never fall back.** A user who typed `device: cuda` on a Mac wants an error, not eight silent CPU hours.

### What it prints — once per process, at first resolution

```
spaCR device: MPS (Apple M3 Max, 36 GB unified)  [auto]
  cellpose gpu=True device=mps bfloat16=True | pin_memory=False | cudnn.benchmark=off
```
```
spaCR device: ROCm (AMD Instinct MI300X, 192.0 GB)  [auto]
  torch 2.9.1+rocm6.2 (HIP 6.2) | cellpose gpu=True device=cuda:0 bfloat16=True
```
```
spaCR device: CPU (AMD Ryzen 9 5950X, 16 cores)  [auto — no CUDA/ROCm/MPS found]
  bfloat16 disabled (no AVX512-BF16 on this CPU: ~3x slower than fp32)
  If you have an AMD GPU, install ROCm torch BEFORE spacr:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
```

That banner alone converts three silent multi-hour slowdowns into one line of output. Add `spacr-doctor` as a console script that prints it plus the wheel/platform summary — it is the first thing to ask for in any bug report.

### On ROCm, specifically

**The good news is larger than expected.** `torch.cuda.is_available()` returns `True` on a ROCm build, so most of spaCR's CUDA paths already work. Audited risk surface:

| Risk | In spaCR? | Verdict |
|---|---|---|
| Compiled CUDA extension | **none** — `setup.py` has no `ext_modules` | pure Python; genuinely good news for the whole matrix |
| AMP / `autocast` / `GradScaler` | **none** | fine |
| `get_device_capability` / SM thresholds | **none** | fine |
| `CUDA_VISIBLE_DEVICES` read directly | **none** | fine — ROCm torch honours it |
| `nvcc` shell-out | `utils.py:7253`, **zero callers** | delete |
| `nvidia-smi` shell-out | transitively via `GPUtil` | make monitoring optional; add an AMD branch (`torch.cuda.memory_allocated()` works under ROCm) |
| `cudnn.benchmark = True` | `deep_spacr.py:2`, **at module import**, before any device is known | safe on CPU (verified). On ROCm it maps to **MIOpen exhaustive search** — the first conv of each new input shape triggers a kernel search that can take minutes. Gate on `device.type == 'cuda' and torch.version.hip is None` |
| Pinned memory | `io.py:623` | works on ROCm; breaks on Mac/CPU — see Wave 4 |
| Hardcoded `cuda:0` | 15 sites | no way to target GPU 1 except `HIP_VISIBLE_DEVICES`, undocumented. `resolve_device('cuda:1')` fixes it |

**The honest ROCm statement for the README:** ROCm torch is **not on PyPI at all** — it lives only at `download.pytorch.org/whl/rocmX.Y`. That cannot be expressed in `install_requires`, and PEP 508 has no GPU-vendor marker. So `pip install spacr` on an MI300X or a Radeon workstation downloads several GB of NVIDIA CUDA-13 runtime (torch's `requires_dist` markers are `platform_system == "Linux"`, full stop) and produces a torch that will never see the user's GPU. **Keep `torch` in `install_requires`** — removing it would break the default install for the majority to serve a minority — but document the pre-install step prominently, and let the banner say what actually happened.

---

## 7. What becomes an extra

Principle: **the core install must succeed, wheel-only, on every cell that has wheels.** Everything that blocks a whole matrix cell for one feature moves out.

```python
install_requires = [
    # numeric / imaging core — wheels everywhere, 3.10–3.14
    'numpy>=1.26.4,<3.0',           # was <2.0  — the 3.13/3.14 blocker
    'pandas>=2.2.1,<3.0',
    'scipy>=1.12.0,<2.0',
    'scikit-image>=0.22.0,<1.0',
    'scikit-learn>=1.4.1,<2.0',
    'scikit-posthocs>=0.10.0,<0.20',
    'statsmodels>=0.14.1,<1.0',
    'opencv-python-headless>=4.9.0.80,<6.0',   # cp37-abi3: fine on 3.13/3.14
    'pillow>=10.2,<13',             # was <11.0
    'tifffile>=2023.4.12',
    'imageio>=2.34.0,<3.0',
    'PyWavelets>=1.6.0,<2.0',
    'psutil>=5.9.8,<8',             # was <6.0  — the aarch64 blocker
    # torch stack (see README for the ROCm / CPU index-url pre-step)
    'torch>=2.0,<3.0', 'torchvision>=0.1,<1.0', 'torchcam>=0.4.0,<1.0',
    'captum>=0.7.0,<1.0',
    'cellpose>=4.0,<5.0',           # cellpose brings segment_anything itself
    'fastremap>=1.14.1',
    # analysis / plotting
    'seaborn>=0.13.2,<1.0', 'matplotlib>=3.8.3,<4.0', 'adjustText>=1.2.0,<2.0',
    'matplotlib_venn>=1.1,<2.0', 'shap>=0.45.0,<1.0', 'xgboost>=2.0.3,<4.0',
    'umap-learn>=0.5.6,<1.0', 'pingouin>=0.5.5,<1.0', 'biopython>=1.80,<2.0',
    'btrack>=0.7.0,<1.0', 'trackpy>=0.6.2,<1.0',
    # io / misc
    'openpyxl>=3.1,<4.0', 'tables>=3.8.0,<4.0', 'rapidfuzz>=3.9,<4.0',
    'pytz>=2023.3.post1', 'tqdm>=4.65.0', 'screeninfo>=0.8.1,<1.0',
    'importlib-metadata>=3.6,<10.0',
    'IPython>=8.18.1,<9.0', 'ipykernel', 'ipywidgets>=8.1.2,<9.0',
    'huggingface-hub>=0.25',        # cap lifts once `transformers` is dropped
]

extras_require = {
    # --- feature extras: each one unblocks a matrix cell -------------
    'zernike':  ['mahotas>=1.4.13,<2.0'],      # no cp313/cp314; no linux-aarch64
    'formats':  ['pylibCZIrw>=5.0.0,<7.0',     # no cp314; no win_arm64
                 'czifile', 'nd2reader>=3.3.0,<4.0', 'readlif'],
    'monitor':  ['gputil>=1.4.0,<2.0', 'gpustat>=1.1.1,<2.0'],   # NVIDIA-only, sdist-only
    'notify':   [],                            # QSystemTrayIcon only; win10toast dropped
    # --- GUI --------------------------------------------------------
    'qt':       ['PySide6>=6.6,<7', 'qtawesome>=1.3,<2'],
    'tk':       ['ttkthemes>=3.2.2,<4.0', 'customtkinter>=5.2.2,<6.0',
                 'ttf_opensans>=2020.10.30'],
    # --- tracking (unchanged) ---------------------------------------
    'trackastra': ['trackastra>=0.5,<1.0'],
    'ultrack':    ['ultrack>=0.6,<1.0'],
    # --- convenience -------------------------------------------------
    'dev':      ['pytest>=8.0,<9', 'pytest-qt>=4.4,<5'],
    'all':      ['spacr[zernike,formats,monitor,qt,trackastra]'],
}
```

**Removed outright:** `segment-anything` (single unattributed 2023 release, 0 imports, arrives via cellpose), `aicspylibczi` (0 imports, blocks ARM Linux), `monai`, `segmentation_models_pytorch`, `torch-geometric`, `wandb`, `gdown`, `brokenaxes` (0 imports each), `transformers` (0 imports; its removal frees the `huggingface-hub<1.0` cap), `lxml` (0 imports), `keyring`, `openai` (verify against the AI Console first), `protobuf` and `numexpr` (transitive-only), the `'full'` and `'headless'` extras (the double-`cv2` bug), and the entire `setup.py:175-190` pip block.

**Kept despite zero direct imports** — do not let a cleanup pass delete these: `tables` (pandas HDF engine, 5 `read_hdf`/`to_hdf` sites), `openpyxl` (pandas Excel engine, 2 sites), `imagecodecs` (transitive, backs `czifile`/`tifffile`), `numba` (transitive via umap/shap).

**Runtime behaviour of an extra that is missing** must be a clear sentence, not an `ImportError` from module scope. `spacr/convert.py:182-186` already has the right pattern (a `find_spec` table with per-format install hints); make `spacr/io.py:1`'s `czifile`/`readlif` imports lazy so that pattern actually holds, and use the same shape for `mahotas` in `_calculate_zernike`.

---

## 8. What can be fixed blind vs what needs the machine

| Wave | Blind-fixable | Needs hardware / that OS |
|---|---|---|
| 0 metadata | **all of it** | — |
| 1 trapezoid | **all of it** | — |
| 2 CI | **all of it** (the YAML) | the runners themselves report |
| 3 pins + extras | the edits | numpy-2 validation on 3.11/3.12/3.13; aarch64 install proof |
| 4 device | the resolver, bf16 gating, pin_memory, GradCAM, the banner | **MPS speedup and op coverage need a Mac; ROCm needs an AMD box.** Everything else is verifiable on CPU-only torch here |
| 5 multiprocessing | contexts, `n_jobs` caps, deletions, worker queue, freeze_support | Windows/macOS spawn behaviour and the abort path need those OSes (CI covers most) |
| 6 Windows correctness | every edit | **every verification.** Writing `_imwrite`, the relative-path migration and the UTF-8 hardening is blind work; proving they fixed anything needs a Windows box or the CI cell |
| 7 packaging | spec/launcher edits | a Mac and a Windows box to run the artefacts |

**A note on honesty in this document.** What I personally executed on this machine: every grep and count in §3, the `ast.parse(feature_version=(3,9))` scan over all 521 files, and every PyPI wheel-metadata query in §1, §2 and §7. What I did not execute: any Windows, macOS, ARM, 3.13, 3.14, ROCm or MPS behaviour — those are inferences from wheel metadata (strong, since a missing wheel is a fact) and from upstream documentation and reproductions the five audits ran on Linux (weaker for anything OS-specific). Where the audits disagreed with each other or with my own re-check, §3 says which one to believe.

---

## 9. If you only do five things

1. **Wave 0** — delete the `pip` subprocess block, add `requires-python = ">=3.10,<3.13"`, fix the classifiers. Half a day, and it turns spaCR's worst first contact (a numpy compiler error on the Python a 2026 laptop ships with) into one clear sentence.
2. **Wave 1** — `np.trapz` → `np.trapezoid`, five sites. One hour, and it is the tripwire on the only path forward: the existing `scipy.integrate.trapz` fallback is *already dead* in scipy versions spaCR already permits.
3. **Wave 2** — the CI matrix. Half a day, and roughly twenty of these findings stop being prose.
4. **Wave 3** — `numpy<3.0`, `psutil<8`, `pillow<13`; `mahotas` and the CZI stack to extras. Two to three days, and 3.13 plus ARM Linux plus a wheel-only core all arrive together.
5. **Wave 4's first hour** — the three Qt Cellpose sites (`gpu=True`) and the eight pipeline sites that pass an explicit CPU device. Every Apple Silicon user gets their GPU back, and it is the single largest performance delta in the entire audit set.
