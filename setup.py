# setup.py must do nothing but describe the package.
#
# It used to end with a module-level loop that shelled out to
# `subprocess.run(['pip', 'install', dep])`. That ran on every build and on
# every `pip install .`, which broke PEP 517 isolated builds (no pip inside
# the isolated env), broke every offline/air-gapped install, swallowed all
# failures behind a bare `except CalledProcessError: pass`, invoked the bare
# name `pip` (absent from PATH in many venv layouts), and installed an entire
# second, unused Qt binding: pyqtgraph, pyqt6, pyqt6.sip, qtpy and superqt —
# a hand-copy of cellpose's `gui` extra, duplicated within itself. 75 files
# under spacr/ import PySide6; zero import PyQt6, pyqtgraph, qtpy or superqt.
# The block is gone; nothing replaced it, because nothing needed it.
#
# Project metadata that PEP 621 owns (name, requires-python, classifiers,
# authors, URLs) now lives in pyproject.toml. The fields below stay here
# because packaging/build_{debian,macos,windows}.* and four tests read this
# file as text or execute it directly; pyproject.toml declares them
# `dynamic` so setuptools takes them from this call.
from setuptools import setup, find_packages

with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# ---------------------------------------------------------------------------
# Every bound below is meant to be evidence, not convention: a floor names an
# API spaCR actually calls, a ceiling names a break that was observed. Bounds
# audited end-to-end on 2026-07-26 against the PyPI JSON API and against two
# throwaway conda envs (CPython 3.12 and 3.13) built with the widened set. The
# audit changed bounds in three directions, and the direction matters:
#
#   * ceilings LOWERED, because the old one was too loose and admitted a
#     resolve that breaks spaCR at runtime (pingouin, scikit-image,
#     statsmodels). These were the most valuable findings in the audit:
#     `<1.0` on a 0.x package is decorative, because 0.x breaks at the MINOR.
#   * ceilings RAISED, because no break exists above them (pillow, psutil,
#     IPython, lxml, xgboost, protobuf).
#   * floors RAISED, because the old one admitted a resolve that is already
#     broken (scikit-learn) or was simply false (torchvision).
#
# Several floors are *higher* than the oldest release that would work — e.g.
# nothing spaCR calls needs matplotlib 3.8.3 (the newest API used, indexing
# `matplotlib.colormaps`, is 3.5), scikit-image 0.22 (the newest is
# `regionprops_table(spacing=)`, 0.20), pandas 2.2.1 (the newest is
# `DataFrame.attrs`, 1.0) or scipy 1.12. They are left alone deliberately:
# an over-tight floor only costs something when it blocks a resolution, and
# none of these do on any Python in `requires-python`. Lowering them would buy
# nothing and would admit combinations no spaCR test has ever run.
#
# ---------------------------------------------------------------------------
# 2026-07-27: the list was reconciled against what spaCR actually imports.
#
# The method, re-run independently rather than inherited: `ast.parse` every one
# of the 159 files under `spacr/`, `ast.walk` each tree (so imports inside
# functions, methods, `try`/`except` bodies and `if TYPE_CHECKING` all count,
# not just module scope), collect every top-level module name, then resolve
# each declared distribution to the import names it actually installs by
# reading the *installed* `top_level.txt` / RECORD rather than guessing from
# the project name — that is what catches opencv-python-headless -> `cv2`,
# pillow -> `PIL`, biopython -> `Bio`, gputil -> `GPUtil`. Every candidate was
# then re-checked with a raw `grep -rIn -w` over the whole tree, non-Python
# files included, to catch dynamic references a syntax tree cannot see.
#
# That last step mattered exactly once, and it is why the census is run rather
# than trusted: `umap-learn` has ZERO import statements anywhere in spaCR. It
# is reached through `umap = _LazyModule('umap.umap_', ...)` at
# spacr/utils.py:197 — the module name is a *string literal*, invisible to an
# import census. It is a core dependency and stays one. Any future pass over
# this list has to grep for the string form before believing a zero.
#
# 18 distributions had zero imports, zero string references and zero dynamic
# references, and are gone: transformers, monai, segmentation_models_pytorch,
# torch-geometric, PyWavelets, rapidfuzz, wandb, gdown, pytz, ipykernel,
# ttkthemes, ttf_opensans (spaCR bundles its own Open Sans under
# spacr/resources/font/), brokenaxes, gpustat, customtkinter, openai, keyring
# and importlib-metadata. Three of those had a *prose* hit and no code hit —
# "transformers" in a GUI blurb, "@openai/codex" in an npm one-liner, "OS
# keyring" in a stale docstring — which is precisely the difference a raw grep
# has to be read for rather than counted.
#
# 9 distributions were imported and never declared, and are now declared. Five
# are module-scope, unguarded imports, so the install was only ever working by
# accident — they arrived transitively and would vanish the day the package
# that dragged them in changed its own mind:
#     requests   spacr/utils.py:2 and spacr/gui_utils.py:1 (module scope).
#                Arrived via huggingface-hub, which is exactly the package
#                that drops requests for httpx at 1.0 — see the hub note.
#     joblib     spacr/utils.py:207 (module scope), spacr/object.py:49.
#                Arrived via scikit-learn.
#     natsort    spacr/submodules.py:36 (module scope). Arrived via cellpose.
#     patsy      spacr/ml.py:33 (module scope). Arrived via statsmodels.
#     sympy      spacr/gui_elements.py:26 (module scope). Arrived via torch.
# and four are function-local and already guarded, so they are declared for
# honesty rather than to fix a break: nvidia-ml-py (imported as ``pynvml`` at
# spacr/qt/widgets/home.py:729 and :743), win10toast (spacr/qt/notify.py:56,
# behind `if system == "Windows"`, so it carries the marker), and
# catboost/lightgbm, which are alternative model backends behind a
# `model_type=` string and live in the `boosting` extra — see the note there.
# ---------------------------------------------------------------------------
dependencies = [
    # -----------------------------------------------------------------------
    # `<2.0` used to be the single most expensive bound in this file: numpy
    # 1.26.4 was the ONLY release satisfying it, its wheels stop at cp312, and
    # that is what forced `requires-python = ">=3.10,<3.13"`.
    #
    # It is now `<3.0`, and every one of the three blockers that held it was
    # closed rather than argued away:
    #
    #   1. `np.trapz` (removed in numpy 2.0) is gone from all three call
    #      sites. spacr/utils.py:7 and spacr/attribution.py:66 now resolve
    #      `_trapezoid = getattr(np, 'trapezoid', None) or np.trapz` once at
    #      module scope, and spacr/timelapse.py:25-28 prefers
    #      `numpy.trapezoid` with a `numpy.trapz` fallback for numpy 1.x. The
    #      old fallback there went through `scipy.integrate.trapz`, which was
    #      already dead — SciPy removed it in 1.14 and `scipy<2.0` resolves
    #      1.18 — so `import spacr.timelapse` was the one module that failed
    #      under numpy 2. It no longer is.
    #   2. torchcam 0.4.0 AND 0.4.1 — i.e. every release that satisfied its
    #      old pin — declare `numpy<2.0.0`. The pin is spurious: torchcam
    #      touches numpy only in an overlay helper (`np.asarray`/`np.uint8`)
    #      and was verified running GradCAM correctly under numpy 2.4.4. But
    #      pip cannot be argued with, so torchcam has left the core list for
    #      the `attribution` extra. It was already imported lazily
    #      (spacr/attribution.py:580, inside `_torchcam_cam`), so the core
    #      install never needed it.
    #   3. tests/test_diameter_estimator.py:1083 called the `ndarray.ptp()`
    #      *method*, removed in 2.0. It now calls `np.ptp(field)`, matching
    #      the five other sites that already used the function form.
    #
    # The floor stays at 1.26.4 rather than moving up with the ceiling: it is
    # the oldest release with cp312 wheels, it costs nothing on 3.13 (where
    # pip picks 2.3+ regardless), and dropping it would gratuitously break
    # anyone pinned to the 1.x line who is happy there. On 3.10 this resolves
    # numpy 2.2.6 (the last line with cp310 wheels); on 3.13, 2.3+.
    # -----------------------------------------------------------------------
    'numpy>=1.26.4,<3.0',
    # `<3.0` is a real ceiling, not a convention. pandas 3.0 makes
    # Copy-on-Write the only mode, which turns the five chained-inplace calls
    # at spacr/submodules.py:1415,1416,1429,1430,1468 into silent no-ops
    # (reproduced: the inf/NaN cleanup simply does not happen, and
    # ChainedAssignmentError is a Warning, so execution continues with wrong
    # data). It also drops `include_groups` from `groupby().apply`, which
    # makes spacr/plot.py:3064 hand seaborn a frame missing its `x_column`,
    # and forbids `read_html` on a raw string, which is exactly what
    # spacr/sim.py:655 passes.
    'pandas>=2.2.1,<3.0',
    'scipy>=1.12.0,<2.0',
    'cellpose>=4.0,<5.0',
    # `segment-anything` removed: PyPI's `segment-anything` has exactly one
    # release (1.0, 2023-04-06) with empty author, homepage and summary —
    # Meta never published SAM to PyPI. spaCR imports it in zero files, and
    # cellpose already depends on segment_anything itself. It was an
    # unpinned, unattributed name in the supply chain for no benefit.
    # spaCR uses the replacement APIs introduced before 0.27:
    # `footprint_rectangle`, `max_size`, `opening`, `closing` and `dilation`.
    # Compatibility fallbacks retain 0.22-0.24 support without importing the
    # aliases removed in 0.27. Keep the next minor as the explicit audit gate.
    'scikit-image>=0.22.0,<0.28',
    # Floor RAISED from 1.4.1, which admitted a resolve that crashes:
    # spacr/utils.py:6247 calls `TSNE(..., max_iter=1000)`, and `max_iter` did
    # not exist on TSNE before scikit-learn 1.5.0 (it was `n_iter`). On 1.4.x
    # that is `TypeError: TSNE.__init__() got an unexpected keyword argument`.
    # utils.py:6245-6246 already documents the rename; the pin never followed.
    'scikit-learn>=1.5.0,<2.0',
    # Only `sp.posthoc_dunn(val_col=, group_col=, p_adjust=)` is used
    # (spacr/sp_stats.py:184-185, spacr/plot.py:3559). Its signature is
    # byte-identical at 0.10.0, 0.11.4 and 0.14.0, so no ceiling is warranted;
    # `<0.20` names a version that does not exist and never bites.
    'scikit-posthocs>=0.10.0,<0.20',
    # `mahotas` REMOVED from the core list -> the `zernike` extra.
    # Not a numpy-2 problem: 1.4.18 computes Zernike moments correctly under
    # numpy 2.4.4, because its wheels are numpy-2 ABI clean. It is a *wheel*
    # problem. Verified against the PyPI JSON API on 2026-07-27: 1.4.18
    # publishes cp310, cp311 and cp312 and nothing else — no cp313 has ever
    # been published at any version, and the last aarch64 build was 1.4.13. In
    # the core list it therefore does not merely block Python 3.13, it makes
    # `pip install spacr` on 3.13 attempt a C++ source build, which succeeds
    # only where a toolchain happens to exist. Zernike moments are the sole
    # consumer, so the feature moves with the package.
    'btrack>=0.7.0,<1.0',
    'trackpy>=0.6.2,<1.0',
    # The deprecated lowercase ``links.logit`` alias is no longer used:
    # spacr/ml.py imports and constructs ``Logit``. Keep the existing 0.15
    # boundary until the complete statistical suite has been qualified
    # against that release; the warning cleanup itself does not claim broader
    # dependency compatibility. The floor is the release that introduced
    # ``statsmodels.othermod.betareg.BetaModel``, imported by spacr/ml.py.
    'statsmodels>=0.13.0,<0.15',
    # ADDED. `from patsy import dmatrices` at spacr/ml.py:33 is module scope
    # and unguarded, so `import spacr.ml` needs it. It was arriving only
    # because statsmodels declares it; statsmodels 0.15 is expected to finish
    # the move to its own formula engine, and the line above already admits
    # 0.14.x, so this was one upstream release away from breaking.
    'patsy>=0.5.6,<2.0',
    'shap>=0.45.0,<1.0',
    'torch>=2.0,<3.0',
    # PyTorch's official SummaryWriter backend. Vision training writes
    # loss/accuracy/F1/LR events to each run folder for an interactive
    # TensorBoard dashboard; 2.21 is the release exercised by the Python 3.12
    # event-file test, while the major-version ceiling avoids an unreviewed
    # event-file/API break.
    'tensorboard>=2.21,<3.0',
    # Floor RAISED from `>=0.1`, which was false — torchvision 0.1.6 is from
    # 2017. spacr/utils.py:53 imports `ResNet18_Weights ... ResNet152_Weights`
    # from torchvision.models.resnet at module scope (the multi-weight API,
    # torchvision 0.13.0) and spacr/deep_spacr.py:2078 defaults to
    # `model_name='maxvit_t'` (0.14.0). torchvision pins torch exactly, and
    # 0.15.x is the release built against torch 2.0 — so `torch>=2.0` above
    # already implies 0.15. This line now says that instead of implying it.
    'torchvision>=0.15,<1.0',
    # `torch-geometric`, `transformers`, `segmentation_models_pytorch` and
    # `monai` REMOVED: zero imports, zero string references, zero dynamic
    # references. Together they were the four most expensive names on the
    # list — monai alone silently overrode the `torch>=2.0` declared two lines
    # up (1.6.0 requires torch>=2.8.0), transformers pinned
    # `huggingface-hub<1.0` across its entire 4.x line, and torch-geometric
    # and smp each drag their own compiled stack — for four packages spaCR
    # never touches. The only hit any of them produced anywhere in the tree
    # was the word "transformers" inside a GUI description string
    # (spacr/qt/screens/app_screen.py:195), which is prose about CNNs, not an
    # import.
    #
    # `torchcam` REMOVED from the core list -> the `attribution` extra. See
    # blocker 2 in the numpy note above: every release satisfying
    # `>=0.4.0,<1.0` declares `numpy<2.0.0`, so leaving it here pins numpy to
    # 1.26.4 no matter what this file says, and on 3.13 that means a numpy
    # source build. (0.4.1 additionally declares `requires-python >=3.11`, so
    # in the core list it also silently backtracked to 0.4.0 on Python 3.10.)
    'captum>=0.7.0,<1.0',
    'seaborn>=0.13.2,<1.0',
    'matplotlib>=3.8.3,<4.0',
    'matplotlib_venn>=1.1,<2.0',
    'adjustText>=1.2.0,<2.0',
    # KEPT despite zero imports, and deliberately so. Both are pandas'
    # optional acceleration backends, and pandas picks them up by *presence*,
    # never by an import spaCR could write: bottleneck accelerates the
    # nan-aware reductions (`mean`/`sum`/`std`/`median` with skipna, which is
    # every per-object aggregation on the measure path) and numexpr backs
    # `pd.eval`/`DataFrame.query` and large elementwise ops. Declaring them is
    # the only way to make the fast path reproducible rather than a property
    # of whichever machine happens to have them. An import census will always
    # report these as unused; that is the nature of a plugin, not a finding.
    'bottleneck>=1.3.6,<2.0',
    'numexpr>=2.8.4,<3.0',
    # Both bounds are weaker than they look. The real API floor is 4.0.0 —
    # spacr/plot.py:110 and :457 unpack `cv2.findContours` as a 2-tuple, which
    # is the OpenCV 4 signature, and `cv2.SIFT_create` (4.4) is hasattr-guarded
    # at spacr/spacrops.py:204. 4.9.0.80 is a *wheel* floor (first tag cut
    # after cp312 support), not an API one. `<5.0` used to be dead text —
    # opencv-python-headless 4.12+ declares `numpy>=2`, so the old `numpy<2.0`
    # capped the resolve at 4.11.0.86 by itself. With numpy widened, 4.12+ is
    # now genuinely reachable and this ceiling starts doing work. OpenCV 5's
    # Python surface was checked against all 57 cv2 symbols spaCR uses and
    # removes none of them, so the cap is precautionary, not a known break.
    'opencv-python-headless>=4.9.0.80,<5.0',
    # Ceiling RAISED two majors. Everything spaCR uses survives Pillow 11 and
    # 12: `Image.Resampling.*` and the `Image.LANCZOS`-style aliases,
    # `resize(box=, reducing_gap=)`, `ImageOps.exif_transpose`, `ImageTk`,
    # `ImageEnhance.*`, `ImageFont.truetype`, and `PIL.ImageQt` with PySide6.
    # The APIs Pillow 11/12 removed (PSFile, PyAccess, ImageMath.eval,
    # isImageType, IPTC internals) appear in zero spaCR files, and the one
    # genuine risk — `Image.fromarray(mode=)` at spacr/measure.py:2654 and
    # spacr/deep_spacr.py:1472,1483 — is a no-op branch there, because all
    # three pass exactly the array's own typemode ('L', 'L', 'RGB').
    # Verified: 12.2.0 installed, full import of every spaCR module + the
    # attribution/packaging/smoke suites green.
    'pillow>=10.2.0,<13',
    'tifffile>=2023.4.12',
    'nd2reader>=3.3.0, <4.0',
    'czifile',
    'pylibCZIrw>=5.0.0,<6.0',
    # `aicspylibczi` removed: zero import statements and zero raw-string
    # references anywhere under spacr/. It ships a manylinux x86_64 wheel
    # only — no linux-aarch64, no cp313, no cp314 — and its sdist needs CMake
    # plus libCZI headers, so it was the single dependency forcing a C++
    # source build on ARM Linux, for a package spaCR never imports.
    'readlif',
    # KEPT despite zero imports: it is the engine pandas requires for the two
    # `pd.read_excel` calls at spacr/plot.py:4994 and spacr/foreign.py:968.
    # pandas raises ImportError naming openpyxl if it is absent, and
    # spacr/plot.py already carries that message as a string — declaring the
    # package is what stops that message ever being shown.
    'openpyxl>=3.1,<4.0',
    'imageio>=2.34.0,<3.0',
    # Ceiling LOWERED from `<1.0`, which protected nothing. pingouin 0.6.0
    # renamed every dashed result column ("p-val" -> "p_val", "W-val" ->
    # "W_val"), and spaCR indexes the old names directly:
    # spacr/plot.py:3493 `.iloc[0][['T', 'p-val']]` and plot.py:3502
    # `.iloc[0][['W-val', 'p-val']]`. Reproduced against pingouin 0.6.1: both
    # raise KeyError. `<1.0` would have shipped that to every fresh install.
    'pingouin>=0.5.5,<0.6',
    # NOT unused, whatever an import census says. spaCR reaches umap through
    # `umap = _LazyModule('umap.umap_', block_roots=_TF_BACKED_ROOTS)` at
    # spacr/utils.py:197 — the module name is a string literal, so there is no
    # `import umap` anywhere in the tree for a syntax tree to find. The
    # indirection is not incidental: importing umap eagerly costs ~6.5 s and
    # ~1.4 GB *per worker process*, and `umap/__init__.py` pulls
    # `parametric_umap` -> TensorFlow, which spaCR refuses. Three call sites
    # use it: spacr/utils.py:6330 and :7019, spacr/timelapse.py:4157.
    'umap-learn>=0.5.6,<1.0',
    # `ttkthemes` REMOVED: zero imports. The Tk GUI is plain tkinter/ttk and
    # the default GUI is PySide6, which has its own theming.
    # Ceiling RAISED. XGBoost 3.0 removed DeviceQuantileDMatrix, `feval`,
    # datatable support and legacy model *saving*; spaCR uses XGBClassifier
    # (spacr/ml.py:2474, hyperparam.py:1844, gui_elements.py:5461) plus
    # booster-level `xgb.DMatrix`/`xgb.train` (spacr/timelapse.py:7030-7050),
    # all of which survive. The resolver handles the interpreter split by
    # itself: xgboost 3.3 needs Python >=3.12, so a 3.10/3.11 install lands
    # on 3.2 without help.
    'xgboost>=2.0.3,<4',
    # `PyWavelets` REMOVED: zero imports and zero references to `pywt`, its
    # import name. It was pulled in as a scikit-image companion; skimage
    # declares it itself where it needs it.
    # `ttf_opensans` REMOVED: zero imports, and redundant besides — spaCR
    # ships its own Open Sans under spacr/resources/font/ and loads it from
    # there, which is why the font works today on machines that never had this
    # package.
    # `customtkinter` REMOVED: zero imports. Same reason as ttkthemes.
    'biopython>=1.80,<2.0',
    # Ceiling REMOVED. spaCR imports lxml in zero files; it is here only as
    # the preferred backend for the single `pd.read_html` call at
    # spacr/sim.py:655, and pandas itself declares `lxml>=4.9.2` with no upper
    # bound (and falls back to bs4+html5lib anyway). Capping a package we
    # never import at a major we never see was constraining users' resolves
    # for nothing. Verified: lxml 6.1.1 installed, suites green.
    'lxml>=5.1.0',
    # Ceiling RAISED two majors. `>=5.9.8,<6.0` admitted exactly ONE release —
    # it was an `==` pin wearing a range's clothes, and it is why the working
    # dev env still sits on psutil 5.9.8. Nothing spaCR calls changed: psutil
    # 6 altered `disk_partitions()` fields and `process_iter()` PID-reuse
    # checking, and 7 removed `memory_info_ex()`; spaCR uses
    # virtual_memory/cpu_percent/cpu_count/cpu_freq/Process/nice/cpu_affinity
    # and already re-validates each process inside a
    # NoSuchProcess/AccessDenied/ZombieProcess handler (spacr/utils.py:1502).
    # `<8` rather than unbounded: psutil marks `Process.info` — used at
    # utils.py:1508,1518 — for deprecation in 8.0.
    'psutil>=5.9.8,<8',
    'gputil>=1.4.0,<2.0',
    # The import name is ``pynvml``, but the maintained distribution is
    # ``nvidia-ml-py``. The separate distribution named ``pynvml`` is now a
    # deprecated compatibility wrapper; declaring it caused both spaCR and
    # torch.cuda imports to print a FutureWarning at startup. Function-local
    # imports remain guarded and fall back to torch when NVML is unavailable.
    # Pure Python, no wheel constraints; major 14 is not yet qualified.
    'nvidia-ml-py>=11.5,<14',
    # `gpustat` REMOVED: zero imports. GPU state is read through GPUtil,
    # nvidia-ml-py (imported as pynvml) and torch.cuda, all declared above.
    # KEPT despite zero imports: PyTables is what backs `pd.HDFStore` at
    # spacr/sequencing.py:77, which is how annotated_reads.h5 is written. The
    # `comp_type` setting documented in spacr/settings.py:2070 is passed
    # straight through to it as `complib`. Without this package that call is
    # an ImportError, so it is a hard requirement that simply has no import
    # statement — pandas owns the import.
    'tables>=3.8.0,<4.0',
    # `rapidfuzz` REMOVED: zero imports. No fuzzy matching exists in spaCR;
    # the name-matching that does exist is exact or regex.
    # `keyring` REMOVED: zero imports. The docstring at spacr/qt/ai/__init__.py
    # still describes an OS-keyring flow, but no code implements it — all
    # three providers in spacr/qt/ai/providers.py shell out to the vendor CLIs
    # (`claude`, `codex`, `gemini`) and never handle an API key. Shipping a
    # credential-storage package for a code path that does not exist is
    # strictly worse than not shipping it.
    'screeninfo>=0.8.1,<1.0',
    # KEPT despite zero imports: cellpose imports fastremap directly (as does
    # fill_voids underneath it), and spaCR's cellpose floor is `>=4.0`. It is
    # declared here so a cellpose that ever stops declaring it does not
    # silently break mask relabelling.
    'fastremap>=1.14.1',
    # `pytz` REMOVED: zero imports. All timestamp handling is stdlib
    # `datetime`; pandas brings its own tz support.
    # KEPT despite zero imports: spaCR *configures* tqdm rather than calling
    # it — spacr/cli.py:1069 sets `TQDM_DISABLE=1` when stdout is not a tty,
    # to stop the progress bars of cellpose, captum, btrack, shap and
    # huggingface-hub from filling a redirected log with carriage returns.
    # That setting is only meaningful if tqdm is present, which is what this
    # line guarantees.
    'tqdm>=4.65.0',
    # ADDED. Module scope and unguarded at spacr/utils.py:2 and
    # spacr/gui_utils.py:1, so `import spacr.utils` — i.e. importing almost
    # anything — already required it. It was arriving only because
    # huggingface-hub declares it, which is precisely the wrong package to
    # rely on: hub 1.x replaces requests with httpx. `<3.0` because the
    # `requests.HTTPError` / `requests.Timeout` names caught at
    # spacr/utils.py:7865 are the whole API surface used.
    'requests>=2.28,<3.0',
    # ADDED. `from joblib import Parallel, delayed` at module scope
    # (spacr/utils.py:207) and function scope (spacr/object.py:49). It was
    # arriving via scikit-learn. `>=1.2` is the release that fixed the
    # pre-1.2 pickle deserialisation issue; nothing above it is used.
    'joblib>=1.2,<2.0',
    # ADDED. `from natsort import natsorted` at spacr/submodules.py:36,
    # module scope. It was arriving via cellpose.
    'natsort>=8.0,<9.0',
    # ADDED. `from sympy import root` at spacr/gui_elements.py:26, module
    # scope. It was arriving via torch, which declares it for TorchDynamo —
    # a dependency that has been proposed for removal upstream more than once.
    'sympy>=1.12,<2.0',
    # `wandb` REMOVED: zero imports. No experiment tracking is wired up; the
    # run journal (spacr/run_journal.py) is spaCR's own.
    # `openai` REMOVED: zero imports. The two textual hits are an npm
    # one-liner (`npm install -g @openai/codex`) and a provider-name match on
    # the string "openai" — the AI Console shells out to the `codex` CLI and
    # never constructs an SDK client.
    # `gdown` REMOVED: zero imports. Downloads go through requests
    # (spacr/model_zoo.py:1290) and huggingface_hub.
    # Ceiling RAISED. Every spaCR import is `from IPython.display import
    # display` (plus HTML and Image in two files); all three still exist in
    # IPython 9.x, which removed only shim modules and pre-8.16 deprecations.
    # The cap was also redundant with the resolver: IPython 9 requires Python
    # >=3.11, so a 3.10 install resolves 8.x on its own. Verified: IPython
    # 9.15.0 installed, suites green.
    'IPython>=8.18.1,<10',
    # `ipykernel` REMOVED: zero imports. Notebooks under Notebooks/ are run by
    # whatever Jupyter the user already has; spaCR is not a kernel provider,
    # and forcing a kernel into every headless cluster install bought nothing.
    'ipywidgets>=8.1.2,<9.0',
    # `brokenaxes` REMOVED: zero imports. No broken-axis plot exists in
    # spacr/plot.py or anywhere else.
    # spacr only calls huggingface_hub.list_repo_files() — verified
    # signature-identical at 0.25 and at 1.24.0, so the API is not what holds
    # this cap down. ONE thing now does, where there used to be two: the
    # transformers half of this note is resolved (transformers is gone, so it
    # no longer pins `huggingface-hub<1.0` across its whole 4.x line), but hub
    # 1.x replaced requests with httpx and `HfHubHttpError` now subclasses
    # `httpx.HTTPError`. spacr/utils.py:7865 and its twin in
    # spacr/gui_utils.py:1007 wrap list_repo_files in
    # `except (requests.HTTPError, requests.Timeout)`, so on hub 1.x a network
    # failure escapes the retry loop uncaught and reaches the user as a
    # traceback. Widening this to `<2.0` is now a two-line source change in
    # those two files, and nothing else.
    'huggingface-hub>=0.25,<1.0',
    # Ceiling REMOVED. spaCR imports protobuf in zero files — it is purely
    # transitive — and this was the single tightest protobuf constraint in the
    # entire resolved graph: google-api-core allows `<8.0.0`, onnxruntime and
    # shap set no ceiling at all. spaCR alone was holding every user's
    # environment at protobuf 5.x for a package it never touches. (The
    # reverse-dependency check was re-run after `wandb` was removed above:
    # shap and onnxruntime still require protobuf, so the floor is still
    # buying a real thing.)
    'protobuf>=5.28.3',
    # ADDED, Windows only. `from win10toast import ToastNotifier` at
    # spacr/qt/notify.py:56 sits inside `if system == "Windows"` and inside a
    # try/except, so the marker is not a convenience — installing it anywhere
    # else would ship a package that can never execute (it imports pywin32).
    # Linux uses `notify-send` and macOS uses `osascript`, both subprocesses,
    # which is why neither has a dependency here.
    'win10toast>=0.9; platform_system == "Windows"',
    #'tensorflow>=2.20.0,<3.0',
    #'stardist>=0.9,<1.0'
]

VERSION = "1.4.9.8"
# The distribution is `spacr` (not `spacr-nightly`) so that
# `pip install -e .` from a working copy replaces any prior PyPI
# `spacr` install instead of coexisting with it — the coexistence
# was the source of stale-metadata warnings after the branch
# rename. The `spacr-nightly` name lives on as a CLI entry-point
# alias below so users still have `spacr-nightly` on their PATH.
name = "spacr"

setup(
    # name/authors/urls/classifiers/requires-python are declared statically in
    # pyproject.toml [project]; name is repeated here only because stdeb and
    # `python setup.py egg_info` want a non-empty distribution before the
    # pyproject config is applied.
    name=name,
    version=VERSION,
    description="Spatial phenotype analysis of crisp screens (SpaCr)",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    package_data={'spacr': ['resources/data/*', 'resources/models/cp', 'resources/icons/*.png', 'resources/icons/loading_spinner.gif', 'resources/font/**/*', 'resources/images/*', 'resources/themes/*.jpg'],},
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'mask=spacr.app_mask:start_mask_app',
            'measure=spacr.app_measure:start_measure_app',
            'make_masks=spacr.app_make_masks:gui_make_masks',
            'annotate=spacr.app_annotate:start_annotate_app',
            'classify=spacr.app_classify:start_classify_app',
            # 'sim=spacr.app_sim:gui_sim' removed: spacr/app_sim.py does not
            # exist, so the installed `sim` command died with ImportError.
            # Simulations run headless via `spacr-run simulation`.
            # New Qt GUI is the default
            'spacr=spacr.qt:run',
            'spacr-qt=spacr.qt:run',
            'spacr-nightly=spacr.qt:run',
            'spacr-tutorial=spacr.qt.tutorial.__main__:main',
            # spacr-repro <run-folder> — replay a recorded run journal
            'spacr-repro=spacr.cli_repro:main',
            # spacr-run <module> --settings f — headless pipeline runner for
            # clusters: no Qt, no Tk, no display. Importing spacr.cli pulls
            # neither torch nor matplotlib, so --help/--list answer instantly.
            'spacr-run=spacr.cli:main',
            # Classic Tk GUI remains available under new names
            'spacr-tk=spacr.gui:gui_app',
            'spacr-legacy=spacr.gui:gui_app',
            'spaceout=spacr.gui:gui_app',
        ],
    },
    extras_require={
        # `tomli` only on 3.10: tests/test_packaging_metadata.py parses
        # pyproject.toml, and tomllib is stdlib from 3.11. The test degrades
        # to a narrow regex without it, so this is about keeping the strong
        # check on the oldest supported interpreter, not about being able to
        # run the suite at all.
        'dev': ['pytest>=8.0,<9', 'pytest-qt>=4.4,<5',
                'tomli>=2.0; python_version < "3.11"'],
        # Pinned identically to the core dependency. Unpinned, this extra
        # silently widened the core `<5.0` cap to "any opencv", so
        # `pip install spacr[headless]` could resolve a different opencv
        # than `pip install spacr`.
        'headless': ['opencv-python-headless>=4.9.0.80,<5.0'],
        # `pip install spacr[trackastra]` — transformer-based object tracking
        # (timelapse_mode='trackastra'). Optional because it pulls its own
        # pretrained weights on first use; trackpy/btrack/iou stay available
        # without it. BSD-3, PyTorch-only, no TensorFlow.
        'trackastra': ['trackastra>=0.5,<1.0'],
        # `pip install spacr[ultrack]` — global-optimisation object tracking
        # (timelapse_mode='ultrack'). Kept alongside trackastra rather than
        # replacing it: ultrack solves segmentation and linking as one integer
        # program, which wins on densely packed and 3D data, while trackastra
        # is the better zero-config generalist. Optional because it brings an
        # ILP solver and a database backend. BSD-3, no TensorFlow.
        #
        # Not installable on Python 3.13: every ultrack release from 0.1.0
        # through 0.7.2 declares `requires-python >=3.9,<3.13` (checked
        # 2026-07-27). pip refuses cleanly, naming the Python version, so a
        # 3.13 user gets a sentence rather than a traceback — but it is why
        # `spacr[all]` cannot be installed on 3.13 either. Use
        # `timelapse_mode='trackastra'` there; trackastra has no such ceiling.
        'ultrack': ['ultrack>=0.6,<1.0'],
        # `pip install spacr[attribution]` — the five torchcam CAM variants
        # (gradcam, gradcam_pp, scorecam, xgradcam, layercam) in
        # spacr/attribution.py. Optional for exactly one reason, and it is a
        # metadata reason rather than a size one: torchcam 0.4.0 and 0.4.1 both
        # declare `numpy<2.0.0`, which is spurious — torchcam touches numpy
        # only in an overlay helper, and GradCAM was verified running under
        # numpy 2.4.4 — but a declared pin is a declared pin, and in the core
        # list it drags every install back to numpy 1.26.4, whose wheels stop
        # at cp312. So: torchcam here, numpy 2 and Python 3.13 there.
        # `spacr/attribution.py:580` already imports it lazily inside
        # `_torchcam_cam`, so nothing else moves. The captum backends
        # (integrated gradients, occlusion, ...) and spaCR's own `smoothgrad`
        # stay available without this extra; captum is a core dependency.
        #
        # KNOWN LIMITATION on Python 3.13, measured rather than inferred, and
        # it has TWO shapes depending on how pip is invoked. Both were
        # reproduced in a throwaway CPython 3.13.14 env on 2026-07-27:
        #
        #   * `pip install --only-binary :all: spacr[attribution]` fails fast:
        #         ERROR: ResolutionImpossible
        #         torchcam 0.4.1 depends on numpy<2.0.0 and >=1.17.2
        #         ... no matching distributions available for your
        #             environment: numpy
        #   * a PLAIN `pip install spacr[attribution]` is worse. pip is happy
        #     to backtrack and answers "Would install ... numpy-1.26.4
        #     torchcam-0.4.1", i.e. it drops the user into a source build of
        #     numpy 1.26.4 — the exact failure this whole change set exists to
        #     remove, and it also silently downgrades opencv to 4.11.0.86.
        #
        # That is torchcam's pin to fix, not spaCR's, and it is why the extra
        # exists rather than a wider numpy cap: the alternative was holding
        # every spaCR user at numpy 1.26.4 and Python 3.12 to keep five CAM
        # variants installable. On 3.13 use `spacr[qt,zernike]` and the captum
        # backends. Upstream: frgfm/torch-cam.
        'attribution': ['torchcam>=0.4.0,<1.0'],
        # `pip install spacr[boosting]` — the two gradient-boosting backends
        # reachable through `model_type='lightgbm'` and `model_type='catboost'`
        # (spacr/ml.py:2477,2483 and spacr/hyperparam.py:1855,1866). Both are
        # already imported inside the `elif` that selects them and both already
        # raise an actionable ImportError naming the package, so an extra is
        # what that guard was always describing. Not core because catboost
        # alone is a ~100 MB wheel for a backend most runs never select, and
        # because xgboost and scikit-learn's HistGradientBoosting — the two
        # defaults — are core already.
        'boosting': ['catboost>=1.2,<2.0', 'lightgbm>=4.0,<5.0'],
        # `pip install spacr[umap]` — declared so the command spaCR already
        # prints is true. `spacr/hyperparam.py:80-85` (UMAP_MISSING_MESSAGE)
        # tells the user to run `pip install umap-learn` "or `pip install
        # spacr[umap]`", and until now the second half named an extra that did
        # not exist. umap-learn is also a core dependency, pinned identically,
        # so this extra is a no-op for anyone who has spaCR installed at all —
        # it exists to stop a printed instruction being a lie.
        'umap': ['umap-learn>=0.5.6,<1.0'],
        'full': ['opencv-python'],
        'qt': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
        ],
        # `spacr-tutorial` — renders narrated MP4 tutorials for every
        # module. ffmpeg is required at runtime (system package) and a
        # Piper voice model is fetched on first run.
        'tutorial': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
            'piper-tts>=1.2,<2',
        ],
        # The AI Console shells out to vendor coding-agent CLIs
        # (`claude`, `codex`, `gemini`) so authentication piggy-backs
        # on the user's chat subscription — no Python API SDKs needed.
        # Users install whichever CLI(s) they want separately; see the
        # Providers… dialog in the AI Console for one-liners.

        # ------------------------------------------------------------------
        # File-format extras.
        #
        # These name the readers for the vendor microscope formats. The three
        # CZI/ND2/LIF readers are *still* listed in `dependencies` above as
        # well, because spacr/io.py imports pylibCZIrw, czifile and nd2reader
        # at module scope — removing them from the core install would turn
        # `import spacr.io` into an ImportError on every platform, not just
        # the thin ones. The core copies come out the moment those imports
        # become lazy/guarded; the diffs live in files this change does not
        # own. `zernike` is different: mahotas has now LEFT the core list, so
        # that extra is the only place it is declared.
        #
        # Which of these actually gate the platform matrix (re-verified
        # against the PyPI JSON API on 2026-07-27, not assumed):
        #   * mahotas 1.4.18 — cp310, cp311, cp312 and nothing else, with
        #     manylinux x86_64 Linux wheels ONLY. No cp313 or cp314 has ever
        #     been published at any version, and the last aarch64 build was
        #     1.4.13. This is the pin that used to cap Python at 3.12 next to
        #     numpy and force a C++ build on ARM Linux; moving it here is what
        #     lets `requires-python` reach <3.14.
        #   * pylibCZIrw — NOT a 3.13 blocker, contrary to earlier analysis.
        #     5.1.1 (inside the `<6.0` cap already declared above) ships
        #     cp39-cp313 including manylinux aarch64 and macosx arm64. It is
        #     only a 3.14 blocker: no cp314 exists even in 6.1.0.
        #   * czifile, readlif, nd2reader — all pure `py3-none-any`. They
        #     constrain no platform and no Python version; these extras are
        #     organisational, not load-bearing.
        # ------------------------------------------------------------------
        'czi': ['pylibCZIrw>=5.0.0,<6.0', 'czifile'],
        'nd2': ['nd2reader>=3.3.0,<4.0'],
        'lif': ['readlif'],
        'zernike': ['mahotas>=1.4.13,<2.0'],

        # `pip install spacr[all]` — every optional feature at once, minus
        # four, each for a stated reason:
        #   * `dev`   — test tooling, not a feature.
        #   * `full`  — the GUI-capable opencv build, which would shadow the
        #               headless one already in the core deps.
        #   * `umap`  — a pure alias for a core dependency; aggregating it
        #               would add nothing and only make the union harder to
        #               read.
        #   * `attribution` — DELIBERATE, and the one that is not obvious.
        #               torchcam declares `numpy<2.0.0`, which has no cp313
        #               wheel. Including it here would mean that on Python
        #               3.13 `pip install spacr[all]` backtracks into a SOURCE
        #               BUILD of numpy 1.26.4 (measured: pip reports "Would
        #               install ... numpy-1.26.4 torchcam-0.4.1" and silently
        #               downgrades opencv to 4.11.0.86) — the exact failure
        #               this change set out to remove, reintroduced through
        #               the extra most likely to be typed by someone who just
        #               wants everything. `spacr[all,attribution]` remains
        #               available on 3.10-3.12 for anyone who wants both.
        #
        # Spelled out as concrete requirements rather than a recursive
        # `spacr[qt,tutorial,...]` self-reference so it resolves identically
        # on old pip, under `pip download`, and in uv/conda resolvers.
        # tests/test_packaging_metadata.py asserts this stays the exact union
        # of the extras it claims to aggregate, so it cannot drift.
        #
        # ---------------------------------------------------------------
        # `spacr[all]` DOES NOT INSTALL ON PYTHON 3.13, and the reason is not
        # anything above. It is `ultrack`: every release from 0.1.0 through
        # 0.7.2 declares `requires-python >=3.9,<3.13`, so pip on 3.13 says
        #
        #     ERROR: Ignored the following versions that require a different
        #     python version: ... 0.7.2 Requires-Python >=3.9,<3.13
        #     ERROR: No matching distribution found for ultrack<1.0,>=0.6
        #
        # (measured 2026-07-27, CPython 3.13.14). That is upstream's ceiling,
        # not ours, and it is a CLEAN refusal — which is precisely why
        # torchcam is still excluded above even though `all` is already
        # unusable on 3.13: a clean "requires a different python version" is a
        # message a user can act on, and a numpy source build is not. When
        # ultrack ships a 3.13 release this note comes out and `all` works
        # again with no other change.
        #
        # On 3.13 use `spacr[qt,tutorial,trackastra,boosting,czi,nd2,lif,zernike]`,
        # which is `all` minus ultrack, or simply `spacr[qt,zernike]`.
        #
        # `spacr[all]` also pulls mahotas, which has no cp313 wheel and builds
        # from sdist. That one is deliberate and fine — asking for
        # *everything* may reasonably require a toolchain, and unlike torchcam
        # it succeeds where one exists (verified: mahotas 1.4.18 built and ran
        # against numpy 2.4.4 on 3.13). A plain `pip install spacr` never
        # needs a compiler, which is the whole point of the split.
        # ---------------------------------------------------------------
        'all': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
            'piper-tts>=1.2,<2',
            'trackastra>=0.5,<1.0',
            'ultrack>=0.6,<1.0',
            'catboost>=1.2,<2.0',
            'lightgbm>=4.0,<5.0',
            'pylibCZIrw>=5.0.0,<6.0',
            'czifile',
            'nd2reader>=3.3.0,<4.0',
            'readlif',
            'mahotas>=1.4.13,<2.0',
        ],
    },
)
