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
#     statsmodels, monai). These were the most valuable findings in the audit:
#     `<1.0` on a 0.x package is decorative, because 0.x breaks at the MINOR.
#   * ceilings RAISED, because no break exists above them (pillow, psutil,
#     IPython, lxml, xgboost, protobuf, customtkinter, openai).
#   * floors RAISED, because the old one admitted a resolve that is already
#     broken (scikit-learn, keyring) or was simply false (torchvision).
#
# Several floors are *higher* than the oldest release that would work — e.g.
# nothing spaCR calls needs matplotlib 3.8.3 (the newest API used, indexing
# `matplotlib.colormaps`, is 3.5), scikit-image 0.22 (the newest is
# `regionprops_table(spacing=)`, 0.20), pandas 2.2.1 (the newest is
# `DataFrame.attrs`, 1.0) or scipy 1.12. They are left alone deliberately:
# an over-tight floor only costs something when it blocks a resolution, and
# none of these do on any Python in `requires-python`. Lowering them would buy
# nothing and would admit combinations no spaCR test has ever run.
# ---------------------------------------------------------------------------
dependencies = [
    # Never imported: spaCR reads its own version through the *stdlib*
    # `importlib.metadata` at all five call sites (spacr/version.py:8,
    # spacr/updater.py:105 and :109, spacr/run_journal.py:93,
    # spacr/qt/ai/issue_report.py:232), and stdlib has had it since 3.8. This
    # backport is a removal candidate, not a bounds problem — its cap is
    # already above the latest release.
    'importlib-metadata>=3.6,<10.0',
    # -----------------------------------------------------------------------
    # `<2.0` is the single most expensive bound in this file: numpy 1.26.4 is
    # the ONLY release that satisfies it, its wheels stop at cp312, and that
    # is what forces `requires-python = ">=3.10,<3.13"` in pyproject.toml.
    #
    # It stays, and the reason is now specific rather than assumed. The
    # 2026-07-26 audit built the whole dependency set against numpy 2.4.4 on
    # CPython 3.12 AND 3.13 and found spaCR's own source almost clean — 64 of
    # 65 modules import unchanged, `np.array(..., copy=False)` appears zero
    # times, and NEP-50 promotion was traced to no risky site. Even mahotas
    # 1.4.18, long assumed to be the blocker, imports and computes Zernike
    # moments correctly under numpy 2.4.4 (its wheels are numpy-2 ABI clean;
    # its real limitation is that it has no cp313 wheel and must build from
    # sdist). Exactly three things stand in the way, all small:
    #
    #   1. `np.trapz` was removed in numpy 2.0. spaCR calls it at
    #      spacr/attribution.py:1410 and spacr/utils.py:4930, and
    #      spacr/timelapse.py:22-24 guards it with a `from scipy.integrate
    #      import trapz` fallback that is ALREADY DEAD (scipy removed
    #      `integrate.trapz` in 1.14, and `scipy>=1.12.0,<2.0` happily
    #      resolves 1.18). `import spacr.timelapse` is the one module that
    #      fails under numpy 2 today. Fix: `np.trapezoid`.
    #   2. torchcam 0.4.0 AND 0.4.1 — i.e. every release satisfying the pin
    #      below — declare `numpy<2.0.0`. The pin is spurious: torchcam
    #      touches numpy only in an overlay helper (`np.asarray`/`np.uint8`)
    #      and was verified running GradCAM correctly under numpy 2.4.4. But
    #      pip cannot be argued with, so torchcam has to leave the core deps
    #      (it is imported lazily at spacr/attribution.py:577, so this is a
    #      small change) before numpy can move.
    #   3. tests/test_diameter_estimator.py:1083 calls the `ndarray.ptp()`
    #      *method*, removed in 2.0. `np.ptp(field)` is the fix; the five
    #      other sites already use the function, which survived.
    #
    # With those three done, `numpy>=1.26.4,<3.0` was proven to install and
    # import cleanly on CPython 3.12 and 3.13, which is what unlocks
    # `requires-python` moving to `<3.14`. Do not widen this line before them:
    # on 3.13 a widened numpy with torchcam still in core resolves back to
    # numpy 1.26.4, finds no cp313 wheel, and drops the user into the numpy
    # source build this pin exists to prevent.
    # -----------------------------------------------------------------------
    'numpy>=1.26.4,<2.0',
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
    # Ceiling LOWERED from the decorative `<1.0`. `skimage.morphology.square`,
    # imported at module scope in spacr/utils.py:14 and used at utils.py:1589
    # and utils.py:7139, is deprecated since 0.25 and removed in 0.27
    # (footprints.py: `removed_version="0.27"`, use `footprint_rectangle`).
    # 0.26 still imports it with a DeprecationWarning, so `<0.27` is the last
    # version that works, not the first that warns.
    'scikit-image>=0.22.0,<0.27',
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
    'mahotas>=1.4.13,<2.0',
    'btrack>=0.7.0,<1.0',
    'trackpy>=0.6.2,<1.0',
    # Ceiling LOWERED from the decorative `<1.0`. spacr/ml.py:28 imports the
    # lowercase link alias `from statsmodels.genmod.families.links import
    # logit` at module scope and ml.py:115 *calls* it as a default argument,
    # so its removal is an ImportError at `import spacr.ml`, not a warning.
    # statsmodels 0.14.6 says so itself: "The logit link alias will be removed
    # after the 0.15.0 release." Floor lowered to the release that introduced
    # `statsmodels.othermod.betareg.BetaModel`, which spacr/ml.py imports.
    'statsmodels>=0.13.0,<0.15',
    'shap>=0.45.0,<1.0',
    'torch>=2.0,<3.0',
    # Floor RAISED from `>=0.1`, which was false — torchvision 0.1.6 is from
    # 2017. spacr/utils.py:53 imports `ResNet18_Weights ... ResNet152_Weights`
    # from torchvision.models.resnet at module scope (the multi-weight API,
    # torchvision 0.13.0) and spacr/deep_spacr.py:2078 defaults to
    # `model_name='maxvit_t'` (0.14.0). torchvision pins torch exactly, and
    # 0.15.x is the release built against torch 2.0 — so `torch>=2.0` above
    # already implies 0.15. This line now says that instead of implying it.
    'torchvision>=0.15,<1.0',
    'torch-geometric>=2.5,<3.0',
    # `>=0.4.0,<1.0` is honest about torchcam itself — GradCAM, GradCAMpp,
    # ScoreCAM, XGradCAM and LayerCAM (spacr/attribution.py:557-563) are all
    # present in 0.4.1 and nothing is deprecated. What this line does NOT say
    # is that every release it admits declares `numpy<2.0.0`, which makes
    # torchcam the hard resolver blocker for numpy 2. See the numpy note.
    'torchcam>=0.4.0,<1.0',
    'transformers>=4.45.2,<5.0',
    'segmentation_models_pytorch>=0.3.3',
    # Ceiling ADDED, because the missing one was doing real damage: monai
    # 1.5.1 raised its torch floor to 2.4.1 and 1.6.0 raised it to 2.8.0, so
    # an unbounded `monai>=1.3.0` silently overrides the `torch>=2.0` declared
    # four lines up and drags torchvision to >=0.23 with it — a multi-GB
    # resolver swing for a package spaCR imports in zero files.
    'monai>=1.3.0,<1.6',
    'captum>=0.7.0,<1.0',
    'seaborn>=0.13.2,<1.0',
    'matplotlib>=3.8.3,<4.0',
    'matplotlib_venn>=1.1,<2.0',
    'adjustText>=1.2.0,<2.0',
    'bottleneck>=1.3.6,<2.0',
    'numexpr>=2.8.4,<3.0',
    # Both bounds are weaker than they look. The real API floor is 4.0.0 —
    # spacr/plot.py:110 and :457 unpack `cv2.findContours` as a 2-tuple, which
    # is the OpenCV 4 signature, and `cv2.SIFT_create` (4.4) is hasattr-guarded
    # at spacr/spacrops.py:204. 4.9.0.80 is a *wheel* floor (first tag cut
    # after cp312 support), not an API one. And `<5.0` is currently dead text:
    # opencv-python-headless 4.12+ declares `numpy>=2`, so `numpy<2.0` above
    # already caps the resolve at 4.11.0.86. OpenCV 5's Python surface was
    # checked against all 57 cv2 symbols spaCR uses and removes none of them,
    # so this ceiling can move the day numpy does.
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
    'openpyxl>=3.1,<4.0',
    'imageio>=2.34.0,<3.0',
    # Ceiling LOWERED from `<1.0`, which protected nothing. pingouin 0.6.0
    # renamed every dashed result column ("p-val" -> "p_val", "W-val" ->
    # "W_val"), and spaCR indexes the old names directly:
    # spacr/plot.py:3493 `.iloc[0][['T', 'p-val']]` and plot.py:3502
    # `.iloc[0][['W-val', 'p-val']]`. Reproduced against pingouin 0.6.1: both
    # raise KeyError. `<1.0` would have shipped that to every fresh install.
    'pingouin>=0.5.5,<0.6',
    'umap-learn>=0.5.6,<1.0',
    'ttkthemes>=3.2.2,<4.0',
    # Ceiling RAISED. XGBoost 3.0 removed DeviceQuantileDMatrix, `feval`,
    # datatable support and legacy model *saving*; spaCR uses XGBClassifier
    # (spacr/ml.py:2474, hyperparam.py:1844, gui_elements.py:5461) plus
    # booster-level `xgb.DMatrix`/`xgb.train` (spacr/timelapse.py:7030-7050),
    # all of which survive. The resolver handles the interpreter split by
    # itself: xgboost 3.3 needs Python >=3.12, so a 3.10/3.11 install lands
    # on 3.2 without help.
    'xgboost>=2.0.3,<4',
    'PyWavelets>=1.6.0,<2.0',
    'ttf_opensans>=2020.10.30',
    # Ceiling RAISED. spaCR imports customtkinter in zero files — the Tk GUI
    # is plain tkinter/ttk and the modern GUI is PySide6 — so customtkinter
    # 6.0's changes have no call site here. (The widget-API breaks people
    # remember, `text_font`->`font` and `orient`->`orientation`, were 5.0,
    # already below this floor.)
    'customtkinter>=5.2.2,<7',
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
    'gpustat>=1.1.1,<2.0',
    'tables>=3.8.0,<4.0',
    'rapidfuzz>=3.9,<4.0',
    # Floor RAISED, ceiling left. `>=15.1` was not a floor, it was an
    # accident: keyring 15.1 is from 2018, and it is exactly what the working
    # dev env resolved (keyring 15.2.0 — a release predating the Python 2
    # drop). A dependency that ships credential storage may not be allowed to
    # resolve seven years stale. 25.0 is the current major line; `<27` keeps
    # one major of headroom above the latest (25.7.0).
    'keyring>=25.0,<27',
    'screeninfo>=0.8.1,<1.0',
    'fastremap>=1.14.1',
    'pytz>=2023.3.post1',
    'tqdm>=4.65.0',
    'wandb>=0.16.2',
    # Ceiling RAISED. spaCR imports `openai` in zero files — all three AI
    # providers in spacr/qt/ai/providers.py shell out to vendor CLIs
    # (`claude`, `codex`, `gemini`) rather than construct an SDK client — so
    # there is no call site for openai-python 2.0's one documented break (the
    # widening of two tool-call `output` fields) to reach.
    'openai>=1.50.2,<3',
    'gdown',
    # Ceiling RAISED. Every spaCR import is `from IPython.display import
    # display` (plus HTML and Image in two files); all three still exist in
    # IPython 9.x, which removed only shim modules and pre-8.16 deprecations.
    # The cap was also redundant with the resolver: IPython 9 requires Python
    # >=3.11, so a 3.10 install resolves 8.x on its own. Verified: IPython
    # 9.15.0 installed, suites green.
    'IPython>=8.18.1,<10',
    'ipykernel',
    'ipywidgets>=8.1.2,<9.0',
    'brokenaxes>=0.6.2,<1.0',
    # spacr only calls huggingface_hub.list_repo_files() — verified
    # signature-identical at 0.25 and at 1.24.0, so the API is not what holds
    # this cap down. Two things do, and both are fixable elsewhere:
    #   * transformers, declared above, pins huggingface-hub<1.0 through the
    #     whole 4.x line (its last release is 4.57.6); only transformers 5.0
    #     moves to hub>=1.3. spaCR imports transformers in ZERO files, so
    #     deleting that line is what actually frees this one.
    #   * hub 1.x replaced requests with httpx, and HfHubHttpError now
    #     subclasses httpx.HTTPError. spacr/utils.py:7732 and the twin in
    #     spacr/gui_utils.py wrap list_repo_files in
    #     `except (requests.HTTPError, requests.Timeout)`, so on hub 1.x a
    #     network failure would escape the retry loop uncaught.
    # Widening to `<2.0` is a two-line change away, not a research problem.
    'huggingface-hub>=0.25,<1.0',
    # Ceiling REMOVED. spaCR imports protobuf in zero files — it is purely
    # transitive (wandb, tensorboard, onnxruntime) — and this was the single
    # tightest protobuf constraint in the entire resolved graph: wandb allows
    # `<7`, google-api-core allows `<8.0.0`, tensorboard and onnxruntime set
    # no ceiling at all. spaCR alone was holding every user's environment at
    # protobuf 5.x for a package it never touches.
    'protobuf>=5.28.3'
    #'tensorflow>=2.20.0,<3.0',
    #'stardist>=0.9,<1.0'
]

VERSION = "1.4.9.4"
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
    package_data={'spacr': ['resources/data/*', 'resources/models/cp', 'resources/icons/*', 'resources/font/**/*', 'resources/images/*', 'resources/themes/*.jpg'],},
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
        'ultrack': ['ultrack>=0.6,<1.0'],
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
        # These name the readers for the vendor microscope formats. They are
        # declared now so `pip install spacr[czi]` is stable from today, but
        # each package is *still* listed in `dependencies` above, because
        # spacr/io.py imports pylibCZIrw, czifile and nd2reader at module
        # scope and spacr/measure.py imports mahotas at module scope —
        # removing them from the core install would turn `import spacr.io`
        # into an ImportError on every platform, not just the thin ones. The
        # core copies come out the moment those imports become lazy/guarded;
        # the diffs live in files this change does not own.
        #
        # Which of these actually gate the platform matrix (verified against
        # the PyPI JSON API on 2026-07-26, not assumed):
        #   * mahotas 1.4.18 — the real blocker. cp310-cp312 only, and its
        #     Linux wheels are manylinux x86_64 ONLY. No cp313 or cp314 has
        #     ever been published, and the last aarch64 build was 1.4.13.
        #     This one pin is what caps Python at 3.12 alongside numpy, and
        #     what forces a C++ build on ARM Linux.
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
        # `dev` (test tooling) and `full` (the GUI-capable opencv build,
        # which would shadow the headless one already in the core deps).
        # Spelled out as concrete requirements rather than a recursive
        # `spacr[qt,tutorial,...]` self-reference so it resolves identically
        # on old pip, under `pip download`, and in uv/conda resolvers.
        # tests/test_packaging_metadata.py asserts this stays the exact union
        # of the extras it claims to aggregate, so it cannot drift.
        'all': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
            'piper-tts>=1.2,<2',
            'trackastra>=0.5,<1.0',
            'ultrack>=0.6,<1.0',
            'pylibCZIrw>=5.0.0,<6.0',
            'czifile',
            'nd2reader>=3.3.0,<4.0',
            'readlif',
            'mahotas>=1.4.13,<2.0',
        ],
    },
)
