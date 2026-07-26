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

dependencies = [
    'importlib-metadata>=3.6,<10.0',
    'numpy>=1.26.4,<2.0',
    'pandas>=2.2.1,<3.0',
    'scipy>=1.12.0,<2.0',
    'cellpose>=4.0,<5.0',
    # `segment-anything` removed: PyPI's `segment-anything` has exactly one
    # release (1.0, 2023-04-06) with empty author, homepage and summary —
    # Meta never published SAM to PyPI. spaCR imports it in zero files, and
    # cellpose already depends on segment_anything itself. It was an
    # unpinned, unattributed name in the supply chain for no benefit.
    'scikit-image>=0.22.0,<1.0',
    'scikit-learn>=1.4.1,<2.0',
    'scikit-posthocs>=0.10.0, <0.20',
    'mahotas>=1.4.13,<2.0',
    'btrack>=0.7.0,<1.0',
    'trackpy>=0.6.2,<1.0',
    'statsmodels>=0.14.1,<1.0',
    'shap>=0.45.0,<1.0',
    'torch>=2.0,<3.0',
    'torchvision>=0.1,<1.0',
    'torch-geometric>=2.5,<3.0',
    'torchcam>=0.4.0,<1.0',
    'transformers>=4.45.2, <5.0',
    'segmentation_models_pytorch>=0.3.3',
    'monai>=1.3.0',
    'captum>=0.7.0, <1.0',
    'seaborn>=0.13.2,<1.0',
    'matplotlib>=3.8.3,<4.0',
    'matplotlib_venn>=1.1,<2.0',
    'adjustText>=1.2.0,<2.0',
    'bottleneck>=1.3.6,<2.0',
    'numexpr>=2.8.4,<3.0',
    'opencv-python-headless>=4.9.0.80,<5.0',
    'pillow>=10.2.0,<11.0',
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
    'pingouin>=0.5.5,<1.0',
    'umap-learn>=0.5.6,<1.0',
    'ttkthemes>=3.2.2,<4.0',
    'xgboost>=2.0.3,<3.0',
    'PyWavelets>=1.6.0,<2.0',
    'ttf_opensans>=2020.10.30',
    'customtkinter>=5.2.2,<6.0', 
    'biopython>=1.80,<2.0',
    'lxml>=5.1.0,<6.0',
    'psutil>=5.9.8, <6.0',
    'gputil>=1.4.0, <2.0', 
    'gpustat>=1.1.1,<2.0',
    'tables>=3.8.0,<4.0',
    'rapidfuzz>=3.9, <4.0',
    'keyring>=15.1, <26.0',
    'screeninfo>=0.8.1,<1.0',
    'fastremap>=1.14.1',
    'pytz>=2023.3.post1',
    'tqdm>=4.65.0',
    'wandb>=0.16.2',
    'openai>=1.50.2, <2.0',
    'gdown',
    'IPython>=8.18.1,<9.0',
    'ipykernel',
    'ipywidgets>=8.1.2,<9.0',
    'brokenaxes>=0.6.2,<1.0',
    # spacr only calls huggingface_hub.list_repo_files() — a stable API
    # across 0.x and 1.x. The primary constraint is transformers
    # (a hard spacr dep) which still pins huggingface-hub<1.0 through
    # its latest 4.57.x release. `datasets>=0.25` (transitive from
    # transformers) resolves inside 0.25..<1.0 fine. Wider `>=1.2` was
    # tempting but it makes `pip install spacr` unresolvable.
    'huggingface-hub>=0.25,<1.0',
    'protobuf>=5.28.3,<6.0'
    #'tensorflow>=2.20.0,<3.0',
    #'stardist>=0.9,<1.0'
]

VERSION = "1.4.9.3"
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
    package_data={'spacr': ['resources/data/*', 'resources/models/cp', 'resources/icons/*', 'resources/font/**/*', 'resources/images/*'],},
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
