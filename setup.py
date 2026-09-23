"""spaCR's packaging: the dependency ranges and the reasons behind them.

The longer notes behind individual bounds are in ``docs/notes/setup.md``.
Two of them stay here, beside the list they govern, because they are what
the next person editing a bound has to see.

WHAT "THE OPTIMAL DEPENDENCY VERSIONS" MEANS. The phrase has two readings
that imply opposite CI jobs -- the oldest set that passes, or the newest set
that passes -- and a list edited under both readings at once drifts in both
directions. spaCR means the NEWEST.

WHAT A USER GETS is the newest. ``pip install spacr`` resolves the top of
every range below, so the upper end of each range is the version spaCR
recommends. The ``Fast / Full suite control`` job in
.github/workflows/tests.yml installs with no constraints file and runs the
suite against whatever pip picks that day.

WHAT A USER IS PROMISED is the floor. A lower bound is for an environment
pinned by something else, and it is worthless unless something installs it,
so the ``Minimum dependencies`` job in the same workflow installs
.github/constraints/minimum-py39.txt on CPython 3.9 and runs the same suite
there. A lower bound is therefore the oldest version CI actually installs
and tests, never a guess. An upper bound is a major version spaCR has not
seen, raised deliberately after testing.

INTEL MAC HAS A CEILING on numba and llvmlite, and it is a fact about wheels
rather than about spaCR: llvmlite 0.46+ publishes no macOS x86_64 wheel, and
numba 0.63+ requires that unavailable line. Without the ceiling pip selects
the newest of each, finds no wheel, falls back to a source build and stops
on a missing ``cmake``. ``install_spacr_unix.sh`` applies the identical pair;
it is declared here too so a developer install from git gets wheels as well.
"""
import sys
import re

from setuptools import setup, find_packages

def pypi_readme(source):
    """Preserve the GitHub README while resolving its repository-relative URLs.

    PyPI serves package metadata without the repository's files. Use the
    public nightly tree, which also carries images not yet promoted to main.
    No network access or spaCR imports are needed during isolated builds.
    """
    def absolute(value, image=False):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", value) or value.startswith(("//", "#")):
            return value
        base = ("https://raw.githubusercontent.com/EinarOlafsson/spacr/nightly/"
                if image else "https://github.com/EinarOlafsson/spacr/blob/nightly/")
        return base + value.removeprefix("./")

    source = re.sub(
        r"(?m)^(\s*\.\. (?:\|[^|]+\| )?(?:image|figure)::\s+)(\S+)",
        lambda match: match[1] + absolute(match[2], image=True), source)
    source = re.sub(
        r"(?m)^(\s*:target:\s+)(\S+)",
        lambda match: match[1] + absolute(match[2]), source)
    return re.sub(r"<([^<>\s]+)>(?=`_)",
                  lambda match: "<" + absolute(match[1]) + ">", source)


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = pypi_readme(fh.read())

dependencies = [
    'numpy>=1.26.4,<3.0',
    'pandas>=2.2.1,<4.0',
    'scipy>=1.12.0,<2.0',
    'cellpose>=4.0.7,<5.0',
    'scikit-image>=0.22.0,<0.28',
    'scikit-learn>=1.5.0,<2.0',
    'scikit-posthocs>=0.10.0,<0.20',
    'trackpy>=0.6.2,<1.0',
    'statsmodels>=0.14.0,<0.15',
    'patsy>=0.5.6,<2.0',
    'shap>=0.47.0,<1.0',
    'torch>=2.0,<3.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3.13"',
    'pyfixest>=0.40.1,<1; python_version >= "3.10"',
    'glum>=3.1.2,<4; python_version >= "3.10"',
    'gpytorch>=1.11,<2; python_version >= "3.10"',
    'tensorboard>=2.21,<3.0',
    'numba>=0.60,<1.0; sys_platform != "darwin" or platform_machine != "x86_64"',
    'llvmlite>=0.43,<1.0; sys_platform != "darwin" or platform_machine != "x86_64"',
    'numba>=0.60,<0.63; sys_platform == "darwin" and platform_machine == "x86_64"',
    'llvmlite>=0.43,<0.46; sys_platform == "darwin" and platform_machine == "x86_64"',
    'torchvision>=0.15,<1.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3.13"',
    'captum>=0.7.0,<1.0',
    'seaborn>=0.13.2,<1.0',
    'matplotlib>=3.8.3,<4.0',
    'matplotlib_venn>=1.1,<2.0',
    'pypdf>=6.16.1,<7.0',
    'cycler>=0.10,<1',
    'PySide6>=6.6,<7',
    'qtawesome>=1.3,<2',
    'pyqtgraph>=0.13.3,<1',
    'mpmath>=1.3,<2',
    'vispy>=0.14,<1.0',
    'win10toast>=0.9; platform_system == "Windows"',
    'adjustText>=1.2.0,<2.0',
    'bottleneck>=1.3.6,<2.0',
    'numexpr>=2.8.4,<3.0',
    'opencv-python-headless>=4.9.0.80,<5.0',
    'pillow>=10.2.0,<13',
    'tifffile>=2023.4.12',
    'nd2reader>=3.3.0, <4.0',
    'czifile',
    'readlif',
    'openpyxl>=3.1,<4.0',
    'imageio>=2.34.0,<3.0',
    'umap-learn>=0.5.11,<1.0',
    'xgboost>=2.0.3,<4',
    'biopython>=1.80,<2.0',
    'lxml>=5.1.0',
    'psutil>=5.9.8,<8',
    'gputil>=1.4.0,<2.0',
    'nvidia-ml-py>=11.450.51,<14',
    'tables>=3.8.0,<4.0',
    'fastremap>=1.14.1',
    'tqdm>=4.65.0',
    'requests>=2.28,<3.0',
    'joblib>=1.2,<2.0',
    'natsort>=8.0,<9.0',
    'IPython>=8.18.1,<10',
    'ipywidgets>=8.1.2,<9.0',
    'huggingface-hub>=0.25,<1.0',
    'protobuf>=5.28.3',
]

VERSION = "1.5.0.9"
name = "spacr"

setup(
    name=name,
    version=VERSION,
    description="Spatial phenotype analysis of CRISPR screens (spaCR)",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    package_data={'spacr': ['resources/release_notes.json', 'resources/layout_policy.json', 'resources/tutorial_index.json', 'resources/module_workflows.json', 'resources/data/*', 'resources/models/cp', 'resources/icons/*.png', 'resources/icons/organism_*.svg', 'resources/icons/loading_spinner.gif', 'resources/font/**/*', 'resources/images/*', 'resources/themes/*.jpg', 'resources/setting_animations/*.json', 'resources/plate_templates/*.json', 'resources/setting_animations/gifs/*.gif'],},
    data_files=(
        [
            ('share/applications', [
                'packaging/linux/io.github.olafssonlab.spacr.desktop']),
        ] + [
            (f'share/icons/hicolor/{size}x{size}/apps', [
                f'packaging/linux/icons/hicolor/{size}x{size}/apps/'
                'io.github.olafssonlab.spacr.png'])
            for size in (16, 32, 48, 64, 128, 256, 512)
        ]
    ) if sys.platform.startswith('linux') else [],
    install_requires=dependencies,
    entry_points={
        'console_scripts': [
            'spacr=spacr.qt:run',
            'spacr-qt=spacr.qt:run',
            'spacr-nightly=spacr.qt:run',
            'spaceout=spacr.qt.spaceout:main',
            'safespacr=spacr.qt.safespacr:main',
            'spacr-server=spacr.qt:run_without_setup',
            'spacr-tutorial=spacr.qt.tutorial.__main__:main',
            'spacr-repro=spacr.cli_repro:main',
            'spacr-workspace=spacr.cli_workspace:main',
            'spacr-make-masks=spacr.cli_make_masks:main',
            'spacr-run=spacr.cli:main',
            'spacr-download=spacr.cli_download:main',
            'spacr-remote=spacr.cli_remote:main',
            'spacr-plugins=spacr.cli_plugins:main',
            'spacr-leakage=spacr.cli_leakage:main',
            'spacr-db-audit=spacr.cli_database:main',
            'spacr-doctor=spacr.doctor:main',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=8.0,<9',
            'pytest-qt>=4.4,<5',
            'pytest-xdist>=3.6.1,<4',
            'tomli>=2.0; python_version < "3.11"',
            'hypothesis>=6.100,<7',
            'docutils>=0.20.1,<0.24',
            'pyarrow>=14.0.2,<26',
            'ruff>=0.9,<1',
            'mypy>=1.11,<2',
            'xenon>=0.9,<1',
            'pingouin>=0.5.5,<2.0',
        ],
        'headless': ['opencv-python-headless>=4.9.0.80,<5.0'],
        'embeddings': ['timm>=0.9,<2.0'],
        'trackastra': [
            'trackastra>=0.5,<1.0; python_version >= "3.10" and '
            '(sys_platform != "darwin" or platform_machine != "x86_64" '
            'or python_version < "3.13")',
        ],
        'ultrack': [
            'ultrack>=0.6,<1.0; python_version >= "3.10" and '
            'python_version < "3.14" and (sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.13")',
        ],
        'attribution': ['torchcam>=0.4.0,<1.0'],
        'rapids': [
            'cuml-cu12>=25.2; python_version >= "3.11" and python_version < "3.13"',
            'cupy-cuda12x>=13.0; python_version >= "3.11" and python_version < "3.13"',
        ],
        'intel-gpu': [
            'intel-extension-for-pytorch>=2.1; platform_system != "Darwin"',
        ],
        'directml': [
            'torch-directml>=0.2; platform_system == "Windows"',
        ],
        'boosting': ['catboost>=1.2,<2.0', 'lightgbm>=4.0,<5.0'],
        'plaque': ['ultralytics>=8.0,<9'],
        'papers': ['ultralytics>=8.4,<9', 'rapidocr-onnxruntime>=1.3,<2',
                   'pdfplumber>=0.11,<1'],
        'umap': ['umap-learn>=0.5.11,<1.0'],
        'anndata': ['anndata>=0.10,<0.13'],
        'dinocell': ['dinocell>=0.74,<1.0'],
        'samcell': ['samcell>=1.2,<2.0'],
        'napari': ['napari>=0.5,<1.0'],
        'full': ['opencv-python'],
        'sweep': ['threadpoolctl>=3.0,<4'],
        'qt': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
            'pyqtgraph>=0.13.3,<1',
            'win10toast>=0.9; platform_system == "Windows"',
        ],
        'flowview': ['PySide6>=6.6,<7'],
        'fractal': ['vispy>=0.14,<1.0'],
        'tutorial': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
            'win10toast>=0.9; platform_system == "Windows"',
            'piper-tts>=1.2,<2; sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.14"',
        ],

        'czi': ['pylibCZIrw>=5.0.0,<7.0; python_version < "3.14"',
                'czifile'],
        'nd2': ['nd2reader>=3.3.0,<4.0'],
        'lif': ['readlif'],
        'zernike': ['mahotas>=1.4.13,<2.0; python_version < "3.13"'],
        'btrack': ['btrack>=0.7.0,<1.0'],

        'numpyro': ['numpyro>=0.13,<1.0', 'jax>=0.4,<1.0'],
        'pymc': ['pymc>=5.10,<6.0'],
        'pyfixest': ['pyfixest>=0.40.1,<1; python_version >= "3.10"'],
        'glum': ['glum>=3.1.2,<4; python_version >= "3.10"'],
        'gpytorch': ['gpytorch>=1.11,<2; python_version >= "3.10"'],

        'zarr': ['zarr>=2.16,<4', 'numcodecs>=0.12,<1'],
        'omero': ['omero-py>=5.17,<6'],

        'all': [
            'PySide6>=6.6,<7',
            'qtawesome>=1.3,<2',
            'pyqtgraph>=0.13.3,<1',
            'vispy>=0.14,<1.0',
            'win10toast>=0.9; platform_system == "Windows"',
            'piper-tts>=1.2,<2; sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.14"',
            'trackastra>=0.5,<1.0; python_version >= "3.10" and '
            '(sys_platform != "darwin" or platform_machine != "x86_64" '
            'or python_version < "3.13")',
            'ultrack>=0.6,<1.0; python_version >= "3.10" and '
            'python_version < "3.14" and (sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.13")',
            'catboost>=1.2,<2.0',
            'lightgbm>=4.0,<5.0',
            'ultralytics>=8.0,<9',
            'pylibCZIrw>=5.0.0,<7.0; python_version < "3.14"',
            'czifile',
            'nd2reader>=3.3.0,<4.0',
            'readlif',
            'mahotas>=1.4.13,<2.0; python_version < "3.13"',
            'btrack>=0.7.0,<1.0',
            'anndata>=0.10,<0.13',
        ],
    },
)
