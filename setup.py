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
import os
import re
from html import unescape

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py as _build_py
from setuptools.command.sdist import sdist as _sdist

from pathlib import Path
import runpy

project_runtime_catalogs = runpy.run_path(
    str(Path(__file__).resolve().parent / "_spacr_catalog_build.py")
)["project_runtime_catalogs"]


class RuntimeCatalogBuild(_build_py):
    """Project copied packages before setuptools byte-compiles them."""

    def build_packages(self):
        """Copy packages normally, then omit development-only catalog fields."""
        super().build_packages()
        project_runtime_catalogs(self.build_lib)


class RuntimeCatalogSdist(_sdist):
    """Keep development provenance in git, outside the release source tree."""

    def make_release_tree(self, base_dir, files):
        """Project only the complete copied source-distribution tree."""
        super().make_release_tree(base_dir, files)
        project_runtime_catalogs(base_dir)


def pypi_readme(source):
    """Preserve the GitHub README while resolving its repository-relative URLs.

    PyPI serves package metadata without the repository's files. By default,
    repository images and links point at the public ``main`` tree, which carries the
    README's images: the image README's panels (docs/readme_images), the
    model cards and the deck are promoted there with every README update,
    and a released description should not move when ``nightly`` does.
    Release builds set ``SPACR_PYPI_README_REF`` to the exact source commit,
    binding repository images and readable pages to that immutable revision.
    Other builds retain the public ``main`` fallback, without requiring Git.
    No network access or spaCR imports are needed during isolated builds.
    GitHub's aligned raw-HTML tile substitutions become ordinary RST images
    here because PyPI's renderer disables raw HTML.

    Each phone/desktop ``<picture>`` row becomes two line blocks, so every
    row is a block of its own as GitHub's ``<div>`` is, never a run of
    inline images beside the row before it. PyPI keeps neither ``<source>``
    nor ``media``, so the desktop drawings carry PyPI's own
    ``hide-below-desktop`` class and the phone drawings ``hide-on-desktop``:
    a desktop browser sees the desktop layout, a phone the phone layout,
    as on GitHub. Each image's width is its share of the column it was
    drawn for (830 px desktop, 358 px phone), so a row of buttons or cards
    fills the column and never wraps when PyPI's column is a little
    narrower than GitHub's. The widths are read from the drawings, which
    sit beside ``setup.py`` in a checkout; a build without them (a wheel
    built from the sdist) reuses the sdist's description, see
    :func:`long_description_for`.
    """
    release_reference = os.environ.get("SPACR_PYPI_README_REF")
    if release_reference is not None and re.fullmatch(r"[0-9a-f]{40}", release_reference) is None:
        raise ValueError("SPACR_PYPI_README_REF must be a full lowercase commit SHA")
    reference = release_reference if release_reference is not None else "main"
    names = set()
    columns = {"desktop": 830, "phone": 358}
    classes = {"desktop": "hide-below-desktop", "phone": "hide-on-desktop"}

    def share(path, layout):
        """``path``'s drawn width as a percentage of its layout's column."""
        from pathlib import Path

        try:
            head = (Path(__file__).resolve().parent / path).read_text(
                encoding="utf-8")[:2000]
        except OSError:
            return None
        match = re.search(r'<svg\b[^>]*?\swidth="([0-9.]+)"', head)
        if match is None:
            return None
        percent = min(100.0, 100.0 * float(match[1]) / columns[layout])
        return f"{int(percent * 1000) / 1000:g}%"

    def pictures(rows):
        """Rows of GitHub ``<picture>`` images as a desktop and a phone
        line block of RST images, each line one row as on GitHub."""
        found = re.compile(
            r'(?:<a href="(?P<target>[^"]*)">)?<picture><source [^>]*?'
            r'srcset="(?P<desktop>[^"]+)"[^>]*>'
            r'<img src="(?P<src>[^"]+)" alt="(?P<alt>[^"]*)"></picture>'
            r'(?:</a>)?')
        blocks, definitions = [], []
        for row in re.findall(r"   <div>(.*)</div>", rows):
            items = list(found.finditer(row))
            for layout in ("desktop", "phone"):
                line = []
                for item in items:
                    image = unescape(item["desktop" if layout == "desktop" else "src"])
                    name = f"pypi-{layout}-" + re.sub(
                        r"\W+", "-", image.rsplit("/", 1)[-1])
                    while name in names:
                        name += "-"
                    names.add(name)
                    line.append(f"|{name}|")
                    lines = [f".. |{name}| image:: {image}",
                             f"   :alt: {unescape(item['alt'])}",
                             f"   :class: {classes[layout]}"]
                    width = share(image, layout)
                    if width:
                        lines.append(f"   :width: {width}")
                    if item["target"]:
                        lines.append(f"   :target: {unescape(item['target'])}")
                    definitions.append("\n".join(lines))
                blocks.append(f".. class:: {classes[layout]}\n\n| "
                              + "\\ ".join(line))
        return "\n\n".join(blocks + definitions) + "\n"

    def absolute(value, image=False):
        for prefix in (
            "https://raw.githubusercontent.com/EinarOlafsson/spacr/",
            "https://github.com/EinarOlafsson/spacr/blob/",
        ):
            if value.startswith(prefix + "main/"):
                return prefix + reference + "/" + value[len(prefix + "main/"):]
        # Warehouse adds a wrapping pseudo-icon to lowercase mailto: links.
        if value.lower().startswith("mailto:"):
            return "MAILTO:" + value[len("mailto:"):]
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", value) or value.startswith(("//", "#")):
            return value
        base = (f"https://raw.githubusercontent.com/EinarOlafsson/spacr/{reference}/"
                if image else f"https://github.com/EinarOlafsson/spacr/blob/{reference}/")
        return base + value.removeprefix("./")

    source = re.sub(
        r'(?m)^\.\. \|(?P<name>[^|\n]+)\| raw:: html\n\n'
        r'   <a href="(?P<target>[^"]+)"><img src="(?P<image>[^"]+)" '
        r'width="(?P<width>[^"]+)" align="middle" alt="(?P<alt>[^"]+)"></a>\n',
        lambda match: (
            f".. |{match['name']}| image:: {unescape(match['image'])}\n"
            f"   :width: {match['width']}\n"
            f"   :alt: {unescape(match['alt'])}\n"
            f"   :target: {unescape(match['target'])}\n"
            "   :align: middle\n"
        ), source,
    )
    source = re.sub(
        r"(?m)^\.\. raw:: html\n\n((?:   <div>[^\n]*</div>(?:\n|$))+)",
        lambda match: pictures(match[1]), source)
    source = space_after_code(source)
    source = re.sub(
        r"(?m)^(\s*\.\. (?:\|[^|]+\| )?(?:image|figure)::\s+)(\S+)",
        lambda match: match[1] + absolute(match[2], image=True), source)
    source = re.sub(
        r"(?m)^(\s*:target:\s+)(\S+)",
        lambda match: match[1] + absolute(match[2]), source)
    return re.sub(r"<([^<>\s]+)>(?=`_)",
                  lambda match: "<" + absolute(match[1]) + ">", source)


def space_after_code(source):
    """Put an empty line block between a code block and the image row after it.

    PyPI sets a code block 30 px below what precedes it and nothing below
    it, so the panel that follows would touch the code; GitHub leaves a gap
    on both sides. One empty line of a line block restores that gap.

    :param source: the converted reStructuredText.
    :returns: the same text with a spacer after each such code block.
    """
    lines, out, in_code = source.split("\n"), [], False
    for line in lines:
        if line.startswith(".. code-block::") or line.startswith(".. code::"):
            in_code = True
        elif in_code and line and not line.startswith(" "):
            if (line.startswith("| |pypi-")
                    or line in {".. class:: hide-below-desktop",
                                ".. class:: hide-on-desktop"}):
                out.extend(["|", ""])
            in_code = False
        out.append(line)
    return "\n".join(out)


def long_description_for(readme, root=None):
    """The PyPI description: the converted README, or the sdist's own.

    A wheel built from the sdist runs this file again in the unpacked sdist,
    which holds ``PKG-INFO`` but not the README's drawings, so their widths
    cannot be read there; the description the sdist already carries is
    reused, and the wheel and the sdist describe the release identically.

    :param readme: the README.rst source text.
    :param root: the folder holding ``setup.py``; this file's own by default.
    :returns: the reStructuredText long description.
    """
    from email import message_from_string
    from pathlib import Path

    here = Path(root) if root is not None else Path(__file__).resolve().parent
    pkg_info = here / "PKG-INFO"
    if not (here / "docs" / "readme_images").is_dir() and pkg_info.is_file():
        metadata = message_from_string(pkg_info.read_text(encoding="utf-8"))
        payload = metadata.get_payload()
        if isinstance(payload, str) and payload.strip():
            return payload
    return pypi_readme(readme)


with open("README.rst", "r", encoding="utf-8") as fh:
    long_description = long_description_for(fh.read())

dependencies = [
    'packaging>=21.3',
    # torch 2.2.2 is the last release with Intel macOS wheels and is built
    # against numpy 1.x: under numpy 2 every torch.from_numpy raises "Numpy is
    # not available". The installer applied this guard only on first install,
    # so an in-app `uv pip install --upgrade spacr` moved numpy to 2.x. Python
    # 3.13 on an Intel Mac gets no torch and no numpy 1.x wheel, so it is left
    # out. opencv 4.12 requires numpy 2 and follows the same split.
    'numpy>=1.26.4,<3.0; python_version < "3.15" and (sys_platform != "darwin" or platform_machine != "x86_64" or python_version >= "3.13")',
    'numpy>=2.5.2,<3.0; python_version >= "3.15"',
    'numpy>=1.26.4,<2.0; sys_platform == "darwin" and platform_machine == "x86_64" and python_version < "3.13"',
    'pandas>=2.2.1,<4.0',
    'scipy>=1.12.0,<2.0',
    'cellpose>=4.0.7,<5.0',
    'scikit-image>=0.22.0,<0.28; python_version < "3.15"',
    'scikit-image>=0.27.0rc1,<0.28; python_version >= "3.15"',
    # Measure's texture confluency calls skimage.restoration.estimate_sigma,
    # which imports PyWavelets internally even though spaCR never imports pywt.
    'PyWavelets>=1.4,<2',
    'scikit-learn>=1.5.0,<2.0',
    'scikit-posthocs>=0.10.0,<0.20',
    'trackpy>=0.6.2,<1.0',
    'statsmodels>=0.14.0,<0.16',
    'patsy>=0.5.6,<2.0',
    'shap>=0.47.0,<1.0',
    'torch>=2.0,<3.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3.13"',
    'pyfixest>=0.40.1,<1; python_version >= "3.10" and python_version < "3.15" and (sys_platform != "darwin" or python_version >= "3.11")',
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
    'PySide6>=6.6,!=6.12.0,<7; sys_platform != "darwin" and python_version < "3.15"',
    'PySide6>=6.12,<7; sys_platform != "darwin" and python_version >= "3.15"',
    'PySide6>=6.6,<6.11; sys_platform == "darwin"',
    'qtawesome>=1.3,<2',
    'pyqtgraph>=0.13.3,<1',
    'mpmath>=1.3,<2',
    'vispy>=0.14,<1.0',
    # VisPy's OpenGL volume renderer needs PyOpenGL for 3D textures. It is pure
    # Python, so one requirement covers every platform and Python.
    'PyOpenGL>=3.1.7,<4',
    # wgpu draws the True 3D view through Vulkan, Metal or Direct3D 12. Its
    # wheels cover Windows x86/x64/arm64, macOS x86_64/arm64 and manylinux_2_28
    # x86_64/aarch64; wgpu 0.32 needs Python 3.11, 0.25-0.31.0 Python 3.10,
    # and Python 3.9 gets none. Without it the view draws with OpenGL (VisPy)
    # or the CPU; `pip uninstall -y wgpu rendercanvas` is the supported way
    # to drop it, and the installers' "Install without wgpu" option does the
    # same. Linux with glibc older than 2.28 cannot install the wheel; the
    # installers detect that and skip it.
    'wgpu>=0.25,<=0.31.0; python_version == "3.10" and (platform_machine == "x86_64" or platform_machine == "AMD64" or platform_machine == "aarch64" or platform_machine == "arm64" or platform_machine == "ARM64")',
    'wgpu>=0.25,<0.33; python_version >= "3.11" and (platform_machine == "x86_64" or platform_machine == "AMD64" or platform_machine == "aarch64" or platform_machine == "arm64" or platform_machine == "ARM64")',
    'win10toast>=0.9; platform_system == "Windows"',
    'adjustText>=1.2.0,<2.0',
    'bottleneck>=1.3.6,<2.0',
    'numexpr>=2.8.4,<3.0',
    'opencv-python-headless>=4.9.0.80,<6.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_version >= "3.13"',
    'opencv-python-headless>=4.9.0.80,<4.12; sys_platform == "darwin" and platform_machine == "x86_64" and python_version < "3.13"',
    'pillow>=10.2.0,<14',
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
    'paramiko>=3.5,<6',
    'joblib>=1.2,<2.0',
    'cloudpickle>=3.1.2,<4',
    'natsort>=8.0,<9.0',
    'IPython>=8.18.1,<10',
    'ipywidgets>=8.1.2,<9.0',
    'huggingface-hub>=0.25,<3',
    'protobuf>=5.28.3',
    # Runtime feature packages are part of normal installs on the qualified
    # Python 3.10+ line; their named extras remain compatible aliases.
    'pyarrow>=14.0.2,<27; python_version >= "3.10"',
    'adlfs>=2023.1; python_version >= "3.10"',
    'aiohttp<4,>=3.8; python_version >= "3.10"',
    'anndata<0.14,>=0.10; python_version >= "3.10"',
    'botocore<2,>=1.29; python_version >= "3.10"',
    # Keep the numeric constant on the left: old pip then safely falls back
    # for non-version Linux kernels instead of parsing them as version values.
    # Darwin releases still use the same numeric deployment boundaries.
    'btrack<1.0,>=0.7.0; python_version >= "3.10" and (sys_platform != "darwin" or python_version < "3.14" or "24" <= platform_release)',
    'catboost<2.0,>=1.2; python_version >= "3.10" and python_version < "3.15"',
    'duckdb<3,>=1.1; python_version >= "3.10"',
    'faiss-cpu<2,>=1.7.4; python_version >= "3.10" and (sys_platform != "darwin" or python_version < "3.14" or "23" <= platform_release)',
    'fsspec>=2023.1; python_version >= "3.10"',
    'gcsfs>=2023.1; python_version >= "3.10"',
    'jax<1.0,>=0.4; python_version >= "3.10"',
    'keyring<27,>=23.0; python_version >= "3.10"',
    'liffile>=2025.1.25; python_version >= "3.10"',
    'lightgbm<5.0,>=4.0; python_version >= "3.10"',
    'mahotas>=1.4.19,<2.0; python_version >= "3.10"',
    'moderngl<6,>=5.12; python_version >= "3.10" and (sys_platform != "darwin" or platform_machine != "x86_64")',
    'moderngl>=5.12,<5.13; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "x86_64"',
    'glcontext>=3.0,<3.1; python_version >= "3.10" and sys_platform == "darwin" and platform_machine == "x86_64"',
    'cryptography<49; sys_platform == "darwin" and platform_machine == "x86_64"',
    'mrcfile<2,>=1.5; python_version >= "3.10"',
    'napari<1.0,>=0.5; python_version >= "3.10"',
    'nd2<1,>=0.10; python_version >= "3.10"',
    # The 0.17 Meson build discovers system NumPy headers through pkg-config;
    # 0.16.6 uses the isolated build's NumPy and builds correctly on 3.15.
    'numcodecs<1,>=0.12; python_version >= "3.10" and python_version < "3.15"',
    'numcodecs>=0.16.6,<0.17; python_version >= "3.15"',
    'numpyro<1.0,>=0.13; python_version >= "3.10"',
    'pdfplumber<1,>=0.11; python_version >= "3.10"',
    'piper-tts<2,>=1.2; python_version < "3.15" and (sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3.14") and python_version >= "3.10" and (sys_platform != "darwin" or python_version < "3.14" or "23" <= platform_release)',
    'psycopg[binary]<4,>=3.1; python_version >= "3.10"',
    'pycromanager<2,>=0.28; python_version >= "3.10"',
    'pylibCZIrw<7.0,>=5.0.0; (python_version < "3.14") and python_version >= "3.10" and (sys_platform != "darwin" or platform_machine != "x86_64")',
    'pymc<7,>=5.10; python_version >= "3.10" and python_version < "3.15"',
    'rdkit>=2023.3; python_version >= "3.10"',
    's3fs>=2023.1; python_version >= "3.10"',
    'samcell<2.0,>=1.2; python_version >= "3.10"',
    'threadpoolctl<4,>=3.0; python_version >= "3.10"',
    'timm<2.0,>=0.9; python_version >= "3.10"',
    'torchcam<1.0,>=0.5; python_version >= "3.11" and (sys_platform != "darwin" or platform_machine != "x86_64")',
    'torchcam>=0.4,<0.5; sys_platform == "darwin" and platform_machine == "x86_64" and python_version < "3.13"',
    'trackastra<1.0,>=0.5; (python_version >= "3.10" and (sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3.13")) and python_version >= "3.10"',
    # Intel macOS is limited to torch 2.2.2; Transformers 5 disables that
    # backend even though pip regards its optional Torch dependency as valid.
    'transformers<6,>=4.40; python_version >= "3.10" and (sys_platform != "darwin" or platform_machine != "x86_64")',
    'transformers>=4.40,<4.58; sys_platform == "darwin" and platform_machine == "x86_64"',
    'ultrack<1.0,>=0.6; (python_version >= "3.10" and python_version < "3.14" and (sys_platform != "darwin" or platform_machine != "x86_64" or python_version < "3.13")) and python_version >= "3.10" and (sys_platform != "linux" or platform_machine == "x86_64" or platform_machine == "AMD64")',
    'zarr<4,>=2.16; python_version >= "3.10"',
]

VERSION = "1.5.1.6"
name = "spacr"

setup(
    cmdclass={"build_py": RuntimeCatalogBuild, "sdist": RuntimeCatalogSdist},
    name=name,
    version=VERSION,
    description="Spatial phenotype analysis of CRISPR screens (spaCR)",
    long_description=long_description,
    long_description_content_type='text/x-rst',
    packages=find_packages(exclude=["tests.*", "tests"]),
    include_package_data=True,
    exclude_package_data={'spacr': ['resources/AI/*', 'resources/AI/**/*']},
    package_data={'spacr': ['resources/release_notes.json', 'resources/layout_policy.json', 'resources/tutorial_index.json', 'resources/module_workflows.json', 'resources/data/*', 'resources/model_cards/*.json', 'resources/models/cp', 'resources/icons/*.png', 'resources/icons/organism_*.svg', 'resources/icons/parascape.svg', 'resources/icons/parascape-LICENSE.txt', 'resources/icons/loading_spinner.gif', 'resources/icons/spaceout.gif', 'resources/font/**/*', 'resources/images/*', 'resources/themes/*.jpg', 'resources/splash/*.jpg', 'resources/splash/*.json', 'resources/setting_animations/*.json', 'resources/plate_templates/*.json', 'resources/setting_animations/gifs/*.gif'],},
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
            'spacr-navigator=spacr.qt:run',
            'spacr-nav=spacr.qt:run',
            'spaceout=spacr.qt.spaceout:main',
            'spaceout-navigator=spacr.qt.spaceout:main',
            'spaceout-nav=spacr.qt.spaceout:main',
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
            'pytest>=8.0,<10',
            'pytest-qt>=4.4,<5',
            'pytest-xdist>=3.6.1,<4',
            'tomli>=2.0; python_version < "3.11"',
            'hypothesis>=6.100,<7',
            'docutils>=0.20.1,<0.24',
            'pypdfium2>=5.13.0,<6',
            'markdown-it-py>=2.2,<5',
            'readme-renderer>=43,<47',
            'pyarrow>=14.0.2,<27',
            'ruff>=0.9,<1',
            'mypy>=1.11,<3',
            'xenon>=0.9,<1',
            'pingouin>=0.5.5,<2.0',
        ],
        'headless': [
            'opencv-python-headless>=4.9.0.80,<6.0; sys_platform != "darwin" or platform_machine != "x86_64" or python_version >= "3.13"',
            'opencv-python-headless>=4.9.0.80,<4.12; sys_platform == "darwin" and platform_machine == "x86_64" and python_version < "3.13"',
        ],
        'embeddings': ['timm>=0.9,<2.0'],
        'trackastra': [
            'trackastra>=0.5,<1.0; python_version >= "3.10" and '
            '(sys_platform != "darwin" or platform_machine != "x86_64" '
            'or python_version < "3.13")',
        ],
        'ultrack': [
            'ultrack>=0.6,<1.0; python_version >= "3.10" and '
            'python_version < "3.14" and (sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.13") and '
            '(sys_platform != "linux" or platform_machine == "x86_64" or '
            'platform_machine == "AMD64")',
        ],
        'attribution': ['torchcam>=0.4.0,<1.0'],
        'gpu': [
            'cucim-cu12>=24.0; platform_system == "Linux"',
            'cupy-cuda12x>=13.0; platform_system == "Linux"',
        ],
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
        'anndata': ['anndata>=0.10,<0.14'],
        'databases': ['duckdb>=1.1,<3', 'psycopg[binary]>=3.1,<4'],
        'dinocell': ['dinocell>=0.74,<1.0'],
        'samcell': ['samcell>=1.2,<2.0'],
        'napari': ['napari>=0.5,<1.0'],
        'tem': ['mrcfile>=1.5,<2'],
        'full': ['opencv-python'],
        'sweep': ['threadpoolctl>=3.0,<4'],
        'qt': [
            'PySide6>=6.6,!=6.12.0,<7; sys_platform != "darwin" and python_version < "3.15"',
            'PySide6>=6.12,<7; sys_platform != "darwin" and python_version >= "3.15"',
            'PySide6>=6.6,<6.11; sys_platform == "darwin"',
            'qtawesome>=1.3,<2',
            'pyqtgraph>=0.13.3,<1',
            'win10toast>=0.9; platform_system == "Windows"',
        ],
        'flowview': [
            'PySide6>=6.6,!=6.12.0,<7; sys_platform != "darwin" and python_version < "3.15"',
            'PySide6>=6.12,<7; sys_platform != "darwin" and python_version >= "3.15"',
            'PySide6>=6.6,<6.11; sys_platform == "darwin"',
        ],
        'flow-gpu': [
            'moderngl>=5.12,<6; sys_platform != "darwin" or platform_machine != "x86_64"',
            'moderngl>=5.12,<5.13; sys_platform == "darwin" and platform_machine == "x86_64"',
            'glcontext>=3.0,<3.1; sys_platform == "darwin" and platform_machine == "x86_64"',
        ],
        'fractal': ['vispy>=0.14,<1.0'],
        'tutorial': [
            'PySide6>=6.6,!=6.12.0,<7; sys_platform != "darwin" and python_version < "3.15"',
            'PySide6>=6.12,<7; sys_platform != "darwin" and python_version >= "3.15"',
            'PySide6>=6.6,<6.11; sys_platform == "darwin"',
            'qtawesome>=1.3,<2',
            'win10toast>=0.9; platform_system == "Windows"',
            'piper-tts>=1.2,<2; python_version < "3.15" and (sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.14") and '
            '(sys_platform != "darwin" or python_version < "3.14" or '
            '"23" <= platform_release)',
        ],

        'czi': ['pylibCZIrw>=5.0.0,<7.0; python_version < "3.14"',
                'czifile'],
        'nd2': ['nd2reader>=3.3.0,<4.0'],
        'lif': ['readlif'],
        'vendor-watch': ['nd2>=0.10,<1; python_version >= "3.10"',
                         'liffile>=2025.1.25; python_version >= "3.12"'],
        'zernike': ['mahotas>=1.4.13,<2.0'],
        'btrack': ['btrack>=0.7.0,<1.0; sys_platform != "darwin" or '
                   'python_version < "3.14" or "24" <= platform_release'],

        'numpyro': ['numpyro>=0.13,<1.0', 'jax>=0.4,<1.0'],
        'pymc': ['pymc>=5.10,<7'],
        'pyfixest': ['pyfixest>=0.40.1,<1; python_version >= "3.10" and '
                    '(sys_platform != "darwin" or python_version >= "3.11")'],
        'glum': ['glum>=3.1.2,<4; python_version >= "3.10"'],
        'gpytorch': ['gpytorch>=1.11,<2; python_version >= "3.10"'],

        'zarr': ['zarr>=2.16,<4',
                 'numcodecs>=0.12,<1; python_version < "3.15"',
                 'numcodecs>=0.16.6,<0.17; python_version >= "3.15"'],
        'omero': ['omero-py>=5.17,<6'],
        'cloud': ['fsspec>=2023.1', 's3fs>=2023.1', 'gcsfs>=2023.1',
                  'adlfs>=2023.1', 'aiohttp>=3.8,<4', 'botocore>=1.29,<2'],
        'survival': ['lifelines>=0.27,<1'],
        'search': ['faiss-cpu>=1.7.4,<2; sys_platform != "darwin" or '
                   'python_version < "3.14" or "23" <= platform_release'],
        'foundation': ['transformers>=4.40,<6'],
        'notify': ['keyring>=23.0,<27'],
        'chem': ['rdkit>=2023.3'],
        'microscope': ['pycromanager>=0.28,<2'],

        'all': [
            'PySide6>=6.6,!=6.12.0,<7; sys_platform != "darwin" and python_version < "3.15"',
            'PySide6>=6.12,<7; sys_platform != "darwin" and python_version >= "3.15"',
            'PySide6>=6.6,<6.11; sys_platform == "darwin"',
            'qtawesome>=1.3,<2',
            'pyqtgraph>=0.13.3,<1',
            'vispy>=0.14,<1.0',
            'win10toast>=0.9; platform_system == "Windows"',
            'piper-tts>=1.2,<2; python_version < "3.15" and (sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.14") and '
            '(sys_platform != "darwin" or python_version < "3.14" or '
            '"23" <= platform_release)',
            'trackastra>=0.5,<1.0; python_version >= "3.10" and '
            '(sys_platform != "darwin" or platform_machine != "x86_64" '
            'or python_version < "3.13")',
            'ultrack>=0.6,<1.0; python_version >= "3.10" and '
            'python_version < "3.14" and (sys_platform != "darwin" or '
            'platform_machine != "x86_64" or python_version < "3.13") and '
            '(sys_platform != "linux" or platform_machine == "x86_64" or '
            'platform_machine == "AMD64")',
            'catboost>=1.2,<2.0',
            'lightgbm>=4.0,<5.0',
            'ultralytics>=8.0,<9',
            'pylibCZIrw>=5.0.0,<7.0; python_version < "3.14"',
            'czifile',
            'nd2reader>=3.3.0,<4.0',
            'readlif',
            'mahotas>=1.4.13,<2.0',
            'btrack>=0.7.0,<1.0; sys_platform != "darwin" or '
            'python_version < "3.14" or "24" <= platform_release',
            'anndata>=0.10,<0.14',
            'mrcfile>=1.5,<2',
        ],
    },
)
