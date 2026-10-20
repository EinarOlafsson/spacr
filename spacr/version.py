"""The installed spaCR version, and the environment report built from it.

A source checkout or editable install reports the version in its own
``spacr/_version.py``, because its installed metadata can be stale; an
installed wheel or frozen installer reports its distribution metadata.

Copyright © 2025 olafsson lab
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, distribution, version as package_version
from pathlib import Path
from platform import python_version

_PACKAGE_CANDIDATES = ("spacr", "spacr-nightly")

_PACKAGE_DIR = Path(__file__).resolve().parent


def _source_version() -> str | None:
    """Return the version literal shipped beside the imported package.

    :returns: ``spacr._version.__version__``, or ``None`` when it is missing.
    """
    try:
        from ._version import __version__ as literal
    except Exception:
        return None
    return str(literal) or None


def _distribution_owns(package_name: str, package_dir: Path) -> bool:
    """Say whether ``package_name``'s installed files are the imported ones.

    :param package_name: Distribution name to inspect.
    :param package_dir: Directory of the imported ``spacr`` package.
    :returns: ``True`` when the distribution's ``spacr/version.py`` is the file
        this module was loaded from.
    """
    try:
        dist = distribution(package_name)
        located = Path(dist.locate_file("spacr/version.py")).resolve()
    except Exception:
        return False
    return located == package_dir / "version.py"


def _runs_from_source(package_dir: Path | None = None) -> bool:
    """Say whether the imported spacr is a source checkout or editable install.

    A checkout or an editable install is imported from files that no installed
    distribution owns, so its dist-info can be stale (installed at 1.5.1.3
    while the checkout is at 1.5.1.4).  A wheel and a frozen installer are
    imported from the files their own metadata lists.

    :param package_dir: Directory of the imported package; defaults to this
        module's directory.
    :returns: ``True`` when the package's own version literal should win.
    """
    package_dir = package_dir or _PACKAGE_DIR
    if (package_dir.parent / "setup.py").is_file():
        return True
    return not any(_distribution_owns(name, package_dir)
                   for name in _PACKAGE_CANDIDATES)


def get_version() -> str:
    """Return the running spacr version, or ``"unknown"`` if it cannot be found.

    A source checkout or editable install reports the version in its own
    ``spacr/_version.py``; an installed wheel or frozen installer reports its
    distribution metadata, then that literal when no metadata is present.

    :returns: The version string of the spacr package that is actually imported.
    """
    if _runs_from_source():
        literal = _source_version()
        if literal:
            return literal
    for package_name in _PACKAGE_CANDIDATES:
        try:
            return package_version(package_name)
        except PackageNotFoundError:
            continue
    return _source_version() or "unknown"


def get_torch_version() -> str:
    """Return the installed PyTorch version, or ``"not available"`` if torch is missing.

    :returns: ``torch.__version__`` when importable, otherwise a placeholder string.
    """
    try:
        import torch
        return torch.__version__
    except Exception:
        return "not available"


def get_version_info() -> dict[str, str]:
    """Return a dict of spacr, platform, Python, and torch version strings.

    :returns: Mapping with keys ``spacr_version``, ``platform``, ``python_version``, ``torch_version``.
    """
    return {
        "spacr_version": get_version(),
        "platform": sys.platform,
        "python_version": python_version(),
        "torch_version": get_torch_version(),
    }


def format_version_info() -> str:
    """Return a human-readable multi-line summary of the current environment.

    :returns: Tab-aligned version report suitable for CLI display.
    """
    info = get_version_info()
    return (
        f"spacr version:\t{info['spacr_version']}\n"
        f"platform:\t{info['platform']}\n"
        f"python version:\t{info['python_version']}\n"
        f"torch version:\t{info['torch_version']}"
    )


__version__ = get_version()


def __getattr__(name: str):
    """Resolve ``version_str`` lazily (PEP 562).

    :param name: Module attribute requested by Python's fallback lookup.
    :returns: The formatted environment report for ``version_str``.
    :raises AttributeError: If ``name`` is not ``version_str``.

    ``format_version_info()`` calls ``get_torch_version()``, which imports
    torch — roughly 0.94 s. Resolving ``version_str`` lazily keeps
    ``import spacr`` from importing torch, so light entry points such as
    :mod:`spacr.validate` can answer in about a second before a run starts.

    ``from spacr.version import version_str`` still works unchanged; torch
    is imported only when ``version_str`` is accessed.
    """
    if name == "version_str":
        return format_version_info()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
