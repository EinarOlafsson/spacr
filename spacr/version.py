"""The installed spaCR version, and the environment report built from it.

Resolves the version from installed package metadata rather than a
hard-coded string, so a source checkout and an installed wheel agree.

Copyright © 2025 olafsson lab
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from platform import python_version

# Prefer the canonical `spacr` distribution. `spacr-nightly` stays
# as a fallback in case a very old install still uses that name.
_PACKAGE_CANDIDATES = ("spacr", "spacr-nightly")


def get_version() -> str:
    """Return the installed spacr package version, or ``"unknown"`` if not found.

    :returns: Version string from the first candidate distribution that resolves.
    """
    for package_name in _PACKAGE_CANDIDATES:
        try:
            return package_version(package_name)
        except PackageNotFoundError:
            continue
    return "unknown"


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

    ``format_version_info()`` calls ``get_torch_version()``, which imports
    torch — roughly 0.94 s. Evaluating it at module scope meant every
    ``import spacr`` paid for torch whether or not anything needed it, which
    defeated the point of light entry points like :mod:`spacr.validate`,
    whose whole value is answering in about a second before a run starts.

    ``from spacr.version import version_str`` still works unchanged; the cost
    is now paid only by the caller that actually asks for it.
    """
    if name == "version_str":
        return format_version_info()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")