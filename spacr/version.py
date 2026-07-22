"""
Copyright © 2025 olafsson lab
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from platform import python_version

_PACKAGE_CANDIDATES = ("spacr-nightly", "spacr")


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
version_str = format_version_info()