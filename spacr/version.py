"""
Copyright © 2025 olafsson lab
"""

from __future__ import annotations

import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from platform import python_version

_PACKAGE_CANDIDATES = ("spacr-nightly", "spacr")


def get_version() -> str:
    for package_name in _PACKAGE_CANDIDATES:
        try:
            return package_version(package_name)
        except PackageNotFoundError:
            continue
    return "unknown"


def get_torch_version() -> str:
    try:
        import torch
        return torch.__version__
    except Exception:
        return "not available"


def get_version_info() -> dict[str, str]:
    return {
        "spacr_version": get_version(),
        "platform": sys.platform,
        "python_version": python_version(),
        "torch_version": get_torch_version(),
    }


def format_version_info() -> str:
    info = get_version_info()
    return (
        f"spacr version:\t{info['spacr_version']}\n"
        f"platform:\t{info['platform']}\n"
        f"python version:\t{info['python_version']}\n"
        f"torch version:\t{info['torch_version']}"
    )


__version__ = get_version()
version_str = format_version_info()