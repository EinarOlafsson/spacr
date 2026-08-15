"""Regression tests for the generated repository facts."""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_refresh_module():
    spec = importlib.util.spec_from_file_location(
        "spacr_skill_refresh", ROOT / "skill" / "refresh.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_facts_version_comes_from_the_declared_package_version():
    """A dynamic ``__version__`` assignment must not become report text."""
    refresh = _load_refresh_module()
    match = re.search(
        r'^VERSION\s*=\s*["\']([^"\']+)["\']',
        (ROOT / "setup.py").read_text(encoding="utf-8"),
        flags=re.MULTILINE,
    )
    assert match is not None
    declared = match.group(1)

    assert refresh._version() == declared
    assert f"version: **{declared}**" in refresh.write_facts()
