"""The README's wording, for tests that check what it says.

README.rst and its nine translations are images and code blocks since
2026-10-12 (``tools/build_readme_images.py``). What they say lives in
``packaging/readme_source.yaml`` and ``docs/i18n/readme/source/<lang>.yaml``;
:func:`readme_words` renders that source back to RST prose -- headings with
underlines, one paragraph per line, lists, code blocks and the generated
blocks as committed -- so a test about the README's words reads the words.
``tests/test_readme_is_built_from_images.py`` holds the images to the source.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_NAME = "spacr_build_readme_images"


def readme_tool():
    """``tools/build_readme_images.py`` as a module, loaded once."""
    module = sys.modules.get(_NAME)
    if module is None:
        spec = importlib.util.spec_from_file_location(
            _NAME, ROOT / "tools" / "build_readme_images.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[_NAME] = module
        spec.loader.exec_module(module)
    return module


def language_of(path) -> str:
    match = re.fullmatch(r"README\.(?P<language>[^.]+)\.rst", Path(path).name)
    return match.group("language") if match else "en"


def readme_words(path=None) -> str:
    """The README at ``path`` (default: the English one) as RST prose."""
    return readme_tool().source_rst(language_of(path or ROOT / "README.rst"))
