"""No directory under ``tools/`` may share a name with an importable package.

WHY THIS IS A TEST AND NOT A CONVENTION. Several tests put ``<root>/tools``
on ``sys.path`` so they can import a tool module, and at least one does it at
MODULE IMPORT time (``tests/test_conda_publication_docs.py``), which means
the entry is there for the rest of the session. A directory under ``tools/``
with no ``__init__.py`` is then a PEP 420 namespace portion competing for
that top-level name.

Usually it loses: a namespace portion is recorded and the path scan
CONTINUES, so a real installed package still wins. It wins when the real
package is NOT installed -- and then the name resolves to an empty namespace
module, and every attribute access on it raises ``AttributeError`` rather
than the ``ImportError`` a missing dependency would give.

That happened. ``tools/coverage/`` held shell scripts. The Fast job does not
install coverage.py. numba does ``class NumbaTracer(coverage.types.Tracer)``
at import time, so numba died, umap died with it, and 18 tests failed with:

    AttributeError: module 'coverage' has no attribute 'types'

It had already cost something quieter: ``spacr/timelapse.py`` lazy-loads
Trackpy partly because this shadow broke its Numba import. The directory was
renamed to ``tools/coverage_scripts`` on 2026-09-13.

Adding an ``__init__.py`` is NOT the repair, and is worse than the disease: a
regular package short-circuits the path scan and wins UNCONDITIONALLY, which
would break ``import coverage`` in the twelve coverage shards that do install
it. The only safe rule is that the name must not collide at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"


def _tool_directories() -> list[str]:
    """Every directory under ``tools/`` importable as a top-level name."""
    if not TOOLS.is_dir():
        return []
    return sorted(
        entry.name for entry in TOOLS.iterdir()
        if entry.is_dir()
        and not entry.name.startswith((".", "_"))
        and entry.name.isidentifier()
    )


def _resolves_without_tools(name: str):
    """The origin of ``name`` when ``tools/`` is not on the path, or None."""
    saved = sys.path[:]
    sys.path = [p for p in sys.path
                if Path(p).resolve() != TOOLS.resolve()]
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    finally:
        sys.path = saved
    return None if spec is None else (spec.origin or "namespace")


@pytest.mark.parametrize("name", _tool_directories())
def test_a_tools_directory_does_not_shadow_an_importable_package(name):
    origin = _resolves_without_tools(name)
    assert origin is None, (
        f"tools/{name}/ shares its name with an importable package "
        f"({origin}). A test that puts tools/ on sys.path would make this "
        f"directory compete for the name '{name}', and it WINS whenever "
        f"that package is not installed -- resolving to an empty namespace "
        f"module whose every attribute raises AttributeError. Rename the "
        f"directory. Do not add __init__.py: that makes it win "
        f"unconditionally, which is worse.")


def test_the_check_has_something_to_check():
    """A parametrised test over an empty list passes by vacuum."""
    assert _tool_directories(), "no directories found under tools/"
