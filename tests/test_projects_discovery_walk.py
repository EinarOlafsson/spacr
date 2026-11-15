"""Project discovery returns a bounded list without walking off a cliff.

:func:`spacr.projects.discover` backs the project browser. It stops at the
result limit (also between roots), never descends into hidden or skipped
folders or plain files, ignores blank roots rather than searching the working
directory, survives a root it cannot list, stops at a project, and visits a
folder reached through a symlink loop once.
"""
from __future__ import annotations

import os

import pytest

from spacr import projects
from spacr.projects import discover


def _make_project(path):
    from spacr import data_manager as _dm
    path.mkdir(parents=True, exist_ok=True)
    (path / _dm.ARCHIVE_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    return path


def test_discover_finds_projects_within_its_limits(subtests, tmp_path):
    """Exact sorted absolute project paths for each walk."""
    flat = tmp_path / "flat"
    for name in ("a", "b", "c", "d"):
        _make_project(flat / name)

    skipped = tmp_path / "skipped"
    _make_project(skipped / ".hidden")
    assert "__pycache__" in projects.SKIP_DIRS
    _make_project(skipped / "__pycache__")
    _make_project(skipped / "real")
    (skipped / "readme.txt").write_text("hello", encoding="utf-8")

    first, second = tmp_path / "first", tmp_path / "second"
    _make_project(first / "a")
    _make_project(first / "b")
    _make_project(second / "c")

    nested = _make_project(tmp_path / "nested" / "outer")
    _make_project(nested / "inner")
    deeper = _make_project(tmp_path / "deep" / "x" / "y" / "z")

    cases = [
        ("discovery stops at the limit", [flat], {"depth": 2, "limit": 2},
         [flat / "a", flat / "b"]),
        ("no limit pressure finds every project", [flat],
         {"depth": 2, "limit": 10},
         [flat / "a", flat / "b", flat / "c", flat / "d"]),
        ("hidden, skipped folders and plain files are not searched",
         [skipped], {"depth": 2, "limit": 10}, [skipped / "real"]),
        ("a second root waits for the limit", [first, second],
         {"depth": 2, "limit": 2}, [first / "a", first / "b"]),
        ("both roots fit under a larger limit", [second, first],
         {"depth": 2, "limit": 10},
         [first / "a", first / "b", second / "c"]),
        ("blank roots are ignored", ["", None, skipped],
         {"depth": 2, "limit": 10}, [skipped / "real"]),
        ("the same root twice is reported once", [flat / "a", flat / "a"],
         {"depth": 2, "limit": 10}, [flat / "a"]),
        ("a root that cannot be listed yields nothing",
         [tmp_path / "missing"], {"depth": 2, "limit": 10}, []),
        ("descent stops at a project", [tmp_path / "nested"],
         {"depth": 3, "limit": 10}, [nested]),
        ("depth bounds the walk", [tmp_path / "deep"],
         {"depth": 2, "limit": 10}, []),
        ("enough depth reaches it", [tmp_path / "deep"],
         {"depth": 3, "limit": 10}, [deeper]),
        ("depth 0 tests only the root", [flat], {"depth": 0, "limit": 10},
         []),
        ("a root that is a project is one", [str(flat / "c")],
         {"depth": 0, "limit": 10}, [flat / "c"]),
    ]
    for case_id, roots, kwargs, expected in cases:
        with subtests.test(msg=case_id):
            assert discover(roots, **kwargs) == tuple(str(p) for p in expected)


@pytest.mark.needs_symlink
def test_discover_visits_a_symlink_loop_once(tmp_path):
    """A folder linked back into itself would otherwise recurse forever."""
    root = tmp_path / "root"
    inner = _make_project(root / "inner")
    os.symlink(root, root / "loop", target_is_directory=True)

    assert discover([root, root / "loop"], depth=3, limit=10) == (str(inner),)
    # Entering through the link first still reports the project once.
    assert discover([root / "loop", root], depth=3, limit=10) == (
        str(root / "loop" / "inner"),)
