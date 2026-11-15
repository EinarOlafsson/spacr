"""Re-rooting climbs a bounded ladder of folders and tries tails deepest first.

Two units of :mod:`spacr.portable_paths`:

* :func:`~spacr.portable_paths.candidate_roots` normalises whatever a caller
  holds -- plate folder, screen folder, ``measurements/``, the database file
  itself, ``~`` or a path object -- into its folder and at most
  ``_MAX_CLIMB`` ancestors, nearest first, each strictly shorter than the
  last, stopping at the filesystem root; nothing yields nothing;
* :func:`~spacr.portable_paths._suffixes` gives the tails a recorded path can
  be re-rooted by (``data``-led first, then longest), and a tail that does
  not end the recorded text is skipped by the re-rooting rather than used to
  slice it.
"""
from __future__ import annotations

import os
from pathlib import Path

from spacr import portable_paths as PP


def _ladder(start):
    """``start`` and its ancestors, as the climb is documented to walk them."""
    out, here = [], os.path.abspath(str(start))
    for _ in range(PP._MAX_CLIMB + 1):
        out.append(here)
        parent = os.path.dirname(here)
        if parent == here:
            break
        here = parent
    return tuple(out)


def test_candidate_roots_climb_from_the_folder_nearest_first(subtests,
                                                             tmp_path,
                                                             monkeypatch):
    """Exact ladders for folders, files, the root, ``~`` and empty input."""
    plate = tmp_path / "screen" / "plate1"
    measurements = plate / "measurements"
    measurements.mkdir(parents=True)
    database = measurements / "measurements.db"
    database.write_bytes(b"")
    loose = tmp_path / "plate.csv"
    loose.write_text("x\n", encoding="utf-8")
    deep = tmp_path.joinpath(*[f"level{n}" for n in range(12)])
    deep.mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert PP._MAX_CLIMB == 2

    cases = [
        ("None yields nothing", None, ()),
        ("an empty string yields nothing", "", ()),
        ("the filesystem root is its own only candidate", "/",
         (os.path.abspath("/"),)),
        ("a top-level folder climbs to the root once", "/a", ("/a", "/")),
        ("a plate folder yields itself, then screen, then above",
         str(plate), (str(plate), str(plate.parent), str(tmp_path))),
        ("a database file is climbed from its folder", str(database),
         (str(measurements), str(plate), str(plate.parent))),
        ("a plain file is climbed from its folder", str(loose),
         _ladder(tmp_path)),
        ("a deep tree is bounded", str(deep), _ladder(deep)),
        ("a path object is accepted", measurements, _ladder(measurements)),
        ("a home-relative path is expanded", "~/screen",
         (str(plate.parent), str(tmp_path), str(tmp_path.parent))),
    ]
    for case_id, root, expected in cases:
        with subtests.test(msg=case_id):
            roots = PP.candidate_roots(root)
            assert roots == expected
            assert len(roots) <= PP._MAX_CLIMB + 1
            assert len(set(roots)) == len(roots)
            assert all(roots)
            for nearer, further in zip(roots, roots[1:]):
                assert further == os.path.dirname(nearer)
                assert len(further) < len(nearer)
            if root and Path(root).is_file():
                assert str(root) not in roots


def test_suffixes_offer_data_led_tails_first_and_reroot_skips_a_misfit(
        subtests, tmp_path):
    """Exact tail order per recorded path; a doubled separator still re-roots."""
    cases = [
        ("a single component has no tail", "measurements.db", []),
        ("an empty path has no tail", "", []),
        ("a single absolute component has no tail", "/single", []),
        ("two components offer their tail", "plate1/measurements.db",
         [("plate1", "measurements.db")]),
        ("a data-led tail comes before longer ones", "/a/b/data/c/d.png",
         [("data", "c", "d.png"), ("a", "b", "data", "c", "d.png"),
          ("b", "data", "c", "d.png"), ("c", "d.png")]),
        ("a doubled separator still splits into components",
         "/gone//plate1/data/crop.png",
         [("data", "crop.png"), ("gone", "plate1", "data", "crop.png"),
          ("plate1", "data", "crop.png")]),
        ("a Windows path splits on backslashes", "C:\\x\\y.png",
         [("C:", "x", "y.png"), ("x", "y.png")]),
    ]
    for case_id, path, expected in cases:
        with subtests.test(msg=case_id):
            assert PP._suffixes(path) == expected

    with subtests.test(msg="a tail that does not end the text is skipped"):
        root = tmp_path / "plate1" / "data"
        root.mkdir(parents=True)
        (root / "crop.png").write_bytes(b"")
        recorded = "/gone//plate1/data/crop.png"
        tails = ["/".join(s) for s in PP._suffixes(recorded)]
        # The first tail ends the text but is not under the root; the second
        # does not end the text at all and must be passed over, not sliced.
        assert recorded.endswith(tails[0])
        assert not recorded.endswith(tails[1])
        mapped, prefixes = PP._reroot_with_prefix(recorded, str(tmp_path))
        assert mapped == str(root / "crop.png")
        assert prefixes == ("/gone//", str(tmp_path) + os.sep)
