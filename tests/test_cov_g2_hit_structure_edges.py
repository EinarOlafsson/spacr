"""Hit-structure artwork edges: headless screen ink and a bitmap-only report."""
from __future__ import annotations

import sys

import pytest

from spacr import sp_stats as sp
from tests.test_chemistry_aware_hits import _scored, _screen

pytest.importorskip("rdkit")


def _result():
    frame, layout, host = _screen()
    return sp._structure_activity(_scored(frame), layout, host=host)


def test_screen_ink_falls_back_without_qt(monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    figure = Figure()
    assert sp._draw_hit_structures(figure, _result(), target="screen") > 0


def test_a_report_without_vector_artwork_lists_only_the_bitmap(tmp_path,
                                                              monkeypatch):
    from pathlib import Path

    def save_bitmap(figure, filename, **kwargs):
        Path(filename).write_bytes(b"png")
        return str(filename)

    monkeypatch.setattr("spacr.plot.save_figure", save_bitmap)
    monkeypatch.setattr(sp, "_write_hit_structures_svg", lambda c, o: None)
    written = sp._write_sar_report(_result(), tmp_path, target="print")
    assert "hit_structures" in written and "hit_structures_svg" not in written
