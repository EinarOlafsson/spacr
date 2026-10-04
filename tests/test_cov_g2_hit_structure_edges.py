"""Hit-structure artwork edges: bad SMILES, empty glyph paths, headless ink."""
from __future__ import annotations

import sys
import types
import xml.etree.ElementTree as ET

import pytest

from spacr import sp_stats as sp
from tests.test_chemistry_aware_hits import _scored, _screen

pytest.importorskip("rdkit")
SVG = "{http://www.w3.org/2000/svg}"


def _result():
    frame, layout, host = _screen()
    return sp._structure_activity(_scored(frame), layout, host=host)


def test_unparseable_hits_are_skipped_and_none_left_writes_nothing(tmp_path):
    result = _result()
    if "smiles_valid" in result.sar.columns:
        result.sar["smiles_valid"] = True
    result.sar.loc[result.sar.hit, "smiles"] = "not((a smiles"
    assert sp._write_hit_structures_svg(result, tmp_path) is None
    assert not list(tmp_path.iterdir())


def test_empty_glyph_paths_are_dropped(tmp_path, monkeypatch):
    chem, structs, draw = sp._rdkit()

    class _Drawer:
        def __init__(self, width, height):
            pass

        def FinishDrawing(self):
            return None

        def GetDrawingText(self):
            return (f'<svg xmlns="{SVG[1:-1]}"><g><path d=" "/>'
                    '<path d="M0 0 L1 1"/></g></svg>')

    fake = types.SimpleNamespace(rdMolDraw2D=types.SimpleNamespace(
        MolDraw2DSVG=_Drawer, PrepareAndDrawMolecule=lambda drawer, mol: None))
    monkeypatch.setattr(sp, "_rdkit", lambda: (chem, structs, fake))
    path = sp._write_hit_structures_svg(_result(), tmp_path)
    paths = ET.parse(path).getroot().findall(".//" + SVG + "path")
    assert paths and all(p.get("d").strip() for p in paths)


def test_screen_ink_falls_back_without_qt(monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setitem(sys.modules, "spacr.qt.preferences", None)
    figure = Figure()
    assert sp._draw_hit_structures(figure, _result(), target="screen") > 0


def test_hits_without_a_validity_column_are_all_drawn(tmp_path):
    result = _result()
    result.sar = result.sar.drop(columns=["smiles_valid"], errors="ignore")
    assert sp._write_hit_structures_svg(result, tmp_path).endswith(".svg")


def test_a_temporary_file_that_cannot_be_made_leaves_nothing(tmp_path,
                                                             monkeypatch):
    import tempfile

    def refuse(*a, **k):
        raise OSError("read-only folder")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse)
    with pytest.raises(OSError, match="read-only"):
        sp._write_hit_structures_svg(_result(), tmp_path)
    assert not list(tmp_path.iterdir())


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
