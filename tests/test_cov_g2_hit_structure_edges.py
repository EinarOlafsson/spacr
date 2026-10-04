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
