"""The vector hit report never exposes a partial SVG or leaves a scratch file."""
from __future__ import annotations

import json
import tempfile
import types
import xml.etree.ElementTree as ET

import pandas as pd
import pytest

from spacr import sp_stats


@pytest.fixture
def structure(monkeypatch):
    frame = pd.DataFrame([{
        "hit": True, "smiles_valid": True, "cluster": 1, "best_rank": 1,
        "smiles": "CCO", "compound": "ethanol", "potency": 0.5,
        "cytotoxicity_index": 2.0,
    }])

    class Drawer:
        def __init__(self, width, height):
            assert (width, height) == (300, 300)

        def FinishDrawing(self):
            pass

        def GetDrawingText(self):
            return ('<svg xmlns="http://www.w3.org/2000/svg">'
                    '<path d="M0 0 L1 1"/></svg>')

    drawing = types.SimpleNamespace(MolDraw2DSVG=Drawer,
                                    PrepareAndDrawMolecule=lambda *_: None)
    monkeypatch.setattr(sp_stats, "_rdkit", lambda: (
        types.SimpleNamespace(MolFromSmiles=lambda smiles: object()), None,
        types.SimpleNamespace(rdMolDraw2D=drawing)))
    return types.SimpleNamespace(sar=frame, clustered=True)


def test_a_vector_report_is_complete_and_has_no_scratch_file(
        structure, tmp_path):
    path = sp_stats._write_hit_structures_svg(structure, tmp_path)
    document = ET.parse(path).getroot()
    metadata = document.find("{http://www.w3.org/2000/svg}metadata")
    assert json.loads(metadata.text)["structures"][0]["compound"] == "ethanol"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["hit_structures.svg"]


def test_a_failed_replace_keeps_the_previous_report_and_cleans_up(
        structure, tmp_path, monkeypatch):
    destination = tmp_path / "hit_structures.svg"
    destination.write_text("previous complete report")

    def refuse_replace(*_args):
        raise OSError("destination is unavailable")

    monkeypatch.setattr("os.replace", refuse_replace)
    with pytest.raises(OSError, match="destination is unavailable"):
        sp_stats._write_hit_structures_svg(structure, tmp_path)
    assert destination.read_text() == "previous complete report"
    assert list(tmp_path.iterdir()) == [destination]


def test_a_failed_scratch_creation_does_not_touch_the_previous_report(
        structure, tmp_path, monkeypatch):
    destination = tmp_path / "hit_structures.svg"
    destination.write_text("previous complete report")

    def refuse_scratch(*_args, **_kwargs):
        raise OSError("cannot create scratch file")

    monkeypatch.setattr(tempfile, "NamedTemporaryFile", refuse_scratch)
    with pytest.raises(OSError, match="cannot create scratch file"):
        sp_stats._write_hit_structures_svg(structure, tmp_path)
    assert destination.read_text() == "previous complete report"
    assert list(tmp_path.iterdir()) == [destination]
