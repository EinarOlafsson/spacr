"""Hit-structure vector artwork and the TVN whitening step of :mod:`spacr.sp_stats`.

``_write_hit_structures_svg`` exports the clustered hit structures of a
chemistry-aware screen as one SVG (skipping hits whose SMILES does not parse,
dropping RDKit's empty glyph paths, writing atomically). ``_whiten_profiles``
is the ``'tvn'`` profiling step: it whitens the complete wells on the
negative-control wells and turns every refusal into the profiling error the
Measure run reports.
"""
from __future__ import annotations

import json
import tempfile
import types
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
import pytest

from spacr import sp_stats as sp
from tests.test_chemistry_aware_hits import _scored, _screen

SVG = "{http://www.w3.org/2000/svg}"


def _result():
    """The clustered chemistry result of the two-plate test screen (5 hits)."""
    frame, layout, host = _screen()
    return sp._structure_activity(_scored(frame), layout, host=host)


def _unparseable(result):
    if "smiles_valid" in result.sar.columns:
        result.sar["smiles_valid"] = True
    result.sar.loc[result.sar.hit, "smiles"] = "not((a smiles"
    return result


def _without_validity(result):
    result.sar = result.sar.drop(columns=["smiles_valid"], errors="ignore")
    return result


def _fake_draw(chem, structs):
    """An RDKit draw module whose drawings hold one empty and one real path."""

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
    return lambda: (chem, structs, fake)


def _refuse_tempfile(*args, **kwargs):
    raise OSError("read-only folder")


def test_write_hit_structures_svg_exports_parseable_hits_atomically(tmp_path, monkeypatch,
                                                                    subtests):
    """_write_hit_structures_svg: one tile per parseable hit, no empty paths, nothing left on failure."""
    pytest.importorskip("rdkit")
    chem, structs, _draw = sp._rdkit()
    hits = ["cpd1", "cpd0", "cpd2", "cpd4", "cpd3"]
    cases = [
        # id, result transform, patches, expected compounds (None = nothing written), error
        ("all_hits_drawn", lambda r: r, {}, hits, None),
        ("no_validity_column_draws_all", _without_validity, {}, hits, None),
        ("empty_glyph_paths_dropped", lambda r: r, {"draw": True}, hits, None),
        ("unparseable_hits_write_nothing", _unparseable, {}, None, None),
        ("temporary_file_refused_leaves_nothing", lambda r: r, {"tempfile": True}, None,
         "read-only"),
    ]
    for case_id, transform, patches, compounds, error in cases:
        with subtests.test(msg=case_id), monkeypatch.context() as patch:
            out_dir = tmp_path / case_id
            out_dir.mkdir()
            if patches.get("draw"):
                patch.setattr(sp, "_rdkit", _fake_draw(chem, structs))
            if patches.get("tempfile"):
                patch.setattr(tempfile, "NamedTemporaryFile", _refuse_tempfile)
            result = transform(_result())
            if error is not None:
                with pytest.raises(OSError, match=error):
                    sp._write_hit_structures_svg(result, out_dir)
                assert list(out_dir.iterdir()) == []
                continue
            path = sp._write_hit_structures_svg(result, out_dir)
            if compounds is None:
                assert path is None
                assert list(out_dir.iterdir()) == []
                continue
            assert path == str(out_dir / "hit_structures.svg")
            assert [p.name for p in out_dir.iterdir()] == ["hit_structures.svg"]
            root = ET.parse(path).getroot()
            assert (root.get("width"), root.get("height")) == ("1500", "400")
            records = json.loads(root.find(SVG + "metadata").text)["structures"]
            assert [r["compound"] for r in records] == compounds
            assert [t.text for t in root.iter(SVG + "title")] == compounds
            paths = root.findall(".//" + SVG + "path")
            assert paths and all(p.get("d").strip() for p in paths)
            if patches.get("draw"):
                assert [p.get("d") for p in paths] == ["M0 0 L1 1"] * len(compounds)


FEATURES = ["a", "b", "c"]


def _wells(n=8, seed=0):
    """``n`` wells with three random features."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    frame.insert(0, "well", [f"w{k}" for k in range(n)])
    return frame


def _mask(n, true_at):
    mask = np.zeros(n, bool)
    mask[true_at] = True
    return mask


def test_whiten_profiles_whitens_complete_wells_on_the_controls(subtests):
    """_whiten_profiles: controls become identity-covariance; refusals are _ProfilingError."""
    with subtests.test(msg="complete_wells_whitened_incomplete_untouched"):
        wells = _wells(n=12)
        negative = _mask(12, slice(0, 6))
        finite = ~_mask(12, [11])
        wells.loc[11, "a"] = np.nan
        notes = []
        out = sp._whiten_profiles(wells, FEATURES, finite, negative, 0.0, notes)
        assert out is not wells
        pd.testing.assert_frame_equal(out.iloc[[-1]], wells.iloc[[-1]])
        assert out["well"].tolist() == wells["well"].tolist()
        controls = out.loc[negative, FEATURES].to_numpy(float)
        np.testing.assert_allclose(np.cov(controls, rowvar=False, ddof=1), np.eye(3),
                                   atol=1e-6)
        assert notes == [
            "TVN: the 3 kept features were whitened (ZCA-cor) on 6 negative-control "
            "wells, keeping the 3 directions they span, epsilon 0."]

    refusals = []
    # one of the two controls has a missing feature -> 1 complete control well
    wells = _wells()
    refusals.append(("one_complete_control_is_too_few", wells, _mask(8, slice(0, 2)),
                     ~_mask(8, [1]),
                     "tvn needs at least 2 negative-control wells with every kept "
                     "feature; 1 have them", False))
    refusals.append(("no_controls", _wells(), _mask(8, []), np.ones(8, bool),
                     "tvn needs at least 2 negative-control wells with every kept "
                     "feature; 0 have them", False))
    constant = _wells()
    constant.loc[_mask(8, slice(0, 3)), FEATURES] = 1.0
    refusals.append(("constant_controls", constant, _mask(8, slice(0, 3)),
                     np.ones(8, bool), "constant in the reference rows", True))
    for case_id, frame, negative, finite, message, chained in refusals:
        with subtests.test(msg=case_id):
            notes = []
            with pytest.raises(sp._ProfilingError) as raised:
                sp._whiten_profiles(frame, FEATURES, finite, negative, 0.0, notes)
            assert message in str(raised.value)
            assert isinstance(raised.value, ValueError)
            assert isinstance(raised.value.__cause__, ValueError) is chained
            assert notes == []
