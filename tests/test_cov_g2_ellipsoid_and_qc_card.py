"""Ellipsoid gate edges and the image-QC classifier card's report reading."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import gate_spec as gs  # noqa: E402
from spacr.qt.widgets import qc_summary as qs  # noqa: E402


def _ellipsoid(**extra):
    values = dict(name="e", x_column="a", y_column="b", z_column="c",
                  x_centre=1.0, y_centre=2.0, z_centre=0.0,
                  x_radius=-2.0, y_radius=2.0, z_radius=2.0)
    values.update(extra)
    return gs._EllipsoidGate(**values)


def test_an_ellipsoid_needs_three_distinct_columns():
    with pytest.raises(gs.GateError, match="has no y_column"):
        _ellipsoid(y_column=" ")
    with pytest.raises(gs.GateError, match="same measurement twice"):
        _ellipsoid(z_column="a")


def test_an_ellipsoid_keeps_radii_positive_and_moves_and_scales():
    gate = _ellipsoid()
    assert gate.kind == gs._ELLIPSOID and gate.x_radius == 2.0
    moved = gate.translated(1, -1)
    assert moved.centre() == (2.0, 1.0)
    grown = gate.scaled(2.0)
    assert (grown.x_radius, grown.z_radius) == (4.0, 4.0)
    assert grown.centre() == gate.centre()
    anchored = gate.scaled(2.0, about=(0.0, 0.0))
    assert anchored.centre() == (2.0, 4.0)


def test_a_flat_ellipsoid_holds_no_rows():
    frame = pd.DataFrame({"a": [1.0], "b": [2.0], "c": [0.0]})
    assert _ellipsoid().mask(frame).tolist() == [True]
    assert _ellipsoid(z_radius=0).mask(frame).tolist() == [False]


def _report(tmp_path, body):
    from spacr.image_quality import REPORT

    path = tmp_path / REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body if isinstance(body, str) else json.dumps(body))
    return path


def _scored():
    return {"policy": {"image_qc_classifier_threshold": 0.5},
            "fields": [{"channels": [{"p_debris": 0.9}, {"p_debris": 0.1}]}]}


def test_a_broken_report_gives_no_card(tmp_path):
    _report(tmp_path, "{not json")
    assert qs._read_image_qc_classifier(str(tmp_path)) is None
    _report(tmp_path, {"fields": []})
    assert qs._read_image_qc_classifier(str(tmp_path)) is None


def test_the_benchmark_rows_are_listed(tmp_path):
    path = _report(tmp_path, _scored())
    pd.DataFrame({"class": ["debris"], "precision": [0.75]}).to_csv(
        path.parent / "image_qc_benchmark.csv", index=False)
    card = qs._read_image_qc_classifier(str(tmp_path))
    assert card.verdict == "warn"
    assert any("precision 0.75" in line for line in card.detail)


def test_an_unreadable_benchmark_is_named(tmp_path, monkeypatch):
    import spacr.tabular as tabular

    path = _report(tmp_path, _scored())
    (path.parent / "image_qc_benchmark.csv").write_text("x\n1\n")

    def broken(*a, **k):
        raise ValueError("bad table")

    monkeypatch.setattr(tabular, "read_table", broken)
    card = qs._read_image_qc_classifier(str(tmp_path))
    assert any("bad table" in line for line in card.detail)
    assert np.isfinite(card.mtime)
