"""333: a downloaded zoo model is offered where the preview picks its model.

The live preview built its Cellpose model from whatever the combo held, and
the combo held Cellpose's stock list. So a user who had selected a zoo model
for the RUN was shown a PREVIEW made with stock cpsam -- while tuning diameter
and thresholds against it.

That is the worst shape this defect could take: the preview does not fail, it
quietly answers a different question than the one being asked, and the answer
looks authoritative.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from spacr import settings as S


class _Entry(SimpleNamespace):
    pass


def test_a_downloaded_cellpose_zoo_model_is_offered(monkeypatch, tmp_path):
    checkpoint = tmp_path / "cpsam_v2_toxo_r2"
    checkpoint.write_bytes(b"weights")

    from spacr import model_zoo
    monkeypatch.setattr(model_zoo, "catalogue", lambda **k: [
        _Entry(kind="cellpose", path=str(checkpoint), key="toxoplasma_pv_v1",
               name="cpsam_v2_toxo_r2"),
    ])
    assert str(checkpoint) in S.downloaded_zoo_models()
    assert str(checkpoint) in S.cellpose_model_menu()


def test_a_model_not_yet_downloaded_is_not_offered(monkeypatch):
    """A dropdown entry that cannot be chosen without starting a 1.2 GB
    download is not a choice; it is a trap. The picker is where downloading
    happens."""
    from spacr import model_zoo
    monkeypatch.setattr(model_zoo, "catalogue", lambda **k: [
        _Entry(kind="cellpose", path="", key="toxoplasma_plaque_v1",
               name="cpsam_plaque_r3"),
    ])
    assert S.downloaded_zoo_models() == ()


def test_a_detector_is_not_offered_as_a_cellpose_model(monkeypatch, tmp_path):
    """The zoo carries the YOLO well detector. CellposeModel cannot load it,
    so offering it here produces a preview that fails on selection."""
    weights = tmp_path / "yolo_welldetect_v3.pt"
    weights.write_bytes(b"weights")

    from spacr import model_zoo
    monkeypatch.setattr(model_zoo, "catalogue", lambda **k: [
        _Entry(kind="detector", path=str(weights),
               key="toxoplasma_well_detector_v1", name="yolo_welldetect_v3.pt"),
    ])
    assert S.downloaded_zoo_models() == ()


def test_an_unreachable_catalogue_yields_an_empty_menu_addition(monkeypatch):
    """A dropdown that could not be built because a catalogue was unreachable
    would be worse than one missing an entry."""
    from spacr import model_zoo

    def boom(**kwargs):
        raise RuntimeError("no network")

    monkeypatch.setattr(model_zoo, "catalogue", boom)
    assert S.downloaded_zoo_models() == ()
    assert "cpsam" in S.cellpose_model_menu(), "the stock menu must survive"
