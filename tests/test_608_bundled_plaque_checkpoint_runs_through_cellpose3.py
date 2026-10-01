"""Item 608: the bundled Cellpose 3 plaque checkpoint no longer breaks Plaque.

Found by spacr-d7 while re-recording tutorials, 2026-10-01: the bundled
plaque checkpoint is a Cellpose 3 model and Cellpose 4 refuses to load it.
The default stays the Cellpose-SAM zoo model, the old checkpoint segments
through the isolated Cellpose 3 backend when that is installed, and without
it the refusal is a sentence saying what to do. Cellpose itself is faked.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from spacr import _segmentation_backends as SB
from spacr import plaque_papers as PP
from spacr import submodules as SUB

REFUSAL = ("This model does not appear to be a CP4 model. CP3 models are "
           "not compatible with CP4.")


def _refuse(**_kw):
    raise ValueError(REFUSAL)


class _Backend:
    """Stands in for the Cellpose 3 worker; records what it was sent."""

    def __init__(self, name, *, model="", **_kw):
        self.name, self.model, self.sent = name, model, []
        self.note = "in its own environment"

    def eval(self, x, batch_size=None, channel_axis=-1, normalize=True,
             diameter=None, flow_threshold=None, cellprob_threshold=0.0,
             min_size=None, resample=None, progress=None,
             should_cancel=None, augment=None):
        """The ``_RemoteBackend.eval`` signature; records what was asked."""
        params = {"diameter": diameter, "flow_threshold": flow_threshold,
                  "cellprob_threshold": cellprob_threshold}
        self.sent.append(([np.asarray(i) for i in x], params))
        labels = [np.ones(np.asarray(i).shape[:2], dtype=np.uint16) for i in x]
        flows = [[np.zeros(i.shape[:2] + (3,), np.uint8), None,
                  np.zeros(i.shape[:2], np.float32), None] for i in x]
        return labels, flows, None


@pytest.fixture
def cellpose3(monkeypatch):
    """The Cellpose 3 backend, installed or not, as the test says."""
    state = {"ready": True}
    monkeypatch.setattr(SB, "_backend_state", lambda name, root=None:
                        SimpleNamespace(ready=state["ready"],
                                        in_process=False))
    monkeypatch.setattr(SB, "_RemoteBackend", _Backend)
    monkeypatch.setattr(SUB.cp_models, "CellposeModel", _refuse)
    return state


def test_the_default_is_a_cellpose_sam_zoo_model_not_the_bundled_one():
    from spacr.model_zoo import catalogue
    from spacr.settings import get_analyze_plaque_settings

    assert SUB.DEFAULT_PLAQUE_MODEL == "toxoplasma_plaque_v2"
    assert get_analyze_plaque_settings({})["plaque_model"] == "toxoplasma_plaque_v2"
    assert SUB._requested_plaque_model({}) == "toxoplasma_plaque_v2"
    assert SUB._requested_plaque_model({"plaque_model": ""}) == "toxoplasma_plaque_v2"
    entry = next(e for e in catalogue(remote=True)
                 if e.key == SUB.DEFAULT_PLAQUE_MODEL)
    assert entry.kind == "cellpose"
    assert "cpsam" in str(entry.name)


def test_the_bundled_checkpoint_segments_through_the_cellpose3_backend(
        cellpose3):
    model = SUB._plaque_cellpose_model("/m/old.CP_model")
    assert isinstance(model, SUB._Cellpose3PlaqueModel)
    backend = model._backend
    assert backend.name == SB._CELLPOSE3
    assert backend.model == "/m/old.CP_model"
    rgb = np.zeros((20, 30, 3), np.uint8)
    rgb[..., 0] = 30
    labels, flows, styles = model.eval(rgb, channel_axis=-1, diameter=40,
                                       flow_threshold=0.5,
                                       cellprob_threshold=-1.0)
    (planes, params), = backend.sent
    assert [p.shape for p in planes] == [(20, 30)]
    assert planes[0].max() == pytest.approx(10.0)
    assert params == {"diameter": 40, "flow_threshold": 0.5,
                      "cellprob_threshold": -1.0}
    assert labels.shape == (20, 30) and styles is None
    assert len(flows) == 4


def test_the_run_reads_masks_and_flows_from_the_backend(cellpose3):
    from spacr.plaque import segment_plaque_image

    model = SUB._plaque_cellpose_model("/m/old.CP_model")
    labels, flows = segment_plaque_image(
        model, np.zeros((16, 16, 3), np.uint8), {"diameter": 0},
        return_flows=True)
    assert labels.shape == (16, 16)
    assert flows["flow_rgb"].shape == (16, 16, 3)
    assert flows["cellprob"].shape == (16, 16)
    assert model._backend.sent[0][1]["diameter"] is None


def test_without_the_backend_the_refusal_says_what_to_do(cellpose3):
    cellpose3["ready"] = False
    with pytest.raises(SUB.Cellpose3Checkpoint) as excinfo:
        SUB._plaque_cellpose_model("/m/old.CP_model")
    text = str(excinfo.value)
    assert "toxoplasma_plaque_v2" in text
    assert "Cellpose 3 backend" in text
    assert isinstance(excinfo.value.__cause__, ValueError)


def test_figure_mode_segments_the_old_checkpoint_the_same_way(cellpose3,
                                                             monkeypatch):
    import cellpose.models

    monkeypatch.setattr(cellpose.models, "CellposeModel", _refuse)
    segment = PP._cellpose_segmenter("/m/old.CP_model")
    assert segment(np.zeros((12, 14, 3), np.uint8)).shape == (12, 14)
    cellpose3["ready"] = False
    with pytest.raises(SUB.Cellpose3Checkpoint):
        PP._cellpose_segmenter("/m/old.CP_model")


def test_another_load_error_is_not_mistaken_for_cellpose3(cellpose3,
                                                          monkeypatch):
    def broken(**_kw):
        raise ValueError("state dict mismatch")

    monkeypatch.setattr(SUB.cp_models, "CellposeModel", broken)
    with pytest.raises(ValueError, match="state dict mismatch"):
        SUB._plaque_cellpose_model("/m/other.pth")
