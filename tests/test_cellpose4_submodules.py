"""Cellpose-4 leftovers in submodules.py.

``train_cellpose`` fine-tunes ``cpsam``, but stamped every checkpoint it
wrote with ``_cyto_`` -- the name of the Cellpose-3 model it used to
fine-tune.

No weights are loaded here: ``cellpose.models.CellposeModel`` is replaced
with a recorder so the call shape is what is under test. CPU only, offline.
"""
from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# submodules.train_cellpose -- the checkpoint name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cached_stock", [False, True])
def test_trained_checkpoint_is_named_cpsam_not_cyto(tmp_path, monkeypatch, cached_stock):
    """train_cellpose fine-tunes cpsam, so the filename must say cpsam."""
    import spacr.submodules as submodules
    from spacr import model_zoo
    from types import SimpleNamespace

    cached = tmp_path / "cached-cpsam"
    if cached_stock:
        cached.write_bytes(b"recording model; never loaded")
    entry = SimpleNamespace(key="cpsam", path=str(cached), uri="")
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])
    monkeypatch.setattr(model_zoo, "fetch",
                        lambda *a, **k: pytest.fail("stock weights must not be fetched"))

    seen = {}

    class _Model:
        def __init__(self, **kw):
            seen["model_kwargs"] = kw
            self.net = object()

    def _train_seg(net, **kw):
        seen["train_kwargs"] = kw

    monkeypatch.setattr(submodules.cp_models, "CellposeModel", _Model)
    monkeypatch.setattr(submodules.train_cp, "train_seg", _train_seg)
    monkeypatch.setattr(submodules, "plot_cellpose_batch", lambda *a, **k: None)

    import tifffile
    for sub in ("images", "masks"):
        (tmp_path / "train" / sub).mkdir(parents=True)
    for i in range(2):
        name = f"img{i}.tif"
        tifffile.imwrite(tmp_path / "train" / "images" / name,
                         np.full((32, 32), 100, np.uint16))
        lbl = np.zeros((32, 32), np.uint16)
        lbl[4:12, 4:12] = 1
        tifffile.imwrite(tmp_path / "train" / "masks" / name, lbl)

    submodules.train_cellpose({
        "src": str(tmp_path), "model_name": "mymodel", "n_epochs": 20,
        "target_size": 16, "augment": False, "batch_size": 2,
        "learning_rate": 0.05, "weight_decay": 1e-4,
    })

    name = seen["train_kwargs"]["model_name"]
    assert name == "mymodel_cpsam_e20_X16_Y16.CP_model"
    assert "_cyto_" not in name
    # It really is the SAM checkpoint being fine-tuned.
    assert seen["model_kwargs"]["pretrained_model"] == (
        str(cached) if cached_stock else "cpsam")
    # The settings snapshot is written under the same name.
    assert (tmp_path / "settings" /
            "mymodel_cpsam_e20_X16_Y16.CP_model.csv").exists()


def test_checkpoints_written_under_the_old_name_still_resolve(tmp_path):
    """Nothing parses the infix -- old names keep working.

    model_zoo recognises a Cellpose checkpoint by its ``.CP_model`` suffix,
    so a checkpoint trained before the rename is still discovered, still
    classified as a Cellpose model, and still loadable by path.
    """
    from spacr import model_zoo
    from spacr.utils import _resolve_cellpose_pretrained

    old = tmp_path / "models" / "cellpose_model" / "toxo_cyto_e25_X512_Y512.CP_model"
    new = tmp_path / "models" / "cellpose_model" / "toxo_cpsam_e25_X512_Y512.CP_model"
    old.parent.mkdir(parents=True)
    for p in (old, new):
        p.write_bytes(b"weights")
        assert model_zoo.classify_kind(str(p)) == "cellpose"
        assert _resolve_cellpose_pretrained(str(p)) == str(p)
