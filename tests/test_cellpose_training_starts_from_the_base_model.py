"""Item 426, step 0's blocker: Cellpose training starts from ``base_model``.

Training handed ``pretrained_model='cpsam'`` to Cellpose whatever was asked,
so a second fine-tuning stage silently restarted from stock weights while its
log said it was continuing.
"""
from __future__ import annotations

import types

import pytest

from spacr import submodules as sm


def test_a_checkpoint_path_is_used_as_is(tmp_path):
    checkpoint = tmp_path / "stage1_model"
    checkpoint.write_bytes(b"w")
    assert sm._resolve_training_base(str(checkpoint)) == str(checkpoint)


def test_a_stock_name_or_nothing_is_passed_through(monkeypatch):
    from spacr import model_zoo

    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [])
    assert sm._resolve_training_base("cpsam") == "cpsam"
    assert sm._resolve_training_base(None) == "cpsam"


def test_a_zoo_key_on_disk_is_used_without_a_download(tmp_path, monkeypatch):
    from spacr import model_zoo

    local = tmp_path / "cpsam_plaque_r5"
    local.write_bytes(b"w")
    entry = types.SimpleNamespace(key="toxoplasma_plaque_v2", path=str(local))
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])
    monkeypatch.setattr(model_zoo, "fetch",
                        lambda *a, **k: pytest.fail("must not download"))
    assert sm._resolve_training_base("toxoplasma_plaque_v2") == str(local)


def test_training_hands_the_base_to_cellpose(tmp_path, monkeypatch):
    checkpoint = tmp_path / "stage1_model"
    checkpoint.write_bytes(b"w")
    seen = {}

    class Stop(Exception):
        pass

    def fake_model(**kw):
        seen.update(kw)
        raise Stop

    monkeypatch.setattr(sm.cp_models, "CellposeModel", fake_model)
    monkeypatch.setattr("spacr.utils.save_settings", lambda *a, **k: None)
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "masks").mkdir(parents=True)
    with pytest.raises(Stop):
        sm.train_cellpose({"src": str(tmp_path), "base_model": str(checkpoint)})
    assert seen["pretrained_model"] == str(checkpoint)


def test_a_stock_zoo_entry_with_nothing_to_download_is_passed_through(monkeypatch):
    """The zoo lists stock cpsam too, with no uri; that is Cellpose's own name,
    not something to fetch (21 training tests failed on the first cut)."""
    from spacr import model_zoo

    entry = types.SimpleNamespace(key="cpsam", path="", uri="")
    monkeypatch.setattr(model_zoo, "catalogue", lambda remote=True: [entry])
    monkeypatch.setattr(model_zoo, "fetch",
                        lambda *a, **k: pytest.fail("must not download"))
    assert sm._resolve_training_base("cpsam") == "cpsam"
