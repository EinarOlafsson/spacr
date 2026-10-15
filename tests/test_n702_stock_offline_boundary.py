"""Offline stock weights refuse before constructor; existing files remain usable."""
from pathlib import Path

import pytest

from spacr import model_zoo as zoo
from spacr import utils as utilities
from tests.test_n702_unknown_cellpose_model_refusal import isolated as isolated


@pytest.fixture
def native_cache(isolated, tmp_path, monkeypatch):
    cache = tmp_path / "configured-native-cache"
    monkeypatch.setattr(utilities.cp_models, "MODEL_DIR", cache)
    monkeypatch.setattr(zoo, "_ensure_model_file", lambda *args, **kwargs: None)
    return cache


@pytest.mark.parametrize("offline", ["0", "off", "FALSE", " No "])
@pytest.mark.parametrize("entry_kind", ["missing", "directory"])
def test_uncached_stock_refuses_before_constructor(native_cache, monkeypatch, offline, entry_kind):
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", offline)
    if entry_kind == "directory":
        (native_cache / "cpsam").mkdir(parents=True)
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs: pytest.fail("Uncached offline stock reached constructor"))
    with pytest.raises(zoo.ModelUnreadable) as error:
        utilities._choose_model("cpsam", device="cpu", object_type="cell")
    assert "SPACR_MODEL_DOWNLOAD" in str(error.value)
    assert str(native_cache / "cpsam") in str(error.value)
    assert "'cpsam'" in str(error.value)
    assert not (native_cache / "cpsam").is_file()


@pytest.mark.parametrize("requested", ["cpsam", None, "cyto3", "sam"])
def test_existing_native_cache_retains_resolved_stock(native_cache, monkeypatch, requested):
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "off")
    native_cache.mkdir()
    checkpoint = native_cache / "cpsam"
    checkpoint.write_bytes(b"private cached checkpoint; never decoded")
    before = checkpoint.read_bytes()
    sentinel = object()
    seen = []
    def construct(**kwargs):
        seen.append(kwargs)
        return sentinel
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", construct)
    assert utilities._choose_model(requested, device="cpu", object_type="cell") is sentinel
    assert seen[0]["pretrained_model"] == "cpsam"
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("requested", ["cpsam", "sam", "explicit.CP_model"])
def test_existing_relative_checkpoint_wins_over_empty_native_cache(native_cache, monkeypatch, requested):
    monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", "0")
    checkpoint = Path(requested)
    checkpoint.write_bytes(b"private relative checkpoint; never decoded")
    before = checkpoint.read_bytes()
    seen = []
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs: seen.append(kwargs) or seen)
    assert utilities._choose_model(requested, device="cpu", object_type="cell") is seen
    assert seen[0]["pretrained_model"] == requested
    assert checkpoint.read_bytes() == before
    assert not native_cache.exists()


@pytest.mark.parametrize("online", [None, "1", "on"])
def test_online_missing_stock_preserves_native_constructor_route(native_cache, monkeypatch, online):
    if online is None:
        monkeypatch.delenv("SPACR_MODEL_DOWNLOAD", raising=False)
    else:
        monkeypatch.setenv("SPACR_MODEL_DOWNLOAD", online)
    seen = []
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs: seen.append(kwargs) or seen)
    assert utilities._choose_model("cpsam", device="cpu", object_type="cell") is seen
    assert seen[0]["pretrained_model"] == "cpsam"
    assert not native_cache.exists()
