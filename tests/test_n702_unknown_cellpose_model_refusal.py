"""Unknown models refuse stock substitution; explicit existing identities win."""
from types import SimpleNamespace

import pytest

from spacr import model_zoo as zoo
from spacr import utils as utilities


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(utilities, "cp_models", SimpleNamespace(
        MODEL_DIR=tmp_path / "native-cellpose-cache",
        CellposeModel=lambda **kwargs: pytest.fail("Unconfigured model reached constructor")))
    monkeypatch.setattr(zoo, "catalogue", lambda **kwargs: [])
    monkeypatch.setattr(zoo, "fetch", lambda *args, **kwargs: pytest.fail("No weights may be fetched"))
    monkeypatch.setattr(utilities, "_installed_cellpose_models", lambda: ("cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"))


@pytest.mark.parametrize("requested", ["spacr_unpublished_checkpoint_731", "spacr_unpublished_checkpoint_731.CP_model", r"C:\Users\lab\spacr_unpublished_checkpoint_731.CP_model"])
def test_unknown_requested_models_refuse_before_model_construction(isolated, monkeypatch, requested):
    monkeypatch.setattr(utilities.cp_models, "CellposeModel", lambda **kwargs: pytest.fail("Unknown model reached constructor"))
    with pytest.raises(zoo.ModelZooError) as error:
        utilities._choose_model(requested, device="cpu", object_type="cell")
    assert repr(requested) in str(error.value)


@pytest.mark.parametrize("requested", [None, "", "  ", "sam", *utilities.LEGACY_CELLPOSE_MODELS])
def test_defaults_legacy_names_and_unmatched_sam_keep_the_default(isolated, requested):
    assert utilities._resolve_cellpose_pretrained(requested, object_type="cell") == utilities.CPSAM_MODEL


@pytest.mark.parametrize("requested", ["cpsam", "cpsam_v2", "cpdino", "cpdino-vitb"])
def test_installed_stock_names_retain_exact_identity(isolated, requested):
    assert utilities._resolve_cellpose_pretrained(requested, object_type="cell") == requested


def test_existing_checkpoint_named_sam_wins_over_the_shorthand(isolated, tmp_path, monkeypatch):
    checkpoint = tmp_path / "sam"
    checkpoint.write_bytes(b"controlled local checkpoint; never decoded")
    before = checkpoint.read_bytes()
    monkeypatch.setattr(zoo, "catalogue", lambda **kwargs: pytest.fail("Existing file queried catalogue"))
    assert utilities._resolve_cellpose_pretrained("sam", object_type="cell") == "sam"
    assert checkpoint.read_bytes() == before


def test_exact_catalogue_key_sam_wins_over_the_shorthand(isolated, tmp_path, monkeypatch):
    checkpoint = tmp_path / "registered-sam.CP_model"
    checkpoint.write_bytes(b"controlled catalogue checkpoint; never decoded")
    before = checkpoint.read_bytes()
    entry = zoo.ModelEntry(key="sam", name=checkpoint.name, kind="cellpose", path=str(checkpoint))
    monkeypatch.setattr(zoo, "catalogue", lambda **kwargs: [entry])
    assert utilities._resolve_cellpose_pretrained("sam", object_type="cell") == str(checkpoint)
    assert checkpoint.read_bytes() == before
