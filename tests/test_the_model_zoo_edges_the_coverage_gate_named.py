"""The model zoo paths the coverage gate named as uncovered on 2026-10-11.

Run 38151063008 reported spacr/model_zoo.py's uncovered statements rising
from 4 to 6 and its branches from 1 to 4. The rise was real, not data the
lost coverage file took with it: the same lines are uncovered when the zoo's
own tests run alone. N685 added the strict repository reader and the
explicit ``alternative`` field; N691 added MitoNet, whose input scaling,
weights cache and tiling had untested edges. Each test here drives one of
those edges and checks what it returns.
"""
from __future__ import annotations

import hashlib
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from spacr import model_zoo as mz


@pytest.mark.parametrize("uri", [
    "https://[huggingface.co/einarolafsson/models/resolve/main/w",
    "https://[::1/einarolafsson/models",
])
def test_a_url_that_does_not_parse_belongs_to_no_repository(uri):
    assert mz.hf_repo_of(uri) == ""


@pytest.mark.parametrize("uri", [
    "https://huggingface.co/-einarolafsson/models/resolve/main/w",
    "https://huggingface.co/einarolafsson/mo dels/resolve/main/w",
    "https://huggingface.co/datasets/einarolafsson/.models/resolve/main/w",
])
def test_an_owner_or_repo_name_hugging_face_would_refuse_is_nobodys(uri):
    assert mz.hf_repo_of(uri) == ""


def test_an_explicit_alternative_field_wins_over_the_display_name():
    assert mz._record_is_alternative(
        {"alternative": True, "display_name": "PV v5"}) is True
    assert mz._record_is_alternative(
        {"alternative": False, "display_name": "PV v5 (alternative)"}) is False
    assert mz._record_is_alternative(
        {"display_name": "PV v5 (alternative)"}) is True
    assert mz._record_is_alternative({"display_name": "PV v4"}) is False


def test_a_weights_file_that_cannot_be_measured_is_kept_as_zero(
        tmp_path, monkeypatch):
    def refuse(uri, timeout):
        raise OSError("HEAD refused")

    monkeypatch.setattr(mz, "_weights_size", refuse)
    monkeypatch.setattr(mz, "_bioimageio_cache", lambda name: tmp_path / name)

    sizes = mz._measure_bioimageio_weights(
        ["https://example.org/w.pt"], {"https://example.org/old.pt": 7}, 1.0)

    assert sizes == {"https://example.org/w.pt": 0,
                     "https://example.org/old.pt": 7}
    assert json.loads((tmp_path / "bioimageio_weight_sizes.json").read_text(encoding="utf-8")
                      ) == sizes


def test_a_shared_catalogue_row_that_is_not_a_mapping_is_skipped(
        monkeypatch, caplog):
    good = dict(next(record for record in mz.BUNDLED_REMOTE_MODELS
                     if record.get("key") == mz.MITONET_KEY),
                key="shared_row", source="spaCR")
    payload = json.dumps({"models": ["not a row", good]}).encode("utf-8")

    class Response(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda target, timeout: Response(payload))
    monkeypatch.setattr(mz, "_SHARED_CATALOGUE_CACHE",
                        {"fetched_at": 0.0, "entries": ()})

    with caplog.at_level("WARNING", logger=mz.LOG.name):
        entries = mz.shared_catalogue("https://example.org/catalogue.json",
                                      force=True, block=True)

    assert [entry.key for entry in entries] == ["shared_row"]
    assert entries[0].source == "shared"
    assert "skipping a shared catalogue entry" in caplog.text


def test_a_scorecard_with_a_finetuned_number_and_no_stock_number():
    entry = SimpleNamespace(
        metrics={"f1": {"finetuned": 0.91234, "vanilla": None}},
        display_name="demo", trained_on="", model_card_url="")

    html = mz.scorecard_html(entry)

    assert "<b>0.9123</b>" in html
    assert "not recorded" in html


def test_a_missing_weights_file_is_not_verified(tmp_path):
    assert mz._verified_once(tmp_path / "absent.pth", "0" * 64) is False


def test_a_file_with_other_bytes_is_not_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(mz, "_MITONET_VERIFIED", {})
    path = tmp_path / "weights.pth"
    path.write_bytes(b"other bytes")

    assert mz._verified_once(path, hashlib.sha256(b"x").hexdigest()) is False
    assert mz._MITONET_VERIFIED == {}
    assert mz._verified_once(
        path, hashlib.sha256(b"other bytes").hexdigest()) is True


def test_a_versioned_copy_beside_the_cache_is_found(tmp_path, monkeypatch):
    payload = b"versioned weights"
    record = dict(mz._mitonet_record(),
                  sha256=hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(mz, "_mitonet_record", lambda: record)
    monkeypatch.setattr(mz, "_MITONET_VERIFIED", {})
    copy = tmp_path / "MitoNet_v1_mini_v2.pth"
    copy.write_bytes(payload)

    assert mz.mitonet_weights(dest=tmp_path, download=False) == copy


def test_a_colour_image_is_averaged_to_one_plane():
    rgb = np.zeros((4, 5, 3), np.uint8)
    rgb[..., 0] = 255
    plane = mz._mitonet_input(rgb)
    assert plane.shape == (4, 5) and plane.dtype == np.float32
    np.testing.assert_allclose(plane, 85 / 255.0, rtol=1e-6)

    floats = np.ones((4, 5, 3), np.float32)
    floats[..., 0] = 0.0
    np.testing.assert_allclose(mz._mitonet_input(floats), 2 / 3, rtol=1e-6)


def test_a_stack_is_refused_not_guessed():
    with pytest.raises(ValueError, match="one 2-D image"):
        mz._mitonet_input(np.zeros((5, 6, 7), np.float32))


def test_an_image_with_nothing_finite_is_a_blank_plane():
    plane = mz._mitonet_input(np.full((3, 3), np.nan, np.float32))
    assert plane.shape == (3, 3) and not plane.any()


def test_a_flat_image_is_a_blank_plane():
    plane = mz._mitonet_input(np.full((3, 3), 1000, np.uint16))
    assert not plane.any()


def test_a_sixteen_bit_image_is_stretched_between_its_percentiles():
    image = np.arange(400, dtype=np.uint16).reshape(20, 20) * 100
    image[0, 0] = np.uint16(65535)
    plane = mz._mitonet_input(image.astype(np.float64))
    assert plane.dtype == np.float32
    assert plane.min() == 0.0 and plane.max() == 1.0
    assert 0.0 < plane[10, 10] < 1.0


def _two_class_stand_in(tmp_path):
    """A TorchScript module whose semantic head has two classes."""
    torch = pytest.importorskip("torch")

    class Net(torch.nn.Module):
        def forward(self, x):
            dark = (x < 0).float()
            logits = torch.cat([(0.5 - dark) * 20.0, (dark - 0.5) * 20.0], 1)
            return {"sem_logits": logits,
                    "ctr_hmp": torch.nn.functional.avg_pool2d(dark, 9, 1, 4),
                    "offsets": torch.zeros(x.shape[0], 2, x.shape[2],
                                           x.shape[3])}

    path = tmp_path / "two_class.pt"
    torch.jit.script(Net()).save(str(path))
    return path


def test_a_two_class_head_on_the_default_device_finds_the_dark_object(
        tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    image = np.full((128, 128), 220, np.uint8)
    image[30:70, 30:70] = 20
    seen = []

    labels = mz.segment_mitonet(image, weights=_two_class_stand_in(tmp_path),
                                min_size=1, on_tile=lambda d, t:
                                seen.append((d, t)))

    np.testing.assert_array_equal(labels > 0, image < 128)
    assert seen == [(1, 1)]


def test_tiles_without_a_progress_callback(tmp_path):
    pytest.importorskip("torch")
    image = np.full((260, 140), 220, np.uint8)
    image[20:60, 20:60] = 20

    labels = mz.segment_mitonet(image, weights=_two_class_stand_in(tmp_path),
                                device="cpu", min_size=1, tile=128, halo=16)

    np.testing.assert_array_equal(labels > 0, image < 128)


def test_a_scorecard_with_a_text_number_and_a_float_stock_number():
    entry = SimpleNamespace(
        metrics={"f1": {"finetuned": "0.9 (n=3)", "vanilla": 0.81234}},
        display_name="demo", trained_on="", model_card_url="")

    html = mz.scorecard_html(entry)

    assert "<b>0.9 (n=3)</b>" in html
    assert "0.8123" in html


def test_weights_asked_for_in_a_folder_that_does_not_exist(tmp_path):
    folder = tmp_path / "not-made-yet"
    with pytest.raises(mz.ModelUnreadable, match="not in"):
        mz.mitonet_weights(dest=folder, download=False)
    assert not folder.exists()


@pytest.mark.parametrize("payload", ["{broken", "[]", '{"preprocess": []}'])
def test_invalid_local_preprocessing_sidecar_retains_declared_model_fallback(tmp_path, payload):
    from pathlib import Path
    path = tmp_path / "model.bin"
    path.write_bytes(b"local checkpoint")
    sidecar = Path(str(path) + ".preprocess.json")
    sidecar.write_text(payload)
    declaration = {"normalization": "percentile", "channels": [0, 1]}
    records = [{"name": "model.bin", "preprocess": declaration}]
    result = mz._model_preprocess(str(path), records=records)
    assert result == declaration
    result["channels"].append(9)
    assert declaration["channels"] == [0, 1]
    assert sidecar.read_text() == payload
