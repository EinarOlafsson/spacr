"""Direct owners for the current-source Activation-search branches."""

from io import BytesIO
from types import SimpleNamespace
import tarfile

import numpy as np
from PIL import Image
import pytest

from spacr import hyperparam as hp


def test_activation_parameter_spellings_reach_the_attribution_backend():
    method, kwargs, samples, sigma = hp._activation_params({
        "cam_type": "saliency_image",
        "target_layer": "features.2",
        "ig_steps": 17,
        "ig_baseline": "mean",
        "occlusion_window": 9,
        "occlusion_stride": 3,
        "smoothgrad_samples": 5,
        "smoothgrad_sigma": 0.2,
    })

    assert method == "saliency"
    assert kwargs == {
        "layer": "features.2",
        "n_steps": 17,
        "baseline": "mean",
        "window": 9,
        "stride": 3,
    }
    assert samples == 5
    assert sigma == pytest.approx(0.2)


def _cheap_activation_backends(monkeypatch):
    from spacr import attribution

    monkeypatch.setattr(
        attribution, "deletion_curve",
        lambda *_args, **_kwargs: SimpleNamespace(auc=0.25))
    monkeypatch.setattr(
        attribution, "insertion_curve",
        lambda *_args, **_kwargs: SimpleNamespace(auc=0.75))
    monkeypatch.setattr(
        attribution, "pointing_game_rate",
        lambda maps, masks: {"rate": 1.0, "hits": len(maps), "n": len(masks)})


def test_activation_fit_reports_masks_and_randomisation(monkeypatch):
    from spacr import attribution

    _cheap_activation_backends(monkeypatch)
    seen = {}

    def sanity(model, image, method, **kwargs):
        seen.update(method=method, kwargs=kwargs)
        return SimpleNamespace(
            gap=0.8, final_similarity=0.1, passed=True,
            verdict=lambda: "passed")

    monkeypatch.setattr(attribution, "randomization_sanity_check", sanity)
    attributed = SimpleNamespace(
        map=np.ones((4, 4), dtype=float), is_flat=lambda: False)
    data = hp.ActivationSearchData(
        model=object(), images=[object()], masks=[np.ones((4, 4), dtype=bool)],
        model_type="tiny")
    fit = hp.activation_fit_fn(
        data, criterion="pointing_game", attribute_fn=lambda *_args: attributed)

    score, details = fit({
        "cam_type": "saliency_channel", "target_layer": "features.2",
        "ig_steps": 7,
    })

    assert score == pytest.approx(1.0)
    assert details["pointing_hits"] == details["pointing_scored"] == 1
    assert details["sanity_passed"] is True
    assert details["sanity_verdict"] == "passed"
    assert seen["method"] == "saliency"
    assert seen["kwargs"]["layer"] == "features.2"
    assert seen["kwargs"]["n_steps"] == 7


@pytest.mark.parametrize(
    ("criterion", "message"),
    [("pointing_game", "no object masks"),
     ("sanity_gap", "sanity check was disabled")],
)
def test_activation_fit_explains_an_unavailable_criterion(
        monkeypatch, criterion, message):
    _cheap_activation_backends(monkeypatch)
    attributed = SimpleNamespace(
        map=np.ones((2, 2), dtype=float), is_flat=lambda: False)
    data = hp.ActivationSearchData(model=object(), images=[object()])
    fit = hp.activation_fit_fn(
        data, criterion=criterion, run_sanity_check=False,
        attribute_fn=lambda *_args: attributed)

    with pytest.raises(ValueError, match=message):
        fit({"cam_type": "saliency"})


def test_activation_grid_search_keeps_masks_and_sanity_in_its_notes(monkeypatch):
    data = hp.ActivationSearchData(
        model=object(), images=[object()], masks=[np.ones((2, 2), dtype=bool)])
    captured = {}
    marker = hp.SearchResult(metric="deletion_auc")

    def grid(fit, space, **kwargs):
        captured.update(fit=fit, space=space, kwargs=kwargs)
        return marker

    monkeypatch.setattr(hp, "grid_search", grid)
    space = hp.SearchSpace({"cam_type": ["saliency"]})

    result = hp.activation_search(data, space, mode="grid",
                                  run_sanity_check=True)

    assert result is marker
    notes = " ".join(captured["kwargs"]["notes"])
    assert "Object masks were available" in notes
    assert "sanity check was skipped" not in notes


class _Model:
    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def _pin_activation_loader_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    from spacr import accelerator

    monkeypatch.setattr(accelerator, "torch_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: _Model())
    return torch


def test_activation_loader_requires_a_real_model_file(tmp_path):
    with pytest.raises(ValueError, match="No trained model"):
        hp.load_activation_data({"model_path": str(tmp_path / "missing.pth")})


def test_activation_loader_requires_a_source_after_loading_the_model(
        tmp_path, monkeypatch):
    _pin_activation_loader_to_cpu(monkeypatch)
    model = tmp_path / "model.pth"
    model.write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="Nothing to attribute"):
        hp.load_activation_data({
            "model_path": str(model), "src": str(tmp_path / "empty"),
            "dataset": str(tmp_path / "missing.tar"),
        })


def test_activation_loader_uses_only_merged_arrays_with_a_nonempty_mask(
        tmp_path, monkeypatch):
    torch = _pin_activation_loader_to_cpu(monkeypatch)
    model = tmp_path / "model.pth"
    model.write_bytes(b"placeholder")
    merged = tmp_path / "experiment" / "merged"
    merged.mkdir(parents=True)

    np.save(merged / "a_wrong_shape.npy", np.zeros((8, 8), dtype=np.float32))
    empty = np.zeros((8, 8, 4), dtype=np.float32)
    np.save(merged / "b_empty_mask.npy", empty)
    usable = empty.copy()
    usable[..., 0] = 2.0
    usable[..., 1] = np.arange(64, dtype=float).reshape(8, 8)
    usable[2:6, 2:6, 3] = 4
    np.save(merged / "c_usable.npy", usable)

    data = hp.load_activation_data({
        "model_path": str(model), "src": str(tmp_path / "experiment"),
        "channels": [0, 1, 2], "mask_dims": {"cell": 3},
        "object_type": "cell", "image_size": 6,
    })

    assert data.model.device == torch.device("cpu")
    assert data.filenames == ["c_usable.npy"]
    assert tuple(data.images[0].shape) == (3, 6, 6)
    assert data.masks[0].shape == (6, 6)
    assert data.masks[0].any()
    assert "pointing-game answer key" in " ".join(data.notes)


def test_activation_loader_can_leave_tar_pixels_unnormalised(
        tmp_path, monkeypatch):
    _pin_activation_loader_to_cpu(monkeypatch)
    model = tmp_path / "model.pth"
    model.write_bytes(b"placeholder")

    pixels = np.full((8, 8, 3), 128, dtype=np.uint8)
    payload = BytesIO()
    Image.fromarray(pixels, mode="RGB").save(payload, format="PNG")
    image_bytes = payload.getvalue()
    dataset = tmp_path / "crops.tar"
    with tarfile.open(dataset, "w") as archive:
        member = tarfile.TarInfo("plate_A1_1.png")
        member.size = len(image_bytes)
        archive.addfile(member, BytesIO(image_bytes))

    data = hp.load_activation_data({
        "model_path": str(model), "dataset": str(dataset),
        "src": str(tmp_path / "no-merged"), "channels": [1, 2, 3],
        "image_size": 8, "normalize_input": False,
    }, n_images=1)

    assert data.masks is None
    assert data.filenames == ["plate_A1_1.png"]
    assert float(data.images[0].mean()) == pytest.approx(128 / 255.0)


def test_activation_app_accepts_preloaded_data(monkeypatch):
    data = hp.ActivationSearchData(model=object(), images=[object()])
    marker = hp.SearchResult(metric="deletion_auc")
    seen = {}

    def search(received, space, **kwargs):
        seen.update(data=received, space=space, kwargs=kwargs)
        return marker

    monkeypatch.setattr(hp, "activation_search", search)
    space = hp.SearchSpace({"cam_type": ["saliency"]})

    result = hp.run_search_for_app(
        "activation", {}, space, data=data, mode="grid")

    assert result is marker
    assert seen["data"] is data
