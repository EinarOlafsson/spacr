"""What a run says about the hardware it ended up on, and the half-precision
block it runs the forward pass in.

The card is shared. :func:`spacr.deep_spacr.pick_device` answers "can this
run use the GPU", not "does a GPU exist", so a run started while somebody
else holds most of the card silently continues on the CPU -- ten to a hundred
times slower. The note it returns is the ONLY notice a user gets of that, so
every entry point that picks a device has to print it, and print it before it
starts working rather than after the run is already slow.

Two smaller pieces of the same block are covered here for the same reason:
``autocasting`` has to be a context manager whether or not half precision was
asked for, since the training loop has one shape rather than a branch around
every forward pass; and releasing the accelerator cache must not touch a
backend that is not there.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn                       # noqa: E402

import numpy as np                          # noqa: E402
from PIL import Image                       # noqa: E402

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt             # noqa: E402

from spacr import deep_spacr                # noqa: E402


#: What ``pick_device`` says when the card is busy and the run fell back.
_NOTE = ("The GPU has 300 MiB free of 24576 and this run needs about 2000 "
         "MiB, so it is running on the CPU instead.")


class _Stop(Exception):
    """Raised by an injected dependency once the notice has been printed."""


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def fell_back_to_cpu(monkeypatch):
    """Every ``pick_device`` call reports a CPU fallback with a note."""
    monkeypatch.setattr(deep_spacr, "pick_device",
                        lambda *a, **k: (torch.device("cpu"), _NOTE))


@pytest.fixture
def stop_after_the_notice(monkeypatch):
    """Make the first real work after the notice raise :class:`_Stop`."""
    import spacr.normalization as normalization

    def stop(*_args, **_kwargs):
        raise _Stop("stopped once the device notice had been printed")

    monkeypatch.setattr(normalization, "normalization_stats", stop)
    monkeypatch.setattr(deep_spacr, "_load_inference_model", stop)
    return stop


# ---------------------------------------------------------------------------
# autocasting / cache release
# ---------------------------------------------------------------------------

def test_autocasting_on_actually_enables_half_precision():
    """The training loop's ``with`` block has to be the real autocast block."""
    device = torch.device("cpu")

    with deep_spacr.autocasting(True, device):
        inside = torch.is_autocast_enabled("cpu")
        dtype = torch.get_autocast_dtype("cpu")

    assert inside is True
    assert dtype is torch.float16
    assert torch.is_autocast_enabled("cpu") is False


def test_autocasting_off_is_still_a_context_manager():
    """One shape for the loop: off must not mean "no ``with`` block"."""
    with deep_spacr.autocasting(False, torch.device("cpu")):
        assert torch.is_autocast_enabled("cpu") is False


def test_the_cache_release_reaches_cuda_when_cuda_is_there(monkeypatch):
    """A CUDA run has to get its cache back between stages."""
    released = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache",
                        lambda: released.append("cuda"))

    deep_spacr._empty_device_cache()

    assert released == ["cuda"]


def test_the_cache_release_leaves_an_absent_backend_alone(monkeypatch):
    """Asking a backend that is not there is how a CPU run crashes."""
    def must_not_run():
        raise AssertionError("empty_cache was called with no CUDA present")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", must_not_run)

    assert deep_spacr._empty_device_cache() is None


# ---------------------------------------------------------------------------
# the notice, per entry point
# ---------------------------------------------------------------------------

def _png(path, seed=0, size=32):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint16).astype(np.uint8)
    Image.fromarray(arr).save(str(path))
    return str(path)


def test_apply_model_says_it_fell_back_before_it_scores_anything(
        tmp_path, capsys, fell_back_to_cpu, stop_after_the_notice):
    """Inference on the CPU is slow; the user learns that at the start."""
    src = tmp_path / "crops"
    src.mkdir()
    _png(src / "plate1_A01_1_0.png")

    with pytest.raises(_Stop):
        deep_spacr.apply_model(str(src), str(tmp_path / "model.pth"),
                               image_size=32)

    assert _NOTE in capsys.readouterr().out


def test_apply_model_to_tar_says_it_fell_back_before_it_scores_anything(
        tmp_path, capsys, fell_back_to_cpu, stop_after_the_notice):
    """The tar entry point picks a device of its own and owes the same notice."""
    settings = {"tar_path": str(tmp_path / "ds.tar"),
                "model_path": str(tmp_path / "model.pth"),
                "normalize": True, "image_size": 32, "batch_size": 2,
                "channels": [1, 2, 3], "n_jobs": 0, "verbose": False,
                "score_threshold": 0.5}

    with pytest.raises(_Stop):
        deep_spacr.apply_model_to_tar(settings)

    assert _NOTE in capsys.readouterr().out


def test_evaluating_a_stage_says_where_it_is_running(capsys, fell_back_to_cpu):
    """Validation on the CPU is the usual reason an epoch takes minutes."""
    class _Refuses(nn.Module):
        def eval(self):
            raise _Stop("stopped once the device notice had been printed")

    with pytest.raises(_Stop):
        deep_spacr.evaluate_model_performance(_Refuses(), [], epoch=1)

    assert _NOTE in capsys.readouterr().out


def test_the_test_stage_says_where_it_is_running(capsys, fell_back_to_cpu):
    """Same notice from the held-out pass, which runs on its own call."""
    class _Refuses(nn.Module):
        def eval(self):
            raise _Stop("stopped once the device notice had been printed")

    with pytest.raises(_Stop):
        deep_spacr.test_model_core(_Refuses(), [], "test", 1, "auto")

    assert _NOTE in capsys.readouterr().out


def test_integrated_gradients_says_where_it_is_running(
        tmp_path, capsys, fell_back_to_cpu, stop_after_the_notice):
    """Attribution maps are the slowest thing here to run on a CPU."""
    src = tmp_path / "pngs"
    src.mkdir()
    _png(src / "only.png", seed=7)

    with pytest.raises(_Stop):
        deep_spacr.visualize_integrated_gradients(
            str(src), str(tmp_path / "model.pth"), image_size=32)

    assert _NOTE in capsys.readouterr().out


def test_smooth_grad_says_where_it_is_running(
        tmp_path, capsys, fell_back_to_cpu, stop_after_the_notice):
    """The second attribution driver picks its own device, and reports it."""
    src = tmp_path / "pngs"
    src.mkdir()
    _png(src / "only.png", seed=9)

    with pytest.raises(_Stop):
        deep_spacr.visualize_smooth_grad(
            str(src), str(tmp_path / "model.pth"), 0, image_size=32)

    assert _NOTE in capsys.readouterr().out


def test_the_activation_map_run_says_where_it_is_running(
        tmp_path, capsys, fell_back_to_cpu):
    """The notice comes before the dataset is even looked for."""
    root = tmp_path / "proj"
    (root / "datasets").mkdir(parents=True)
    settings = {"dataset": str(root / "datasets" / "nope.tar"),
                "model_path": str(root / "no_such_model.pth"),
                "model_type": "resnet18", "cam_type": "saliency_image",
                "target_layer": None, "image_size": 32, "batch_size": 2,
                "channels": [1, 2, 3], "normalize": False, "save": False,
                "plot": False, "correlation": False, "overlay": True,
                "shuffle": False, "n_jobs": 0}

    assert deep_spacr.generate_activation_map(settings) is None

    assert _NOTE in capsys.readouterr().out


def test_training_says_where_it_is_running_and_what_precision_it_uses(
        tmp_path, capsys, fell_back_to_cpu, monkeypatch):
    """``amp=True`` on a CPU run is answered, not obeyed -- and said out loud."""
    monkeypatch.setattr("spacr.utils.choose_model",
                        lambda *a, **k: nn.Sequential(nn.Linear(4, 2)))
    generator = torch.Generator().manual_seed(0)
    batch = (torch.rand(4, 4, generator=generator),
             torch.tensor([0, 1, 0, 1]),
             [f"f{i}.png" for i in range(4)])
    src = tmp_path / "data"
    src.mkdir()
    dst = tmp_path / "model"
    dst.mkdir()

    _model, path = deep_spacr.train_model(
        str(src), str(dst), "resnet18", [batch], epochs=1,
        val_loaders=None, num_classes=2, schedule=None,
        settings={"mixed_precision": True}, tensorboard=False)

    out = capsys.readouterr().out
    assert _NOTE in out
    assert "training in full precision instead" in out
    assert path.endswith(".pth")
