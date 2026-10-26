"""Saliency maps come out the same for the same crops and settings.

The Activation Maps tutorial (lesson 16) compares the saved maps with an
independent recomputation and allows one grey level of rounding. On the
workstation two runs differed by 17: ``spacr.deep_spacr`` turns on
``cudnn.benchmark`` at import, so cuDNN picked its convolution algorithm by
timing, and convolutions ran in TF32. These tests run on the CPU (no GPU at
Home) and check both the kernel flags in force while a map is computed and
that two runs save identical pixels.
"""
from __future__ import annotations

import tarfile

import numpy as np
import pytest
from PIL import Image

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg", force=True)
torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    from spacr.utils import TorchModel
    torch.manual_seed(3)
    path = tmp_path_factory.mktemp("model") / "binary_model.pth"
    torch.save(TorchModel(model_name="resnet18", pretrained=False,
                          num_classes=1, image_size=32), str(path))
    return str(path)


def _dataset(root):
    (root / "measurements").mkdir(parents=True)
    (root / "datasets").mkdir()
    raw = root / "raw"
    raw.mkdir()
    tar_path = root / "datasets" / "ds.tar"
    with tarfile.open(tar_path, "w") as tar:
        for i in range(5):
            rng = np.random.default_rng(i)
            name = f"plate1_A01_1_{i}.png"
            Image.fromarray(rng.integers(0, 256, (32, 32, 3)).astype(np.uint8)).save(raw / name)
            tar.add(raw / name, arcname=name)
    return str(tar_path)


def _settings(tar_path, model_path, cam_type):
    return {
        "dataset": tar_path, "model_path": model_path, "model_type": "resnet18",
        "cam_type": cam_type, "target_layer": None, "image_size": 32,
        "batch_size": 5, "channels": [1, 2, 3], "normalize": False,
        "normalize_input": True, "save": True, "plot": False, "correlation": False,
        "overlay": False, "shuffle": False, "n_jobs": 0,
        "manders_thresholds": [15, 50, 75],
    }


def _maps(root, cam_type):
    folder = root / "datasets" / "ds" / cam_type
    return {p.name: np.asarray(Image.open(p)).astype(np.int16)
            for p in sorted(folder.rglob("*.png"))}


def _flags():
    return dict(benchmark=torch.backends.cudnn.benchmark,
                deterministic=torch.backends.cudnn.deterministic,
                cudnn_tf32=torch.backends.cudnn.allow_tf32,
                matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
                algorithms=torch.are_deterministic_algorithms_enabled())


def test_saliency_is_computed_with_deterministic_full_precision_kernels(tmp_path, model_path, monkeypatch):
    from spacr import deep_spacr
    from spacr.utils import SaliencyMapGenerator

    seen = []
    original = SaliencyMapGenerator.compute_saliency_and_predictions

    def spy(self, inputs):
        seen.append(_flags())
        return original(self, inputs)

    monkeypatch.setattr(SaliencyMapGenerator, "compute_saliency_and_predictions", spy)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    before = _flags()
    deep_spacr.generate_activation_map(
        _settings(_dataset(tmp_path / "p"), model_path, "saliency_channel"))

    assert seen, "the saliency generator never ran"
    for flags in seen:
        assert flags == dict(benchmark=False, deterministic=True, cudnn_tf32=False,
                             matmul_tf32=False, algorithms=True)
    assert _flags() == before


@pytest.mark.parametrize("cam_type", ["saliency_image", "saliency_channel"])
def test_two_runs_save_the_same_saliency_pixels(tmp_path, model_path, cam_type):
    from spacr.deep_spacr import generate_activation_map

    runs = []
    for index in range(2):
        root = tmp_path / f"run{index}"
        generate_activation_map(_settings(_dataset(root), model_path, cam_type))
        runs.append(_maps(root, cam_type))

    assert runs[0] and runs[0].keys() == runs[1].keys()
    worst = max(int(np.abs(runs[0][k] - runs[1][k]).max()) for k in runs[0])
    assert worst <= 1
