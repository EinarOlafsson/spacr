"""The GPU measurement kernels give the CPU measurement's numbers.

The kernels are PyTorch code that runs unchanged on a CUDA device or on the
CPU, so running them on the CPU here checks the very arithmetic the GPU run
uses against the scikit-image / Mahotas path that stays the default.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from skimage.segmentation import expand_labels                     # noqa: E402

from spacr import measure as M                                      # noqa: E402

CPU = torch.device("cpu")


def _field(seed=0, size=160, objects=40):
    """A crowded 2-D label field with touching objects and one gap in the numbering."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    for i, (r, c) in enumerate(rng.integers(3, size - 3, (objects, 2)), 1):
        labels[r, c] = i
    labels = expand_labels(labels, 7)
    labels[labels == 5] = 0
    labels[size - 3:, :3] = objects + 3
    return labels, rng


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_the_intensity_table_matches_regionprops(dtype):
    labels, rng = _field()
    image = rng.gamma(2.0, 300.0, labels.shape).astype(dtype)
    image[:30, :30] = image[0, 0]
    image[60:64, :] = 0
    percentiles = M._field_reference_percentiles(image)

    cpu = M._extended_regionprops_table(
        labels, image, list(M._GPU_INTENSITY_PROPS),
        field_percentiles=percentiles)
    gpu = M._torch_intensity_table(labels, image, percentiles, CPU)

    assert list(gpu.columns) == list(cpu.columns)
    assert list(gpu.dtypes) == list(cpu.dtypes)
    np.testing.assert_array_equal(gpu["label"], cpu["label"])
    scale = float(np.max(image))
    for column in cpu.columns:
        np.testing.assert_allclose(
            gpu[column].astype(float), cpu[column].astype(float),
            rtol=1e-5, atol=1e-5 * scale if dtype == np.float32 else 1e-9 * scale,
            err_msg=column)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_homogeneity_matches_the_co_occurrence_matrix(dtype):
    labels, rng = _field(seed=1)
    image = rng.gamma(2.0, 300.0, labels.shape).astype(dtype)
    image[labels == 3] = 17
    distances = [1, 2, 4, 8, 16, 32, 200]

    cpu = M._calculate_homogeneity(labels, image, distances)
    gpu = M._torch_homogeneity(labels, image, distances, CPU)

    assert list(gpu.columns) == list(cpu.columns)
    np.testing.assert_array_equal(np.isnan(gpu.values), np.isnan(cpu.values))
    np.testing.assert_allclose(gpu.values, cpu.values, rtol=0, atol=1e-12)
    assert np.isnan(gpu["homogeneity_distance_200"]).all()


def test_zernike_matches_mahotas():
    pytest.importorskip("mahotas")
    labels, _ = _field(seed=2)
    n_objects = len(np.unique(labels)) - 1
    frame = pd.DataFrame({"label": np.arange(n_objects)})

    cpu = M._calculate_zernike(labels, frame, degree=8)
    gpu = M._calculate_zernike(labels, frame, degree=8, device=CPU)

    assert list(gpu.columns) == list(cpu.columns)
    assert sum(c.startswith("zernike_") for c in gpu.columns) == 25
    np.testing.assert_allclose(gpu.values, cpu.values, rtol=0, atol=1e-12)


def test_the_setting_is_off_by_default_and_the_cpu_path_stays():
    from spacr.settings import get_measure_crop_settings

    settings = get_measure_crop_settings({})
    assert settings["measure_gpu"] is False
    assert M._measurement_device(settings) is None


def test_without_cuda_the_run_falls_back_to_the_cpu(monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert M._measurement_device({"measure_gpu": True}) is None
    assert "measuring on the CPU" in capsys.readouterr().out


def test_only_a_2d_unspaced_finite_input_takes_the_gpu_kernels():
    labels, rng = _field()
    image = rng.random(labels.shape).astype(np.float32)
    assert M._gpu_measurable(labels, image)
    assert not M._gpu_measurable(labels, image, spacing=(1.0, 0.5))
    assert not M._gpu_measurable(np.stack([labels, labels]))
    image[0, 0] = np.nan
    assert not M._gpu_measurable(labels, image)


def test_intensity_measurements_route_through_the_device(monkeypatch):
    labels, rng = _field(seed=3, size=96, objects=12)
    channels = rng.gamma(2.0, 300.0, labels.shape + (2,)).astype(np.float32)
    empty = np.zeros_like(labels)
    settings = {
        "radial_dist": False, "calculate_correlation": False,
        "homogeneity": True, "homogeneity_distances": [2, 4],
        "distance_gaussian_sigma": 0, "cell_mask_dim": 0,
        "nucleus_mask_dim": None, "pathogen_mask_dim": None,
    }
    cpu = M._intensity_measurements(labels, empty, empty, empty, empty,
                                    channels, settings)[0]
    monkeypatch.setattr(M, "_measurement_device", lambda _settings: CPU)
    gpu = M._intensity_measurements(labels, empty, empty, empty, empty,
                                    channels, settings)[0]

    assert list(gpu.columns) == list(cpu.columns)
    numeric = cpu.select_dtypes("number").columns
    np.testing.assert_allclose(gpu[numeric].astype(float),
                               cpu[numeric].astype(float),
                               rtol=1e-4, atol=1e-2)
