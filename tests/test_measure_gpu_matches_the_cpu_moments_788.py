"""Item 788: measure_gpu and the CPU path agree on skew and kurtosis.

The GPU kernels compute the central moments in float64 with a centred
two-pass sum. The CPU path handed SciPy the object's pixels in the image's
own dtype, so on a float32 image the third and fourth moments were
accumulated in float32 and moved by up to about 1 % on bright, low-contrast
objects. Both paths now compute in float64; these tests pin the agreement.
The "GPU" kernels run on the torch CPU device, which executes the same code.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

torch = pytest.importorskip("torch")

from skimage.segmentation import expand_labels                     # noqa: E402

from spacr import measure as M                                      # noqa: E402

CPU = torch.device("cpu")

#: Stated parity: skew and kurtosis agree to this relative tolerance.
MOMENT_RTOL = 1e-9

MOMENTS = ("skew_intensity", "kurtosis_intensity")


def _bright_low_contrast_field(dtype, seed=0, size=200, objects=40):
    """Large objects on a bright pedestal with little spread: float32's worst case."""
    rng = np.random.default_rng(seed)
    labels = np.zeros((size, size), np.int32)
    for i, (r, c) in enumerate(rng.integers(3, size - 3, (objects, 2)), 1):
        labels[r, c] = i
    labels = expand_labels(labels, 12)
    image = (20000.0 + rng.gamma(2.0, 4.0, labels.shape)).astype(dtype)
    return labels, image


def _tables(labels, image):
    percentiles = M._field_reference_percentiles(image)
    cpu = M._extended_regionprops_table(
        labels, image, list(M._GPU_INTENSITY_PROPS),
        field_percentiles=percentiles)
    gpu = M._torch_intensity_table(labels, image, percentiles, CPU)
    return cpu, gpu


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.uint16])
def test_skew_and_kurtosis_agree_within_the_stated_tolerance(dtype):
    labels, image = _bright_low_contrast_field(dtype)
    cpu, gpu = _tables(labels, image)

    assert len(cpu) == 40
    for column in MOMENTS:
        assert np.isfinite(cpu[column]).all(), column
        np.testing.assert_allclose(
            gpu[column].to_numpy(float), cpu[column].to_numpy(float),
            rtol=MOMENT_RTOL, atol=0, err_msg=column)


def test_the_cpu_moments_are_the_float64_moments_of_a_float32_image():
    labels, image = _bright_low_contrast_field(np.float32, seed=1)
    cpu, _ = _tables(labels, image)

    for row, label in enumerate(cpu["label"]):
        pixels = image[labels == label].astype(np.float64)
        assert cpu["skew_intensity"].iloc[row] == pytest.approx(
            skew(pixels), rel=1e-12)
        assert cpu["kurtosis_intensity"].iloc[row] == pytest.approx(
            kurtosis(pixels), rel=1e-12)


def test_float32_moments_would_have_missed_the_tolerance():
    """The control: the old float32 accumulation is outside the stated bound."""
    labels, image = _bright_low_contrast_field(np.float32, seed=2)
    _, gpu = _tables(labels, image)

    worst = 0.0
    for row, label in enumerate(gpu["label"]):
        pixels = image[labels == label]
        narrow = float(kurtosis(pixels))
        reference = float(gpu["kurtosis_intensity"].iloc[row])
        worst = max(worst, abs(narrow - reference) / abs(reference))
    assert worst > 100 * MOMENT_RTOL


def test_the_routed_measurement_matches_the_cpu_moments(monkeypatch):
    labels, image = _bright_low_contrast_field(np.float32, seed=3, size=120,
                                               objects=12)
    channels = np.stack([image, image[::-1].copy()], axis=-1)
    empty = np.zeros_like(labels)
    settings = {
        "radial_dist": False, "calculate_correlation": False,
        "homogeneity": False, "homogeneity_distances": [2],
        "distance_gaussian_sigma": 0,
        "cell_mask_dim": 0, "nucleus_mask_dim": None,
        "pathogen_mask_dim": None,
    }
    cpu = M._intensity_measurements(labels, empty, empty, empty, empty,
                                    channels, settings)[0]
    monkeypatch.setattr(M, "_measurement_device", lambda _settings: CPU)
    gpu = M._intensity_measurements(labels, empty, empty, empty, empty,
                                    channels, settings)[0]

    moment_columns = [c for c in cpu.columns
                      if c.endswith(MOMENTS)]
    assert len(moment_columns) == 4
    for column in moment_columns:
        np.testing.assert_allclose(
            gpu[column].to_numpy(float), cpu[column].to_numpy(float),
            rtol=MOMENT_RTOL, atol=0, err_msg=column)
