"""Edge cases of Measure's vectorised tables and its DNA-content gates."""
from __future__ import annotations

import numpy as np
import pytest

from spacr import measure as M

torch = pytest.importorskip("torch")
CPU = torch.device("cpu")


def _field():
    labels = np.zeros((16, 16), np.int32)
    labels[1:6, 1:6] = 1
    labels[8:14, 8:14] = 2
    image = np.random.default_rng(3).gamma(2.0, 100.0, labels.shape).astype(np.float32)
    return labels, image


def test_no_field_percentiles_leave_the_fractions_missing():
    labels, image = _field()
    table = M._torch_intensity_table(labels, image, (np.nan, np.nan), CPU)
    assert table["frac_high90"].isna().all() and table["frac_low10"].isna().all()


def test_homogeneity_uses_its_default_distances_and_handles_no_objects():
    labels, image = _field()
    table = M._torch_homogeneity(labels, image, None, CPU)
    assert [c for c in table.columns if c.startswith("homogeneity")] == [
        f"homogeneity_distance_{d}" for d in (2, 4, 8, 16, 32, 64)]
    empty = M._torch_homogeneity(np.zeros_like(labels), image, [2], CPU)
    assert list(empty.columns) == ["homogeneity_distance_2"] and empty.empty


def _fit(densities):
    class _Fit:
        g1, g2 = 1.0, 2.0

        def densities(self, grid):
            return densities(grid)

    return _Fit()


def test_gates_fall_back_to_the_g1_g2_crossing_when_s_never_wins():
    def densities(grid):
        g1 = np.exp(-((grid - 1.0) ** 2) * 20)
        g2 = np.exp(-((grid - 2.0) ** 2) * 20)
        return np.stack([g1, np.zeros_like(grid), g2], axis=1)

    first, second = M._dna_gates(_fit(densities))
    assert first == pytest.approx(second, abs=1e-6)
    assert 1.4 < first < 1.6


def test_gates_with_no_crossing_at_all_fall_back_to_the_midpoint():
    def densities(grid):
        return np.stack([np.ones_like(grid), np.zeros_like(grid),
                         np.zeros_like(grid)], axis=1)

    assert M._dna_gates(_fit(densities)) == (1.5, 1.5)


def test_a_single_population_has_no_mixture_crossing():
    values = np.random.default_rng(0).normal(5.0, 0.1, 400)
    fit = M._two_population_fit(values)
    assert np.isfinite(fit["cut"])
