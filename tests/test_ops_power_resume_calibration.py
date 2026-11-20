"""OPS power resume rows belong to their calibration and resolved design."""

import copy

import numpy as np
import pandas as pd
import pytest

from spacr import power_ops as po


@pytest.fixture
def calibration():
    """A small empirical calibration with two genes and two control guides."""
    rng = np.random.default_rng(4)
    return {
        "n_genes": 2, "guides_per_gene": {"median": 2}, "nt_guides": 2,
        "cells_per_gene": {"q50": 10}, "call_rate": {"value": 1.0},
        "fdr_pct": {"value": 0}, "nt_ratio": 0.2, "guide_alpha": 100,
        "dims": 2, "nt_mean": [0, 0], "nt_cov": [[1, 0], [0, 1]],
        "_pool": (rng.standard_normal((40, 2)),
                  np.repeat(np.array(["nt1", "nt2"], dtype=object), 20)),
    }


def _scan(calibration, path=None):
    """Run one small design without optional analysis packages."""
    grid = {"cells_per_gene": [10], "hit_fraction": [0], "effect_dims": [1]}
    return po.scan_ops_designs(calibration, grid, [0], scorers=[],
                               progress_file=str(path) if path else None)


def test_changed_calibration_invalidates_the_power_resume_cache(calibration, tmp_path):
    """Changing the calibrated gene count must not reuse a smaller simulated screen."""
    path = tmp_path / "designs.tsv"
    first = _scan(calibration, path)
    assert first.loc[0, "status"] == "ok" and first.loc[0, "cells"] == 24
    changed = {**calibration, "n_genes": 4}
    resumed = _scan(changed, path)
    fresh = _scan(changed)
    assert resumed.loc[0, "cells"] == fresh.loc[0, "cells"] == 48
    assert resumed.loc[0, "run_key"] != first.loc[0, "run_key"]


@pytest.mark.parametrize("change", ["mean", "pool", "pool-guide"])
def test_calibration_changes_invalidate_rows_with_the_same_resolved_design(
        calibration, tmp_path, change):
    """Phenotype values and control-pool contents must participate in cache identity."""
    path = tmp_path / "designs.tsv"
    first = _scan(calibration, path)
    changed = copy.deepcopy(calibration)
    if change == "mean":
        changed["nt_mean"] = [5, 5]
    elif change == "pool":
        changed["_pool"][0][20, 0] += 5
    else:
        changed["_pool"][1][20] = "nt1"
    resumed = _scan(changed, path)
    assert resumed.loc[0, "status"] == "ok"
    assert resumed.loc[0, "design_key"] == first.loc[0, "design_key"]
    assert resumed.loc[0, "run_key"] != first.loc[0, "run_key"]


def test_an_unchanged_calibration_resumes_without_simulating_again(
        calibration, tmp_path, monkeypatch):
    """Copies of an in-memory control pool have stable identity across runs."""
    path = tmp_path / "designs.tsv"
    first = _scan(calibration, path)

    def unexpected_simulation(*args, **kwargs):
        """An unchanged completed row must be read from the progress table."""
        raise AssertionError("a completed simulation was repeated")

    monkeypatch.setattr(po, "_run_point", unexpected_simulation)
    resumed = _scan(copy.deepcopy(calibration), path)
    pd.testing.assert_frame_equal(resumed, first)


def test_changed_resolved_defaults_invalidate_the_power_resume_cache(
        calibration, tmp_path, monkeypatch):
    """A default absent from the explicit grid still changes the simulated design."""
    path = tmp_path / "designs.tsv"
    first = _scan(calibration, path)
    monkeypatch.setitem(po.OPS_POWER_DESIGN, "penetrance", 0.5)
    resumed = _scan(calibration, path)
    fresh = _scan(calibration)
    assert resumed.loc[0, "design_key"] == fresh.loc[0, "design_key"]
    assert resumed.loc[0, "run_key"] != first.loc[0, "run_key"]


def test_invalid_designs_remain_recorded_as_failed_rows(calibration, tmp_path):
    """Resolving cache identity must preserve the sweep failure-reporting path."""
    rows = po.scan_ops_designs(calibration, {"effect_type": ["unknown"]}, [0],
                               scorers=[], progress_file=str(tmp_path / "designs.tsv"))
    assert rows.loc[0, "status"] == "failed"
    assert "effect_type" in rows.loc[0, "error"]
