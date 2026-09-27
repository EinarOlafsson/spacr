"""The segmentation parameter-robustness report.

A few fields are re-segmented with the diameter, the flow and cell-probability
thresholds and contrast enhancement each moved alone; the report says how much
object counts, areas, intensities and the objects themselves move, and flags a
setting whose results are fragile. Here a synthetic plate is segmented by a
threshold whose level follows the cell-probability threshold, so raising it
deliberately drops the dim half of the objects.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from skimage.measure import label as sk_label

from spacr.seg_qc import (_match_fraction, _number_list, _robustness_grid,
                          _score_robustness)


def _field(seed):
    """A 96 x 96 field with six bright and six dim square objects."""
    rng = np.random.default_rng(seed)
    image = rng.uniform(0.0, 0.05, (96, 96)).astype(np.float32)
    for index in range(12):
        row, col = divmod(index, 4)
        level = 0.9 if index % 2 else 0.45
        image[8 + 28 * row: 20 + 28 * row, 6 + 23 * col: 18 + 23 * col] = level
    return image


def _segment(image, point):
    """Threshold at 0.3 moved by a tenth of the cell-probability threshold."""
    plane = image if image.ndim == 2 else image[..., 0]
    return sk_label(plane > 0.3 + 0.1 * float(point["cellprob_threshold"]))


def _plate(tmp_path, n_fields=3):
    """A mask source folder with one ``.npz`` batch of single-channel fields."""
    masks = tmp_path / "plate" / "masks"
    masks.mkdir(parents=True)
    stack = np.stack([_field(i) for i in range(n_fields)])[..., None]
    names = np.array([f"plate1_A0{i + 1}_1.npy" for i in range(n_fields)])
    np.savez(masks / "batch_0.npz", data=(stack * 1000).astype(np.uint16),
             filenames=names)
    return masks


def _settings(**extra):
    from spacr.settings import set_default_settings_preprocess_generate_masks

    settings = set_default_settings_preprocess_generate_masks(
        dict(src="unused", cell_channel=0, nucleus_channel=None,
             pathogen_channel=None, robustness_report=True, **extra))
    return settings


def test_number_lists_read_from_text_and_lists():
    assert _number_list("0.5, 2") == [0.5, 2.0]
    assert _number_list("[-2; 2]") == [-2.0, 2.0]
    assert _number_list([1, "x", 3]) == [1.0, 3.0]
    assert _number_list("", (0.2,)) == [0.2]
    assert _number_list(None, (1,)) == [1.0]


def test_the_grid_moves_one_parameter_at_a_time():
    grid = _robustness_grid(_settings(cell_diameter=None), "cell")
    assert grid[0]["parameter"] == "baseline"
    assert [p["parameter"] for p in grid[1:]] == [
        "diameter", "diameter", "flow_threshold", "flow_threshold",
        "cellprob_threshold", "cellprob_threshold", "enhancement"]
    assert [p["diameter"] for p in grid[1:3]] == [22.5, 37.5]
    base = grid[0]
    for point in grid[1:]:
        moved = [k for k in ("diameter", "flow_threshold", "cellprob_threshold", "enhance")
                 if point[k] != base[k]]
        assert len(moved) == 1
    grid = _robustness_grid(_settings(robustness_enhancement=False,
                                      robustness_diameter_factors="1, 2",
                                      cell_diameter=40), "cell")
    assert [p["diameter"] for p in grid if p["parameter"] == "diameter"] == [80.0]
    assert all(p["parameter"] != "enhancement" for p in grid)


def test_match_fraction_counts_objects_found_again():
    a = np.zeros((20, 20), int)
    a[2:8, 2:8] = 1
    a[10:16, 10:16] = 2
    assert _match_fraction(a, a) == 1.0
    b = a.copy()
    b[b == 2] = 0
    assert _match_fraction(a, b) == 0.5
    assert _match_fraction(np.zeros_like(a), np.zeros_like(a)) == 1.0


def test_a_deliberately_fragile_threshold_is_flagged():
    fields = [(f"f{i}", _field(i)) for i in range(3)]
    grid = _robustness_grid(_settings(), "cell")
    per_field, summary = _score_robustness(fields, _segment, grid, 0.2)
    assert len(per_field) == 3 * len(grid)
    row = summary.set_index(["parameter", "value"])
    assert row.loc[("cellprob_threshold", "2"), "fragile"]
    assert row.loc[("cellprob_threshold", "2"), "object_count_change"] == 0.5
    assert "object_count" in row.loc[("cellprob_threshold", "2"), "reason"] or \
        "baseline objects" in row.loc[("cellprob_threshold", "2"), "reason"]
    assert not row.loc[("baseline", "run settings"), "fragile"]
    assert not row.loc[("flow_threshold", "0.2"), "fragile"]
    assert not row.loc[("cellprob_threshold", "-2"), "fragile"]


def test_the_report_runs_on_a_plate_sample_and_writes_its_tables(tmp_path, capsys):
    from spacr.object import _run_robustness_report
    from spacr.tabular import read_table

    masks = _plate(tmp_path)
    settings = _settings(robustness_fields=2, robustness_crop=64)
    summary = _run_robustness_report(str(masks), settings, "cell", segment=_segment)
    assert isinstance(summary, pd.DataFrame)
    assert summary["n_fields"].eq(2).all()
    fragile = summary[summary["fragile"]]
    assert list(fragile["parameter"]) == ["cellprob_threshold"]
    qc = tmp_path / "plate" / "qc"
    table = read_table(qc / "segmentation_robustness_cell.csv")
    assert len(table) == len(summary)
    assert len(read_table(qc / "segmentation_robustness_cell_fields.csv")) == 2 * len(summary)
    assert any(p.name.startswith("segmentation_robustness_cell.") and p.suffix != ".csv"
               for p in qc.iterdir())
    out = capsys.readouterr().out
    assert "FRAGILE" in out and "cellprob_threshold" in out


def test_the_report_is_off_by_default_and_never_raises(tmp_path):
    from spacr.object import _run_robustness_report
    from spacr.settings import set_default_settings_preprocess_generate_masks

    defaults = set_default_settings_preprocess_generate_masks({"src": "x", "cell_channel": 0})
    assert defaults["robustness_report"] is False
    assert _run_robustness_report(str(tmp_path), defaults, "cell", segment=_segment) is None

    def broken(image, point):
        raise RuntimeError("model fell over")

    masks = _plate(tmp_path)
    assert _run_robustness_report(str(masks), _settings(), "cell", segment=broken) is None
