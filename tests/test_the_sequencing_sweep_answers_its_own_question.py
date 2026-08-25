"""Two thresholds, two questions, one switch -- and the run names which is which.

``target_unique_count`` asks how many gRNAs a well should end up with and
answers it from the counts. ``calibrate_fraction_threshold`` asks which
cut-off makes imaging and sequencing agree and answers it from the control
wells. Both numbers are worth having, so the run reports both; only one
switch decides which is in force, and it is read where the score table is
already loaded.

The failure these pin is not a crash. It is a bare number: "Closest Fraction
Threshold: 0.0168" with no source, printed on the very path a screen falls
through to when it asked for the calibration and could not have it.
"""
from __future__ import annotations

import inspect

import matplotlib
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import sequencing as SEQ  # noqa: E402


def _count_csv(path, n_wells=9, n_grna=15, seed=3):
    """One count table: a random subset of gRNAs per well with read counts."""
    rng = np.random.default_rng(seed)
    rows = []
    for well in range(n_wells):
        row_id = f"r{(well % 3) + 1}"
        column_id = f"c{(well // 3) + 1}"
        for guide in range(n_grna):
            if rng.random() < 0.6:
                rows.append({"plateID": "plate1", "rowID": row_id,
                             "columnID": column_id, "grna": f"g{guide}",
                             "count": int(rng.integers(5, 500))})
    pd.DataFrame(rows).to_csv(path, index=False)


def _settings(csv, **extra):
    base = {
        "count_data": str(csv),
        "target_unique_count": 5,
        "filter_column": "columnID",
        "control_wells": ["c1"],
    }
    base.update(extra)
    return base


def test_the_switch_does_not_change_what_the_counts_answer(tmp_path):
    """The same threshold with the calibration switch on and off.

    The counts module cannot run the calibration -- it is never given the
    per-cell scores or the named control blocks -- so a switch that changed
    its answer here could only be changing it to something it did not
    measure.
    """
    csv = tmp_path / "counts.csv"
    _count_csv(str(csv))

    without = SEQ.graph_sequencing_stats(_settings(csv))
    with_switch = SEQ.graph_sequencing_stats(
        _settings(csv, calibrate_fraction_threshold=True,
                  positive_control_wells="c2",
                  negative_control_wells="c3"))

    assert float(with_switch) == float(without)


def test_the_threshold_is_handed_back_with_the_question_it_answers(tmp_path,
                                                                   capsys):
    """The returned number names ``target_unique_count`` as its source.

    And says, in the same breath, that it is not the imaging/sequencing
    agreement cut-off -- because a screen that ticked the calibration box and
    was refused lands on exactly this number.
    """
    csv = tmp_path / "counts.csv"
    _count_csv(str(csv))

    threshold = SEQ.graph_sequencing_stats(_settings(csv))
    printed = capsys.readouterr().out

    assert f"fraction_threshold={threshold}" in printed
    assert "target_unique_count=5" in printed
    assert "calibrate_fraction_threshold" in printed
    assert "imaging and sequencing agree" in printed


def test_the_sweep_line_says_where_its_number_came_from(tmp_path, capsys):
    """"Closest Fraction Threshold" carries its source on the same line."""
    csv = tmp_path / "counts.csv"
    _count_csv(str(csv))

    SEQ.graph_sequencing_stats(_settings(csv, target_unique_count=4))
    printed = capsys.readouterr().out

    line = next(l for l in printed.splitlines()
                if l.startswith("Closest Fraction Threshold:"))

    assert "target_unique_count=4" in line


def test_the_counts_module_never_reads_the_calibration_switch():
    """One reader for the switch, where the imaging side exists.

    A second reader in a module that cannot run the sweep could only
    disagree with the first -- or, worse, quietly return the counts answer
    while appearing to honour the switch.
    """
    source = inspect.getsource(SEQ.graph_sequencing_stats)
    body = source.split('"""', 2)[2]

    assert "settings.get('calibrate_fraction_threshold')" not in body
    assert 'settings["calibrate_fraction_threshold"]' not in body
    assert "calibrate_fraction_threshold" in source, (
        "the docstring has to say why it is not read here")


def test_the_regression_path_is_the_one_reader():
    """``ml._perform_regression`` reads it, and nothing else in spaCR does."""
    from spacr import ml

    readers = [name for name, obj in vars(ml).items()
               if inspect.isfunction(obj)
               and "calibrate_fraction_threshold" in _safe_source(obj)]

    assert readers == ["_perform_regression"], readers


def _safe_source(obj) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):                              # noqa: BLE001
        return ""
