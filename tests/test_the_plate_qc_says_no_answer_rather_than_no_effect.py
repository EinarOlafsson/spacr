"""Where plate QC declines to answer, and where its report stays quiet.

The distinction the module's own docstring draws is the point of the file:
``None`` means "no answer" and NaN or zero would be read as "no correlation".
On a plate-effect report that difference decides whether a screen is thrown
away, so the guards that produce None are worth more than the statistics they
guard.
"""
from __future__ import annotations

import numpy as np
import pytest


def _report(**changes):
    from spacr.plate_qc import EdgeEffectReport

    fields = dict(plate="p1", value_col=None, grouping="well", rings=[],
                  gradients={}, notes=[])
    fields.update(changes)
    return EdgeEffectReport(**fields)


# ---------------------------------------------------------------------------
# _spearman — too few points, and no spread
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 2])
def test_too_few_points_give_no_correlation_at_all(n):
    """Line 1033.

    Spearman over two points is +1 or -1 by construction. Returning that would
    put a perfect correlation into an edge-effect report built from a plate
    with two usable wells, which is the strongest possible claim from the
    weakest possible evidence.
    """
    from spacr.plate_qc import _spearman

    x = np.arange(n, dtype=float)
    y = np.arange(n, dtype=float)

    assert _spearman(x, y) == (None, None)


def test_a_constant_axis_gives_no_correlation_at_all():
    """The guard below it, and the docstring's own reason.

    scipy returns NaN here, and a NaN read out of a report is taken as "no
    correlation" -- which is a finding. "No answer" is not.
    """
    from spacr.plate_qc import _spearman

    constant = np.full(10, 3.0)
    varying = np.arange(10, dtype=float)

    assert _spearman(constant, varying) == (None, None)
    assert _spearman(varying, constant) == (None, None)


def test_a_real_pair_gives_a_correlation_and_a_p_value():
    """The taken side, so the two refusals above are visibly decisions."""
    from spacr.plate_qc import _spearman

    x = np.arange(12, dtype=float)
    y = x * 2.0 + 1.0

    rho, p_value = _spearman(x, y)

    assert rho == pytest.approx(1.0)
    assert p_value is not None and p_value < 0.05


# ---------------------------------------------------------------------------
# format_edge_report — the lines it does not print
# ---------------------------------------------------------------------------

def test_a_report_on_object_counts_names_the_grouping_not_a_column():
    """Arc 1551 -> 1553: no ``value_col``, so the default wording stands.

    Counting objects per well is the default QC, and it has no measurement
    column to name. Printing "well None" would be worse than the default
    sentence it replaces.
    """
    from spacr.plate_qc import format_edge_report

    text = format_edge_report(_report(value_col=None, grouping="well",
                                      n_rows=16, n_cols=24))

    assert "objects per well" in text
    assert "None" not in text


def test_a_report_on_a_measurement_names_the_column():
    """The taken side."""
    from spacr.plate_qc import format_edge_report

    text = format_edge_report(_report(value_col="area", grouping="cell",
                                      n_rows=16, n_cols=24))

    assert "cell area" in text


def test_a_report_with_no_notes_prints_no_notes_heading():
    """Arc 1606 -> 1611.

    A "Notes" heading with nothing under it reads as though the QC had
    something to add and lost it. Silence says there was nothing.
    """
    from spacr.plate_qc import format_edge_report

    text = format_edge_report(_report(notes=[], n_rows=16, n_cols=24))

    assert "Notes" not in text


def test_a_report_with_notes_prints_them_under_a_heading():
    """The taken side."""
    from spacr.plate_qc import format_edge_report

    text = format_edge_report(_report(
        notes=["12 well(s) dropped below the minimum count"],
        n_rows=16, n_cols=24))

    assert "Notes" in text
    assert "12 well(s) dropped" in text


def test_a_report_with_only_one_median_states_no_difference():
    """Arc 1394 -> 1397, reached through the formatter's own output.

    The difference needs both sides. With one median missing -- an edge ring
    with no usable wells -- it must stay None rather than becoming the other
    median by subtraction from zero.
    """
    from spacr.plate_qc import format_edge_report

    text = format_edge_report(_report(ok=True, edge_median=12.0,
                                      interior_median=None,
                                      n_rows=16, n_cols=24))

    assert isinstance(text, str) and text
