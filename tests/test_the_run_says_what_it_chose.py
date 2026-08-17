"""A setting the run chose for itself is shown, and saved.

Two reports, 2026-08-17.

1. "if no fraction threshold and min cell cound is set these are set
   automatically, these automatic values should be shown in the runs values
   rows". The settings table is printed -- and `save_settings` writes the CSV
   -- BEFORE either value is derived, so both showed `None` there and the
   numbers appeared only in passing prose. A settings record that says None
   for a value the run chose is a record you cannot reproduce the run from.

2. `analysis_mode='guide_permutation'` with `analysis_unit='cell'` failed
   nine frames deep, phrased as a data-integrity problem:

       ValueError: Phenotype/block/nuisance values are not constant within
       well 'plate1_r1_c12'.

   after a 20-second run that had already written its regression data, three
   summary plots and their statistics. It is not a corrupt table; it is two
   settings that cannot both be honoured.
"""
from __future__ import annotations

import inspect

import pandas as pd
import pytest


# --------------------------------------------------------------------------- #
#  The automatic values are reported
# --------------------------------------------------------------------------- #

def test_the_derived_values_are_recorded():
    from spacr.ml import _AUTOMATIC_SETTINGS

    assert isinstance(_AUTOMATIC_SETTINGS, dict)


def test_both_automatic_settings_are_recorded_where_they_are_derived():
    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    assert "_AUTOMATIC_SETTINGS['min_cell_count']" in source
    assert "_AUTOMATIC_SETTINGS['fraction_threshold']" in source


def test_they_are_printed_under_a_heading_that_says_they_are_automatic():
    """"0.0168" in a wall of output is not an answer to "what did the run
    use"; "Chosen automatically" is."""
    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    assert "Chosen automatically (not set by the user)" in source


def test_the_settings_csv_is_re_saved_once_they_are_known():
    """It is written before either value exists, so the file on disk said
    None for both. Re-saving is what makes the run reproducible from it."""
    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    body = source.split("Chosen automatically", 1)[1]
    assert "save_settings(settings, name='regression'" in body


def test_the_record_is_cleared_per_run():
    """A GUI session runs many. A second run must not report the first one's
    choices as its own."""
    from spacr.ml import perform_regression

    source = inspect.getsource(perform_regression)
    clear_at = source.index("_AUTOMATIC_SETTINGS.clear()")
    first_write = source.index("_AUTOMATIC_SETTINGS['min_cell_count']")
    assert clear_at < first_write, (
        "the record is written before it is cleared, so a second run in one "
        "process inherits the first one's values")


# --------------------------------------------------------------------------- #
#  The impossible combination is refused early
# --------------------------------------------------------------------------- #

def _frame():
    return pd.DataFrame({"prc": ["plate1_r1_c1"] * 3,
                         "pred": [0.1, 0.2, 0.3],
                         "grna": ["g_1", "g_2", "g_3"],
                         "plateID": ["plate1"] * 3})


@pytest.mark.parametrize("unit", ["cell", "object", "nucleus"])
def test_a_per_object_unit_is_refused(unit, tmp_path):
    from spacr.ml import _run_guide_permutation_analysis

    with pytest.raises(ValueError) as caught:
        _run_guide_permutation_analysis(
            _frame(), "pred", str(tmp_path),
            {"analysis_unit": unit, "agg_type": None})

    message = str(caught.value)
    assert "guide_permutation" in message
    assert unit in message
    # It names BOTH ways out, because either may be what the user meant.
    assert "analysis_unit='well'" in message
    assert "analysis_mode='regression'" in message


def test_the_well_unit_is_not_refused(tmp_path):
    """The refusal must not fire on the combination that works -- it is
    checked before any real work, so a wrong check would block every
    permutation run."""
    from spacr.ml import _run_guide_permutation_analysis

    try:
        _run_guide_permutation_analysis(
            _frame(), "pred", str(tmp_path),
            {"analysis_unit": "well", "agg_type": "mean"})
    except ValueError as error:
        assert "guide_permutation tests each guide across WELLS" not in str(error)
    except Exception:
        pass          # any other failure is downstream, not this guard


def test_the_default_unit_is_not_refused(tmp_path):
    """`analysis_unit` absent means well."""
    from spacr.ml import _run_guide_permutation_analysis

    try:
        _run_guide_permutation_analysis(_frame(), "pred", str(tmp_path), {})
    except ValueError as error:
        assert "needs one row per well" not in str(error)
    except Exception:
        pass


def test_it_refuses_before_doing_the_work():
    """The report came after a 20-second run that had already written its
    regression data and three plots. The check must precede the call it
    guards."""
    from spacr.ml import _run_guide_permutation_analysis

    source = inspect.getsource(_run_guide_permutation_analysis)
    guard = source.index("tests each guide across")
    work = source.index("analyse_long_guide_table(")
    assert guard < work
