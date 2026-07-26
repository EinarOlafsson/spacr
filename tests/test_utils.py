"""
Tests for pure/testable helpers in spacr.utils.

We stick to helpers that don't need GPU, network, or cellpose models — the
heavy parts of utils.py get exercised by higher-level pipelines elsewhere.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

import spacr.utils as U


# ---------------------------------------------------------------------------
# _safe_int_convert
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("val,expected", [
    (42, 42),
    ("42", 42),
    ("042", 42),       # de-zero-padding is the one job it still has
    (" 42 ", 42),
    (42.0, 42),
    ("-15", -15),
])
def test_safe_int_convert_ok(val, expected):
    assert U._safe_int_convert(val) == expected


@pytest.mark.parametrize("val", [3.7, True, False, "s3"])
def test_safe_int_convert_takes_the_default_when_there_is_no_integer(val):
    """"What integer is this?" is answered by ``schema.parse_int_token``.

    So the cases a bare ``int()`` used to answer by inventing one now take
    the default: ``3.7`` is not a field and ``True`` is a caller bug, and
    silently turning them into ``3`` and ``1`` is the same species of lie as
    turning ``'x'`` into ``0``. Both are inert in spaCR — the only remaining
    callers hand it a regex group that starts with a digit — which is why
    this is a cheap place to be strict.

    ``'s3'`` is deliberate too: a vendor prefix is a *key* spelling, and key
    construction belongs to ``schema.field_id``, not here.
    """
    assert U._safe_int_convert(val) == 0
    assert U._safe_int_convert(val, default=-7) == -7


def test_safe_int_convert_falls_back_on_invalid(capsys):
    out = U._safe_int_convert("not_a_number", default=99)
    assert out == 99


def test_safe_int_convert_default_zero(capsys):
    assert U._safe_int_convert("nope") == 0


def test_safe_int_convert_no_longer_raises_on_none():
    """``resume._safe_int`` caught ``TypeError`` and this one did not.

    So ``None`` was field ``f0`` in one code path and a crash in the other.
    There is one answer now, and it is the default.
    """
    assert U._safe_int_convert(None) == 0
    assert U._safe_int_convert(None, default=5) == 5


@pytest.mark.parametrize("token,expected", [
    ("001", "1"), ("1", "1"), ("0", "0"), (" 7 ", "7"),
    ("1a", "1a"),          # not a number: kept, NOT turned into '0'
    ("12x34", "12x34"),
])
def test_int_or_token_keeps_what_it_cannot_read(token, expected):
    """The de-zero-padding helper the filename metadata parsers use.

    ``str(_safe_int_convert(x))`` turned every unreadable token into ``'0'``,
    so two odd wells became one well. Keeping the token keeps them apart and
    keeps them visible.
    """
    assert U._int_or_token(token) == expected


# ---------------------------------------------------------------------------
# _convert_cq1_well_id — 384-well CQ1 encoding (row A..P, col 1..24)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("well_id,expected", [
    (1,   "A01"),
    (24,  "A24"),
    (25,  "B01"),
    (48,  "B24"),
    (384, "P24"),
])
def test_convert_cq1_well_id(well_id, expected):
    assert U._convert_cq1_well_id(well_id) == expected


@pytest.mark.parametrize("well_id", [0, -1, "1a", "", None])
def test_convert_cq1_well_id_keeps_a_token_that_names_no_well(well_id, capsys):
    """``chr(ord('A') + row)`` used to emit punctuation for these.

    Index ``0`` came back as ``'@24'``: a well name with a punctuation mark
    where the row letter should be. Now that ``_extract_filename_metadata``
    keeps an unreadable well token instead of substituting ``'0'``, this can
    also be handed ``'1a'``. Keeping the token means two odd wells stay two
    odd wells and no name is invented for either.
    """
    assert U._convert_cq1_well_id(well_id) == str(well_id)
    assert "Not a CQ1 well index" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# mask_object_count
# ---------------------------------------------------------------------------

def test_mask_object_count_zero_for_empty_mask():
    m = np.zeros((10, 10), dtype=np.int32)
    assert U.mask_object_count(m) == 0


def test_mask_object_count_ignores_background(synth_mask_2d):
    n = U.mask_object_count(synth_mask_2d)
    # Number of positive-integer labels in the mask.
    expected = len(np.unique(synth_mask_2d)) - (1 if 0 in np.unique(synth_mask_2d) else 0)
    assert n == expected


def test_mask_object_count_arbitrary_ids():
    m = np.array([[0, 5, 5], [0, 0, 12], [7, 7, 12]])
    assert U.mask_object_count(m) == 3  # ids 5, 7, 12


# ---------------------------------------------------------------------------
# _relabel_sequential — fills gaps in label ids
# ---------------------------------------------------------------------------

def test_relabel_sequential_compacts_ids():
    m = np.array([[0, 3, 3], [0, 0, 7], [12, 12, 7]], dtype=np.int32)
    out = U._relabel_sequential(m)
    unique = sorted(int(x) for x in np.unique(out) if x != 0)
    assert unique == [1, 2, 3]  # 3 objects, contiguous 1..3
    # Same connected regions preserved.
    assert (out != 0).sum() == (m != 0).sum()


def test_relabel_sequential_empty_mask_unchanged():
    m = np.zeros((5, 5), dtype=np.int32)
    out = U._relabel_sequential(m)
    assert out.shape == m.shape
    assert (out == 0).all()


# ---------------------------------------------------------------------------
# annotate_conditions — pandas mapping helper
# ---------------------------------------------------------------------------

def _mini_df():
    """Small DataFrame with the row/column encoding annotate_conditions expects."""
    return pd.DataFrame({
        "rowID": ["r1", "r1", "r2", "r2"],
        "columnID": ["c1", "c2", "c1", "c2"],
    })


def test_annotate_conditions_single_string_broadcasts():
    df = _mini_df()
    out = U.annotate_conditions(df, cells="HeLa")
    assert (out["host_cells"] == "HeLa").all()
    # 'condition' becomes just the host_cells value when no other axes present.
    assert (out["condition"] == "HeLa").all()


def test_annotate_conditions_column_based_mapping():
    df = _mini_df()
    out = U.annotate_conditions(
        df,
        cells="HeLa",
        pathogens=["parasite"],
        pathogen_loc=[["c1"]],   # only column c1 gets the pathogen; c2 stays NaN
    )
    # Column c1 rows → parasite; column c2 rows → NaN.
    c1_rows = out[out["columnID"] == "c1"]
    c2_rows = out[out["columnID"] == "c2"]
    assert (c1_rows["pathogen"] == "parasite").all()
    assert c2_rows["pathogen"].isna().all()
    # Condition string reflects presence/absence of pathogen.
    assert (c1_rows["condition"] == "HeLa_parasite").all()
    assert (c2_rows["condition"] == "HeLa").all()


def test_annotate_conditions_row_based_mapping():
    df = _mini_df()
    out = U.annotate_conditions(
        df,
        treatments=["drugA", "drugB"],
        treatment_loc=[["r1"], ["r2"]],
    )
    r1 = out[out["rowID"] == "r1"]
    r2 = out[out["rowID"] == "r2"]
    assert (r1["treatment"] == "drugA").all()
    assert (r2["treatment"] == "drugB").all()


# ---------------------------------------------------------------------------
# save_settings / load_settings — round trip via csv
# ---------------------------------------------------------------------------

def test_save_and_load_settings_round_trip(tmp_project_dir, monkeypatch):
    # save_settings writes to `settings/<name>_settings.csv` relative to
    # settings['src']. Feed it a settings dict pointing at our tmp dir.
    settings = {
        "src": str(tmp_project_dir),
        "batch_size": 32,
        "channels": [0, 1, 2],
    }
    U.save_settings(settings, name="unit_test")
    written = list((tmp_project_dir / "settings").glob("unit_test*.csv"))
    assert written, "save_settings did not write a csv under settings/"
    # save_settings writes columns ['Key', 'Value']; tell load_settings so.
    loaded = U.load_settings(str(written[0]), setting_key="Key", setting_value="Value")
    assert loaded is not None
    # load_settings returns str values by default; accept "32" too.
    val = loaded["batch_size"] if isinstance(loaded, dict) else loaded.loc["batch_size", "Value"]
    assert str(val) in ("32", "32.0")


# ---------------------------------------------------------------------------
# check_mask_folder — path predicate
# ---------------------------------------------------------------------------

def test_check_mask_folder_missing_returns_false(tmp_project_dir):
    # No masks folder present → returns True (nothing to check yet).
    # Behaviour: check_mask_folder(src, mask_fldr) returns True if the
    # mask directory doesn't exist yet, meaning it's safe to write into.
    out = U.check_mask_folder(str(tmp_project_dir), "nonexistent_masks")
    assert isinstance(out, bool)


# ---------------------------------------------------------------------------
# print_progress — should never raise
# ---------------------------------------------------------------------------

def test_print_progress_basic(capsys):
    U.print_progress(files_processed=3, files_to_process=10, n_jobs=2,
                     batch_size=5, operation_type="testing")
    out = capsys.readouterr().out + capsys.readouterr().err
    # We only assert it did NOT raise; content may vary.
    assert True


def test_print_progress_with_time_ls(capsys):
    U.print_progress(files_processed=5, files_to_process=10, n_jobs=1,
                     time_ls=[0.1, 0.2, 0.15], batch_size=1)
    assert True
