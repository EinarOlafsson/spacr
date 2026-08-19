"""The sweep, and the three mistakes it exists to stop a user making.

Instruction 175. Each was made by hand while sweeping ONE gene, in about ten
minutes, and each would be made again by anyone doing this themselves.
"""
import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_sweep import (SweepResult, is_measurement,
                                          measurement_columns, sweep)


@pytest.fixture()
def screen():
    """80 wells on 2 plates; guide A moves `real`, nothing moves `noise`."""
    rng = np.random.default_rng(0)
    n = 80
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    a = rng.random(n)
    wells = pd.DataFrame({
        "real": a * 3.0 + rng.normal(0, 0.2, n),
        "noise": rng.normal(0, 1, n),
        "object_label": np.arange(n),          # an identifier
        "pathogen_pathogen": np.arange(n) + 0.001,   # its twin, innocent name
    }, index=index)
    fractions = pd.DataFrame({"A": a, "B": rng.random(n)}, index=index)
    plates = [i.split("_")[0] for i in index]
    return wells, fractions, plates


# ------------------------------------------------- identifiers are not data


def test_an_identifier_is_not_a_measurement():
    assert not is_measurement("object_label")
    assert not is_measurement("pathogen_object_label")
    assert not is_measurement("plateID")
    assert is_measurement("pathogen_area")
    assert is_measurement("cell_channel_1_mean_intensity")


def test_a_column_that_DUPLICATES_an_identifier_is_dropped_too(screen):
    """`pathogen_pathogen` passes the name test and is the object label to
    four decimal places. The only way to catch it is to look."""
    wells, _fractions, _plates = screen

    kept = measurement_columns(wells)

    assert "pathogen_pathogen" not in kept
    assert "object_label" not in kept
    assert "real" in kept and "noise" in kept


def test_no_identifier_reaches_the_table(screen):
    wells, fractions, plates = screen

    result = sweep(wells, fractions, blocks=plates)

    assert set(result.table["measurement"]) <= {"real", "noise"}
    assert "object_label" in result.dropped


# --------------------------------------------------------- the correction


def test_the_table_carries_a_corrected_q(screen):
    wells, fractions, plates = screen

    result = sweep(wells, fractions, blocks=plates)

    assert (result.table["q"] >= result.table["p"] - 1e-12).all()
    assert result.survivors()["measurement"].tolist()[:1] == ["real"]


def test_a_screen_where_nothing_moves_produces_no_survivors():
    rng = np.random.default_rng(1)
    n = 60
    index = [f"plate1_r{i}_c1" for i in range(n)]
    wells = pd.DataFrame({"a": rng.normal(0, 1, n), "b": rng.normal(0, 1, n)},
                         index=index)
    fractions = pd.DataFrame({"G": rng.random(n)}, index=index)

    result = sweep(wells, fractions, blocks=["plate1"] * n)

    assert len(result.survivors()) == 0


# ------------------------------------------------------- effective sample


def test_a_guide_in_a_handful_of_wells_does_not_get_the_whole_screens_power():
    """A guide present in 7 of 1,366 wells reported p = 0.0 and sat at the top
    of the table. Its correlation is carried by those seven points."""
    rng = np.random.default_rng(2)
    n = 400
    index = [f"plate1_r{i}_c1" for i in range(n)]
    sparse = np.zeros(n); sparse[:7] = rng.random(7)
    wells = pd.DataFrame({"m": rng.normal(0, 1, n)}, index=index)
    wells.loc[wells.index[:7], "m"] += 5.0          # a strong but tiny signal
    fractions = pd.DataFrame({"rare": sparse}, index=index)

    result = sweep(wells, fractions, blocks=["plate1"] * n, min_wells=5)

    row = result.table.iloc[0]
    assert row["effective_wells"] < 30, (
        f"a 7-well guide was given {row['effective_wells']} wells of power")
    assert row["p"] > 0.0, "p = 0 from seven wells"


def test_the_effective_count_is_reported_beside_the_raw_one(screen):
    wells, fractions, plates = screen

    result = sweep(wells, fractions, blocks=plates)

    assert "n_wells" in result.table.columns
    assert "effective_wells" in result.table.columns


# ---------------------------------------------------------- circularity


def test_circularity_is_reported_when_the_score_joins(screen):
    wells, fractions, plates = screen
    score = pd.Series(wells["real"].to_numpy(), index=wells.index)

    result = sweep(wells, fractions, blocks=plates, scores=score)

    assert result.circularity_known
    real = result.table[result.table["measurement"] == "real"]
    assert real["circularity"].iloc[0] > 0.9


def test_a_score_that_joins_to_NOTHING_must_not_read_as_not_circular(screen):
    """The score CSVs say 'pplate1' where the databases say 'plate1', so an
    un-canonicalised join matches no well -- and the all-NaN column was
    reported as '0 of 5,959 hits are circular'."""
    wells, fractions, plates = screen
    nowhere = pd.Series([np.nan] * len(wells), index=wells.index)

    result = sweep(wells, fractions, blocks=plates, scores=nowhere)

    assert not result.circularity_known
    assert "NOT computed" in result.describe()
    with pytest.raises(ValueError, match="pplate1"):
        result.survivors(max_circularity=0.15)


def test_survivors_without_a_circularity_bar_still_work(screen):
    wells, fractions, plates = screen
    nowhere = pd.Series([np.nan] * len(wells), index=wells.index)

    result = sweep(wells, fractions, blocks=plates, scores=nowhere)

    assert len(result.survivors()) >= 0          # no raise


# ------------------------------------------------------------- the shape


def test_the_effect_grid_is_guides_by_measurements(screen):
    wells, fractions, plates = screen

    result = sweep(wells, fractions, blocks=plates)

    assert result.effects.shape == (2, 2)
    assert list(result.effects.index) == ["A", "B"]


def test_it_says_what_it_did(screen):
    wells, fractions, plates = screen

    text = sweep(wells, fractions, blocks=plates).describe()

    assert "measurement(s)" in text and "identifier column(s)" in text
