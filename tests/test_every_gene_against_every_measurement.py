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


# ------------------------------------------------------------- the picture


def test_nothing_surviving_draws_nothing(screen):
    """A picture of noise is worse than no picture."""
    from spacr.gene_measurement_sweep import plot_sweep

    rng = np.random.default_rng(3)
    n = 60
    index = [f"plate1_r{i}_c1" for i in range(n)]
    wells = pd.DataFrame({"a": rng.normal(0, 1, n)}, index=index)
    fractions = pd.DataFrame({"G": rng.random(n)}, index=index)
    result = sweep(wells, fractions, blocks=["plate1"] * n)

    assert plot_sweep(result) is None


def test_it_draws_what_survived(screen, tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from spacr.gene_measurement_sweep import plot_sweep

    wells, fractions, plates = screen
    result = sweep(wells, fractions, blocks=plates)

    out = tmp_path / "sweep.png"
    figure = plot_sweep(result, path=str(out))

    assert figure is not None
    assert out.exists() and out.stat().st_size > 1000


def test_the_grid_is_ordered_so_neighbours_are_alike():
    """A heatmap in arrival order hides every block structure in it."""
    from spacr.gene_measurement_sweep import _order_like_neighbours

    # Two blocks of correlated rows, interleaved on the way in.
    grid = pd.DataFrame(
        [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1],
         [1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]],
        index=["a1", "b1", "a2", "b2"], dtype=float)

    ordered = _order_like_neighbours(grid)

    names = list(ordered.index)
    assert abs(names.index("a1") - names.index("a2")) == 1
    assert abs(names.index("b1") - names.index("b2")) == 1


# ---------------------------------------------------------------- gene level


def test_a_guide_name_maps_to_its_gene_in_both_spellings():
    """`hits.gene_of` reads a DESIGN TERM and truncates at the first
    underscore. Handed the bare `TGGT1_225160_2` a count table carries, that
    returns `TGGT1` -- the organism, for every guide in the screen."""
    from spacr.gene_measurement_sweep import gene_of_guide

    assert gene_of_guide("TGGT1_225160_2") == "225160"
    assert gene_of_guide("225160_2") == "225160"
    assert gene_of_guide("fraction:grna[225160_1]") == "225160"
    assert gene_of_guide("TGME49_239740_3") == "239740"


def test_a_genes_fraction_is_the_sum_of_its_guides():
    """The same rule the regression applies, so 'does this gene move this
    measurement' is not a different arithmetic from the fit."""
    from spacr.gene_measurement_sweep import gene_fractions

    guides = pd.DataFrame(
        {"TGGT1_225160_1": [0.1, 0.2], "TGGT1_225160_2": [0.2, 0.1],
         "TGGT1_239740_1": [0.3, 0.3]},
        index=["plate1_r1_c1", "plate1_r1_c2"])

    genes = gene_fractions(guides)

    assert sorted(genes.columns) == ["225160", "239740"]
    assert genes["225160"].tolist() == pytest.approx([0.3, 0.3])


def test_the_sweep_can_run_at_gene_level(screen):
    wells, fractions, plates = screen
    fractions = fractions.rename(columns={"A": "TGGT1_111_1",
                                          "B": "TGGT1_222_1"})

    result = sweep(wells, fractions, blocks=plates, level="gene")

    assert set(result.table["guide"]) == {"111", "222"}
    assert set(result.table["level"]) == {"gene"}


def test_both_keeps_the_guide_rows_reachable(screen):
    wells, fractions, plates = screen
    fractions = fractions.rename(columns={"A": "TGGT1_111_1",
                                          "B": "TGGT1_222_1"})

    result = sweep(wells, fractions, blocks=plates, level="both")

    assert set(result.table["level"]) == {"gene", "guide"}
    assert "TGGT1_111_1" in set(result.table["guide"])
    assert "111" in set(result.table["guide"])


def test_a_gene_and_its_guide_are_different_rows(screen):
    """`233460` the gene and `233460_1` the guide must not collide."""
    wells, fractions, plates = screen
    fractions = fractions.rename(columns={"A": "TGGT1_111_1",
                                          "B": "TGGT1_222_1"})

    result = sweep(wells, fractions, blocks=plates, level="both")
    rows = result.table[result.table["measurement"] == "real"]

    assert len(rows) == len(set(zip(rows["level"], rows["guide"])))


def test_an_unknown_level_is_refused(screen):
    wells, fractions, plates = screen

    with pytest.raises(ValueError, match="level must be"):
        sweep(wells, fractions, blocks=plates, level="nonsense")


def test_the_picture_draws_one_level_not_both(screen, tmp_path):
    """A gene row and its own guide rows drawn together are the same effect
    counted several times -- a block of near-identical rows that reads as
    agreement between independent things."""
    import matplotlib
    matplotlib.use("Agg")
    from spacr.gene_measurement_sweep import plot_sweep

    wells, fractions, plates = screen
    fractions = fractions.rename(columns={"A": "TGGT1_111_1",
                                          "B": "TGGT1_222_1"})
    result = sweep(wells, fractions, blocks=plates, level="both")

    figure = plot_sweep(result, path=str(tmp_path / "s.png"))

    assert figure is not None
    drawn = [t.get_text() for t in figure.axes[0].get_yticklabels()]
    assert not any("_" in name for name in drawn), (
        f"guide rows reached a gene-level picture: {drawn}")


# ------------------------------------------------- representation and controls


def test_representation_is_on_every_row(screen):
    """Measured on the real screen: 220950 sits in ALL 1,536 wells at a median
    fraction of 0.176 while the median gene is in 73. Ranking by q therefore
    ranks by representation as much as by biology, and a reader cannot see
    that unless it is on the row."""
    wells, fractions, plates = screen

    result = sweep(wells, fractions, blocks=plates)

    assert "share" in result.table.columns
    assert "ubiquitous" in result.table.columns
    assert (result.table["share"] > 0).any()


def test_a_gene_in_every_well_is_flagged():
    rng = np.random.default_rng(4)
    n = 60
    index = [f"plate1_r{i}_c1" for i in range(n)]
    wells = pd.DataFrame({"m": rng.normal(0, 1, n)}, index=index)
    everywhere = rng.random(n) * 0.2 + 0.1
    somewhere = np.where(np.arange(n) < 10, rng.random(n), 0.0)
    fractions = pd.DataFrame({"all": everywhere, "few": somewhere}, index=index)

    result = sweep(wells, fractions, blocks=["plate1"] * n)
    flags = dict(zip(result.table["guide"], result.table["ubiquitous"]))

    assert flags["all"] is True or flags["all"] == True   # noqa: E712
    assert not flags["few"]


def test_controls_are_MARKED_and_not_removed(screen):
    """The regression drops the control COLUMNS because they are not part of
    the contrast it fits. This asks whether a gene moves a measurement, and a
    control is exactly the thing whose answer you want to see."""
    wells, fractions, plates = screen

    result = sweep(wells, fractions, blocks=plates, controls=["A"])

    marked = result.table[result.table["control"]]
    assert set(marked["guide"]) == {"A"}
    # Present, not filtered away.
    assert "A" in set(result.table["guide"])
    assert "B" in set(result.table["guide"])
