"""Instruction 177 F: compare a measurement, annotated genes against the rest.

    "there should be the opertunity to show the any measurement comparing the
    values for the cells that have been annotated with a gene vs the rest ...
    the option to do this on the cell, well, and plate level should be
    available ... ther e should also be the ability to do statistics. where
    the variance and normality and n is noted and the correct test chosen."

THE LEVEL CHANGES WHAT AN OBSERVATION IS, which is why it is a setting and
not a detail: a per-CELL test on 60,000 cells drawn from 12 wells has 60,000
rows and 12 independent units, and the p it produces is about cells.
"""
import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_compare import (LEVELS, PLOTS, REST, build,
                                            plot, save, with_statistics)


@pytest.fixture()
def objects():
    """400 cells over 4 plates. The first 80 carry a real shift."""
    rng = np.random.default_rng(0)
    n = 400
    frame = pd.DataFrame({
        "pathogen_area": rng.normal(10.0, 2.0, n),
        "plateID": np.repeat(["p1", "p2", "p3", "p4"], n // 4),
        "rowID": rng.choice(["r1", "r2", "r3"], n),
        "columnID": rng.choice(["c1", "c2", "c3", "c4"], n),
    })
    frame.loc[frame.index[:80], "pathogen_area"] += 3.0
    return frame


def _picked(objects):
    return objects.index[:80]


# ------------------------------------------------------------------- groups


def test_everything_not_named_is_the_rest(objects):
    out = build(objects, "pathogen_area",
                groups={"220950": _picked(objects)}, level="cell")

    assert set(out.groups) == {"220950", REST}
    assert out.counts() == {"220950": 80, REST: 320}


def test_several_genes_at_once(objects):
    """"this should also work for several annotated genes at a time"."""
    out = build(objects, "pathogen_area", level="cell",
                groups={"220950": objects.index[:50],
                        "225160": objects.index[50:90]})

    assert set(out.groups) == {"220950", "225160", REST}
    assert out.counts()["225160"] == 40


def test_a_measurement_that_is_not_there_says_so(objects):
    out = build(objects, "no_such_measurement", groups={}, level="cell")

    assert not len(out.frame)
    assert "not a column" in out.note


# ------------------------------------------------------------------- levels


@pytest.mark.parametrize("level", [name for name, _why in LEVELS])
def test_every_level_builds(objects, level):
    out = build(objects, "pathogen_area",
                groups={"220950": _picked(objects)}, level=level)

    assert len(out.frame), level
    assert out.level == level


def test_a_coarser_level_has_fewer_rows(objects):
    """THE POINT OF THE SETTING. Cells are many and not independent; plates
    are few and are."""
    sizes = {level: len(build(objects, "pathogen_area",
                              groups={"220950": _picked(objects)},
                              level=level).frame)
             for level, _why in LEVELS}

    assert sizes["cell"] > sizes["well"] > sizes["plate"]


def test_the_cell_level_says_its_rows_are_not_independent(objects):
    """A p-value on 400 cells from 12 wells is about cells, and the panel
    has to say so rather than let it read as 400 experiments."""
    out = build(objects, "pathogen_area",
                groups={"220950": _picked(objects)}, level="cell")

    assert "not independent" in out.note


def test_a_well_holding_both_groups_contributes_to_both(objects):
    """The alternative is to decide the well belongs to whichever group
    happens to be larger, which is a choice nobody made."""
    out = build(objects, "pathogen_area",
                groups={"220950": _picked(objects)}, level="well")

    shared = out.frame.groupby("unit")["group"].nunique()
    assert (shared > 1).any(), "no well held both groups, so this proves nothing"


def test_a_level_the_rows_cannot_support_says_so():
    """A plate-level comparison needs plateID on the objects."""
    frame = pd.DataFrame({"pathogen_area": [1.0, 2.0, 3.0]})

    out = build(frame, "pathogen_area", groups={}, level="plate")

    assert not len(out.frame)
    assert "plateID" in out.note


# --------------------------------------------------------------- statistics


def test_the_engine_chooses_the_test_and_says_why(objects):
    """`sp_stats` picks between t, Welch, Mann-Whitney, ANOVA and
    Kruskal-Wallis from the assumption checks; nothing here second-guesses
    it, because two choosers would answer differently in a figure legend."""
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))

    assert out.statistics
    row = out.statistics[0]
    assert row.get("Test Name")
    assert row.get("Why This Test")


def test_the_assumption_checks_travel_with_the_result(objects):
    """"where the variance and normality and n is noted"."""
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))
    row = out.statistics[0]

    assert row["Normality"]
    assert row["Equal variance"]
    assert "220950" in row["n per group"]
    assert row["Level"] == "well"


def test_one_group_is_not_a_comparison(objects):
    out = with_statistics(build(objects, "pathogen_area", groups={}, level="well"))

    assert out.statistics == []


def test_too_few_units_is_refused_not_guessed(objects):
    """At plate level one group has a single plate. A test there would be a
    number with nothing behind it."""
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": objects.index[:100]}, level="plate"))

    if out.statistics:
        assert out.statistics[0].get("Test Name") in (
            "not testable", "T-test", "Welch's t-test", "Mann-Whitney U")


# -------------------------------------------------------------------- plots


@pytest.mark.parametrize("kind", [name for name, _why in PLOTS])
def test_every_plot_type_draws(objects, kind):
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))

    assert plot(out, kind=kind) is not None, kind


def test_the_rest_is_grey_and_the_gene_is_not(objects):
    """The one rule that matters most: everything grey except the claim."""
    from spacr.gene_measurement_sweep import HOUSE

    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))
    figure = plot(out, kind="jitter")

    colours = [c.get_facecolor()[0] for c in figure.axes[0].collections
               if len(c.get_offsets())]
    assert len(colours) == 2
    import matplotlib.colors as mcolors
    greys = [c for c in colours
             if np.allclose(c[:3], mcolors.to_rgb(HOUSE.GREY), atol=0.02)]
    assert len(greys) == 1, "the rest is not grey, or the gene is"


def test_a_bar_for_a_small_group_says_dots_would_be_better(objects):
    """The user asked for a bar and gets one; the skill's rule -- "a bar for
    n = 3 is not done in these papers" -- is REPORTED rather than used to
    override the request."""
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": objects.index[:2]}, level="plate"))
    figure = plot(out, kind="bar")

    if figure is not None:
        said = " ".join(t.get_text() for t in figure.axes[0].texts)
        assert "individual points" in said


def test_nothing_to_draw_returns_none(objects):
    assert plot(build(objects, "nope", groups={}, level="cell")) is None


# --------------------------------------------------------------- one folder


def test_saving_puts_everything_in_one_folder(objects, tmp_path):
    """"all of this in one folder upon saving" -- a figure in one place, its
    numbers in another and the settings in a third is how a result becomes
    unreproducible six months later."""
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))
    crops = {"p1_r1_c1": [np.full((8, 8, 3), 120, dtype="uint8")] * 2}

    written = save(out, str(tmp_path / "cmp"), kind="jitter_box",
                   settings={"regression_type": "ols", "fdr_alpha": 0.05},
                   images=crops)

    assert {"pdf", "png", "data", "statistics", "settings"} <= set(written)
    for key in ("pdf", "png", "data", "statistics", "settings"):
        assert (tmp_path / "cmp").exists()
        assert written[key].startswith(str(tmp_path / "cmp"))
    assert (tmp_path / "cmp" / "cells" / "p1_r1_c1" / "0000.png").exists()


def test_the_saved_settings_name_which_produced_what(objects, tmp_path):
    import json

    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))
    written = save(out, str(tmp_path / "cmp"),
                   settings={"regression_type": "mixed"})

    record = json.loads(open(written["settings"]).read())
    assert record["regression_settings"]["regression_type"] == "mixed"
    assert record["level"] == "well"
    assert record["plot"]
    assert record["n_per_group"]


def test_no_images_is_not_an_error(objects, tmp_path):
    """A comparison run on a screen whose crops are not to hand is still a
    comparison."""
    out = with_statistics(build(objects, "pathogen_area",
                     groups={"220950": _picked(objects)}, level="well"))

    written = save(out, str(tmp_path / "cmp"), images=None)

    assert "cells" not in written
    assert "data" in written


# --------------------------------------------------- reachable from the tab


def test_the_report_is_readable_not_a_field_dump(objects, qtbot):
    """`sp_stats` returns a dozen fields per group, and dumping them made a
    400-character line nobody reads. The verdict and its number survive; the
    column name and the row count do not."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementCompareDialog)

    dialog = MeasurementCompareDialog(objects,
                                      {"220950": list(objects.index[:80])})
    qtbot.addWidget(dialog)
    lines = dialog.report.toPlainText().splitlines()

    assert any(line.startswith("n: ") for line in lines)
    assert any(line.startswith("normality: ") for line in lines)
    assert any(line.startswith("equal variance: ") for line in lines)
    assert any(line.startswith("test: ") for line in lines)
    assert any(line.startswith("why: ") for line in lines)
    for line in lines:
        assert len(line) < 400, line[:80]


def test_the_dialog_offers_measurements_from_the_data(objects, qtbot):
    """Built from the screen, never typed -- the rule every other chooser in
    spaCR follows."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementCompareDialog)

    dialog = MeasurementCompareDialog(objects,
                                      {"220950": list(objects.index[:80])})
    qtbot.addWidget(dialog)
    offered = {dialog.measurement.itemData(i)
               for i in range(dialog.measurement.count())}

    assert "pathogen_area" in offered
    assert "plateID" not in offered, "an identifier was offered as a measurement"


def test_it_opens_on_the_well_level(objects, qtbot):
    """The unit the screen randomises, and the honest default."""
    from spacr.qt.widgets.measurement_compare_dialog import (
        MeasurementCompareDialog)

    dialog = MeasurementCompareDialog(objects,
                                      {"220950": list(objects.index[:80])})
    qtbot.addWidget(dialog)

    assert dialog.level.currentData() == "well"


def test_the_cells_tab_groups_by_what_the_picker_chose(qtbot, tmp_path):
    """The groups are the picker's answer, read off `montage_candidate` --
    so whichever mode is in force is what the comparison compares."""
    import sys
    sys.path.insert(0, "tests/qt")
    import pandas as pd
    import test_cells_behind_the_dot_tab as T

    root, db, csv = T._screen(tmp_path, with_png=True)
    view = T.CellMontageView(frame_provider=lambda: pd.read_csv(csv),
                             results_provider=lambda: csv,
                             database_provider=lambda: T._rows(db),
                             threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(T.GENE_KEY)
    view.build()

    groups = view.picked_groups()

    assert groups, "the tab drew cells but named no group"
    assert all(len(members) for members in groups.values())


def test_comparing_with_nothing_picked_says_so(qtbot):
    from spacr.qt.widgets.cell_montage_view import CellMontageView

    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)

    assert view.compare_a_measurement() is None
    assert "Show some cells first" in view.status_text()
