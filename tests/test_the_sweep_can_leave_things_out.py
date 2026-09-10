"""Instruction 175: the sweep can be told what to leave out.

Asked for 2026-08-19: "there should be the option to remove columns befor the
sweep and remove specific genes or guides and to remove over represented
guides".

THREE SEPARATE JUDGEMENTS, so three separate controls: a column you do not
trust, a gene you already know about, and a guide whose BREADTH is doing the
work its biology is being credited with. The third is the one the maintainer
raised first -- "220950 is waaay over represented" -- and it is not the same
question as the second.
"""
import numpy as np
import pandas as pd
import pytest

from spacr.gene_measurement_sweep import sweep


@pytest.fixture()
def screen():
    """120 wells, 3 plates. 999 is in EVERY well; 111 carries a real signal."""
    rng = np.random.default_rng(0)
    n = 120
    index = [f"plate{1 + i // 40}_r{i}_c1" for i in range(n)]
    signal = rng.random(n)
    wells = pd.DataFrame({
        "pathogen_area": signal * 5 + rng.normal(0, 0.3, n),
        "cell_area": rng.normal(0, 1, n),
        "nucleus_area": rng.normal(0, 1, n),
    }, index=index)
    fractions = pd.DataFrame({
        "TGGT1_111_1": signal * 0.5,
        "TGGT1_222_1": rng.random(n),
        "TGGT1_999_1": np.full(n, 0.2),        # in every well
    }, index=index)
    return wells, fractions, [i.split("_")[0] for i in index]


def _genes(result):
    return set(result.table["guide"].astype(str))


def _measures(result):
    return set(result.table["measurement"].astype(str))


# ------------------------------------------------------------- measurements


def test_a_named_measurement_is_left_out(screen):
    wells, fractions, plates = screen
    out = sweep(wells, fractions, blocks=plates, level="gene",
                drop_measurements=["cell_area"])

    assert "cell_area" not in _measures(out)
    assert "pathogen_area" in _measures(out)


def test_naming_a_column_that_is_not_there_is_not_an_error(screen):
    """A settings file outlives the screen it was written for."""
    wells, fractions, plates = screen
    out = sweep(wells, fractions, blocks=plates, level="gene",
                drop_measurements=["no_such_column"])

    assert _measures(out)


# -------------------------------------------------------- genes and guides


def test_a_named_gene_is_left_out(screen):
    wells, fractions, plates = screen
    out = sweep(wells, fractions, blocks=plates, level="gene",
                drop_guides=["222"])

    assert "222" not in _genes(out)
    assert "111" in _genes(out)


def test_a_gene_id_matches_at_guide_level_too(screen):
    """"Matched at BOTH levels": a user typing a gene id should not have to
    know which level the sweep happens to be running at."""
    wells, fractions, plates = screen
    out = sweep(wells, fractions, blocks=plates, level="guide",
                drop_guides=["222"])

    assert not any("222" in g for g in _genes(out))
    assert any("111" in g for g in _genes(out))


# ------------------------------------------------------ over-representation


def test_a_guide_in_too_many_wells_is_left_out(screen):
    wells, fractions, plates = screen
    out = sweep(wells, fractions, blocks=plates, level="gene",
                max_wells_fraction=0.9)

    assert "999" not in _genes(out), (
        "the guide present in every well survived the breadth filter")


def test_breadth_and_share_are_different_filters(screen):
    """A guide can be in EVERY well at a low fraction, or in a handful at a
    high one. Measured on the maintainer's screen 220950 is extreme on both,
    which is exactly why one number cannot stand for the other."""
    wells, fractions, plates = screen
    fractions = fractions.copy()
    fractions["TGGT1_777_1"] = 0.0
    # In three wells only, but taking most of each.
    fractions.iloc[:3, fractions.columns.get_loc("TGGT1_777_1")] = 0.8

    wide = sweep(wells, fractions, blocks=plates, level="gene",
                 min_wells=1, max_wells_fraction=0.9)
    deep = sweep(wells, fractions, blocks=plates, level="gene",
                 min_wells=1, max_share=0.5)

    assert "777" in _genes(wide), "a rare guide was caught by a BREADTH filter"
    assert "777" not in _genes(deep), "a concentrated guide survived max_share"


def test_no_filter_leaves_everything(screen):
    wells, fractions, plates = screen
    out = sweep(wells, fractions, blocks=plates, level="gene")

    assert {"111", "222", "999"} <= _genes(out)


def test_what_was_left_out_is_said(screen, capsys):
    """A sweep that quietly dropped a gene the user was looking for would
    send them hunting through the table for a row that was never computed."""
    wells, fractions, plates = screen
    sweep(wells, fractions, blocks=plates, level="gene", drop_guides=["222"])

    said = capsys.readouterr().out
    assert "left out" in said
    assert "222" in said


# ---------------------------------------------------------------- the panel


def test_the_panel_offers_all_three(qtbot):
    from spacr.qt.widgets.sweep_panel import SweepPanel

    panel = SweepPanel()
    qtbot.addWidget(panel)

    assert panel.exclusions() == {}, "a fresh panel excludes nothing"

    panel.drop_columns.setText("cell_area, plateID")
    panel.drop_genes.setText("220950")
    panel.cap_wells.setChecked(True)
    panel.cap_wells_value.setValue(0.4)

    asked = panel.exclusions()
    assert asked["drop_measurements"] == ["cell_area", "plateID"]
    assert asked["drop_guides"] == ["220950"]
    assert asked["max_wells_fraction"] == pytest.approx(0.4)


def test_a_blank_box_is_an_absent_key_not_an_empty_list(qtbot):
    """`sweep` distinguishes "no filter" from "a filter that matches
    nothing", and the second would keep everything while looking like a
    choice."""
    from spacr.qt.widgets.sweep_panel import SweepPanel

    panel = SweepPanel()
    qtbot.addWidget(panel)
    panel.drop_columns.setText("   ,  ")

    assert "drop_measurements" not in panel.exclusions()
