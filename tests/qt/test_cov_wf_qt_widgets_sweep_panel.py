"""What the sweep panel does with well keys it did not have to compose.

:func:`spacr.qt.widgets.sweep_panel.sweep_inputs` has four ways in: the
measurements frame may already carry ``prc`` or may need it built, and the
score table may be keyed by ``prc``, by plate/row/column, or by nothing that
identifies a well at all. Every one of those decides which score lands on
which well, and getting it wrong is silent -- the join simply matches
nothing, the circularity column comes back all-NaN, and an all-NaN column
reads to the user as "nothing here restates the score", which is the most
confident possible way to say nothing.

The last test covers the other end of the panel: the gene the profile
picture is drawn about, when the row the user has selected does not name one.

Everything here uses the real ``sweep_inputs`` and, for the panel, the real
engine, so the frames asserted on are the frames the sweep is actually fed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")


# ---------------------------------------------------------------------------
# The measurements side: prc already there, or composed
# ---------------------------------------------------------------------------

@pytest.fixture()
def keyed_cells():
    """Measurements whose ``prc`` disagrees with its own plate/row/column.

    Deliberately contradictory: ``prc`` says three different wells on two
    plates while ``plateID``/``rowID``/``columnID`` say one well repeated
    three times. Whichever the function trusts is then visible in the answer
    instead of being a coincidence.
    """
    return pd.DataFrame({
        "prc": ["pplate1_r1_c1", "pplate1_r1_c2", "plate2_r3_c4"],
        "plateID": ["plateX"] * 3,
        "rowID": ["r9"] * 3,
        "columnID": ["c9"] * 3,
        "area": [1.0, 2.0, 3.0],
    })


@pytest.fixture()
def keyed_counts():
    """Guide fractions keyed the way the count CSVs key them."""
    return pd.DataFrame([
        {"prc": "plate1_r1_c1", "grna": "A", "fraction": 0.5},
        {"prc": "plate1_r1_c2", "grna": "A", "fraction": 0.25},
        {"prc": "plate2_r3_c4", "grna": "B", "fraction": 1.0},
    ])


def test_a_frame_that_already_has_prc_keeps_its_own_wells(keyed_cells,
                                                          keyed_counts):
    """The merged frame's own well keys win, and they are un-doubled first.

    A merged measurements frame arrives with ``prc`` already on it. If the
    panel rebuilt it from ``plateID``/``rowID``/``columnID`` anyway, three
    wells would collapse into one and every guide fraction would be averaged
    across wells that have nothing to do with each other -- a sweep over
    invented wells, reported with the same confidence as a real one. The
    ``pplate1`` prefix is collapsed on the way through, because that is the
    spelling a measurements database stamps and the counts say ``plate1``.
    """
    from spacr.qt.widgets.sweep_panel import sweep_inputs

    wells, fractions, plates, _found = sweep_inputs(keyed_cells, keyed_counts)

    assert list(wells.index) == ["plate1_r1_c1", "plate1_r1_c2",
                                 "plate2_r3_c4"]
    assert list(wells["area"]) == [1.0, 2.0, 3.0]
    assert list(plates) == ["plate1", "plate1", "plate2"]
    # The counts joined, which is the whole point of un-doubling the prefix.
    assert fractions.loc["plate1_r1_c1", "A"] == pytest.approx(0.5)
    assert fractions.loc["plate2_r3_c4", "B"] == pytest.approx(1.0)

    # And the contrast, so "kept its own" is a decision rather than the only
    # thing the function can do: without prc, the key IS composed, and these
    # three rows really are one well.
    composed, _f, composed_plates, _s = sweep_inputs(
        keyed_cells.drop(columns=["prc"]), keyed_counts)
    assert list(composed.index) == ["plateX_r9_c9"]
    assert list(composed_plates) == ["plateX"]
    assert composed.loc["plateX_r9_c9", "area"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# The score side: three ways a score table can be keyed
# ---------------------------------------------------------------------------

def test_a_score_table_keyed_by_prc_is_joined_on_the_key_it_has(keyed_cells,
                                                                keyed_counts):
    """A score CSV that already names its wells is not re-keyed from plateID.

    The score CSVs of a real run carry ``prc``. Composing a fresh key from
    the plate/row/column columns sitting beside it would overwrite a correct
    key with whatever those columns happen to hold -- here, one well repeated
    -- and the scores would land on the wrong wells rather than fail to land,
    which no error message would ever report.
    """
    from spacr.qt.widgets.sweep_panel import sweep_inputs

    by_prc = pd.DataFrame({
        "prc": ["pplate1_r1_c1", "pplate1_r1_c2", "plate2_r3_c4"],
        "plateID": ["plateX"] * 3,
        "rowID": ["r9"] * 3,
        "columnID": ["c9"] * 3,
        "pred": [0.1, 0.2, 0.3],
    })
    wells, _f, _p, found = sweep_inputs(keyed_cells, keyed_counts,
                                        scores=by_prc)
    assert list(found.index) == list(wells.index)
    assert list(found) == pytest.approx([0.1, 0.2, 0.3])

    # A table keyed the OTHER way -- no prc, plate/row/column only -- has its
    # key built, and reaches the same wells.
    by_plate = pd.DataFrame({
        "plateID": ["pplate1", "pplate1", "plate2"],
        "rowID": ["r1", "r1", "r3"],
        "columnID": ["c1", "c2", "c4"],
        "pred": [0.4, 0.5, 0.6],
    })
    _w, _f2, _p2, built = sweep_inputs(keyed_cells, keyed_counts,
                                       scores=by_plate)
    assert list(built.index) == list(wells.index)
    assert list(built) == pytest.approx([0.4, 0.5, 0.6])


def test_a_score_table_without_the_score_column_leaves_circularity_unknown(
        keyed_cells, keyed_counts):
    """No score column means no scores -- not a column of zeros.

    ``sweep`` reports circularity only when it was handed scores, and the
    panel hides rows above the circularity bar. Inventing a score series out
    of a table that does not contain one would hide real hits against a
    number nobody measured; returning nothing lets the panel say the
    circularity is unknown and keep the bar off.
    """
    from spacr.qt.widgets.sweep_panel import sweep_inputs

    without = pd.DataFrame({
        "prc": ["pplate1_r1_c1", "pplate1_r1_c2", "plate2_r3_c4"],
        "recall": [0.9, 0.8, 0.7],
    })
    _w, _f, _p, missing = sweep_inputs(keyed_cells, keyed_counts,
                                       scores=without)
    assert missing is None

    # The same table under the name the caller asked for DOES produce scores,
    # so the None above is the missing column and not a dead code path.
    named = without.rename(columns={"recall": "pred"})
    wells, _f2, _p2, found = sweep_inputs(keyed_cells, keyed_counts,
                                          scores=named)
    assert list(found.index) == list(wells.index)
    assert list(found) == pytest.approx([0.9, 0.8, 0.7])


def test_scores_with_no_well_key_are_refused_rather_than_joined_by_position(
        keyed_cells, keyed_counts):
    """Rows in file order are not wells; a positional join is a wrong answer.

    A score frame with a ``pred`` column and nothing that identifies a well
    cannot be attached to anything. Lining it up by position would look like
    a successful join and would silently attribute every score to whichever
    well happened to sort into that slot, which is worse than saying the
    circularity is unknown.
    """
    from spacr.qt.widgets.sweep_panel import sweep_inputs

    keyless = pd.DataFrame({"pred": [0.1, 0.2, 0.3]})
    _w, _f, _p, refused = sweep_inputs(keyed_cells, keyed_counts,
                                       scores=keyless)
    assert refused is None

    # Give those same three numbers a key and they arrive, so the refusal is
    # about the missing key and not about the frame.
    keyed = keyless.assign(prc=["pplate1_r1_c1", "pplate1_r1_c2",
                                "plate2_r3_c4"])
    wells, _f2, _p2, found = sweep_inputs(keyed_cells, keyed_counts,
                                          scores=keyed)
    assert list(found.index) == list(wells.index)
    assert list(found) == pytest.approx([0.1, 0.2, 0.3])


def test_a_score_column_in_the_measurements_beats_the_score_csv(keyed_cells,
                                                                keyed_counts):
    """When the merged frame already carries the score, that is the score.

    The measurements are what the sweep is actually run over. If a stale
    score CSV could override the column sitting in the frame, circularity
    would be computed against numbers that never touched these rows.
    """
    from spacr.qt.widgets.sweep_panel import sweep_inputs

    with_pred = keyed_cells.assign(pred=[0.11, 0.22, 0.33])
    csv = pd.DataFrame({"prc": ["plate1_r1_c1", "plate1_r1_c2",
                                "plate2_r3_c4"],
                        "pred": [9.0, 9.0, 9.0]})
    _w, _f, _p, found = sweep_inputs(with_pred, keyed_counts, scores=csv)
    assert list(found) == pytest.approx([0.11, 0.22, 0.33])


# ---------------------------------------------------------------------------
# The panel: which gene the profile picture is about
# ---------------------------------------------------------------------------

@pytest.fixture()
def swept(qtbot):
    """A panel that has really run a sweep over a small synthetic screen.

    Guide ``A`` moves ``real``; ``noise`` moves for nobody. The result is the
    engine's own, so the table the panel reads back is the table it ships.
    """
    from spacr.qt.widgets.sweep_panel import SweepPanel

    rng = np.random.default_rng(0)
    n = 80
    a = rng.random(n)
    cells = pd.DataFrame({
        "plateID": ["plate1"] * 40 + ["plate2"] * 40,
        "rowID": [f"r{i}" for i in range(n)],
        "columnID": ["c1"] * n,
        "real": a * 3.0 + rng.normal(0, 0.2, n),
        "noise": rng.normal(0, 1, n),
        "pred": rng.random(n),
    })
    counts = pd.DataFrame(
        [{"prc": f"plate{1 + i // 40}_r{i}_c1", "grna": g,
          "fraction": (a[i] if g == "A" else 1 - a[i])}
         for i in range(n) for g in ("A", "B")])
    widget = SweepPanel(lambda: cells, lambda: counts, threaded=False)
    qtbot.addWidget(widget)
    assert widget.start() is True
    assert widget._result is not None
    assert widget.table.rowCount() > 0
    return widget


def _put_gene(panel, text):
    """Write ``text`` into the gene cell of the first row, sorting off."""
    from spacr.qt.widgets.sortable_table import table_item

    panel.table.setSortingEnabled(False)
    panel.table.setItem(0, 1, table_item(text))
    panel.table.setSortingEnabled(True)


def test_a_selected_row_with_no_gene_in_it_falls_back_to_the_best_gene(swept):
    """A blank gene cell must not decide that there is no gene to draw.

    "One gene's fingerprint" is the one picture that needs a subject, and the
    subject is the selected row. A row can be selected while its gene cell
    holds nothing -- blank text, or no item at all where a column was never
    filled -- and treating that as "no gene" would answer the Show picture
    button with "nothing to draw" while a table full of survivors is on
    screen. The honest fallback is the strongest survivor, which the picture
    then names in its title.
    """
    best = str(swept._result.table.sort_values("q").iloc[0]["guide"])

    # A real name in the cell is what the user picked, and it is returned --
    # so the fallback below is a decision about blankness, not the only
    # answer this panel can give.
    _put_gene(swept, "a-gene-nobody-swept")
    swept.table.selectRow(0)
    assert swept.selected_gene() == "a-gene-nobody-swept"

    # Blank text in the selected row: the strongest survivor instead.
    _put_gene(swept, "   ")
    swept.table.selectRow(0)
    assert swept.selected_gene() == best

    # And no item at all in the gene column of the selected row.
    swept.table.setSortingEnabled(False)
    swept.table.takeItem(0, 1)
    swept.table.setSortingEnabled(True)
    swept.table.selectRow(0)
    assert swept.table.item(0, 1) is None
    assert swept.selected_gene() == best


def test_the_profile_picture_is_drawn_for_the_fallback_gene(swept):
    """The fallback reaches the picture, not just the accessor.

    If the blank-cell fallback stopped at :meth:`selected_gene`, the profile
    view would still refuse to draw and the user would see an unexplained
    "nothing to draw" for a gene the table is showing.
    """
    _put_gene(swept, "")
    swept.table.selectRow(0)
    figure = swept.figure(kind="profile")
    assert figure is not None
    best = str(swept._result.table.sort_values("q").iloc[0]["guide"])
    assert best in " ".join(text.get_text()
                            for axes in figure.axes
                            for text in [axes.title])
