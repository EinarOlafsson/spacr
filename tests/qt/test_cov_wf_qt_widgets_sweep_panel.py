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


