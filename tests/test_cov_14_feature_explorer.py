"""The explorer says why it is empty instead of drawing an empty picture.

Three ways there is nothing to rank or draw, and each has to leave the panel
readable rather than blank:

* no table has been loaded yet, so the summary says so and the panel does not
  try to rank columns it does not have;
* a ranking that kept nothing still repaints the canvas, because a stale
  picture under a new empty result is a picture of the previous table;
* a feature whose column holds no finite value has no bin edges to draw on,
  and a bar chart drawn on no edges is a row of nothing with a title over it
  claiming a score.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def test_a_panel_with_no_table_says_so_and_ranks_nothing(qtbot):
    """Ranking before a table is loaded refuses and explains the refusal."""
    from spacr.qt.widgets.feature_explorer import FeatureExplorerPanel

    panel = FeatureExplorerPanel()
    qtbot.addWidget(panel)

    assert panel.rank_now() is None
    assert panel.summary() == "no table loaded"
    assert panel.result is None
