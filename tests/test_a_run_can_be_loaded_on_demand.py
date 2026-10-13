"""Double-click a run, or right-click it and choose Load, and it loads.

REPORTED: "i also still cannot loade the second run the first run is
perpetually loaded. double clicking on the second run or right clicking on a
run and clicking load should have that run be laoded."

Both were literally absent. `doubleClicked` was connected to nothing, and the
row menu offered Remove / Open beside / Delete and no Load -- so the only route
to another run was a single click, and when that was refused for any reason
there was no second route at all.
"""

import spacr


import pandas as pd
import pytest


def _panel(qtbot):
    from spacr.qt.widgets.sweep_runs import SweepRunsPanel

    panel = SweepRunsPanel()
    qtbot.addWidget(panel)
    panel.set_frame(pd.DataFrame({
        "run": ["mixed_1", "ols_1"],
        "folder": ["/runs/mixed_1", "/runs/ols_1"],
        "status": ["ok", "ok"],
        "regression_type": ["mixed", "ols"],
    }))
    return panel


def test_double_click_loads_the_row(qtbot):
    """It was connected to nothing. This is the first half of the report.

    Driven by EMITTING the table's own doubleClicked, so the test fails if the
    connection is removed -- checking only that the handler exists would pass
    on a handler nothing calls, which is the bug that was there.
    """
    panel = _panel(qtbot)
    rows = panel._all_rows()
    second = [r for r in rows if r.get("run") == "ols_1"][0]
    # Select the second run, then double-click it.
    panel.set_loaded_run(panel._row_key(second))
    seen = []
    panel.loaded_run_changed.connect(seen.append)
    panel.trial_activated.connect(seen.append)

    table = panel.table.table
    for row in range(table.rowCount()):
        item = table.item(row, 0)
        if item is not None and "ols_1" in "".join(
                table.item(row, c).text() if table.item(row, c) else ""
                for c in range(table.columnCount())):
            table.selectRow(row)
            table.doubleClicked.emit(table.model().index(row, 0))
            break
    else:
        pytest.fail("the ols_1 row is not in the table")
    assert seen, "double-clicking a run told nobody"


def test_loading_the_second_run_announces_it(qtbot):
    panel = _panel(qtbot)
    seen = []
    panel.loaded_run_changed.connect(seen.append)
    panel.trial_activated.connect(seen.append)

    second = [r for r in panel._all_rows() if r.get("run") == "ols_1"][0]
    assert panel.load_this_run(second) is True
    assert seen, "loading a run told nobody"
    assert any(r.get("run") == "ols_1" for r in seen if isinstance(r, dict))


def test_asking_twice_still_announces(qtbot):
    """The stuck case: the mark already claims the run and the screen does not.

    `set_loaded_run` is idempotent and returns early, which is right for a run
    announcing itself and is exactly wrong for a user asking again. If the two
    have drifted apart, an idempotent load is the one that cannot repair it.
    """
    panel = _panel(qtbot)
    second = [r for r in panel._all_rows() if r.get("run") == "ols_1"][0]
    panel.load_this_run(second)

    seen = []
    panel.loaded_run_changed.connect(seen.append)
    panel.trial_activated.connect(seen.append)
    assert panel.load_this_run(second) is True
    assert seen, "asking for the run already marked loaded did nothing"


def test_the_menu_offers_load_first(qtbot):
    panel = _panel(qtbot)
    row = [r for r in panel._all_rows() if r.get("run") == "ols_1"][0]
    menu = panel._build_run_menu([row])
    verbs = [a.data() for a in menu.actions() if a.data()]
    assert "load" in verbs, verbs
    assert verbs[0] == "load", f"Load must come first, got {verbs}"


def test_the_menu_verb_loads(qtbot):
    panel = _panel(qtbot)
    row = [r for r in panel._all_rows() if r.get("run") == "ols_1"][0]
    seen = []
    panel.loaded_run_changed.connect(seen.append)
    panel.trial_activated.connect(seen.append)
    assert panel._apply_run_menu("load", [row]) is True
    assert seen
