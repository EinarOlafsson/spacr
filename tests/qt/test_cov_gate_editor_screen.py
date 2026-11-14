"""What the Gate Editor SCREEN does with the gates the canvas produced.

The canvas draws them; this is everything that happens afterwards -- writing
them back into the database as filter columns, turning the shown ones into an
annotation, saving the strategy to a file and loading it against a different
table, and saving the picture.

The load-bearing property throughout is that none of these steps is allowed to
be silent. Every one of them can fail for a reason the user can act on -- the
table came from a CSV so there is nowhere to write, one gate names a
measurement this database has not got, a strategy was saved against another
plate -- and each of those has to arrive as a sentence naming the thing that
went wrong, not as an empty label or a traceback.

Jobs run unthreaded, which :class:`~spacr.qt.job_runner.JobRunner` supports
explicitly: the same signals in the same order, inline.
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.gate_editor import GateEditorScreen      # noqa: E402
from spacr.qt.widgets.gate_spec import (                        # noqa: E402
    GateSet, RectGate,
)


# ---------------------------------------------------------------------------
# A measurement database with a real object identity
# ---------------------------------------------------------------------------

def _objects(n=8):
    """One object table, with the identity the filters table merges on."""
    return pd.DataFrame({
        "plateID": ["p1"] * n,
        "rowID": ["A"] * n,
        "columnID": ["1"] * n,
        "fieldID": ["f1"] * n,
        "object_label": list(range(1, n + 1)),
        "area": np.linspace(10.0, 80.0, n),
        "intensity": np.linspace(100.0, 800.0, n),
    })


@pytest.fixture
def database(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        _objects().to_sql("cell", db, index=False)
    return path


@pytest.fixture
def screen(qtbot):
    widget = GateEditorScreen(threaded=False)
    qtbot.addWidget(widget)
    return widget


@pytest.fixture
def loaded(screen, database):
    """The screen as it is after a database has been read and gated."""
    screen._path = database
    screen._table = "cell"
    screen.set_frame(_objects())
    screen.gates.set_gates(GateSet().add(
        RectGate(name="big", x_column="area", y_column="intensity",
                 x_low=30.0, x_high=90.0, y_low=0.0, y_high=1000.0)))
    return screen


class _Dialogs:
    """Every modal the screen can raise, recorded instead of shown."""

    def __init__(self, monkeypatch, module):
        self.shown = []
        self.item = (None, False)
        self.text = ("", False)
        self.save_path = ""
        self.open_path = ""

        class _Box:
            @staticmethod
            def information(_parent, title, body):
                self.shown.append(("information", title, body))

            @staticmethod
            def warning(_parent, title, body):
                self.shown.append(("warning", title, body))

        class _Input:
            @staticmethod
            def getItem(*_a, **_k):
                return self.item

            @staticmethod
            def getText(*_a, **_k):
                return self.text

        import PySide6.QtWidgets as qtw
        monkeypatch.setattr(qtw, "QMessageBox", _Box)
        monkeypatch.setattr(qtw, "QInputDialog", _Input)
        monkeypatch.setattr(
            module.QFileDialog, "getSaveFileName",
            staticmethod(lambda *a, **k: (self.save_path, "")))
        monkeypatch.setattr(
            module.QFileDialog, "getOpenFileName",
            staticmethod(lambda *a, **k: (self.open_path, "")))

    @property
    def titles(self):
        return [title for _kind, title, _body in self.shown]


@pytest.fixture
def dialogs(monkeypatch):
    import spacr.qt.screens.gate_editor as module

    return _Dialogs(monkeypatch, module)


# ---------------------------------------------------------------------------
# Writing the gates back to the database
# ---------------------------------------------------------------------------

def test_exporting_with_nothing_drawn_says_so(screen, dialogs):
    screen.export_gates()
    assert dialogs.titles == ["No gates"]


def test_an_export_that_wrote_nothing_says_nothing_written(screen):
    """Not an empty label. `_on_exported` is the only thing that reports the
    result, and a blank one reads as the export still running."""
    screen._on_exported(([], [("g", "no such column")]))
    assert screen._source.text() == (
        "nothing written · could not export g (no such column)")


# ---------------------------------------------------------------------------
# Saving and loading the strategy
# ---------------------------------------------------------------------------

def test_a_file_that_is_not_a_strategy_is_reported_rather_than_raised(
        screen, tmp_path):
    """Pointing the loader at the wrong file is an ordinary slip, and the
    screen has to survive it with a sentence rather than a traceback out of
    a button press."""
    path = tmp_path / "not_gates.json"
    path.write_text("{not json at all", encoding="utf-8")

    assert screen.load_gates(str(path)) is False
    assert "could not load those gates" in screen._source.text()


def test_a_json_that_simply_has_no_gates_in_it_reports_the_zero(screen,
                                                                tmp_path):
    """It loads -- a strategy with nothing in it is a legitimate file -- and
    the count is what says the user picked the wrong one."""
    path = tmp_path / "other.json"
    path.write_text('{"something": "else"}', encoding="utf-8")

    assert screen.load_gates(str(path)) is True
    assert screen._source.text() == "0 gate(s) from other.json"


# ---------------------------------------------------------------------------
# Saving and loading the filter set
# ---------------------------------------------------------------------------

def test_a_filter_file_that_cannot_be_read_is_reported(screen, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json at all", encoding="utf-8")

    assert screen.load_filters(str(path)) == []
    assert "could not load that filter set" in screen._source.text()


# ---------------------------------------------------------------------------
# Saving the picture
# ---------------------------------------------------------------------------

def test_saving_a_graph_before_anything_is_drawn_says_so(screen):
    """The console, not a modal: nothing has gone wrong, there is simply
    nothing there yet."""
    assert screen.save_graph("/tmp/never-written.png") == ""


# ---------------------------------------------------------------------------
# A read that failed
# ---------------------------------------------------------------------------

def test_a_table_that_could_not_be_read_names_the_file_and_the_reason(screen):
    """The runner's `job_failed` lands here, and a status line saying only
    "failed" leaves the user with nothing to fix."""
    screen._path = "/data/plate1/measurements.db"
    screen._on_load_failed("no such table: cell")
    assert screen._source.text() == (
        "could not read measurements.db: no such table: cell")


# ---------------------------------------------------------------------------
# Getting a table on screen
# ---------------------------------------------------------------------------

def test_choosing_one_file_loads_it_and_several_merge(screen, dialogs,
                                                      monkeypatch, database):
    """One file behaves exactly as it did before the screen learned to take
    three plates at once, which is the whole compatibility claim of that
    change."""
    import spacr.qt.screens.gate_editor as module

    chosen = []
    monkeypatch.setattr(screen, "load_path", lambda p, t=None: chosen.append(p))
    monkeypatch.setattr(screen, "load_paths", lambda p, t=None: chosen.append(p))

    for offered in ([], [database], [database, database]):
        monkeypatch.setattr(
            module.QFileDialog, "getOpenFileNames",
            staticmethod(lambda *a, _o=offered, **k: (_o, "")))
        screen.choose_table()

    assert chosen == [database, [database, database]], (
        "cancelling the dialog loaded something")


# ---------------------------------------------------------------------------
# The right-click menu on the plot
# ---------------------------------------------------------------------------


class _Menu:
    """A QMenu that records instead of popping up.

    A real one cannot be used here: an offscreen Qt still runs `exec`'s
    nested event loop, with nobody to dismiss the popup, and the run hangs
    for as long as anyone lets it. Patching `QMenu.exec` does not help --
    `_show_graph_menu` imports the class inside the function, so the class
    itself is what gets replaced.
    """

    built = []

    def __init__(self, _parent=None):
        self.rows = []
        self.at = None
        _Menu.built.append(self)

    def addSeparator(self):                      # noqa: N802 - Qt name
        self.rows.append(None)

    def addAction(self, label):                  # noqa: N802 - Qt name
        action = _Action(label)
        self.rows.append(action)
        return action

    def exec(self, point):
        self.at = point


class _Action:
    def __init__(self, label):
        self.label = label
        self.enabled = True
        self.tooltip = ""
        self.calls = []

    def setEnabled(self, on):                    # noqa: N802 - Qt name
        self.enabled = bool(on)

    def setToolTip(self, text):                  # noqa: N802 - Qt name
        self.tooltip = text

    @property
    def triggered(self):
        return self

    def connect(self, slot):
        self.calls.append(slot)


def test_a_greyed_row_carries_its_reason_into_the_menu(screen, monkeypatch):
    """The tooltip is the only place the reason can be read once the item is
    in the menu, and a greyed row with no reason looks like a bug."""
    from PySide6.QtCore import QPoint
    import PySide6.QtWidgets as qtw
    from spacr.qt.widgets import figure_settings

    _Menu.built = []
    monkeypatch.setattr(qtw, "QMenu", _Menu)
    monkeypatch.setattr(figure_settings, "_add_figure_tools",
                        lambda *_args, **_kwargs: None)

    screen._show_graph_menu(QPoint(0, 0))

    rows = {row.label: row for row in _Menu.built[0].rows if row is not None}
    assert rows["Save graph…"].enabled is False
    assert rows["Save graph…"].tooltip == "Draw a graph first."
    assert rows["Save graph…"].calls == [], (
        "a disabled item was still wired to fire")


@pytest.mark.parametrize("raises", [False, True])
def test_graph_menu_remains_usable_without_a_figure(screen, monkeypatch, raises):
    import PySide6.QtWidgets as qtw
    from PySide6.QtCore import QPoint

    from spacr.qt.widgets import figure_settings

    def figure():
        if raises:
            raise RuntimeError("canvas is closing")
        return None

    _Menu.built = []
    monkeypatch.setattr(qtw, "QMenu", _Menu)
    monkeypatch.setattr(screen, "axis_under", lambda _point: None)
    monkeypatch.setattr(screen.gates.canvas, "figure", figure)
    figure_tools = []
    monkeypatch.setattr(figure_settings, "_add_figure_tools",
                        lambda *_args: figure_tools.append(True))
    screen._show_graph_menu(QPoint(2, 3))

    menu = _Menu.built[0]
    rows = {row.label: row for row in menu.rows if row is not None}
    assert menu.at == screen.gates.canvas.mapToGlobal(QPoint(2, 3))
    assert rows["Reset view"].calls
    assert rows["Graph settings…"].calls
    assert figure_tools == []


# ---------------------------------------------------------------------------
# Gating on components instead of measurements
# ---------------------------------------------------------------------------

def _wide(n=60, seed=1):
    """More measurements than can be drawn, which is what xD is for."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({f"m{i}": rng.normal(i, 1.0, n) for i in range(6)})
    frame["plateID"] = "p1"
    return frame


def test_projecting_with_no_table_says_which_step_is_missing(screen):
    assert screen.reduce_to_components() == "Load a table first."


def test_the_button_that_claimed_a_projection_is_put_back_when_it_failed(
        screen, monkeypatch):
    """Leaving it on would say the editor is gating components when it is
    gating the measurements themselves."""
    monkeypatch.setattr(screen, "reduce_to_components", lambda: "no.")

    screen._on_projection_requested(True)

    assert screen._settings.xd_projection is False


def test_turning_the_projection_off_does_not_take_the_components_away(screen):
    """The components are ordinary columns by then and gates may be drawn on
    them; dropping the columns those gates name would break them."""
    screen.set_frame(_wide())
    screen._on_projection_requested(True)
    assert "PC1" in screen._frame.columns

    screen._on_projection_requested(False)

    assert screen._settings.xd_projection is False
    assert "PC1" in screen._frame.columns


def test_a_projection_with_no_variance_to_report_still_names_the_components(
        screen):
    """UMAP and t-SNE report none, and "3 component(s)" is honest where a
    percentage would be invented."""
    components = pd.DataFrame({"UMAP1": [0.0], "UMAP2": [1.0]})
    assert GateEditorScreen._variance_label(components, []) == \
        "2 component(s)"


# ---------------------------------------------------------------------------
# The working set of tables
# ---------------------------------------------------------------------------

@pytest.fixture
def two_tables(tmp_path):
    path = str(tmp_path / "measurements.db")
    with sqlite3.connect(path) as db:
        _objects().to_sql("cell", db, index=False)
        _objects().rename(columns={"area": "nucleus_area"}).to_sql(
            "nucleus", db, index=False)
    return path


def test_removing_a_table_takes_it_out_of_the_merge(screen, two_tables):
    screen.load_path(two_tables, table="cell")
    screen._table_picker.setCurrentText("nucleus")
    screen._on_table_added(0)

    screen.remove_table("nucleus")

    assert screen._tables == ["cell"]


def test_reloading_a_working_set_that_has_no_table_does_nothing(screen):
    screen._path = None
    screen._reload_working_set()
    assert screen._frame is None


# ---------------------------------------------------------------------------
# The per-column merge rules
# ---------------------------------------------------------------------------

def test_the_merge_rules_need_measurements_to_be_rules_about(screen, dialogs):
    screen.show_aggregation_rules()
    assert dialogs.titles == ["No table"]


# ---------------------------------------------------------------------------
# The screen as the registry sees it
# ---------------------------------------------------------------------------

def test_the_factory_the_registry_calls_builds_the_screen(qtbot):
    from spacr.qt.screens.gate_editor import make_gate_editor_screen

    made = make_gate_editor_screen("gate_editor")
    qtbot.addWidget(made)
    assert isinstance(made, GateEditorScreen)


def test_the_screen_reports_whether_a_read_is_still_running(screen):
    """The activity spinner asks these two, and a screen that always said
    "idle" would let the window close over a running read."""
    assert screen.active_jobs() == 0
    assert screen.is_busy() is False


# ---------------------------------------------------------------------------
# The Filter/Search tab strip
# ---------------------------------------------------------------------------

def test_the_side_tabs_are_styled_in_a_freshly_built_stylesheet(qapp):
    """A tab strip with no rule falls through to the blanket
    ``QWidget { background-color: bg }`` -- #000000 on dark -- so it is not
    slightly off, it is a black slab beside the plot.

    The screen tried to register the block from its own ``__init__``, and
    asked the theme for a `register_qss` that does not exist. The ImportError
    landed in the except beside it, which says the styling "is not worth
    taking the screen down for" -- so the styling was never applied and
    nothing said so.
    """
    from spacr.qt import theme
    from spacr.qt.screens.gate_editor import SIDE_TABS_NAME

    theme.load_widget_qss_registrars()

    assert SIDE_TABS_NAME in theme.stylesheet("dark"), (
        f"{SIDE_TABS_NAME} has no rule in a freshly built stylesheet")


def test_the_side_tabs_block_renders_something_for_both_themes(qapp):
    """The signature is ``fn(palette, opacity)``. The lambda the screen
    registered took one argument, so even a registration that landed would
    have raised the moment the sheet was composed."""
    from spacr.qt import theme
    from spacr.qt.screens.gate_editor import SIDE_TABS_NAME

    block = theme._WIDGET_QSS[SIDE_TABS_NAME]
    for name in ("dark", "light"):
        palette = dict(theme.palette_for(name))
        palette["theme"] = name
        assert str(block(palette, 1.0) or "").strip()


# ---------------------------------------------------------------------------
# The small guards
# ---------------------------------------------------------------------------


def test_a_table_the_formula_panel_cannot_compute_is_not_pushed_downstream(
        screen, monkeypatch):
    """Handing the gates a None frame would take the plot down for a broken
    formula, which the formula panel reports on its own."""
    monkeypatch.setattr(screen.formulas, "computed_frame", lambda: None)
    before = screen.gates.canvas.population()

    screen._push_frame()

    assert screen.gates.canvas.population() is before
