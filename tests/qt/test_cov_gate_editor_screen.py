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
# Annotating from the gates on screen
# ---------------------------------------------------------------------------

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


def test_merging_no_files_at_all_does_nothing(screen):
    screen.load_paths([])
    assert screen._frame is None


def test_a_file_whose_tables_cannot_be_listed_names_it(screen, tmp_path):
    """A path that is not a database at all is an ordinary mis-drop, and the
    screen has to say which file it could not read."""
    bad = tmp_path / "notes.db"
    bad.write_text("this is not sqlite", encoding="utf-8")

    screen.load_paths([str(bad)])

    assert "could not read" in screen._source.text()
    assert str(bad) in screen._source.text()


def test_a_database_with_no_tables_in_it_says_there_is_nothing_to_merge(
        screen, tmp_path, monkeypatch):
    import spacr.qt.screens.gate_editor as module

    monkeypatch.setattr(module, "table_names", lambda _p: [])
    screen.load_paths([str(tmp_path / "empty.db")])
    assert screen._source.text() == "no table to merge in the chosen files"


def test_two_plates_of_the_same_name_are_reported_not_pooled(screen,
                                                             tmp_path):
    """Two databases that each hold a plate called p1 are two experiments.
    Pooling them would compute every per-well number over both at once with
    nothing on screen to say so, so the clash is named and refused."""
    paths = []
    for name in ("a.db", "b.db"):
        path = str(tmp_path / name)
        with sqlite3.connect(path) as db:
            _objects().to_sql("cell", db, index=False)
        paths.append(path)

    screen.load_paths(paths)

    assert screen._frame is None, "the collision was pooled anyway"
    assert "p1" in screen._source.text()


def test_two_plates_that_do_not_clash_merge_into_one_frame(screen, tmp_path):
    paths = []
    for name, plate in (("a.db", "p1"), ("b.db", "p2")):
        path = str(tmp_path / name)
        frame = _objects()
        frame["plateID"] = plate
        with sqlite3.connect(path) as db:
            frame.to_sql("cell", db, index=False)
        paths.append(path)

    screen.load_paths(paths)

    assert screen._frame is not None and len(screen._frame) == 16
    assert "2 databases · cell" in screen._source.text()
    assert screen._merge_plan is not None


def test_a_merge_that_fails_for_any_other_reason_still_says_so(screen,
                                                              monkeypatch,
                                                              tmp_path):
    """Not a traceback out of a file dialog. The count of files is in the
    message because that is what the user just did."""
    import spacr.qt.screens.gate_editor as module
    import spacr.multi_database as multi

    monkeypatch.setattr(module, "table_names", lambda _p: ["cell"])
    monkeypatch.setattr(multi, "describe_merge",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("the disk went away")))

    screen.load_paths([str(tmp_path / "a.db"), str(tmp_path / "b.db")])

    assert screen._source.text() == (
        "could not merge 2 files: the disk went away")


def test_a_csv_is_read_whole_because_the_sampling_is_done_in_sql(screen,
                                                                 tmp_path):
    """Reading a whole file only to throw four rows in five away costs more
    than it saves, so the fraction applies to databases and the cap to both."""
    path = tmp_path / "plate.csv"
    _objects().to_csv(path, index=False)

    screen.load_path(str(path))

    assert screen._frame is not None and len(screen._frame) == 8
    assert screen._table_picker.isHidden() is True, (
        "a CSV has no tables to pick between")


def test_a_database_the_loader_cannot_open_names_the_file_not_the_path(
        screen, tmp_path):
    bad = tmp_path / "broken.db"
    bad.write_text("not sqlite either", encoding="utf-8")

    screen.load_path(str(bad))

    assert screen._source.text().startswith("could not read broken.db: ")
    assert screen._frame is None


# ---------------------------------------------------------------------------
# The right-click menu on the plot
# ---------------------------------------------------------------------------

def test_a_greyed_row_says_why_it_is_greyed(screen):
    """A disabled item with no reason is a dead end that looks like a bug."""
    items = {label: (enabled, why)
             for label, enabled, _cb, why in screen.graph_menu_items()
             if label}
    assert items["Save graph…"] == (False, "Draw a graph first.")
    assert items["Copy image to clipboard"] == (False, "Draw a graph first.")


def test_a_canvas_that_cannot_say_whether_it_drew_offers_the_rest_anyway(
        screen, monkeypatch):
    """The figure query is the only thing that can raise here, and losing the
    whole menu over it would cost the user six working actions."""
    canvas = screen.gates.canvas
    monkeypatch.setattr(canvas, "figure", lambda: (_ for _ in ()).throw(
        RuntimeError("no figure here")))

    items = {label: enabled
             for label, enabled, _cb, _why in screen.graph_menu_items()
             if label}

    assert items["Save graph…"] is False
    assert items["Export gates to the database…"] is True


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
# The settings window
# ---------------------------------------------------------------------------

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


def test_a_projection_needs_two_measurements_to_project(screen):
    """The xD tab can be narrowed to one column, and one column has no
    projection -- refused with the reason rather than a component that is
    the column again."""
    screen.set_frame(_wide())
    screen.apply_settings(screen._settings.replaced(reduction_columns=("m0",)))

    assert "fewer than two measurements" in screen.reduce_to_components()


def test_a_reduction_the_data_refuses_is_reported_on_the_screen(screen,
                                                                monkeypatch):
    import spacr.merge_tables as merge

    def _refuse(*_a, **_k):
        raise merge.ReductionError("t-SNE needs more rows than that")

    monkeypatch.setattr(merge, "reduce_dimensions", _refuse)
    screen.set_frame(_wide())

    assert screen.reduce_to_components() == "t-SNE needs more rows than that"
    assert screen._source.text() == "t-SNE needs more rows than that"


def test_the_components_arrive_as_ordinary_columns_on_the_axes(screen):
    """A gate on PC1 vs PC2 is the same kind of object as a gate on area vs
    intensity, which is only true if the components are just columns."""
    screen.set_frame(_wide())

    assert screen.reduce_to_components() is None

    assert (screen._x.currentText(), screen._y.currentText()) == ("PC1", "PC2")
    assert screen._z.currentText() == "PC3"
    assert "PC1" in screen._frame.columns
    assert "projected onto PC1 " in screen._source.text()


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


def _lopsided(n=200, seed=0):
    """One morphology measurement against twenty-four intensity ones.

    A shape somebody really has -- an object table with one area column and
    a channel sweep -- and one where ticking "morphology" puts 4% of the
    projection's variance on screen.
    """
    rng = np.random.default_rng(seed)
    columns = {"cell_area": rng.normal(100.0, 10.0, n)}
    for i in range(1, 25):
        columns[f"cell_channel_{i}_mean_intensity"] = rng.normal(
            50.0 * i, 5.0, n)
    return pd.DataFrame(columns)


def test_a_group_carrying_almost_none_of_the_variance_is_named(screen):
    """A group can be ticked and carry almost nothing, and nobody notices,
    because a projection always produces a picture."""
    screen.set_frame(_lopsided())
    screen.apply_settings(screen._settings.replaced(
        reduction_groups={"family": ["morphology", "intensity"]}))

    assert screen.reduce_to_components() is None

    said = screen._source.text()
    assert "family:morphology carries 4% of the variance" in said, said


def test_a_balanced_split_is_not_reported_at_all(screen):
    """Saying it every time would train the user to ignore the line, and the
    balanced split is the expected case."""
    frame = _lopsided()
    keep = ["cell_area", "cell_perimeter"] + [
        f"cell_channel_{i}_mean_intensity" for i in (1, 2)]
    frame["cell_perimeter"] = frame["cell_area"] * 0.9
    screen.set_frame(frame[keep])
    screen.apply_settings(screen._settings.replaced(
        reduction_groups={"family": ["morphology", "intensity"]}))

    assert screen.reduce_to_components() is None
    assert "of the variance" not in screen._source.text()


def test_a_diagnostic_that_fails_does_not_take_the_projection_with_it(
        screen, monkeypatch):
    """A note about the projection that costs the projection has cost more
    than it explained."""
    import spacr.merge_tables as merge

    monkeypatch.setattr(merge, "group_variance_share",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            RuntimeError("no variance today")))
    monkeypatch.setattr(merge, "missingness_leak",
                        lambda *_a, **_k: (_ for _ in ()).throw(
                            RuntimeError("nor that")))
    screen.set_frame(_wide())
    screen.apply_settings(screen._settings.replaced(
        reduction_columns=("m0", "m1", "m2")))

    assert screen.reduce_to_components() is None
    assert "PC1" in screen._frame.columns


def test_a_projection_that_split_on_whether_something_was_measured_says_so(
        screen):
    """`reduce_dimensions` fills gaps with the column median rather than
    dropping the row, which is right -- dropping loses every measurement the
    object DID have -- but it puts every uninfected cell on the same point of
    every pathogen column. Separating infected from uninfected on the FACT of
    measurement is real, reproducible, and not a phenotype: exactly the split
    somebody would otherwise write up.

    Half the objects here have no pathogen measurements at all, which is what
    an uninfected cell looks like in a real table.
    """
    rng = np.random.default_rng(0)
    n = 240
    frame = pd.DataFrame({"cell_area": rng.normal(0.0, 1.0, n)})
    uninfected = np.zeros(n, dtype=bool)
    uninfected[: n // 2] = True
    for i in range(6):
        values = rng.normal(0.0, 50.0, n)
        values[uninfected] = np.nan
        frame[f"pathogen_channel_{i}_mean_intensity"] = values
    screen.set_frame(frame)

    assert screen.reduce_to_components() is None

    said = screen._source.text()
    assert "was measured" in said, said
    assert "50% missing" in said and "not a phenotype" in said


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


def test_picking_a_second_table_adds_it_rather_than_switching(screen,
                                                              two_tables):
    """"Picking nucleus does not switch to nucleus" -- it merges nucleus
    measurements alongside the ones already loaded, so a gate can put a cell
    measurement on one axis and a nuclear one on another."""
    screen.load_path(two_tables, table="cell")
    assert screen._tables == ["cell"]

    screen._table_picker.setCurrentText("nucleus")
    screen._on_table_added(0)

    assert screen._tables == ["cell", "nucleus"]
    # The table went through the merge rather than being read straight: its
    # measurements are qualified by the table they came from, which is what
    # lets one axis be a cell measurement and the other a nuclear one.
    assert screen._frame is not None
    assert "cell_area" in screen._frame.columns
    assert "area" not in screen._frame.columns


def test_picking_a_table_that_is_already_in_the_set_changes_nothing(
        screen, two_tables):
    screen.load_path(two_tables, table="cell")
    screen._table_picker.setCurrentText("cell")

    screen._on_table_added(0)

    assert screen._tables == ["cell"]


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


def test_a_row_cap_thins_a_merge_evenly_rather_than_taking_the_first_rows(
        two_tables):
    """The first N rows of a merged table are one plate's worth of one
    field. Stepping keeps the cap honest about what it is a sample OF."""
    merged = GateEditorScreen._read_working_set(
        two_tables, ["cell", "nucleus"], 1.0, 4, None)
    assert len(merged) <= 4


def test_a_sample_fraction_thins_a_merge_the_same_way(two_tables):
    merged = GateEditorScreen._read_working_set(
        two_tables, ["cell", "nucleus"], 0.5, None, None)
    whole = GateEditorScreen._read_working_set(
        two_tables, ["cell", "nucleus"], 1.0, None, None)
    assert 0 < len(merged) < len(whole)


def test_one_table_in_the_working_set_is_read_straight(two_tables):
    """Merging a table onto itself only renames its columns, and every saved
    gate on a single-table session would stop matching."""
    frame = GateEditorScreen._read_working_set(
        two_tables, ["cell"], 1.0, None, None)
    assert "area" in frame.columns


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


def test_the_screen_still_names_its_tab_strip_that(qtbot):
    """The QSS is keyed by objectName; a rename on one side and not the
    other is the same black slab with a rule nobody matches."""
    from spacr.qt.screens.gate_editor import SIDE_TABS_NAME

    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)
    assert screen.side_tabs.objectName() == SIDE_TABS_NAME


# ---------------------------------------------------------------------------
# The small guards
# ---------------------------------------------------------------------------

def test_a_theme_that_cannot_make_the_panel_transparent_still_builds_it(
        qtbot, monkeypatch):
    """Transparency is decoration. A theme build that cannot apply it costs
    the backdrop and must not cost the screen."""
    from PySide6.QtWidgets import QTabWidget
    from spacr.qt import theme

    real = theme.make_transparent

    def _only_the_tabs(widget):
        if isinstance(widget, QTabWidget):
            raise RuntimeError("no compositing here")
        return real(widget)

    monkeypatch.setattr(theme, "make_transparent", _only_the_tabs)

    screen = GateEditorScreen(threaded=False)
    qtbot.addWidget(screen)

    assert screen.side_tabs is not None
    assert screen.side_tabs.count() == 2


def test_a_table_the_formula_panel_cannot_compute_is_not_pushed_downstream(
        screen, monkeypatch):
    """Handing the gates a None frame would take the plot down for a broken
    formula, which the formula panel reports on its own."""
    monkeypatch.setattr(screen.formulas, "computed_frame", lambda: None)
    before = screen.gates.canvas.population()

    screen._push_frame()

    assert screen.gates.canvas.population() is before


def test_the_plot_actions_are_silent_when_there_is_no_plot_to_act_on(screen):
    """`GateEditorPanel` builds its canvas, so this is the state during
    construction and teardown rather than one a user reaches -- but all four
    entry points hang off Qt signals, and a signal arriving then must not
    raise into the event loop.

    Put back by hand rather than by `monkeypatch`: pytest-qt closes the
    widgets it was given BEFORE monkeypatch's finalisers run, and the panel's
    own closeEvent closes the canvas.
    """
    from PySide6.QtCore import QPoint

    canvas, screen.gates.canvas = screen.gates.canvas, None
    try:
        assert screen.graph_menu_items() == []
        screen._install_graph_context_menu()       # must not raise
        screen._show_graph_menu(QPoint(0, 0))
        screen._copy_graph_to_clipboard()
    finally:
        screen.gates.canvas = canvas

    assert screen.console.log.toPlainText() == ""
