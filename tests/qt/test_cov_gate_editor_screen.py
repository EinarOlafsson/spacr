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

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.gate_editor import GateEditorScreen      # noqa: E402
from spacr.qt.widgets.gate_spec import (                        # noqa: E402
    GateSet, RectGate, ThresholdGate,
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


def _filters_of(path):
    with sqlite3.connect(path) as db:
        return pd.read_sql("SELECT * FROM filters", db)


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

def test_a_gate_becomes_a_column_of_the_filters_table(loaded, dialogs):
    """The whole point of the screen: a shape drawn on a scatter comes back
    as a 1/0 column every other module can join on."""
    loaded.export_gates()

    filters = _filters_of(loaded._path)
    assert "big" in filters.columns
    # Every object gets an answer, including the ones outside the gate: null
    # is not what "outside the gate" means.
    assert set(filters["big"]) == {0, 1}
    assert int(filters["big"].sum()) == 6
    assert "wrote big (6 objects)" in loaded._source.text()


def test_exporting_with_nothing_drawn_says_so(screen, dialogs):
    screen.export_gates()
    assert dialogs.titles == ["No gates"]


def test_a_table_read_from_a_csv_has_nowhere_to_put_the_gates(loaded, dialogs,
                                                              tmp_path):
    """"Not a database" rather than a failed write: a CSV is a legitimate
    thing to gate on, and the refusal has to name what is missing."""
    loaded._path = str(tmp_path / "plate.csv")
    loaded.export_gates()
    assert dialogs.titles == ["Not a database"]
    assert "measurement database" in dialogs.shown[0][2]


def test_a_database_with_no_table_chosen_is_refused(loaded, dialogs):
    """The gates were drawn on SOME table's measurements, and the export
    writes an object identity read from that table."""
    loaded._table = ""
    loaded.export_gates()
    assert dialogs.titles == ["No table"]


def test_one_gate_that_cannot_be_applied_does_not_cost_the_others(loaded):
    """A gate on a computed column the database has not got is an ordinary
    thing to have. Sinking the export would cost the user the five gates that
    were fine."""
    loaded.gates.set_gates(
        GateSet()
        .add(ThresholdGate(name="bright", column="intensity", low=400.0))
        .add(ThresholdGate(name="ratio", column="area_over_intensity",
                           low=0.5)))

    loaded.export_gates()

    said = loaded._source.text()
    assert "bright" in said and "could not export ratio" in said
    assert "big" not in _filters_of(loaded._path).columns
    assert "bright" in _filters_of(loaded._path).columns


def test_an_export_that_wrote_nothing_says_nothing_written(screen):
    """Not an empty label. `_on_exported` is the only thing that reports the
    result, and a blank one reads as the export still running."""
    screen._on_exported(([], [("g", "no such column")]))
    assert screen._source.text() == (
        "nothing written · could not export g (no such column)")


# ---------------------------------------------------------------------------
# Annotating from the gates on screen
# ---------------------------------------------------------------------------

def test_the_shown_gates_become_one_annotation_column(loaded, dialogs):
    """Ticking a gate on and off is already how the user says which ones
    count, so the annotation is built from those and asks nothing further."""
    dialogs.item = ("binary", True)
    dialogs.text = ("phenotype", True)

    loaded.annotate_from_gates()

    filters = _filters_of(loaded._path)
    assert "phenotype" in filters.columns
    assert "wrote phenotype" in loaded._source.text()


def test_annotating_with_no_gates_ticked_says_which_gesture_is_missing(
        loaded, dialogs):
    for name in loaded.gates.canvas.enabled_gates:
        loaded.gates.canvas.set_gate_enabled(name, False)

    loaded.annotate_from_gates()

    assert dialogs.titles == ["No gates shown"]


def test_annotating_without_a_table_is_refused(screen, dialogs):
    screen.gates.set_gates(GateSet().add(
        ThresholdGate(name="bright", column="intensity", low=400.0)))
    screen.annotate_from_gates()
    assert dialogs.titles == ["No table"]


def test_cancelling_the_mode_writes_nothing(loaded, dialogs):
    dialogs.item = ("binary", False)
    loaded.annotate_from_gates()
    assert "filters" not in _tables_of(loaded._path)


def test_an_annotation_with_no_name_writes_nothing(loaded, dialogs):
    """A blank name and a cancelled dialog mean the same thing, and neither
    may leave a column called '' behind."""
    dialogs.item = ("binary", True)
    dialogs.text = ("   ", True)
    loaded.annotate_from_gates()
    assert "filters" not in _tables_of(loaded._path)


def test_a_mode_the_annotator_refuses_is_reported_not_raised(loaded, dialogs):
    dialogs.item = ("sideways", True)
    dialogs.text = ("phenotype", True)

    loaded.annotate_from_gates()

    assert dialogs.titles == ["Could not annotate"]


def test_a_csv_still_gets_its_counts_even_with_nowhere_to_write(
        loaded, dialogs, tmp_path):
    """The counts ARE the answer to "how many objects did I just label".
    Refusing outright because there is no database would hide them."""
    dialogs.item = ("binary", True)
    dialogs.text = ("phenotype", True)
    loaded._path = str(tmp_path / "plate.csv")

    loaded.annotate_from_gates()

    said = loaded._source.text()
    assert said.startswith("binary annotation — ")
    assert "not written: this table came from a file" in said


def _tables_of(path):
    with sqlite3.connect(path) as db:
        return {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}


# ---------------------------------------------------------------------------
# Saving and loading the strategy
# ---------------------------------------------------------------------------

def test_a_gating_strategy_round_trips_through_a_file(loaded, tmp_path,
                                                      qtbot):
    path = str(tmp_path / "gates.json")
    assert loaded.save_gates(path) == path
    assert "gates saved to gates.json" == loaded._source.text()

    other = GateEditorScreen(threaded=False)
    qtbot.addWidget(other)
    other.set_frame(_objects())

    assert other.load_gates(path) is True
    assert [g.name for g in other.gates.gates.gates] == ["big"]
    assert other._source.text() == "1 gate(s) from gates.json"


def test_a_file_that_is_not_a_strategy_is_reported_rather_than_raised(
        screen, tmp_path):
    """Pointing the loader at the wrong file is an ordinary slip, and the
    screen has to survive it with a sentence rather than a traceback out of
    a button press."""
    path = tmp_path / "not_gates.json"
    path.write_text("{not json at all")

    assert screen.load_gates(str(path)) is False
    assert "could not load those gates" in screen._source.text()


def test_a_json_that_simply_has_no_gates_in_it_reports_the_zero(screen,
                                                                tmp_path):
    """It loads -- a strategy with nothing in it is a legitimate file -- and
    the count is what says the user picked the wrong one."""
    path = tmp_path / "other.json"
    path.write_text('{"something": "else"}')

    assert screen.load_gates(str(path)) is True
    assert screen._source.text() == "0 gate(s) from other.json"


def test_the_save_dialog_is_only_obeyed_when_it_returns_a_path(loaded,
                                                               dialogs,
                                                               tmp_path):
    """Cancelling a Save As must not write a file called ''."""
    dialogs.save_path = ""
    loaded.choose_save_gates()
    assert loaded._source.text() != "gates saved to "

    dialogs.save_path = str(tmp_path / "chosen.json")
    loaded.choose_save_gates()
    assert os.path.isfile(dialogs.save_path)


def test_the_load_dialog_is_only_obeyed_when_it_returns_a_path(loaded,
                                                               dialogs,
                                                               tmp_path):
    path = str(tmp_path / "chosen.json")
    loaded.save_gates(path)
    loaded.gates.set_gates(GateSet())

    dialogs.open_path = ""
    loaded.choose_load_gates()
    assert len(loaded.gates.gates) == 0

    dialogs.open_path = path
    loaded.choose_load_gates()
    assert [g.name for g in loaded.gates.gates.gates] == ["big"]


# ---------------------------------------------------------------------------
# Saving and loading the filter set
# ---------------------------------------------------------------------------

def test_a_filter_set_saves_and_loads_through_the_screen(loaded, tmp_path,
                                                         qtbot):
    loaded.filters.add_column("area")
    path = str(tmp_path / "filters.json")

    assert loaded.save_filters(path) == path
    assert loaded._source.text() == "filters saved to filters.json"

    other = GateEditorScreen(threaded=False)
    qtbot.addWidget(other)
    other.set_frame(_objects())

    assert other.load_filters(path) == []
    assert other._source.text() == "filters loaded from filters.json"


def test_a_filter_set_saved_against_another_plate_names_what_is_missing(
        loaded, tmp_path, qtbot):
    """A set that half-applies selects the wrong rows while looking like it
    worked, which is why the missing columns are named rather than counted."""
    loaded.filters.add_column("area")
    path = str(tmp_path / "filters.json")
    loaded.save_filters(path)

    other = GateEditorScreen(threaded=False)
    qtbot.addWidget(other)
    other.set_frame(_objects().drop(columns=["area"]))

    assert other.load_filters(path) == ["area"]
    assert other._source.text() == "filters.json loaded; this table has no area"


def test_a_filter_file_that_cannot_be_read_is_reported(screen, tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json at all")

    assert screen.load_filters(str(path)) == []
    assert "could not load that filter set" in screen._source.text()


def test_the_filter_dialogs_are_only_obeyed_when_they_return_a_path(
        loaded, dialogs, tmp_path):
    dialogs.save_path = ""
    loaded.choose_save_filters()
    assert "filters saved" not in loaded._source.text()

    dialogs.save_path = str(tmp_path / "f.json")
    loaded.filters.add_column("area")
    loaded.choose_save_filters()
    assert os.path.isfile(dialogs.save_path)

    dialogs.open_path = ""
    loaded._source.setText("untouched")
    loaded.choose_load_filters()
    assert loaded._source.text() == "untouched"

    dialogs.open_path = dialogs.save_path
    loaded.choose_load_filters()
    assert "filters loaded from f.json" == loaded._source.text()


# ---------------------------------------------------------------------------
# Saving the picture
# ---------------------------------------------------------------------------

def test_saving_a_graph_before_anything_is_drawn_says_so(screen):
    """The console, not a modal: nothing has gone wrong, there is simply
    nothing there yet."""
    assert screen.save_graph("/tmp/never-written.png") == ""


def test_the_graph_is_written_through_the_print_restyler(loaded, tmp_path,
                                                         monkeypatch):
    """`savefig` would write the dark theme's colours -- white text on black,
    unusable on paper -- which is the whole reason the figure queue's
    restyling renderer is used instead."""
    import spacr.qt.widgets.figure_queue as queue

    called = {}

    def _render(figure, path):
        called["path"] = path
        open(path, "wb").close()
        return True

    monkeypatch.setattr(queue, "render_figure_to_png", _render)
    target = str(tmp_path / "graph.png")

    assert loaded.save_graph(target) == target
    assert called["path"] == target
    assert "Saved the graph to" in loaded.console.log.toPlainText()


def test_a_path_with_no_suffix_gets_the_preferred_one(loaded, tmp_path,
                                                      monkeypatch):
    """The format comes from the figure-format PREFERENCE. A second place to
    answer "am I making PDFs" is one too many."""
    import spacr.qt.preferences as preferences
    import spacr.qt.widgets.figure_queue as queue

    monkeypatch.setattr(preferences, "get_figure_format", lambda: "pdf")
    written = []

    def _render(figure, path):
        written.append(path)
        open(path, "wb").close()
        # A PDF run writes the vector file beside the PNG.
        open(os.path.splitext(path)[0] + ".pdf", "wb").close()
        return True

    monkeypatch.setattr(queue, "render_figure_to_png", _render)

    out = loaded.save_graph(str(tmp_path / "graph"))

    assert written == [str(tmp_path / "graph.png")]
    assert out == str(tmp_path / "graph.pdf"), (
        "the user asked for a PDF and was handed the PNG")


def test_a_render_that_refuses_says_so_instead_of_claiming_a_file(
        loaded, tmp_path, monkeypatch):
    import spacr.qt.widgets.figure_queue as queue

    monkeypatch.setattr(queue, "render_figure_to_png", lambda *_a: False)

    assert loaded.save_graph(str(tmp_path / "graph.png")) == ""
    assert "Could not save the graph." in loaded.console.log.toPlainText()


def test_cancelling_the_graph_dialog_writes_nothing(loaded, dialogs):
    dialogs.save_path = ""
    assert loaded.save_graph() == ""


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
