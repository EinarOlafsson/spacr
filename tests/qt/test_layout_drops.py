"""Drop anywhere, and the drop resolves through spaCR's project layout.

Two properties are under test here, and the first one is the reason the
second one is safe:

**A drop and an auto-chain give the same answer.** Both go through
:func:`spacr.chaining.resolve_drop` / :func:`spacr.chaining.chained_inputs`,
which ask the artifact registry the same question. Two answers to "where is
the database" is how a screen and the run it launches come to disagree, and
the disagreement is invisible in normal use because both answers *work* --
``spacr.ports.project_root`` hops a trailing ``merged``, so a field holding
the plate and a field holding ``<plate>/merged`` run identically. It only
shows when a settings CSV written by one is read beside the other. So the
agreement is asserted explicitly, per module, against a real project tree.

**Every screen that reads a path takes a drop.** The screens landed in the
last day mostly had none; they are wired through
:func:`spacr.qt.dnd.install_for` and
:data:`spacr.qt.dnd_handlers._HANDLERS`. The three screens where a drop is
genuinely meaningless are named in
:data:`spacr.qt.dnd_handlers.NO_DROP_TARGET` and this file asserts they are
named rather than merely absent.

The fixture is a real project tree on disk -- ``merged/*.npy``,
``measurements/measurements.db`` with real tables, ``masks/``, ``results/``,
``settings/`` -- because every assertion here is about what is found on
disk, and a mock of the layout would only test that the mock matches the
code that reads it.
"""
from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import pytest
from PySide6.QtCore import QObject, QPoint, QPointF, Qt, QTimer, QUrl
from PySide6.QtCore import QMimeData
from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PySide6.QtWidgets import QApplication

from spacr import artifacts as art
from spacr import chaining as ch
from spacr import ports as P
from spacr.qt import dnd_handlers as dh


# ---------------------------------------------------------------------------
# A real project on disk
# ---------------------------------------------------------------------------

def _make_project(root, *, images: bool = True) -> str:
    """Build a spaCR project with one of everything the ports declare."""
    root = str(root)
    for name in ("merged", "measurements", "masks", "results", "settings",
                 "model"):
        os.makedirs(os.path.join(root, name), exist_ok=True)
    np.save(os.path.join(root, "merged", "plate1_A01_1.npy"),
            np.zeros((8, 8, 3), np.uint16))
    db = os.path.join(root, "measurements", "measurements.db")
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE cell (object_label int, area real)")
    connection.execute("INSERT INTO cell VALUES (1, 3.0)")
    connection.execute("CREATE TABLE png_list (png_path text)")
    connection.execute("INSERT INTO png_list VALUES ('a.png')")
    connection.commit()
    connection.close()
    open(os.path.join(root, "masks", "plate1_A01_1.tif"), "wb").close()
    with open(os.path.join(root, "results", "results_gene.csv"), "w") as fh:
        fh.write("gene,coefficient\nabc,1.0\n")
    with open(os.path.join(root, "settings",
                           "gen_mask_settings.csv"), "w") as fh:
        fh.write("Key,Value\nsrc,x\n")
    open(os.path.join(root, "model", "best.pth"), "wb").close()
    if images:
        open(os.path.join(root, "plate1_A01_f01_ch1.tif"), "wb").close()
    return root


@pytest.fixture
def project(tmp_path):
    """One plate folder, laid out the way spaCR lays one out."""
    return _make_project(tmp_path / "plate1")


@pytest.fixture
def experiment(tmp_path):
    """Two plate folders under one parent -- the ambiguous drop."""
    parent = tmp_path / "experiment"
    parent.mkdir()
    return (parent, _make_project(parent / "plate1"),
            _make_project(parent / "plate2"))


# ---------------------------------------------------------------------------
# The invariant: a dropped root and an auto-chained one agree
# ---------------------------------------------------------------------------

#: Every pipeline module that consumes something and is reachable from a
#: dropped project folder. Taken from the port registry, not typed out, so a
#: module that joins the graph joins this test.
CHAINED_MODULES = tuple(
    key for key in P.known_modules()
    if P.module_ports(key).consumes
    and any(port.kind != P.SEQUENCING_READS
            for port in P.module_ports(key).consumes))


def _register_outputs(root: str) -> None:
    """Record a Mask and a Measure run, as a real pipeline would."""
    registry = art.open_registry(root, create=True)
    art.register_run_outputs("mask", {"src": root}, registry=registry)
    art.register_run_outputs("measure", {"src": root}, registry=registry)


@pytest.mark.parametrize("module", CHAINED_MODULES)
def test_a_dropped_project_agrees_with_auto_chaining(project, module,
                                                     tmp_path, monkeypatch):
    """Dropping the root fills the field with what chaining would have chosen.

    Not "with something that also works" -- with the same string. Asserted
    against a project whose runs are *registered*, because that is the case
    where the two could differ: the drop could re-derive ``<root>/merged``
    while chaining reports where the producer says it wrote.
    """
    monkeypatch.setenv(ch.PIN_STATE_ENV, str(tmp_path / "pins.json"))
    _register_outputs(project)

    setting = ch.source_key(module)
    chained = ch.resolve_settings(module, {setting: ""}, root=project,
                                  pins=ch.PinStore(str(tmp_path / "p.json")))
    dropped = ch.resolve_drop(module, project)

    if setting not in chained.filled:
        pytest.skip(f"{module} has nothing to chain from this project")
    target = next((t for t in dropped.targets if t.setting == setting), None)
    assert target is not None, (
        f"auto-chaining fills {setting} for {module} but a drop on the same "
        f"folder resolved nothing: {dropped.reason}")
    assert target.value == chained.settings[setting], (
        f"{module}: a dropped project fills {setting} with "
        f"{target.value!r} and auto-chaining fills it with "
        f"{chained.settings[setting]!r}. One project, two answers.")
    assert target.source == ch.FROM_REGISTRY, (
        "the run was registered, so the drop must have read the registry "
        "rather than re-derived the path")


@pytest.mark.parametrize("module", CHAINED_MODULES)
def test_the_layout_fallback_lands_where_chaining_would(project, module,
                                                        tmp_path):
    """With no registry at all, the declared layout gives the same answer.

    The half of the agreement that has no registry to agree through: a plate
    somebody copied in has no ``artifacts.db``, auto-chaining offers nothing,
    and a drop still has to fill the field with the folder a subsequent run
    will read. It does that from :data:`spacr.ports.PORTS`, so what the drop
    fills in is by construction where ``check_ready`` looks.
    """
    assert not os.path.exists(os.path.join(project, art.ARTIFACTS_DB_NAME))
    dropped = ch.resolve_drop(module, project)
    if not dropped.targets:
        pytest.skip(f"{module} reads nothing this project has")
    for target in dropped.targets:
        assert target.source == ch.FROM_LAYOUT
        assert os.path.exists(target.location)
    readiness = P.check_ready(module, {ch.source_key(module):
                                       dropped.targets[0].value})
    assert readiness.ok, (
        f"a drop filled {module}'s source with "
        f"{dropped.targets[0].value!r} and check_ready then refused it: "
        f"{readiness.reason}")


def test_the_measure_field_holds_the_plate_not_the_merged_folder(project):
    """The one disagreement this change actually fixed, pinned by name.

    ``MeasureDropHandler`` drilled into ``merged/``; chaining filled the same
    key with the plate.
    """
    dropped = ch.resolve_drop("measure", project)
    assert dropped.targets[0].value == project
    assert dropped.targets[0].location == os.path.join(project, "merged")


# ---------------------------------------------------------------------------
# The layout walk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("inside", [
    "",
    "merged",
    "masks",
    "results",
    os.path.join("measurements", "measurements.db"),
    os.path.join("merged", "plate1_A01_1.npy"),
    os.path.join("settings", "gen_mask_settings.csv"),
])
def test_every_place_inside_a_project_resolves_to_the_project(project, inside):
    """Drop the database, the merged folder or a CSV -- same plate comes back."""
    assert ch.project_root_of(os.path.join(project, inside)) == project


def test_the_layout_folders_are_read_off_the_port_declarations():
    """Not typed out: a plugin's port makes its folder part of the layout."""
    names = ch.layout_directories()
    assert {"merged", "measurements", "masks", "data", "model", "results",
            "settings", "orig", "consolidated"} <= set(names)

    P.register_module_ports(P.ModulePorts(
        key="_layout_probe",
        produces=(P.Port(P.EMBEDDING, "out", "plugin_outputs", "*.csv"),)))
    try:
        assert "plugin_outputs" in ch.layout_directories()
    finally:
        P.PORTS.pop("_layout_probe", None)
    assert "plugin_outputs" not in ch.layout_directories()


# ---------------------------------------------------------------------------
# The direct drop still works
# ---------------------------------------------------------------------------

def test_dropping_the_database_itself_still_works(project):
    db = os.path.join(project, "measurements", "measurements.db")
    resolution = ch.resolve_drop("graph_builder", db,
                                 kinds=(P.MEASUREMENTS_DB,))
    assert resolution.targets[0].location == db


def test_dropping_a_folder_of_images_still_works(tmp_path):
    """A plate nothing has been run on: no layout, and the folder is the answer."""
    folder = tmp_path / "raw"
    folder.mkdir()
    open(folder / "plate1_A01_f01_ch1.tif", "wb").close()
    resolution = ch.resolve_drop("mask", folder)
    assert resolution.targets[0].value == str(folder)


def test_dropping_a_result_csv_is_taken_as_the_file_it_is(project):
    """A file the screen can read directly skips the layout walk entirely."""
    from pathlib import Path

    csv = Path(project) / "results" / "results_gene.csv"
    handler = dh.get_handler("profiler")
    assert handler.can_accept(csv) is True

    loaded = []

    class Screen:
        def load_coefficients(self, path):
            loaded.append(path)

    handler.apply(csv, Screen())
    assert loaded == [str(csv)]


# ---------------------------------------------------------------------------
# Ambiguity is asked about, never guessed
# ---------------------------------------------------------------------------

def test_two_projects_under_the_dropped_folder_become_a_question(experiment):
    parent, plate1, plate2 = experiment
    resolution = ch.resolve_drop("measure", parent)
    assert resolution.ambiguous, (
        "a folder holding two plates resolved to one of them without asking")
    assert resolution.choices[0].options == (plate1, plate2)
    assert not resolution.targets, "it picked one anyway"


def test_two_databases_in_one_project_become_a_question(project):
    open(os.path.join(project, "measurements", "spare.db"), "wb").close()
    resolution = ch.resolve_drop("umap", project)
    assert resolution.ambiguous
    options = resolution.choices[0].options
    assert len(options) == 2
    assert options[0].endswith("measurements.db"), (
        "the declared location should be offered first, not sorted away")


def test_an_ambiguous_drop_reaches_the_did_you_mean_dialog(experiment):
    """The handler answers ``can_accept`` False and offers the candidates.

    That is what :mod:`spacr.qt.dnd` needs to show the chooser, so ambiguity
    routes through the machinery that already exists rather than a second
    dialog of its own.
    """
    from pathlib import Path

    parent, plate1, plate2 = experiment
    handler = dh.get_handler("graph_builder")
    assert handler.can_accept(Path(parent)) is False
    assert [str(p) for p in handler.suggest_alternatives(Path(parent))] == \
        [plate1, plate2]


def test_a_multi_table_database_is_asked_about_rather_than_taken(project,
                                                                 monkeypatch):
    """``load_path`` takes the first table silently; a drop must not."""
    from pathlib import Path

    asked = {}

    def fake_ask(screen, headline, question, options):
        asked["options"] = list(options)
        return options[1]

    monkeypatch.setattr(dh, "_ask_for_one", fake_ask)

    class Screen:
        def __init__(self):
            self.loaded = None

        def load_path(self, path, table=None):
            self.loaded = (path, table)

    screen = Screen()
    dh.get_handler("graph_builder").apply(Path(project), screen)
    assert asked["options"] == ["cell", "png_list"]
    assert screen.loaded == (
        os.path.join(project, "measurements", "measurements.db"), "png_list")


def test_cancelling_the_table_question_loads_nothing(project, monkeypatch):
    from pathlib import Path

    monkeypatch.setattr(dh, "_ask_for_one",
                        lambda *args, **kwargs: None)

    class Screen:
        loaded = None

        def load_path(self, path, table=None):
            self.loaded = (path, table)

    screen = Screen()
    dh.get_handler("trellis").apply(Path(project), screen)
    assert screen.loaded is None, (
        "the question was cancelled and a table was loaded anyway")


# ---------------------------------------------------------------------------
# A drop that resolves to nothing says what is missing
# ---------------------------------------------------------------------------

def test_an_unsatisfiable_drop_names_what_the_module_needs(tmp_path):
    empty = tmp_path / "holiday_photos"
    empty.mkdir()
    resolution = ch.resolve_drop("measure", empty)
    assert not resolution.ok
    assert not resolution.targets
    problem = resolution.problems[0]
    assert P.MERGED_ARRAYS in problem.message
    assert "merged" in problem.message
    assert "mask" in problem.fix, "it should say who produces what is missing"
    assert problem.message in resolution.reason
    assert problem.fix in resolution.reason


def test_the_missing_input_sentence_is_check_ready_s_own(tmp_path):
    """Not a paraphrase: the same Problem objects."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    dropped = ch.resolve_drop("umap", empty)
    readiness = P.check_ready("umap", root=str(empty))
    assert ([(p.message, p.fix) for p in dropped.problems]
            == [(p.message, p.fix) for p in readiness.problems])


def test_a_screen_with_no_module_still_gets_the_port_s_sentence(tmp_path):
    """A table explorer is not a module, and says the same thing anyway."""
    empty = tmp_path / "nothing"
    empty.mkdir()
    dropped = ch.resolve_drop("graph_builder", empty,
                              kinds=(P.MEASUREMENTS_DB,))
    assert not dropped.ok
    assert P.MEASUREMENTS_DB in dropped.problems[0].message
    assert "measurements.db" in dropped.problems[0].message


def test_an_unsatisfiable_drop_is_reported_on_the_screen(tmp_path,
                                                         monkeypatch):
    from pathlib import Path

    reported = []
    monkeypatch.setattr(
        dh, "_report_drop_problem",
        lambda screen, path, reason, suggestion, **kw:
            reported.append((reason, suggestion)))

    empty = tmp_path / "nothing"
    empty.mkdir()

    class Screen:
        def load_path(self, path, table=None):
            raise AssertionError("nothing should have been loaded")

    dh.get_handler("outliers").apply(Path(empty), Screen())
    assert reported, "a drop that resolved to nothing said nothing"
    assert "measurements-db" in reported[0][0]


# ---------------------------------------------------------------------------
# It says what it resolved to
# ---------------------------------------------------------------------------

def test_the_resolved_path_is_reported_not_just_applied(project):
    from pathlib import Path

    class Screen:
        def __init__(self):
            self.lines = []
            self.loaded = None

        class _Console:
            def __init__(self, lines):
                self._lines = lines

            def append_stdout(self, text):
                self._lines.append(text)

        def load_path(self, path, table=None):
            self.loaded = path

    screen = Screen()
    screen._console = Screen._Console(screen.lines)
    dh.get_handler("feature_explorer").apply(Path(project), screen)
    text = "".join(screen.lines)
    assert os.path.join(project, "measurements", "measurements.db") in text
    assert "resolved measurements-db" in text
    assert project in text, "the project it resolved through is not named"


# ---------------------------------------------------------------------------
# Coverage: every registered app either takes a drop or says why it cannot
# ---------------------------------------------------------------------------

def _app_keys():
    """Every app key registered at import time.

    Deliberately *not* calling ``register_self_registering_modules``. That is
    what ``spacr.qt.run`` does before the window opens, and calling it here
    would register rows globally for the whole test session — including the
    ones a screen's own test asserts are still switched off. A collection-time
    side effect on a module registry is not something to do for the
    convenience of a parametrize.
    """
    from spacr.qt.app import APPS
    return [row[0] for row in APPS]


def _self_registering_keys():
    """The app keys that only appear once ``spacr.qt.run`` has registered them."""
    from spacr.qt import SELF_REGISTERING_MODULES
    return {module.rsplit(".", 1)[-1] for module in SELF_REGISTERING_MODULES}


@pytest.mark.parametrize("app_key", _app_keys())
def test_every_registered_app_that_reads_a_path_accepts_a_drop(
        qtbot, qt_theme_applied, app_key):
    """Built through the real factory, and asked whether it takes drops.

    ``acceptDrops()`` is the property, and it is asked of the screen the
    ``MainWindow`` factory actually builds rather than of a class constructed
    by hand -- a screen can only be dropped on if the window's own factory
    installs the dropzone.

    Roughly twenty-five screens landed in one day with no drop handling at
    all. The three that legitimately have none are named in
    :data:`spacr.qt.dnd_handlers.NO_DROP_TARGET`, so "this one has no target"
    is a statement somebody wrote down rather than an omission.

    One screen per test, like ``test_all_module_smoke`` next door: sixty live
    AppScreens in one function is sixty resource pollers, and they take the
    interpreter down before any assertion is reached.
    """
    from spacr.qt.app import MainWindow
    from .test_all_module_smoke import _FactoryHost

    screen = MainWindow._build_screen(_FactoryHost(), app_key)
    qtbot.addWidget(screen)
    if app_key in dh.NO_DROP_TARGET:
        assert not screen.acceptDrops(), (
            f"{app_key} is listed in NO_DROP_TARGET but accepts drops; "
            "remove the exemption")
        return
    assert screen.acceptDrops(), (
        f"{app_key} accepts no drops and no reason is recorded for it. Add a "
        "handler to spacr.qt.dnd_handlers._HANDLERS, or an entry to "
        "NO_DROP_TARGET saying what the screen reads instead.")


def test_the_screens_without_a_drop_target_really_read_no_path():
    """The excuse has to be true, not just written down.

    A screen listed in :data:`~spacr.qt.dnd_handlers.NO_DROP_TARGET` must not
    quietly grow a path field and keep its exemption.
    """
    known = set(_app_keys()) | _self_registering_keys() | {"feature_dict"}
    for key, reason in dh.NO_DROP_TARGET.items():
        assert key in known, f"{key} is not a registered app any more"
        assert reason and reason[0].islower(), (
            f"{key}: the reason reads as a sentence fragment following the "
            f"screen name; got {reason!r}")
        assert key not in dh._HANDLERS, (
            f"{key} is both exempt from drops and has a drop handler")


def test_the_new_screens_are_all_covered():
    """The screens this change was written for, named so the list is checkable."""
    landed = [
        "graph_builder", "trellis", "gate_editor", "feature_explorer",
        "outliers", "control_chart", "dose_response", "image_scatter",
        "lineage", "profiler", "pipeline_graph", "run_compare",
        "qc_dashboard", "data_manager", "project_browser", "run_history",
        "methods_export", "curate", "napari_bridge", "hit_list",
        "classifier_evaluation", "distributed_jobs", "layer_viewer",
    ]
    missing = [key for key in landed if key not in dh._HANDLERS]
    assert not missing, f"no drop policy for: {', '.join(missing)}"
    for key in landed:
        assert isinstance(dh.get_handler(key), dh.LayoutDropHandler), (
            f"{key} has a handler that does not resolve through the layout")


def test_explain_cv_drop_keeps_database_and_predictions_distinct(
        qtbot, project, tmp_path):
    from spacr.qt.screens.model_explanation import ModelExplanationScreen

    screen = ModelExplanationScreen()
    qtbot.addWidget(screen)
    handler = dh.get_handler("explain_cv")
    database = Path(project) / "measurements" / "measurements.db"
    predictions = tmp_path / "predictions.csv"
    predictions.write_text("path,pred\na.png,1\n", encoding="utf-8")

    handler.apply(Path(project), screen)
    handler.apply(predictions, screen)

    assert screen.explain.database.text() == str(database)
    assert screen.explain.predictions.text() == str(predictions)
    assert screen.explain.prediction_column.currentText() == "pred"


def test_investigate_hit_drop_identifies_fraction_table_by_schema(
        qtbot, project, tmp_path):
    from spacr.qt.screens.model_explanation import InvestigateHitScreen

    screen = InvestigateHitScreen()
    qtbot.addWidget(screen)
    handler = dh.get_handler("investigate_hit")
    fractions = tmp_path / "fractions.csv"
    fractions.write_text(
        "plateID,rowID,columnID,guide,fraction\np1,A,01,g1,0.5\n",
        encoding="utf-8",
    )
    results = tmp_path / "regression-results"
    results.mkdir()

    handler.apply(Path(project), screen)
    handler.apply(fractions, screen)
    handler.apply(results, screen)

    assert screen.investigate.database.text().endswith(
        "measurements/measurements.db")
    assert screen.investigate.fractions.text() == str(fractions)
    assert screen.investigate.regression_folder.text() == str(results)


@pytest.mark.parametrize("key", [
    "graph_builder", "trellis", "gate_editor", "feature_explorer",
    "outliers", "control_chart", "dose_response", "pca", "tabulate",
])
def test_every_table_screen_reads_the_project_s_database(project, key,
                                                         monkeypatch):
    """Nine screens, one handler: drop the plate, get its measurements."""
    from pathlib import Path

    monkeypatch.setattr(dh, "_ask_for_one",
                        lambda screen, headline, question, options: options[0])

    class Screen:
        loaded = None

        def load_path(self, path, table=None):
            self.loaded = (path, table)

    screen = Screen()
    dh.get_handler(key).apply(Path(project), screen)
    assert screen.loaded == (
        os.path.join(project, "measurements", "measurements.db"), "cell")


def test_a_project_screen_opens_the_plate_a_dropped_file_lives_in(project):
    """Drop the database on the pipeline graph and the *project* opens."""
    from pathlib import Path

    opened = []

    class Screen:
        def load_project(self, root):
            opened.append(root)

    handler = dh.get_handler("pipeline_graph")
    db = Path(project) / "measurements" / "measurements.db"
    assert handler.can_accept(db) is True
    handler.apply(db, Screen())
    assert opened == [project]


def test_the_data_manager_scans_what_was_dropped_on_it(project):
    from pathlib import Path

    calls = []

    class Screen:
        def set_project(self, root):
            calls.append(("set", root))

        def scan(self):
            calls.append(("scan",))
            return True

    dh.get_handler("data_manager").apply(Path(project), Screen())
    assert calls == [("set", project), ("scan",)]


def test_curate_resolves_the_mask_folder_and_asks_which_mask(project,
                                                             monkeypatch):
    from pathlib import Path

    open(os.path.join(project, "masks", "plate1_A02_1.tif"), "wb").close()
    asked = {}

    def fake_ask(screen, headline, question, options):
        asked["options"] = list(options)
        return options[0]

    monkeypatch.setattr(dh, "_ask_for_one", fake_ask)

    class Screen:
        def __init__(self):
            self.opened = None

        class _Edit:
            text_value = ""

            def setText(self, value):
                self.text_value = value

        def open_mask(self):
            self.opened = self._mask_edit.text_value

    screen = Screen()
    screen._mask_edit = Screen._Edit()
    dh.get_handler("curate").apply(Path(project), screen)
    assert len(asked["options"]) == 2, "it took one of two masks without asking"
    assert screen.opened == os.path.join(project, "masks",
                                         "plate1_A01_1.tif")


def test_napari_bridge_takes_a_mask_dropped_directly(project):
    from pathlib import Path

    class Screen:
        def __init__(self):
            self.paths = {}

        def set_paths(self, mask="", image=""):
            self.paths = {"mask": mask, "image": image}

    screen = Screen()
    mask = Path(project) / "masks" / "plate1_A01_1.tif"
    dh.get_handler("napari_bridge").apply(mask, screen)
    assert screen.paths["mask"] == str(mask)


def test_distributed_jobs_finds_the_settings_snapshot_in_a_plate(project):
    from pathlib import Path

    class Combo:
        current = ""

        def setCurrentText(self, value):
            self.current = value

    class Edit:
        value = ""

        def setText(self, text):
            self.value = text

    class Screen:
        def __init__(self):
            self._settings_path = Edit()
            self._module = Combo()

    screen = Screen()
    dh.get_handler("distributed_jobs").apply(Path(project), screen)
    assert screen._settings_path.value == os.path.join(
        project, "settings", "gen_mask_settings.csv")
    assert screen._module.current == "mask"


# ---------------------------------------------------------------------------
# It still costs about a millisecond
# ---------------------------------------------------------------------------

#: The longest a drop event may take to return, in seconds. The event is what
#: the user is holding the mouse button through; anything the handler defers
#: to a worker is not measured here and is covered by
#: tests/qt/test_dnd_handlers_full.py's watchdog.
#:
#: Stated, not derived, the same way tests/qt/test_gui_responsiveness.py
#: states its budget, and matching that file's 400 ms for the same reason: the
#: box this was written on runs about twenty suites at once, and a drop
#: measured at a millisecond idle was seen at 101 ms under that load. A
#: responsiveness test that goes red because the machine is busy gets deleted
#: rather than fixed, so the tight number is asserted separately and without
#: Qt in the way, by
#: :func:`test_resolving_a_drop_really_does_cost_about_a_millisecond`.
DISPATCH_BUDGET_S = 0.400

#: What resolving one drop is allowed to cost, in seconds, measured directly.
#: The real number on an idle machine is 0.04-0.11 ms against a project with
#: 40 000 crops under it; five milliseconds is fifty times that and still
#: three orders of magnitude below the recursive glob it refuses to do.
RESOLVE_BUDGET_S = 0.005


def _drop(widget, paths):
    """Replay the window system's enter -> move -> drop on ``widget``."""
    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
    QApplication.sendEvent(widget, QDragEnterEvent(
        QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    QApplication.sendEvent(widget, QDragMoveEvent(
        QPoint(4, 4), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier))
    event = QDropEvent(QPointF(4, 4), Qt.CopyAction, mime,
                       Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(widget, event)
    return event


@pytest.fixture(scope="session")
def crowded_project(tmp_path_factory):
    """A project whose ``data/`` holds 40 000 crops in 400 folders.

    The layout walk must not be tempted into it. ``resolve_drop`` resolves
    only the ports the module *consumes*, and the one recursive pattern among
    them (Classify's optional ``data/**/*_png``) is not what fills the field --
    so the cost of a drop is a handful of ``stat`` calls no matter how much is
    underneath. This fixture is what would make that untrue if it stopped
    being so.
    """
    root = _make_project(tmp_path_factory.mktemp("crowded") / "plate1")
    for well in range(1, 21):
        for field in range(1, 21):
            leaf = os.path.join(root, "data", f"A{well:02d}",
                                f"f{field:03d}_png")
            os.makedirs(leaf)
            for obj in range(100):
                open(os.path.join(leaf, f"{obj}.png"), "wb").close()
    return root


#: How much cheaper resolving a drop has to be than the recursive glob it
#: avoids. Stated as a ratio rather than a duration because both halves are
#: measured on the same machine in the same test: an absolute budget large
#: enough not to be flaky on CI is also large enough to pass with the glob
#: left in, which is the failure this exists to catch.
CHEAPER_THAN_THE_WALK = 20


@pytest.mark.parametrize("key,kinds", [
    ("measure", ()),
    ("umap", ()),
    ("classify", ()),
    ("graph_builder", (P.MEASUREMENTS_DB,)),
    ("pipeline_graph", ()),
])
def test_resolving_a_drop_really_does_cost_about_a_millisecond(
        crowded_project, key, kinds):
    """The claim, measured without Qt between it and the clock.

    The *best* of five runs, not the mean: this box runs many suites at once
    and a scheduler stall would otherwise be attributed to the resolution.
    A best-of cannot be faked in the other direction -- work that is really
    being done cannot vanish from every run.
    """
    best = min(_time(lambda: ch.resolve_drop(key, crowded_project,
                                             kinds=kinds))
               for _ in range(5))
    assert best < RESOLVE_BUDGET_S, (
        f"resolving a drop for {key} took {best * 1000:.2f} ms against a "
        f"{RESOLVE_BUDGET_S * 1000:.0f} ms budget; something is reading the "
        "tree rather than the layout")


def _time(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def test_resolving_a_drop_is_far_cheaper_than_the_walk_it_avoids(
        crowded_project):
    """Guards the guard: the thing being avoided has to actually cost something.

    The expensive thing is not ``os.walk``; it is the one recursive glob among
    the declared ports, Classify's ``data/**/*_png``. Filling a field must
    never reach for it, and this measures both sides so the comparison holds
    on a slow disk as well as a fast one.
    """
    port = P.module_ports("classify").port("crops")
    start = time.perf_counter()
    resolved = P.resolve_port(port, crowded_project)
    walk = time.perf_counter() - start
    assert resolved.count >= 400

    start = time.perf_counter()
    ch.resolve_drop("classify", crowded_project)
    drop = time.perf_counter() - start

    assert drop * CHEAPER_THAN_THE_WALK < walk, (
        f"resolving a drop took {drop * 1000:.1f} ms against "
        f"{walk * 1000:.1f} ms for the recursive glob it is supposed to be "
        f"avoiding — less than {CHEAPER_THAN_THE_WALK}x apart, so something "
        "in the resolution is walking the crops.")


@pytest.mark.parametrize("key", ["measure", "umap", "graph_builder",
                                 "pipeline_graph", "qc_dashboard"])
def test_a_drop_on_a_crowded_project_still_costs_a_millisecond(
        qtbot, crowded_project, key):
    """The drop event returns immediately, whatever is under the folder."""
    from spacr.qt.dnd import install_dropzone

    class Screen(QObject):
        """Enough of a screen to receive whatever the handler hands over."""

        def __init__(self):
            super().__init__()
            self.seen = []

        def load_path(self, path, table=None):
            self.seen.append(path)

        def load_project(self, root):
            self.seen.append(root)

        def set_source(self, src):
            self.seen.append(src)

    from PySide6.QtWidgets import QWidget

    widget = QWidget()
    qtbot.addWidget(widget)
    screen = Screen()
    install_dropzone(widget, dh.get_handler(key), screen)

    start = time.perf_counter()
    event = _drop(widget, [crowded_project])
    elapsed = time.perf_counter() - start

    assert event.isAccepted()
    assert elapsed < DISPATCH_BUDGET_S, (
        f"the drop event on {key} took {elapsed * 1000:.0f} ms to return "
        f"(budget {DISPATCH_BUDGET_S * 1000:.0f} ms); something in the "
        "resolution is walking the tree on the GUI thread")


def test_resolving_a_crowded_project_never_walks_its_crops(crowded_project,
                                                           monkeypatch):
    """Measured as an absence, not as a duration: the walk never happens.

    A timing budget passes on a fast disk with a warm cache even when the
    walk is there. This asserts the ``data/`` tree is not descended at all.
    """
    visited = []
    real_walk = os.walk

    def counting(top, *args, **kwargs):
        visited.append(str(top))
        return real_walk(top, *args, **kwargs)

    monkeypatch.setattr(os, "walk", counting)
    for module in ("measure", "umap", "mask"):
        ch.resolve_drop(module, crowded_project)
    crops = os.path.join(crowded_project, "data")
    assert not [v for v in visited if v.startswith(crops)], visited


# ---------------------------------------------------------------------------
# Small things that would only ever fail in front of a user
# ---------------------------------------------------------------------------

def test_the_resolution_cache_does_not_outlive_its_answer(tmp_path):
    """Run a step, drop the same folder again, and find what appeared.

    One drop asks the handler four questions -- can you take it, why not,
    what else would work, take it -- so the answer is memoised. Memoising it
    for the session instead would mean a plate that had nothing in it when
    you first tried still has nothing in it after Measure has run.
    """
    from pathlib import Path

    plate = tmp_path / "plate1"
    plate.mkdir()
    handler = dh.get_handler("graph_builder")
    assert handler.can_accept(Path(plate)) is False

    _make_project(plate)
    os.utime(plate)
    assert handler.can_accept(Path(plate)) is True


def test_a_database_in_a_folder_with_a_question_mark_still_lists_its_tables(
        tmp_path):
    """A URI is not a path. Quoting is what makes the read-only open safe."""
    from pathlib import Path

    odd = tmp_path / "screen #3 (rep?)"
    odd.mkdir()
    db = odd / "measurements.db"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE cell (a int)")
    connection.commit()
    connection.close()
    assert dh.table_names(Path(db)) == ["cell"]


def test_dropping_a_root_the_browser_already_watches_is_not_an_error(project):
    """``add_root`` returns False for a duplicate; that is a no-op, not a fault."""
    from pathlib import Path

    class Screen:
        def __init__(self):
            self.roots = []

        def add_root(self, path, scan=True):
            if path in self.roots:
                return False
            self.roots.append(path)
            return True

    screen = Screen()
    handler = dh.get_handler("project_browser")
    handler.apply(Path(project), screen)
    handler.apply(Path(project), screen)        # must not raise
    assert screen.roots == [project]


def test_the_layer_viewer_takes_an_image_and_its_mask_in_one_drop(project):
    """A viewer stacks layers, so a multi-drop lands as two, not as the first."""
    from pathlib import Path

    class Screen:
        def __init__(self):
            self.images = []
            self.labels = []

        def add_image_file(self, path):
            self.images.append(path)

        def add_labels_file(self, path):
            self.labels.append(path)

    screen = Screen()
    handler = dh.get_handler("layer_viewer")
    assert handler.accepts_multiple() is True
    image = Path(project) / "plate1_A01_f01_ch1.tif"
    mask = Path(project) / "masks" / "plate1_A01_1.tif"
    handler.apply(image, screen)
    handler.apply(mask, screen)
    assert screen.images == [str(image)]
    assert screen.labels == [str(mask)], (
        "a file out of masks/ is a label array, not another image")


class _SweepList:
    """Stand-in for ``FilePathListWidget``: the two methods a drop uses."""

    def __init__(self):
        self.paths = []

    def add_paths(self, paths):
        self.paths.extend(str(path) for path in paths)
        return len(self.paths)


class _SweepScreen:
    """Parameter Sweep reduced to the two inputs the handler fills."""

    def __init__(self):
        self.score_data = _SweepList()
        self.count_data = _SweepList()


def test_the_sweep_sorts_a_dropped_csv_by_its_header_not_its_name(tmp_path):
    """A count table filed as a score is a wrong sweep, not a visible error.

    Parameter Sweep holds its two inputs in separate list widgets, so the
    side has to be decided before the file is added -- unlike Regression,
    whose paired table can show the user a mistake. The rule is the same one
    Regression uses (:func:`spacr.qt.widgets.file_list.side_for_header`), and
    it reads the header, so the filenames here are deliberately swapped: the
    count export is called ``scores.csv``. Sorting by name would put both
    files on the wrong side and nothing downstream would say so.
    """
    counts = tmp_path / "scores.csv"
    counts.write_text("grna,count\ng1,5\n", encoding="utf-8")
    scores = tmp_path / "counts.csv"
    scores.write_text("prc,pred\nplate1_r1_c1,0.5\n", encoding="utf-8")

    screen = _SweepScreen()
    handler = dh.get_handler("parameter_sweep")
    assert handler.accepts_multiple() is True, (
        "a sweep is many plates; a multi-drop must land as many, not as one")
    handler.apply(counts, screen)
    handler.apply(scores, screen)

    assert screen.count_data.paths == [str(counts)]
    assert screen.score_data.paths == [str(scores)]


def test_dropping_a_folder_of_tables_fills_both_sweep_lists(tmp_path):
    """"Drop the folder" is the gesture a many-plate screen is for."""
    folder = tmp_path / "plates"
    folder.mkdir()
    (folder / "plate1_counts.csv").write_text(
        "grna,count\ng1,5\n", encoding="utf-8")
    (folder / "plate1_scores.csv").write_text(
        "prc,pred\nplate1_r1_c1,0.5\n", encoding="utf-8")
    (folder / "notes.txt").write_text("ignored", encoding="utf-8")

    screen = _SweepScreen()
    handler = dh.get_handler("parameter_sweep")
    assert handler.can_accept(folder) is True
    handler.apply(folder, screen)

    assert screen.count_data.paths == [str(folder / "plate1_counts.csv")]
    assert screen.score_data.paths == [str(folder / "plate1_scores.csv")]


def test_the_sweep_refuses_a_path_that_holds_no_table(tmp_path):
    """Refusing is what puts the reason on screen instead of silence."""
    empty = tmp_path / "empty"
    empty.mkdir()
    handler = dh.get_handler("parameter_sweep")

    assert handler.can_accept(empty) is False
    assert "score" in handler.error_message(empty).lower()
