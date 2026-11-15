"""``MainWindow`` and its module screens: reaching, building, rebuilding, seeding.

Every test drives the real window offscreen.  A probe app (:data:`PROBE`)
registered with a factory this file controls stands in where a test needs a
screen whose collaborators can be made to fail; everything else is the
shipping registry and the shipping screens.
"""
from __future__ import annotations

import logging
import types

import pytest
from PySide6.QtCore import QPointF
from PySide6.QtGui import QEnterEvent
from PySide6.QtWidgets import QApplication, QLineEdit, QPushButton, QWidget

from spacr.qt import app as app_mod
from spacr.qt import i18n as i18n_mod
from spacr.qt.app import APPS, MainWindow
from spacr.qt.screens import app_screen as app_screen_mod
from spacr.qt.screens import settings_model as settings_model_mod
from spacr.qt.widgets.home import AppTile
from tests.qt.test_cov_qt_app import _close_owned_screens, win  # noqa: F401
from tests.qt.test_startup_ai_reporting_choice import private_ai_profile  # noqa: F401

pytestmark = pytest.mark.qt

PROBE = "main_window_probe"


class _ProbeScreen(QWidget):
    """A module screen that records what the window asks of it."""

    def __init__(self, boom=None):
        super().__init__()
        self.boom = boom if boom is not None else {}
        self.workspaces = 0
        self.seeds: list = []
        self.built_for = None

    def register_workspace(self):
        self.workspaces += 1
        if self.boom.get("workspace"):
            raise RuntimeError("the workspace store is gone")

    def apply_seed(self, seed):
        self.seeds.append(seed)
        if self.boom.get("seed"):
            raise ValueError("this screen cannot take that seed")


@pytest.fixture
def probe_app():
    """Register :data:`PROBE`; ``boom`` breaks a collaborator, ``fail_build``
    breaks the factory, ``built`` lists every screen made."""
    state = {"boom": {}, "built": [], "fail_build": False}

    def factory():
        if state["fail_build"]:
            raise RuntimeError("this screen cannot be built today")
        screen = _ProbeScreen(state["boom"])
        screen.built_for = dict(
            app_screen_mod.AppScreen.values_the_next_screen_is_built_for or {})
        state["built"].append(screen)
        return screen

    app_mod.register_app(PROBE, "Probe", "a screen this file owns",
                         app_mod.SECTION_TOOLS, factory=factory)
    return state


class _Errors(logging.Handler):
    """Collect what ``spacr.qt.app`` logs, whatever the propagation."""

    def __init__(self):
        super().__init__(logging.DEBUG)
        self.messages: list = []

    def emit(self, record):
        self.messages.append(record.getMessage())

    def text(self):
        return "\n".join(self.messages)


@pytest.fixture
def app_log():
    handler = _Errors()
    logger = logging.getLogger("spacr.qt.app")
    level, disabled = logger.level, logging.root.manager.disable
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logging.disable(logging.NOTSET)
    try:
        yield handler
    finally:
        logging.disable(disabled)
        logger.setLevel(level)
        logger.removeHandler(handler)


def _spacr_menu(window):
    return next(top.menu() for top in window.menuBar().actions()
                if top.text().replace("&", "") == "spaCR")


def _menu_entries(menu, found=None):
    """Every leaf action under ``menu``, as {text: {statusTips}} and actions."""
    found = found if found is not None else {}
    for act in menu.actions():
        if act.isSeparator():
            continue
        if act.menu() is not None:
            _menu_entries(act.menu(), found)
        else:
            found.setdefault(act.text(), []).append(act)
    return found


# ---------------------------------------------------------------------------
# reaching a module
# ---------------------------------------------------------------------------

def test_every_app_key_builds_its_own_screen_once_and_names_it(
        win, qtbot, qt_theme_applied, subtests):
    """Navigation builds a screen once, reuses it, names it, and Home returns."""
    with subtests.test(msg="built once, then reused"):
        before = win._stack.count()
        win._on_nav_selected("measure")
        first = win._screens["measure"]
        assert win._stack.count() == before + 1
        win._on_nav_selected("mask")
        win._on_nav_selected("measure")
        assert win._screens["measure"] is first
        assert win._stack.currentWidget() is first
        assert win._stack.count() == before + 2

    with subtests.test(msg="__home__ returns to the startup page"):
        win._on_nav_selected("umap")
        assert win._stack.currentWidget() is win._screens["umap"]
        win._on_nav_selected("__home__")
        assert win._stack.currentWidget() is win._startup
        assert win._status_app_label.text() == "Home"
        assert "__home__" not in win._screens

    with subtests.test(msg="every registered key"):
        wrong = []
        for key, name, _d, _s in APPS:
            win._on_nav_selected(key)
            screen = win._screens.get(key)
            if not (isinstance(screen, QWidget)
                    and win._stack.currentWidget() is screen
                    and win._status_app_label.text() == name):
                wrong.append(key)
        assert wrong == []
        assert set(win._screens) == {k for k, *_r in APPS}

    with subtests.test(msg="the window can open straight into an app"):
        straight = MainWindow(initial_app="mask")
        qtbot.addWidget(straight, before_close_func=_close_owned_screens)
        assert straight._stack.currentWidget() is straight._screens["mask"]
        assert straight._status_app_label.text() == "Mask"
        assert list(straight._screens) == ["mask"]


def test_every_surface_of_the_window_navigates_where_it_says(
        qtbot, qt_theme_applied, private_ai_profile, subtests):
    """Home tiles, dock rows, the spaCR menu and dock hover all reach their module."""
    from spacr.qt.organisms import ORGANISMS

    preferences, _, _ = private_ai_profile
    preferences.set_easy_mode(False)
    win = MainWindow()
    qtbot.addWidget(win, before_close_func=_close_owned_screens)

    with subtests.test(msg="clicking a Home tile"):
        tiles = {t.text_label: t for t in win._startup.findChildren(AppTile)}
        tiles["Measure"].click()
        assert win._status_app_label.text() == "Measure"
        assert win._stack.currentWidget() is win._screens["measure"]

    with subtests.test(msg="clicking a dock row, then the Home row"):
        rows = {b.property("navKey"): b
                for b in win._sidebar.findChildren(QPushButton)}
        rows["regression"].click()
        assert win._status_app_label.text() == "Regression"
        assert win._stack.currentWidget() is win._screens["regression"]
        rows["__home__"].click()
        assert win._stack.currentWidget() is win._startup

    with subtests.test(msg="the spaCR menu lists every app with its summary"):
        summaries = {}
        for guide in ORGANISMS.values():
            for key, _title, summary, _icon in guide["modules"]:
                if key:
                    summaries.setdefault(key, set()).add(summary)
        entries = _menu_entries(_spacr_menu(win))
        missing = [key for key, name, desc, _s in APPS
                   if not ({a.statusTip() for a in entries.get(name, [])}
                           & summaries.get(key, {desc}))]
        assert missing == []

    with subtests.test(msg="a menu entry and the Home entry navigate"):
        entries = _menu_entries(_spacr_menu(win))
        entries["Image UMAP"][0].trigger()
        assert win._status_app_label.text() == "Image UMAP"
        assert win._stack.currentWidget() is win._screens["umap"]
        entries["Home"][0].trigger()
        assert win._stack.currentWidget() is win._startup

    with subtests.test(msg="dock hover announces help on Home at once"):
        win.resize(1400, 1000)
        win.show()
        page = win._startup
        assert page._hint_bar is not None
        win._dock_edge.set_collapsed(False)
        row = next(r for r in win._sidebar.rows() if r.key == "measure")
        qtbot.waitUntil(lambda: row.isVisibleTo(win))
        announced: list = []
        win._sidebar.module_hovered.connect(announced.append)
        point = row.rect().center()
        QApplication.sendEvent(row, QEnterEvent(
            QPointF(point), QPointF(row.mapTo(win, point)),
            QPointF(row.mapToGlobal(point))))
        assert announced == ["measure"]
        assert page._hint_bar.module_key == "measure"
        assert page._hint_bar.text() != ""
        assert page._hint_bar.isVisibleTo(page)
        assert row.is_hovered() is True
        assert win._stack.currentWidget() is page


# ---------------------------------------------------------------------------
# building a screen
# ---------------------------------------------------------------------------

DEDICATED = {
    "toxoplasma": "OrganismScreen", "plasmodium": "OrganismScreen",
    "candida": "OrganismScreen",
    "annotate": "AnnotateScreen", "make_masks": "MakeMasksScreen",
    "queue": "QueueScreen", "db_browser": "DbBrowserScreen",
    "agreement": "AgreementScreen", "plate_view": "PlateViewScreen",
    "model_compare": "ModelCompareScreen", "align": "AlignScreen",
    "convert": "ConvertScreen", "foreign": "ForeignScreen",
    "batch": "BatchScreen", "distributed_jobs": "DistributedJobsScreen",
    "model_zoo": "ModelZooScreen", "report": "ReportScreen",
    "train_compare": "TrainCompareScreen",
    "classifier_evaluation": "ClassifierEvaluationScreen",
    "run_history": "RunHistoryScreen",
    "train_cellpose": "CellposeWorkbenchScreen",
}

#: Keys whose screen is built for its own job rather than being the generic
#: settings form (``AppScreen``) over a pipeline entry point.
DEDICATED_KEYS = set(DEDICATED) | {
    "trypanosoma", "leishmania", "giardia", "virus", "mammalian",
    "curate", "data_manager", "experiment_design", "graph_builder",
    "hit_list", "image_scatter", "layer_viewer", "lineage", "methods_export",
    "pca", "pipeline_graph", "power", "profiler", "qc_dashboard",
    "run_compare", "tabulate", "explain_cv", "investigate_hit",
    "volcano_explorer", "parameter_sweep", "dose_response", "gate_editor",
    "project_browser"}


def test_build_screen_picks_the_dedicated_class_the_plugin_or_the_generic_form(
        win, probe_app, monkeypatch, caplog, subtests):
    """Dedicated keys get their class, others an AppScreen for that key; plugin
    screens must be widgets; a broken registry or help pass costs nothing."""
    from spacr import plugins
    from spacr.qt.screens.app_screen import AppScreen

    with subtests.test(msg="dedicated classes by key"):
        built = {}
        for key in DEDICATED:
            screen = win._build_screen(key)
            built[key] = type(screen).__name__
            screen.deleteLater()
        assert built == DEDICATED

    with subtests.test(msg="every registered key: generic or dedicated"):
        generic, dedicated = set(), set()
        for key, *_r in APPS:
            screen = win._build_screen(key)
            if isinstance(screen, AppScreen):
                assert screen.app_key == key
                generic.add(key)
            elif key != PROBE:
                dedicated.add(key)
            screen.deleteLater()
        keys = {k for k, *_r in APPS}
        assert generic, "expected some generic AppScreen apps"
        assert dedicated == DEDICATED_KEYS & keys

    with subtests.test(msg="the Cellpose workbench is wired to the window"):
        submissions: list = []
        with monkeypatch.context() as m:
            m.setattr(win, "_on_remote_submit_requested",
                      lambda key, settings: submissions.append((key, settings)))
            screen = win._build_screen("train_cellpose")
            screen.remote_submit_requested.emit("train_cellpose", {"epochs": 3})
        assert submissions == [("train_cellpose", {"epochs": 3})]
        assert screen.close() is True
        assert [page.isVisible() for page in screen._screens] == [
            False] * len(screen._screens)
        screen.deleteLater()

    with subtests.test(msg="a plugin screen must be a widget"), \
            monkeypatch.context() as m:
        class _Contribution:
            key = "__main_window_plugin__"
            screen_factory = "main_window_plugin:make_screen"

        m.setattr(plugins, "get_app", lambda key: _Contribution())
        made = QWidget()
        m.setattr(plugins, "load_object", lambda path: (lambda app_key: made))
        assert win._build_screen("__main_window_plugin__") is made
        m.setattr(plugins, "load_object",
                  lambda path: (lambda app_key: "a string"))
        with pytest.raises(TypeError) as raised:
            win._build_screen("__main_window_plugin__")
        assert "main_window_plugin:make_screen" in str(raised.value)
        assert "expected QWidget" in str(raised.value)

    with subtests.test(msg="an unreadable plugin registry falls back to the "
                           "built-in factory"), monkeypatch.context() as m:
        m.setattr(plugins, "get_app", lambda key: (_ for _ in ()).throw(
            RuntimeError("plugin index gone")))
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
            screen = win._build_screen(PROBE)
        assert screen is probe_app["built"][-1]
        assert f"Could not inspect plugin screen contribution {PROBE}" \
            in caplog.text
        m.setattr(plugins, "get_app", lambda key: None)
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
            again = win._build_screen(PROBE)
        assert again is probe_app["built"][-1] and again is not screen
        assert "Could not inspect plugin screen contribution" not in caplog.text

    with subtests.test(msg="a help pass that raises does not stop the module "
                           "opening"), monkeypatch.context() as m:
        seen: list = []
        boom = {"on": True}

        def _retarget(widget):
            if isinstance(widget, _ProbeScreen):
                seen.append(widget)
            if boom["on"]:
                raise RuntimeError("the tooltip table is unreadable")

        m.setattr(settings_model_mod, "retarget_field_tooltips", _retarget)
        win._screens.pop(PROBE, None)
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
            win._on_nav_selected(PROBE)
        screen = probe_app["built"][-1]
        assert seen == [screen]
        assert f"Could not retarget help on the {PROBE} screen" in caplog.text
        assert win._stack.currentWidget() is screen
        assert win._status_app_label.text() == "Probe"
        boom["on"] = False
        win._on_nav_selected("__home__")
        win._on_nav_selected(PROBE)
        assert seen == [screen, screen]
        assert win._stack.currentWidget() is screen


# ---------------------------------------------------------------------------
# rebuilding a screen
# ---------------------------------------------------------------------------

def test_rebuild_app_screen_swaps_in_a_dressed_replacement_or_keeps_the_old(
        win, probe_app, monkeypatch, caplog, subtests):
    """A rebuild shows the replacement built from the values; a failing build,
    dressing pass or retire never costs the working screen."""
    from spacr.qt.screens.app_screen import AppScreen

    sentinel = {"left_over": "from an earlier build"}
    monkeypatch.setattr(AppScreen, "values_the_next_screen_is_built_for",
                        sentinel)

    with subtests.test(msg="never built before"), monkeypatch.context() as m:
        fresh = QWidget()
        m.setattr(win, "_build_screen", lambda key: fresh)
        assert "mask" not in win._screens
        win.rebuild_app_screen("mask", {"nucleus_channel": 1})
        assert win._screens["mask"] is fresh
        assert win._stack.currentWidget() is fresh
        assert AppScreen.values_the_next_screen_is_built_for is sentinel

    with subtests.test(msg="a failed build keeps the old formless screen"):
        old = QWidget()
        win._screens[PROBE] = old
        win._stack.addWidget(old)
        probe_app["fail_build"] = True
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
            win.rebuild_app_screen(PROBE, {"nucleus_channel": 2})
        assert f"could not rebuild the {PROBE} screen" in caplog.text
        assert win._screens[PROBE] is old
        assert probe_app["built"] == []
        assert AppScreen.values_the_next_screen_is_built_for is sentinel
        probe_app["fail_build"] = False
        win.rebuild_app_screen(PROBE, {"nucleus_channel": 2})
        fresh = probe_app["built"][-1]
        assert fresh.built_for == {"nucleus_channel": 2}
        assert win._screens[PROBE] is fresh
        assert win._stack.currentWidget() is fresh
        assert AppScreen.values_the_next_screen_is_built_for is sentinel

    with subtests.test(msg="every dressing pass failing still shows it"), \
            monkeypatch.context() as m:
        boom = {"on": True}
        themed, translated, retargeted = [], [], []

        def _raise_when_on():
            if boom["on"]:
                raise RuntimeError("the dressing pass failed")

        m.setattr(MainWindow, "_theme_screen",
                  lambda self, screen, key: themed.append(screen)
                  or _raise_when_on())
        m.setattr(i18n_mod, "retranslate_widget_tree",
                  lambda w: translated.append(w) or _raise_when_on())
        m.setattr(settings_model_mod, "retarget_field_tooltips",
                  lambda w: retargeted.append(w) or _raise_when_on())
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
            win.rebuild_app_screen(PROBE, {})
        fresh = probe_app["built"][-1]
        assert [themed[-1], translated[-1], retargeted[-1]] == [fresh] * 3
        for message in (f"Could not theme the rebuilt {PROBE} screen",
                        f"Could not translate the rebuilt {PROBE} screen",
                        f"Could not retarget help on the {PROBE} screen"):
            assert message in caplog.text
        assert win._screens[PROBE] is fresh
        assert win._stack.currentWidget() is fresh
        boom["on"] = False
        caplog.clear()
        with caplog.at_level(logging.ERROR, logger="spacr.qt.app"):
            win.rebuild_app_screen(PROBE, {})
        second = probe_app["built"][-1]
        assert second is not fresh
        assert [themed[-1], translated[-1], retargeted[-1]] == [second] * 3
        assert "rebuilt" not in caplog.text
        assert win._stack.currentWidget() is second

    with subtests.test(msg="the old screen will not retire"):
        old = QWidget()
        win._screens[PROBE] = old
        win._stack.addWidget(old)
        probe_app["boom"]["workspace"] = True
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
            win.rebuild_app_screen(PROBE, {})
        fresh = probe_app["built"][-1]
        assert fresh.workspaces == 2, "the failed publish is retried once"
        assert f"could not retire the {PROBE} screen" in caplog.text
        assert f"could not restore {PROBE} workspace" in caplog.text
        assert win._screens[PROBE] is fresh
        assert win._stack.currentWidget() is fresh
        assert win._stack.indexOf(old) != -1
        probe_app["boom"]["workspace"] = False
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="spacr.qt.app"):
            win.rebuild_app_screen(PROBE, {})
        second = probe_app["built"][-1]
        assert second.workspaces == 1
        assert win._stack.indexOf(fresh) == -1
        assert f"could not retire the {PROBE} screen" not in caplog.text


# ---------------------------------------------------------------------------
# seeding a screen
# ---------------------------------------------------------------------------

def test_a_train_request_lands_its_seed_where_a_form_can_take_it(
        win, probe_app, monkeypatch, caplog, subtests):
    """Seeds land on the screen carrying the key; unknown keys, rejected
    values, formless screens and Home are navigation only."""
    with subtests.test(msg="classify resolves to the merged Classify"):
        win._on_train_requested("classify", {"src": "/data/plate1",
                                             "epochs": 7})
        assert "classify" not in win._screens, "an orphan screen was built"
        screen = win._screens["classify_merged"]
        assert win._stack.currentWidget() is screen
        assert screen._settings_model.collect()["src"] == "/data/plate1"
        widgets = screen._settings_model._widgets
        if "epochs" in widgets:
            assert widgets["epochs"].value() == 7
        paths = ["/data/plate1", "/data/plate2"]
        win._on_train_requested("classify", {"src": paths})
        assert screen._settings_model.collect()["src"] == paths

    with subtests.test(msg="a rejected value keeps the rest of the seed"):
        win._on_train_requested("classify", {"src": "/ok",
                                             "epochs": "not-a-number"})
        assert win._screens["classify_merged"]._settings_model.collect()[
            "src"] == "/ok"

    no_ops = [
        ("unknown setting keys", "classify", {"no_such_setting_at_all": 1},
         "classify_merged"),
        ("a screen without settings", "db_browser", {"src": "/x"}, "db_browser"),
        ("Home", "__home__", {"src": "/x"}, None),
    ]
    for name, key, seed, shown in no_ops:
        with subtests.test(msg=name):
            win._on_train_requested(key, seed)
            expected = win._startup if shown is None else win._screens[shown]
            assert win._stack.currentWidget() is expected

    with subtests.test(msg="only a form takes a seed"), \
            monkeypatch.context() as m:
        m.setattr(win, "open_module", lambda key: key)
        landed: list = []
        real_apply = MainWindow._apply_seed_value
        m.setattr(MainWindow, "_apply_seed_value", staticmethod(
            lambda w, value: landed.append(value) or real_apply(w, value)))
        win._screens["mw_formless"] = QWidget()
        win._on_train_requested("mw_formless", {"nucleus_channel": "1"})
        assert landed == []
        field = QLineEdit()
        seeded = QWidget()
        seeded._settings_model = types.SimpleNamespace(
            _widgets={"nucleus_channel": field})
        win._screens["mw_seeded"] = seeded
        win._on_train_requested("mw_seeded",
                                {"nucleus_channel": "1", "absent_setting": 7})
        assert landed == ["1"]
        assert field.text() == "1"

    with subtests.test(msg="a seed the screen refuses is logged with it"):
        probe_app["boom"]["seed"] = True
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
            win._on_train_requested(PROBE, {"target_gene": "PTEN"})
        screen = probe_app["built"][-1]
        assert screen.seeds == [{"target_gene": "PTEN"}]
        assert f"Could not seed {PROBE} with {{'target_gene': 'PTEN'}}" \
            in caplog.text
        assert win._stack.currentWidget() is screen
        probe_app["boom"]["seed"] = False
        caplog.clear()
        with caplog.at_level(logging.WARNING, logger="spacr.qt.app"):
            win._on_train_requested(PROBE, {"target_gene": "TP53"})
        assert screen.seeds == [{"target_gene": "PTEN"}, {"target_gene": "TP53"}]
        assert f"Could not seed {PROBE}" not in caplog.text


# ---------------------------------------------------------------------------
# model zoo, demos and the current-screen snapshot
# ---------------------------------------------------------------------------

def test_the_zoo_hands_models_to_model_compare_only_when_it_can_take_them(
        win, subtests):
    """Both models and the field count arrive; absent keys leave controls be;
    a screen without ``configure`` drops the request but still navigates."""
    with subtests.test(msg="two models and four fields"):
        win._on_zoo_compare_requested({"model_a": "cyto3", "model_b": "nuclei",
                                       "folder": "", "n_fields": 4})
        screen = win._screens["model_compare"]
        assert win._stack.currentWidget() is screen
        assert (screen._panel_a.model_edit.text(),
                screen._panel_b.model_edit.text(),
                screen._fields_box.value()) == ("cyto3", "nuclei", 4)

    with subtests.test(msg="missing keys leave the screen alone"):
        win._on_zoo_compare_requested({})
        assert (screen._panel_a.model_edit.text(),
                screen._panel_b.model_edit.text(),
                screen._fields_box.value()) == ("cyto3", "nuclei", 4)

    with subtests.test(msg="no configure: dropped, navigation kept"):
        placeholder = QWidget()
        win._screens["model_compare"] = placeholder
        win._on_zoo_compare_requested({"model_a": "zoo-drop-probe"})
        assert win._visit_order[-1] == "model_compare"
        assert screen._panel_a.model_edit.text() == "cyto3"
        win._screens["model_compare"] = screen
        win._on_zoo_compare_requested({"model_a": "zoo-drop-probe"})
        assert screen._panel_a.model_edit.text() == "zoo-drop-probe"


def test_a_demo_is_generated_by_key_and_applied_by_the_best_route(
        win, tmp_path, monkeypatch, subtests):
    """Every demo key names a generator and an app; the generated layout goes
    to the settings model first, then ``_open_source``, then ``_open_folder``."""
    import spacr.utils as sutils
    from spacr.qt import synthetic as syn

    with subtests.test(msg="every demo target resolves"):
        keys = {k for k, *_r in APPS}
        assert [d for d, (app, gen) in win.DEMO_TARGETS.items()
                if not callable(getattr(syn, gen, None)) or app not in keys] == []

    with subtests.test(msg="the generator is chosen by demo key"), \
            monkeypatch.context() as m:
        seen: list = []
        m.setattr(syn, "generate_crop_demo",
                  lambda dst, **kw: seen.append(dst) or "sentinel-layout")
        assert win._run_demo_generator("crop", str(tmp_path)) == "sentinel-layout"
        assert seen == [str(tmp_path)]

    layout = win._run_demo_generator("mask", str(tmp_path / "mask"))
    src = str(layout.src)

    class _Settings:
        def __init__(self, seen):
            self.seen = seen

        def apply_settings_dict(self, settings):
            self.seen.append(("settings", settings["src"]))

    class _Source:
        def __init__(self, seen):
            self.seen = seen

        def _open_source(self, path):
            self.seen.append(("source", path))

    class _Folder:
        def __init__(self, seen):
            self.seen = seen

        def _open_folder(self, path):
            self.seen.append(("folder", path))

    class _Both(_Settings, _Source):
        pass

    cases = [
        ("settings model preferred", _Both, False, [("settings", src)]),
        ("open_source fallback", _Source, False, [("source", src)]),
        ("open_folder fallback", _Folder, False, [("folder", src)]),
        ("an unreadable csv falls through", _Both, True, [("source", src)]),
    ]
    for name, cls, unreadable, expected in cases:
        with subtests.test(msg=name), monkeypatch.context() as m:
            if unreadable:
                m.setattr(sutils, "load_settings", lambda *a, **k: None)
            seen: list = []
            win._apply_demo_to_screen(cls(seen), layout)
            assert seen == expected


def test_the_snapshot_names_the_last_app_screen_viewed(win, subtests):
    """"Add current plate" takes the most recently VIEWED settings screen."""
    with subtests.test(msg="a non-form screen falls back to the last form"):
        win._on_nav_selected("classify")
        win._on_nav_selected("queue")
        assert win._stack.currentWidget() is win._screens["queue"]
        key, settings = win._snapshot_current_screen_settings()
        assert key == "classify"
        assert settings == win._screens["classify"]._settings_model.collect()

    with subtests.test(msg="last viewed, not last opened"):
        win._on_nav_selected("mask")
        win._on_nav_selected("measure")
        win._on_nav_selected("mask")
        win._on_nav_selected("queue")
        key, settings = win._snapshot_current_screen_settings()
        assert key == "mask"
        assert settings == win._screens["mask"]._settings_model.collect()
