"""spacr.qt.app — the home-screen shell: sections, tiles, navigation, menus.

Everything here drives the REAL :class:`MainWindow` offscreen. The only
things stubbed are true externalities:

* modal dialogs — a modal blocks the run forever, so every
  ``QMessageBox`` / ``QFileDialog`` entry point is swapped for a
  recorder that also lets a test script the user's answers;
* the PyPI and Hugging Face network calls;
* ``AppScreen._on_run``, which would otherwise start a real
  segmentation/measurement pipeline when the end-to-end demo fires.

Nothing else is faked: the sections, tiles, sidebar, menu bar, screens
and settings models are the shipping objects.
"""
from __future__ import annotations

from contextlib import contextmanager

import os
import sys
import threading
import types

import pytest
from PySide6.QtCore import QObject
from PySide6.QtGui import QFontMetrics, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
)

from spacr.qt import app as app_mod
from spacr.qt.app import (
    _FORCE_GLYPH,
    _ICON_OVERRIDES,
    APPS,
    MAX_APPS_PER_SECTION,
    SECTION_ASSAYS,
    SECTION_CORE,
    SECTIONS,
    MainWindow,
    Sidebar,
    _icon_for_app,
    _PipelinePreloader,
    app_stage,
    home_bands,
    make_home_page,
    section_members,
    tiled_apps,
)
from spacr.qt.widgets.home import AppTile, HomePage

# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _normal_home_preferences(tmp_path, monkeypatch):
    """Select ordinary Home in a private copy of current preferences."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    source = preferences._settings()
    path = str(tmp_path / "normal-home.ini")
    private = QSettings(path, QSettings.IniFormat)
    for key in source.allKeys():
        private.setValue(key, source.value(key))
    private.sync()
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    preferences.set_easy_mode(False)


@pytest.fixture
def normal_win(_normal_home_preferences, request):
    """Build the normal host only after its private preference is selected."""
    return request.getfixturevalue("win")


@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    w = MainWindow()
    qtbot.addWidget(w, before_close_func=_close_owned_screens)
    return w


def _close_owned_screens(window):
    """Retire screens before their MainWindow loses ownership of them.

    Several pyqtgraph views create parentless context-menu windows. Closing
    only the outer MainWindow leaves those menus reachable through Python
    signal cycles even after Qt deletes the parent screen, so a test process
    that opens many windows accumulates hundreds of live top-level widgets.
    The window's own screen registry is the exact ownership boundary; close
    only those children and let pytest-qt delete the window normally.
    """
    for screen in list(getattr(window, "_screens", {}).values()):
        # pyqtgraph's ViewBox menus are deliberately parentless top-level
        # windows and ViewBox.close() does not retire them. Find only the
        # ViewBoxes in graphics scenes owned by this screen; never sweep the
        # QApplication or touch another test's widgets.
        try:
            from PySide6.QtWidgets import QGraphicsView, QMenu
            from pyqtgraph import PlotItem, ViewBox

            def _retire_menu(menu):
                if menu is None:
                    return
                for child in reversed(menu.findChildren(QMenu)):
                    child.close()
                    child.deleteLater()
                menu.close()
                menu.deleteLater()

            seen = set()
            for graphics_view in screen.findChildren(QGraphicsView):
                scene = graphics_view.scene()
                items = list(scene.items()) if scene is not None else []
                for item in items:
                    if isinstance(item, PlotItem):
                        _retire_menu(getattr(item, "ctrlMenu", None))
                for item in items:
                    if not isinstance(item, ViewBox) or id(item) in seen:
                        continue
                    seen.add(id(item))
                    menu = getattr(item, "menu", None)
                    item.close()
                    _retire_menu(menu)
                    item.menu = None
        except (ImportError, RuntimeError):
            pass
        try:
            screen.close()
            screen.deleteLater()
        except RuntimeError:
            pass


class _ModalRecorder:
    """Stand-in for every QMessageBox entry point.

    ``answers`` is a queue of return values for :meth:`question`; when it
    runs dry the answer is No, so a test that forgets to script an answer
    stops the flow rather than silently continuing.
    """

    def __init__(self):
        self.about: list = []
        self.warning: list = []
        self.information: list = []
        self.questions: list = []
        self.answers: list = []
        self.threads: list = []
        #: (objectName, windowTitle, every label's text) per `QDialog.exec`.
        self.dialogs: list = []

    def titles(self, kind: str) -> list:
        return [t for t, _text in getattr(self, kind)]

    def texts(self, kind: str) -> list:
        return [x for _t, x in getattr(self, kind)]


@pytest.fixture
def modals(monkeypatch):
    rec = _ModalRecorder()

    def _question(parent, title, text, *a, **kw):
        rec.questions.append((title, text))
        rec.threads.append(threading.current_thread())
        return rec.answers.pop(0) if rec.answers else QMessageBox.No

    def _warning(parent, title, text, *a, **kw):
        rec.warning.append((title, text))
        rec.threads.append(threading.current_thread())
        return QMessageBox.Ok

    def _information(parent, title, text, *a, **kw):
        rec.information.append((title, text))
        rec.threads.append(threading.current_thread())
        return QMessageBox.Ok

    def _about(parent, title, text, *a, **kw):
        rec.about.append((title, text))
        return None

    monkeypatch.setattr(QMessageBox, "question", _question)
    monkeypatch.setattr(QMessageBox, "warning", _warning)
    monkeypatch.setattr(QMessageBox, "information", _information)
    monkeypatch.setattr(QMessageBox, "about", _about)

    # `QDialog.exec` too, and this one is not optional. b530f70a replaced
    # `QMessageBox.about(...)` with a hand-built About panel that ends in
    # `dialog.exec()`, so `_show_about()` stopped going through the seam above
    # and started a NESTED MODAL EVENT LOOP instead. Offscreen there is nobody
    # to close it: the test did not fail, it hung, and a hung test takes the
    # whole 60-minute Qt job down with it rather than one assertion.
    #
    # Recording the dialog's own title and the text of every label it built
    # keeps the assertions saying what they said about the QMessageBox --
    # "About spaCR" was shown, and the version was in it -- against the widget
    # that is actually shown now.
    from PySide6.QtWidgets import QDialog, QLabel

    def _exec(dialog, *a, **kw):
        labels = [w.text() for w in dialog.findChildren(QLabel)]
        rec.dialogs.append((dialog.objectName(), dialog.windowTitle(),
                            "\n".join(labels)))
        if dialog.objectName() == "AboutDialog":
            rec.about.append((dialog.windowTitle(), "\n".join(labels)))
        return QDialog.Accepted

    monkeypatch.setattr(QDialog, "exec", _exec)
    monkeypatch.setattr(QDialog, "exec_", _exec, raising=False)
    return rec


def _counts() -> dict:
    counts: dict = {}
    for *_r, section in APPS:
        counts[section] = counts.get(section, 0) + 1
    return counts


def _img(icon: QIcon, px: int = 24):
    """Rasterise an icon so two QIcons can be compared by pixels."""
    return icon.pixmap(px, px).toImage()


def _tiles(page: HomePage) -> dict:
    return {t.text_label: t for t in page.findChildren(AppTile)}


# ===========================================================================
# 1. The app registry: sections
# ===========================================================================

#: The section every app is filed under. This is a deliberate ledger, not
#: a mirror of the code: moving an app between sections is a product
#: decision, and this table is where it gets recorded.
#:
#: #16i rewrote 22 of the original rows into two maturity sections, which
#: emptied Data, Segmentation models and Results & QC completely. #16j
#: put every one of them back: an app is filed under what it DOES, and
#: how finished it is lives in :data:`EXPECTED_STAGES` below and is
#: drawn as the tile's hover colour rather than as a place.
EXPECTED_SECTIONS = {
    "toxoplasma": "Organism", "plasmodium": "Organism", "candida": "Organism",
    # 634 (eb2201537): five alpha organism pages under Organism.
    "trypanosoma": "Organism", "leishmania": "Organism", "giardia": "Organism",
    "virus": "Organism", "mammalian": "Organism",
    # REWRITTEN 2026-08-31, when Home was cut from seven categories to
    # four. The user wrote out the tiles they wanted, in the order they
    # wanted them, and this ledger is the record of where every app
    # landed -- including the ones that no longer draw a tile at all.
    #
    # Explore, Results & QC, Design and Segmentation models are gone as
    # PLACES. Every app that lived in one was re-filed:
    #
    #   Explore        -> Tools (Graph Builder, Gate Editor, QC) or Data
    #                     (Lineage, Tabulate, Pipeline Graph)
    #   Results & QC   -> Tools (Image UMAP, Plate Viewer) or Data
    #                     (Report, Run History) or Core (Training Runs,
    #                     Investigate Hit)
    #   Design         -> Data (Experiment Design, Power, Dose-Response)
    #
    # A folded module keeps a section even with no tile: the section is
    # what says which host it belongs behind.
    'align': 'Tools',
    'analyze_plaques': 'Organism',
    'annotate': 'Core',
    'batch': 'Data',
    'classify_merged': 'Core',
    'convert': 'Data',
    'data_manager': 'Data',
    'db_browser': 'Data',
    'distributed_jobs': 'Data',
    # Filed where the two comments above already said they were: an
    # EC50 is fitted to pick the next experiment's concentration, so
    # Dose-Response went to Data with Power, and a gate is an
    # instrument pointed at a finished table, so the Gate Editor went
    # to Tools. Only their ROWS were missing until 2026-09-05 --
    # `app._SELF_REGISTERING_APPS` now registers them (and the Project
    # Browser, which the dock files under Help) at import rather than
    # only when `run()` walks SELF_REGISTERING_MODULES.
    'dose_response': 'Data',
    'experiment_design': 'Data',
    'external_masks': 'Data',
    'foreign': 'Data',
    'gate_editor': 'Tools',
    'graph_builder': 'Tools',
    'host_pathogen': 'Organism',
    'invasion': 'Organism',
    # investigate_hit, profiler and train_compare moved Core -> Tools in
    # 571b6e77c, which split `app_is_visible` from the new `tiled_apps()`
    # so folding a module stopped DELETING its door in the command
    # palette. The move was deliberate and the table was not updated in
    # the same commit, which is the whole reason this table exists.
    'investigate_hit': 'Tools',
    'layer_viewer': 'Data',
    'lineage': 'Data',
    'make_masks': 'Tools',
    'map_barcodes': 'Core',
    'mask': 'Core',
    'measure': 'Core',
    'pipeline_graph': 'Data',
    'plate_view': 'Tools',
    'power': 'Data',
    'profiler': 'Tools',
    'project_browser': 'Data',
    'qc_dashboard': 'Data',
    'queue': 'Data',
    'recruitment': 'Organism',
    'regression': 'Core',
    'replication': 'Organism',
    'report': 'Data',
    'run_compare': 'Data',
    'run_history': 'Data',
    'tabulate': 'Data',
    'train_compare': 'Tools',
    'umap': 'Tools',
}

#: How finished every app is, as a second ledger on the same app keys.
#: Absent from ``APP_STAGE`` means stable, so the two are checked
#: against each other rather than against a copy of the same dict.
EXPECTED_STAGES = {
    "host_pathogen": "alpha",
    "plasmodium": "alpha", "candida": "alpha",
    "trypanosoma": "alpha", "leishmania": "alpha", "giardia": "alpha",
    "virus": "alpha", "mammalian": "alpha",
    # The merged Classify assessment is recorded in maturity.AFFIRMED.
    "classify_merged": "stable",
    "align": "alpha", "convert": "alpha",
    "foreign": "alpha", "external_masks": "alpha",
    "queue": "alpha",
    "batch": "alpha", "distributed_jobs": "alpha",
    "invasion": "alpha", "db_browser": "alpha",
    "plate_view": "alpha", "train_compare": "alpha",
    "run_history": "alpha", "report": "alpha",
    "layer_viewer": "alpha", "graph_builder": "alpha",
    "data_manager": "alpha",
    "power": "alpha", "run_compare": "alpha",
    "lineage": "alpha",
    "pipeline_graph": "alpha", "profiler": "alpha",
    "experiment_design": "alpha", "qc_dashboard": "alpha",
    # The three that a launched GUI had and a bare `import spacr.qt.app` did
    # not. Their rows now arrive from `app._SELF_REGISTERING_APPS`; the stage
    # is the one `app_catalog` has declared for each all along, so nothing was
    # promoted or demoted here -- this ledger simply covers them now.
    "dose_response": "alpha", "gate_editor": "alpha",
    "project_browser": "alpha",
    # Tabulate joined APPS when app.py's _SELF_REGISTERING_APPS started
    # calling its register(); it arrived alpha, like every screen that is
    # built and reachable but not yet trusted end to end. Absent from this
    # table it would read as "stable", which is the one claim nobody has
    # earned yet. PCA arrived with it and has since been folded onto Image
    # UMAP; what its button lights in now is the host fold fallback's to
    # say.
    "tabulate": "alpha", "investigate_hit": "alpha",
    # The Volcano Explorer's entry stood here beside the Parameter Sweep's
    # until it too stopped being a registry row; a stage is a property of a
    # tile on Home. What the fold button lights in afterwards is the host
    # fold fallback's to say, and it is asserted there.
    # `cellpose_masks` stood beside `train_cellpose` here until the two
    # became one Cellpose Workbench tile, and `train_cellpose` until that
    # tile became a button on the Make Masks masthead. A maturity is a
    # property of a tile on Home, so a key that no longer has one drops out;
    # `make_masks.FOLD_FALLBACK` says what its button lights in, and
    # `test_the_fold_fallback_agrees_with_the_registry` asserts it there.
    # Activation, the Hit List and Methods & Results are not here: each
    # folded onto a host and lost its registry row, and a maturity is a
    # property of a tile. What their buttons light up in now comes from
    # the host's own fold record, checked in the fold tests.
    # Make Masks, Analyze Plaques, Replication and Image UMAP were beta
    # until 2026-10-20 and are stable now, so they read the default.
    # A stage is a property of a tile on Home, so a key folded out of the
    # registry drops out of this ledger with its row. What the button
    # lights in afterwards is recorded in the host's fold fallback, and
    # `test_the_switch_lights_in_the_stage_the_tile_LIT` holds it there.
}


def test_every_app_is_filed_under_the_section_it_belongs_to():
    """Section assignment for every app, one entry at a time.

    Five keys left this table with their registry rows: the modules they
    named are folded into a host screen and are reached from there, so
    they have no section because they have no tile. A section is a
    property of a TILE.
    """
    actual = {key: section for key, _n, _d, section in APPS}
    assert actual == EXPECTED_SECTIONS, (
        "an app changed section (or was added/removed). That is allowed — "
        "update EXPECTED_SECTIONS in the same commit so the move is "
        "recorded rather than accidental.")


def test_every_app_carries_the_maturity_it_was_given():
    """Pin every current tile stage, including the assessed Classify screen."""
    actual = {key: app_stage(key) for key, *_r in APPS}
    expected = {key: EXPECTED_STAGES.get(key, "stable")
                for key, *_r in APPS}
    assert actual == expected, (
        "an app changed maturity. That is allowed — record it in "
        "EXPECTED_STAGES in the same commit.")
    counts = {s: sum(1 for v in actual.values() if v == s)
              for s in ("alpha", "beta", "stable")}
    # 36 alpha since PCA and Tabulate started registering. The beta and stable
    # columns have still not moved, which is the shape the docstring above
    # describes: alpha is the column that grows, and only use empties it.
    # 39 alpha since the two model-explanation stages arrived.
    # 39 -> 41 on 2026-08-17: the Volcano Explorer and the Parameter Sweep
    # were registered without any of this file's three ledgers being updated.
    # 8 -> 6 stable on 2026-08-23: Classify (CV) and Classify (ML) were
    # removed from the registry. Both were stable, and the merged Classify
    # screen that replaces them is the one entry now. Their entry points are
    # untouched -- see HEADLESS_ONLY in test_app_registry_parity.
    # 41 -> 40 alpha: the Parameter Sweep gave up its registry row to become
    # the Regression screen's sweep card. Nothing was signed off; the count
    # falls because there is one fewer tile, not one more trusted app.
    # 9 -> 8 beta, for the same kind of reason: Cellpose Masks and Train
    # Cellpose became the two tabs of one Cellpose Workbench tile. Both keys
    # still run; only one of them is now a tile, and this counts tiles.
    # 40 -> 28 alpha and 8 -> 5 beta: the folded modules gave up their
    # rows, Illumination among them -- its settings had been on Measure's
    # panel for some time while its tile stayed on Home. Nothing was
    # signed off and nothing regressed: this counts TILES, and every one
    # of those modules is now a button or a settings category on the
    # screen it was folded into.
    # 28 -> 26 alpha and 5 -> 4 beta: the last three folds gave up their
    # rows. Activation became a tab on Classify, and the Hit List and
    # Methods & Results became pages on Regression. All three still run and
    # all three keep the colour they were assessed in -- a folded module's
    # maturity lives in its host's fold record now, because the registry
    # answers for a key it no longer holds exactly as it answers a typo.
    # 26 -> 29 alpha on 2026-09-05, and NOT because anything was assessed:
    # Dose-Response, the Gate Editor and the Project Browser registered only
    # when `run()` walked SELF_REGISTERING_MODULES, so a bare
    # `import spacr.qt.app` -- which is the registry this file counts -- never
    # saw the three rows a launched GUI had. `app._SELF_REGISTERING_APPS`
    # registers them now. Each has declared stage='alpha' in `app_catalog`
    # since it was written; the column grew by three tiles, not by three
    # demotions.
    # 33 -> 38: the five 634 alpha organism pages (eb2201537).
    # 38 -> 37 alpha, 6 -> 7 stable: Toxoplasma is not an alpha species.
    # 4 -> 0 beta, 8 -> 12 stable on 2026-10-20: Make Masks, Analyze
    # Plaques, Replication and Image UMAP were signed off for the paper.
    assert counts == {"alpha": 36, "beta": 0, "stable": 12}


def test_no_section_is_used_that_was_never_declared():
    """Every declared section is used, and nothing else is.

    It was relaxed to a subset for #16i, when three declared sections
    had nothing filed under them. They are all populated again, so the
    equality is back — with the subset check kept as the first, more
    specific failure message.
    """
    used = {section for *_r, section in APPS}
    assert used <= set(SECTIONS), f"undeclared sections in APPS: {used - set(SECTIONS)}"
    assert used == set(SECTIONS), (
        f"declared sections nothing is filed under: {set(SECTIONS) - used}")
    assert len(SECTIONS) == len(set(SECTIONS)), "duplicate section name"
    assert {s for s, _rows in home_bands()} == used, (
        "Home bands and the filed sections have come apart")
    for section in SECTIONS:
        assert section_members(section), f"{section} has no tab to open"


def test_no_section_holds_more_than_the_cap():
    """Keep every current section within the explicit readability cap.

    The cap is a design constraint rather than a count inferred from Core.
    Crossing it requires a deliberate, meaningfully named split instead of
    silently lengthening a row.
    """
    counts = _counts()
    # 40 since 2026-09-05, raised at the maintainer's instruction. Data hit
    # exactly twenty -- the old ceiling -- the moment the three
    # self-registering modules joined the table, so the next registration
    # there would have tripped the cap rather than caught a mistake.
    #
    # The number is pinned here on purpose: it is a design constraint, and
    # moving it should be a decision somebody makes rather than a number
    # that drifts to fit whatever the registry has become.
    assert MAX_APPS_PER_SECTION == 40
    over = {s: n for s, n in counts.items() if n > MAX_APPS_PER_SECTION}
    assert not over, (
        f"sections over the {MAX_APPS_PER_SECTION}-app cap: {over}. Add a "
        "section with a name that means something instead of lengthening "
        "a row nobody reads to the end of.")


def test_the_core_section_leads_the_ctrl_number_slots():
    """Ctrl+1..9 address APPS[0..8]; Core has to lead them.

    Core now contains the six primary pipeline modules. The first six number
    shortcuts therefore open that complete block, and Ctrl+7..9 continue into
    the next apps in sidebar order. The assertion deliberately follows the
    registry rather than a fixed Core count."""
    core = [k for k, _n, _d, s in APPS if s == SECTION_CORE]
    assert 0 < len(core) <= MAX_APPS_PER_SECTION
    assert [k for k, *_r in APPS[:len(core)]] == core


def test_no_section_is_empty():
    """It used to require three per section ("too small to deserve a
    heading"). The floor that survives is one: no tab opens on an empty
    pane, and no Home band is drawn with nothing under it."""
    empty = sorted(s for s in SECTIONS if not section_members(s))
    assert not empty, f"declared sections with no members: {empty}"
    assert all(rows for _s, rows in home_bands()), "an empty Home band"


def test_sections_are_contiguous_blocks_in_declaration_order():
    """A section split across two blocks would render two identical
    headings on Home and in the sidebar — both walk APPS in order and
    emit a heading whenever the section changes."""
    order: list = []
    for *_r, section in APPS:
        if not order or order[-1] != section:
            order.append(section)
    # Compared as a subsequence, which is the same as equality while
    # every section has apps in it — and stays a statement about ORDER
    # rather than about population if one ever does not.
    assert order == [s for s in SECTIONS if s in set(order)]
    assert len(order) == len(set(order)), (
        f"a section appears in two separate runs of APPS: {order}")


def test_sidebar_draws_exactly_one_heading_per_section_in_order(
        qtbot, qt_theme_applied):
    bar = Sidebar()
    qtbot.addWidget(bar)
    headings = [lbl.text() for lbl in bar.findChildren(QLabel)
                if lbl.objectName() == "SidebarSection"]
    # The sidebar walks `dock_rows()` and heads each run, so it shows the
    # DOCK's grouping.
    #
    # NOT `home_bands()` ANY MORE, and the difference is the point. A section
    # is Home's categorisation and every Help module is tileless, so Help can
    # never be a Home band -- `test_no_section_is_empty` says as much. The
    # dock lists modules whether or not they have a tile, so it gets one more
    # heading than Home does. Comparing the two was what made this test read
    # as "the dock is Home", which it is not.
    from spacr.qt.app import dock_rows

    expected = []
    for _key, _name, _desc, section in dock_rows():
        if not expected or expected[-1] != section:
            expected.append(section)
    assert headings == expected
    # And the extra one is Help, last -- asserted so a future change that
    # quietly drops it fails here rather than in the maintainer's dock.
    assert headings[-1] == "Help"
    assert [s for s, _rows in home_bands()] == headings[:-1]


def test_sidebar_has_one_row_per_app_plus_home_in_apps_order(
        qtbot, qt_theme_applied):
    bar = Sidebar()
    qtbot.addWidget(bar)
    # TOP-LEVEL ROWS ONLY, in APPS order. Folded modules are nested under
    # their hosts in the dock now, so the button list also carries their
    # child rows -- indented, hidden until the host is opened, and not part
    # of the order this test is about.
    # IN DOCK ORDER, which is APPS with the Help-menu modules moved to the
    # end under their own heading -- `dock_rows` is the function that does
    # it, and comparing against raw APPS asserted an order the dock stopped
    # drawing.
    from spacr.qt.app import dock_rows

    keys = [b.property("navKey") for b in bar.findChildren(QPushButton)
            if not b.property("isFoldChild")]
    assert keys == ["__home__"] + [k for k, *_r in dock_rows()]
    # THE DOCK IS FLAT NOW, and that is the change this had to follow. It
    # once nested a folded module as an indented child row, so every key in
    # APPS had a row here; the rewrite gives the dock one row per registry
    # entry it draws and reaches the folded ones through their host's fold
    # strip instead. `train_compare` is the one that caught it: a real
    # module, reachable in the running application, with no dock row at all.
    #
    # So the question this asks is "does the dock explain the rows it DOES
    # draw", and reachability is asserted where it belongs --
    # tests/qt/test_the_dock_mirrors_the_home_tiles.py, which checks every
    # module is a dock row or a fold child of one.
    by_key = {b.property("navKey"): b
              for b in bar.findChildren(QPushButton)}
    for key, name, desc, _s in APPS:
        btn = by_key.get(key)
        if btn is None:
            continue            # folded: reached through its host, not here
        # Each row announces itself to a screen reader with name AND
        # description. The summary is drawn nowhere on the row -- it goes to
        # the strip along the bottom -- so for a screen reader the accessible
        # description is the only route to it.
        assert btn.accessibleName().strip() == name
        assert btn.accessibleDescription() == desc, (
            f"the dock row for {key} describes itself to nobody")
        # AND NO TOOLTIP. The module popup came off on 2026-09-03 ("remove
        # the popup window tooltip on the moduals"); the sentence is in the
        # accessible description above and in the hint strip at the foot of
        # the page, with its API and Tutorial links.
        assert btn.toolTip() == "", "the module popup is back"
        # "&&" is how Qt is told to DRAW an ampersand: a lone "&" is a
        # mnemonic, and "Align & Stitch" was rendering as "Align _Stitch"
        # in this column. The accessible name above carries the real
        # string.
        # NO LEADING SPACES ANY MORE. The old dock padded a row's label with
        # two spaces to leave room for the icon it painted itself; the
        # rewritten dock gives the icon its own slot, so the text starts at
        # the text.
        assert btn.full_text() == name.replace("&", "&&")


# ===========================================================================
# 2. Sidebar width policy + label elision
# ===========================================================================

def _fake_apps(names):
    return [(f"k{i}", n, f"desc {i}", SECTION_CORE)
            for i, n in enumerate(names)]


#: The two zooms the width policy has to hold at: 100 %, which is what this
#: file used to assume without saying so, and 150 %, which is what the app has
#: SHIPPED since ``DEFAULT_FONT_SCALE`` was raised in b530f70a (2026-08-03) —
#: "spaCR was laid out on a 1080p display and reads small on the 4K panels it
#: is used on".
SIDEBAR_SCALES = (1.0, 1.5)


@pytest.fixture
def at_font_scale(qapp, qt_theme_applied):
    """Put the font-scale PREFERENCE and the STYLESHEET on the same scale.

    These tests compare a bound computed from the preference
    (``scaled_px(WIDTH_MIN)``) against a width driven by the size hint of
    text the application stylesheet draws. The two agree only if the sheet
    was built at the scale the preference reports, and nothing in the
    harness guarantees that: ``qt_theme_applied`` builds the sheet with
    ``stylesheet()``, whose ``font_scale`` default is still 1.0, while the
    per-test QSettings sandbox is empty so ``get_font_scale()`` answers the
    shipped ``DEFAULT_FONT_SCALE`` of 1.5. Bounds at 150 %, glyphs at
    100 % — under which a policy that is correct at BOTH scales measures as
    broken at neither.

    So these tests state their scale rather than inherit it. Restores the
    shared application's 100 % sheet on the way out: leaving it at 150 %
    would move every later test that measures a pixel.
    """
    from spacr.qt import preferences
    from spacr.qt.theme import stylesheet

    def _apply(scale: float) -> float:
        preferences.set_font_scale(scale)
        qapp.setStyleSheet(stylesheet(font_scale=scale))
        qapp.processEvents()
        return scale

    yield _apply
    qapp.setStyleSheet(stylesheet())
    qapp.processEvents()


def _lay_out(widget):
    """Show and settle a widget so elision actually runs.

    Qt computes elision in ``resizeEvent``; a widget that was never shown
    reports nothing elided however narrow it is, which turns
    ``assert not clipped_items()`` into a tautology.
    """
    widget.show()
    QApplication.processEvents()
    widget.resize(widget.width(), widget.height())
    QApplication.processEvents()


@pytest.mark.parametrize("scale", SIDEBAR_SCALES)
def test_sidebar_column_never_narrows_below_its_floor(
        qtbot, at_font_scale, monkeypatch, scale):
    """Short names must not produce a stubby column — and the floor is the
    SCALED floor, so 150 % text does not sit in a 100 % column."""
    from spacr.qt.preferences import scaled_px
    at_font_scale(scale)
    monkeypatch.setattr(app_mod, "APPS", _fake_apps(["A", "B", "C"]))
    bar = Sidebar()
    qtbot.addWidget(bar)
    assert bar.width() == scaled_px(Sidebar.WIDTH_MIN)
    # SHOW IT FIRST. `clipped_items()` reads `_elided`, which is only ever
    # set in `resizeEvent`, so this assertion on an unshown widget can never
    # fail -- proven by mutation: a column 52 px too narrow for its own text
    # still passed. Showing it is what makes the claim real.
    _lay_out(bar)
    assert not bar.clipped_items()


@pytest.mark.parametrize("scale", SIDEBAR_SCALES)
def test_sidebar_column_widens_for_a_longer_name(
        qtbot, at_font_scale, monkeypatch, scale):
    """A name the floor cannot hold widens the column — off the floor, under
    the cap, and far enough that nothing is cut.

    Both bounds track the zoom, so this is asserted at 100 % and at the
    shipped 150 %. Measured here, the column lands at 309 px (floor 220, cap
    320) and at 449 px (floor 330, cap 480) — off the floor and short of the
    cap at both, which is the whole claim.
    """
    from spacr.qt.preferences import scaled_px
    at_font_scale(scale)
    long_ish = "Cellpose Model Comparison Workbench"
    monkeypatch.setattr(app_mod, "APPS", _fake_apps(["A", long_ish]))
    bar = Sidebar()
    qtbot.addWidget(bar)
    assert scaled_px(Sidebar.WIDTH_MIN) < bar.width() <= scaled_px(
        Sidebar.WIDTH_MAX)
    # Widening that still clips the name it widened for is not widening --
    # but only a LAID-OUT widget can report clipping. See the note in
    # test_sidebar_column_never_narrows_below_its_floor.
    _lay_out(bar)
    assert not bar.clipped_items()


@pytest.mark.parametrize("scale", SIDEBAR_SCALES)
def test_sidebar_caps_its_width_and_elides_a_pathological_name(
        qtbot, qapp, at_font_scale, monkeypatch, scale):
    """A name no column could hold elides — with the full name on hover —
    instead of pushing the sidebar across the window.

    The cap is ``scaled_px(WIDTH_MAX)``, not ``WIDTH_MAX``: at the shipped
    150 % zoom the column is allowed 480 px, and pinning the raw 320 would
    demand a column too narrow for its own text.
    """
    from spacr.qt.preferences import scaled_px
    at_font_scale(scale)
    huge = "An Extraordinarily Long Hypothetical Module Name For Testing"
    monkeypatch.setattr(app_mod, "APPS", _fake_apps([huge]))
    bar = Sidebar()
    qtbot.addWidget(bar)
    bar.resize(bar.width(), 400)
    bar.show()
    qapp.processEvents()

    assert bar.width() == scaled_px(Sidebar.WIDTH_MAX)
    clipped = bar.clipped_items()
    assert [b.full_text().strip() for b in clipped] == [huge]
    assert clipped[0].text() != huge and "…" in clipped[0].text()
    # THE FULL NAME IS IN THE ACCESSIBLE NAME, not in a tooltip. The row's
    # popup came off on 2026-09-03, so what carries a name too long to draw
    # is the text a screen reader reads -- and it is the same string, not a
    # truncation of it.
    assert clipped[0].accessibleName().strip() == huge


def test_refresh_icons_leaves_a_row_with_no_nav_key_alone(
        qtbot, qt_theme_applied):
    from spacr.qt.widgets.eliding import ElidingPushButton
    bar = Sidebar()
    qtbot.addWidget(bar)
    stray = ElidingPushButton("not a nav row", bar)
    bar._items.append(stray)
    bar.refresh_icons()
    assert stray.icon().isNull(), "a row with no navKey has no app to look up"
    assert not bar.findChildren(QPushButton)[0].icon().isNull()


# ===========================================================================
# 3. Home tiles: the clipped-label bug that motivated the regrouping
# ===========================================================================

@contextmanager
def _at_design_scale():
    """Keep the page and its pixel contract at the same explicit scale."""
    from spacr.qt import gui_scale, preferences

    settings = preferences._settings()
    keys = ("prefs/font_scale", "prefs/gui_scale")
    previous = {key: settings.value(key) for key in keys
                if settings.contains(key)}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(gui_scale, "_SCALE", 1.0)
        try:
            preferences.set_font_scale(1.0)
            preferences.set_gui_scale(1.0)
            yield
        finally:
            for key in keys:
                if key in previous:
                    settings.setValue(key, previous[key])
                else:
                    settings.remove(key)


@pytest.fixture
def _home_geometry_at_design_scale():
    with _at_design_scale():
        yield


@pytest.fixture(scope="module")
def home_page(qapp, qt_theme_applied, tmp_path_factory, request):
    # Alpha tiles are off for a new user (DEFAULT_SHOW_ALPHA is False) and
    # this page is built at module scope, before conftest's per-test
    # fixture raises the fallback, so raise it here for the page's life.
    from spacr.qt import preferences as _prefs
    from PySide6.QtCore import QSettings
    source = _prefs._settings()
    path = str(tmp_path_factory.mktemp("classic-home") / "preferences.ini")
    private = QSettings(path, QSettings.IniFormat)
    for key in source.allKeys():
        private.setValue(key, source.value(key))
    private.sync()
    patch = pytest.MonkeyPatch()
    request.addfinalizer(patch.undo)
    patch.setattr(_prefs, "_settings",
                  lambda: QSettings(path, QSettings.IniFormat))
    _prefs.set_easy_mode(False)
    patch.setattr(_prefs, "DEFAULT_SHOW_ALPHA", True)
    with _at_design_scale():
        page = make_home_page()  # the page MainWindow ships
        page.resize(1500, 950)
        page.show()
        qapp.processEvents()
    yield page
    page.hide()
    page.deleteLater()
    qapp.processEvents()


def test_home_renders_one_tile_per_app_under_every_section_heading(home_page):
    # TILED apps -- a folded module draws no tile by design.
    assert set(_tiles(home_page)) == {n for _k, n, *_r in tiled_apps()}
    headings = {lbl.text() for lbl in home_page.findChildren(QLabel)}
    for section in SECTIONS:
        assert section.upper() in headings


def test_no_home_tile_label_is_silently_clipped(home_page):
    """Either the name fits the label drawing it, or it elides AND the
    tooltip carries the whole name. Silent truncation is the bug."""
    bad = []
    for name, tile in _tiles(home_page).items():
        label = tile.name_label
        needed = QFontMetrics(label.font()).horizontalAdvance(name)
        if needed <= label.available_text_width():
            if label.is_elided():
                bad.append(f"{name}: elided although it fits")
            continue
        if not label.is_elided():
            bad.append(f"{name}: needs {needed}px, has "
                       f"{label.available_text_width()}px, not elided")
        elif label.toolTip() != name:
            bad.append(f"{name}: elided without a tooltip")
    assert not bad, bad


def test_todays_names_all_fit_without_eliding(home_page):
    assert [t.text_label for t in home_page.findChildren(AppTile)
            if t.is_name_elided()] == []


@pytest.mark.usefixtures("_home_geometry_at_design_scale")
def test_a_tile_is_wide_enough_for_the_longest_name(home_page):
    """Was ``test_a_tile_is_as_wide_as_the_name_it_has_to_draw``.

    That version asked ``HTile.required_width()`` — a per-tile
    measurement from when each section was a horizontal scroller and
    every tile sized itself to its own name. Every tile is the same size
    now, so the contract is that the ONE size fits the longest name; a
    tile that were wider than its neighbour would be the bug.

    Which tile that is comes from the page rather than being named here:
    it was "Annotator Agreement" until that module folded onto Annotate
    and lost its tile, and a hard-coded name turns "the longest tile is
    now a different one" into a KeyError that says nothing about width.
    """
    from spacr.qt.preferences import scaled_px
    tiles = _tiles(home_page)
    assert tiles, "the home page drew no tiles"
    name = max(tiles, key=len)
    longest = tiles[name]
    label = longest.name_label
    assert QFontMetrics(label.font()).horizontalAdvance(
        name) <= label.available_text_width()
    assert not label.is_elided()
    assert longest.width() <= scaled_px(HomePage.TILE_MAX_W)


def test_an_unfittable_name_elides_and_keeps_the_row_sane(
        qtbot, qt_theme_applied):
    from spacr.qt.preferences import scaled_px
    long_name = "Extremely Long Hypothetical Module Name For Testing"
    page = HomePage([("mask", long_name, "d", SECTION_CORE)],
                    _icon_for_app)
    qtbot.addWidget(page)
    page.resize(1400, 600)
    page.show()
    qtbot.waitExposed(page)

    # The tile on the CURRENT tab: every app is drawn once on Home and
    # once on its category tab, and the one on the hidden tab has never
    # been laid out, so it has nothing to elide against.
    tile = next(t for t in page.findChildren(AppTile) if t.isVisible())
    label = tile.name_label
    assert label.is_elided()
    assert "…" in label.text() and label.text() != long_name
    assert label.full_text() == long_name == label.toolTip()
    assert tile.width() <= scaled_px(HomePage.TILE_MAX_W)


# ===========================================================================
# 4. Icons
# ===========================================================================

def test_icon_overrides_all_point_at_files_that_exist():
    here = os.path.dirname(os.path.abspath(app_mod.__file__))
    icons = os.path.normpath(os.path.join(here, "..", "resources", "icons"))
    missing = {k: v for k, v in _ICON_OVERRIDES.items()
               if not os.path.isfile(os.path.join(icons, v))}
    assert not missing, f"override points at a missing file: {missing}"


@pytest.mark.parametrize("key", ["agreement", "plate_view", "model_compare",
                                 "analyze_plaques", "model_zoo"])
def test_a_module_with_its_own_art_no_longer_borrows(key):
    """The other half of the change above, asserted rather than assumed.

    Each of these has a `<key>.png` and must now draw it. Without this a
    later "tidy-up" could put the override back and nothing would notice:
    the borrow renders a perfectly good icon, just the wrong one.
    """
    from spacr.qt import iconset

    assert key not in _ICON_OVERRIDES, (
        f"{key} has artwork of its own; an override would hide it")
    own = iconset.bundled_icon_path(key)
    assert own is not None and own.endswith(f"{key}.png")


def test_forced_glyph_keys_ignore_the_bundled_pngs(qapp, monkeypatch):
    """The set is empty, and the mechanism still works when it is not.

    It used to assert ``_FORCE_GLYPH == {"align"}``: no bundled artwork
    said "tiles registered into ONE canvas", so Align & Stitch drew a
    qtawesome glyph. The user then picked ``cellpose_all_01`` for it,
    which is that call overruled by the person whose app it is, and
    ``align.png`` is installed. ``invasion`` had left the set earlier
    for the same reason.

    The loop below is what actually matters and is kept rather than
    deleted with the last entry — it is the contract any future member
    has to satisfy, exercised here against a key put in temporarily."""
    from spacr.qt import iconset
    assert _FORCE_GLYPH == set()

    # `mask` has a bundled PNG and no override, so "the glyph won" is a
    # real observation about it rather than a fallback that would have
    # happened anyway.
    assert _img(_icon_for_app("mask")) != _img(iconset.icon("mask"))
    monkeypatch.setattr(app_mod, "_FORCE_GLYPH", {"mask"})
    assert _img(_icon_for_app("mask")) == _img(iconset.icon("mask")), (
        "a key in _FORCE_GLYPH still drew its PNG")

    # …and nothing in the set may also carry an override: the two
    # mechanisms would silently disagree about which picture wins.
    assert not (set(app_mod._FORCE_GLYPH) & set(_ICON_OVERRIDES))


def test_an_unknown_key_falls_back_to_a_themed_glyph():
    from spacr.qt import iconset
    icon = _icon_for_app("no_such_app_key_at_all")
    assert isinstance(icon, QIcon) and not icon.isNull()
    assert _img(icon) == _img(iconset.icon("no_such_app_key_at_all"))


# ===========================================================================
# 5. Navigation
# ===========================================================================

# -- the search / filter path ------------------------------------------------

def test_the_command_palette_filters_by_name_and_navigates(win, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(win)
    qtbot.addWidget(palette)

    # Was "annotator agree" until Annotator Agreement folded onto Annotate
    # and lost its row; the palette lists tiles, so the filter has to name
    # one that still is one. Two words, one app, and nothing else close.
    palette._on_filter("plate viewer")
    rows = [palette._list.item(i).text()
            for i in range(palette._list.count())]
    assert any("Plate Viewer" in r for r in rows)
    assert not any("Mask" in r for r in rows), (
        f"filter let unrelated commands through: {rows}")

    palette._on_activate()
    assert win._status_app_label.text() == "Plate Viewer"
    assert win._stack.currentWidget() is win._screens["plate_view"]


def test_the_command_palette_filters_by_section_name(win, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    palette._on_filter(SECTION_ASSAYS)
    rows = [palette._list.item(i).text()
            for i in range(palette._list.count())]
    from spacr.qt.app import visible_apps

    # Only apps the user can see are offered; 634's alpha organism pages
    # (eb2201537) are hidden while alpha features are off.
    for name in (n for _k, n, _d, s in visible_apps() if s == SECTION_ASSAYS):
        assert any(name in r for r in rows), f"{name} not found by section"


def test_an_empty_filter_restores_every_command(win, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    total = palette._list.count()
    palette._on_filter("zzzz-no-such-command")
    assert palette._list.count() == 0
    palette._on_filter("")
    assert palette._list.count() == total


# ===========================================================================
# 6. Menu actions: about / urls / logs / preferences
# ===========================================================================

def test_resolve_version_reports_the_package_version(win):
    import spacr
    assert win._resolve_version() == spacr.__version__


def test_resolve_version_falls_back_to_dev(win, monkeypatch):
    import spacr
    monkeypatch.setattr(spacr, "__version__", "")
    assert win._resolve_version() == "dev"
    monkeypatch.setitem(sys.modules, "spacr", None)
    assert win._resolve_version() == "dev"
    # Put spacr back before qtbot closes the window: closing imports from it.
    monkeypatch.setitem(sys.modules, "spacr", spacr)


def test_help_menu_urls_open_in_a_browser(win, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    wanted = {"Tutorial", "Documentation"}
    for top in win.menuBar().actions():
        if top.text().replace("&", "") != "Help":
            continue
        for act in top.menu().actions():
            if act.text().replace("&", "") in wanted:
                act.trigger()
        break
    # Asserted against the module's own constants rather than literals. This
    # used to hard-code the singular `/spacr/tutorial/`, which is a 404 that no
    # page has ever been served from -- so the test pinned the broken link in
    # place and went red when the link was finally fixed. The published library
    # is at `/tutorials/` (plural), because `docs/source/conf.py` copies
    # `_extra/tutorials/` into the site root via `html_extra_path`.
    from spacr.qt.app import DOCS_URL, TUTORIALS_URL

    assert opened == [TUTORIALS_URL, DOCS_URL]
    # Keep the plural pinned: it is the whole point of the fix.
    assert TUTORIALS_URL.endswith("/tutorials/")


def test_a_failing_browser_open_reports_in_the_status_bar(win, monkeypatch):
    import webbrowser

    def _boom(url):
        raise RuntimeError("no browser")

    monkeypatch.setattr(webbrowser, "open", _boom)
    win._open_url("https://example.invalid/x")
    assert win.statusBar().currentMessage() == (
        "Failed to open https://example.invalid/x: no browser")


def test_open_log_folder_points_at_the_real_log_directory(win, monkeypatch):
    import webbrowser

    from spacr.qt.verbose_logger import log_dir
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    win._open_log_folder()
    assert opened == [f"file://{log_dir()}"]


def test_open_log_folder_reports_failures(win, monkeypatch):
    import webbrowser

    def _boom(url):
        raise OSError("nope")

    monkeypatch.setattr(webbrowser, "open", _boom)
    win._open_log_folder()
    assert win.statusBar().currentMessage() == (
        "Failed to open log folder: nope")


# ===========================================================================
# 7. Demos
# ===========================================================================

def test_apply_demo_to_a_screen_that_supports_nothing_is_a_no_op(
        win, tmp_path, monkeypatch):
    """None of the three hooks -> nothing happens, and in particular the
    demo's settings CSV is never even read.

    The second half is the control: the identical call against a screen
    that *does* take settings reads the CSV and receives it, so "the CSV
    was never read" is a measurement that can come out either way."""
    import spacr.utils as sutils
    layout = win._run_demo_generator("mask", str(tmp_path))
    read: list = []
    real_load = sutils.load_settings

    def _spy(*a, **k):
        read.append(a[0] if a else None)
        return real_load(*a, **k)
    monkeypatch.setattr(sutils, "load_settings", _spy)


    bare = types.SimpleNamespace()
    win._apply_demo_to_screen(bare, layout)
    assert read == [], (
        f"the demo CSV was read for a screen that cannot take it: {read}")
    assert vars(bare) == {}, "something was pushed onto the screen anyway"

    # Control — same layout, same call, a screen that can take settings.
    got: list = []

    class _Settings:
        def apply_settings_dict(self, settings):
            got.append(settings)

    win._apply_demo_to_screen(_Settings(), layout)
    assert read == [str(layout.settings_csv)]
    assert got and isinstance(got[0], dict) and got[0]


# ===========================================================================
# 8. The end-to-end demo chain
# ===========================================================================

# ===========================================================================
# 9. Update check
# ===========================================================================

def _update_info(installed="1.0.0", latest="2.0.0", error=None):
    from spacr.updater import UpdateInfo
    return UpdateInfo(installed_version=installed, latest_release=latest,
                      nightly_sha=None, error=error)


def _run_update_check(win, qtbot, modals):
    win._check_for_updates()
    worker = win._update_worker
    qtbot.waitUntil(
        lambda: bool(modals.information or modals.warning), timeout=5000)

    def _done():
        try:
            return worker.isFinished()
        except RuntimeError:      # already deleteLater()'d — it finished
            return True

    qtbot.waitUntil(_done, timeout=5000)


def test_an_unreachable_update_server_is_reported(win, qtbot, modals,
                                                  monkeypatch):
    import spacr.updater as updater
    monkeypatch.setattr(updater, "check_for_updates",
                        lambda: _update_info(latest=None, error="timed out"))
    _run_update_check(win, qtbot, modals)
    assert modals.warning[0][0] == "Updates"
    assert "timed out" in modals.warning[0][1]
    assert not modals.information


def test_the_result_is_delivered_on_the_gui_thread(win, qtbot, modals,
                                                   monkeypatch):
    """The worker runs the network call off-thread, but the dialog it
    ends in must be raised on the GUI thread — building a QWidget from a
    QThread is how this app used to abort on quit."""
    import spacr.updater as updater
    monkeypatch.setattr(updater, "check_for_updates",
                        lambda: _update_info(installed="9.9.9",
                                             latest="9.9.9"))
    _run_update_check(win, qtbot, modals)
    assert modals.threads == [threading.main_thread()]


def test_being_up_to_date_says_so(win, qtbot, modals, monkeypatch):
    import spacr.updater as updater
    monkeypatch.setattr(updater, "check_for_updates",
                        lambda: _update_info(installed="9.9.9",
                                             latest="9.9.9"))
    _run_update_check(win, qtbot, modals)
    assert modals.information[0][1] == "You're on 9.9.9. No updates."
    assert not modals.warning


def test_accepting_an_upgrade_waits_for_authenticated_shutdown_handoff(
        win, qtbot, modals, monkeypatch, tmp_path):
    from spacr import install_cleanup, restart_state
    from spacr.qt.bridge import registry
    import spacr.updater as updater

    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(updater, "check_for_updates", _update_info)
    monkeypatch.setattr(restart_state, "state_path", lambda: tmp_path / "restart.json")
    calls = []
    events = []
    ready = [False]
    plan = {"adapter": "pip-private-posix-v1", "command": ["inert-helper"],
            "version": "2.0.0", "handshake": {"token": "test-only-token"}}

    def prepare(records, version, *, ticked):
        calls.append((records, version, ticked))
        return plan

    class Handshake:
        """Stand in for the authenticated helper; execute no installer code."""

        def __init__(self, received):
            assert received is plan
            events.append("prepared")

        def status(self):
            return {"state": "ready"} if ready[0] else None

        def check(self):
            events.append("checked")

        def approve(self):
            assert win._frozen_update_closing
            assert restart_state.peek()["module"] == "__home__"
            events.append("approved")

        def cancel(self):
            events.append("cancelled")

    monkeypatch.setattr(install_cleanup, "start_update_helper", prepare)
    monkeypatch.setattr(install_cleanup, "_FrozenUpdateHandshake", Handshake)
    monkeypatch.setattr(updater, "run_pip_upgrade", lambda **kwargs: pytest.fail(
        "The live GUI must delegate replacement instead of upgrading itself"))
    monkeypatch.setattr(win, "_restart_after_package_upgrade", lambda: pytest.fail(
        "The delegated helper owns restart after the GUI closes"))
    modals.answers = [QMessageBox.Yes]
    win.show()
    qtbot.waitUntil(lambda: not registry().is_busy(), timeout=5000)
    win._check_for_updates()
    qtbot.waitUntil(lambda: events == ["prepared"], timeout=5000)
    assert calls == [([], "2.0.0", ())]
    assert "1.0.0" in modals.questions[0][1]
    assert "2.0.0" in modals.questions[0][1]
    assert win.isVisible() and not win._closing
    assert win._frozen_update_plan is plan
    assert win._frozen_update_timer.isActive()
    assert not (tmp_path / "restart.json").exists()
    ready[0] = True
    restart_prompts = []

    def confirm_restart(message):
        restart_prompts.append(message)
        return QMessageBox.Ok

    monkeypatch.setattr(win, "_confirm_update_restart", confirm_restart)
    win._poll_frozen_update()
    assert len(restart_prompts) == 1 and "2.0.0" in restart_prompts[0]
    assert events == ["prepared", "checked", "approved"]
    assert win._closing and not win.isVisible()
    assert win._frozen_update_handshake is None
    assert not win._frozen_update_timer.isActive()
    assert not modals.warning


def test_an_unavailable_update_plan_reports_its_error_without_closing(
        win, qtbot, modals, monkeypatch):
    from spacr import install_cleanup
    from spacr.qt.bridge import registry
    import spacr.updater as updater

    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(updater, "check_for_updates", _update_info)
    calls = []

    def prepare(records, version, *, ticked):
        calls.append((records, version, ticked))
        return {"command": None, "error": "fixture preparation refused"}

    monkeypatch.setattr(install_cleanup, "start_update_helper", prepare)
    monkeypatch.setattr(updater, "run_pip_upgrade", lambda **kwargs: pytest.fail(
        "A refused plan must not mutate the live environment"))
    modals.answers = [QMessageBox.Yes]
    win.show()
    qtbot.waitUntil(lambda: not registry().is_busy(), timeout=5000)
    _run_update_check(win, qtbot, modals)
    assert calls == [([], "2.0.0", ())]
    assert modals.warning == [("Updates", "Upgrade unavailable: fixture preparation refused")]
    assert win.isVisible() and not win._closing
    assert getattr(win, "_frozen_update_handshake", None) is None
    assert not modals.information


def test_declining_an_upgrade_does_nothing(win, qtbot, modals, monkeypatch):
    from spacr.qt.bridge import registry
    import spacr.updater as updater
    monkeypatch.setattr(updater, "editable_install_location", lambda: None)
    monkeypatch.setattr(updater, "check_for_updates", _update_info)

    def _must_not_run():
        raise AssertionError("pip must not run when the user declines")

    monkeypatch.setattr(updater, "run_pip_upgrade", _must_not_run)
    modals.answers = [QMessageBox.No]
    qtbot.waitUntil(lambda: not registry().is_busy(), timeout=5000)
    win._check_for_updates()
    worker = win._update_worker
    qtbot.waitUntil(lambda: bool(modals.questions), timeout=5000)

    def _done():
        try:
            return worker.isFinished()
        except RuntimeError:
            return True

    qtbot.waitUntil(_done, timeout=5000)
    assert not modals.information and not modals.warning


def test_a_missing_updater_module_is_reported_not_raised(win, modals,
                                                         monkeypatch):
    monkeypatch.setitem(sys.modules, "spacr.updater", None)
    win._check_for_updates()
    assert modals.warning[0][0] == "Updates"
    assert "Update check unavailable" in modals.warning[0][1]
    assert not hasattr(win, "_update_worker")


# ===========================================================================
# 10. Shutdown
# ===========================================================================

def test_closing_the_window_waits_for_a_running_update_check(
        win, qtbot, modals, monkeypatch):
    """Quitting mid-update destroyed a live QThread — the exact abort
    ``closeEvent`` drains the consoles to avoid."""
    import time

    import spacr.updater as updater

    def _slow_check():
        time.sleep(0.4)
        return _update_info(installed="9.9.9", latest="9.9.9")

    monkeypatch.setattr(updater, "check_for_updates", _slow_check)
    win._check_for_updates()
    worker = win._update_worker
    assert not worker.isFinished(), "the check should still be in flight"

    win.close()
    assert worker.isFinished(), (
        "close() returned while the update QThread was still running")


def test_a_news_fetch_that_outlasts_the_close_is_detached_not_destroyed(
        win, monkeypatch):
    """A running QThread destroyed with its parent window aborts the process.

    Coverage shard 0 of CI run 36020180024 lost three workers to SIGABRT
    ("QThread: Destroyed while thread '' is still running") this way: Home
    asked for a GitHub release fetch, the window closed, and the next test's
    deferred-delete flush destroyed the window with the fetch still running.
    The socket timeout does not bound name resolution, so the close wait can
    run out; what outlives it has to leave the window before the window goes.
    Shutdown now also refuses destruction until that parked fetch finishes,
    since detaching it alone does not make interpreter exit safe.
    """
    import threading

    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    import spacr.updater as updater
    from spacr.qt import bridge

    release = threading.Event()

    def _unanswered_fetch():
        release.wait(10)
        return []

    monkeypatch.delenv("SPACR_NO_NETWORK", raising=False)
    monkeypatch.setattr(updater, "fetch_release_notes", _unanswered_fetch)
    monkeypatch.setattr(MainWindow, "_UPDATE_WORKER_WAIT_MS", 50)
    retire = bridge.retire_live_work
    monkeypatch.setattr(bridge, 'retire_live_work',
                        lambda timeout_ms: retire(min(timeout_ms, 50)))
    warnings = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: warnings.append(args[2]))
    win._refresh_news()
    worker = win._news_worker
    assert worker.isRunning(), "the fetch should still be in flight"

    try:
        assert win.close() is False
        assert not win._closing
        assert warnings and 'No worker was force-terminated' in warnings[0]
        assert worker.parent() is None, (
            "a fetch still running at close stayed a child of the window, "
            "so deleting the window destroys a running QThread")
        assert any(thread is worker
                   for thread, _ in list(bridge._PARKED_THREADS))
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        assert worker.isRunning(), "the fetch survives the refused close"
    finally:
        release.set()
    assert worker.wait(10_000)
    bridge.prune_parked_threads()
    assert not any(thread is worker
                   for thread, _ in list(bridge._PARKED_THREADS))
    assert win.close() is True
    assert win._closing
    win.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)


def test_a_news_refresh_asked_for_after_close_starts_no_thread(
        win, monkeypatch):
    """Home asks for the news one event-loop turn after it is shown.

    A window closed before that turn used to start the fetch anyway, after
    its close had already waited for the workers it knew about.
    """
    import spacr.updater as updater

    fetched = []
    monkeypatch.setattr(updater, "fetch_release_notes",
                        lambda: fetched.append(True) or [])
    win.close()
    win._refresh_news()
    assert getattr(win, "_news_worker", None) is None
    assert fetched == []


# ===========================================================================
# 11. Cross-screen hand-offs
# ===========================================================================

def test_snapshot_returns_the_settings_of_the_visible_app_screen(win):
    win._on_nav_selected("measure")
    key, settings = win._snapshot_current_screen_settings()
    assert key == "measure"
    assert isinstance(settings, dict) and settings
    assert settings == win._screens["measure"]._settings_model.collect()


def test_snapshot_raises_when_no_app_has_been_opened(win):
    win._on_nav_selected("db_browser")
    with pytest.raises(RuntimeError, match="No active plate settings"):
        win._snapshot_current_screen_settings()


def test_the_legacy_explain_error_hook_is_inert(win):
    assert win._on_explain_error("Traceback…", "mask") is None


# ===========================================================================
# 12. Train hand-off + seed values
# ===========================================================================

# The Train buttons emit ``classify``, and every run journal ever written
# names the key that ran. Neither is a screen any more: Classify (CV) and
# Classify (ML) became one merged Classify, so the hand-off is resolved
# through ``chaining._SUCCEEDED_BY`` before it navigates. It used to
# navigate to the raw key, which BUILT a screen for it -- a page with no
# sidebar row, no tile and no way back to it -- and seeded that instead of
# the screen the user can actually reach.

def test_seed_values_are_applied_per_widget_type(qtbot):
    apply = MainWindow._apply_seed_value

    box = QCheckBox()
    qtbot.addWidget(box)
    apply(box, 1)
    assert box.isChecked() is True
    apply(box, 0)
    assert box.isChecked() is False

    spin = QSpinBox()
    qtbot.addWidget(spin)
    spin.setRange(0, 100)
    apply(spin, "42.9")
    assert spin.value() == 42

    dspin = QDoubleSpinBox()
    qtbot.addWidget(dspin)
    dspin.setRange(0, 100)
    apply(dspin, "3.5")
    assert dspin.value() == pytest.approx(3.5)

    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Alpha", "a")
    combo.addItem("Beta", "b")
    apply(combo, "b")
    assert combo.currentIndex() == 1
    apply(combo, "Alpha")
    assert combo.currentIndex() == 0

    edit = QLineEdit()
    qtbot.addWidget(edit)
    apply(edit, None)
    assert edit.text() == ""
    apply(edit, 12)
    assert edit.text() == "12"

    label = QLabel("untouched")
    qtbot.addWidget(label)
    apply(label, "ignored")           # unknown widget type: no-op
    assert label.text() == "untouched"


def test_a_combo_seed_that_matches_nothing_leaves_the_index_alone(qtbot):
    combo = QComboBox()
    qtbot.addWidget(combo)
    combo.addItem("Alpha", "a")
    combo.addItem("Beta", "b")
    combo.setCurrentIndex(1)
    MainWindow._apply_seed_value(combo, "gamma")
    assert combo.currentIndex() == 1


# ===========================================================================
# 13. Pipeline preloader
# ===========================================================================

def test_every_preloaded_module_actually_exists():
    import importlib.util
    missing = [m for m in _PipelinePreloader._MODULES
               if importlib.util.find_spec(m) is None]
    assert not missing, f"preloader points at modules that don't exist: {missing}"


@pytest.mark.parametrize("navigator", (False, True))
def test_the_window_still_opens_without_shortcuts_or_the_tour(
        qtbot, qt_theme_applied, monkeypatch, tmp_path, navigator):
    """Refused optional systems leave only the current Home's own bindings."""
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtGui import QKeySequence
    from spacr.qt import preferences
    import spacr.qt

    source = preferences._settings()
    path = str(tmp_path / "optional-shortcuts.ini")
    private = QSettings(path, QSettings.IniFormat)
    for key in source.allKeys():
        private.setValue(key, source.value(key))
    private.sync()
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    preferences.set_easy_mode(navigator)
    monkeypatch.setitem(sys.modules, "spacr.qt.shortcuts", None)
    monkeypatch.setitem(sys.modules, "spacr.qt.first_run", None)
    monkeypatch.delattr(spacr.qt, "shortcuts", raising=False)
    monkeypatch.delattr(spacr.qt, "first_run", raising=False)
    w = MainWindow()
    qtbot.addWidget(w, before_close_func=_close_owned_screens)
    assert w._stack.currentWidget() is w._startup
    from PySide6.QtGui import QShortcut
    if navigator:
        escape = w._startup._workflow_escape
        assert escape.key() == QKeySequence(Qt.Key_Escape)
        assert escape.parent() is w._startup
        assert w.findChildren(QShortcut) == [escape]
    else:
        assert not w.findChildren(QShortcut), (
            "shortcuts should be absent when the module failed to import")


def test_the_main_window_does_not_preload_pipelines_by_default(win):
    # The default is intentionally loaded-on-call: eager startup spent twenty
    # seconds importing torch/compiler/distributed stacks on the maintainer's
    # machine. The eager path remains covered in
    # test_libraries_load_when_called.py.
    assert win._preloader is None
    assert win._loading_screen is None


# ===========================================================================
# 14. Bundled fonts
# ===========================================================================

# ===========================================================================
# 15. launch()
# ===========================================================================

class _FakeSignal:
    def __init__(self):
        self.callbacks: list = []

    def connect(self, cb):
        self.callbacks.append(cb)


class _AppShim(QObject):
    """Stands in for the QApplication ``launch`` would construct.

    Delegates everything it doesn't override to the real, already-running
    QApplication so ``apply_preferences_to_app`` still does its work —
    except ``setStyleSheet`` (a global re-polish would touch every widget
    other tests left behind) and ``exec`` (which would block forever).

    A REAL QObject, not a bare class. `launch` installs
    `_DialogTranslationFilter(app)`, which passes the application as the
    filter's Qt PARENT, and PySide6 refuses a parent that is not a QObject:
    "QObject.__init__ called with wrong argument types: _AppShim". A plain
    duck-typed double cannot stand in for something the C++ side takes
    ownership through, so it inherits instead of pretending.

    `__getattr__` still forwards everything QObject does not already define,
    which is what keeps the delegation working.
    """

    def __init__(self, real, argv):
        super().__init__()
        self._real = real
        self.argv = list(argv)
        self.aboutToQuit = _FakeSignal()
        self.app_name = None
        self.org_name = None
        self.stylesheet = None
        self.exec_calls = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def setApplicationName(self, name):
        self.app_name = name

    def setOrganizationName(self, name):
        self.org_name = name

    def setStyleSheet(self, qss):
        self.stylesheet = qss

    def exec(self):
        self.exec_calls += 1
        return 0


class _ThreadShim:
    instances: list = []

    def __init__(self, target=None, name=None, daemon=None):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False
        _ThreadShim.instances.append(self)

    def start(self):
        self.started = True


@pytest.fixture
def launched(qapp, qtbot, monkeypatch, tmp_path):
    """Run the real :func:`launch` against a stand-in QApplication."""
    monkeypatch.setenv("SPACR_LOG_DIR", str(tmp_path / "logs"))
    made: list = []

    # THE STUB HAS TO CARRY `setAttribute`, and carrying it is not a
    # formality. `launch` sets AA_DontUseNativeDialogs BEFORE constructing
    # the QApplication, because Qt ignores that attribute afterwards --
    # silently, which looks exactly like it worked. A bare function has no
    # such attribute, so the call raised and every test through this fixture
    # failed on `'function' object has no attribute 'setAttribute'`.
    #
    # Recorded rather than swallowed, so the ordering can be asserted: the
    # attribute must be set while nothing has been constructed yet.
    attributes: list = []

    def _factory(argv):
        shim = _AppShim(qapp, argv)
        made.append(shim)
        return shim

    def _set_attribute(*args):
        attributes.append((args, len(made)))

    _factory.setAttribute = _set_attribute
    _factory.instance = lambda: made[-1] if made else qapp

    _ThreadShim.instances = []
    monkeypatch.setattr(app_mod, "QApplication", _factory)
    # ``app_mod`` also uses Event for the optional pipeline preloader. A
    # Thread-only namespace made every MainWindow constructor fail before the
    # launch test reached the pre-warm it meant to record. Keep the real
    # threading surface and replace only Thread on an isolated proxy.
    threading_proxy = types.SimpleNamespace(
        **{name: getattr(threading, name) for name in dir(threading)}
    )
    threading_proxy.Thread = _ThreadShim
    monkeypatch.setattr(app_mod, "threading", threading_proxy)

    state = {"shims": made, "threads": _ThreadShim.instances,
             "attributes": attributes,
             "before": set(qapp.topLevelWidgets())}

    def _window():
        new = [w for w in qapp.topLevelWidgets()
               if w not in state["before"] and isinstance(w, MainWindow)]
        assert len(new) == 1, f"expected exactly one new window, got {new}"
        return new[0]

    state["window"] = _window
    try:
        yield state
    finally:
        # CLOSE WHAT `launch` OPENED. The real `exec()` returns only once the
        # window is closed; `_AppShim.exec` returns at once, so a test that
        # does not close the window by hand leaves it SHOWN, and a shown
        # window's backdrop keeps its `spacr-ambient-shade` thread running
        # for the rest of the worker. Every user in another module leaked
        # this way; 49a88481a put one in the same CI batch as
        # test_no_backdrop_thread_outlives_a_test.py, whose counts then read
        # 4 == 1 (three launches' threads plus its own) whenever xdist gave
        # both files one worker. Closed the way the tests that close by hand
        # do; hidden if the close is refused, since hiding is what stops the
        # backdrop.
        #
        # AND DELETED, the ones a test closed by hand included. `launch`
        # owns its window, not pytest-qt, and a MainWindow is not
        # WA_DeleteOnClose: closed, it was still a LIVE top-level window,
        # with its menus, dock and home page, for the rest of the process.
        # A serial `pytest tests/qt` carried nine of them by a third of the
        # way through, restyled on every theme change after that
        # (features/new/47).
        for window in [w for w in qapp.topLevelWidgets()
                       if w not in state["before"]
                       and isinstance(w, MainWindow)]:
            if window.isVisible():
                try:
                    _close_owned_screens(window)
                    window.close()
                except Exception:                            # noqa: BLE001
                    pass
                if window.isVisible():
                    window.hide()
            window.deleteLater()


#: What the background pre-warm is expected to import, written out rather
#: than read back out of the code it guards.
#:
#: The pre-warm exists for one moment: the first time the user opens a
#: module, the settings form for it is built, and building it reaches
#: ``spacr.settings`` -- about a second of import that would otherwise land
#: on the GUI thread exactly when the window is supposed to snap open.
#: Importing it on a daemon thread while the user is still looking at Home
#: is the whole feature, so what the list holds is the thing worth asserting.
#:
#: ``spacr.gui_utils`` was the other name here, and by far the heavier one:
#: it pulled torch and cv2 in behind it. The Qt settings form no longer
#: reaches it -- the one function it wanted is in ``spacr.settings_spec``,
#: which imports nothing -- and the module itself went with the rest of the
#: Tk interface. Asserting a deleted module is warmed cannot pass; asserting
#: it is *absent* would pass for the wrong reason, since a name nothing can
#: import can only ever be missing. So the list is asserted for what it
#: holds. Adding a name here is a claim that a screen pays for that import
#: on its first open; record it in this table in the same change.
#:
#: ``spacr.qt.screens.settings_model`` and ``spacr.qt.imagery`` were added
#: 2026-09-09, and the moment they are for is PREFERENCES rather than a
#: module screen. The first ``PreferencesDialog(...)`` of a session was
#: measured at 902 ms against 30 ms for the second -- all of the difference
#: import work, on the GUI thread, with the animated backdrop stopped for
#: the whole of it -- and settings_model was 474 ms of that 902, imagery 83.
#: Warmed off-thread the first open costs 139 ms. Nobody can reach
#: Preferences before the home screen exists, so this thread always wins
#: the race it has to win.
PREWARMED_MODULES = ("spacr.settings", "spacr.qt.screens.settings_model",
                     "spacr.qt.imagery", "spacr.settings_advisor",
                     "pandas", "scipy.linalg",
                     "scipy.stats", "matplotlib.figure",
                     "matplotlib.font_manager", "skimage.data",
                     "skimage.draw", "skimage.filters", "skimage.measure",
                     "skimage.morphology")


def _prewarmed_module_names():
    """The module names ``launch``'s background pre-warm imports.

    Read out of the prewarm helper so that the two tests below compare
    the code against :data:`PREWARMED_MODULES` rather than against each
    other.
    """
    import inspect
    import re

    source = inspect.getsource(app_mod._start_settings_prewarm)
    match = re.search(r"for mod in \(([^)]*)\):", source)
    assert match, "launch no longer pre-warms a tuple of module names"
    return [name.strip().strip("\"'")
            for name in match.group(1).split(",") if name.strip()]


def test_the_prewarm_list_holds_what_a_first_module_open_pays_for():
    """The list is the feature; a silent change to it is a silent regression.

    Dropping a name here costs the user a frozen window on the first module
    open, and nothing anywhere says so -- the pre-warm swallows its failures
    and a warm that never happened looks exactly like one that did.
    """
    assert _prewarmed_module_names() == list(PREWARMED_MODULES), (
        "the pre-warm list moved. That is allowed -- update "
        "PREWARMED_MODULES in the same change, so what a first module open "
        "is expected to pay for stays written down")


def test_the_prewarm_list_names_modules_that_exist():
    """The pre-warm swallows every failure, because it is an optimisation and
    a slow first open is better than a crash on start. That silence has to be
    paid for here: a name that no longer imports warms nothing, and the module
    screen goes back to freezing on its first import with nothing said.
    """
    import importlib.util

    names = _prewarmed_module_names()
    assert names, "the pre-warm list is empty; nothing is warmed"
    missing = [name for name in names
               if importlib.util.find_spec(name) is None]
    assert not missing, (
        f"launch pre-warms {missing}, which no longer exist -- the import "
        "fails into the debug log and the screen it was warming for is slow "
        "again")


def test_concurrent_home_news_requests_keep_bundled_rows_until_one_real_fetch_finishes(
        normal_win, qtbot, monkeypatch):
    from PySide6.QtTest import QSignalSpy
    from spacr import updater
    from spacr.qt import preferences
    from tests.qt.test_the_news_panel_shows_the_latest_release import NEWER

    window = normal_win
    page = window._startup
    assert isinstance(page, HomePage) and window._stack.currentWidget() is page
    automatic = QSignalSpy(page.news_refresh_requested)
    monkeypatch.setenv('SPACR_NO_NETWORK', '1')
    window.show()
    qtbot.waitUntil(lambda: automatic.count() == 1)
    assert getattr(window, '_news_worker', None) is None
    panel = page.news_panel
    bundled = panel.releases
    assert bundled and NEWER not in bundled
    entered, release = threading.Event(), threading.Event()
    called_from = []
    worker = None

    def fetch():
        called_from.append(threading.get_ident())
        entered.set()
        assert release.wait(10)
        return [dict(NEWER)]

    monkeypatch.setattr(updater, 'fetch_release_notes', fetch)
    preferences.set_refresh_news(True)
    monkeypatch.delenv('SPACR_NO_NETWORK', raising=False)
    try:
        page.news_refresh_requested.emit()
        qtbot.waitUntil(entered.is_set)
        worker = window._news_worker
        assert worker.isRunning()
        finished = QSignalSpy(worker.finished)
        page.news_refresh_requested.emit()
        assert window._news_worker is worker and worker.isRunning()
        assert len(called_from) == 1 and called_from[0] != threading.get_ident()
        assert panel.releases == bundled
        assert all(NEWER['name'] not in label.text() for label in panel.findChildren(QLabel))
        release.set()
        qtbot.waitUntil(lambda: panel.releases[0] == NEWER)
        qtbot.waitUntil(lambda: finished.count() == 1)
        assert len(called_from) == 1
        assert panel.releases[1:] == bundled
        labels = [label.text() for label in panel.findChildren(QLabel)]
        assert any(NEWER['name'] in text for text in labels)
        assert any(NEWER['url'] in text for text in labels)
        assert window._startup is page and window._stack.currentWidget() is page
    finally:
        release.set()
        if worker is None:
            worker = getattr(window, '_news_worker', None)
        if worker is not None:
            try:
                assert worker.wait(10000)
            except RuntimeError:
                pass
