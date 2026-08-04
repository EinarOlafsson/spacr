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

import importlib
import os
import sys
import threading
import types

import pytest

from PySide6.QtGui import QFontMetrics, QIcon
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QWidget,
)

from spacr.qt import app as app_mod
from spacr.qt.app import (
    APPS,
    MAX_APPS_PER_SECTION,
    SECTION_CORE,
    SECTION_DATA,
    SECTION_DESIGN,
    SECTION_EXPLORE,
    SECTION_MODELS,
    SECTION_RESULTS,
    SECTION_TOXO,
    SECTIONS,
    MainWindow,
    Sidebar,
    _FORCE_GLYPH,
    _ICON_OVERRIDES,
    _PipelinePreloader,
    _icon_for_app,
    _load_bundled_fonts,
    app_stage,
    home_bands,
    make_home_page,
    section_members,
)
from spacr.qt.widgets.home import AppTile, HomePage


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def win(qtbot, qt_theme_applied):
    """A live MainWindow, cleaned up by pytest-qt."""
    w = MainWindow()
    qtbot.addWidget(w)
    return w


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
    return rec


@pytest.fixture
def pick_dir(monkeypatch):
    """Script :meth:`QFileDialog.getExistingDirectory`.

    Returns a one-item list; set ``[0]`` to the directory the "user"
    picks, or to ``""`` to simulate Cancel. Every call is recorded in
    ``.calls``.
    """
    box = [""]
    calls: list = []

    def _get(parent, caption, directory="", options=None):
        calls.append((caption, directory))
        return box[0]

    monkeypatch.setattr(QFileDialog, "getExistingDirectory", _get)
    box.append(calls)          # box[1] is the call log
    return box


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
    "mask":            SECTION_CORE,
    "timelapse":       SECTION_CORE,
    "motility":        SECTION_CORE,
    "measure":         SECTION_CORE,
    "annotate":        SECTION_CORE,
    "classify":        SECTION_CORE,
    "ml_analyze":      SECTION_CORE,
    "map_barcodes":    SECTION_CORE,
    "regression":      SECTION_CORE,
    "align":           SECTION_DATA,
    "convert":         SECTION_DATA,
    "foreign":         SECTION_DATA,
    "external_masks":  SECTION_DATA,
    "queue":           SECTION_DATA,
    "batch":           SECTION_DATA,
    "distributed_jobs": SECTION_DATA,
    "db_browser":      SECTION_DATA,
    "make_masks":      SECTION_MODELS,
    "train_cellpose":  SECTION_MODELS,
    "cellpose_masks":  SECTION_MODELS,
    "model_compare":   SECTION_MODELS,
    "model_zoo":       SECTION_MODELS,
    # Illumination correction is done TO the images on the way in, like
    # stitching and conversion -- not a reading of what came out.
    "illumination":    SECTION_DATA,
    "data_manager":    SECTION_DATA,
    "plate_view":      SECTION_RESULTS,
    "agreement":       SECTION_RESULTS,
    "umap":            SECTION_RESULTS,
    "activation":      SECTION_RESULTS,
    "train_compare":   SECTION_RESULTS,
    "classifier_evaluation": SECTION_RESULTS,
    "run_history":     SECTION_RESULTS,
    "report":          SECTION_RESULTS,
    "barcode_qc":      SECTION_RESULTS,
    # Two runs of the same project read against each other is a reading of
    # what a finished run produced, which is what this section is.
    "run_compare":     SECTION_RESULTS,
    # Explore's first two. The section was declared and empty until they
    # registered -- "page through image layers" is the example in its own
    # definition, and the Graph Builder family is what it was named for.
    "layer_viewer":    SECTION_EXPLORE,
    "graph_builder":   SECTION_EXPLORE,
    # Explore's third, and it is Explore rather than Results & QC for the
    # same reason as the other two: handing measurements.db to scanpy or
    # scvi-tools is asking the numbers something spaCR did not plan for.
    "anndata_export":  SECTION_EXPLORE,
    "analyze_plaques": SECTION_TOXO,
    "recruitment":     SECTION_TOXO,
    "invasion":        SECTION_TOXO,
    "replication":     SECTION_TOXO,
    # Design's first and, for now, only app. The section has been declared
    # and empty since the sections were named; its note ("Plan the
    # experiment before it runs: power, sample size, plate layout,
    # controls and replicates") was written for this.
    "power":           SECTION_DESIGN,
}

#: How finished every app is, as a second ledger on the same app keys.
#: Absent from ``APP_STAGE`` means stable, so the two are checked
#: against each other rather than against a copy of the same dict.
EXPECTED_STAGES = {
    "align": "alpha", "model_zoo": "alpha", "convert": "alpha",
    "foreign": "alpha", "external_masks": "alpha",
    "model_compare": "alpha", "queue": "alpha",
    "batch": "alpha", "distributed_jobs": "alpha",
    "invasion": "alpha", "db_browser": "alpha",
    "plate_view": "alpha", "agreement": "alpha", "train_compare": "alpha",
    "classifier_evaluation": "alpha",
    "run_history": "alpha", "report": "alpha",
    "illumination": "alpha", "barcode_qc": "alpha",
    "layer_viewer": "alpha", "graph_builder": "alpha",
    "data_manager": "alpha",
    "power": "alpha", "anndata_export": "alpha", "run_compare": "alpha",
    "make_masks": "beta", "train_cellpose": "beta", "cellpose_masks": "beta",
    "timelapse": "beta", "motility": "beta", "analyze_plaques": "beta",
    "replication": "beta", "umap": "beta", "activation": "beta",
}


def test_every_app_is_filed_under_the_section_it_belongs_to():
    """Section assignment for every app, one entry at a time."""
    actual = {key: section for key, _n, _d, section in APPS}
    assert actual == EXPECTED_SECTIONS, (
        "an app changed section (or was added/removed). That is allowed — "
        "update EXPECTED_SECTIONS in the same commit so the move is "
        "recorded rather than accidental.")


def test_every_app_carries_the_maturity_it_was_given():
    """The other axis, one entry at a time.

    Twenty-five alpha, nine beta, eight stable -- Illumination, Barcode
    QC, Layer Viewer and Graph Builder each arrived alpha, and so do
    Power / Design, AnnData Export and Run Compare, which is what
    "built and reachable, not yet trusted end to end" means for a
    feature whose first users are reading this sentence. Signing an app
    off is deleting a line from ``APP_STAGE`` and from here; nothing
    else moves, which is the whole point of maturity not being a
    section."""
    actual = {key: app_stage(key) for key, *_r in APPS}
    expected = {key: EXPECTED_STAGES.get(key, "stable")
                for key, *_r in APPS}
    assert actual == expected, (
        "an app changed maturity. That is allowed — record it in "
        "EXPECTED_STAGES in the same commit.")
    counts = {s: sum(1 for v in actual.values() if v == s)
              for s in ("alpha", "beta", "stable")}
    assert counts == {"alpha": 25, "beta": 9, "stable": 8}


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
    """The cap was 9 and is now 13.

    Nine was the width of the Core pipeline and nothing more, so it
    would have fired on the next Core app rather than when a row stopped
    being readable. #16i raised it to thirteen for a staging section
    that no longer exists; the number stays, because thirteen is still
    about where a row stops being scannable and nine was never about
    anything but a coincidence."""
    counts = _counts()
    assert MAX_APPS_PER_SECTION == 13
    over = {s: n for s, n in counts.items() if n > MAX_APPS_PER_SECTION}
    assert not over, (
        f"sections over the {MAX_APPS_PER_SECTION}-app cap: {over}. Add a "
        "section with a name that means something instead of lengthening "
        "a row nobody reads to the end of.")


def test_the_core_section_leads_the_ctrl_number_slots():
    """Ctrl+1..9 address APPS[0..8]; Core has to lead them.

    Core is nine again — #16i staged Timelapse and Motility Assay out of
    it and #16j put them back — so the nine slots are exactly the
    pipeline. The assertion is written not to care either way: what has
    to stay true is that Core is the first contiguous block, so the low
    numbers reach the pipeline."""
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
    # The sidebar walks APPS and heads each run, so it shows exactly the
    # sections apps are filed under — the same list as the Home bands.
    assert headings == [s for s, _rows in home_bands()]


def test_sidebar_has_one_row_per_app_plus_home_in_apps_order(
        qtbot, qt_theme_applied):
    bar = Sidebar()
    qtbot.addWidget(bar)
    keys = [b.property("navKey") for b in bar.findChildren(QPushButton)]
    assert keys == ["__home__"] + [k for k, *_r in APPS]
    # Each row announces itself to a screen reader with name + description
    by_key = {b.property("navKey"): b for b in bar.findChildren(QPushButton)}
    for key, name, desc, _s in APPS:
        btn = by_key[key]
        assert btn.accessibleName() == name
        assert btn.accessibleDescription() == desc
        assert btn.toolTip() == f"{name} — {desc}"
        # "&&" is how Qt is told to DRAW an ampersand: a lone "&" is a
        # mnemonic, and "Align & Stitch" was rendering as "Align _Stitch"
        # in this column. The accessible name and the tooltip above
        # carry the real string.
        assert btn.full_text() == f"  {name}".replace("&", "&&")


def test_sidebar_emits_the_key_of_the_row_that_was_clicked(
        qtbot, qt_theme_applied):
    bar = Sidebar()
    qtbot.addWidget(bar)
    by_key = {b.property("navKey"): b for b in bar.findChildren(QPushButton)}
    for key in ("__home__", "mask", "invasion"):
        with qtbot.waitSignal(bar.nav_selected, timeout=1000) as blocker:
            by_key[key].click()
        assert blocker.args == [key]


# ===========================================================================
# 2. Sidebar width policy + label elision
# ===========================================================================

def _fake_apps(names):
    return [(f"k{i}", n, f"desc {i}", SECTION_CORE)
            for i, n in enumerate(names)]


def test_sidebar_column_never_narrows_below_its_floor(
        qtbot, qt_theme_applied, monkeypatch):
    """Short names must not produce a stubby column."""
    from spacr.qt.preferences import scaled_px
    monkeypatch.setattr(app_mod, "APPS", _fake_apps(["A", "B", "C"]))
    bar = Sidebar()
    qtbot.addWidget(bar)
    assert bar.width() == scaled_px(Sidebar.WIDTH_MIN)
    assert not bar.clipped_items()


def test_sidebar_column_widens_for_a_longer_name(
        qtbot, qt_theme_applied, monkeypatch):
    from spacr.qt.preferences import scaled_px
    long_ish = "Cellpose Model Comparison Workbench"
    monkeypatch.setattr(app_mod, "APPS", _fake_apps(["A", long_ish]))
    bar = Sidebar()
    qtbot.addWidget(bar)
    assert scaled_px(Sidebar.WIDTH_MIN) < bar.width() <= scaled_px(
        Sidebar.WIDTH_MAX)


def test_sidebar_caps_its_width_and_elides_a_pathological_name(
        qtbot, qapp, qt_theme_applied, monkeypatch):
    """A name no column could hold elides — with the full name on hover —
    instead of pushing the sidebar across the window."""
    from spacr.qt.preferences import scaled_px
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
    assert huge in clipped[0].toolTip()


def test_the_sidebar_re_inks_its_icons_when_the_theme_changes(
        qtbot, qt_theme_applied, monkeypatch):
    """A QIcon bakes its pixmap when it is built, so a stylesheet swap
    alone leaves white glyphs white — on a white light-theme column."""
    from spacr.qt import preferences as prefs
    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "dark")
    bar = Sidebar()
    qtbot.addWidget(bar)
    rows = {b.property("navKey"): b
            for b in bar.findChildren(QPushButton)}
    dark = {k: _img(b.icon()) for k, b in rows.items()}
    assert dark and not any(b.icon().isNull() for b in rows.values())

    monkeypatch.setattr(prefs, "resolve_effective_theme", lambda: "light")
    bar.refresh_icons()
    unchanged = [k for k, b in rows.items() if _img(b.icon()) == dark[k]]
    assert not unchanged, (
        f"these rows kept their dark ink after a theme switch: {unchanged}")


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


def test_no_shipping_sidebar_label_is_clipped(qtbot, qapp, qt_theme_applied):
    bar = Sidebar()
    qtbot.addWidget(bar)
    bar.resize(bar.width(), 1400)
    bar.show()
    qapp.processEvents()
    assert [b.full_text().strip() for b in bar.clipped_items()] == []


# ===========================================================================
# 3. Home tiles: the clipped-label bug that motivated the regrouping
# ===========================================================================

@pytest.fixture(scope="module")
def home_page(qapp, qt_theme_applied):
    page = make_home_page()  # the page MainWindow ships
    page.resize(1500, 950)
    page.show()
    qapp.processEvents()
    yield page
    page.hide()
    qapp.processEvents()


def test_home_renders_one_tile_per_app_under_every_section_heading(home_page):
    assert set(_tiles(home_page)) == {n for _k, n, *_r in APPS}
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


def test_a_tile_is_wide_enough_for_the_longest_name(home_page):
    """Was ``test_a_tile_is_as_wide_as_the_name_it_has_to_draw``.

    That version asked ``HTile.required_width()`` — a per-tile
    measurement from when each section was a horizontal scroller and
    every tile sized itself to its own name. Every tile is the same size
    now, so the contract is that the ONE size fits the longest name; a
    tile that were wider than its neighbour would be the bug.
    """
    from spacr.qt.preferences import scaled_px
    tiles = _tiles(home_page)
    longest = tiles["Annotator Agreement"]
    label = longest.name_label
    assert QFontMetrics(label.font()).horizontalAdvance(
        "Annotator Agreement") <= label.available_text_width()
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

def test_every_app_has_a_non_null_icon():
    blank = [k for k, *_r in APPS if _icon_for_app(k).isNull()]
    assert not blank, f"apps rendering a blank tile: {blank}"


def test_icon_overrides_all_point_at_files_that_exist():
    here = os.path.dirname(os.path.abspath(app_mod.__file__))
    icons = os.path.normpath(os.path.join(here, "..", "resources", "icons"))
    missing = {k: v for k, v in _ICON_OVERRIDES.items()
               if not os.path.isfile(os.path.join(icons, v))}
    assert not missing, f"override points at a missing file: {missing}"


@pytest.mark.parametrize("key,twin", [
    ("train_cellpose", "cellpose_masks"),   # shares the Cellpose glyph
    ("agreement",      "annotate"),         # scores annotation columns
    ("plate_view",     "map_barcodes"),     # ruled bars read as a well grid
        ("model_compare",  "mask"),             # one field, segmented two ways
])
def test_an_override_makes_two_keys_share_one_glyph(key, twin):
    """The override table is the only reason these render alike; a typo in
    it would send one of the pair to the fallback glyph instead."""
    assert _ICON_OVERRIDES[key].endswith(".png")
    assert _img(_icon_for_app(key)) == _img(_icon_for_app(twin))


def test_apps_without_a_shared_source_do_not_render_alike():
    """Guards the test above against a degenerate "everything is the same
    blank icon" pass."""
    assert _img(_icon_for_app("mask")) != _img(_icon_for_app("measure"))


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

def test_every_app_key_navigates_to_its_own_screen(win):
    """Every tile leads somewhere, and the status bar names where."""
    for key, name, _d, _s in APPS:
        win._on_nav_selected(key)
        screen = win._screens.get(key)
        assert isinstance(screen, QWidget), f"{key} built no screen"
        assert win._stack.currentWidget() is screen, f"{key} not shown"
        assert win._status_app_label.text() == name
    assert set(win._screens) == {k for k, *_r in APPS}


def test_a_screen_is_built_once_and_then_reused(win):
    before = win._stack.count()
    win._on_nav_selected("measure")
    first = win._screens["measure"]
    assert win._stack.count() == before + 1
    win._on_nav_selected("mask")
    win._on_nav_selected("measure")
    assert win._screens["measure"] is first
    assert win._stack.currentWidget() is first
    assert win._stack.count() == before + 2


def test_home_returns_to_the_startup_page(win):
    win._on_nav_selected("umap")
    assert win._stack.currentWidget() is not win._startup
    win._on_nav_selected("__home__")
    assert win._stack.currentWidget() is win._startup
    assert win._status_app_label.text() == "Home"
    assert "__home__" not in win._screens


def test_build_screen_returns_the_dedicated_class_where_there_is_one(win):
    expected = {
        "annotate":      "AnnotateScreen",
        "make_masks":    "MakeMasksScreen",
        "queue":         "QueueScreen",
        "db_browser":    "DbBrowserScreen",
        "agreement":     "AgreementScreen",
        "plate_view":    "PlateViewScreen",
        "model_compare": "ModelCompareScreen",
        "align":         "AlignScreen",
        "convert":       "ConvertScreen",
        "foreign":       "ForeignScreen",
        "batch":         "BatchScreen",
        "distributed_jobs": "DistributedJobsScreen",
        "model_zoo":     "ModelZooScreen",
        "report":        "ReportScreen",
        "train_compare": "TrainCompareScreen",
        "classifier_evaluation": "ClassifierEvaluationScreen",
        "run_history":   "RunHistoryScreen",
    }
    for key, cls_name in expected.items():
        screen = win._build_screen(key)
        assert type(screen).__name__ == cls_name, key
        screen.deleteLater()


def test_every_other_key_builds_a_generic_app_screen(win):
    from spacr.qt.screens.app_screen import AppScreen
    dedicated = {"annotate", "make_masks", "queue", "db_browser", "agreement",
                 "plate_view", "model_compare", "align", "convert", "foreign",
                 "batch", "distributed_jobs", "model_zoo", "report", "train_compare",
                 "classifier_evaluation", "run_history"}
    generic = [k for k, *_r in APPS if k not in dedicated]
    assert generic, "expected some generic AppScreen apps"
    for key in generic:
        screen = win._build_screen(key)
        assert isinstance(screen, AppScreen)
        assert screen.app_key == key
        screen.deleteLater()


def test_clicking_a_home_tile_navigates(win, qtbot):
    tile = _tiles(win._startup)["Measure"]
    tile.click()
    assert win._status_app_label.text() == "Measure"
    assert win._stack.currentWidget() is win._screens["measure"]


def test_clicking_a_sidebar_row_navigates(win):
    by_key = {b.property("navKey"): b
              for b in win._sidebar.findChildren(QPushButton)}
    by_key["regression"].click()
    assert win._status_app_label.text() == "Regression"
    by_key["__home__"].click()
    assert win._stack.currentWidget() is win._startup


def test_the_menu_bar_lists_every_app_and_its_entries_navigate(win):
    seen = {}
    for top in win.menuBar().actions():
        if top.text().replace("&", "") != "spaCR":
            continue
        for act in top.menu().actions():
            if not act.isSeparator():
                seen[act.text()] = act.statusTip()
        break
    for _key, name, desc, _s in APPS:
        assert seen.get(name) == desc, f"{name} missing/mislabelled in menu"
    for top in win.menuBar().actions():
        if top.text().replace("&", "") != "spaCR":
            continue
        act = next(a for a in top.menu().actions() if a.text() == "Image UMAP")
        act.trigger()
        break
    assert win._status_app_label.text() == "Image UMAP"


def test_the_home_menu_entry_returns_to_the_startup_page(win):
    win._on_nav_selected("mask")
    for top in win.menuBar().actions():
        if top.text().replace("&", "") != "spaCR":
            continue
        next(a for a in top.menu().actions() if a.text() == "Home").trigger()
        break
    assert win._stack.currentWidget() is win._startup


# -- the search / filter path ------------------------------------------------

def test_the_command_palette_filters_by_name_and_navigates(win, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(win)
    qtbot.addWidget(palette)

    palette._on_filter("annotator agree")
    rows = [palette._list.item(i).text()
            for i in range(palette._list.count())]
    assert any("Annotator Agreement" in r for r in rows)
    assert not any("Mask" in r for r in rows), (
        f"filter let unrelated commands through: {rows}")

    palette._on_activate()
    assert win._status_app_label.text() == "Annotator Agreement"
    assert win._stack.currentWidget() is win._screens["agreement"]


def test_the_command_palette_filters_by_section_name(win, qtbot):
    from spacr.qt.command_palette import CommandPalette
    palette = CommandPalette(win)
    qtbot.addWidget(palette)
    palette._on_filter(SECTION_TOXO)
    rows = [palette._list.item(i).text()
            for i in range(palette._list.count())]
    for name in (n for _k, n, _d, s in APPS if s == SECTION_TOXO):
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

def test_about_shows_the_installed_version(win, modals):
    import spacr
    win._show_about()
    assert len(modals.about) == 1
    title, body = modals.about[0]
    assert title == "About spaCR"
    assert spacr.__version__ in body


def test_about_says_unknown_when_the_version_cannot_be_read(
        win, modals, monkeypatch):
    import spacr
    monkeypatch.delattr(spacr, "__version__")
    win._show_about()
    assert "<b>Version:</b> unknown" in modals.about[0][1]


def test_resolve_version_reports_the_package_version(win):
    import spacr
    assert win._resolve_version() == spacr.__version__


def test_resolve_version_falls_back_to_dev(win, monkeypatch):
    import spacr
    monkeypatch.setattr(spacr, "__version__", "")
    assert win._resolve_version() == "dev"
    monkeypatch.setitem(sys.modules, "spacr", None)
    assert win._resolve_version() == "dev"


def test_help_menu_urls_open_in_a_browser(win, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    for top in win.menuBar().actions():
        if top.text().replace("&", "") != "Help":
            continue
        for act in top.menu().actions():
            if act.text().endswith("(web)"):
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


def test_preferences_opens_the_dialog_and_rebuilds_home(win, monkeypatch):
    from spacr.qt import preferences as prefs
    calls = []

    class _FakeDialog:
        def __init__(self, parent=None):
            calls.append(parent)

        def exec(self):
            calls.append("exec")
            return 1

    monkeypatch.setattr(prefs, "PreferencesDialog", _FakeDialog)
    old_home = win._startup
    win._on_nav_selected("__home__")
    win._open_preferences()

    assert calls == [win, "exec"]
    assert win._startup is not old_home, "Home was not rebuilt"
    assert win._stack.indexOf(old_home) == -1, "old Home left in the stack"
    assert win._stack.currentWidget() is win._startup


def test_preferences_reports_when_the_dialog_is_unavailable(
        win, monkeypatch):
    from spacr.qt import preferences as prefs
    monkeypatch.delattr(prefs, "PreferencesDialog")
    old_home = win._startup
    win._open_preferences()
    assert "Preferences unavailable" in win.statusBar().currentMessage()
    assert win._startup is old_home, "Home should not be rebuilt on failure"


def test_preferences_survives_a_home_rebuild_failure(win, monkeypatch):
    from spacr.qt import preferences as prefs

    class _FakeDialog:
        def __init__(self, parent=None):
            pass

        def exec(self):
            return 1

    def _boom():
        raise RuntimeError("cannot rebuild")

    monkeypatch.setattr(prefs, "PreferencesDialog", _FakeDialog)
    monkeypatch.setattr(win, "_install_startup_page", _boom)
    win._open_preferences()          # must not propagate
    assert win._stack.indexOf(win._startup) != -1


def test_refresh_theme_re_inks_the_sidebar_and_rebuilds_home(win, monkeypatch):
    calls = []
    monkeypatch.setattr(win._sidebar, "refresh_icons",
                        lambda: calls.append("icons"))
    old_home = win._startup
    win.refresh_theme()
    assert calls == ["icons"]
    assert win._startup is not old_home
    assert win._stack.indexOf(old_home) == -1


def test_refresh_theme_survives_either_half_falling_over(win, monkeypatch):
    def _boom(*args):
        raise RuntimeError("restyle failed")

    monkeypatch.setattr(win._sidebar, "refresh_icons", _boom)
    old_home = win._startup
    win.refresh_theme()               # icons blew up; Home still rebuilt
    assert win._startup is not old_home

    monkeypatch.setattr(win, "_install_startup_page", _boom)
    still_home = win._startup
    win.refresh_theme()               # both halves blew up; no exception
    assert win._startup is still_home


def test_rebuilding_home_keeps_a_non_home_screen_on_screen(win):
    win._on_nav_selected("mask")
    current = win._stack.currentWidget()
    old_home = win._startup
    win._rebuild_startup_page()
    assert win._startup is not old_home
    assert win._stack.currentWidget() is current
    assert win._stack.indexOf(old_home) == -1


# ===========================================================================
# 7. Demos
# ===========================================================================

def test_demo_targets_all_resolve_to_a_real_generator(win):
    from spacr.qt import synthetic as syn
    for demo_key, (target_app, gen_name) in win.DEMO_TARGETS.items():
        assert callable(getattr(syn, gen_name, None)), demo_key
        assert target_app in {k for k, *_r in APPS}, demo_key


def test_run_demo_generator_dispatches_on_the_demo_key(win, monkeypatch,
                                                       tmp_path):
    from spacr.qt import synthetic as syn
    seen = []

    def _fake(dst, **kw):
        seen.append(dst)
        return "sentinel-layout"

    monkeypatch.setattr(syn, "generate_crop_demo", _fake)
    assert win._run_demo_generator("crop", str(tmp_path)) == "sentinel-layout"
    assert seen == [str(tmp_path)]


def test_cancelling_the_folder_picker_does_nothing(win, modals, pick_dir):
    pick_dir[0] = ""
    before = dict(win._screens)
    win._on_load_demo("mask")
    assert win._screens == before
    assert not modals.warning
    assert pick_dir[1], "the folder picker was never shown"


def test_a_failing_generator_warns_and_stays_put(win, modals, pick_dir,
                                                 tmp_path, monkeypatch):
    from spacr.qt import synthetic as syn

    def _boom(dst, **kw):
        raise RuntimeError("disk full")

    monkeypatch.setattr(syn, "generate_measure_demo", _boom)
    pick_dir[0] = str(tmp_path)
    win._on_load_demo("measure")
    assert modals.warning == [("Demo generation failed", "disk full")]
    assert "measure" not in win._screens


def test_the_mask_demo_lands_in_the_mask_screen(win, modals, pick_dir,
                                                tmp_path):
    pick_dir[0] = str(tmp_path)
    win._on_load_demo("mask")
    assert not modals.warning
    assert win._stack.currentWidget() is win._screens["mask"]
    src_widget = win._screens["mask"]._settings_model._widgets["src"]
    assert str(tmp_path) in src_widget.text()
    assert win.statusBar().currentMessage().startswith("Loaded mask demo from")


def test_a_demo_whose_screen_never_opened_is_dropped_quietly(
        win, modals, pick_dir, tmp_path, monkeypatch):
    pick_dir[0] = str(tmp_path)
    monkeypatch.setattr(win, "_on_nav_selected", lambda key: None)
    win._on_load_demo("mask")
    assert not modals.warning
    assert "mask" not in win._screens


def test_a_demo_that_cannot_be_applied_warns(win, modals, pick_dir,
                                             tmp_path, monkeypatch):
    def _boom(widget, layout):
        raise ValueError("bad settings csv")

    pick_dir[0] = str(tmp_path)
    monkeypatch.setattr(win, "_apply_demo_to_screen", _boom)
    win._on_load_demo("mask")
    assert modals.warning == [("Demo load failed", "bad settings csv")]


def test_apply_demo_prefers_the_settings_model(win, tmp_path):
    layout = win._run_demo_generator("mask", str(tmp_path))
    applied = []

    class _Screen:
        def apply_settings_dict(self, settings):
            applied.append(settings)

    win._apply_demo_to_screen(_Screen(), layout)
    assert len(applied) == 1
    assert applied[0]["src"] == str(layout.src)


def test_apply_demo_falls_back_to_open_source(win, tmp_path):
    layout = win._run_demo_generator("mask", str(tmp_path))
    seen = []

    class _AnnotateLike:
        def _open_source(self, src):
            seen.append(src)

    win._apply_demo_to_screen(_AnnotateLike(), layout)
    assert seen == [str(layout.src)]


def test_apply_demo_falls_back_to_open_folder(win, tmp_path):
    layout = win._run_demo_generator("mask", str(tmp_path))
    seen = []

    class _MakeMasksLike:
        def _open_folder(self, src):
            seen.append(src)

    win._apply_demo_to_screen(_MakeMasksLike(), layout)
    assert seen == [str(layout.src)]


def test_apply_demo_with_an_unreadable_csv_falls_through(win, tmp_path,
                                                         monkeypatch):
    """When load_settings hands back something that isn't a dict the
    settings path must NOT be taken — the folder fallback runs instead."""
    import spacr.utils as sutils
    layout = win._run_demo_generator("mask", str(tmp_path))
    monkeypatch.setattr(sutils, "load_settings", lambda *a, **k: None)
    seen = []

    class _Both:
        def apply_settings_dict(self, settings):
            seen.append(("settings", settings))

        def _open_source(self, src):
            seen.append(("source", src))

    win._apply_demo_to_screen(_Both(), layout)
    assert seen == [("source", str(layout.src))]


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

    class _Bare:
        """No apply_settings_dict, no _open_source, no _open_folder."""

    bare = _Bare()
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

@pytest.fixture
def no_pipeline_runs(monkeypatch):
    """Stop the E2E chain from actually launching a segmentation run."""
    from spacr.qt.screens.app_screen import AppScreen
    runs = []
    monkeypatch.setattr(AppScreen, "_on_run",
                        lambda self: runs.append(self.app_key))
    return runs


def _write_settings_pack(root, app_key, rows):
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{app_key}_settings.csv"
    path.write_text("\n".join(",".join(str(c) for c in r) for r in rows))
    return path


def test_the_e2e_demo_stops_when_the_user_says_no(win, modals):
    modals.answers = [QMessageBox.No]
    win._on_e2e_demo()
    assert len(modals.questions) == 1
    assert modals.questions[0][0] == "End-to-end demo"
    assert not modals.warning


def test_the_e2e_demo_stops_when_the_folder_picker_is_cancelled(
        win, modals, pick_dir):
    modals.answers = [QMessageBox.Yes]
    pick_dir[0] = ""
    win._on_e2e_demo()
    assert pick_dir[1], "the folder picker was never shown"
    assert not modals.warning


def test_a_failed_download_is_reported(win, modals, pick_dir, tmp_path,
                                       monkeypatch):
    from spacr.qt import hf_download

    def _fake(parent, dest, callback):
        callback(None, "404 not found")

    monkeypatch.setattr(hf_download, "download_toxo_mito_demo", _fake)
    modals.answers = [QMessageBox.Yes]
    pick_dir[0] = str(tmp_path)
    win._on_e2e_demo()
    assert modals.warning[0][0] == "Download"
    assert "404 not found" in modals.warning[0][1]


def test_a_successful_download_starts_the_chain(win, modals, pick_dir,
                                                tmp_path, monkeypatch):
    from spacr.qt import hf_download

    class _Result:
        dataset_path = tmp_path / "data"
        settings_path = tmp_path / "settings"

    def _fake(parent, dest, callback):
        callback(_Result(), None)

    monkeypatch.setattr(hf_download, "download_toxo_mito_demo", _fake)
    # Yes to the demo, then No at the first pipeline stage.
    modals.answers = [QMessageBox.Yes, QMessageBox.No]
    pick_dir[0] = str(tmp_path)
    win._on_e2e_demo()

    assert [t for t, _x in modals.questions] == ["End-to-end demo",
                                                 "Mask generation"]
    assert win.statusBar().currentMessage() == (
        "E2E chain stopped at 'mask' stage.")


def test_the_chain_runs_mask_then_measure_then_opens_annotate(
        win, modals, tmp_path, no_pipeline_runs, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen
    applied = []
    real_apply = AppScreen.apply_settings_dict

    def _record(self, settings):
        applied.append((self.app_key, dict(settings)))
        return real_apply(self, settings)

    monkeypatch.setattr(AppScreen, "apply_settings_dict", _record)

    data = tmp_path / "toxo"
    pack = tmp_path / "pack"
    _write_settings_pack(pack, "mask", [
        ["# a comment row", "ignored"],
        ["orphan_row_with_one_column"],
        [],
        ["  nucleus_channel  ", "2"],
        ["cell_diameter", "37.5"],
        ["save", "TRUE"],
        ["verbose", "false"],
        ["custom_model", "/models/cyto3"],
    ])
    _write_settings_pack(pack, "measure", [["save_measurements", "true"]])

    modals.answers = [QMessageBox.Yes, QMessageBox.Yes, QMessageBox.Yes]
    win._run_e2e_chain(data, pack)

    assert [t for t, _x in modals.questions] == [
        "Mask generation", "Measurement", "Annotation"]
    # mask + measure auto-run; annotate is interactive and must not.
    assert no_pipeline_runs == ["mask", "measure"]
    assert set(win._screens) >= {"mask", "measure", "annotate"}

    mask_settings = dict(applied)["mask"]
    assert mask_settings["src"] == str(data)
    assert mask_settings["nucleus_channel"] == 2
    assert mask_settings["cell_diameter"] == 37.5
    assert mask_settings["save"] is True
    assert mask_settings["verbose"] is False
    assert mask_settings["custom_model"] == "/models/cyto3"
    assert "# a comment row" not in mask_settings
    assert "orphan_row_with_one_column" not in mask_settings
    # Defaults survive alongside the overrides
    assert len(mask_settings) > 8
    assert dict(applied)["measure"]["save_measurements"] is True
    assert win.statusBar().currentMessage().startswith("E2E chain launched")


def test_the_chain_without_a_settings_pack_uses_plain_defaults(
        win, modals, tmp_path, no_pipeline_runs, monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen
    from spacr.qt.screens.settings_model import resolve_default_settings
    applied = []
    monkeypatch.setattr(AppScreen, "apply_settings_dict",
                        lambda self, s: applied.append((self.app_key, dict(s))))

    modals.answers = [QMessageBox.Yes, QMessageBox.No]
    win._run_e2e_chain(tmp_path / "imgs", tmp_path / "missing-pack")

    key, settings = applied[0]
    assert key == "mask"
    expected = dict(resolve_default_settings("mask"))
    expected["src"] = str(tmp_path / "imgs")
    assert settings == expected


def test_the_chain_reports_a_screen_that_will_not_open(win, modals,
                                                       tmp_path):
    class _NoMask(dict):
        def get(self, key, default=None):
            return None if key == "mask" else super().get(key, default)

    win._screens = _NoMask(win._screens)
    modals.answers = [QMessageBox.Yes]
    win._run_e2e_chain(tmp_path, tmp_path)
    assert modals.warning == [("E2E", "Couldn't open the 'mask' screen.")]


def test_the_chain_reports_a_stage_that_blows_up(win, modals, tmp_path,
                                                 monkeypatch):
    from spacr.qt.screens.app_screen import AppScreen

    def _boom(self, settings):
        raise RuntimeError("settings rejected")

    monkeypatch.setattr(AppScreen, "apply_settings_dict", _boom)
    modals.answers = [QMessageBox.Yes, QMessageBox.Yes]
    win._run_e2e_chain(tmp_path, tmp_path)
    assert modals.warning == [("E2E: mask failed", "settings rejected")]
    # It stopped: the measure stage was never offered.
    assert [t for t, _x in modals.questions] == ["Mask generation"]


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


def test_accepting_an_upgrade_runs_pip(win, qtbot, modals, monkeypatch):
    import spacr.updater as updater
    monkeypatch.setattr(updater, "check_for_updates", _update_info)
    calls = []
    monkeypatch.setattr(updater, "run_pip_upgrade",
                        lambda: (calls.append(1), 0)[1])
    modals.answers = [QMessageBox.Yes]
    _run_update_check(win, qtbot, modals)
    assert calls == [1]
    assert "1.0.0" in modals.questions[0][1]
    assert "2.0.0" in modals.questions[0][1]
    assert modals.information[0][1].startswith("Upgrade finished")


def test_a_failing_pip_upgrade_shows_its_exit_code(win, qtbot, modals,
                                                   monkeypatch):
    import spacr.updater as updater
    monkeypatch.setattr(updater, "check_for_updates", _update_info)
    monkeypatch.setattr(updater, "run_pip_upgrade", lambda: 3)
    modals.answers = [QMessageBox.Yes]
    _run_update_check(win, qtbot, modals)
    assert "pip returned exit code 3" in modals.warning[0][1]


def test_declining_an_upgrade_does_nothing(win, qtbot, modals, monkeypatch):
    import spacr.updater as updater
    monkeypatch.setattr(updater, "check_for_updates", _update_info)

    def _must_not_run():
        raise AssertionError("pip must not run when the user declines")

    monkeypatch.setattr(updater, "run_pip_upgrade", _must_not_run)
    modals.answers = [QMessageBox.No]
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

def test_closing_the_window_drains_every_console_panel(win, monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel
    win._on_nav_selected("mask")
    win._on_nav_selected("measure")
    panels = win.findChildren(ConsolePanel)
    assert len(panels) >= 2, "expected one console per app screen"
    drained = []
    monkeypatch.setattr(ConsolePanel, "shutdown",
                        lambda self: drained.append(self))
    win.close()
    assert set(drained) == set(panels)


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


def test_closing_the_window_is_fine_with_no_update_check_running(win):
    assert not hasattr(win, "_update_worker")
    win.close()
    assert not win.isVisible()


def test_closing_after_the_update_worker_was_reaped_is_safe(win):
    """``worker.finished`` schedules ``deleteLater``; by quit time the C++
    object may be gone and PySide raises on any call to it."""

    class _Reaped:
        def wait(self, msecs=None):
            raise RuntimeError("Internal C++ object already deleted.")

    win._update_worker = _Reaped()
    win.close()
    assert not win.isVisible()


def test_a_panel_that_refuses_to_shut_down_does_not_block_the_close(
        win, monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel
    win._on_nav_selected("mask")

    def _boom(self):
        raise RuntimeError("stream stuck")

    monkeypatch.setattr(ConsolePanel, "shutdown", _boom)
    win.close()                       # must not propagate
    assert not win.isVisible()


# ===========================================================================
# 11. Cross-screen hand-offs
# ===========================================================================

def test_the_model_zoo_hands_two_models_to_model_compare(win):
    win._on_zoo_compare_requested({
        "model_a": "cyto3", "model_b": "nuclei",
        "folder": "", "n_fields": 4,
    })
    screen = win._screens["model_compare"]
    assert win._stack.currentWidget() is screen
    assert screen._panel_a.model_edit.text() == "cyto3"
    assert screen._panel_b.model_edit.text() == "nuclei"
    assert screen._fields_box.value() == 4


def test_a_zoo_request_with_missing_keys_leaves_the_screen_alone(win):
    """Absent keys become empty strings / 0, which ModelCompareScreen
    documents as "leave that control alone" — not "blank it"."""
    win._on_nav_selected("model_compare")
    screen = win._screens["model_compare"]
    before = (screen._panel_a.model_edit.text(),
              screen._panel_b.model_edit.text(),
              screen._fields_box.value())
    win._on_zoo_compare_requested({})
    assert (screen._panel_a.model_edit.text(),
            screen._panel_b.model_edit.text(),
            screen._fields_box.value()) == before


def test_a_zoo_request_is_dropped_if_compare_cannot_be_configured(win):
    """A ``model_compare`` entry without ``configure`` swallows the request.

    Measured on the real screen, which is held aside and checked before
    and after: it must be untouched. The control at the end puts it back
    and fires the identical request, so "untouched" is a measurement that
    demonstrably comes out differently when the hand-off does land.
    """
    win._on_nav_selected("model_compare")
    real = win._screens["model_compare"]
    before = real._panel_a.model_edit.text()
    probe = "zoo-drop-probe"
    assert before != probe, "the control below could not tell the two apart"

    placeholder = QWidget()                        # no .configure
    assert not hasattr(placeholder, "configure")
    win._screens["model_compare"] = placeholder
    win._on_zoo_compare_requested({"model_a": probe})

    # Navigation still happened — the user is looking at Model Compare,
    # only the preload was dropped.
    assert win._visit_order[-1] == "model_compare"
    assert real._panel_a.model_edit.text() == before, (
        "the request reached the real screen anyway")

    # Control — same request, a screen that can take it.
    win._screens["model_compare"] = real
    win._on_zoo_compare_requested({"model_a": probe})
    assert real._panel_a.model_edit.text() == probe


def test_snapshot_returns_the_settings_of_the_visible_app_screen(win):
    win._on_nav_selected("measure")
    key, settings = win._snapshot_current_screen_settings()
    assert key == "measure"
    assert isinstance(settings, dict) and settings
    assert settings == win._screens["measure"]._settings_model.collect()


def test_snapshot_falls_back_to_the_last_app_screen_visited(win):
    win._on_nav_selected("classify")
    win._on_nav_selected("queue")     # QueueScreen is not an AppScreen
    assert win._stack.currentWidget() is win._screens["queue"]
    key, settings = win._snapshot_current_screen_settings()
    assert key == "classify"
    assert isinstance(settings, dict) and settings


def test_snapshot_means_last_VIEWED_not_last_opened(win):
    """Open Mask, open Measure, go back to Mask, then hit "Add current
    plate" on the Queue screen: the plate being added is Mask's.

    Walking ``_screens`` (creation order) instead of the visit order
    silently queued Measure's settings here.
    """
    win._on_nav_selected("mask")
    win._on_nav_selected("measure")
    win._on_nav_selected("mask")           # revisit — now the newest
    win._on_nav_selected("queue")
    key, settings = win._snapshot_current_screen_settings()
    assert key == "mask"
    assert settings == win._screens["mask"]._settings_model.collect()


def test_snapshot_raises_when_no_app_has_been_opened(win):
    win._on_nav_selected("db_browser")
    with pytest.raises(RuntimeError, match="No active plate settings"):
        win._snapshot_current_screen_settings()


def test_the_legacy_explain_error_hook_is_inert(win):
    assert win._on_explain_error("Traceback…", "mask") is None


# ===========================================================================
# 12. Train hand-off + seed values
# ===========================================================================

def test_train_requested_navigates_and_seeds_the_target(win):
    win._on_train_requested("classify", {"src": "/data/plate1",
                                         "epochs": 7})
    screen = win._screens["classify"]
    assert win._stack.currentWidget() is screen
    widgets = screen._settings_model._widgets
    assert widgets["src"].text() == "/data/plate1"
    if "epochs" in widgets:
        assert widgets["epochs"].value() == 7


def test_train_requested_ignores_keys_the_target_does_not_have(win):
    win._on_train_requested("classify", {"no_such_setting_at_all": 1})
    assert win._stack.currentWidget() is win._screens["classify"]


def test_train_requested_survives_a_value_the_widget_rejects(win):
    """A bad seed must not take the navigation down with it."""
    win._on_train_requested("classify", {"src": "/ok", "epochs": "not-a-number"})
    widgets = win._screens["classify"]._settings_model._widgets
    assert widgets["src"].text() == "/ok"


def test_train_requested_to_home_is_a_no_op(win):
    win._on_train_requested("__home__", {"src": "/x"})
    assert win._stack.currentWidget() is win._startup


def test_train_requested_to_a_screen_without_settings_is_a_no_op(win):
    win._on_train_requested("db_browser", {"src": "/x"})
    assert win._stack.currentWidget() is win._screens["db_browser"]


def test_seed_values_are_applied_per_widget_type(qtbot):
    apply = MainWindow._apply_seed_value

    box = QCheckBox(); qtbot.addWidget(box)
    apply(box, 1)
    assert box.isChecked() is True
    apply(box, 0)
    assert box.isChecked() is False

    spin = QSpinBox(); qtbot.addWidget(spin); spin.setRange(0, 100)
    apply(spin, "42.9")
    assert spin.value() == 42

    dspin = QDoubleSpinBox(); qtbot.addWidget(dspin); dspin.setRange(0, 100)
    apply(dspin, "3.5")
    assert dspin.value() == pytest.approx(3.5)

    combo = QComboBox(); qtbot.addWidget(combo)
    combo.addItem("Alpha", "a")
    combo.addItem("Beta", "b")
    apply(combo, "b")
    assert combo.currentIndex() == 1
    apply(combo, "Alpha")
    assert combo.currentIndex() == 0

    edit = QLineEdit(); qtbot.addWidget(edit)
    apply(edit, None)
    assert edit.text() == ""
    apply(edit, 12)
    assert edit.text() == "12"

    label = QLabel("untouched"); qtbot.addWidget(label)
    apply(label, "ignored")           # unknown widget type: no-op
    assert label.text() == "untouched"


def test_a_combo_seed_that_matches_nothing_leaves_the_index_alone(qtbot):
    combo = QComboBox(); qtbot.addWidget(combo)
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


def test_the_preloader_walks_its_module_list_one_tick_at_a_time(monkeypatch):
    from PySide6.QtCore import QTimer
    scheduled: list = []
    imported: list = []
    real_import = importlib.import_module

    def _fake_import(name):
        imported.append(name)
        if name == "spacr.no_such_module":
            raise ImportError("nope")
        return real_import(name)

    monkeypatch.setattr(QTimer, "singleShot",
                        lambda ms, cb: scheduled.append((ms, cb)))
    monkeypatch.setattr(importlib, "import_module", _fake_import)

    pre = _PipelinePreloader()
    monkeypatch.setattr(pre, "_MODULES", ("spacr.no_such_module", "json"))
    pre.start()

    # A failing import is swallowed and the chain continues.
    assert imported == ["spacr.no_such_module"]
    assert [ms for ms, _cb in scheduled] == [50]

    pre.start()          # already started: must not restart the chain
    assert imported == ["spacr.no_such_module"]
    assert len(scheduled) == 1

    scheduled.pop()[1]()
    assert imported == ["spacr.no_such_module", "json"]
    assert len(scheduled) == 1

    scheduled.pop()[1]()          # past the end: stops, imports nothing more
    assert imported == ["spacr.no_such_module", "json"]
    assert scheduled == []
    assert pre._i == 2


def test_the_window_can_open_straight_into_an_app(qtbot, qt_theme_applied):
    """``spacr-qt mask`` opens on Mask, not on Home."""
    w = MainWindow(initial_app="mask")
    qtbot.addWidget(w)
    assert w._stack.currentWidget() is w._screens["mask"]
    assert w._status_app_label.text() == "Mask"


def test_the_window_still_opens_without_shortcuts_or_the_tour(
        qtbot, qt_theme_applied, monkeypatch):
    """Neither optional subsystem may take the whole window down with it."""
    import spacr.qt
    monkeypatch.setitem(sys.modules, "spacr.qt.shortcuts", None)
    monkeypatch.setitem(sys.modules, "spacr.qt.first_run", None)
    monkeypatch.delattr(spacr.qt, "shortcuts", raising=False)
    monkeypatch.delattr(spacr.qt, "first_run", raising=False)
    w = MainWindow()
    qtbot.addWidget(w)
    assert w._stack.currentWidget() is w._startup
    from PySide6.QtGui import QShortcut
    assert not w.findChildren(QShortcut), (
        "shortcuts should be absent when the module failed to import")


def test_shortcuts_are_installed_when_the_module_is_available(win):
    from PySide6.QtGui import QShortcut
    bound = {sc.key().toString() for sc in win.findChildren(QShortcut)}
    for keys in ("Ctrl+H", "Ctrl+K", "Ctrl+1", "Ctrl+9", "F1"):
        assert keys in bound, f"{keys} was never bound"


def test_the_main_window_schedules_the_preloader_but_not_immediately(win):
    assert isinstance(win._preloader, _PipelinePreloader)
    assert win._preloader._started is False, (
        "the preloader must not import torch/cellpose during __init__")


# ===========================================================================
# 14. Bundled fonts
# ===========================================================================

def test_bundled_open_sans_is_registered(qapp):
    from PySide6.QtGui import QFontDatabase
    _load_bundled_fonts()
    families = QFontDatabase.families()
    assert any("Open Sans" in f for f in families)
    _load_bundled_fonts()          # idempotent
    assert QFontDatabase.families() == families


def test_missing_font_directory_is_not_an_error(monkeypatch):
    """No fonts directory -> nothing is registered, quietly.

    ``QFontDatabase.families()`` cannot show this: a font already loaded
    by another test stays loaded, so the family list is unchanged either
    way. The registration call itself is what is counted — and the
    control at the end (same call, directory present) proves the counter
    is wired to something that really does fire.
    """
    import PySide6.QtGui as qtgui

    added: list = []

    class _SpyDatabase:
        @staticmethod
        def addApplicationFont(path):
            added.append(path)
            return len(added)
    monkeypatch.setattr(qtgui, "QFontDatabase", _SpyDatabase)

    hidden = {"on": True}
    real_isdir = os.path.isdir
    monkeypatch.setattr(
        os.path, "isdir",
        lambda p: False if (hidden["on"]
                            and str(p).endswith(
                                os.path.join("resources", "fonts")))
        else real_isdir(p))

    assert _load_bundled_fonts() is None
    assert added == [], f"fonts were registered from nowhere: {added}"

    # Control — the same call with the directory visible does register the
    # bundled TTFs, so the empty list above is a real observation.
    hidden["on"] = False
    _load_bundled_fonts()
    assert added, "the bundled fonts directory registered nothing at all"
    assert all(p.lower().endswith((".ttf", ".otf")) for p in added)
    assert any("OpenSans" in os.path.basename(p).replace(" ", "")
               for p in added), added


# ===========================================================================
# 15. launch()
# ===========================================================================

class _FakeSignal:
    def __init__(self):
        self.callbacks: list = []

    def connect(self, cb):
        self.callbacks.append(cb)


class _AppShim:
    """Stands in for the QApplication ``launch`` would construct.

    Delegates everything it doesn't override to the real, already-running
    QApplication so ``apply_preferences_to_app`` still does its work —
    except ``setStyleSheet`` (a global re-polish would touch every widget
    other tests left behind) and ``exec`` (which would block forever).
    """

    def __init__(self, real, argv):
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

    def _factory(argv):
        shim = _AppShim(qapp, argv)
        made.append(shim)
        return shim

    _ThreadShim.instances = []
    monkeypatch.setattr(app_mod, "QApplication", _factory)
    monkeypatch.setattr(app_mod, "threading",
                        types.SimpleNamespace(Thread=_ThreadShim))

    state = {"shims": made, "threads": _ThreadShim.instances,
             "before": set(qapp.topLevelWidgets())}

    def _window():
        new = [w for w in qapp.topLevelWidgets()
               if w not in state["before"] and isinstance(w, MainWindow)]
        assert len(new) == 1, f"expected exactly one new window, got {new}"
        return new[0]

    state["window"] = _window
    yield state


def test_launch_opens_the_requested_app(launched, qtbot):
    rc = app_mod.launch(["measure"])
    assert rc == 0

    shim = launched["shims"][0]
    assert shim.exec_calls == 1
    assert shim.app_name == "spaCR"
    assert shim.org_name == "Olafsson Lab"
    assert "QWidget" in (shim.stylesheet or ""), "theme was never applied"

    win = launched["window"]()
    qtbot.addWidget(win)
    assert win.isVisible()
    assert win._status_app_label.text() == "Measure"
    assert "measure" in win._screens
    win.close()


def test_launch_with_no_arguments_opens_on_home(launched, qtbot,
                                                monkeypatch):
    monkeypatch.setattr(sys, "argv", ["spacr-qt"])
    rc = app_mod.launch()
    assert rc == 0
    assert launched["shims"][0].argv == ["spacr-qt"]

    win = launched["window"]()
    qtbot.addWidget(win)
    assert win._screens == {}
    assert win._stack.currentWidget() is win._startup
    assert win._status_app_label.text() == "Home"
    win.close()


def test_launch_prewarms_the_heavy_imports_off_thread(launched, qtbot,
                                                      monkeypatch):
    app_mod.launch([])
    win = launched["window"]()
    qtbot.addWidget(win)

    assert len(launched["threads"]) == 1
    thread = launched["threads"][0]
    assert thread.name == "spacr-prewarm"
    assert thread.daemon is True
    assert thread.started is True

    thread.target()
    assert "spacr.settings" in sys.modules
    assert "spacr.gui_utils" in sys.modules

    # A prewarm import that fails must stay silent — it is an optimisation.
    with monkeypatch.context() as m:
        def _boom(name):
            raise ImportError("no module today")
        m.setattr(importlib, "import_module", _boom)
        thread.target()
    win.close()


def test_launch_drains_the_ai_consoles_on_quit(launched, qtbot, monkeypatch):
    from spacr.qt.widgets.console_panel import ConsolePanel
    app_mod.launch(["mask"])
    win = launched["window"]()
    qtbot.addWidget(win)

    shim = launched["shims"][0]
    assert len(shim.aboutToQuit.callbacks) == 1
    drain = shim.aboutToQuit.callbacks[0]

    drained = []
    monkeypatch.setattr(ConsolePanel, "shutdown",
                        lambda self: drained.append(self))
    drain()
    assert drained and set(drained) == set(win.findChildren(ConsolePanel))

    # The AI providers are asked to kill any surviving subprocess.
    from spacr.qt import ai as ai_mod
    cancelled = []

    class _Provider:
        def cancel_stream(self):
            cancelled.append(self)

    monkeypatch.setattr(ai_mod, "list_providers", lambda: [_Provider()])
    drain()
    assert len(cancelled) == 1

    # A panel that raises on shutdown, or a provider registry that cannot
    # even be read, must not stop the drain.
    def _boom(self):
        raise RuntimeError("stuck")

    def _no_providers():
        raise RuntimeError("providers unavailable")

    monkeypatch.setattr(ConsolePanel, "shutdown", _boom)
    monkeypatch.setattr(ai_mod, "list_providers", _no_providers)
    drain()
    win.close()


def test_launch_survives_a_qimagereader_without_an_allocation_limit(
        launched, qtbot, monkeypatch):
    import PySide6.QtGui as qtgui

    class _OldQImageReader:
        @staticmethod
        def setAllocationLimit(limit):
            raise AttributeError("not in this Qt build")

    monkeypatch.setattr(qtgui, "QImageReader", _OldQImageReader)
    assert app_mod.launch([]) == 0
    win = launched["window"]()
    qtbot.addWidget(win)
    win.close()
