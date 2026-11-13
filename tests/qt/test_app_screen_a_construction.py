"""How :class:`spacr.qt.screens.app_screen.AppScreen` builds itself.

``AppScreen`` is the settings screen every non-interactive spaCR module
renders through, so a widget built with the wrong type, dropped from its
section, or a runtime card wired to the wrong app is wrong for every module.
Each test below states one construction invariant and walks a case table of
apps / patched build states through it with ``subtests``.

This module also holds the small helpers the other ``test_app_screen_a_*``
modules import: a per-case screen builder that tears the screen down when the
case ends (so cases cannot leak widgets or threads into each other), the
hover-tooltip reset, and the no-modal guard.

Everything runs offscreen, offline, on the CPU, and no test opens a modal.
"""
from __future__ import annotations

import contextlib
import csv
import os
import sys
import types
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent                                  # noqa: E402
from PySide6.QtWidgets import (                                    # noqa: E402
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
)

from spacr.qt.screens.app_screen import (                          # noqa: E402
    APP_INTROS,
    APP_TITLES,
    AppScreen,
    _hyperparam_searchable,
)
from spacr.qt.widgets.section import Section                       # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Shared helpers (imported by the sibling test_app_screen_a_* modules)
# ---------------------------------------------------------------------------

def reset_hover_tooltip() -> None:
    """Drop the process-wide hover tooltip's anchor and pending hide timer.

    ``HoverTooltip`` keeps a raw reference to the widget it was last shown
    for; a screen torn down inside the 250 ms hide window would leave a
    dangling anchor that blows up in the next case's event loop.
    """
    from spacr.qt.widgets.hover_tooltip import HoverTooltip
    tip = HoverTooltip._INSTANCE
    if tip is not None:
        tip._hide_timer.stop()
        tip._anchor = None
        tip.hide()


@contextlib.contextmanager
def built(app_key: str):
    """Build one screen for one case and tear it fully down afterwards."""
    from spacr.qt.widget_cleanup import retire_pyqtgraph_menus
    scr = AppScreen(app_key)
    try:
        yield scr
    finally:
        reset_hover_tooltip()
        retire_pyqtgraph_menus(scr)
        scr.close()
        scr.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


def console_text(console) -> str:
    """Concatenate every stdout/error block rendered in a ConsolePanel."""
    from spacr.qt.widgets.console_panel import _StdoutBlock
    return "\n".join(b.text() for b in console.findChildren(_StdoutBlock))


def section_titles(screen) -> list:
    return [s.title() for s in screen.findChildren(Section)]


def label_texts(screen) -> list:
    return [w.text() for w in screen.findChildren(QLabel)]


def settle(qtbot, scr, timeout: int = 20000):
    """Wait until the screen's background jobs (usage poll, issue filing) land."""
    qtbot.waitUntil(lambda: not scr.is_busy() and scr.active_jobs() == 0,
                    timeout=timeout)
    return scr


def write_csv(path: Path, rows, header=("Key", "Value")) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        for k, v in rows:
            w.writerow([k, v])
    return path


def forbid_modals(mp) -> None:
    """Turn any modal dialog into an AssertionError instead of a hang.

    ``QMessageBox`` overrides ``exec``, so it is patched separately from
    ``QDialog.exec``; an instance-level ``box.exec()`` once wedged the whole
    qt suite for 24 minutes.
    """
    def _boom(*a, **k):
        raise AssertionError("a modal dialog was opened during the test")

    for name in ("information", "warning", "critical", "question", "about"):
        mp.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        mp.setattr(QFileDialog, name, staticmethod(_boom))
    mp.setattr(QDialog, "exec", _boom, raising=False)
    for name in ("exec", "exec_"):
        mp.setattr(QMessageBox, name, _boom, raising=False)


def recording_message_box(mp, seen: list) -> None:
    """Replace the screen's QMessageBox with one that records (kind, *args)."""
    mp.setattr("spacr.qt.screens.app_screen.QMessageBox", type("MB", (), {
        "information": staticmethod(lambda *a: seen.append(("info",) + a)),
        "warning": staticmethod(lambda *a: seen.append(("warn",) + a)),
    }))


@pytest.fixture(autouse=True)
def _reset_hover_tooltip_around_each_test():
    reset_hover_tooltip()
    yield
    reset_hover_tooltip()


# ---------------------------------------------------------------------------
# Widget kinds and their defaults
# ---------------------------------------------------------------------------

def test_each_setting_type_builds_its_widget_kind_carrying_the_default(
        qtbot, subtests):
    """bool -> Toggle/QCheckBox, int -> QSpinBox, float -> QDoubleSpinBox,
    enumerated -> combo field, str/None -> _ScalarEdit; each widget starts at
    its setting's DEFAULT and a written value is collected back typed.
    """
    from spacr.qt.screens.settings_model import _MetadataTypeField, _ScalarEdit
    from spacr.qt.widgets.toggle import Toggle

    with built("mask") as scr:
        widgets = scr._settings_model._widgets
        defaults = scr._settings_model._defaults
        # Bucketed by what the widget IS (subclasses count): enumerated
        # settings are _ValueCombo, a QComboBox subclass.
        for kind in (Toggle, QSpinBox, QDoubleSpinBox, QComboBox, _ScalarEdit):
            with subtests.test(case=f"mask builds a {kind.__name__}"):
                assert any(isinstance(w, kind) for w in widgets.values())

        cases = [
            ("batch_size", QSpinBox,
             lambda w: w.value() == int(defaults["batch_size"])),
            ("plot", QCheckBox,
             lambda w: w.isChecked() == bool(defaults["plot"])),
            ("metadata_type", _MetadataTypeField,
             lambda w: (isinstance(w.combo, QComboBox)
                        and w.get_value() == defaults["metadata_type"]
                        and w.combo.currentData() == defaults["metadata_type"])),
            ("seg_qc_count_ratio", QDoubleSpinBox,
             lambda w: w.value() == pytest.approx(
                 float(defaults["seg_qc_count_ratio"]))),
        ]
        for key, kind, holds_default in cases:
            with subtests.test(case=f"mask {key}"):
                assert isinstance(widgets[key], kind)
                assert holds_default(widgets[key])

    with subtests.test(case="cellpose custom_model is a nullable path"):
        with built("cellpose_masks") as scr:
            widget = scr._settings_model._widgets["custom_model"]
            assert isinstance(widget, _ScalarEdit)
            assert not isinstance(widget, QCheckBox)
            assert scr._settings_model.collect()["custom_model"] is None
            widget.set_value("/models/cpsam.CP_model")
            assert scr._settings_model.collect()["custom_model"] == \
                "/models/cpsam.CP_model"


def test_every_widget_is_laid_out_and_hover_help_lives_on_its_label(
        qtbot, subtests):
    """No widget is built and dropped; the hint/HTML-tip maps cover exactly
    the model's widgets via their LABELS, fields carry no tooltip and no
    per-setting API dot, and each tip links the documentation.
    """
    from spacr.qt.widgets.info_link import InfoLink

    for app_key in ("measure", "mask"):
        with subtests.test(case=app_key), built(app_key) as scr:
            widgets = scr._settings_model._widgets
            laid_out = set()
            for key, w in widgets.items():
                node = w.parentWidget()
                while node is not None:
                    if isinstance(node, Section):
                        laid_out.add(key)
                        break
                    node = node.parentWidget()
            assert laid_out == set(widgets)

            assert len(scr._hint_map) == len(widgets)
            assert all(w.toolTip() == "" for w in widgets.values())
            assert set(scr._hint_map) == set(scr._html_tip_map)
            for lbl, hint in scr._hint_map.items():
                assert isinstance(lbl, QLabel)
                assert hint and "https://" in hint
                html = scr._html_tip_map[lbl]
                assert "<a href=" in html and ">API<" in html
                assert "/api/" in html or "settings_flow.html#" in html, html
            assert not [link for link in scr.findChildren(InfoLink)
                        if link.objectName() == "SettingInfoLink"]


def test_masthead_names_the_app_and_carries_its_blurb(qtbot, subtests):
    """The header shows the app title, its intro and the instruction; the API
    link is in the blurb's hover help (no info dot); an unknown key titles
    itself, has no blurb and falls back to ``{"src": "path"}``.
    """
    from spacr.qt.screens.settings_model import api_docs_url
    from spacr.qt.widgets.info_link import InfoLink

    with subtests.test(case="mask"), built("mask") as scr:
        texts = label_texts(scr)
        assert APP_TITLES["mask"] in texts
        assert APP_INTROS["mask"] in texts
        assert "Configure settings, then press Run." in texts
        assert not [link for link in scr.findChildren(InfoLink)
                    if link.objectName() in ("ModuleInfoLink", "InfoLink")]
        help_label = scr._header.api_help
        assert help_label is not None
        assert APP_INTROS["mask"] in help_label.help_html()
        assert help_label.url() == api_docs_url("mask")
        assert not any("Docs" in t for t in texts)

    with subtests.test(case="unknown app key"), built("not_a_real_app") as scr:
        texts = label_texts(scr)
        assert "Not_A_Real_App" in texts
        assert not any("Docs" in t for t in texts if isinstance(t, str))
        assert scr._settings_model.collect()["src"] == "path"


# ---------------------------------------------------------------------------
# Sections / categories
# ---------------------------------------------------------------------------

SECTION_CASES = [
    # app, titles present, titles absent, widget keys present, keys absent,
    # collected values
    ("mask", {"INPUT & METADATA", "CELL SEGMENTATION"},
     {"TIMELAPSE", "MOTILITY (BETA)", "MOTILITY ADVANCED (BETA)"},
     {"timelapse"}, {"motility_analysis"}, {"timelapse": False}),
    ("measure", {"MASK & CHANNEL MAPPING"}, set(), {"timelapse"}, set(), {}),
    ("classify", set(), {"CELLPOSE", "OTHER"},
     {"custom_model_path"}, {"custom_model"}, {}),
    ("classify_merged", set(), {"CELLPOSE", "OTHER"},
     {"custom_model_path"}, {"custom_model"}, {}),
    ("umap", {"INPUT DATA"}, set(), set(), set(), {}),
]


def test_sections_group_each_apps_categories(qtbot, subtests):
    """Each app renders its declared categories once each, suppresses the
    per-app hidden ones (mask: no motility; classify: no Cellpose, no Other
    bucket), and keeps/omits the keys those categories own.
    """
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    for app, present, absent, keys_in, keys_out, collected in SECTION_CASES:
        with subtests.test(case=app), built(app) as scr:
            titles = section_titles(scr)
            assert len(titles) == len(set(titles)), "a category was emitted twice"
            assert present <= set(titles)
            assert not (absent & set(titles))
            widgets = scr._settings_model._widgets
            out = scr._settings_model.collect()
            assert keys_in <= set(widgets)
            assert not (keys_out & set(widgets))
            assert not (keys_out & set(out))
            for key, value in collected.items():
                assert out[key] is value
            # Only the object-grid apps mount the per-object table, and they
            # mount it whenever their settings model is built.
            has_grid = app in AppScreen.OBJECT_GRID_APPS
            assert (getattr(scr, "_object_grid", None) is not None) is has_grid
            assert ("PER-OBJECT SETTINGS" in {t.upper() for t in titles}) \
                is has_grid
            if app == "umap":
                assert titles[0] == "INPUT DATA"
            if app.startswith("classify"):
                assert "custom_model" not in scr._settings_model._defaults
                assert "custom_model" not in _APP_HIDDEN_KEYS[app]
                _check_full_retranslation(scr)
            if app == "classify":
                from spacr.qt.screens.app_screen import SECTION_HINTS
                by_title = {s.title(): s for s in scr.findChildren(Section)}
                curated = SECTION_HINTS["PLATE SOURCES & WORKFLOW"]
                assert by_title["PLATE SOURCES & WORKFLOW"]._header.toolTip() \
                    == curated
                assert "Settings that control" not in curated

    with subtests.test(case="maturity notice names hidden stages"), \
            built("mask") as scr, pytest.MonkeyPatch.context() as mp:
        notice = scr._maturity_notice
        assert notice is not None
        scr.refresh_maturity_visibility()
        assert notice.isHidden()
        mp.setattr("spacr.qt.preferences.maturity_is_visible",
                   lambda stage: stage not in ("alpha", "beta"))
        stages = {s.maturity() for s in scr.rendered_settings_sections()
                  if s.maturity() in ("alpha", "beta")}
        assert stages, "mask renders no alpha/beta section to hide"
        scr.refresh_maturity_visibility()
        expected = ("Alpha and Beta" if len(stages) > 1
                    else stages.pop().title())
        assert not notice.isHidden()
        assert notice.text().startswith(expected)
        assert "hidden by Preferences" in notice.text()

    with subtests.test(case="section explainers rerender"), \
            built("regression") as scr, pytest.MonkeyPatch.context() as mp:
        boxes = scr._section_explainers
        model_box = scr._model_explainer
        assert model_box is not None and model_box in boxes.values()
        others = [b for b in boxes.values() if b is not model_box]
        assert others
        refreshed = []
        mp.setattr(scr, "_refresh_model_explainer", refreshed.append)
        for box in others:
            box.setHtml("")
        scr._retheme_section_explainers()
        assert refreshed == [None]
        assert all(box.toPlainText().strip() for box in others)
        # The browser's own locale hook routes back to the same rerender.
        others[0].retranslate_dynamic_content()
        assert refreshed == [None, None]
        mp.setitem(boxes, "Gone", None)
        scr._retheme_section_explainers()
        assert len(refreshed) == 3

    with subtests.test(case="fallback hint for an uncurated title"):
        from spacr.qt.screens.app_screen import SECTION_HINTS
        from spacr.qt.screens.settings_model import category_tooltip
        assert "OTHER" not in SECTION_HINTS
        stray = Section("Other")
        stray.set_hint(category_tooltip("classify", "Other"))
        assert stray._header.toolTip() == "Settings that control other."


def test_a_broken_or_unusual_settings_build_still_gives_a_usable_screen(
        qtbot, subtests):
    """Whatever the settings model hands back (an uncategorised key, an extra
    row it does not own, no sections, an exception) and whichever optional
    runtime hook fails, the screen constructs, says what happened, and its
    Run button stays usable.
    """
    from spacr.qt.screens import settings_model as sm
    real_build = sm.SettingsWidgets.build_sections
    real_categories = sm.categories_for_app

    def without_epochs(app_key, categories):
        return {name: [k for k in keys if k != "epochs"]
                for name, keys in real_categories(app_key, categories).items()}

    stray = {}

    def with_stray_row(self):
        sections = real_build(self)
        w = QLineEdit()
        w.setToolTip("I own my tooltip")
        stray["w"] = w
        sections.append(("Extras", [("Stray", w)]))
        return sections

    def boom(self):
        raise RuntimeError("kaboom")

    def raising(message):
        return lambda *a, **k: (_ for _ in ()).throw(RuntimeError(message))

    def check_other(scr):
        assert section_titles(scr)[-1] == "OTHER"

    def check_stray(scr):
        assert stray["w"].toolTip() == "I own my tooltip"
        assert "EXTRAS" in section_titles(scr)
        assert all(lbl.text() != "Stray" for lbl in scr._hint_map)
        assert len(scr._hint_map) == len(scr._settings_model._widgets)

    def check_empty(scr):
        assert "No settings defined for this app." in label_texts(scr)
        assert scr.findChildren(Section) == []

    def check_failed(scr):
        assert any(t.startswith("Failed to build settings for 'mask': kaboom")
                   for t in label_texts(scr))

    def check_runtime_built(scr):
        assert scr._console is not None

    cases = [
        ("uncategorised key lands in OTHER", "classify",
         (sm, "categories_for_app", without_epochs), check_other),
        ("row widget the model does not own", "mask",
         (sm.SettingsWidgets, "build_sections", with_stray_row), check_stray),
        ("no sections", "mask",
         (sm.SettingsWidgets, "build_sections", lambda self: []), check_empty),
        ("build raises", "mask",
         (sm.SettingsWidgets, "build_sections", boom), check_failed),
        ("console target registration fails", "mask",
         ("spacr.qt.verbose_logger.register_console_target",
          raising("no logger")), check_runtime_built),
        ("dropzone install fails", "measure",
         ("spacr.qt.dnd.install_dropzone", raising("no dnd")),
         check_runtime_built),
    ]
    for name, app, patch, check in cases:
        with subtests.test(case=name), pytest.MonkeyPatch.context() as mp:
            mp.setattr(*patch)
            with built(app) as scr:
                check(scr)
                assert scr._btn_run.isEnabled()


# ---------------------------------------------------------------------------
# Runtime preview / hyperparameter cards
# ---------------------------------------------------------------------------

def _check_full_retranslation(scr) -> None:
    """A full language pass over a classify screen remembers each tree
    header's source captions and writes their translations back, and the
    form's buddy labels take part in the same pass.
    """
    from PySide6.QtWidgets import QTreeWidget
    from spacr.qt.i18n import _A11Y_DERIVED_NAME, retranslate_widget_tree, tr

    # A field captioned by a buddy label: Qt names it after the label, so a
    # name an earlier pass derived for it is withdrawn rather than kept stale.
    field = QLineEdit(scr)
    caption = QLabel("Plate folder", scr)
    caption.setBuddy(field)
    field.setAccessibleName("stale derived name")
    field.setProperty(_A11Y_DERIVED_NAME, "stale derived name")
    from PySide6.QtWidgets import QMenu
    menu = QMenu("Export", scr)
    retranslate_widget_tree(scr)
    assert field.accessibleName() != "stale derived name"
    # An unnamed menu is named after its (translated) title.
    assert menu.accessibleName() == tr("Export")
    assert menu.property(_A11Y_DERIVED_NAME) == tr("Export")
    trees = scr.findChildren(QTreeWidget)
    assert trees, "classify builds no tree whose header is retranslated"
    for tree in trees:
        sources = tree._spacr_i18n_header_sources
        assert len(sources) == tree.columnCount()
        header = tree.headerItem()
        for index, source in enumerate(sources):
            if source:
                assert header.text(index) == tr(source)


def _check_measure_preview_status(qtbot, panel) -> None:
    """The measure card's status line says why nothing is drawn, shows a load
    error from the CURRENT load only, and always names a non-RGB display
    mapping (falling back to plain RGB when the preference cannot be read).
    """
    from spacr.qt.i18n import tr
    from spacr.qt.widgets.preview_contract import PRIMARY_NOTES

    prefs = "spacr.qt.preferences.image_display_primaries"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(prefs, lambda: "rgb")
        panel.refresh()
        qtbot.waitUntil(lambda: not panel._render_plan, timeout=10000)
        assert panel._crops == []
        assert panel._status.text() == tr(panel.PREVIEW_SOURCE_HINT)
        assert panel.display_primaries_note() == ""

        panel._on_array_loaded(panel._load_token + 1, {"error": "stale"})
        assert panel._status.text() == tr(panel.PREVIEW_SOURCE_HINT)
        panel._on_array_loaded(panel._load_token,
                               {"error": "cannot read the array",
                                "sets": None})
        assert panel._status.text() == "cannot read the array"

        mp.setattr(prefs, lambda: "cmy")
        note = panel.display_primaries_note()
        assert panel.display_primaries() == "cmy"
        assert note == tr(PRIMARY_NOTES["cmy"])
        panel.set_preview_status("Ready.")
        assert panel._status.text() == f"{tr('Ready.')}  ·  {note}"
        panel.set_preview_status("")
        assert panel._status.text() == note

        mp.setattr(prefs, lambda: "sepia")
        assert "sepia" in panel.display_primaries_note()
        mp.setattr(prefs, lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert panel.display_primaries() == "rgb"


PREVIEW_ATTRS = {"_live_preview", "_measure_preview", "_timelapse_preview",
                 "_motility_preview", "_hyperparam"}


def test_each_preview_app_builds_exactly_its_own_collapsed_card(
        qtbot, subtests):
    """A preview app gets its own panel + card (others stay None), the card
    starts hidden behind a "Live" switch that shows/hides it, tuned values
    route back via ``_propagate_live_settings``, and the runtime splitter
    holds figures / preview / console / System / buttons in that order.
    Apps without a card ignore the switch handlers.
    """
    for app_key, attr in (("timelapse", "_timelapse_preview"),
                          ("motility", "_motility_preview"),
                          ("measure", "_measure_preview"),
                          ("mask", "_live_preview")):
        with subtests.test(case=app_key), built(app_key) as scr:
            panel = getattr(scr, attr)
            card = getattr(scr, attr + "_card")
            assert panel is not None and card is not None
            assert scr._runtime_splitter is not None
            for other in PREVIEW_ATTRS - {attr}:
                assert getattr(scr, other) is None, other
            assert scr._preview_switch.text() == "Live"
            assert card.isHidden() and scr._preview_switch.isChecked() is False
            scr._preview_switch.setChecked(True)
            assert not card.isHidden()
            scr._preview_switch.setChecked(False)
            assert card.isHidden()
            if app_key == "mask":
                from spacr.qt.screens.app_screen import _LIVE_PREVIEW
                for owed, calls in ((True, []), (False, [True])):
                    with pytest.MonkeyPatch.context() as mp:
                        regrouped = []
                        mp.setattr(scr, "_part_is_owed",
                                   lambda part, _o=owed: _o and
                                   part == _LIVE_PREVIEW)
                        mp.setattr(scr, "_live_preview",
                                   types.SimpleNamespace(
                                       regroup_the_folder=lambda:
                                       regrouped.append(True)))
                        scr._regroup_the_live_preview()
                        assert regrouped == calls, f"owed={owed}"
                assert scr._lp_switch.isChecked() is False
                scr._lp_switch.setChecked(True)
                assert not card.isHidden()
                scr._lp_switch.setChecked(False)
                assert card.isHidden()
                continue
            splitter = scr._runtime_splitter
            panes = [splitter.widget(i) for i in range(splitter.count())]
            assert panes == [scr._figures_card, card, scr._console_wrap,
                             scr._usage_card, scr._actions_section]
            assert scr._console.isAncestorOf(scr._console_wrap) is False
            assert scr._console_wrap.isAncestorOf(scr._console)
            assert panel._propagate_cb == scr._propagate_live_settings
            if app_key == "measure":
                _check_measure_preview_status(qtbot, panel)

    with subtests.test(case="map_barcodes has no card"), \
            built("map_barcodes") as scr:
        assert scr._live_preview_card is None
        scr._on_lp_switch(True)
        assert not hasattr(scr, "_lp_switch")
        assert scr._hyperparam_card is None
        scr._on_hyperparam_switch(True)
        assert scr._hyperparam is None
        assert not hasattr(scr, "_hp_switch")


def test_hyperparameter_search_card_follows_the_live_settings(
        qtbot, subtests, monkeypatch):
    """Searchable apps get a hidden search card whose switch opens it seeded
    from the current settings, and a search reads ``src`` at run time (not
    when the card opened); the card opens even without a settings model;
    ``_hyperparam_searchable`` is False for non-searchable apps and when the
    hyperparam screen cannot be imported.
    """
    from spacr.hyperparam import SearchResult

    for app_key in ("umap", "classify", "ml_analyze"):
        with subtests.test(case=app_key), built(app_key) as scr:
            assert scr._hyperparam is not None and scr._live_preview is None
            assert scr._hp_switch.isChecked() is False
            assert scr._hyperparam_card.isHidden()
            scr._hp_switch.setChecked(True)
            assert not scr._hyperparam_card.isHidden()
            assert scr._hyperparam._settings["src"] == \
                scr._settings_model.collect()["src"]
            if app_key != "umap":
                continue
            captured = {}

            def _search(request, _on_trial, _should_stop):
                captured["src"] = request.settings.get("src")
                return SearchResult(space=request.space,
                                    metric=request.criterion)

            scr._hyperparam.set_search_fn(_search)
            scr._settings_model.set_value_for_key(
                "src", "/data/dropped-after-search-opened")
            with qtbot.waitSignal(scr._hyperparam.search_finished,
                                  timeout=5000):
                assert scr._hyperparam.run_search()
            assert captured["src"] == "/data/dropped-after-search-opened"

    with subtests.test(case="umap without a settings model"), \
            built("umap") as scr:
        scr._settings_model = None
        scr._on_hyperparam_switch(True)
        assert not scr._hyperparam_card.isHidden()

    with subtests.test(case="searchable per app"):
        assert _hyperparam_searchable("umap") is True
        assert _hyperparam_searchable("mask") is False
    with subtests.test(case="searchable when the import fails"):
        monkeypatch.setitem(sys.modules, "spacr.qt.screens.hyperparam", None)
        assert _hyperparam_searchable("umap") is False
