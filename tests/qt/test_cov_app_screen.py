"""Coverage + behaviour tests for :mod:`spacr.qt.screens.app_screen`.

``AppScreen`` is the settings screen every non-interactive spaCR module
renders through, so a value that does not survive the widget round trip is
silently wrong for *every* module.  These tests therefore assert on VALUES:

* one widget per branch of ``SettingsWidgets._widget_for`` is built, and the
  value written into it comes back out of ``collect()`` with its Python type
  intact (``"64"`` -> ``64``, ``"True"`` -> ``True``, ``"0.25"`` -> ``0.25``);
* the collect -> apply -> collect round trip is a fixed point for every app;
* categories are grouped, per-app-suppressed and hinted as declared;
* Run/Stop/progress form a state machine that a real worker thread drives
  through both the success and the failure arm.

Everything runs offscreen, offline, on the CPU, and no test opens a modal.
"""
from __future__ import annotations

import csv
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSpinBox,
)

from spacr.qt.screens.app_screen import (
    APP_INTROS,
    APP_TITLES,
    COLUMN_TABLES,
    HINT_STRIP_LINES,
    AppScreen,
    QtGui_QListWidgetItem_helper,
    _hyperparam_searchable,
)
from spacr.qt.screens.settings_model import _ListEdit, _ScalarEdit
from spacr.qt.widgets.section import Section


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _console_text(console) -> str:
    """Concatenate every stdout/error block rendered in a ConsolePanel."""
    from spacr.qt.widgets.console_panel import _StdoutBlock
    return "\n".join(b.text() for b in console.findChildren(_StdoutBlock))


def _pct(usage_bar) -> int:
    """The integer percentage a :class:`UsageBar` is currently showing."""
    assert usage_bar._pct.text() == f"{usage_bar._bar.value()}%"
    return usage_bar._bar.value()


def _sections(screen) -> list:
    """Every :class:`Section` the screen's settings panel built, in order."""
    return screen.findChildren(Section)


def _section_titles(screen) -> list:
    return [s.title() for s in _sections(screen)]


def _make_screen(qtbot, app_key: str) -> AppScreen:
    scr = AppScreen(app_key)
    qtbot.addWidget(scr)
    return scr


def _settle(qtbot, scr, timeout: int = 20000) -> AppScreen:
    """Wait for the screen's background jobs to deliver.

    The usage poll and the issue report run on worker threads -- ``GPUtil``
    shells out to ``nvidia-smi`` (25 ms, every 2 s) and filing an issue shells
    out to ``gh`` and then talks to api.github.com, which together froze the
    window for up to 28 seconds when they ran inline. The consequence for a
    test is that the effect of ``_refresh_usage()`` or ``_on_file_issue()`` is
    not visible when the call returns; it is visible once the event loop has
    run. This is that wait, kept an explicit call rather than hidden inside
    ``_make_screen`` so each test says which of its steps is asynchronous.
    """
    qtbot.waitUntil(lambda: not scr.is_busy() and scr.active_jobs() == 0,
                    timeout=timeout)
    return scr


def _write_csv(path: Path, rows, header=("Key", "Value")) -> Path:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(header))
        for k, v in rows:
            w.writerow([k, v])
    return path


@pytest.fixture(autouse=True)
def _reset_hover_tooltip():
    """Drop the hover-tooltip singleton's anchor around every test.

    ``HoverTooltip`` is process-wide and keeps a raw reference to the widget
    it was last shown for, plus a pending 250 ms hide timer. A screen torn
    down inside that window leaves a dangling anchor that blows up in the
    NEXT test's event loop (see ``test_hover_tooltip_survives_a_deleted_
    anchor`` — the product bug this works around).
    """
    def _reset():
        from spacr.qt.widgets.hover_tooltip import HoverTooltip
        tip = HoverTooltip._INSTANCE
        if tip is not None:
            tip._hide_timer.stop()
            tip._anchor = None
            tip.hide()

    _reset()
    yield
    _reset()


@pytest.fixture
def no_modals(monkeypatch):
    """Blow up loudly if any code path under test opens a modal dialog.

    A real QMessageBox/QFileDialog hangs a headless run forever; failing
    instead turns that into a red test in under a second.
    """
    def _boom(*a, **k):
        raise AssertionError("a modal dialog was opened during the test")

    for name in ("information", "warning", "critical", "question", "about"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(_boom))
    for name in ("getOpenFileName", "getSaveFileName", "getExistingDirectory"):
        monkeypatch.setattr(QFileDialog, name, staticmethod(_boom))
    monkeypatch.setattr(QDialog, "exec", _boom, raising=False)
    # QMessageBox OVERRIDES exec, so patching QDialog.exec does not reach it
    # and an instance-level `box.exec()` sailed straight through this guard
    # into a real modal loop. That is what wedged the whole qt suite: one
    # test reached `shutdown.ask_how_to_quit`, which builds a QMessageBox and
    # calls exec() on it, and the run sat there for 24 minutes with no output
    # until it was killed. `exec_` too, for the Qt5-style spelling.
    for name in ("exec", "exec_"):
        monkeypatch.setattr(QMessageBox, name, _boom, raising=False)
    return True


class _RecordingMenu(QMenu):
    """QMenu whose exec() records instead of blocking on a modal loop."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.exec_calls = []

    def exec(self, *args):          # noqa: A003 - shadowing QMenu.exec is the point
        self.exec_calls.append(args)
        return None


class _FakeProvider:
    def __init__(self, name, label):
        self.name = name
        self.label = label


def _recording_file_issue(seen: dict, url: str = "https://example/issue"):
    """Stand-in for ``ai.issue_report.file_issue`` that records its args.

    Filing a real issue is a network call to GitHub — the only externality
    mocked in this module.
    """
    def _fn(tb, active_app="", settings=None):
        seen["tb"] = tb
        seen["app"] = active_app
        seen["settings"] = settings
        return url
    return _fn


# ---------------------------------------------------------------------------
# A. Settings-widget construction — one branch of _widget_for per type
# ---------------------------------------------------------------------------

class TestWidgetConstruction:

    def test_mask_screen_builds_one_widget_per_expected_type(self, qtbot):
        """Every widget kind the settings model can emit is actually built."""
        scr = _make_screen(qtbot, "mask")
        widgets = scr._settings_model._widgets
        kinds = {}
        for key, w in widgets.items():
            kinds.setdefault(type(w).__name__, []).append(key)

        # bool -> Toggle, int -> QSpinBox, float -> QDoubleSpinBox,
        # enumerated -> QComboBox, str/None -> _ScalarEdit.
        for kind in ("Toggle", "QSpinBox", "QDoubleSpinBox",
                     "QComboBox", "_ScalarEdit"):
            assert kinds.get(kind), f"no {kind} built for the mask app"

        # And each one carries the DEFAULT of its setting, not a blank widget.
        defaults = scr._settings_model._defaults
        assert isinstance(widgets["batch_size"], QSpinBox)
        assert widgets["batch_size"].value() == int(defaults["batch_size"])
        assert isinstance(widgets["denoise"], QCheckBox)
        assert widgets["denoise"].isChecked() == bool(defaults["denoise"])
        assert isinstance(widgets["metadata_type"], QComboBox)
        assert widgets["metadata_type"].currentText() == str(
            defaults["metadata_type"])
        assert isinstance(widgets["seg_qc_count_ratio"], QDoubleSpinBox)
        assert widgets["seg_qc_count_ratio"].value() == pytest.approx(
            float(defaults["seg_qc_count_ratio"]))

    def test_every_rendered_widget_reaches_a_section_row(self, qtbot):
        """No widget is built and then dropped on the floor."""
        scr = _make_screen(qtbot, "measure")
        built = set(scr._settings_model._widgets)
        laid_out = set()
        for key, w in scr._settings_model._widgets.items():
            # A widget that made it into a Section has a parent chain that
            # reaches one of the screen's Section frames.
            node = w.parentWidget()
            while node is not None:
                if isinstance(node, Section):
                    laid_out.add(key)
                    break
                node = node.parentWidget()
        assert laid_out == built

    def test_tooltip_moves_from_field_to_label(self, qtbot):
        """Hover targets are the LABELS; fields keep no tooltip of their own."""
        scr = _make_screen(qtbot, "mask")
        for w in scr._settings_model._widgets.values():
            assert w.toolTip() == ""
        # Every label registered in the hint map carries BOTH a plain hint
        # (for the bottom strip) and the HTML tip (for the sticky popup).
        assert len(scr._hint_map) == len(scr._settings_model._widgets)
        assert set(scr._hint_map) == set(scr._html_tip_map)
        for lbl, hint in scr._hint_map.items():
            assert isinstance(lbl, QLabel)
            assert hint, "empty plain hint"
            assert "<a href=" in scr._html_tip_map[lbl]
            assert "Open spaCR API documentation" in scr._html_tip_map[lbl]
            assert "https://" in hint

        # And no API link dot beside the field. There was one per setting
        # -- 191 on this form -- each carrying a tooltip of its own, sitting
        # between the label and the field. Hovering the right-hand side of a
        # row popped help, which from the user's side of the screen is
        # indistinguishable from "the field has a tooltip", and was reported
        # as exactly that.
        #
        # The API link is not lost: it is in the label's tooltip HTML, which
        # the `<a href=` and `/api/` assertions above and below cover.
        from spacr.qt.widgets.info_link import InfoLink
        setting_links = [
            link for link in scr.findChildren(InfoLink)
            if link.objectName() == "SettingInfoLink"
        ]
        assert not setting_links, (
            f"{len(setting_links)} API link dots are still on the settings "
            f"form; the help belongs to the label alone")
        for html in scr._html_tip_map.values():
            assert "/api/" in html

    def test_a_row_widget_the_model_does_not_own_keeps_its_own_tooltip(
            self, qtbot, monkeypatch):
        """Rows whose widget isn't in ``_widgets`` get no hint and are left
        with whatever tooltip they arrived with."""
        from PySide6.QtWidgets import QLineEdit
        from spacr.qt.screens import settings_model as sm
        real = sm.SettingsWidgets.build_sections
        stray = {}

        def _patched(self):
            sections = real(self)
            w = QLineEdit()
            w.setToolTip("I own my tooltip")
            stray["w"] = w
            sections.append(("Extras", [("Stray", w)]))
            return sections

        monkeypatch.setattr(sm.SettingsWidgets, "build_sections", _patched)
        scr = _make_screen(qtbot, "mask")
        assert stray["w"].toolTip() == "I own my tooltip"
        assert "EXTRAS" in _section_titles(scr)
        assert all(lbl.text() != "Stray" for lbl in scr._hint_map)
        assert len(scr._hint_map) == len(scr._settings_model._widgets)

    def test_header_shows_title_and_intro_blurb(self, qtbot):
        from spacr.qt.widgets.info_link import InfoLink

        scr = _make_screen(qtbot, "mask")
        texts = [w.text() for w in scr.findChildren(QLabel)]
        assert APP_TITLES["mask"] in texts
        assert APP_INTROS["mask"] in texts
        assert "Configure settings, then press Run." in texts
        module_links = [
            link for link in scr.findChildren(InfoLink)
            if link.objectName() == "ModuleInfoLink"
        ]
        assert len(module_links) == 1
        assert not any("Docs" in text for text in texts)

    def test_unknown_app_key_titles_itself_and_has_no_blurb(self, qtbot):
        """An app key with no APP_TITLES/APP_INTROS entry still constructs."""
        scr = _make_screen(qtbot, "not_a_real_app")
        texts = [w.text() for w in scr.findChildren(QLabel)]
        assert "Not_A_Real_App" in texts        # app_key.title()
        # No intro row => no docs link label.
        assert not any("Docs" in t for t in texts if isinstance(t, str))
        # resolve_default_settings falls back to {"src": "path"}.
        assert scr._settings_model.collect()["src"] == "path"


# ---------------------------------------------------------------------------
# B. Category grouping + conditional categories
# ---------------------------------------------------------------------------

class TestCategoryGrouping:

    def test_mask_hides_timelapse_and_motility_categories(self, qtbot):
        """Tracking and the motility assay are separate modules now."""
        scr = _make_screen(qtbot, "mask")
        titles = _section_titles(scr)
        assert "INPUT & METADATA" in titles
        assert "CELL SEGMENTATION" in titles
        for gone in ("TIMELAPSE", "MOTILITY (BETA)",
                     "MOTILITY ADVANCED (BETA)"):
            assert gone not in titles
        # ... and the keys themselves are not silently parked in "Other".
        keys = set(scr._settings_model._widgets)
        assert "timelapse" not in keys
        assert "motility_analysis" not in keys

    def test_measure_still_shows_timelapse_category(self, qtbot):
        """Measure keeps its timelapse controls in the channel-mapping group."""
        scr = _make_screen(qtbot, "measure")
        assert "MASK & CHANNEL MAPPING" in _section_titles(scr)
        assert "timelapse" in scr._settings_model._widgets

    def test_classify_hides_cellpose_and_needs_no_other_bucket(self, qtbot):
        """It used to assert ``titles[-1] == "OTHER"``.

        The "Other" section is not a heading anyone chose -- it is the
        trailing bucket ``build_sections`` emits for keys in no category at
        all. Classify rendered one holding exactly ``custom_model``, because
        that key was filed under "Cellpose" and Classify hides Cellpose. The
        key now lives beside ``model_type``, which is the question it
        answers, so there is nothing left to bucket.

        The two section names are read off the module's own ordering rather
        than spelled here: the Classify overhaul renamed "Model
        Architecture" to "Model & Regularization" and "Optimization & Loss"
        to "Training & Loss" (commit 30500970), and hard-coded titles made
        that deliberate rename look like a regression. What this test is
        actually defending is the absence of CELLPOSE and OTHER.

        The escape hatch itself still works and is covered by
        ``test_uncategorised_keys_still_land_in_other`` below.
        """
        scr = _make_screen(qtbot, "classify")
        titles = _section_titles(scr)
        assert "CELLPOSE" not in titles
        assert "OTHER" not in titles
        assert "custom_model" in scr._settings_model._widgets
        # custom_model is rendered under some heading, and that heading is
        # the model one -- asserted by where the key landed, not by its name.
        model_section = next(
            (name for name, rows in scr._settings_model.build_sections()
             if any(label_or_key == "custom_model"
                    or getattr(widget, "property", lambda _p: None)(
                        "settingKey") == "custom_model"
                    for label_or_key, widget in rows)),
            None)
        assert model_section is not None, (
            "custom_model is not rendered in any section")
        assert "MODEL" in model_section.upper()

    def test_uncategorised_keys_still_land_in_other(self, qtbot, monkeypatch):
        """The bucket is a safety net, not a section anyone should see."""
        from spacr.qt.screens import settings_model as sm
        real = sm.categories_for_app

        def without_epochs(app_key, categories):
            return {
                name: [key for key in keys if key != "epochs"]
                for name, keys in real(app_key, categories).items()
            }

        monkeypatch.setattr(sm, "categories_for_app", without_epochs)
        scr = _make_screen(qtbot, "classify")
        titles = _section_titles(scr)
        assert titles[-1] == "OTHER"

    def test_sections_are_ordered_and_each_row_is_labelled(self, qtbot):
        scr = _make_screen(qtbot, "umap")
        titles = _section_titles(scr)
        assert titles[0] == "PATHS"
        assert len(titles) == len(set(titles)), "a category was emitted twice"

    def test_curated_and_fallback_section_hints(self, qtbot):
        """Curated blurbs are used verbatim; anything else gets the generic.

        Classify's "Plate Sources & Workflow" used to be asserted as the
        FALLBACK here -- it was one of nine Classify categories that had no
        curated entry. Every rendered category has one now (see
        tests/qt/test_category_tooltips.py), so it is the curated arm, and
        the fallback is exercised on a title that is not a category at all.
        """
        from spacr.qt.screens.app_screen import SECTION_HINTS as HINTS
        from spacr.qt.screens.settings_model import category_tooltip
        from spacr.qt.widgets.section import Section

        scr = _make_screen(qtbot, "classify")
        by_title = {s.title(): s for s in _sections(scr)}
        curated = HINTS["PLATE SOURCES & WORKFLOW"]
        assert by_title["PLATE SOURCES & WORKFLOW"]._header.toolTip() == curated
        assert "Settings that control" not in curated

        # "Other" is the trailing bucket for keys in no category -- not a
        # heading anyone chose, so it deliberately has no blurb. Classify no
        # longer renders one (see
        # test_classify_hides_cellpose_and_needs_no_other_bucket), so the
        # fallback is exercised on a section built directly instead.
        assert "OTHER" not in HINTS
        stray = Section("Other")
        stray.set_hint(category_tooltip("classify", "Other"))
        assert stray._header.toolTip() == "Settings that control other."

    def test_no_settings_defined_banner(self, qtbot, monkeypatch):
        from spacr.qt.screens import settings_model as sm
        monkeypatch.setattr(sm.SettingsWidgets, "build_sections",
                            lambda self: [])
        scr = _make_screen(qtbot, "mask")
        texts = [w.text() for w in scr.findChildren(QLabel)]
        assert "No settings defined for this app." in texts
        assert _sections(scr) == []

    def test_settings_build_failure_is_reported_inline(self, qtbot):
        """A broken settings dict must not take the whole screen down."""
        from spacr.qt.screens import settings_model as sm

        def _boom(self):
            raise RuntimeError("kaboom")

        orig = sm.SettingsWidgets.build_sections
        sm.SettingsWidgets.build_sections = _boom
        try:
            scr = _make_screen(qtbot, "mask")
        finally:
            sm.SettingsWidgets.build_sections = orig
        texts = [w.text() for w in scr.findChildren(QLabel)]
        assert any(t.startswith("Failed to build settings for 'mask': kaboom")
                   for t in texts)
        # The runtime half is still fully built, so the screen is usable.
        assert scr._btn_run.isEnabled()


# ---------------------------------------------------------------------------
# C. Save / load round trips — the whole point of the screen
# ---------------------------------------------------------------------------

class TestRoundTrip:

    @pytest.mark.parametrize("app_key", [
        "mask", "measure", "classify", "umap", "ml_analyze", "regression",
        "map_barcodes", "activation", "invasion",
    ])
    def test_collect_apply_collect_is_a_fixed_point(self, qtbot, app_key):
        """apply(collect()) must not perturb a single value, for any app."""
        scr = _make_screen(qtbot, app_key)
        first = scr._settings_model.collect()
        applied = scr.apply_settings_dict(first)
        second = scr._settings_model.collect()
        assert applied == len(scr._settings_model._widgets)
        assert set(first) == set(second)
        drifted = {k: (first[k], second[k])
                   for k in first if first[k] != second[k]}
        assert drifted == {}, f"{app_key} lost values on round trip: {drifted}"

    def test_csv_strings_are_coerced_back_to_python_types(self, qtbot):
        """Values arrive from a CSV as strings; they must land typed."""
        scr = _make_screen(qtbot, "mask")
        n = scr.apply_settings_dict({
            "src": "/data/plate1",
            "batch_size": "64",
            "denoise": "True",
            "seg_qc_count_ratio": "0.25",
            "metadata_type": "cq1",
        })
        assert n == 5
        out = scr._settings_model.collect()
        assert out["src"] == "/data/plate1"
        assert out["batch_size"] == 64 and isinstance(out["batch_size"], int)
        assert out["denoise"] is True
        assert out["seg_qc_count_ratio"] == pytest.approx(0.25)
        assert out["metadata_type"] == "cq1"

    def test_apply_settings_dict_skips_keys_this_app_lacks(self, qtbot):
        scr = _make_screen(qtbot, "map_barcodes")
        before = scr._settings_model.collect()
        n = scr.apply_settings_dict({
            "src": "/seq/run1",
            "cell_diameter": 42,          # a Mask key; map_barcodes lacks it
            "totally_made_up": "x",
        })
        assert n == 1
        after = scr._settings_model.collect()
        assert after["src"] == "/seq/run1"
        assert "cell_diameter" not in after
        # Nothing else moved.
        del before["src"], after["src"]
        assert before == after

    def test_apply_value_rejects_garbage_without_disturbing_the_widget(
            self, qtbot):
        """A malformed CSV cell leaves the previous value in place."""
        scr = _make_screen(qtbot, "mask")
        w_int = scr._settings_model._widgets["batch_size"]
        w_float = scr._settings_model._widgets["seg_qc_count_ratio"]
        w_int.setValue(7)
        w_float.setValue(0.5)
        scr._apply_value(w_int, "not-a-number")
        scr._apply_value(w_float, "not-a-number")
        scr._apply_value(w_int, None)
        assert w_int.value() == 7
        assert w_float.value() == pytest.approx(0.5)

        # A combo given an option it does not have keeps its selection.
        combo = scr._settings_model._widgets["metadata_type"]
        combo.setCurrentIndex(0)
        keep = combo.currentText()
        scr._apply_value(combo, "no_such_option")
        assert combo.currentText() == keep

        # A line edit given None clears rather than writing the text "None".
        edit = scr._settings_model._widgets["src"]
        scr._apply_value(edit, None)
        assert edit.text() == ""
        assert scr._settings_model.collect()["src"] is None

    def test_apply_settings_dict_counts_only_what_it_could_write(self, qtbot):
        """One unwritable widget must not abort the rest of the import."""
        scr = _make_screen(qtbot, "mask")
        real = scr._apply_value
        broken = scr._settings_model._widgets["denoise"]

        def _flaky(widget, val):
            if widget is broken:
                raise RuntimeError("widget is gone")
            return real(widget, val)

        scr._apply_value = _flaky
        n = scr.apply_settings_dict({"src": "/data/x", "denoise": "True",
                                     "batch_size": "5"})
        assert n == 2
        out = scr._settings_model.collect()
        assert out["src"] == "/data/x"
        assert out["batch_size"] == 5

    def test_apply_value_ignores_a_widget_type_it_does_not_handle(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        stray = QLabel("untouched")
        scr._apply_value(stray, "value")      # must be a silent no-op
        assert stray.text() == "untouched"

    @pytest.mark.parametrize("raw,expected", [
        ("True", True), ("true", True), (" TRUE ", True),
        ("1", True), ("yes", True), ("YES", True),
        ("False", False), ("false", False), ("0", False), ("", False),
        ("no", False), ("None", False),
        (True, True), (False, False), (1, True), (0, False), (None, False),
    ])
    def test_truthy(self, raw, expected):
        assert AppScreen._truthy(raw) is expected

    def test_list_edit_round_trips_a_python_list(self, qtbot):
        """_ListEdit is the repr()-based branch of the widget factory."""
        w = _ListEdit()
        w.set_value([0, 1, 2])
        assert w.text() == "[0, 1, 2]"
        assert w.get_value() == [0, 1, 2]
        w.set_value(None)
        assert w.text() == "" and w.get_value() is None
        w.setText("not python")
        assert w.get_value() == "not python"

    def test_scalar_edit_empty_collects_as_none(self, qtbot):
        w = _ScalarEdit()
        w.set_value(None)
        assert w.get_value() is None
        w.set_value(3)
        assert w.text() == "3" and w.get_value() == "3"

    # Was a strict xfail; fixed in settings_model._read_widget. A combo whose
    # selected option is Python None used to collect as the STRING 'None', so
    # every Qt run shipped strict_errors='None' -- and errors.strict_errors()
    # saw a non-None value and took bool('None') == True, silently turning
    # strict error handling ON for every GUI run. cov_type and 'transform'
    # reached statsmodels the same way.
    def test_none_option_collects_as_none_not_the_string(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        combo = scr._settings_model._widgets["strict_errors"]
        idx = combo.findText("None")
        assert idx >= 0
        combo.setCurrentIndex(idx)
        assert scr._settings_model.collect()["strict_errors"] is None

    # Was a strict xfail; fixed by collect() coercing against
    # settings.expected_types. A setting whose DEFAULT is None gets a
    # free-text widget, so it came back as a raw string and cellpose received
    # diameter='37'. The Tk GUI never had this because it runs check_settings
    # before dispatch.
    def test_numeric_text_fields_collect_as_numbers(self, qtbot):
        from spacr.settings import expected_types
        assert expected_types["cell_diameter"] is int
        scr = _make_screen(qtbot, "mask")
        scr._settings_model._widgets["cell_diameter"].setText("37")
        assert scr._settings_model.collect()["cell_diameter"] == 37


# ---------------------------------------------------------------------------
# D. Import settings (the load half of the round trip)
# ---------------------------------------------------------------------------

class TestImportSettings:

    def test_import_key_value_csv_applies_values(self, qtbot, monkeypatch,
                                                 tmp_path):
        """The header spacr.utils.save_settings writes."""
        path = _write_csv(tmp_path / "s.csv",
                          [("src", "/data/p1"), ("batch_size", 12),
                           ("denoise", "True")])
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(path), "")))
        scr = _make_screen(qtbot, "mask")
        scr._on_import_settings()
        out = scr._settings_model.collect()
        assert out["src"] == "/data/p1"
        assert out["batch_size"] == 12
        assert out["denoise"] is True
        assert "Loaded 3 settings" in _console_text(scr._console)

    def test_import_setting_key_setting_value_csv_applies_values(
            self, qtbot, monkeypatch, tmp_path, no_modals):
        """The OTHER header spaCR writes — spacr/io.py:save_settings_to_db,
        spacr/object.py and the documented default of load_settings all use
        ``setting_key,setting_value``. Importing one must not fail."""
        path = _write_csv(tmp_path / "s.csv",
                          [("src", "/data/p2"), ("batch_size", 9)],
                          header=("setting_key", "setting_value"))
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(path), "")))
        scr = _make_screen(qtbot, "mask")
        scr._on_import_settings()          # no_modals => must not warn
        out = scr._settings_model.collect()
        assert out["src"] == "/data/p2"
        assert out["batch_size"] == 9

    def test_import_cancelled_changes_nothing(self, qtbot, monkeypatch,
                                              no_modals):
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: ("", "")))
        scr = _make_screen(qtbot, "mask")
        before = scr._settings_model.collect()
        scr._on_import_settings()
        assert scr._settings_model.collect() == before
        assert _console_text(scr._console) == ""

    def test_import_of_a_non_settings_csv_warns_and_changes_nothing(
            self, qtbot, monkeypatch, tmp_path):
        path = tmp_path / "wrong.csv"
        path.write_text("alpha,beta\n1,2\n")
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(path), "")))
        seen = []
        monkeypatch.setattr("spacr.qt.screens.app_screen.QMessageBox",
                            type("MB", (), {
                                "warning": staticmethod(
                                    lambda *a: seen.append(a)),
                                "information": staticmethod(
                                    lambda *a: seen.append(a)),
                            }))
        scr = _make_screen(qtbot, "mask")
        before = scr._settings_model.collect()
        scr._on_import_settings()
        assert len(seen) == 1 and seen[0][1] == "Import failed"
        assert scr._settings_model.collect() == before

    def test_moved_settings_are_called_out_on_import(self, qtbot, monkeypatch,
                                                     tmp_path, no_modals):
        """An old Mask CSV with timelapse=True must say so, not go quiet."""
        path = _write_csv(tmp_path / "old.csv",
                          [("src", "/data/p3"), ("timelapse", "True"),
                           ("motility_analysis", "True")])
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            staticmethod(lambda *a, **k: (str(path), "")))
        scr = _make_screen(qtbot, "mask")
        scr._on_import_settings()
        text = _console_text(scr._console)
        # Only src was applicable; the two moved keys are reported.
        assert "Loaded 1 settings" in text
        assert "timelapse=True was ignored" in text
        assert "Timelapse module" in text
        assert "motility_analysis=True was ignored" in text
        assert "Motility Assay module" in text

    def test_moved_settings_notice_is_mask_only(self, qtbot):
        scr = _make_screen(qtbot, "measure")
        scr._warn_about_moved_settings({"timelapse": True,
                                        "motility_analysis": True})
        assert _console_text(scr._console) == ""

    def test_moved_settings_stay_quiet_when_the_flags_are_off(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr._warn_about_moved_settings({"timelapse": "False"})
        scr._warn_about_moved_settings({})
        assert _console_text(scr._console) == ""


# ---------------------------------------------------------------------------
# E. The empty-state banner + src plumbing
# ---------------------------------------------------------------------------

class TestEmptyStateAndSrc:

    def test_banner_shows_for_a_placeholder_src_and_hides_on_a_real_path(
            self, qtbot):
        scr = _make_screen(qtbot, "mask")
        card = scr._empty_state_card
        assert card is not None
        assert card.objectName() == "EmptyStateBanner"
        assert not card.isHidden()

        src = scr._settings_model._widgets["src"]
        src.setText("/path/to/src")     # still a placeholder
        assert not card.isHidden()
        src.setText("/data/real_plate")
        assert card.isHidden()

    def test_banner_absent_when_src_already_points_somewhere(
            self, qtbot, monkeypatch):
        from spacr.qt.screens import settings_model as sm
        real = sm.resolve_default_settings

        def _with_src(app_key):
            s = real(app_key)
            s["src"] = "/data/already_set"
            return s

        monkeypatch.setattr(sm, "resolve_default_settings", _with_src)
        scr = _make_screen(qtbot, "mask")
        assert scr._empty_state_card is None
        assert scr._settings_src_path() == "/data/already_set"

    def test_banner_absent_without_a_src_widget(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr._settings_model._widgets = {}
        assert scr._build_empty_state_banner() is None

    def test_banner_absent_when_the_model_cannot_be_read(self, qtbot):
        """Defensive arm: a settings model with no _widgets at all."""
        scr = _make_screen(qtbot, "mask")
        scr._settings_model = object()
        assert scr._build_empty_state_banner() is None

    def test_maybe_hide_empty_state_is_a_noop_without_a_card(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        card, scr._empty_state_card = scr._empty_state_card, None
        scr._maybe_hide_empty_state("/data/x")      # must not raise
        assert not card.isHidden(), "the detached card must be left alone"

    def test_banner_copes_with_a_src_widget_that_is_not_a_text_field(
            self, qtbot):
        """No text to read means "src is unset" — show the banner, wire
        nothing."""
        scr = _make_screen(qtbot, "mask")
        scr._settings_model._widgets["src"] = QCheckBox()
        card = scr._build_empty_state_banner()
        assert card is not None
        assert card.objectName() == "EmptyStateBanner"
        assert scr._settings_src_path() == ""

    def test_settings_src_path_follows_the_field(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr._settings_model._widgets["src"].setText("  /data/plate7  ")
        assert scr._settings_src_path() == "/data/plate7"
        scr._settings_model._widgets.pop("src")
        assert scr._settings_src_path() == ""

    def test_column_fields_get_a_sql_button_bound_to_the_live_src(self, qtbot):
        """The picker must read src when clicked, not when built."""
        from spacr.qt.widgets.column_picker import ColumnPickerButton
        scr = _make_screen(qtbot, "classify")
        assert "annotation_column" in COLUMN_TABLES
        field = scr._settings_model._widgets["annotation_column"]
        buttons = [b for b in scr.findChildren(ColumnPickerButton)
                   if b.field is field]
        assert len(buttons) == 1
        btn = buttons[0]
        assert btn.table == COLUMN_TABLES["annotation_column"] == "png_list"
        scr._settings_model.set_value_for_key("src", ["/data/plate9"])
        assert btn.db_path() == "/data/plate9"

    def test_non_column_fields_get_no_picker(self, qtbot):
        from spacr.qt.widgets.column_picker import ColumnPickerButton
        scr = _make_screen(qtbot, "classify")
        field = scr._settings_model._widgets["src"]
        assert [b for b in scr.findChildren(ColumnPickerButton)
                if b.field is field] == []

    def test_attach_column_picker_ignores_an_unbound_widget(self, qtbot):
        from spacr.qt.widgets.column_picker import ColumnPickerButton
        scr = _make_screen(qtbot, "mask")
        holder = QLabel(parent=scr)
        scr._attach_column_picker(None, holder)        # no key -> no picker
        scr._attach_column_picker("batch_size", holder)  # not a column field
        assert holder.findChildren(ColumnPickerButton) == []


# ---------------------------------------------------------------------------
# F. Hover hints (event filter)
# ---------------------------------------------------------------------------

class TestHoverHints:

    def test_hint_strip_reserves_four_lines_without_moving_run_stop(
            self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr.resize(1200, 900)
        scr.show()
        qtbot.wait(1)

        expected_height = (
            scr._hint_strip.fontMetrics().lineSpacing() * HINT_STRIP_LINES
        )
        assert scr._hint_strip.minimumHeight() == expected_height
        assert scr._hint_strip.maximumHeight() == expected_height
        before = (scr._btn_run.pos(), scr._btn_stop.pos())

        scr._hint_strip.setText(
            "A deliberately long setting description. " * 30
        )
        qtbot.wait(1)
        assert (scr._btn_run.pos(), scr._btn_stop.pos()) == before

    def test_enter_and_leave_drive_the_hint_strip_and_the_popup(self, qtbot):
        from spacr.qt.widgets.hover_tooltip import HoverTooltip, split_api_link
        scr = _make_screen(qtbot, "mask")
        label, hint = next(iter(scr._hint_map.items()))
        html = scr._html_tip_map[label]
        tip = HoverTooltip.instance()

        assert scr._hint_strip.text() == scr._default_hint()
        scr.eventFilter(label, QEvent(QEvent.Enter))
        assert scr._hint_strip.text() == hint
        assert tip._anchor is label
        # The popup renders the body's trailing documentation link as its own
        # blue "API" word, so the prose it shows is that body without the
        # anchor. The URL is not lost — it moves to the word.
        body, url = split_api_link(html)
        assert tip._label.text() == body
        assert tip.api_url() == url
        assert url.startswith("https://")

        scr.eventFilter(label, QEvent(QEvent.Leave))
        assert scr._hint_strip.text() == scr._default_hint()
        assert tip._hide_timer.isActive()
        tip.cancel_hide()

    def test_enter_on_an_unregistered_widget_leaves_the_strip_alone(
            self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr._hint_strip.setText("untouched")
        scr.eventFilter(QLabel("stray"), QEvent(QEvent.Enter))
        assert scr._hint_strip.text() == "untouched"

    def test_hover_before_the_hint_strip_exists_still_shows_the_popup(
            self, qtbot):
        """Labels install the filter while the SETTINGS panel is built — the
        hint strip only exists once the runtime panel is built."""
        from spacr.qt.widgets.hover_tooltip import HoverTooltip
        scr = _make_screen(qtbot, "mask")
        label = next(iter(scr._hint_map))
        strip = scr._hint_strip
        del scr._hint_strip
        try:
            scr.eventFilter(label, QEvent(QEvent.Enter))
            assert HoverTooltip.instance()._anchor is label
            scr.eventFilter(label, QEvent(QEvent.Leave))
            assert HoverTooltip.instance()._hide_timer.isActive()
        finally:
            scr._hint_strip = strip
            HoverTooltip.instance().cancel_hide()
        assert strip.text() == scr._default_hint()

    def test_other_events_fall_through_to_qwidget(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        label = next(iter(scr._hint_map))
        assert scr.eventFilter(label, QEvent(QEvent.Show)) is False

    # Was a strict xfail; fixed in widgets/hover_tooltip.py. _maybe_hide
    # dereferenced self._anchor after the anchored widget's C++ object was
    # gone -- hover a settings label, switch module inside the 250 ms hide
    # delay, and the slot raised RuntimeError inside the Qt event loop.
    def test_hover_tooltip_survives_a_deleted_anchor(self, qtbot):
        import shiboken6
        from spacr.qt.widgets.hover_tooltip import HoverTooltip
        scr = AppScreen("mask")          # deliberately NOT qtbot-owned
        label = next(iter(scr._hint_map))
        scr.eventFilter(label, QEvent(QEvent.Enter))
        tip = HoverTooltip.instance()
        assert tip._anchor is label
        shiboken6.delete(scr)
        assert not shiboken6.isValid(label)
        tip._maybe_hide()                # must not raise


# ---------------------------------------------------------------------------
# G. Run / Stop state machine
# ---------------------------------------------------------------------------

class TestRunStopStateMachine:

    def _patch_entry(self, monkeypatch, fn):
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.resolve_pipeline_entry",
            lambda key: fn)

    def test_run_then_finish_walks_the_full_button_state_machine(
            self, qtbot, monkeypatch, no_modals):
        # `_on_stop` asks how to quit before it stops anything. `no_modals`
        # turns that prompt into a failure rather than a hang, which is what
        # it is for -- so the answer has to be stubbed for the state machine
        # under test to get past it.
        from spacr.qt import shutdown
        monkeypatch.setattr(shutdown, "ask_how_to_quit",
                            lambda *a, **k: shutdown.GRACEFUL)
        gate = threading.Event()
        seen = {}

        def _fn(settings):
            seen["src"] = settings.get("src")
            print("worker-alive")
            gate.wait(10)
            print("worker-done")

        self._patch_entry(monkeypatch, _fn)
        scr = _make_screen(qtbot, "mask")
        scr._settings_model._widgets["src"].setText("/data/state_machine")

        # idle
        assert scr._btn_run.isEnabled()
        assert not scr._btn_stop.isEnabled()
        assert not scr._progress.isVisibleTo(scr)
        assert scr._thread is None

        try:
            scr._on_run()
            # running
            assert not scr._btn_run.isEnabled()
            assert scr._btn_stop.isEnabled()
            assert scr._progress.isVisibleTo(scr)
            assert scr._thread is not None and scr._worker is not None
            qtbot.waitUntil(lambda: "worker-alive" in _console_text(
                scr._console), timeout=5000)
            assert seen["src"] == "/data/state_machine"

            # Stop is advisory — it asks, it does not kill.
            scr._on_stop()
            assert "Requesting stop" in _console_text(scr._console)
            assert scr._thread.isInterruptionRequested()
        finally:
            gate.set()

        # finished
        qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=10000)
        assert not scr._btn_stop.isEnabled()
        assert not scr._progress.isVisibleTo(scr)
        text = _console_text(scr._console)
        assert "Starting mask" in text
        assert "worker-done" in text
        assert "✓ Finished" in text
        # References are released only once the QThread has really stopped.
        qtbot.waitUntil(lambda: scr._thread is None, timeout=5000)
        assert scr._worker is None

    def test_a_failing_pipeline_reaches_the_failed_arm(self, qtbot,
                                                       monkeypatch, no_modals):
        def _fn(settings):
            raise RuntimeError("pipeline exploded")

        self._patch_entry(monkeypatch, _fn)
        scr = _make_screen(qtbot, "mask")
        scr._on_run()
        qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=10000)
        qtbot.wait(50)
        text = _console_text(scr._console)
        assert "✗ Failed — see traceback above" in text
        assert "pipeline exploded" in text
        assert "RuntimeError" in scr._last_error_text
        assert not scr._btn_stop.isEnabled()

    def test_run_on_an_interactive_only_app_explains_itself(self, qtbot,
                                                            monkeypatch):
        seen = []
        monkeypatch.setattr("spacr.qt.screens.app_screen.QMessageBox",
                            type("MB", (), {
                                "information": staticmethod(
                                    lambda *a: seen.append(a)),
                                "warning": staticmethod(
                                    lambda *a: seen.append(("warn",) + a)),
                            }))
        scr = _make_screen(qtbot, "annotate")
        scr._on_run()
        assert len(seen) == 1
        assert seen[0][1] == "Not runnable"
        assert "'annotate' app is interactive-only" in seen[0][2]
        # Nothing started.
        assert scr._btn_run.isEnabled()
        assert not scr._btn_stop.isEnabled()
        assert scr._thread is None

    def test_run_with_unreadable_settings_warns_and_does_not_start(
            self, qtbot, monkeypatch):
        self._patch_entry(monkeypatch, lambda s: None)
        seen = []
        monkeypatch.setattr("spacr.qt.screens.app_screen.QMessageBox",
                            type("MB", (), {
                                "warning": staticmethod(
                                    lambda *a: seen.append(a)),
                                "information": staticmethod(
                                    lambda *a: seen.append(("info",) + a)),
                            }))
        scr = _make_screen(qtbot, "mask")

        def _boom():
            raise ValueError("channels must be a list")

        scr._settings_model.collect = _boom
        scr._on_run()
        assert len(seen) == 1
        assert seen[0][1] == "Bad settings"
        assert seen[0][2] == "channels must be a list"
        assert scr._btn_run.isEnabled()
        assert scr._thread is None

    def test_stop_without_a_running_thread_is_silent(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        assert scr._thread is None
        scr._on_stop()
        assert _console_text(scr._console) == ""

    def test_stop_survives_a_thread_that_refuses_to_be_interrupted(
            self, qtbot, monkeypatch):
        """Stop must survive a thread whose requestInterruption throws.

        `_on_stop` now asks how to quit first, and this test predates that.
        Without an answer stubbed in it opened a real modal and hung the
        entire suite -- see the `no_modals` fixture. The prompt is stubbed
        rather than suppressed so the assertion below still exercises the
        graceful path it was written for.
        """
        from spacr.qt import shutdown
        monkeypatch.setattr(shutdown, "ask_how_to_quit",
                            lambda *a, **k: shutdown.GRACEFUL)
        scr = _make_screen(qtbot, "mask")

        class _Stubborn:
            # `_on_stop` asks whether the thread is running before it asks it
            # to stop, so a double that answers only requestInterruption is
            # no longer a faithful stand-in for a QThread.
            def isRunning(self):
                return True

            def requestInterruption(self):
                raise RuntimeError("no")

        scr._thread = _Stubborn()
        scr._on_stop()          # must not raise
        assert "Requesting stop" in _console_text(scr._console)
        scr._thread = None

    def test_run_context_failure_does_not_block_the_run(self, qtbot,
                                                        monkeypatch,
                                                        no_modals):
        """set_run_context is best-effort decoration around the console."""
        self._patch_entry(monkeypatch, lambda s: print("ran-anyway"))
        scr = _make_screen(qtbot, "mask")

        def _boom(*a, **k):
            raise RuntimeError("console busy")

        scr._console.set_run_context = _boom
        scr._on_run()
        qtbot.waitUntil(lambda: scr._btn_run.isEnabled(), timeout=10000)
        qtbot.wait(50)
        assert "ran-anyway" in _console_text(scr._console)

    def test_on_finished_reports_elapsed_even_without_a_start_stamp(
            self, qtbot):
        """_on_finished is also reachable straight from a signal replay."""
        scr = _make_screen(qtbot, "mask")
        scr._btn_run.setEnabled(False)
        scr._btn_stop.setEnabled(True)
        scr._progress.setVisible(True)
        scr._on_finished(False)
        assert scr._btn_run.isEnabled()
        assert not scr._btn_stop.isEnabled()
        assert not scr._progress.isVisibleTo(scr)
        assert "✗ Failed" in _console_text(scr._console)

    def test_a_broken_desktop_notifier_does_not_break_the_finish(
            self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "spacr.qt.notify.announce_pipeline_finished",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no dbus")))
        scr = _make_screen(qtbot, "mask")
        scr._btn_run.setEnabled(False)
        scr._on_finished(True)
        assert scr._btn_run.isEnabled()
        assert "✓ Finished" in _console_text(scr._console)

    def test_finish_reports_elapsed_time_to_the_notifier(self, qtbot,
                                                        monkeypatch):
        import time
        seen = []
        monkeypatch.setattr(
            "spacr.qt.notify.announce_pipeline_finished",
            lambda app, status, elapsed: seen.append((app, status, elapsed)))
        scr = _make_screen(qtbot, "mask")
        scr._run_started_at = time.time() - 2.0
        scr._on_finished(True)
        assert len(seen) == 1
        app, status, elapsed = seen[0]
        assert (app, status) == ("mask", "success")
        assert 1.5 <= elapsed <= 10.0
        scr._on_finished(False)
        assert seen[1][1] == "failed"

    def test_clear_thread_refs(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr._thread, scr._worker = object(), object()
        scr._clear_thread_refs()
        assert scr._thread is None and scr._worker is None

    def test_clear_console_button(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr._console.append_stdout("noise\n")
        assert "noise" in _console_text(scr._console)
        scr._btn_clear.click()
        assert _console_text(scr._console) == ""


# ---------------------------------------------------------------------------
# H. Error routing / issue filing
# ---------------------------------------------------------------------------

class TestErrorRouting:

    def test_error_prints_raw_when_ai_is_off(self, qtbot, monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: False)
        scr = _make_screen(qtbot, "mask")
        scr._on_pipeline_error("Traceback ...\nValueError: boom")
        assert "ValueError: boom" in _console_text(scr._console)
        assert scr._last_error_text.endswith("boom")
        assert not scr._btn_file_issue.isVisible()
        assert not scr._btn_file_issue.isEnabled()

    def test_error_routes_through_ai_and_hides_the_raw_traceback(
            self, qtbot, monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.get_provider",
                            lambda name: _FakeProvider(name, name.title()))
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_route_errors_through_ai", lambda: True)
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: False)
        scr = _make_screen(qtbot, "mask")
        scr._console.set_ai_active(True)
        scr._console.set_ai_provider("claude")

        calls = []
        scr._console.open_error_flow = lambda tb, **kw: calls.append((tb, kw))
        scr._on_pipeline_error("SECRET-TRACEBACK")

        assert calls == [("SECRET-TRACEBACK",
                          {"active_app": "mask", "show_raw": False})]
        assert "SECRET-TRACEBACK" not in _console_text(scr._console)

    def test_error_falls_back_to_raw_when_routing_is_disabled(
            self, qtbot, monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.get_provider",
                            lambda name: _FakeProvider(name, name.title()))
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_route_errors_through_ai", lambda: False)
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: False)
        scr = _make_screen(qtbot, "mask")
        scr._console.set_ai_active(True)
        scr._console.set_ai_provider("claude")
        scr._console.open_error_flow = lambda *a, **k: pytest.fail(
            "must not route when the preference is off")
        scr._on_pipeline_error("VISIBLE-TRACEBACK")
        assert "VISIBLE-TRACEBACK" in _console_text(scr._console)

    def test_error_falls_back_to_raw_when_the_ai_lookup_explodes(
            self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_route_errors_through_ai",
            lambda: (_ for _ in ()).throw(RuntimeError("prefs gone")))
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: False)
        scr = _make_screen(qtbot, "mask")
        scr._console.set_ai_active(True)
        scr._console.set_ai_provider("claude")
        scr._on_pipeline_error("STILL-VISIBLE")
        assert "STILL-VISIBLE" in _console_text(scr._console)

    def test_opt_in_auto_files_the_issue_and_reveals_the_button(
            self, qtbot, monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: True)
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_route_errors_through_ai", lambda: False)
        seen = {}

        def _file_issue(tb, active_app="", settings=None):
            seen["tb"] = tb
            seen["app"] = active_app
            seen["settings"] = settings
            return "https://github.com/EinarOlafsson/spacr/issues/new?x=" + "y" * 200

        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue", _file_issue)
        scr = _make_screen(qtbot, "mask")
        scr._settings_model._widgets["src"].setText("/data/for_issue")
        scr._settings_model._widgets["batch_size"].setValue(11)
        scr._settings_model._widgets["denoise"].setChecked(True)
        scr._on_pipeline_error("BOOM-TB")
        # The button is revealed synchronously; the report is filed on a
        # worker, so everything below it waits for the job.
        assert scr._btn_file_issue.isEnabled()
        _settle(qtbot, scr)
        assert seen["tb"] == "BOOM-TB"
        assert seen["app"] == "mask"
        # The snapshot carries the user's actual settings, typed.
        assert seen["settings"]["src"] == "/data/for_issue"
        assert seen["settings"]["batch_size"] == 11
        assert seen["settings"]["denoise"] is True
        assert seen["settings"]["metadata_type"] == \
            scr._settings_model._widgets["metadata_type"].currentText()
        text = _console_text(scr._console)
        assert "[issue] opened pre-filled report" in text
        # The URL is truncated to 100 chars in the console line.
        assert "y" * 101 not in text

    def test_auto_file_failure_is_reported_not_swallowed(self, qtbot,
                                                        monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.settings.get_auto_file_issues",
                            lambda: True)
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_route_errors_through_ai", lambda: False)

        def _boom(*a, **k):
            raise RuntimeError("github down")

        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue", _boom)
        scr = _make_screen(qtbot, "mask")
        scr._on_pipeline_error("TB")
        # The failure now happens on the worker, so it comes back through the
        # completion handler rather than out of the `except` around the call.
        # It must still reach the console: an auto-filed report that silently
        # fails to send is worse than one that fails loudly.
        _settle(qtbot, scr)
        assert "[issue] auto-file failed: github down" in _console_text(
            scr._console)

    def test_unreadable_issue_preference_keeps_the_button_hidden(
            self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_route_errors_through_ai", lambda: False)
        monkeypatch.setattr(
            "spacr.qt.ai.settings.get_auto_file_issues",
            lambda: (_ for _ in ()).throw(RuntimeError("registry gone")))
        scr = _make_screen(qtbot, "mask")
        scr._on_pipeline_error("TB-VISIBLE")
        assert "TB-VISIBLE" in _console_text(scr._console)
        assert not scr._btn_file_issue.isEnabled()

    def test_file_issue_snapshot_failure_still_files_the_issue(self, qtbot,
                                                               monkeypatch):
        seen = {}
        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue",
                            _recording_file_issue(seen))
        scr = _make_screen(qtbot, "mask")
        scr._last_error_text = "TB"

        class _HostileWidgets:
            def items(self):
                raise RuntimeError("widget map corrupted")

        scr._settings_model._widgets = _HostileWidgets()
        scr._on_file_issue()
        _settle(qtbot, scr)
        assert seen["settings"] == {}
        assert "[issue] opened pre-filled report" in _console_text(
            scr._console)

    def test_issue_snapshot_skips_widget_types_it_cannot_read(self, qtbot,
                                                              monkeypatch):
        seen = {}
        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue",
                            _recording_file_issue(seen))
        scr = _make_screen(qtbot, "mask")
        scr._last_error_text = "TB"
        box = QCheckBox()
        box.setChecked(True)
        scr._settings_model._widgets = {"a_flag": box, "a_label": QLabel("x")}
        scr._on_file_issue()
        _settle(qtbot, scr)
        assert seen["settings"] == {"a_flag": True}

    def test_file_issue_button_is_inert_without_a_traceback(self, qtbot,
                                                            monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue",
                            lambda *a, **k: pytest.fail("must not file"))
        scr = _make_screen(qtbot, "mask")
        scr._on_file_issue()
        assert _console_text(scr._console) == ""

    def test_file_issue_survives_an_unreadable_settings_model(self, qtbot,
                                                              monkeypatch):
        seen = {}
        monkeypatch.setattr("spacr.qt.ai.issue_report.file_issue",
                            _recording_file_issue(seen))
        scr = _make_screen(qtbot, "mask")
        scr._last_error_text = "TB"
        scr._settings_model = None
        scr._on_file_issue()
        _settle(qtbot, scr)
        assert seen["settings"] == {}

    def test_explain_error_opens_the_flow_and_emits_for_mainwindow(
            self, qtbot):
        scr = _make_screen(qtbot, "mask")
        emitted = []
        scr.error_explain_requested.connect(
            lambda tb, app: emitted.append((tb, app)))

        scr._on_explain_error()                     # no error yet -> no-op
        assert emitted == []

        opened = []
        scr._console.open_error_flow = lambda *a, **k: opened.append((a, k))
        scr._last_error_text = "TB-TEXT"
        scr._on_explain_error()
        assert opened == [(("TB-TEXT", "mask"), {})]
        assert emitted == [("TB-TEXT", "mask")]


# ---------------------------------------------------------------------------
# I. AI toggle + provider menu
# ---------------------------------------------------------------------------

class TestAiControls:

    def test_menu_without_a_provider_offers_only_the_dialog(self, qtbot,
                                                            monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.configured_providers", lambda: [])
        scr = _make_screen(qtbot, "mask")
        scr._refresh_ai_menu()
        acts = scr._ai_menu.actions()
        labels = [a.text() for a in acts if not a.isSeparator()]
        assert labels == ["(no vendor CLI installed)", "Providers…"]
        assert not acts[0].isEnabled()

    def test_menu_lists_providers_and_marks_the_current_one(self, qtbot,
                                                            monkeypatch):
        providers = [_FakeProvider("claude", "Claude Code"),
                     _FakeProvider("codex", "Codex")]
        monkeypatch.setattr("spacr.qt.ai.configured_providers",
                            lambda: providers)
        monkeypatch.setattr(
            "spacr.qt.ai.get_provider",
            lambda name: next((p for p in providers if p.name == name), None))
        scr = _make_screen(qtbot, "mask")
        scr._console.set_ai_provider("codex")
        scr._refresh_ai_menu()
        acts = [a for a in scr._ai_menu.actions() if not a.isSeparator()]
        assert [a.text() for a in acts] == ["Claude Code", "Codex",
                                            "Providers…"]
        assert acts[0].isChecked() is False
        assert acts[1].isChecked() is True

        # Triggering a provider action selects it and rebuilds the menu.
        acts[0].trigger()
        assert scr._console._current_provider_name == "claude"
        acts2 = [a for a in scr._ai_menu.actions() if not a.isSeparator()]
        assert acts2[0].isChecked() is True

    def test_ai_switch_autoselects_the_first_configured_provider(
            self, qtbot, monkeypatch):
        providers = [_FakeProvider("codex", "Codex")]
        monkeypatch.setattr("spacr.qt.ai.configured_providers",
                            lambda: providers)
        monkeypatch.setattr("spacr.qt.ai.get_provider",
                            lambda name: providers[0]
                            if name == "codex" else None)
        scr = _make_screen(qtbot, "mask")
        assert scr._console._current_provider_name is None
        scr._ai_switch.setChecked(True)
        assert scr._console._ai_active is True
        assert scr._console._current_provider_name == "codex"

    def test_ai_switch_keeps_a_provider_the_user_already_chose(self, qtbot,
                                                               monkeypatch):
        providers = [_FakeProvider("claude", "Claude Code"),
                     _FakeProvider("codex", "Codex")]
        monkeypatch.setattr("spacr.qt.ai.configured_providers",
                            lambda: providers)
        monkeypatch.setattr(
            "spacr.qt.ai.get_provider",
            lambda name: next((p for p in providers if p.name == name), None))
        scr = _make_screen(qtbot, "mask")
        scr._console.set_ai_provider("codex")
        scr._ai_switch.setChecked(True)
        assert scr._console._ai_active is True
        # NOT overwritten with providers[0].
        assert scr._console._current_provider_name == "codex"

    def test_ai_switch_refuses_to_stay_on_without_a_provider(self, qtbot,
                                                             monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.configured_providers", lambda: [])
        scr = _make_screen(qtbot, "mask")
        scr._ai_switch.setChecked(True)
        assert "No vendor CLI installed" in _console_text(scr._console)
        assert scr._ai_switch.isChecked() is False

    def test_ai_switch_off_deactivates_without_touching_the_provider(
            self, qtbot, monkeypatch):
        providers = [_FakeProvider("codex", "Codex")]
        monkeypatch.setattr("spacr.qt.ai.configured_providers",
                            lambda: providers)
        monkeypatch.setattr("spacr.qt.ai.get_provider",
                            lambda name: providers[0])
        scr = _make_screen(qtbot, "mask")
        scr._ai_switch.setChecked(True)
        scr._ai_switch.setChecked(False)
        assert scr._console._ai_active is False
        assert scr._console._current_provider_name == "codex"

    def test_providers_dialog_refreshes_the_menu_when_accepted(
            self, qtbot, monkeypatch):
        state = {"providers": []}
        monkeypatch.setattr("spacr.qt.ai.configured_providers",
                            lambda: list(state["providers"]))
        monkeypatch.setattr(
            "spacr.qt.ai.get_provider",
            lambda name: next((p for p in state["providers"]
                               if p.name == name), None))
        scr = _make_screen(qtbot, "mask")
        assert [a.text() for a in scr._ai_menu.actions()
                if not a.isSeparator()][0] == "(no vendor CLI installed)"

        class _Dlg:
            def __init__(self, parent=None):
                pass

            def exec(self):
                state["providers"] = [_FakeProvider("gemini", "Gemini")]
                return QDialog.Accepted

        monkeypatch.setattr(
            "spacr.qt.widgets.ai_chat_panel._ProvidersDialog", _Dlg)
        scr._on_open_providers_dialog()
        assert [a.text() for a in scr._ai_menu.actions()
                if not a.isSeparator()] == ["Gemini", "Providers…"]

    def test_providers_dialog_rejected_leaves_the_menu_alone(self, qtbot,
                                                             monkeypatch):
        monkeypatch.setattr("spacr.qt.ai.configured_providers", lambda: [])
        scr = _make_screen(qtbot, "mask")

        class _Dlg:
            def __init__(self, parent=None):
                pass

            def exec(self):
                return QDialog.Rejected

        monkeypatch.setattr(
            "spacr.qt.widgets.ai_chat_panel._ProvidersDialog", _Dlg)
        before = [a.text() for a in scr._ai_menu.actions()]
        scr._on_open_providers_dialog()
        assert [a.text() for a in scr._ai_menu.actions()] == before


# ---------------------------------------------------------------------------
# J. Per-app runtime panels: live preview, timelapse, motility, hyperparam
# ---------------------------------------------------------------------------

class TestRuntimePanels:

    def test_mask_gets_a_live_preview_behind_an_lp_toggle(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        assert scr._live_preview is not None
        assert scr._runtime_splitter is not None
        # Starts collapsed.
        assert scr._live_preview_card.isHidden()
        assert scr._lp_switch.isChecked() is False
        scr._lp_switch.setChecked(True)
        assert not scr._live_preview_card.isHidden()
        scr._lp_switch.setChecked(False)
        assert scr._live_preview_card.isHidden()

    def test_lp_switch_is_a_noop_where_there_is_no_card(self, qtbot):
        scr = _make_screen(qtbot, "map_barcodes")
        assert scr._live_preview_card is None
        scr._on_lp_switch(True)      # must not raise
        assert not hasattr(scr, "_lp_switch")

    @pytest.mark.parametrize("app_key,attr", [
        ("timelapse", "_timelapse_preview"),
        ("motility", "_motility_preview"),
        ("measure", "_measure_preview"),
    ])
    def test_each_preview_app_builds_its_own_card(self, qtbot, app_key, attr):
        scr = _make_screen(qtbot, app_key)
        panel = getattr(scr, attr)
        card = getattr(scr, attr + "_card")
        assert panel is not None and card is not None
        assert scr._runtime_splitter is not None
        assert scr._runtime_splitter.count() == 2
        # The other slots stay None rather than leaking from another screen.
        others = {"_live_preview", "_measure_preview", "_timelapse_preview",
                  "_motility_preview", "_hyperparam"} - {attr}
        for name in others:
            assert getattr(scr, name) is None, name
        # The propagate callback routes tuned values into the settings panel.
        assert panel._propagate_cb == scr._propagate_live_settings

    @pytest.mark.parametrize("app_key,attr,label", [
        ("mask", "_live_preview_card", "Live"),
        ("timelapse", "_timelapse_preview_card", "Live"),
        ("motility", "_motility_preview_card", "Live"),
        ("measure", "_measure_preview_card", "Live"),
    ])
    def test_every_runtime_preview_has_a_bottom_right_toggle(
            self, qtbot, app_key, attr, label):
        scr = _make_screen(qtbot, app_key)
        card = getattr(scr, attr)
        assert card.isHidden()
        assert scr._preview_switch.text() == label
        assert scr._preview_switch.isChecked() is False
        scr._preview_switch.setChecked(True)
        assert not card.isHidden()
        scr._preview_switch.setChecked(False)
        assert card.isHidden()

    @pytest.mark.parametrize("app_key", ["umap", "classify", "ml_analyze"])
    def test_hyperparam_apps_get_a_search_card(self, qtbot, app_key):
        scr = _make_screen(qtbot, app_key)
        assert scr._hyperparam is not None
        assert scr._live_preview is None
        assert scr._hp_switch.isChecked() is False
        assert scr._hyperparam_card.isHidden()
        scr._hp_switch.setChecked(True)
        assert not scr._hyperparam_card.isHidden()
        # Opening it seeds the search from the panel's current settings.
        assert scr._hyperparam._settings["src"] == \
            scr._settings_model.collect()["src"]

    def test_umap_search_reads_src_changed_while_panel_is_already_open(
            self, qtbot):
        """Typing or dropping a path after opening Search must not go stale."""
        from spacr.hyperparam import SearchResult

        scr = _make_screen(qtbot, "umap")
        scr._hp_switch.setChecked(True)
        captured = {}

        def _search(request, _on_trial, _should_stop):
            captured["src"] = request.settings.get("src")
            return SearchResult(
                space=request.space, metric=request.criterion)

        scr._hyperparam.set_search_fn(_search)
        scr._settings_model.set_value_for_key(
            "src", "/data/dropped-after-search-opened")
        with qtbot.waitSignal(scr._hyperparam.search_finished, timeout=5000):
            assert scr._hyperparam.run_search()
        assert captured["src"] == "/data/dropped-after-search-opened"

    def test_hyperparam_switch_without_a_settings_model(self, qtbot):
        scr = _make_screen(qtbot, "umap")
        scr._settings_model = None
        scr._on_hyperparam_switch(True)          # must not raise
        assert not scr._hyperparam_card.isHidden()

    def test_hyperparam_switch_is_a_noop_where_there_is_no_card(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        assert scr._hyperparam_card is None
        scr._on_hyperparam_switch(True)          # must not raise
        assert scr._hyperparam is None
        assert not hasattr(scr, "_hp_switch")

    def test_hyperparam_searchable_reports_per_app(self):
        assert _hyperparam_searchable("umap") is True
        assert _hyperparam_searchable("mask") is False

    def test_hyperparam_searchable_is_false_when_the_import_fails(
            self, monkeypatch):
        monkeypatch.setitem(sys.modules, "spacr.qt.screens.hyperparam", None)
        assert _hyperparam_searchable("umap") is False

    def test_propagate_live_settings_writes_into_the_panel(self, qtbot):
        """The values the Live Preview tunes must land in the run settings."""
        scr = _make_screen(qtbot, "mask")
        before = scr._settings_model.collect()
        scr._propagate_live_settings({
            "cell_CP_prob": 3,          # QSpinBox
            "cell_FT": 0.75,            # QDoubleSpinBox
            "normalize": False,         # QCheckBox
            "cell_diameter": 37,        # free-text field (default is None)
            "metadata_type": "cq1",     # QComboBox
            "not_a_setting_here": 1,    # unknown -> silently skipped
        })
        out = scr._settings_model.collect()
        assert out["cell_CP_prob"] == 3
        assert out["cell_FT"] == pytest.approx(0.75)
        assert out["normalize"] is False
        assert out["metadata_type"] == "cq1"
        assert str(out["cell_diameter"]) == "37"
        # Nothing else was disturbed.
        moved = {k for k in before if before[k] != out[k]}
        assert moved == {"cell_CP_prob", "cell_FT", "normalize",
                         "cell_diameter", "metadata_type"}

    def test_propagate_live_settings_without_a_model(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        model, scr._settings_model = scr._settings_model, None
        before = model.collect()
        scr._propagate_live_settings({"cell_FT": 0.1})   # must not raise
        assert model.collect() == before

    def test_console_target_registration_failure_does_not_break_the_screen(
            self, qtbot, monkeypatch):
        monkeypatch.setattr(
            "spacr.qt.verbose_logger.register_console_target",
            lambda panel: (_ for _ in ()).throw(RuntimeError("no logger")))
        scr = _make_screen(qtbot, "mask")
        assert scr._console is not None
        assert scr._btn_run.isEnabled()

    def test_dropzone_failure_does_not_break_the_screen(self, qtbot,
                                                        monkeypatch):
        monkeypatch.setattr(
            "spacr.qt.dnd.install_dropzone",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no dnd")))
        scr = _make_screen(qtbot, "measure")
        assert scr._btn_run.isEnabled()


# ---------------------------------------------------------------------------
# K. Live-preview autoload from src
# ---------------------------------------------------------------------------

class TestLivePreviewAutoload:

    def _tiff(self, tmp_path: Path) -> Path:
        import tifffile
        sub = tmp_path / "plate1" / "well"
        sub.mkdir(parents=True)
        img = (np.arange(32 * 32, dtype=np.uint16) % 500).reshape(32, 32)
        p = sub / "plate1_A01_T01F01L01A01Z01C00.tif"
        tifffile.imwrite(str(p), img)
        return p

    def test_typing_a_src_folder_loads_its_first_tile_into_live_preview(
            self, qtbot, tmp_path):
        """Regression: the debounce timer used to be wired against a
        ``_live_preview`` attribute that did not exist yet (the settings
        panel is built BEFORE the runtime panel), so this never fired."""
        tif = self._tiff(tmp_path)
        scr = _make_screen(qtbot, "mask")
        assert scr._live_preview is not None
        scr._settings_model._widgets["src"].setText(str(tmp_path))
        qtbot.waitUntil(
            lambda: scr._live_preview._image_path is not None, timeout=5000)
        assert scr._live_preview._image_path == tif
        assert scr._live_preview._image is not None

    def test_autoload_accepts_a_direct_image_path(self, qtbot, tmp_path):
        tif = self._tiff(tmp_path)
        scr = _make_screen(qtbot, "mask")
        scr._autoload_live_preview(str(tif))
        qtbot.waitUntil(
            lambda: scr._live_preview._image_path is not None, timeout=5000)
        assert scr._live_preview._image_path == tif

    def test_autoload_ignores_placeholders_and_missing_folders(
            self, qtbot, tmp_path):
        scr = _make_screen(qtbot, "mask")
        for value in ("", "   ", "path", "/path/to/src", "/path",
                      str(tmp_path / "nope"), __file__):
            scr._autoload_live_preview(value)
        qtbot.waitUntil(
            lambda: not scr._live_preview._image_loaders, timeout=5000)
        for value in ("", "   ", "path", "/path/to/src", "/path",
                      str(tmp_path / "nope"), __file__):
            assert scr._live_preview._image_path is None, value

    def test_autoload_ignores_a_folder_with_no_images(self, qtbot, tmp_path):
        (tmp_path / "empty").mkdir()
        scr = _make_screen(qtbot, "mask")
        scr._autoload_live_preview(str(tmp_path / "empty"))
        qtbot.waitUntil(
            lambda: not scr._live_preview._image_loaders, timeout=5000)
        assert scr._live_preview._image_path is None

    def test_autoload_is_a_noop_on_screens_without_a_preview(self, qtbot,
                                                             tmp_path):
        self._tiff(tmp_path)
        scr = _make_screen(qtbot, "map_barcodes")
        assert scr._live_preview is None
        scr._autoload_live_preview(str(tmp_path))   # must not raise
        # No debounce timer is wired either, so typing src costs nothing.
        assert not hasattr(scr, "_live_src_timer")


# ---------------------------------------------------------------------------
# L. Demos menu
# ---------------------------------------------------------------------------

class TestDemosMenu:

    def test_empty_state_cta_opens_the_demos_menu(self, qtbot):
        win = QMainWindow()
        qtbot.addWidget(win)
        menu = _RecordingMenu("&Demos", win)
        win.menuBar().addMenu(menu)
        scr = AppScreen("mask")
        win.setCentralWidget(scr)
        scr._open_demos_menu()
        assert len(menu.exec_calls) == 1

    def test_no_demos_menu_means_nothing_happens(self, qtbot):
        win = QMainWindow()
        qtbot.addWidget(win)
        other = _RecordingMenu("&File", win)
        win.menuBar().addMenu(other)
        scr = AppScreen("mask")
        win.setCentralWidget(scr)
        scr._open_demos_menu()
        assert other.exec_calls == []

    def test_a_demos_entry_without_a_submenu_is_skipped(self, qtbot):
        win = QMainWindow()
        qtbot.addWidget(win)
        win.menuBar().addAction("&Demos")       # bare action, no submenu
        scr = AppScreen("mask")
        win.setCentralWidget(scr)
        scr._open_demos_menu()                  # must not raise
        assert _console_text(scr._console) == ""

    def test_open_demos_menu_without_a_window_is_silent(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        scr.window = lambda: None
        scr._open_demos_menu()          # must not raise
        assert _console_text(scr._console) == ""

    def test_open_demos_menu_survives_a_parent_without_a_menu_bar(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        assert not hasattr(scr.window(), "menuBar")
        scr._open_demos_menu()          # top-level QWidget: no menuBar()
        assert _console_text(scr._console) == ""


# ---------------------------------------------------------------------------
# M. Figures
# ---------------------------------------------------------------------------

class TestFigures:

    def _fig(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(2, 2))
        ax.plot([0, 1, 2], [0, 1, 4])
        return fig

    def test_first_figure_reveals_the_figures_card(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        assert scr._figures_card.isHidden()
        assert scr._figure_queue.count() == 0
        scr._on_figure_ready(self._fig(), "")
        assert not scr._figures_card.isHidden()
        assert scr._figure_queue.count() == 1
        scr._on_figure_ready(self._fig())
        assert scr._figure_queue.count() == 2

    def test_thumbnail_item_helper_renders_an_icon(self, qtbot):
        item = QtGui_QListWidgetItem_helper(self._fig(), 4)
        assert item.text() == "#5"
        assert item.textAlignment() == int(Qt.AlignCenter)
        assert not item.icon().isNull()

    def test_thumbnail_item_helper_skips_an_undecodable_render(self):
        """savefig succeeded but the bytes are not a PNG -> no icon, no crash."""
        class _Garbage:
            def savefig(self, buf, **k):
                buf.write(b"definitely not a png")

            def get_facecolor(self):
                return "white"

        item = QtGui_QListWidgetItem_helper(_Garbage(), 2)
        assert item.text() == "#3"
        assert item.icon().isNull()

    def test_thumbnail_item_helper_survives_an_unrenderable_figure(self):
        class _Bad:
            def savefig(self, *a, **k):
                raise RuntimeError("no canvas")

            def get_facecolor(self):
                return "white"

        item = QtGui_QListWidgetItem_helper(_Bad(), 0)
        assert item.text() == "#1"
        assert item.icon().isNull()


# ---------------------------------------------------------------------------
# N. Usage bars
# ---------------------------------------------------------------------------

class TestUsage:

    def test_refresh_usage_writes_real_percentages(self, qtbot):
        scr = _make_screen(qtbot, "mask")
        _settle(qtbot, scr)               # the poll the constructor started
        scr._refresh_usage()
        _settle(qtbot, scr)
        import psutil                      # already a spaCR dependency
        assert _pct(scr._usage_ram) == pytest.approx(
            psutil.virtual_memory().percent, abs=5.0)
        for bar in (scr._usage_cpu, scr._usage_gpu, scr._usage_vram):
            assert 0 <= _pct(bar) <= 100

    def test_gpu_absent_pins_the_gpu_bars_to_zero(self, qtbot, monkeypatch):
        import types
        fake = types.ModuleType("GPUtil")
        fake.getGPUs = lambda: []
        monkeypatch.setitem(sys.modules, "GPUtil", fake)
        scr = _make_screen(qtbot, "mask")
        _settle(qtbot, scr)
        scr._usage_gpu.set_value(55)
        scr._usage_vram.set_value(55)
        scr._refresh_usage()
        _settle(qtbot, scr)
        assert _pct(scr._usage_gpu) == 0
        assert _pct(scr._usage_vram) == 0

    def test_gpu_present_reports_load_and_memory(self, qtbot, monkeypatch):
        import types

        class _Gpu:
            load = 0.25
            memoryUtil = 0.5

        fake = types.ModuleType("GPUtil")
        fake.getGPUs = lambda: [_Gpu()]
        monkeypatch.setitem(sys.modules, "GPUtil", fake)
        scr = _make_screen(qtbot, "mask")
        _settle(qtbot, scr)
        scr._refresh_usage()
        _settle(qtbot, scr)
        assert _pct(scr._usage_gpu) == 25
        assert _pct(scr._usage_vram) == 50

    def test_psutil_failure_leaves_the_bars_untouched(self, qtbot,
                                                     monkeypatch):
        import types
        fake = types.ModuleType("psutil")

        def _boom(*a, **k):
            raise RuntimeError("no /proc")

        fake.virtual_memory = _boom
        fake.cpu_percent = _boom
        fake.cpu_count = _boom
        scr = _make_screen(qtbot, "mask")
        _settle(qtbot, scr)
        scr._usage_ram.set_value(42)
        monkeypatch.setitem(sys.modules, "psutil", fake)
        scr._refresh_usage()
        _settle(qtbot, scr)
        assert _pct(scr._usage_ram) == 42

    def test_per_core_toggle_creates_one_bar_per_core_and_fills_them(
            self, qtbot):
        import psutil
        n = int(psutil.cpu_count(logical=True) or 0)
        assert n > 0
        scr = _make_screen(qtbot, "mask")
        assert scr._per_core_bars == []
        assert scr._per_core_wrap.isHidden()

        scr._btn_cpu_toggle.setChecked(True)
        assert len(scr._per_core_bars) == n
        assert not scr._per_core_wrap.isHidden()
        assert scr._per_core_bars[0]._label.text() == "C00"
        assert scr._per_core_bars[-1]._label.text() == f"C{n - 1:02d}"

        scr._refresh_usage()
        for bar in scr._per_core_bars:
            assert 0 <= _pct(bar) <= 100

        # Re-opening reuses the bars rather than duplicating them.
        scr._btn_cpu_toggle.setChecked(False)
        assert scr._per_core_wrap.isHidden()
        scr._btn_cpu_toggle.setChecked(True)
        assert len(scr._per_core_bars) == n

    def test_per_core_toggle_with_an_unknown_core_count(self, qtbot,
                                                       monkeypatch):
        import types
        fake = types.ModuleType("psutil")
        fake.cpu_count = lambda logical=True: (_ for _ in ()).throw(
            RuntimeError("nope"))
        scr = _make_screen(qtbot, "mask")
        monkeypatch.setitem(sys.modules, "psutil", fake)
        scr._on_toggle_per_core(True)
        assert scr._per_core_bars == []
        assert not scr._per_core_wrap.isHidden()


# ---------------------------------------------------------------------------
# O. Teardown
# ---------------------------------------------------------------------------

class TestCloseEvent:

    def test_close_stops_a_running_thread_and_clears_the_queue(
            self, qtbot, monkeypatch, no_modals):
        gate = threading.Event()
        monkeypatch.setattr(
            "spacr.qt.screens.app_screen.resolve_pipeline_entry",
            lambda key: (lambda s: gate.wait(10)))
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._on_run()
        assert scr._thread is not None
        gate.set()
        scr.close()
        assert scr._thread is None
        assert scr._worker is None

    def test_close_without_a_run_still_clears_the_figure_queue(self, qtbot):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        fig, _ = plt.subplots(figsize=(1, 1))
        scr._on_figure_ready(fig, "")
        assert scr._figure_queue.count() == 1
        scr.close()
        assert scr._figure_queue.count() == 0

    def test_close_without_a_figure_queue(self, qtbot):
        scr = AppScreen("mask")
        qtbot.addWidget(scr)
        scr._figure_queue = None
        scr.close()                 # must not raise
        assert scr._thread is None

    def test_close_survives_a_thread_and_a_queue_that_misbehave(self, qtbot):
        scr = AppScreen("mask")
        qtbot.addWidget(scr)

        class _BadThread:
            def requestInterruption(self):
                raise RuntimeError("nope")

        class _BadQueue:
            def clear(self):
                raise RuntimeError("nope")

        scr._thread = _BadThread()
        scr._figure_queue = _BadQueue()
        scr.close()                 # must not raise
        assert scr._thread is None
