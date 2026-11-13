"""Values in and out of the ``AppScreen`` settings form.

A value that does not survive the widget round trip is silently wrong for
every module, so these tests assert on VALUES: collect -> apply -> collect is
a fixed point, CSV strings land typed, unknown keys and garbage are refused
without disturbing anything, imported files are read in both header styles,
and the ``src`` field drives the empty-state banner, the column pickers and
the live-preview autoload.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QCheckBox, QFileDialog, QLabel, QLineEdit  # noqa: E402

from spacr.qt.screens.app_screen import COLUMN_TABLES, AppScreen   # noqa: E402
from spacr.qt.screens.settings_model import _ListEdit, _ScalarEdit  # noqa: E402

from tests.qt.test_app_screen_a_construction import (               # noqa: E402,F401
    _reset_hover_tooltip_around_each_test,
    built,
    console_text,
    forbid_modals,
    recording_message_box,
    write_csv,
)

pytestmark = pytest.mark.qt

ROUND_TRIP_APPS = ["mask", "measure", "classify", "umap", "ml_analyze",
                   "regression", "map_barcodes", "activation", "invasion"]


def test_collect_apply_collect_is_a_fixed_point_for_every_app(qtbot, subtests):
    """apply(collect()) writes every accepted key, perturbs no value, and a
    key the app does not declare is refused (count 0, never collected).
    """
    from spacr.qt.screens.settings_model import _APP_HIDDEN_KEYS

    for app_key in ROUND_TRIP_APPS:
        with subtests.test(case=app_key), built(app_key) as scr:
            model = scr._settings_model
            first = model.collect()
            accepted = set(model._widgets) | set(model._defaults) | set(
                _APP_HIDDEN_KEYS.get(app_key, set()))
            assert scr.apply_settings_dict(first) == len(set(first) & accepted)
            second = model.collect()
            assert set(first) == set(second)
            drifted = {k: (first[k], second[k])
                       for k in first if first[k] != second[k]}
            assert drifted == {}
            unknown = "not_a_declared_run_setting"
            assert unknown not in accepted
            assert scr.apply_settings_dict({unknown: 123}) == 0
            assert unknown not in model.collect()


def test_apply_settings_dict_writes_typed_values_and_counts_what_landed(
        qtbot, subtests):
    """String values from a CSV land as their Python types, keys an app lacks
    are skipped and nothing else moves, one unwritable widget does not abort
    the rest, a None combo option collects as None (not "None"), numeric free
    text collects as a number, and Live-Preview propagation writes only the
    keys it was given (silently with no model).
    """
    from spacr.settings import expected_types

    with subtests.test(case="CSV strings are coerced"), built("mask") as scr:
        n = scr.apply_settings_dict({
            "src": "/data/plate1", "batch_size": "64", "plot": "True",
            "seg_qc_count_ratio": "0.25", "metadata_type": "cq1"})
        assert n == 5
        out = scr._settings_model.collect()
        assert out["src"] == "/data/plate1"
        assert out["batch_size"] == 64 and isinstance(out["batch_size"], int)
        assert out["plot"] is True
        assert out["seg_qc_count_ratio"] == pytest.approx(0.25)
        assert out["metadata_type"] == "cq1"

    with subtests.test(case="keys the app lacks are skipped"), \
            built("map_barcodes") as scr:
        before = scr._settings_model.collect()
        n = scr.apply_settings_dict({"src": "/seq/run1", "cell_diameter": 42,
                                     "totally_made_up": "x"})
        assert n == 1
        after = scr._settings_model.collect()
        assert after["src"] == "/seq/run1"
        assert "cell_diameter" not in after
        del before["src"], after["src"]
        assert before == after

    with subtests.test(case="one unwritable widget"), built("mask") as scr:
        real = scr._apply_value
        broken = scr._settings_model._widgets["plot"]

        def _flaky(widget, val):
            if widget is broken:
                raise RuntimeError("widget is gone")
            return real(widget, val)

        scr._apply_value = _flaky
        assert scr.apply_settings_dict(
            {"src": "/data/x", "plot": "True", "batch_size": "5"}) == 2
        out = scr._settings_model.collect()
        assert out["src"] == "/data/x" and out["batch_size"] == 5

    with subtests.test(case="None option and numeric text"), \
            built("mask") as scr:
        combo = scr._settings_model._widgets["strict_errors"]
        idx = combo.findText("None")
        assert idx >= 0
        combo.setCurrentIndex(idx)
        assert expected_types["cell_diameter"] is int
        scr._settings_model._widgets["cell_diameter"].setText("37")
        out = scr._settings_model.collect()
        assert out["strict_errors"] is None
        assert out["cell_diameter"] == 37

    with subtests.test(case="live preview propagation"), built("mask") as scr:
        before = scr._settings_model.collect()
        scr._propagate_live_settings({
            "cell_cellprob_threshold": 3, "cell_flow_threshold": 0.75,
            "normalize": False, "cell_diameter": 37, "metadata_type": "cq1",
            "not_a_setting_here": 1})
        out = scr._settings_model.collect()
        assert out["cell_cellprob_threshold"] == 3
        assert out["cell_flow_threshold"] == pytest.approx(0.75)
        assert out["normalize"] is False
        assert out["metadata_type"] == "cq1"
        assert str(out["cell_diameter"]) == "37"
        assert {k for k in before if before[k] != out[k]} == {
            "cell_cellprob_threshold", "cell_flow_threshold", "normalize",
            "cell_diameter", "metadata_type"}
        model, scr._settings_model = scr._settings_model, None
        frozen = model.collect()
        scr._propagate_live_settings({"cell_flow_threshold": 0.1})
        assert model.collect() == frozen


def test_widgets_accept_only_values_they_can_hold(qtbot, subtests):
    """A malformed value leaves a spin box / combo / format field at its
    previous value, None clears a text field (never the text "None"), and an
    unhandled widget type is a silent no-op. ``AppScreen._truthy`` maps every
    CSV spelling of a flag to a bool; ``_ListEdit`` round-trips a Python list
    via repr; ``_ScalarEdit`` collects empty as None and text as text.
    """
    with built("mask") as scr:
        widgets = scr._settings_model._widgets
        naming = widgets["metadata_type"]
        keep = naming.combo.currentData()
        assert keep is not None
        widgets["batch_size"].setValue(7)
        widgets["seg_qc_count_ratio"].setValue(0.5)
        cases = [
            ("int garbage", widgets["batch_size"], "not-a-number",
             lambda: widgets["batch_size"].value() == 7),
            ("int None", widgets["batch_size"], None,
             lambda: widgets["batch_size"].value() == 7),
            ("float garbage", widgets["seg_qc_count_ratio"], "not-a-number",
             lambda: widgets["seg_qc_count_ratio"].value() == pytest.approx(0.5)),
            ("combo unknown option", naming.combo, "no_such_option",
             lambda: naming.combo.currentData() == keep),
            ("format field unknown", naming, "no_such_option",
             lambda: naming.get_value() == keep),
            ("src None clears", widgets["src"], None,
             lambda: (widgets["src"].text() == ""
                      and scr._settings_model.collect()["src"] is None)),
        ]
        for name, widget, value, holds in cases:
            with subtests.test(case=name):
                scr._apply_value(widget, value)
                assert holds()
        with subtests.test(case="unhandled widget type"):
            stray = QLabel("untouched")
            scr._apply_value(stray, "value")
            assert stray.text() == "untouched"
        with subtests.test(case="bare line edit None"):
            box = QLineEdit()
            qtbot.addWidget(box)
            box.setText("/data/plate1")
            scr._apply_value(box, None)
            assert box.text() == ""

    truthy = [("True", True), ("true", True), (" TRUE ", True), ("1", True),
              ("yes", True), ("YES", True), ("False", False), ("false", False),
              ("0", False), ("", False), ("no", False), ("None", False),
              (True, True), (False, False), (1, True), (0, False),
              (None, False)]
    for raw, expected in truthy:
        with subtests.test(case=f"truthy {raw!r}"):
            assert AppScreen._truthy(raw) is expected

    with subtests.test(case="_ListEdit"):
        w = _ListEdit()
        w.set_value([0, 1, 2])
        assert w.text() == "[0, 1, 2]" and w.get_value() == [0, 1, 2]
        w.set_value(None)
        assert w.text() == "" and w.get_value() is None
        w.setText("not python")
        assert w.get_value() == "not python"
    with subtests.test(case="_ScalarEdit"):
        w = _ScalarEdit()
        w.set_value(None)
        assert w.get_value() is None
        w.set_value(3)
        assert w.text() == "3" and w.get_value() == "3"


@pytest.mark.usefixtures("console_info_enabled")
def test_import_settings_reads_both_csv_headers_and_explains_what_it_skipped(
        qtbot, subtests, tmp_path):
    """Importing applies a ``Key,Value`` or ``setting_key,setting_value`` CSV
    and logs the count; a cancelled dialog or a non-settings CSV changes
    nothing (the latter warns "Import failed"); a Mask import applies
    ``timelapse`` and reports ``motility_analysis`` as ignored (Measure owns
    it), while the moved-settings notice is Mask-only and quiet for off flags.
    """
    def pick(mp, path):
        mp.setattr(QFileDialog, "getOpenFileName",
                   staticmethod(lambda *a, **k: (str(path), "")))

    with subtests.test(case="Key,Value"), pytest.MonkeyPatch.context() as mp:
        pick(mp, write_csv(tmp_path / "kv.csv", [
            ("src", "/data/p1"), ("batch_size", 12), ("plot", "True")]))
        with built("mask") as scr:
            scr._on_import_settings()
            out = scr._settings_model.collect()
            assert (out["src"], out["batch_size"], out["plot"]) == (
                "/data/p1", 12, True)
            assert "Loaded 3 settings" in console_text(scr._console)

    with subtests.test(case="setting_key,setting_value"), \
            pytest.MonkeyPatch.context() as mp:
        forbid_modals(mp)
        pick(mp, write_csv(tmp_path / "sk.csv", [
            ("src", "/data/p2"), ("batch_size", 9)],
            header=("setting_key", "setting_value")))
        with built("mask") as scr:
            scr._on_import_settings()
            out = scr._settings_model.collect()
            assert (out["src"], out["batch_size"]) == ("/data/p2", 9)

    with subtests.test(case="cancelled"), pytest.MonkeyPatch.context() as mp:
        forbid_modals(mp)
        pick(mp, "")
        with built("mask") as scr:
            before = scr._settings_model.collect()
            console_before = console_text(scr._console)
            scr._on_import_settings()
            assert scr._settings_model.collect() == before
            assert console_text(scr._console) == console_before

    with subtests.test(case="not a settings CSV"), \
            pytest.MonkeyPatch.context() as mp:
        wrong = tmp_path / "wrong.csv"
        wrong.write_text("alpha,beta\n1,2\n", encoding="utf-8")
        pick(mp, wrong)
        seen = []
        recording_message_box(mp, seen)
        with built("mask") as scr:
            before = scr._settings_model.collect()
            scr._on_import_settings()
            assert len(seen) == 1 and seen[0][:1] == ("warn",)
            assert seen[0][2] == "Import failed"
            assert scr._settings_model.collect() == before

    with subtests.test(case="moved settings on Mask"), \
            pytest.MonkeyPatch.context() as mp:
        forbid_modals(mp)
        pick(mp, write_csv(tmp_path / "old.csv", [
            ("src", "/data/p3"), ("timelapse", "True"),
            ("motility_analysis", "True")]))
        with built("mask") as scr:
            from spacr.qt.screens.mask import fold_set
            assert fold_set(scr) is None
            assert scr._settings_model.collect()["timelapse"] is False
            scr._on_import_settings()
            text = console_text(scr._console)
            out = scr._settings_model.collect()
            assert out["src"] == "/data/p3" and out["timelapse"] is True
            assert "motility_analysis" not in out
            assert scr.dimension_switch("t") is not None
            assert "Loaded 2 settings" in text
            assert "timelapse=True was ignored" not in text
            assert "carrying no Timelapse switch" not in text
            assert "motility_analysis=True was ignored" in text
            assert "Measure" in text and "sidebar" not in text

    for app, payloads in (("measure", [{"timelapse": True,
                                        "motility_analysis": True}]),
                          ("mask", [{"timelapse": "False"}, {}])):
        with subtests.test(case=f"moved-settings notice quiet on {app}"), \
                built(app) as scr:
            before = console_text(scr._console)
            for payload in payloads:
                scr._warn_about_moved_settings(payload)
            assert console_text(scr._console) == before


def test_src_field_drives_the_empty_state_banner_and_column_pickers(
        qtbot, subtests, monkeypatch):
    """The banner shows while ``src`` is a placeholder and hides on a real
    path; it is absent when src is preset, there is no src widget, or the
    model cannot be read; a non-text src widget reads as unset. Column
    fields get one SQL picker bound to the LIVE src; other fields get none.
    """
    from spacr.qt.screens import settings_model as sm
    from spacr.qt.widgets.column_picker import ColumnPickerButton

    with subtests.test(case="banner follows the field"), built("mask") as scr:
        card = scr._empty_state_card
        assert card is not None and card.objectName() == "EmptyStateBanner"
        assert not card.isHidden()
        src = scr._settings_model._widgets["src"]
        src.setText("/path/to/src")
        assert not card.isHidden()
        src.setText("/data/real_plate")
        assert card.isHidden()

    with subtests.test(case="src path is stripped and empty when gone"), \
            built("mask") as scr:
        card, scr._empty_state_card = scr._empty_state_card, None
        scr._maybe_hide_empty_state("/data/x")
        assert not card.isHidden(), "the detached card must be left alone"
        scr._settings_model._widgets["src"].setText("  /data/plate7  ")
        assert scr._settings_src_path() == "/data/plate7"
        scr._settings_model._widgets["src"] = QCheckBox()
        rebuilt = scr._build_empty_state_banner()
        assert rebuilt is not None and rebuilt.objectName() == "EmptyStateBanner"
        assert scr._settings_src_path() == ""
        scr._settings_model._widgets.pop("src")
        assert scr._settings_src_path() == ""
        scr._settings_model._widgets = {}
        assert scr._build_empty_state_banner() is None
        scr._settings_model = object()
        assert scr._build_empty_state_banner() is None

    with subtests.test(case="banner absent when src is preset"), \
            pytest.MonkeyPatch.context() as mp:
        real = sm.resolve_default_settings

        def _with_src(app_key):
            s = real(app_key)
            s["src"] = "/data/already_set"
            return s

        mp.setattr(sm, "resolve_default_settings", _with_src)
        with built("mask") as scr:
            assert scr._empty_state_card is None
            assert scr._settings_src_path() == "/data/already_set"

    with subtests.test(case="column picker bound to live src"), \
            built("ml_analyze") as scr:
        assert "annotation_column" in COLUMN_TABLES
        field = scr._settings_model._widgets["annotation_column"]
        buttons = [b for b in scr.findChildren(ColumnPickerButton)
                   if b.field is field]
        assert len(buttons) == 1
        assert buttons[0].table == COLUMN_TABLES["annotation_column"] == \
            "png_list"
        scr._settings_model.set_value_for_key("src", "/data/plate9")
        assert buttons[0].db_path() == "/data/plate9"

    with subtests.test(case="no picker on non-column fields"), \
            built("classify") as scr:
        field = scr._settings_model._widgets["src"]
        assert [b for b in scr.findChildren(ColumnPickerButton)
                if b.field is field] == []
        holder = QLabel(parent=scr)
        scr._attach_column_picker(None, holder)
        scr._attach_column_picker("batch_size", holder)
        assert holder.findChildren(ColumnPickerButton) == []


def _tiff(root: Path) -> Path:
    import tifffile
    sub = root / "plate1" / "well"
    sub.mkdir(parents=True)
    img = (np.arange(32 * 32, dtype=np.uint16) % 500).reshape(32, 32)
    p = sub / "plate1_A01_T01F01L01A01Z01C00.tif"
    tifffile.imwrite(str(p), img)
    return p


def test_live_preview_autoloads_the_first_tile_a_src_points_at(
        qtbot, subtests, tmp_path):
    """Typing a folder into src (debounced) or handing a direct image path
    loads that tile; placeholders, missing folders, non-images and imageless
    folders load nothing; screens without a preview ignore it and wire no
    debounce timer.
    """
    with subtests.test(case="typed src folder"), built("mask") as scr:
        root = tmp_path / "typed"
        tif = _tiff(root)
        assert scr._live_preview is not None
        scr._settings_model._widgets["src"].setText(str(root))
        qtbot.waitUntil(lambda: scr._live_preview._image_path is not None,
                        timeout=5000)
        assert scr._live_preview._image_path == tif
        assert scr._live_preview._image is not None

    with subtests.test(case="direct image path"), built("mask") as scr:
        tif = _tiff(tmp_path / "direct")
        scr._autoload_live_preview(str(tif))
        qtbot.waitUntil(lambda: scr._live_preview._image_path is not None,
                        timeout=5000)
        assert scr._live_preview._image_path == tif

    (tmp_path / "empty").mkdir()
    ignored = ["", "   ", "path", "/path/to/src", "/path",
               str(tmp_path / "nope"), __file__, str(tmp_path / "empty")]
    with built("mask") as scr:
        for value in ignored:
            with subtests.test(case=f"ignored {value!r}"):
                scr._autoload_live_preview(value)
                qtbot.waitUntil(lambda: not scr._live_preview._image_loaders,
                                timeout=5000)
                assert scr._live_preview._image_path is None

    with subtests.test(case="no preview on map_barcodes"), \
            built("map_barcodes") as scr:
        _tiff(tmp_path / "nopreview")
        assert scr._live_preview is None
        scr._autoload_live_preview(str(tmp_path / "nopreview"))
        assert not hasattr(scr, "_live_src_timer")
