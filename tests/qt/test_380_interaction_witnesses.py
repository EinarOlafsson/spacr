"""Interaction receipts must witness real input, including every repeat."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def harness():
    """Import the executable diagnostic without invoking its command line."""
    path = Path(__file__).resolve().parents[2] / "tools/perf_paint.py"
    spec = importlib.util.spec_from_file_location("perf_380_witness", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("state", ["hidden", "disabled", "readonly", "full"])
def test_text_input_rejects_unusable_or_unchanged_fields(qtbot, qapp, harness, state):
    """A fast no-op must fail rather than look like successful typing."""
    from PySide6.QtWidgets import QLineEdit

    field = QLineEdit()
    qtbot.addWidget(field)
    field.show()
    if state == "hidden":
        field.hide()
    elif state == "disabled":
        field.setEnabled(False)
    elif state == "readonly":
        field.setReadOnly(True)
    else:
        field.setMaxLength(1)
        field.setText("x")
        field.deselect()
    qapp.processEvents()
    with pytest.raises(RuntimeError):
        harness._type_character(field, qapp)


def test_text_input_has_a_real_witness_without_recording_text(qtbot, qapp, harness):
    """Receipts expose lengths, never a path entered in the field."""
    from PySide6.QtWidgets import QLineEdit

    field = QLineEdit("private-source-path")
    qtbot.addWidget(field)
    field.show()
    qapp.processEvents()
    row = harness._type_character(field, qapp)
    assert row == {"text_length_before": 19, "text_length_after": 20}
    assert field.text().endswith("a")


def test_scroll_repeats_move_even_from_the_bottom(qtbot, qapp, harness):
    """Repeated scrolling reverses instead of measuring a clamped endpoint."""
    from PySide6.QtWidgets import QScrollArea, QWidget

    area = QScrollArea()
    qtbot.addWidget(area)
    content = QWidget()
    content.resize(100, 2000)
    area.setWidget(content)
    area.resize(180, 180)
    area.show()
    qapp.processEvents()
    bar = area.verticalScrollBar()
    bar.setValue(bar.maximum())
    rows = [harness._scroll_settings(area, qapp) for _ in range(20)]
    assert all(row["scroll_value_before"] != row["scroll_value_after"] for row in rows)
    assert any(row["scroll_value_after"] > row["scroll_value_before"] for row in rows)
    assert any(row["scroll_value_after"] < row["scroll_value_before"] for row in rows)


def test_short_scroll_area_is_rejected(qtbot, qapp, harness):
    """No scrollable content cannot be reported as instant scrolling."""
    from PySide6.QtWidgets import QScrollArea, QWidget

    area = QScrollArea()
    qtbot.addWidget(area)
    area.setWidget(QWidget())
    area.show()
    qapp.processEvents()
    with pytest.raises(RuntimeError):
        harness._scroll_settings(area, qapp)


@pytest.mark.parametrize("kind", ["section", "collapsible"])
def test_section_input_changes_actual_state(qtbot, qapp, harness, kind):
    """Both actual section implementations are driven through their headers."""
    from PySide6.QtWidgets import QWidget

    from spacr.qt.widgets.collapsible_section import CollapsibleSection
    from spacr.qt.widgets.section import Section

    section = (Section("Settings") if kind == "section"
               else CollapsibleSection("Settings", QWidget(), expanded=False))
    qtbot.addWidget(section)
    section.show()
    qapp.processEvents()
    assert harness._toggle_section(section, qapp) == {
        "expanded_before": False, "expanded_after": True}
    assert harness._toggle_section(section, qapp) == {
        "expanded_before": True, "expanded_after": False}


def test_failed_section_input_is_not_a_successful_timing(qtbot, qapp, harness, monkeypatch):
    """A disconnected heading must fail the instrument's state witness."""
    from spacr.qt.widgets.section import Section

    section = Section("Settings")
    qtbot.addWidget(section)
    section.show()
    qapp.processEvents()
    monkeypatch.setattr(section, "is_expanded", lambda: False)
    with pytest.raises(RuntimeError, match="did not change"):
        harness._toggle_section(section, qapp)


def test_actual_mask_screen_has_witnessed_trials(qapp, harness):
    """Run the bounded real-screen acceptance path without data or inference."""
    with harness._settings_elsewhere():
        rows = harness.measure_interaction("mask")
    assert len(rows) == 4
    assert all(row["status"] == "observed" for row in rows)
    for row in rows:
        assert "frames_dropped" not in row
        assert "physical presented/dropped frames unknown" in row["frame_scope"]
        assert len(row["samples"]) == 3
        assert all(sample["input_dispatch_and_events_ms"] is not None
                   for sample in row["samples"])
        if row["action"] == "expand a section":
            assert all(sample["expanded_after"] for sample in row["samples"])
        elif row["action"] == "collapse a section":
            assert all(not sample["expanded_after"] for sample in row["samples"])
