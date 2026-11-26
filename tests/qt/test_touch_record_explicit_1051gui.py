"""Item 1051 (GUI): the form records which fields the user edited, so a field
edited to its default still beats a preset, and untouched fields get it."""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

KEY = "ops_pixel_quality"


def _form(qtbot):
    from PySide6.QtWidgets import QWidget

    from spacr.qt.screens.settings_model import SettingsWidgets

    holder = QWidget()
    qtbot.addWidget(holder)
    form = SettingsWidgets("ops", holder)
    form.build_sections()
    form._enable_commit_observation(lambda key: None)
    return form


def _resolved(form):
    from spacr import ops_decode

    return ops_decode.resolve_settings(form.with_explicit_record(form.collect()))


def test_an_untouched_field_gets_the_preset(qtbot):
    form = _form(qtbot)
    assert form.touched_keys() == []
    assert _resolved(form)[KEY] == 0.1


def test_a_field_edited_to_the_default_is_explicit_and_wins(qtbot):
    form = _form(qtbot)
    widget = form._widgets[KEY]
    default = form._defaults[KEY]
    widget.setValue(0.5)
    widget.setValue(float(default))
    assert KEY in form.touched_keys()
    assert _resolved(form)[KEY] == float(default)


def test_a_programmatic_write_is_not_an_edit(qtbot):
    form = _form(qtbot)
    assert form.set_value_for_key(KEY, 0.4)
    assert form.touched_keys() == []


def test_a_loaded_file_keeps_its_record(qtbot):
    from spacr.settings import LEGACY_EXPLICIT

    form = _form(qtbot)
    form.note_loaded_record([KEY])
    assert form.touched_keys() == []
    assert form.explicit_record() == [KEY]
    assert _resolved(form)[KEY] == float(form._defaults[KEY])
    form.note_loaded_record(LEGACY_EXPLICIT)
    assert form.explicit_record() == LEGACY_EXPLICIT
    assert _resolved(form)[KEY] == 0.1


def test_a_reset_to_defaults_clears_touches(qtbot):
    form = _form(qtbot)
    form._widgets[KEY].setValue(0.5)
    form.note_loaded_record([KEY])
    form._widgets[KEY].setValue(0.6)
    assert KEY in form.touched_keys()
    form.reset_to_defaults()
    assert form.touched_keys() == []
    assert form.explicit_record() == []
    assert form.collect()[KEY] == float(form._defaults[KEY])
    assert _resolved(form)[KEY] == 0.1
