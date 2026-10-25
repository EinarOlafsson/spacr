"""Source-folder lookup survives deferred controls and deleted Qt fields."""
from pathlib import Path

import pytest
import shiboken6
from PySide6.QtWidgets import QLineEdit

from spacr.qt.screens.settings_model import SettingsWidgets


pytestmark = pytest.mark.qt


def test_a_deleted_source_field_falls_back_to_the_imported_folder(qapp):
    model = SettingsWidgets("measure")
    model._defaults["src"] = ["/data/imported", "/data/second"]
    field = QLineEdit("/data/edited")
    model._widgets = {"src": field}

    assert model._current_source_folder() == "/data/edited"

    shiboken6.delete(field)

    assert model._current_setting("src") == ["/data/imported", "/data/second"]
    assert model._current_source_folder() == "/data/imported"


@pytest.mark.parametrize("value, expected", [
    ([], ""),
    ((), ""),
    (None, ""),
    (("/data/first", "/data/second"), "/data/first"),
    (Path("/data/first"), str(Path("/data/first"))),
])
def test_a_source_without_a_built_field_reads_the_imported_value(
        qapp, value, expected):
    model = SettingsWidgets("measure")
    model._widgets = {}
    model._defaults["src"] = value

    assert model._current_setting("src") == value
    assert model._current_source_folder() == expected
