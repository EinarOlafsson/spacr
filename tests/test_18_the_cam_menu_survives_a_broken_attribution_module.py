"""Item 288: the Activation Maps form's ``cam_type`` menu never goes empty.

``spacr.settings_spec._cam_type_choices`` asks the loaded attribution module
for its methods and falls back to the frozen list when that module is not
loaded -- so the settings panel never imports torch. The third case is a
loaded module that cannot answer (half-imported, or broken by an optional
backend): the menu must still offer every method, from the frozen list,
rather than raise into the settings panel.
"""
from __future__ import annotations

import sys
import types

from spacr import settings_spec


def test_a_module_that_cannot_answer_leaves_the_frozen_menu(monkeypatch):
    broken = types.ModuleType("spacr.attribution")

    def cam_type_choices():
        raise RuntimeError("attribution half-imported")

    broken.cam_type_choices = cam_type_choices
    monkeypatch.setitem(sys.modules, "spacr.attribution", broken)
    assert settings_spec._cam_type_choices() == \
        list(settings_spec._CAM_TYPE_CHOICES)


def test_no_module_loaded_uses_the_frozen_menu(monkeypatch):
    monkeypatch.delitem(sys.modules, "spacr.attribution", raising=False)
    choices = settings_spec._cam_type_choices()
    assert choices == list(settings_spec._CAM_TYPE_CHOICES)
    assert "hirescam" in choices and "chefer" in choices
    assert "spacr.attribution" not in sys.modules
