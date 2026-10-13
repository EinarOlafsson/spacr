"""Which object settings a form leaves out when the run names no plane.

``SettingsWidgets._keys_of_objects_the_run_has_no_channel_for`` decides,
when a panel is built, which nucleus and pathogen settings belong to an
object the run does not have: the object's switch (``*_channel`` or, for
Measure, ``*_mask_dim``) names no plane. Cell is never gated, an object
the module has no switch for is not this gate's to decide, and the switch
itself always stays so the object can be turned on.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import SettingsWidgets  # noqa: E402

gate = SettingsWidgets._keys_of_objects_the_run_has_no_channel_for


def test_an_object_without_a_plane_loses_its_settings_but_not_its_switch():
    settings = {"cell_channel": 0, "cell_diameter": 30,
                "nucleus_channel": None, "nucleus_diameter": 12,
                "nucleus_min_size": 5, "pathogen_channel": 2,
                "pathogen_diameter": 8}
    assert gate(settings) == {"nucleus_diameter", "nucleus_min_size"}


def test_measures_mask_dims_switch_objects_and_answers_may_come_separately():
    settings = {"nucleus_mask_dim": None, "nucleus_min_size": 5,
                "pathogen_mask_dim": None, "pathogen_min_size": 3}
    assert gate(settings) == {"nucleus_min_size", "pathogen_min_size"}
    deciding = {"nucleus_mask_dim": 3, "pathogen_mask_dim": "none"}
    assert gate(settings, deciding) == {"pathogen_min_size"}


def test_a_module_without_an_objects_switch_keeps_its_settings():
    assert gate({"nucleus_diameter": 12, "plot": True}) == set()


def test_a_numbered_background_switch_belongs_to_its_slot():
    from spacr.organelle_types import organelle_role
    from spacr.qt.screens.settings_model import object_of_setting

    assert object_of_setting("remove_background_organelle_7") == \
        organelle_role(7)
