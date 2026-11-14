"""Table-driven test of ``spacr.object_roles.organelle_settings_view``.

The classical organelle segmenter reads legacy ``organelle_*`` keys; the
view translates one slot's ``<role>_*`` keys (and its recorded Cellpose
channel) into those names on a copy, refuses a role that is not a slot,
and never mutates the caller's settings.
"""
from __future__ import annotations

import pytest

from spacr import object_roles


def test_organelle_settings_view_translates_one_slot_onto_legacy_keys(subtests):
    """Exact view per slot/settings pair; non-slots are refused by name."""
    cases = [
        ("legacy slot is a plain copy",
         {"organelle_channel": 1, "unrelated": "x"}, "organelle",
         {"organelle_channel": 1, "unrelated": "x"}),
        ("recorded cellpose channel reaches the legacy key",
         {"organellec_channel": 2, "organellec_diameter": 11,
          "cellpose_organellec_channel": 5, "cellpose_organelle_channel": 0},
         "organellec",
         {"organellec_channel": 2, "organellec_diameter": 11,
          "cellpose_organellec_channel": 5, "cellpose_organelle_channel": 5,
          "organelle_channel": 2, "organelle_diameter": 11}),
        ("unrecorded slot channel leaves the legacy channel alone",
         {"organellec_channel": 2, "cellpose_organelle_channel": 0},
         "organellec",
         {"organellec_channel": 2, "cellpose_organelle_channel": 0,
          "organelle_channel": 2}),
        ("a recorded None does not overwrite the legacy channel",
         {"organelleb_channel": 3, "cellpose_organelleb_channel": None,
          "cellpose_organelle_channel": 1},
         "organelleb",
         {"organelleb_channel": 3, "cellpose_organelleb_channel": None,
          "cellpose_organelle_channel": 1, "organelle_channel": 3}),
        ("another slot's keys are not translated",
         {"organelleb_channel": 3, "organellec_channel": 4}, "organellec",
         {"organelleb_channel": 3, "organellec_channel": 4,
          "organelle_channel": 4}),
    ]
    for case_id, settings, role, expected in cases:
        with subtests.test(msg=case_id):
            before = dict(settings)
            view = object_roles.organelle_settings_view(settings, role)
            assert view == expected
            assert view is not settings
            assert settings == before

    for role in ("cell", "nucleus", "cytoplasm", ""):
        with subtests.test(msg=f"{role!r} is not a slot"):
            with pytest.raises(ValueError,
                               match=f"^unknown organelle role {role!r}$"):
                object_roles.organelle_settings_view({"cell_channel": 0},
                                                     role)
