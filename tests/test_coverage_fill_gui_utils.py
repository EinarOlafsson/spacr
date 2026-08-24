"""What the settings panel makes of a settings dict.

WHAT THIS FILE USED TO BE: coverage-fill for `spacr.gui_utils`, the
Tkinter interface's helper module. That interface is gone and so is the
module, and with it `parse_list` and `convert_to_number`, which existed
to read values out of Tk entry widgets and have no definition anywhere in
the tree any more.

`convert_settings_dict_for_gui` outlived it: `gui_utils` only re-exported
it, it lives in `spacr.settings_spec`, and it is what the Qt settings
model reads to decide whether a setting is a combo, a check or an entry.
So that test stays, pointed at the real module.
"""
from __future__ import annotations

from spacr import settings_spec as GU


def test_convert_settings_dict_for_gui():
    settings = {
        "metadata_type": "cellvoyager",   # special case → combo
        "verbose": True,                  # bool → check
        "diameter": 30,                   # int → entry
        "src": "/data",                   # str → entry
        "cov_type": None,                 # None (special) → combo
        "channels_list": [0, 1, 2],       # list → entry (str)
    }
    out = GU.convert_settings_dict_for_gui(settings)
    assert out["metadata_type"][0] == "combo"
    assert out["verbose"] == ("check", None, True)
    assert out["diameter"] == ("entry", None, 30)
    assert out["src"] == ("entry", None, "/data")
    assert out["channels_list"][0] == "entry"
    assert out["cov_type"][0] == "combo"
