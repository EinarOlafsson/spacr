"""Exercise Trackastra's settings and preview routing without model loading."""

import json

from PySide6.QtWidgets import QApplication

from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
from spacr.settings_spec import convert_settings_dict_for_gui


app = QApplication([])
settings = {
    "timelapse_mode": "trackastra",
    "trackastra_model": "audit-model",
    "trackastra_linking": "ilp",
    "timelapse_objects": ["cell"],
}
ui = convert_settings_dict_for_gui(settings)
assert ui["timelapse_mode"][0] == "combo"
assert "trackastra" in ui["timelapse_mode"][1]
assert ui["trackastra_model"] == ("entry", None, "audit-model")
assert ui["trackastra_linking"] == ("entry", None, "ilp")
panel = TimelapsePreviewPanel()
panel.apply_settings(settings)
params = panel._track_params()
assert params["mode"] == "trackastra"
assert params["trackastra_model"] == "audit-model"
assert params["trackastra_linking"] == "ilp"
assert "trackastra" in [panel._mode_box.itemText(index)
                         for index in range(panel._mode_box.count())]
print(json.dumps({"ui_mode": ui["timelapse_mode"],
                  "ui_model": ui["trackastra_model"],
                  "ui_linking": ui["trackastra_linking"],
                  "preview_track_params": params}, sort_keys=True))
panel.close()
