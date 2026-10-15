"""The new reference choice is discoverable at its real planner and flow page."""
import json
from pathlib import Path


def test_reference_plate_documents_actual_consumer_and_flow():
    from spacr.qt.screens.settings_flow_index import SETTINGS_WITH_A_FLOW_SECTION

    root = Path(__file__).resolve().parents[1]
    key = 'intensity_calibration_reference_plate'
    consumers = json.loads((root / 'docs/setting_consumers.json').read_text())['consumers'][key]
    assert any(row['module'] == 'spacr.intensity_rescale' and
               row['qualname'] == '_build_calibration_plan' for row in consumers)
    assert key in SETTINGS_WITH_A_FLOW_SECTION
    page = (root / 'docs/source/_generated/settings_flow.rst').read_text()
    anchor = '.. _setting-flow-' + key + ':'
    section = page.split(anchor, 1)[1].split('.. _setting-flow-', 1)[0]
    assert 'first plate by name' in section and '_build_calibration_plan' in section
