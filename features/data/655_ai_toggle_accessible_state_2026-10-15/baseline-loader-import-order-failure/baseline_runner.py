import importlib.util
import sys
from pathlib import Path
import spacr.qt.widgets

path=Path(__file__).with_name('baseline_ai_toggle_label.py')
name='spacr.qt.widgets.ai_toggle_label'
assert name not in sys.modules
spec=importlib.util.spec_from_file_location(name,path)
module=importlib.util.module_from_spec(spec)
sys.modules[name]=module
spec.loader.exec_module(module)
import pytest
raise SystemExit(pytest.main(['-q','tests/qt/test_655_ai_toggle_accessible_state.py','-k','exposes_checkbox or accessibility_action_uses or real_changes_emit']))
