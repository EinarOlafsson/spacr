import importlib.abc
import importlib.util
import sys
from pathlib import Path

path=Path(__file__).with_name('baseline_ai_toggle_label.py')
name='spacr.qt.widgets.ai_toggle_label'
assert name not in sys.modules
class BaselineFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname==name:
            return importlib.util.spec_from_file_location(fullname,globals()['path'])
        return None
finder=BaselineFinder()
sys.meta_path.insert(0,finder)
import pytest
result=pytest.main(['-q','tests/qt/test_655_ai_toggle_accessible_state.py','-k','exposes_checkbox or accessibility_action_uses or real_changes_emit'])
assert Path(sys.modules[name].__file__).resolve()==path.resolve()
raise SystemExit(result)
