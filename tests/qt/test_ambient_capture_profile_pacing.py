"""Exercise the actual private pacing API without constructing GPU widgets."""
import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from spacr.qt.widgets import ambient as a

ROOT = Path(__file__).resolve().parents[2]

class Field:
    _rate = a.AmbientWidget._rate
    _apply_rate = a.AmbientWidget._apply_rate
    def __init__(self, fps=60, held=True, run=False):
        self._fps, self._hold_paced, self._run_paced = fps, held, run
        self._exact_fps = None
        self.timer_rates, self.producer_rates = [], []
        self._timer = SimpleNamespace(setInterval=self.timer_rates.append)
        self._producer_box = [SimpleNamespace(set_fps=self.producer_rates.append)]

@pytest.fixture(autouse=True)
def reset(monkeypatch):
    monkeypatch.setattr(a, '_CAPTURE_HELD_FPS', None)
    monkeypatch.setattr(a, '_live_ambient_widgets', lambda: [])

def test_default_remains_twenty():
    assert Field()._rate() == 20

def test_existing_later_and_reset(monkeypatch):
    old = Field()
    monkeypatch.setattr(a, '_live_ambient_widgets', lambda: [old])
    a._set_capture_held_fps(30)
    assert old._rate() == 30
    assert old.timer_rates == [33]
    assert old.producer_rates == [30]
    assert Field()._rate() == 30
    a._set_capture_held_fps(None)
    assert old._rate() == Field()._rate() == 20
    assert old.producer_rates == [30,20]

@pytest.mark.parametrize('fps,held,run', [(15,True,False),(60,False,False),(60,True,True)])
def test_other_caps_remain_authoritative(fps,held,run):
    a._set_capture_held_fps(30)
    field = Field(fps,held,run)
    expected = min(fps,a._RUN_FPS) if run else fps
    if held: expected = min(expected,30)
    assert field._rate() == expected
    field._apply_rate()
    assert field.producer_rates == [expected]

@pytest.mark.parametrize('bad', [True,30.0,'30',0,-1,a.MAX_FPS+1])
def test_invalid_request_does_not_change_state(bad):
    with pytest.raises(ValueError): a._set_capture_held_fps(bad)
    assert a._CAPTURE_HELD_FPS is None

def test_capture_profile_request_precedes_application_start():
    path = ROOT/'tools/tutorials/capture_host_mode.py'
    tree = ast.parse(path.read_text())
    profile = next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='CAPTURE_PROFILE' for t in n.targets))
    entries = {k.value:v for k,v in zip(profile.keys,profile.values)}
    assert ast.literal_eval(entries['field_held_fps']) == 30
    child = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_child')
    request = [n for n in ast.walk(child) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='_set_capture_held_fps']
    launch = [n for n in ast.walk(child) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and ast.unparse(n.func)=='spacr.qt.run']
    assert len(request)==len(launch)==1 and request[0].lineno < launch[0].lineno
    assert ast.unparse(request[0].args[0]) == "CAPTURE_PROFILE['field_held_fps']"
    prepare = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_profile')
    actual = next(n.value for n in prepare.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='actual' for t in n.targets))
    keys = [k.value for k in actual.keys]
    assert 'field_held_fps' not in keys
    assert 'requested_field_held_fps' in keys
    skip = next(n.value for n in prepare.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='skip' for t in n.targets))
    assert ast.literal_eval(skip) == {'theme', 'field_held_fps'}
