"""Exact production wrapping must not reject a genuinely shown fold tooltip."""
import hashlib
import importlib.util
from pathlib import Path

import pytest

from spacr.qt.tooltip_policy import _wrapped_tooltip

SOURCE = Path(__file__).resolve().parents[1] / 'capture_classify_overview.py'
spec = importlib.util.spec_from_file_location('private_capture_classify_overview', SOURCE)
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
matches = capture._actual_tooltip_matches
import spacr.qt.tooltip_policy as policy
print('ACTUAL_CAPTURE_IMPORT', capture.__file__, hashlib.sha256(SOURCE.read_bytes()).hexdigest())
print('ACTUAL_PRODUCTION_IMPORT', policy.__file__, hashlib.sha256(Path(policy.__file__).read_bytes()).hexdigest())

RAW = ('Classifier Evaluation\nInspect held-out predictions, grouped or nested '
       'cross-validation, calibration, confusion matrices, per-plate performance and leakage checks.')


def test_actual_production_wrapped_full_text_is_accepted():
    displayed = _wrapped_tooltip(RAW)
    assert displayed != RAW
    assert matches(True, displayed, RAW)


def test_short_production_tooltip_is_accepted():
    assert matches(True, _wrapped_tooltip('Activation'), 'Activation')


@pytest.mark.parametrize('visible,displayed,authored', [
    (False, _wrapped_tooltip(RAW), RAW),
    (True, '', RAW),
    (True, _wrapped_tooltip(RAW), ''),
    (True, _wrapped_tooltip('Explain CV: inspect another model'), RAW),
    (True, _wrapped_tooltip(RAW.replace('held-out', 'training')), RAW),
    (True, RAW, RAW),
])
def test_hidden_empty_stale_changed_or_unwrapped_tooltip_is_rejected(visible, displayed, authored):
    assert not matches(visible, displayed, authored)
