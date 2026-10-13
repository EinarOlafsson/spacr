"""Standalone guide commands load captions owned by the first-run screen."""
from pathlib import Path
import os
import subprocess
import sys


def test_standalone_guide_lookup_loads_the_animation_caption():
    root = Path(__file__).resolve().parents[1]
    probe = '''
from pathlib import Path
import sys
root = Path(sys.argv[1]).resolve()
sys.meta_path = [finder for finder in sys.meta_path
    if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / 'tools'))
import spacr.qt.i18n as i18n
assert Path(i18n.__file__).resolve().is_relative_to(root)
assert i18n._exact_translation('Animation', 'hi') is None
import build_guide_i18n as guide
assert guide.runtime_ui_name('Animation', 'hi') == 'एनिमेशन'
assert guide.runtime_ui_name('Animation', 'is') == 'Hreyfimynd'
assert guide.runtime_ui_name('Animation', 'sv') == 'Animation'
'''
    result = subprocess.run([sys.executable, '-c', probe, str(root)],
        cwd=root, capture_output=True, text=True, timeout=60,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'QT_QPA_PLATFORM': 'offscreen'})
    assert result.returncode == 0, result.stderr
