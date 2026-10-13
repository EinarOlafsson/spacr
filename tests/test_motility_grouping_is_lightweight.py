"""Opening a motility plate must not import the model stack to parse filenames."""

import subprocess
import sys


def test_grouping_names_uses_no_analysis_or_model_imports(tmp_path):
    for name in ('plate1_A03_002_t010.npy', 'plate1_A03_002_t002.npy',
                 'plate1_B04_001_t001.npy', 'readme.txt'):
        (tmp_path / name).touch()
    script = """
import sys
from spacr.qt.widgets.motility_preview import group_merged_files
blocked = ('spacr.timelapse', 'spacr.utils', 'torch', 'matplotlib.pyplot')
assert not [name for name in blocked if name in sys.modules]
groups = group_merged_files(sys.argv[1])
assert set(groups) == {('plate1', 'A03', '002')}
assert [row['timeID'] for row in groups[('plate1', 'A03', '002')]] == [2, 10]
assert not [name for name in blocked if name in sys.modules]
"""
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path)],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
