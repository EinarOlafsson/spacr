"""The shipped media-reset handler releases only the discarded seek guard."""
import json
from pathlib import Path
import re
import shutil
import subprocess

import pytest

PLAYER = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials/app_v2.js'


@pytest.mark.parametrize('pending', [False, True])
def test_source_replacement_cannot_leave_a_cancelled_seek_pending(pending):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node executes the actual shipped event handler')
    match = re.search(r'elements\.video\.addEventListener\("emptied", \(\) => \{(.*?)\n\}\);',
                      PLAYER.read_text(), re.S)
    assert match, 'A replaced source may never dispatch seeked'
    script = f'let videoClockCorrectionPending = {json.dumps(pending)};\n' + match[1]
    script += '\nconsole.log(JSON.stringify(videoClockCorrectionPending));'
    assert json.loads(subprocess.check_output([node, '-e', script], text=True)) is False
