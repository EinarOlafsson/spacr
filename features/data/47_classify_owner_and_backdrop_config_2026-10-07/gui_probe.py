import os
import subprocess
import sys

for args in [
    [sys.executable, 'tools/can_this_display_be_measured.py'],
    [sys.executable, '-m', 'pytest', '-q', '-p', 'no:randomly', 'tests/qt/test_one_backdrop_for_the_window.py::test_the_home_screen_is_not_black_on_a_real_display'],
]:
    print('SOFTWARE_X_DISPLAY_COMMAND', repr(args), flush=True)
    result = subprocess.run(args, env=os.environ.copy())
    print('RETURN_CODE', result.returncode, flush=True)
    if result.returncode:
        sys.exit(result.returncode)
