from pathlib import Path
import os
import subprocess
import sys

stage = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005/tutorial-standard-install-neutral-current-r1/current-ui-reference-r2')
repo = Path('/media/carruthers/mnt3/codex/spacr-worktrees/docs-completion-20261005')
stage.mkdir(exist_ok=False)
for name in ('app-state', 'example_data', 'cache', 'tmp', 'logs', 'mpl'):
    (stage / name).mkdir()
environment = dict(os.environ)
environment.update(CUDA_VISIBLE_DEVICES='', QT_QPA_PLATFORM='offscreen', OMP_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2', SPACR_TUTORIAL_CACHE_ISOLATED='1',
                   XDG_CACHE_HOME='/tmp/spacr-install-ui/cache', SPACR_HOME='/tmp/spacr-install-ui/app-state',
                   SPACR_LOG_DIR='/tmp/spacr-install-ui/logs', MPLCONFIGDIR='/tmp/spacr-install-ui/mpl',
                   TMPDIR='/tmp/spacr-install-ui/tmp')
code = '''import sys
from pathlib import Path
sys.meta_path[:] = [finder for finder in sys.meta_path if not getattr(finder, '__module__', '').startswith('__editable__')]
sys.path.insert(0, '/tmp/spacr-code')
sys.path.insert(0, '/tmp/spacr-code/tools/tutorials')
import capture_refresh
sys.argv = ['capture_refresh.py', '--module', 'home', '--stage', '/tmp/spacr-install-ui', '--openings', '--openings-set', 'home', '--capture-name', 'current_install_openings_r2', '--platform', 'offscreen']
raise SystemExit(capture_refresh.main())
'''
subprocess.run(['bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
                '--dev-bind', '/dev', '/dev', '--tmpfs', '/tmp', '--ro-bind', str(repo), '/tmp/spacr-code',
                '--bind', str(stage), '/tmp/spacr-install-ui',
                '--bind', str(stage / 'app-state'), str(Path.home() / '.spacr'),
                '--bind', str(stage / 'example_data'), str(Path.home() / '.cache/spacr/example_data'),
                '--chdir', '/tmp/spacr-code', '--', sys.executable, '-c', code], env=environment,
               check=True, timeout=180)
print('PASS: current source Home/Performance native reference recording; no released-application claim.', flush=True)
