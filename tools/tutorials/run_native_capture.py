"""Run the normal recorder on a private X11 display and fresh neutral profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--xvfb', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('capture_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stage = args.stage.resolve()
    root = Path(__file__).resolve().parents[2]
    if not args.child:
        stage.mkdir(parents=True, exist_ok=False)
    for name in ('cache', 'example_data', 'app-state', 'logs', 'mpl', 'tmp', 'profile'):
        (stage / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='', SPACR_TUTORIAL_CACHE_ISOLATED='1',
                      SPACR_HOME=str(stage/'app-state'), SPACR_LOG_DIR=str(stage/'logs'),
                      XDG_CACHE_HOME=str(stage/'cache'), SPACR_NEWS_CACHE=str(stage/'cache/news'),
                      MPLCONFIGDIR=str(stage/'mpl'), TMPDIR=str(stage/'tmp'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    capture_args = args.capture_args[1:] if args.capture_args[:1] == ['--'] else args.capture_args
    if not args.child:
        neutral = '/tmp/spacr-tutorial-current'
        subprocess.run([
            'bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
            '--dev-bind', '/dev', '/dev', '--tmpfs', '/tmp', '--bind', str(stage), neutral,
            '--bind', str(stage/'app-state'), str(Path.home()/'.spacr'),
            '--bind', str(stage/'example_data'), str(Path.home()/'.cache/spacr/example_data'),
            '--setenv', 'HOME', neutral + '/profile',
            '--', sys.executable, str(Path(__file__).resolve()), '--stage', neutral,
            '--xvfb', str(args.xvfb.resolve()), '--child', '--', *capture_args,
        ], check=True)
        return 0
    display = ':' + str(200 + os.getpid() % 2000)
    with (stage/'xvfb.log').open('wb') as log:
        server = subprocess.Popen([str(args.xvfb), display, '-screen', '0', '3840x2160x24',
                                   '-ac', '-nolisten', 'tcp'], stdout=log, stderr=log)
        try:
            deadline = time.monotonic()+10
            while not Path('/tmp/.X11-unix/X'+display[1:]).exists():
                if server.poll() is not None or time.monotonic()>deadline:
                    raise RuntimeError('Private X11 server did not start')
                time.sleep(.02)
            os.environ.update(DISPLAY=display, QT_QPA_PLATFORM='xcb')
            sys.meta_path[:] = [finder for finder in sys.meta_path
                               if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
            sys.path[:0] = [str(root), str(root/'tools/tutorials')]
            import spacr
            assert Path(spacr.__file__).resolve().parent == root/'spacr'
            sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for pattern in ('spacr/qt/**/*.py', 'tools/tutorials/capture*.py')
                       for p in root.glob(pattern)}
            (stage/'exact-source.json').write_text(json.dumps({
                'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                'sha256': sources, 'fresh': True, 'private_profile': True,
                'surface': 'native Qt application on isolated X11; no native compositor claim',
            }, indent=2)+'\n')
            import capture_refresh
            sys.argv = ['capture_refresh', '--stage', str(stage), '--platform', 'xcb', *capture_args]
            result = capture_refresh.main()
            if any(hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest
                   for name,digest in sources.items()):
                raise RuntimeError('Application or recorder source changed during capture')
            return result
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == '__main__':
    raise SystemExit(main())
