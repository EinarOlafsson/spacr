"""Run the normal recorder on a private X11 display and fresh neutral profile."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

NEUTRAL = '/tmp/spacr-tutorial-current'


def copy_inputs(source, stage):
    """Copy a prepared private input tree into the new stage and verify every byte.

    File and folder modification times are kept, so an input that depends on
    age (for example old run folders for Storage -> Prune now) stays real.
    """
    source = Path(source).resolve()
    if not source.is_dir():
        raise ValueError(f'Prepared inputs are not a directory: {source}')
    manifest = {}
    for path in sorted(p for p in source.rglob('*') if p.is_file()):
        relative = path.relative_to(source)
        if relative.parts[0] in ('captures', 'exact-source.json', 'inputs-manifest.json'):
            raise ValueError(f'Prepared inputs cannot provide capture evidence: {relative}')
        target = stage / relative
        if target.exists():
            raise FileExistsError(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Prepared input changed while copying: {relative}')
        manifest[str(relative)] = digest
    if not manifest:
        raise ValueError('Prepared inputs contain no files')
    for folder in sorted((p for p in source.rglob('*') if p.is_dir()),
                         key=lambda p: len(p.parts), reverse=True):
        shutil.copystat(folder, stage / folder.relative_to(source))
    (stage / 'inputs-manifest.json').write_text(json.dumps(
        {'source': str(source), 'sha256': manifest}, indent=2) + '\n', encoding="utf-8")
    return manifest


def read_only_binds(pairs):
    """Map SOURCE=RELATIVE pairs to bwrap read-only binds inside the neutral stage."""
    arguments = []
    for pair in pairs:
        source, separator, relative = pair.partition('=')
        relative_path = PurePosixPath(relative)
        if (not separator or not source or relative_path.is_absolute()
                or not relative_path.parts or '..' in relative_path.parts):
            raise ValueError(f'Use SOURCE=RELATIVE inside the stage for a read-only bind: {pair}')
        if not Path(source).exists():
            raise FileNotFoundError(source)
        arguments += ['--ro-bind', str(Path(source).resolve()), f'{NEUTRAL}/{relative_path}']
    return arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--xvfb', type=Path, required=True)
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--inputs', type=Path,
                        help='Prepared private input tree copied byte-for-byte into the new stage')
    parser.add_argument('--ro-bind', action='append', default=[], metavar='SOURCE=RELATIVE',
                        help='Read-only host path visible at RELATIVE inside the neutral stage, e.g. a backend environment')
    parser.add_argument('capture_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    stage = args.stage.resolve()
    root = Path(__file__).resolve().parents[2]
    if not args.child:
        binds = read_only_binds(args.ro_bind)
        stage.mkdir(parents=True, exist_ok=False)
        if args.inputs is not None:
            copy_inputs(args.inputs, stage)
    for name in ('cache', 'example_data', 'app-state', 'logs', 'mpl', 'tmp', 'profile'):
        (stage / name).mkdir(parents=True, exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES='', SPACR_TUTORIAL_CACHE_ISOLATED='1', SPACR_NO_SPLASH='1',
                      SPACR_HOME=str(stage/'app-state'), SPACR_LOG_DIR=str(stage/'logs'),
                      XDG_CACHE_HOME=str(stage/'cache'), SPACR_NEWS_CACHE=str(stage/'cache/news'),
                      MPLCONFIGDIR=str(stage/'mpl'), TMPDIR=str(stage/'tmp'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    capture_args = args.capture_args[1:] if args.capture_args[:1] == ['--'] else args.capture_args
    if not args.child:
        neutral = NEUTRAL
        subprocess.run([
            'bwrap', '--die-with-parent', '--unshare-net', '--ro-bind', '/', '/',
            '--dev-bind', '/dev', '/dev', '--tmpfs', '/tmp', '--bind', str(stage), neutral,
            '--bind', str(stage/'app-state'), str(Path.home()/'.spacr'),
            '--bind', str(stage/'example_data'), str(Path.home()/'.cache/spacr/example_data'),
            *binds,
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
                'prepared_inputs': (json.loads((stage/'inputs-manifest.json').read_text(encoding="utf-8"))
                                    if (stage/'inputs-manifest.json').exists() else None),
            }, indent=2)+'\n', encoding="utf-8")
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
