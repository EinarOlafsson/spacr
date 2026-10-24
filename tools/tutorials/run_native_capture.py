"""Run the normal recorder on a private display and fresh neutral profile.

``--host xvfb`` (default): a private Xvfb; the recorder builds the window itself.

``--host gnome``: the headless GNOME capture host (``gnome_capture_host.py``):
the real application started by ``spacr.qt.run``, the scene drivers' QTest
calls performed by a human-like real pointer (``capture_host_adapter.py``),
every clip a GNOME screen recording checked by ``capture_clip_gate.py``, and
the same ``captures/<module>/`` files for the staging tools
(``capture_host_mode.py``). For example::

    tools/gpu_turn.sh lesson-05-host tools/run_capped.sh 12G \
      python tools/tutorials/run_native_capture.py --host gnome --stage NEW_STAGE -- \
      --module home --home-host-smoke --clip-seconds 6
"""
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

#: Modules whose private example capture_refresh prepares and binds only outside run_native (SPACR_TUTORIAL_CACHE_ISOLATED unset).
PREPARED_MODULES = ('lineage', 'image_scatter', 'pipeline_graph', 'project_browser', 'data_manager')
PREPARE_SCRIPT = r'''
import sys, json
from pathlib import Path
sys.path[:0] = ['/tmp/spacr-tutorial-current/app/tools/tutorials', '/tmp/spacr-tutorial-current/app']
module, name, source = sys.argv[1:4]
stage = Path('/tmp/spacr-tutorial-current')
if module in ('lineage', 'image_scatter'):
    from lineage_data import prepare
    kind = module + '_state'
    prepared = prepare(stage, run_directory=module + '_runs', copy_format_markers=module == 'image_scatter')
elif module == 'pipeline_graph':
    from pipeline_graph_data import prepare
    kind = 'pipeline_graph_state'; prepared = prepare(stage)
elif module == 'project_browser':
    from project_browser_data import prepare
    kind = 'project_browser_state'; prepared = prepare(stage)
elif module == 'data_manager':
    from manager_data import prepare
    kind = 'data_manager_state'; prepared = prepare(stage, source=Path(source) if source else None)
state = stage / kind / (name + '.json')
if state.exists():
    raise RuntimeError('Use a new capture name for a fresh private example: ' + str(state))
state.parent.mkdir(parents=True, exist_ok=True)
state.write_text(json.dumps(prepared, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print('prepared', kind, name)
'''


def flag_value(flags, name):
    return flags[flags.index(name) + 1] if name in flags and flags.index(name) + 1 < len(flags) else None


def copy_inputs(source, stage, allow_captures=False):
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
        if relative.parts[0] in (() if allow_captures else ('captures',)) + ('exact-source.json', 'inputs-manifest.json'):
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
        (stage / folder.relative_to(source)).mkdir(parents=True, exist_ok=True)     # an empty prepared folder is part of the tree
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
    # Keep a native crash's core: the child (and bwrap, gnome-shell, the app)
    # inherit this limit, so apport can write a core into the app's working
    # directory, the stage, next to the faulthandler log in stage/logs.
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CORE, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ImportError, ValueError, OSError):
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', type=Path, required=True)
    parser.add_argument('--host', choices=('xvfb', 'gnome'), default='xvfb')
    parser.add_argument('--fps', choices=('auto', '30', '60'), default='auto',
                        help='--host gnome recording rate. auto = 30: at 2560x1440 the GNOME '
                             'screencast delivers only ~40 frames/s of the 60 fps field (compositor-bound, '
                             'measured 2026-10-12), so 60 is choppy; see capture_host_mode.AUTO_FPS')
    parser.add_argument('--refresh', type=int, default=60,
                        help='--host gnome virtual monitor refresh (Hz). 60 by default: at 30 Hz the '
                             'screencast delivered only ~20 frames/s (measured 2026-10-12)')
    parser.add_argument('--size', default='2560x1440', help='--host gnome virtual monitor (16:9)')
    parser.add_argument('--cuda', action='store_true',
                        help='--host gnome: let the app use CUDA (model previews and runs inside a GPU turn); '
                             'default CPU only, as on the Xvfb recorder')
    parser.add_argument('--splash', action='store_true',
                        help='--host gnome: leave SPACR_NO_SPLASH unset so the app plays its startup logo '
                             '(2 s hold, 4 s blow-away). Needs capture_refresh --record-startup, which ends '
                             'the 01_startup_sequence clip that capture_host_mode begins before the app launches')
    parser.add_argument('--xvfb', type=Path, help='The Xvfb binary (--host xvfb)')
    parser.add_argument('--child', action='store_true')
    parser.add_argument('--in-host', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--host-stage', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--inputs', type=Path,
                        help='Prepared private input tree copied byte-for-byte into the new stage')
    parser.add_argument('--stage-journal', action='store_true',
                        help='Bind stage/runs as the app run journal (profile/.spacr/runs and app-state/runs), as capture_refresh did; for the history lessons')
    parser.add_argument('--legacy-root', type=Path,
                        help='Read-only view of this folder at /mnt/firecuda2/Claude (the drivers default to that old workspace path)')
    parser.add_argument('--input-captures', action='store_true',
                        help='Let --inputs carry the retained captures/<name> folders of an earlier accepted lesson capture')
    parser.add_argument('--ro-bind', action='append', default=[], metavar='SOURCE=RELATIVE',
                        help='Read-only host path visible at RELATIVE inside the neutral stage, e.g. a backend environment')
    parser.add_argument('capture_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.host == 'xvfb' and args.xvfb is None:
        parser.error('--host xvfb needs --xvfb')
    capture_flags = args.capture_args[1:] if args.capture_args[:1] == ['--'] else args.capture_args
    if args.splash and (args.host != 'gnome' or '--record-startup' not in capture_flags):
        parser.error('--splash is for --host gnome with the capture argument --record-startup: '
                     'every other recording runs with SPACR_NO_SPLASH=1')
    if '--record-startup' in capture_flags and not args.splash:
        parser.error('--record-startup needs --splash (the logo is switched off otherwise)')
    stage = args.stage.resolve()
    root = Path(__file__).resolve().parents[2]
    if args.host == 'gnome' and not args.child and not args.in_host:
        # Outer step: a new stage, then the GNOME capture host around the rest.
        read_only_binds(args.ro_bind)
        stage.mkdir(parents=True, exist_ok=False)
        if args.inputs is not None:
            copy_inputs(args.inputs, stage, args.input_captures)
        host_stage = stage / 'gnome-host'
        capture_args = args.capture_args[1:] if args.capture_args[:1] == ['--'] else args.capture_args
        return subprocess.run([
            sys.executable, str(Path(__file__).with_name('gnome_capture_host.py')), '--stage', str(host_stage),
            '--refresh', str(args.refresh), '--size', args.size,
            '--', sys.executable, str(Path(__file__).resolve()), '--host', 'gnome', '--in-host',
            '--stage', str(stage), '--host-stage', str(host_stage), '--fps', args.fps,
            *(['--cuda'] if args.cuda else []), *(['--splash'] if args.splash else []),
            *(['--legacy-root', str(args.legacy_root)] if args.legacy_root else []),
            *(['--stage-journal'] if args.stage_journal else []),
            *[f'--ro-bind={pair}' for pair in args.ro_bind], '--', *capture_args]).returncode
    if not args.child:
        binds = read_only_binds(args.ro_bind)
        if not args.in_host:
            stage.mkdir(parents=True, exist_ok=False)
            if args.inputs is not None:
                copy_inputs(args.inputs, stage, args.input_captures)
    for name in ('cache', 'example_data', 'app-state', 'logs', 'mpl', 'tmp', 'profile', 'runs'):
        (stage / name).mkdir(parents=True, exist_ok=True)
    if not args.cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    # Lessons record the shipped default: field card compositing off (02540ce97).
    os.environ.pop('SPACR_COMPOSITE_CARDS', None)
    # The startup logo is off in every recording except the one launch clip
    # (--splash, see capture_refresh --record-startup).
    if args.splash:
        os.environ.pop('SPACR_NO_SPLASH', None)
    else:
        os.environ['SPACR_NO_SPLASH'] = '1'
    # No accessibility bridge inside the capture session: the AT-SPI bridge
    # walks item views over D-Bus while they change (a crash class in lessons
    # 39/42/44) and costs D-Bus traffic in every recording. Nothing visible
    # depends on it.
    # SPACR_TUTORIAL_AT_BRIDGE=1 keeps the inherited accessibility settings (A/B check for the live file picker).
    if os.environ.get('SPACR_TUTORIAL_AT_BRIDGE') != '1':
        os.environ.pop('QT_LINUX_ACCESSIBILITY_ALWAYS_ON', None)
        os.environ.update(NO_AT_BRIDGE='1', QT_ACCESSIBILITY='0')
    os.environ.update(SPACR_TUTORIAL_CACHE_ISOLATED='1',
                      SPACR_HOME=str(stage/'app-state'), SPACR_LOG_DIR=str(stage/'logs'),
                      XDG_CACHE_HOME=str(stage/'cache'), SPACR_NEWS_CACHE=str(stage/'cache/news'),
                      MPLCONFIGDIR=str(stage/'mpl'), TMPDIR=str(stage/'tmp'),
                      OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    capture_args = args.capture_args[1:] if args.capture_args[:1] == ['--'] else args.capture_args
    if not args.child:
        neutral = NEUTRAL
        if args.host == 'gnome':
            # Inside the capture host: keep its sockets reachable. The session
            # bus is an abstract socket (needs the host network namespace, so
            # no --unshare-net); Xwayland's socket lives in /tmp/.X11-unix;
            # PipeWire and Wayland sockets in the host stage.
            host_stage = args.host_stage.resolve()
            isolation = ['--bind', '/tmp/.X11-unix', '/tmp/.X11-unix',
                         '--bind', str(host_stage), str(host_stage)]
            if args.cuda:
                # CUDA initialisation fails (error 304) on the read-only /proc
                # of --ro-bind / /; a writable /proc bind lets it start.
                isolation += ['--bind', '/proc', '/proc']
            display = ['--host', 'gnome', '--fps', args.fps, *(['--cuda'] if args.cuda else []),
                       *(['--splash'] if args.splash else [])]
        else:
            isolation = ['--unshare-net']
            display = ['--xvfb', str(args.xvfb.resolve())]
        def sandbox(cache_source, manager_binds):
            return [
            'bwrap', '--die-with-parent', *(isolation if args.host == 'xvfb' else []),
                '--ro-bind', '/', '/',
                '--dev-bind', '/dev', '/dev', '--tmpfs', '/tmp',
                *(['--tmpfs', '/mnt', '--ro-bind', str(args.legacy_root), '/mnt/firecuda2/Claude'] if args.legacy_root else []),
                # drivers also look for stage.parent/synthetic, /derived, /catalog and /test_datasets (the old author tree beside the stage)
                *[x for name, rel in (('synthetic', 'toxoplasma_projects/tutorials/synthetic'), ('derived', 'toxoplasma_projects/tutorials/derived'),
                                      ('catalog', 'toxoplasma_projects/tutorials/catalog'), ('test_datasets', 'toxoplasma_projects/test_datasets'))
                  if args.legacy_root and (args.legacy_root / rel).is_dir() for x in ('--ro-bind', str(args.legacy_root / rel), '/tmp/' + name)], '--bind', str(stage), neutral, '--ro-bind', str(root), neutral + '/app',
                *(isolation if args.host == 'gnome' else []),
                # The app's HOME is the neutral profile: bind its private state there.
                *(['--bind', str(stage/'app-state'), neutral + '/profile/.spacr',
                   '--bind', str(cache_source), neutral + '/profile/.cache/spacr/example_data']
                  if args.host == 'gnome' else
                  ['--bind', str(stage/'app-state'), str(Path.home()/'.spacr'),
                   '--bind', str(stage/'example_data'), str(Path.home()/'.cache/spacr/example_data')]),
                *binds, *manager_binds,
                *(['--bind', str(stage/'runs'), neutral + '/profile/.spacr/runs', '--bind', str(stage/'runs'), neutral + '/app-state/runs'] if args.stage_journal else []),
                '--setenv', 'HOME', neutral + '/profile',
            ]

        cache_source, manager_binds = stage/'example_data', []
        module, capture_name = flag_value(capture_flags, '--module'), flag_value(capture_flags, '--capture-name')
        if module in PREPARED_MODULES:
            capture_name = capture_name or module
            subprocess.run([*sandbox(cache_source, []), '--', sys.executable, '-c', PREPARE_SCRIPT, module, capture_name,
                            flag_value(capture_flags, '--manager-source') or ''], check=True)
            kind = {'lineage': 'lineage_state', 'image_scatter': 'image_scatter_state'}.get(module, module + '_state')
            prepared = json.loads((stage / kind / (capture_name + '.json')).read_text(encoding='utf-8'))
            real = lambda path: stage / Path(path).relative_to(neutral)
            if module in ('lineage', 'image_scatter'):
                cache_source = real(prepared['cache'])
            else:
                # keep the original readable at an immutable alias, then shadow its own pathname with the disposable clone
                manager_binds = ['--ro-bind', str(real(prepared['source'])), prepared['original_readonly'],
                                 '--bind', str(real(prepared['clone'])), prepared['source']]
        # SPACR_TUTORIAL_CAPTURE_GDB=1 (crash retakes only; gdb slows start-up): run the
        # app under gdb in batch mode so a native crash prints bt and py-bt
        # into the job output.
        debugger = (['gdb', '-batch', '-ex', 'set pagination off', '-ex', 'run',
                     '-ex', 'bt', '-ex', 'py-bt', '--args']
                    if os.environ.get('SPACR_TUTORIAL_CAPTURE_GDB') == '1' else [])
        result = subprocess.run([
            *sandbox(cache_source, manager_binds),
            '--', *debugger, sys.executable, neutral + '/app/tools/tutorials/run_native_capture.py', '--stage', neutral,
            *display, '--child', '--', *capture_args,
        ], check=args.host == 'xvfb')
        return result.returncode
    if args.host == 'gnome':
        return gnome_child(stage, root, capture_args, args.fps)
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


def gnome_child(stage, root, capture_args, fps='auto'):
    """In the capture host and the neutral stage: the real app, human pointer, GNOME clips."""
    if not os.environ.get('SPACR_CAPTURE_HOST'):
        raise RuntimeError('--host gnome runs inside tools/tutorials/gnome_capture_host.py')
    os.environ.update(QT_QPA_PLATFORM='xcb', SPACR_CAPTURE_FPS=fps)
    sys.meta_path[:] = [finder for finder in sys.meta_path
                       if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
    sys.path[:0] = [str(root), str(root/'tools/tutorials')]
    import spacr
    assert Path(spacr.__file__).resolve().parent == root/'spacr'
    sources = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
               for pattern in ('spacr/qt/**/*.py', 'tools/tutorials/capture*.py',
                               'tools/tutorials/human_input.py', 'tools/tutorials/gnome_capture_host.py')
               for p in root.glob(pattern)}
    (stage/'exact-source.json').write_text(json.dumps({
        'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        'sha256': sources, 'fresh': True, 'private_profile': True,
        'surface': ('actual spaCR launched by spacr.qt.run in a private headless GNOME Shell '
                    '(mutter compositor, Xwayland); GNOME ScreenCast with the real cursor; '
                    'pointer and keyboard through mutter RemoteDesktop'),
        'monitor': os.environ.get('SPACR_CAPTURE_HOST_SIZE'),
        'top_bar': os.environ.get('SPACR_CAPTURE_HOST_TOP_BAR'),
        'prepared_inputs': (json.loads((stage/'inputs-manifest.json').read_text())
                            if (stage/'inputs-manifest.json').exists() else None),
    }, indent=2)+'\n')
    import capture_host_mode
    # File dialogs open in the working directory: make that the neutral stage,
    # never the checkout's (mounted, personal) path.
    os.chdir(stage)
    result = capture_host_mode.run_child(stage, capture_args)
    if any(hashlib.sha256((root/name).read_bytes()).hexdigest() != digest
           for name, digest in sources.items()):
        raise RuntimeError('Application or recorder source changed during capture')
    return result


if __name__ == '__main__':
    raise SystemExit(main())
