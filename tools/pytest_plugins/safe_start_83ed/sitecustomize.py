"""Diagnostic child breadcrumbs and periodic stacks; no launcher mutations."""
import os
import sys

_MARKER = 'SPACR_SAFE_COLLECTOR_ERROR:'
_errors = []


def _error(operation, exc):
    """Use a separate integrity channel that cannot interrupt the child."""
    try:
        import json
        record = {'operation': operation, 'error_type': type(exc).__name__,
                  'pid': os.getpid()}
        if len(_errors) < 100:
            _errors.append(record)
        sys.__stderr__.write(_MARKER + json.dumps(record) + '\n')
        sys.__stderr__.flush()
    except Exception:
        pass


def _initialize():
    import atexit
    import faulthandler
    import json
    from pathlib import Path
    import time

    root = Path(os.environ['SPACR_SAFE_DIAGNOSTIC_DIR'])
    root.mkdir(parents=True, exist_ok=True)
    target = os.path.abspath(sys.argv[0])
    phase = Path(sys.argv[3]).stem
    evidence = root / (phase + '-pid-' + str(os.getpid()))
    evidence.mkdir(exist_ok=False)
    journal = (evidence / 'phases.jsonl').open('w', buffering=1)
    faults = os.open(evidence / 'periodic-stacks.log', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    count = 0

    def note(event, **detail):
        nonlocal count
        try:
            if count >= 2000:
                if count == 2000:
                    count += 1
                    journal.write(json.dumps({'event': 'phase_detail_truncated', 'limit': 2000}) + '\n')
                    journal.flush()
                return
            count += 1
            journal.write(json.dumps({'event': event, 'monotonic': time.monotonic(),
                                      'pid': os.getpid(), 'phase': phase, **detail}) + '\n')
            journal.flush()
        except Exception as exc:
            _error('phase-journal', exc)

    def finish():
        try:
            sys.setprofile(None)
            note('child_python_atexit_started')
            if _errors:
                (evidence / 'collector-integrity-child.json').write_text(
                    json.dumps({'errors': _errors}, indent=2) + '\n', encoding="utf-8")
        except Exception as exc:
            _error('child-integrity-receipt', exc)

    atexit.register(finish)
    try:
        import coverage
        if os.environ.get('COVERAGE_PROCESS_CONFIG') or os.environ.get('COVERAGE_PROCESS_START'):
            if coverage.Coverage.current() is None:
                coverage.process_startup(slug='diagnostic_sitecustomize')
        coverage_active = coverage.Coverage.current() is not None
    except Exception as exc:
        coverage_active = False
        _error('coverage-start', exc)
    note('child_boot', coverage_active=coverage_active,
         cuda_blank=os.environ.get('CUDA_VISIBLE_DEVICES') == '',
         platform=os.environ.get('QT_QPA_PLATFORM'),
         nvidia_nodes=[n for n in os.listdir('/dev') if n.startswith('nvidia')],
         dri_present=os.path.exists('/dev/dri'))
    faulthandler.dump_traceback_later(30, repeat=True, file=faults)

    def profile(frame, event, arg):
        try:
            if event not in ('call', 'return'):
                return
            file = frame.f_code.co_filename
            function = frame.f_code.co_name
            if file == target and function in {'<module>', 'exec_module', '__init__', 'write', '_snapshot', '_repair_through_preferences', 'in_the_dialog', '_look_and_leave', '_driven_exec', '_watchdog'}:
                note('launcher_' + event, function=function, line=frame.f_lineno)
            elif function == 'run' and file.endswith('/spacr/qt/__init__.py'):
                note('qt_run_' + event, line=frame.f_lineno)
            elif function == '__init__' and file.endswith('/spacr/qt/app.py') and frame.f_code.co_qualname == 'MainWindow.__init__':
                note('main_window_' + event, line=frame.f_lineno)
            elif function == '__init__' and file.endswith('/spacr/qt/widgets/ambient.py') and frame.f_code.co_qualname == 'AmbientWidget.__init__':
                note('ambient_' + event, line=frame.f_lineno)
        except Exception as exc:
            _error('profile-callback', exc)

    sys.setprofile(profile)


if os.path.basename(sys.argv[0]) == 'launch_once.py' and len(sys.argv) == 6 and sys.argv[2] in {'normal', 'safe'} and sys.argv[4:] == ['blobs', 'data_art_impulse_lens'] and os.environ.get('SPACR_SAFE_DIAGNOSTIC_DIR'):
    try:
        _initialize()
    except Exception as exc:
        _error('child-collector-initialization', exc)
