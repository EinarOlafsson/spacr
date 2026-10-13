"""Show unchanged, locally saved PNG plots in the real system image viewer.

This never inserts an image into spaCR's Figures card or changes application
code. A private X11 display and D-Bus session are mandatory.
"""
import hashlib
from pathlib import Path
import subprocess


def check_saved_file_run(run):
    """Accept only a completed worker and the specifically diagnosed src error.

The stricter in-app figure acceptance remains separate and unchanged.
Independent CSV checks and actual viewer capture are additionally required.
"""
    outcome = run['outcome']
    if not outcome['finished'] or not outcome['ok'] or outcome['errors']:
        raise ValueError('The actual worker did not finish successfully')
    errors = [line.strip() for line in run['settings_errors']]
    if errors not in ([], ['[settings] ERROR [src]: src is missing from the settings.']):
        raise ValueError('Unexpected settings errors cannot be accepted by this workaround')


def show_saved_plots(app, window, stage, capture, settle, paths):
    from PIL import Image
    from capture_diagnostics import PrivateDesktop

    if not 1 <= len(paths) <= 8:
        raise ValueError('Use one to eight already verified saved plots')
    checked = []
    for path in map(Path, paths):
        if path.is_symlink() or not path.resolve().is_relative_to(stage.resolve()) or path.suffix != '.png':
            raise ValueError('Only original PNG files within private tutorial staging are allowed')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with Image.open(path) as image:
            size = list(image.size)
            if image.format != 'PNG' or min(size) < 100:
                raise ValueError('Expected a nonempty PNG figure')
            image.verify()
        checked.append(dict(path=str(path), sha256=digest, dimensions=size))
    desktop = PrivateDesktop(stage)
    try:
        for index, record in enumerate(checked, 9):
            path = Path(record['path'])
            viewer = subprocess.Popen(['eog', '--new-instance', '--disable-gallery', str(path)],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                wid, title = desktop.find(path.name, settle)
                desktop.show(wid); settle(2)
                capture(f'{index:02d}_saved_{path.stem}', desktop=True)
                record.update(window_title=title, viewer='eog', actual_desktop_capture=True)
            finally:
                if viewer.poll() is None:
                    viewer.terminate()
                    try:
                        viewer.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        viewer.kill(); viewer.wait(timeout=5)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
                raise ValueError('Viewing changed the saved plot')
        desktop.x.XMapRaised(desktop.display, int(window.winId()))
        desktop.x.XFlush(desktop.display); settle()
        capture('11_back_to_actual_gui')
    finally:
        desktop.close()
    return dict(accepted=True, saved_plots=checked, saved_files_unchanged=True,
                application_figures_fixed=False, scope='External system viewer, not spaCR Figures')
