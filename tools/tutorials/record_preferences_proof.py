"""Proof recording for the headless GNOME capture host: Home -> Preferences -> close.

Run INSIDE the host (it refuses otherwise)::

    tools/gpu_turn.sh capture-host-proof \\
      python tools/tutorials/gnome_capture_host.py --stage STAGE -- \\
      python tools/tutorials/record_preferences_proof.py --out-dir OUT

THE ACTUAL APPLICATION. spaCR is started through its real entry point,
``spacr.qt.run(['--fresh', '--no-setup'])`` -- the function the ``spacr``
command calls -- from this checkout's source (not the installed copy). So
everything a user's launch installs is there: the dialog filters (glass
card, rounded translucent frameless dialogs), fonts, GUI scale, the ambient
startup. In-process recorders that build ``MainWindow`` themselves skip all
of that, which is why the earlier Preferences walkthrough was square and
opaque.

The private profile is prepared first, in a separate offscreen process, by
the same capture_policy calls every tutorial uses (dark theme, spaCR field
``data_art_impulse_lens`` at its shipped density and palette, alpha features
off, fresh start, "What's new" marked seen, font scale 1.5 x width/3840).

The Qt main thread only runs the app. A driver thread is the human: it
moves the real compositor pointer (``human_input.Human``) along eased
paths, maximises nothing by magic (the window is maximised before the
recording starts), opens spaCR -> Preferences... from the menu bar, hovers
two controls, opens a category and scrolls it with the wheel, and closes the
dialog with its Cancel button -- while GNOME's own ScreenCast stream is
recorded with the cursor embedded.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def use_this_checkout():
    sys.meta_path[:] = [f for f in sys.meta_path
                        if '__editable__' not in (getattr(f, '__module__', '') or type(f).__module__)]
    sys.path[:0] = [str(REPO), str(HERE)]


def app_environment(profile: Path) -> dict:
    for name in ('config', 'state', 'spacr-home', 'logs', 'mpl', 'cache', 'example_data'):
        (profile / name).mkdir(parents=True, exist_ok=True)
    return {
        # Xwayland: mutter composites ARGB windows, and global Qt
        # coordinates equal compositor coordinates for human_input.
        'QT_QPA_PLATFORM': 'xcb', 'QT_SCALE_FACTOR': '1', 'QT_AUTO_SCREEN_SCALE_FACTOR': '0',
        'QT_FONT_DPI': '96', 'SPACR_LANGUAGE': 'en',
        'XDG_CONFIG_HOME': str(profile / 'config'), 'XDG_STATE_HOME': str(profile / 'state'),
        'SPACR_HOME': str(profile / 'spacr-home'), 'SPACR_LOG_DIR': str(profile / 'logs'),
        'MPLCONFIGDIR': str(profile / 'mpl'), 'XDG_CACHE_HOME': str(profile / 'cache'),
        'SPACR_EXAMPLE_DATA': str(profile / 'example_data'), 'SPACR_TUTORIAL_CACHE_ISOLATED': '1',
    }


def prepare_profile(font_scale: float, settings=()) -> None:
    """Child process (offscreen): write the tutorial capture policy into the private profile."""
    use_this_checkout()
    from PySide6.QtWidgets import QApplication

    import spacr.qt
    spacr.qt.register_self_registering_modules()
    from spacr.qt import app as gui
    from spacr.qt.first_run import mark_tour_seen
    from spacr.qt.preferences import (_note_running_version, set_ai_on_by_default,
                                      set_font_scale, set_preload_policy)
    from spacr.qt.walkthrough import mark_seen
    from spacr.qt.widgets.fold_strip import folded_modules
    from spacr.updater import _installed_version
    from capture_policy import configure_appearance

    app = QApplication.instance() or QApplication(['spacr'])  # noqa: F841 - QSettings/Qt state
    mark_tour_seen()
    for key, *_ in gui.APPS:
        mark_seen(key)
    for key in folded_modules():
        mark_seen(key)
    set_preload_policy('on_demand')
    set_ai_on_by_default(False)
    configure_appearance('dark', 'data_art_impulse_lens', default_field=True)
    set_font_scale(font_scale)
    # No "What's new" dialog: the app compares against installed metadata.
    _note_running_version(_installed_version())
    _note_running_version(_installed_version())
    if settings:
        # Extra preference-store values (e.g. a user's rim/popup_backdrop), written
        # exactly as Preferences would store them; recorded in the receipt.
        from spacr.qt.preferences import _settings
        store = _settings()
        for pair in settings:
            key, _, value = pair.partition('=')
            store.setValue(key, value)
        store.sync()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--font-scale', type=float, default=None,
                        help='Default: 1.5 x monitor width / 3840, i.e. the 4K lessons\' 1.5 '
                             'at the same proportion of the screen (1.0 at 2560x1440)')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='Extra preference value in the fresh capture profile, e.g. rim/popup_backdrop=blobs')
    parser.add_argument('--pace', type=float, default=1.0,
                        help='Multiplier on the scripted pauses (0.4 gives a ~15 s clip)')
    parser.add_argument('--card-still', type=float, default=0.0,
                        help='Extra seconds with the pointer still right after Preferences opens '
                             '(for measuring the card body over time)')
    parser.add_argument('--prepare-profile', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_profile:
        prepare_profile(args.font_scale, args.set)
        return 0
    if not os.environ.get('SPACR_CAPTURE_HOST'):
        parser.error('run inside tools/tutorials/gnome_capture_host.py')
    if args.font_scale is None:
        args.font_scale = round(1.5 * int(os.environ['SPACR_CAPTURE_HOST_SIZE'].split('x')[0]) / 3840, 3)
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    os.environ.update(app_environment(out / 'app-profile'))
    subprocess.run([sys.executable, str(Path(__file__).resolve()), '--out-dir', str(out),
                    '--font-scale', str(args.font_scale), *[f'--set={kv}' for kv in args.set],
                    '--prepare-profile'],
                   env={**os.environ, 'QT_QPA_PLATFORM': 'offscreen'}, check=True)
    use_this_checkout()

    import spacr
    assert Path(spacr.__file__).resolve().parent == REPO / 'spacr', spacr.__file__
    import spacr.qt
    from human_input import GuiCaller, HumanProcess, widget_center, widget_rect
    from capture_policy import exclude_release_history, running_version, verify_appearance, verify_fresh_start

    receipt = {'repo_head': subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'],
                                           capture_output=True, text=True).stdout.strip(),
               'spacr_module': spacr.__file__, 'spacr___version__': spacr.__version__,
               'installed_metadata_version': running_version(),
               'setup_py_VERSION': next(line.split('"')[1] for line in (REPO / 'setup.py').read_text().splitlines()
                                        if line.startswith('VERSION = ')),
               'launch': "spacr.qt.run(['--fresh', '--no-setup'])", 'font_scale': args.font_scale,
               'monitor': os.environ['SPACR_CAPTURE_HOST_SIZE'], 'extra_settings': args.set,
               'steps': []}
    failure = []
    t0 = [time.monotonic()]
    go = threading.Event()

    def step(name):
        receipt['steps'].append({'t': round(time.monotonic() - t0[0], 3), 'step': name})
        print(f'[proof] {receipt["steps"][-1]["t"]:7.2f}s {name}', flush=True)

    pace = args.pace

    def human_script():
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QAbstractButton, QApplication, QComboBox, QDialog,
                                       QLabel, QMenu, QScrollArea, QSlider, QTabWidget)
        human = None
        app = None
        try:
            human = HumanProcess(log=(out / 'human_input.log').open('w'))
            human.press('Escape')                      # leave GNOME's startup overview
            human.warp(human.width * 0.62, human.height * 0.58)
            time.sleep(0.8)
            go.set()                                   # now launch spaCR
            deadline = time.monotonic() + 120
            while (app := QApplication.instance()) is None and time.monotonic() < deadline:
                time.sleep(0.05)
            from spacr.qt import app as spacr_app

            def main_window():
                return next((w for w in QApplication.topLevelWidgets()
                             if isinstance(w, spacr_app.MainWindow) and w.isVisible()), None)

            def wait_until(fn, timeout=10.0, poll=0.05):
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    value = gui(fn)
                    if value:
                        return value
                    time.sleep(poll)
                raise TimeoutError(getattr(fn, '__name__', 'condition'))

            window = wait_until(main_window, timeout=120, poll=0.2)
            gui(window.showMaximized)          # set-up, before the recording
            time.sleep(6.0)                    # startup work and the field settle
            gui(lambda: verify_fresh_start(window))
            gui(lambda: exclude_release_history(window))
            receipt['appearance'] = str(gui(lambda: verify_appearance(window, default_field=True)))
            receipt['ui_version_labels'] = gui(lambda: sorted({
                label.text() for label in window.findChildren(QLabel)
                if label.isVisible() and (spacr.__version__ in label.text()
                                          or receipt['installed_metadata_version'] in label.text())}))
            receipt['window_rect'] = gui(lambda: widget_rect(window))
            receipt['qpa'] = gui(app.platformName)

            def menu_center(title_part):
                bar = window.menuBar()
                for action in bar.actions():
                    if title_part in action.text().replace('&', ''):
                        p = bar.mapToGlobal(bar.actionGeometry(action).center())
                        return p.x(), p.y()
                raise RuntimeError(f'No menu {title_part!r}')

            def open_item_center(text_part):
                menu = QApplication.activePopupWidget()
                if isinstance(menu, QMenu):
                    for action in menu.actions():
                        if text_part in action.text():
                            p = menu.mapToGlobal(menu.actionGeometry(action).center())
                            return p.x(), p.y()
                return None

            def dialog():
                return next((w for w in QApplication.topLevelWidgets()
                             if isinstance(w, QDialog) and w.isVisible()
                             and w.findChild(QTabWidget, 'PreferencesTabs')), None)

            def visible(w):
                return w.isVisible() and w.width() > 4 and w.height() > 4

            def facts():
                d = dialog()
                if d is None:
                    return None
                tabs = d.findChild(QTabWidget, 'PreferencesTabs')
                page = tabs.currentWidget()
                handle = d.windowHandle()
                scroll = page if isinstance(page, QScrollArea) else next(
                    (s for s in page.findChildren(QScrollArea) if visible(s)), None)
                view = scroll.viewport() if scroll else None
                close = next((b for b in d.findChildren(QAbstractButton)
                              if visible(b) and b.text().replace('&', '').lower() == 'cancel'), None)
                bar = tabs.tabBar()
                return {
                    'dialog_rect': widget_rect(d), 'tab': tabs.tabText(tabs.currentIndex()),
                    'frameless': bool(d.windowFlags() & Qt.FramelessWindowHint),
                    'translucent': d.testAttribute(Qt.WA_TranslucentBackground),
                    'alpha_buffer': handle.format().alphaBufferSize() if handle else None,
                    'has_mask': not d.mask().isEmpty(),
                    'window_opacity': d.windowOpacity(),
                    'tab_centers': {tabs.tabText(i): (lambda p: (p.x(), p.y()))(bar.mapToGlobal(bar.tabRect(i).center()))
                                    for i in range(tabs.count())},
                    'hover': [(type(w).__name__, widget_center(w)) for w in
                              ([c for c in page.findChildren(QComboBox) if visible(c)][:1]
                               + [s for s in page.findChildren(QSlider) if visible(s)][:1])],
                    # The label column: wheel events there reach the scroll area,
                    # not a spin box or combo that would eat them.
                    'scroll_point': (lambda r: (r[0] + r[2] * 0.10, r[1] + r[3] * 0.55))(widget_rect(view)) if view else None,
                    'scroll_value': scroll.verticalScrollBar().value() if scroll else None,
                    'scroll_max': scroll.verticalScrollBar().maximum() if scroll else None,
                    'cancel': widget_center(close) if close else None,
                }

            from spacr.qt.widgets.ambient import AmbientWidget

            def field():
                widgets = [w for w in window.findChildren(AmbientWidget)
                           if w.isVisible() and not w.property('spacrPopupBackdrop')]
                engines = [getattr((getattr(w, '_producer_box', None) or [None])[0], '_engine', None)
                           or getattr(w, '_engine', None) for w in widgets]
                return {'t': time.monotonic(), 'frames_painted': sum(w.frames_painted for w in widgets),
                        'fps_cap': [w.fps() for w in widgets],
                        'gpu_refusal': [getattr(e, '_graphics_refusal', 'n/a') for e in engines]}

            human.warp(human.width * 0.62, human.height * 0.58)
            time.sleep(0.5)
            receipt['field_samples'] = [dict(gui(field), at='before recording')]
            t0[0] = time.monotonic()
            human.start_recording(str(out / 'preferences_proof.mp4'))
            t0[0] = time.monotonic()
            step('recording: Home with the animated spaCR field')
            human.pause(3.5 * pace)
            receipt['field_samples'].append(dict(gui(field), at='end of the still Home segment'))
            human.move_to(human.width * 0.40, human.height * 0.36)
            human.pause(0.7 * pace)
            step('glide to the spaCR menu')
            human.click_at(*gui(lambda: menu_center('spaCR')))
            item = wait_until(lambda: open_item_center('Preferences'))
            human.pause(0.45 * pace)
            step('glide to Preferences...')
            human.move_to(*item)
            human.pause(0.3 * pace)
            human.click()
            first = wait_until(facts, timeout=20)
            receipt['dialog_on_open'] = first

            step(f'Preferences open ({first["tab"]})')
            def popup_backdrop():
                d = dialog()
                found = [w for w in d.findChildren(AmbientWidget)] if d else []
                from spacr.qt import preferences as prefs
                return {'pref_popup_backdrop': prefs.get_popup_backdrop(),
                        'pref_popup_darkness': prefs._popup_backdrop_darkness(),
                        'pref_popup_motion': prefs._popup_backdrop_motion(),
                        'widgets': [{'theme': w.theme(), 'visible': w.isVisible(),
                                     'popup_backdrop': bool(w.property('spacrPopupBackdrop')),
                                     'rect': widget_rect(w), 'frames_painted': w.frames_painted,
                                     'fps_cap': w.fps(), 't': time.monotonic() - t0[0]} for w in found]}

            receipt['popup_backdrop'] = [gui(popup_backdrop)]
            if args.card_still:
                step('card still (pointer at rest)')
                human.pause(args.card_still)
                receipt['popup_backdrop'].append(gui(popup_backdrop))

                def grabs():
                    # Qt's own rendering, independent of the compositor: the
                    # dialog as Qt paints it, and its popup backdrop alone.
                    d = dialog()
                    d.grab().save(str(out / 'qt_grab_dialog.png'))
                    for k, w in enumerate(d.findChildren(AmbientWidget)):
                        w.grab().save(str(out / f'qt_grab_popup_backdrop_{k}.png'))

                gui(grabs)
            human.pause(1.6 * pace)
            for kind, point in first['hover']:
                step(f'hover {kind}')
                human.move_to(*point)
                human.pause(1.3 * pace)
            target = 'Performance' if 'Performance' in first['tab_centers'] else list(first['tab_centers'])[2]
            step(f'open the {target} category')
            human.click_at(*first['tab_centers'][target])
            human.pause(0.9 * pace)
            page = gui(facts)
            receipt['dialog_category'] = page
            if not page['scroll_point'] or not page['scroll_max']:
                raise RuntimeError(f'{target} does not scroll: {page}')
            step('scroll the category')
            human.move_to(*page['scroll_point'])
            human.pause(0.4 * pace)
            human.scroll(+7)
            human.pause(0.9 * pace)
            receipt['scroll_value_after_down'] = gui(facts)['scroll_value']
            human.scroll(-7)
            human.pause(0.8 * pace)
            receipt['scroll_value_after_up'] = gui(facts)['scroll_value']
            cancel = gui(facts)['cancel']
            step('close with Cancel')
            human.click_at(*cancel)
            wait_until(lambda: dialog() is None, timeout=10)
            step('back on Home')
            human.pause(0.4 * pace)
            human.move_to(human.width * 0.56, human.height * 0.62)
            human.pause(2.5 * pace)
            receipt['field_samples'].append(dict(gui(field), at='end of recording'))
            receipt['recording'] = human.stop_recording()
            step('stopped')
            human.save_log(str(out / 'pointer_trace.json'), origin=t0[0])
        except BaseException:
            failure.append(traceback.format_exc())
            print(failure[-1], file=sys.stderr, flush=True)
            go.set()
        finally:
            if human is not None:
                human.close()
            try:
                if app is not None:
                    gui(lambda: (dialog() and dialog().reject(), app.quit()))
            except Exception:
                os._exit(4)

    gui = GuiCaller()                  # main thread, before spaCR builds its QApplication
    threading.Thread(target=human_script, name='human', daemon=True).start()
    go.wait()
    if not failure:
        rc = spacr.qt.run(['--fresh', '--no-setup'])
        receipt['app_exit_code'] = rc
    (out / 'proof_receipt.json').write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    if failure:
        (out / 'failure.txt').write_text('\n'.join(failure))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
