"""Lesson 04 (installers): Home and Performance of the current nightly, recorded in the real app.

The pip-install desktop driver (capture_pip_installation.py) records the install
terminal. Its last two scenes show the application itself; in the GNOME capture
host they are recorded here, in the running app, with the pointer doing the
work: 13_current_nightly_home (the Home tabs Core, Data, Tools and Organism, each
opened by a click) and 14_current_nightly_performance (spaCR menu, Preferences,
the Performance tab and the profile list; the profile is kept).

Run: ``capture_refresh --module home --openings --openings-set install_nightly``.
Outside the host (Xvfb) the old stills are taken after the same steps.
"""
from __future__ import annotations

import capture_ui
from capture_ui_drv_a import Rec
from capture_ui_drv_e import guarded


@guarded
def record_install_nightly(app, window, captures, capture, settle, write_json):
    from PySide6.QtWidgets import QComboBox

    from spacr.qt import preferences  # noqa: F401  (the dialog class must import)

    import capture_home_live as h

    evidence = {'lesson': '04_platform_installers', 'home_tabs': [], 'performance': {}}
    if not window._startup.isVisible():
        capture_ui.open_module(capture, window, '__home__', settle)
    tabs = window._startup._tabs
    with capture_ui.clip(capture, '13_current_nightly_home'):
        for index in range(tabs.count()):
            capture_ui.select_tab(capture, tabs, index)
            settle(0.6)
            evidence['home_tabs'].append(tabs.tabText(index))
            capture_ui.wait(capture, 1.5, settle)
        capture_ui.select_tab(capture, tabs, 0)
    if not evidence['home_tabs']:
        raise RuntimeError('Home has no tabs')

    # 14: the pointer opens spaCR > Preferences..., the Performance tab and its profile list.
    rec = Rec(capture, '14_current_nightly_performance', settle).start()

    def steps(dialog):
        try:
            selector = dialog.findChild(QComboBox, 'PerformanceLevel')
            if selector is None:
                raise RuntimeError('Preferences has no Performance selector')
            page = next(w for w in h._ancestors(selector) if (w.objectName() or '').startswith('PreferencesTab'))
            h.show_tab(capture, dialog, page.objectName())
            h.hover(selector)
            h.pause(capture, 1.2)
            before = selector.currentText()
            h.choose_combo(capture, selector)          # the list opens; the current profile stays
            if selector.currentText() != before:
                raise RuntimeError('The Performance profile changed')
            evidence['performance'] = {'profile': before,
                                       'profiles': [selector.itemText(i) for i in range(selector.count())]}
            h.pause(capture, 1.0)
            rec.stop()
            h.close_preferences(dialog, 'cancel')
        except BaseException:
            rec.abort()
            raise

    if capture_ui.live(capture):
        h.open_preferences(app, window, capture, steps)
    else:
        from spacr.qt.preferences import PreferencesDialog
        from PySide6.QtWidgets import QTabWidget
        dialog = PreferencesDialog(window)
        try:
            tabs = dialog.findChild(QTabWidget, 'PreferencesTabs')
            selector = dialog.findChild(QComboBox, 'PerformanceLevel')
            for index in range(tabs.count()):
                if tabs.widget(index).findChild(QComboBox, 'PerformanceLevel') is selector:
                    tabs.setCurrentIndex(index)
                    break
            dialog.resize(1200, 1200)
            dialog.move(window.mapToGlobal(window.rect().center()) - dialog.rect().center())
            dialog.show()
            settle()
            evidence['performance'] = {'profile': selector.currentText()}
            rec.stop()
        finally:
            dialog.close()
            dialog.deleteLater()
            settle(0.5)
    write_json(captures / 'install_nightly.json', evidence)
