"""The main window and Make Masks fit a 1280 x 720 or 1366 x 768 screen (707).

The maintainer's 2026-10-18 native check: "positions look fine at both
resolutions and 1080p look perfect. but 720 cant fit the entire window on
screen." After the taskbar or menu bar the work area of those displays is
about 1280 x 680 and 1366 x 728. The window held a 1200 x 720 minimum, and
Make Masks wrapped TOOLS and IO onto so many lines beside its side column
that the bars ran out of room and the image shrank to a strip.
"""
from __future__ import annotations

import os
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRect, Qt
from PySide6.QtWidgets import (QAbstractButton, QApplication, QLabel,
                               QWidget)

from spacr.qt import app as app_mod

WORK_AREAS = [(1280, 680), (1366, 728)]


class _Screen:
    """A display whose work area is ``rect``."""

    def __init__(self, rect):
        self._rect = rect

    def availableGeometry(self):                     # noqa: N802 (Qt name)
        return QRect(self._rect)


@pytest.fixture
def folder(tmp_path):
    fields = tmp_path / "fields"
    fields.mkdir()
    for index in range(3):
        imageio.imwrite(fields / f"field_{index}.tif",
                        np.full((64, 64), index + 1, dtype=np.uint16))
    return fields


def _open_make_masks(qtbot, folder, size):
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    window._on_nav_selected("make_masks")
    screen = window._stack.currentWidget()
    assert screen._open_folder(str(folder))
    qtbot.waitUntil(lambda: screen._image_container.isVisible())
    qtbot.waitUntil(lambda: (window.width(), window.height()) == size)
    qtbot.wait(60)
    QApplication.processEvents()
    return window, screen


def _close(screen):
    screen._magnifier.close()
    screen.close_folded()


def _rect_in(window, widget):
    return QRect(widget.mapTo(window, QPoint(0, 0)), widget.size())


def _whole(window, widget):
    rect = _rect_in(window, widget)
    return (widget.isVisible() and window.rect().contains(rect)
            and widget.visibleRegion().boundingRect() == widget.rect())


def test_the_main_window_minimum_fits_the_smallest_work_area(qtbot):
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    minimum = window.minimumSize()
    assert minimum.width() <= 1200 and minimum.height() <= 640
    assert (window.width(), window.height()) == app_mod._MAIN_WINDOW_OPENING


def test_native_home_content_fits_new_desktop_minimum(tmp_path):
    """Check real Home text and icons at the newly legal native desktop floor."""
    import ast
    import json
    import subprocess
    import sys
    import textwrap

    if sys.platform != "darwin" or os.environ.get("QT_QPA_PLATFORM", "").lower() != "cocoa":
        pytest.skip("Requires the owned native Cocoa acceptance runner")
    root = Path(app_mod.__file__).resolve().parents[2]
    source = ast.parse((root / "tests/qt/test_the_macos_application_menu.py").read_text())
    bootstrap = next(node.value.args[0] for node in source.body
                     if isinstance(node, ast.Assign) and any(
                         isinstance(target, ast.Name) and target.id == "_OWNED_SETTINGS_PROBE"
                         for target in node.targets))
    child = textwrap.dedent(ast.literal_eval(bootstrap)) + textwrap.dedent(r'''
        import subprocess, time, traceback
        from pathlib import Path
        from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer
        from PySide6.QtGui import QFontInfo, QRegion, QTextDocument, QTextOption
        from PySide6.QtWidgets import (QAbstractButton, QLabel, QMenuBar, QPushButton, QScrollArea,
            QStyle, QStyleOptionButton, QStyleOptionToolButton, QToolButton)
        from spacr.qt import gui_scale, preferences, theme, timing
        from spacr.qt.app import MainWindow
        from spacr.qt.widgets.home import AppTile, _AppTileIcon, _find_logo_pixmap
        from spacr.qt.widgets.eliding import ElidingLabel

        assert timing.ENABLED, 'Readiness requires explicit SPACR_TIMING=1'
        assert Path(__import__('spacr').__file__).resolve().parent.parent == Path(os.environ['SPACR_HOME_TEST_ROOT'])
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == os.environ['SPACR_HOME_TEST_HEAD']
        preferences.set_gui_scale(2.)
        preferences.set_font_scale(2.)
        report = {'errors': [], 'checks': [], 'programmatic_not_physical': True}
        output = Path(os.environ['SPACR_HOME_TEST_OUTPUT'])
        state = {}

        def persist():
            """Persist actual partial evidence before any strict assertion."""
            output.write_text(json.dumps(report, indent=2))

        def close():
            """Use the real owned MainWindow close route even on failure."""
            if 'timer' in state:
                state['timer'].stop()
            report['normal_close_accepted'] = state['window'].close()
            persist()

        def signature(window, target):
            """Observe only the target subtree and its actual layout ancestors."""
            snapshot = []
            widgets = [target, *target.findChildren(QWidget)]
            ancestor = target.parentWidget()
            while ancestor is not None:
                if ancestor not in widgets:
                    widgets.append(ancestor)
                if ancestor is window:
                    break
                ancestor = ancestor.parentWidget()
            for w in widgets:
                if not w.isVisible():
                    continue
                fitter = getattr(w, '_text_fit', None)
                if fitter is not None and (fitter._timer.isActive() or fitter._busy):
                    return None
                if any(getattr(w, key, False) for key in
                        ('_rebalance_queued', '_limits_queued', '_refit_queued', '_fitting_module_actions', '_back_fit_busy')):
                    return None
                for key in ('_floor_timer', '_module_action_fit_timer', '_back_fit_timer'):
                    timer = getattr(w, key, None)
                    if timer is not None and timer.isActive():
                        return None
                layout = w.layout()
                snapshot.append((id(w), w.property('size').toTuple(), w.pos().toTuple(),
                    w.property('minimumSize').toTuple(), w.property('maximumSize').toTuple(),
                    layout.geometry().getRect() if layout else None, w.font().toString()))
            return snapshot

        def native_menu_child(widget):
            """Keep external macOS menu-bar objects out of in-window claims."""
            parent = widget.parentWidget()
            while parent is not None:
                if isinstance(parent, QMenuBar) and parent.isNativeMenuBar():
                    return True
                parent = parent.parentWidget()
            return False

        def startup_snapshot(window):
            """Observe normal covers and owned onboarding without dismissing them."""
            def geometry(widget, controls=False):
                if widget is None:
                    return None
                try:
                    return {'class': type(widget).__name__, 'name': widget.objectName(),
                        'visible': widget.isVisible(), 'size': widget.property('size').toTuple(),
                        'global_rect': widget.rect().translated(widget.mapToGlobal(QPoint())).getRect(),
                        'mask_empty': widget.mask().isEmpty(),
                        'mask_bounds': widget.mask().boundingRect().getRect(),
                        'mask_rect_count': widget.mask().rectCount(),
                        'visible_region': widget.visibleRegion().boundingRect().getRect(),
                        'controls': [{'class': type(child).__name__, 'text': child.text(),
                            'name': child.objectName(), 'visible': child.isVisible(), 'enabled': child.isEnabled()}
                            for child in widget.findChildren(QAbstractButton) if controls and child.isVisible()]}
                except RuntimeError:
                    return {'destroyed': True}
            running = window._startup_sequence_running()
            overlays = [w for w in window.findChildren(QWidget)
                        if w.isVisible() and ('Overlay' in type(w).__name__ or w.isWindow())]
            modal = QApplication.activeModalWidget()
            if modal is not None and modal not in overlays:
                overlays.append(modal)
            return {'sequence_running': running,
                'loading_screen': geometry(getattr(window, '_loading_screen', None)),
                'blowing_cover': geometry(getattr(window, '_blowing_cover', None)),
                'current_stack': geometry(window._stack.currentWidget()),
                'owned_overlays': [geometry(w, controls=True) for w in overlays],
                'active_modal': geometry(modal, controls=True)}

        def check(widget):
            """Check the current native text or icon after one scroll reveal."""
            window = state['window']
            row = {'class': type(widget).__name__, 'name': widget.objectName(),
                   'rect': widget.rect().getRect(), 'visible': widget.visibleRegion().boundingRect().getRect(),
                   'font': widget.font().toString(), 'startup_observation': startup_snapshot(window)}
            row['ancestors'] = []
            ancestor = widget
            while ancestor is not None:
                row['ancestors'].append({'class': type(ancestor).__name__, 'name': ancestor.objectName(),
                    'actual_size': ancestor.property('size').toTuple(),
                    'in_window': ancestor.rect().translated(ancestor.mapTo(window, QPoint())).getRect(),
                    'minimum_hint': ancestor.minimumSizeHint().toTuple()})
                if ancestor is window:
                    break
                ancestor = ancestor.parentWidget()
            report['checks'].append(row)
            def visible(rect):
                return (QRegion(rect).subtracted(widget.visibleRegion()).isEmpty()
                        and window.rect().contains(rect.translated(widget.mapTo(window, QPoint()))))
            if isinstance(widget, QLabel):
                pixmap = widget.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    shown = pixmap.deviceIndependentSize()
                    placed = QStyle.alignedRect(widget.layoutDirection(), widget.alignment(),
                        QSize(round(shown.width()), round(shown.height())), widget.contentsRect())
                    row.update(icon_size=(shown.width(), shown.height()), dpr=pixmap.devicePixelRatio(),
                               placed=placed.getRect())
                    source_size = None
                    if isinstance(widget, _AppTileIcon):
                        sizes = widget._icon.availableSizes()
                        assert sizes, row
                        source_size = max(sizes, key=lambda s: s.width()*s.height())
                    elif widget.objectName() == 'HeroMark':
                        source_size = _find_logo_pixmap().size()
                    if source_size is not None:
                        assert abs(shown.width()*source_size.height()-shown.height()*source_size.width()) <= max(source_size.width(), source_size.height()), row
                    assert pixmap.devicePixelRatio() == 2. and not placed.isEmpty(), row
                    assert widget.contentsRect().contains(placed) and visible(placed), row
                if widget.text():
                    flags = int(widget.alignment() | (Qt.TextWordWrap if widget.wordWrap() else Qt.TextSingleLine))
                    text = widget.style().itemTextRect(widget.fontMetrics(), widget.contentsRect(),
                        flags, widget.isEnabled(), widget.text())
                    if widget.textFormat() == Qt.RichText or (widget.textFormat() == Qt.AutoText and '<' in widget.text()):
                        document = QTextDocument()
                        document.setDocumentMargin(0)
                        document.setDefaultFont(widget.font())
                        option = QTextOption()
                        option.setWrapMode(QTextOption.WordWrap if widget.wordWrap() else QTextOption.NoWrap)
                        document.setDefaultTextOption(option)
                        document.setHtml(widget.text())
                        document.setTextWidth(widget.contentsRect().width() if widget.wordWrap() else -1)
                        needed = document.size().toSize()
                        text = QStyle.alignedRect(widget.layoutDirection(), widget.alignment(), needed, widget.contentsRect())
                    row.update(text=widget.text(), text_rect=text.getRect())
                    if isinstance(widget, ElidingLabel) and widget.is_elided():
                        row['intentional_elision'] = widget.full_text()
                        assert widget.toolTip() == widget.full_text() and '…' in widget.text(), row
                    persist()
                    assert widget.contentsRect().contains(text) and visible(text), row
            elif isinstance(widget, AppTile):
                glyphs = widget.findChildren(_AppTileIcon)
                for glyph in glyphs:
                    assert not glyph.geometry().intersects(widget.name_label.geometry()), row
            elif isinstance(widget, QPushButton):
                option = QStyleOptionButton()
                option.initFrom(widget)
                option.rect, option.text = widget.rect(), widget.text()
                option.icon, option.iconSize = widget.icon(), widget.property('iconSize')
                for active, feature in ((widget.isDefault(), QStyleOptionButton.DefaultButton),
                        (widget.autoDefault(), QStyleOptionButton.AutoDefaultButton),
                        (widget.isFlat(), QStyleOptionButton.Flat)):
                    if active:
                        option.features |= feature
                field = widget.style().subElementRect(QStyle.SE_PushButtonContents, option, widget)
                text = widget.style().itemTextRect(widget.fontMetrics(), field,
                    int(Qt.AlignCenter | Qt.TextSingleLine), widget.isEnabled(), widget.text())
                row.update(text=widget.text(), field=field.getRect(), text_rect=text.getRect())
                row['icon_scope'] = 'Child QLabel and native QToolButton icons checked separately'
                persist()
                assert widget.icon().isNull(), ('Additional QPushButton icon placement requires native proof', row)
                assert field.contains(text) and visible(text), row
            elif isinstance(widget, QToolButton):
                row['tool_button_style'] = widget.toolButtonStyle().value
                assert not widget.text() or widget.toolButtonStyle() == Qt.ToolButtonIconOnly, ('Unmeasured displayed tool-button caption', row)
                option = QStyleOptionToolButton()
                option.initFrom(widget)
                option.rect, option.icon, option.iconSize = widget.rect(), widget.icon(), widget.property('iconSize')
                option.toolButtonStyle = widget.toolButtonStyle()
                option.text, option.arrowType = widget.text(), widget.arrowType()
                field = widget.style().subControlRect(QStyle.CC_ToolButton, option, QStyle.SC_ToolButton, widget)
                if widget.arrowType() != Qt.NoArrow:
                    row.update(native_arrow=widget.arrowType().value, field=field.getRect())
                    persist()
                    assert not field.isEmpty() and widget.rect().contains(field) and visible(field), row
                    return
                pixmap = widget.icon().pixmap(widget.property('iconSize'), 2.)
                shown = pixmap.deviceIndependentSize()
                placed = QStyle.alignedRect(widget.layoutDirection(), Qt.AlignCenter,
                    QSize(round(shown.width()), round(shown.height())), field)
                row.update(field=field.getRect(), placed=placed.getRect(), dpr=pixmap.devicePixelRatio(),
                           actual_icon_size=widget.property('iconSize').toTuple(),
                           source_sizes=[size.toTuple() for size in widget.icon().availableSizes()])
                persist()
                assert not pixmap.isNull() and pixmap.devicePixelRatio() == 2., row
                assert field.contains(placed) and visible(placed), row
            else:
                assert False, ('Unmeasured native button type', row)
            persist()

        def tick():
            """Keep inspection inside its original single twenty-second budget."""
            try:
                assert time.perf_counter()-state['started'] < 20, 'Home inspection exceeded20s'
                window = state['window']
                handle = window.windowHandle()
                assert handle.size() == QSize(1200, 640), handle.size()
                observed_startup = startup_snapshot(window)
                if observed_startup != state.get('last_startup'):
                    report.setdefault('startup_history', []).append(
                        {'elapsed': time.perf_counter()-state['started'], **observed_startup})
                    state['last_startup'] = observed_startup
                    persist()
                if observed_startup['sequence_running']:
                    return
                blockers = [entry for entry in observed_startup['owned_overlays']
                            if entry and entry.get('visible') and
                                ('TourOverlay' in entry.get('class', '') or observed_startup['active_modal'] == entry)]
                assert not blockers, ('Normal startup has an interactive owned overlay; no automatic dismissal', blockers)
                if not state.get('font_witness_checked'):
                    witness = state['font_witness']
                    fitter = witness.parentWidget()._text_fit
                    if fitter._timer.isActive() or fitter._busy:
                        return
                    selected = fitter._selected.get(witness)
                    assert selected is not None, 'Missing actual selected Hero subtitle font'
                    selected_font = selected['font']
                    report['startup_witness'].update(
                        selected_subtitle_font=selected_font.toString(),
                        selected_subtitle_pixels=selected_font.pixelSize(),
                        selected_subtitle_base_sheet=selected['base'],
                        actual_subtitle_font=witness.font().toString(),
                        actual_subtitle_pixels=witness.font().pixelSize())
                    persist()
                    assert selected_font.pixelSize() == report['startup_witness']['expected_subtitle_pixels']
                    assert QFontInfo(selected_font).family() == QFontInfo(witness.font()).family() == 'Open Sans'
                    assert 0 < witness.font().pixelSize() <= selected_font.pixelSize()
                    state['font_witness_checked'] = True
                if 'widgets' not in state:
                    report['external_native_menu_objects'] = [w.objectName() for w in window.findChildren(QWidget)
                        if w.isVisible() and native_menu_child(w) and isinstance(w, (QLabel, QAbstractButton))]
                    state['widgets'] = [w for w in window.findChildren(QWidget)
                        if w.isVisible() and w.window() is window and not native_menu_child(w)
                            and (isinstance(w, QLabel) and (w.text() or (w.pixmap() is not None and not w.pixmap().isNull()))
                            or isinstance(w, QAbstractButton))]
                    state['index'] = 0
                    report['target_count'] = len(state['widgets'])
                if state['index'] == len(state['widgets']):
                    assert report['checks'] and any(row.get('icon_size') for row in report['checks'])
                    close()
                    return
                widget = state['widgets'][state['index']]
                current = signature(window, widget)
                if current is None:
                    state['previous'] = None
                    return
                if not state.get('revealed'):
                    if state.get('previous') != current:
                        state['previous'] = current
                        return
                    parent = widget.parentWidget()
                    while parent is not None and parent is not window:
                        if isinstance(parent, QScrollArea):
                            parent.ensureWidgetVisible(widget, 0, 0)
                        parent = parent.parentWidget()
                    state['revealed'] = True
                    state['after_reveal'] = signature(window, widget)
                    return
                assert state['after_reveal'] is not None and current == state['after_reveal'], 'Layout changed after the single reveal'
                try:
                    check(widget)
                except AssertionError:
                    report['errors'].append(traceback.format_exc())
                    persist()
                state['index'] += 1
                state['revealed'] = False
                if state['index'] < len(state['widgets']):
                    state['previous'] = signature(window, state['widgets'][state['index']])
            except Exception:
                report['errors'].append(traceback.format_exc())
                persist()
                close()

        def prepare(entry):
            """Set the real client minimum only after normal Home readiness."""
            if entry.get('detail') != '__home__' or state:
                return
            app = QApplication.instance()
            windows = [w for w in app.topLevelWidgets() if isinstance(w, MainWindow)]
            assert len(windows) == 1
            window = windows[0]
            handle = window.windowHandle()
            state.update(window=window, started=time.perf_counter())
            report['initial_startup_observation'] = startup_snapshot(window)
            witness = window._startup.findChild(QLabel, 'HeroSubtitle')
            expected_subtitle = gui_scale.scale_int(theme.font_px('subtitle', 2.), 2.)
            report['startup_witness'] = {
                'platform': app.platformName(), 'exposed': handle.isExposed(),
                'screen': handle.screen().geometry().getRect(),
                'work': handle.screen().availableGeometry().getRect(), 'dpr': handle.devicePixelRatio(),
                'live_gui': gui_scale.current_scale(), 'saved_gui': preferences.get_gui_scale(),
                'saved_font': preferences.get_font_scale(), 'app_font': app.font().toString(),
                'app_family': QFontInfo(app.font()).family(), 'declared_role': 'subtitle',
                'expected_subtitle_pixels': expected_subtitle,
                'actual_subtitle_font': witness.font().toString() if witness is not None else None,
                'actual_subtitle_pixels': witness.font().pixelSize() if witness is not None else None}
            persist()
            assert app.platformName() == 'cocoa' and handle.isExposed()
            assert handle.screen().geometry().size() == QSize(1440, 900)
            assert handle.screen().availableGeometry().size() == QSize(1440, 870)
            assert handle.devicePixelRatio() == 2.
            assert gui_scale.current_scale() == preferences.get_gui_scale() == 2.
            assert preferences.get_font_scale() == 2.
            assert QFontInfo(app.font()).family() == 'Open Sans'
            assert witness is not None
            state['font_witness'] = witness
            window.showNormal()
            handle.resize(QSize(1200, 640))
            report['native'] = {'client': handle.size().toTuple(), 'work': handle.screen().availableGeometry().getRect(),
                                'dpr': handle.devicePixelRatio(), 'app_font': app.font().toString(),
                                'styled_witness_font': witness.font().toString()}
            assert handle.screen().availableGeometry().contains(handle.frameGeometry())
            timer = QTimer(window)
            timer.setInterval(50)
            timer.timeout.connect(tick)
            state['timer'] = timer
            timer.start()
            persist()

        def ready(entry):
            """Record callback failures and close rather than silently timing out."""
            try:
                prepare(entry)
            except Exception:
                report['errors'].append(traceback.format_exc())
                persist()
                if 'window' in state:
                    close()
                else:
                    windows = [w for w in QApplication.instance().topLevelWidgets() if isinstance(w, MainWindow)]
                    assert len(windows) == 1
                    state['window'] = windows[0]
                    close()

        from PySide6.QtWidgets import QWidget
        timing.subscribe_readiness(ready)
        from spacr.qt import run
        result = run(['--no-setup', '--fresh'])
        timing.unsubscribe_readiness(ready)
        from spacr.qt.widgets import ambient
        report.update(normal_return=result, compiler_threads=[t.name for t in ambient._COMPILER_THREADS if t.is_alive()],
                      compiler_helper_present=ambient._PRIME_PROCESS is not None)
        persist()
        assert result == 0 and not report['errors'] and report.get('normal_close_accepted'), report
        assert not report['compiler_threads'] and not report['compiler_helper_present'], report
    ''')
    settings = (tmp_path / "settings").resolve()
    settings.mkdir()
    expected_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    env = os.environ.copy()
    env.update(QT_QPA_PLATFORM="cocoa", SPACR_ISOLATED_PROFILE="1", SPACR_HOME=str(tmp_path / "profile"),
               SPACR_MENU_TEST_SETTINGS=str(tmp_path.resolve()), SPACR_HOME_TEST_ROOT=str(root),
               SPACR_HOME_TEST_HEAD=expected_head,
               SPACR_HOME_TEST_OUTPUT=str(tmp_path / "native-home.json"),
               XDG_CONFIG_HOME=str(settings), XDG_CACHE_HOME=str(tmp_path / "cache"),
               MPLCONFIGDIR=str(tmp_path / "cache" / "matplotlib"),
               SPACR_TIMING_LOG=str(tmp_path / "timing.txt"),
               NUMBA_CACHE_DIR=str(tmp_path / "numba"), SPACR_NO_NETWORK="1", SPACR_NO_SETUP="1", SPACR_TIMING="1")
    for key in tuple(env):
        if key.startswith("SPACR_BENCHMARK_"):
            env.pop(key)
    result = subprocess.run([sys.executable, "-c", child], cwd=root, env=env,
                            capture_output=True, text=True, timeout=180)
    (tmp_path / "native-home-stdout.txt").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "native-home.json").read_text())
    assert not report["errors"] and report["normal_close_accepted"]


@pytest.mark.parametrize("font_scale", (2., .1))
def test_main_window_live_scaled_floor_fits_native_work_area_without_losing_baseline(qtbot, qapp, font_scale):
    from spacr.qt import gui_scale, preferences

    old_gui = gui_scale.current_scale()
    old_pref_gui = preferences.get_gui_scale()
    old_font = preferences.get_font_scale()
    old_sheet, old_app_font, old_palette = qapp.styleSheet(), qapp.font(), qapp.palette()
    window = None
    try:
        gui_scale.install_scaling_layer()
        preferences.set_gui_scale(2.)
        gui_scale.set_gui_scale_live(2.)
        preferences.set_font_scale(font_scale)
        preferences.apply_preferences_to_app(qapp)
        window = app_mod.MainWindow(initial_app="__home__")
        qtbot.addWidget(window)
        child = QWidget(window)
        child.setMinimumSize(1600, 1000)
        child.hide()
        app_mod._show_at_launch(window)
        qtbot.waitExposed(window)
        # Resize a restored window; a maximized offscreen window manager
        # immediately reapplies the desktop size to native resize requests.
        window.showNormal()
        qapp.processEvents()
        for scale in (2., .1, 2.):
            gui_scale.set_gui_scale_live(scale)
            app_mod._fit_to_the_work_area(window)
            qapp.processEvents()
            handle = window.windowHandle()
            frame, work = handle.frameGeometry(), handle.screen().availableGeometry()
            actual = handle.minimumSize()
            assert actual.width() <= work.width() and actual.height() <= work.height(), (scale, actual, work)
            assert work.contains(frame), (scale, frame, work)
            if scale > 1.:
                assert actual.width() <= 1200 and actual.height() <= 640, (scale, actual)
                from PySide6.QtCore import QSize
                margins = handle.frameMargins()
                base = QSize(min(handle.width(), work.width() - margins.left() - margins.right()),
                             min(handle.height(), work.height() - margins.top() - margins.bottom()))
                assert base.width() >= actual.width() and base.height() >= actual.height()
                # The offscreen desktop can be only 800 px wide: its minimum
                # fills that axis, leaving resize room on the other axis only.
                smaller = (base - QSize(20, 20)).expandedTo(actual)
                for target in (smaller, base, smaller):
                    handle.resize(target)
                    qtbot.waitUntil(lambda: handle.size() == target, timeout=2000)
                    assert work.contains(handle.frameGeometry())
            assert window._gs_geo['minw'] == 1200 and window._gs_geo['minh'] == 640
            assert handle.maximumSize().width() == handle.maximumSize().height() == 16777215
            assert gui_scale._actual_minimum_size(child).width() == gui_scale.scale_int(1600, scale)
            assert gui_scale._actual_minimum_size(child).height() == gui_scale.scale_int(1000, scale)
    finally:
        try:
            if window is not None:
                window.close()
        finally:
            preferences.set_font_scale(old_font)
            preferences.set_gui_scale(old_pref_gui)
            gui_scale.set_gui_scale_live(old_gui)
            qapp.setStyleSheet(old_sheet)
            qapp.setFont(old_app_font)
            qapp.setPalette(old_palette)


@pytest.mark.parametrize("size", WORK_AREAS)
def test_make_masks_fits_with_every_group_and_the_image_whole(
        qtbot, qt_theme_applied, folder, size):
    from spacr.qt.preferences import get_folded_panels

    window, screen = _open_make_masks(qtbot, folder, size)
    width, height = size
    minimum = window.minimumSize()
    assert minimum.width() <= width and minimum.height() <= height
    hint = window.minimumSizeHint()
    assert hint.width() <= width and hint.height() <= height
    assert screen._view_pane.is_collapsed("Shortcuts")
    assert "make_masks/Shortcuts" not in get_folded_panels()
    groups = (screen._tools_group, screen._annotation_group,
              screen._io_group, screen._navigation_group)
    for group in groups:
        assert _whole(window, group), group.title_label.text()
        for control in group.findChildren(QAbstractButton):
            if control.isVisibleTo(group):
                assert _whole(window, control), (group.title_label.text(),
                                                 control.text())
    canvas = _rect_in(window, screen._canvas)
    assert _whole(window, screen._canvas)
    assert canvas.height() >= 120 and canvas.width() >= 600
    assert _whole(window, screen._image_footer)
    assert canvas.bottom() < _rect_in(window, screen._io_group).top()
    for label in screen._image_footer.findChildren(QLabel):
        if label.isVisibleTo(screen._image_footer):
            assert _whole(window, label)
    shots = os.environ.get("SPACR_TEST_SCREENSHOTS")
    if shots:
        path = Path(shots) / f"make-masks-{width}x{height}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert window.grab().save(str(path))
    _close(screen)


@pytest.mark.parametrize("language,size,font_scale", [
    pytest.param("en", WORK_AREAS[0], .9, id="en-small-font90"),
    pytest.param("en", WORK_AREAS[1], .9, id="en-medium-font90"),
    pytest.param("de", WORK_AREAS[0], .9, id="de-small-font90"),
    pytest.param("de", WORK_AREAS[1], .9, id="de-medium-font90"),
    pytest.param("de", WORK_AREAS[0], 2., id="de-small-font200"),
])
def test_overview_reset_ruler_and_active_tab_fit_at_ninety_percent(
        qtbot, qt_theme_applied, folder, monkeypatch, size, language, font_scale):
    from PySide6.QtWidgets import QToolButton
    from PySide6.QtGui import QTextDocument
    from spacr.qt import gui_scale, preferences, theme
    from spacr.qt.i18n import retranslate_widget_tree, tr

    previous = gui_scale.current_scale()
    previous_saved_gui = preferences.get_gui_scale()
    previous_font = preferences.get_font_scale()
    monkeypatch.setenv("SPACR_LANGUAGE", language)
    preferences.set_gui_scale(.9)
    preferences.set_font_scale(font_scale)
    gui_scale.install_scaling_layer()
    gui_scale.set_gui_scale_live(.9)
    assert gui_scale.current_scale() == pytest.approx(.9)
    screen = None
    try:
        window, screen = _open_make_masks(qtbot, folder, size)
        preferences.apply_preferences_to_app(QApplication.instance())
        retranslate_widget_tree(window, language)
        QApplication.processEvents()
        assert (window.width(), window.height()) == size
        overview = screen.findChild(QToolButton, "MakeMasksFieldOverview")
        assert overview is not None
        caption = tr("Field-of-view overview")
        assert overview.accessibleName() == caption
        tooltip = QTextDocument()
        tooltip.setHtml(overview.toolTip())
        assert tooltip.toPlainText().splitlines()[-2:] == [caption, "API"]
        assert overview.toolButtonStyle() == Qt.ToolButtonIconOnly
        assert not overview.icon().isNull()
        assert screen._io_group.isAncestorOf(overview)
        assert _rect_in(window, screen._io_group).contains(_rect_in(window, overview))
        assert overview.font().pixelSize() == gui_scale.scale_int(theme.font_px("body", font_scale), .9)
        tabs = screen._view_tabs
        bar = tabs.tabBar()
        for index, caption in enumerate(("Mask", "Cell probability", "Flows")):
            assert tabs.tabText(index) == tr(caption, language=language)
        display = window.screen()
        print("707 geometry", QApplication.platformName(), language, size,
              "client", (window.width(), window.height()),
              "display", display.geometry().getRect(), "work", display.availableGeometry().getRect(),
              "DPR", display.devicePixelRatio(), "font_px", overview.font().pixelSize())
        buttons = (screen._btn_reset_zoom, screen._btn_ruler, overview)
        for index in range(tabs.count()):
            tabs.setCurrentIndex(index)
            QApplication.processEvents()
            assert (window.width(), window.height()) == size
            assert _whole(window, bar)
            selected = bar.tabRect(index)
            assert bar.rect().contains(selected), (language, size, index)
            selected = QRect(bar.mapTo(window, selected.topLeft()), selected.size())
            rectangles = []
            for button in buttons:
                assert _whole(window, button), (language, size, button.text())
                assert button.width() >= button.sizeHint().width()
                assert button.height() >= button.sizeHint().height()
                rect = _rect_in(window, button)
                assert not rect.intersects(selected)
                assert not any(rect.intersects(other) for other in rectangles)
                rectangles.append(rect)
            assert _whole(window, screen._image_footer)
            content = _rect_in(window, tabs.currentWidget())
            assert _rect_in(window, screen._image_footer).top() > content.bottom()
    finally:
        if screen is not None:
            _close(screen)
        preferences.set_gui_scale(previous_saved_gui)
        preferences.set_font_scale(previous_font)
        gui_scale.set_gui_scale_live(previous)


def test_a_side_column_opened_by_hand_stays_open_while_narrow(
        qtbot, qt_theme_applied, folder):
    window, screen = _open_make_masks(qtbot, folder, (1280, 680))
    pane = screen._view_pane
    assert pane.is_collapsed("Shortcuts")
    pane.toggle_pane("Shortcuts", by_user=True)
    assert not pane.is_collapsed("Shortcuts")
    window.resize(1300, 680)
    qtbot.waitUntil(lambda: window.width() == 1300)
    assert not pane.is_collapsed("Shortcuts")
    _close(screen)


def test_a_wide_screen_keeps_the_side_column_open(
        qtbot, qt_theme_applied, folder):
    window, screen = _open_make_masks(qtbot, folder, (1920, 1050))
    assert not screen._view_pane.is_collapsed("Shortcuts")
    assert screen._shortcut_panel.isVisible()
    window.resize(1280, 680)
    qtbot.waitUntil(lambda: screen._view_pane.is_collapsed("Shortcuts"))
    window.resize(1920, 1050)
    qtbot.waitUntil(
        lambda: not screen._view_pane.is_collapsed("Shortcuts"))
    _close(screen)


def test_a_folded_side_column_does_not_hold_the_screen_tall(
        qtbot, qt_theme_applied, folder):
    window, screen = _open_make_masks(qtbot, folder, (1280, 680))
    folded = screen.minimumSizeHint().height()
    screen._view_pane.toggle_pane("Shortcuts", by_user=True)
    qtbot.waitUntil(lambda: screen.minimumSizeHint().height() > folded,
                    timeout=2000)
    assert window.minimumSize().height() <= 640
    _close(screen)


@pytest.mark.parametrize("work", [QRect(0, 0, 1280, 680),
                                  QRect(0, 40, 1366, 728)])
def test_an_oversized_window_is_clamped_into_the_work_area(qtbot, work):
    window = QWidget()
    qtbot.addWidget(window)
    window.setMinimumSize(1200, 720)
    window.show()
    window.setGeometry(400, 300, 1920, 1080)
    assert app_mod._fit_to_the_work_area(window, _Screen(work))
    frame = window.frameGeometry()
    assert work.contains(frame)
    from spacr.qt.gui_scale import _actual_minimum_size

    assert _actual_minimum_size(window).height() <= work.height()


def test_a_window_that_fits_is_left_where_it_is(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    window.show()
    window.setGeometry(100, 100, 1200, 640)
    before = window.geometry()
    assert not app_mod._fit_to_the_work_area(
        window, _Screen(QRect(0, 0, 1920, 1050)))
    assert window.geometry() == before


def test_an_unplaced_window_is_sized_but_left_for_the_window_manager(qtbot):
    window = QWidget()
    qtbot.addWidget(window)
    from PySide6.QtCore import QSize
    from spacr.qt.gui_scale import _resize_actual

    _resize_actual(window, QSize(1200, 720))
    assert app_mod._fit_to_the_work_area(
        window, _Screen(QRect(0, 32, 1280, 680)))
    assert window.height() <= 680
    assert not window.testAttribute(Qt.WidgetAttribute.WA_Moved)


@pytest.mark.parametrize("scale", [.5, 2.])
def test_generic_window_clamping_uses_actual_pixels_at_nondefault_scale(qtbot, scale):
    from spacr.qt import gui_scale

    old_scale = gui_scale.current_scale()
    window = None
    try:
        gui_scale.install_scaling_layer()
        gui_scale.set_gui_scale_live(scale)
        window = QWidget()
        qtbot.addWidget(window)
        window.setMinimumSize(1200, 720)
        window.show()
        window.setGeometry(400, 300, 1920, 1080)
        work = QRect(0, 32, 1280, 680)
        frame, inner = window.frameGeometry(), window.geometry()
        border = frame.size() - inner.size()
        assert app_mod._fit_to_the_work_area(window, _Screen(work))
        assert window.geometry().size() == work.size() - border
        assert work.contains(window.frameGeometry())
        minimum = gui_scale._actual_minimum_size(window)
        assert minimum.width() <= window.width() and minimum.height() <= window.height()
    finally:
        if window is not None:
            window.close()
        gui_scale.set_gui_scale_live(old_scale)


def test_a_framed_launch_on_a_720p_screen_fits(qtbot, qt_theme_applied,
                                               monkeypatch):
    from spacr.qt import preferences

    preferences.set_borderless_window(False)
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    work = QRect(0, 0, 1280, 680)
    monkeypatch.setattr(window, "screen", lambda: _Screen(work))
    app_mod._show_at_launch(window)
    assert window.isVisible() and not window.isMaximized()
    assert window.width() <= work.width()
    assert window.height() <= work.height()
    window.close()


def test_restoring_a_maximised_window_lands_inside_the_work_area(
        qtbot, qt_theme_applied, monkeypatch):
    window = app_mod.MainWindow()
    qtbot.addWidget(window)
    window.show()
    window.setGeometry(200, 200, 1900, 1000)
    window.showMaximized()
    work = QRect(0, 0, 1280, 680)
    monkeypatch.setattr(window, "screen", lambda: _Screen(work))
    window._toggle_maximised()
    qtbot.waitUntil(lambda: not window.isMaximized())
    assert work.contains(window.frameGeometry())


def _module_keys():
    return [entry[0] for entry in app_mod.APPS] + ["make_masks"]


@pytest.mark.parametrize("size", WORK_AREAS)
def test_every_module_screen_fits_without_being_squeezed(
        qtbot, qt_theme_applied, size):
    """A page whose minimum exceeds the stack has its controls cut off."""
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    window.resize(*size)
    window.show()
    qtbot.waitUntil(lambda: (window.width(), window.height()) == size)
    squeezed = {}
    for key in _module_keys():
        window._on_nav_selected(key)
        qtbot.wait(20)
        QApplication.processEvents()
        page = window._stack.currentWidget()
        stack = window._stack.size()
        hint = page.minimumSizeHint()
        if hint.width() > stack.width() or hint.height() > stack.height():
            squeezed[key] = (hint.toTuple(), stack.toTuple())
        scroll = page.findChild(app_mod._CrampedScroll)
        if scroll is not None:
            content = scroll.widget()
            need = content.minimumSizeHint()
            assert content.width() >= need.width(), key
            assert content.height() >= need.height(), key
    assert not squeezed
    window.close()


def _geometry(page):
    skip = {"CrampedScreenScroll", "CrampedScreenContent",
            "qt_scrollarea_viewport", "qt_scrollarea_hcontainer",
            "qt_scrollarea_vcontainer"}
    rows = []
    for widget in page.findChildren(QWidget):
        if widget.objectName() in skip or not widget.isVisible():
            continue
        rows.append((type(widget).__name__, widget.objectName(),
                     QRect(widget.mapTo(page, QPoint(0, 0)),
                           widget.size()).getRect()))
    return sorted(rows)


@pytest.mark.parametrize("key", sorted(app_mod._CRAMPED_SCREENS))
def test_a_roomy_screen_lays_out_exactly_as_before_it_scrolled(
        qtbot, qt_theme_applied, key):
    """At 1920 x 1050 the wrapper changes no widget's place or size."""
    window = app_mod.MainWindow(initial_app="__home__")
    qtbot.addWidget(window)
    page = window._build_screen_timed(key)
    host = QWidget()
    qtbot.addWidget(host)
    page.setParent(host)
    host.resize(1920, 995)
    page.setGeometry(0, 0, 1920, 995)
    host.show()
    qtbot.wait(60)
    QApplication.processEvents()
    before = _geometry(page)
    assert app_mod._scroll_when_cramped(page)
    qtbot.wait(60)
    QApplication.processEvents()
    scroll = page.findChild(app_mod._CrampedScroll)
    assert scroll.widget().size() == page.size()
    assert not scroll.horizontalScrollBar().isVisible()
    assert not scroll.verticalScrollBar().isVisible()
    assert _geometry(page) == before
    host.close()


_TESTED_ROOT = Path(__file__).resolve().parents[2]


def _tested_tree(env):
    """Make a fresh probe process import this tree's spacr, not an install.

    :param env: the probe's environment, changed in place.
    :returns: ``env``.
    """
    env["SPACR_TESTED_ROOT"] = str(_TESTED_ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_TESTED_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    return env


GUARD = 'import os as _os, spacr as _spacr\nassert _os.path.realpath(_spacr.__file__).startswith(_os.path.realpath(_os.environ["SPACR_TESTED_ROOT"]) + _os.sep), ("probe imported spacr from " + _spacr.__file__)\n'


_DPR_SCRIPT = r"""
import sys
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget
app = QApplication([])
from spacr.qt import app as app_mod
page = QWidget()
layout = QVBoxLayout(page)
tall = QWidget()
tall.setMinimumSize(900, 1200)
layout.addWidget(tall)
app_mod._scroll_when_cramped(page)
page.resize(1280, 625)
page.show()
for _ in range(5):
    app.processEvents()
scroll = page.findChild(app_mod._CrampedScroll)
small = (scroll.verticalScrollBar().isVisible(),
         scroll.widget().height() >= 1200)
page.resize(1920, 995)
for _ in range(5):
    app.processEvents()
roomy = (scroll.verticalScrollBar().isVisible(),
         scroll.widget().size() == page.size())
print(page.devicePixelRatioF(), small, roomy)
"""


def test_the_threshold_is_in_logical_pixels_at_device_pixel_ratio_two():
    import subprocess
    import sys

    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", QT_SCALE_FACTOR="2",
               CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-c", GUARD + _DPR_SCRIPT], env=_tested_tree(env),
                          capture_output=True, text=True, timeout=240)
    assert done.returncode == 0, done.stderr[-2000:]
    line = done.stdout.strip().splitlines()[-1]
    assert line == "2.0 (True, True) (False, True)", line
