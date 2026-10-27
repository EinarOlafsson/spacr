"""Record native Classify family choices and its five current nested routes."""
import capture_ui
from capture_barcode_saved_plots import launch
from capture_ui_drv_a import Rec


def _actual_tooltip_matches(visible, displayed, authored):
    from spacr.qt.tooltip_policy import _wrapped_tooltip
    return bool(visible and authored and displayed and displayed == _wrapped_tooltip(authored))


def record_overview(app, window, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt, QPoint, QEvent
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QToolTip
    from spacr.qt.tooltip_policy import _wrapped_tooltip
    from spacr.qt.widgets.fold_strip import FoldButton
    from spacr.qt.screens.classify import FOLDED_APPS
    from spacr.qt.screens.feature_explorer import FeatureExplorerScreen

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled() or widget.visibleRegion().isEmpty():
            raise ValueError('The actual overview control is unavailable')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.4)

    tiles = [w for w in window.findChildren(QAbstractButton) if w.isVisible() and
             (w.property('moduleAppKey') == 'classify_merged' or w.property('navKey') == 'classify_merged')]
    if not tiles:
        raise ValueError('No native Home Classify route')
    # Host: the clip starts at Home; the pointer clicks the Classify tile.
    with Rec(capture, '01_classify_home_route', settle):
        click(max(tiles, key=lambda w: w.width() * w.height())); settle(1)
    host = window._screens['classify_merged']
    selector = host._settings_model._widgets.get('classifier_family')
    if not isinstance(selector, QComboBox) or selector.currentData() != 'cv':
        raise ValueError('Expected the actual initial CV family selector')
    index = selector.findData('ml')
    if index < 0:
        raise ValueError('The actual tabular family choice is missing')
    # Host: the clip holds the pointer's click that drops the list down.
    with Rec(capture, '02_family_choices', settle, desktop=True):
        click(selector)
        if not selector.view().isVisible():
            raise ValueError('The actual family choices popup did not open')
    with Rec(capture, '03_tabular_family', settle):
        QTest.keyClick(selector.view(), Qt.Key_Home)
        for _ in range(index):
            QTest.keyClick(selector.view(), Qt.Key_Down)
        QTest.keyClick(selector.view(), Qt.Key_Return); settle(.7)
        if host._settings_model.collect().get('classifier_family') != 'ml':
            raise ValueError('Visible family selection did not reach ML')
    with Rec(capture, '04_cv_restored', settle):
        click(selector); QTest.keyClick(selector.view(), Qt.Key_Home)
        for _ in range(selector.findData('cv')):
            QTest.keyClick(selector.view(), Qt.Key_Down)
        QTest.keyClick(selector.view(), Qt.Key_Return); settle(.7)
        if host._settings_model.collect().get('classifier_family') != 'cv':
            raise ValueError('The original family was not restored')
    buttons = [w for w in host.findChildren(FoldButton) if w.isVisible()]
    if len(buttons) != 5 or {w.app_key for w in buttons} != set(FOLDED_APPS):
        raise ValueError('The five expected Classify nested routes changed')
    rows = []
    for index, key in enumerate(FOLDED_APPS):
        button = next(w for w in buttons if w.app_key == key)
        # Host: the clip starts with the pointer elsewhere, then it glides to
        # the button and rests until the real tooltip shows.
        tip = Rec(capture, '05_' + key + '_tooltip', settle, desktop=True).start()
        QTest.mouseMove(button, button.rect().center()); settle(2.6)  # tooltip_policy waits ~2 s
        # Private Xvfb has no window manager to activate native hover timers.
        # Deliver Qt's real help event to the real button; its production
        # filters, text and popup rendering remain in charge. No popup text
        # is supplied or fabricated by the recorder. (The host's real pointer
        # shows the genuine tooltip, so this fallback is Xvfb only.)
        # Match the production wrapping of this button's own complete text.
        tip_ok = lambda: _actual_tooltip_matches(QToolTip.isVisible(), QToolTip.text(), button.toolTip())
        simulated_help_event = not tip_ok()
        if simulated_help_event:
            point = button.rect().center()
            app.sendEvent(button, QHelpEvent(QEvent.ToolTip, point, button.mapToGlobal(point)))
            settle(2.6)
        if not tip_ok():
            tip.abort()
            raise ValueError('The actual hover tooltip did not appear: ' + key)
        tip.stop()
        rows.append(dict(key=key, tooltip=_wrapped_tooltip(button.toolTip()),
                         authored_tooltip=button.toolTip(),
                         displayed_tooltip=QToolTip.text(),
                         simulated_native_help_event=simulated_help_event,
                         x=button.mapTo(window, QPoint(0, 0)).x()))
    QTest.mouseMove(window, QPoint(400, 800)); settle(.5)
    with Rec(capture, '06_feature_explorer_nested', settle):
        click(next(w for w in buttons if w.app_key == 'feature_explorer')); settle(.8)
        panels = [w for w in app.allWidgets() if isinstance(w, FeatureExplorerScreen) and w.isVisible()]
        if len(panels) != 1:
            raise ValueError('The actual nested Feature Explorer did not open')
    if host._worker_thread_is_running() or host.active_jobs() or panels[0].active_jobs():
        raise ValueError('Navigation-only recording unexpectedly started work')
    write_json(captures / 'scientific_acceptance.json', dict(accepted=True,
               scope='Native navigation and family choices only; no model or dataset operation',
               family_restored='cv', folds=rows, nested_feature_explorer_opened=True,
               model_started=False, provider_contacted=False, app_source_modified=False,
               published=False))


if __name__ == '__main__':
    raise SystemExit(launch('classify_merged', 'classify_main_family_and_folds_v4',
                           '--classify-overview', 120))
