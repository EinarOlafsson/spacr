"""Record native Classify family choices and its five current nested routes."""
from capture_barcode_saved_plots import launch


def record_overview(app, window, captures, capture, settle, write_json):
    from PySide6.QtCore import Qt, QPoint, QEvent
    from PySide6.QtGui import QHelpEvent
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QAbstractButton, QComboBox, QToolTip
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
    click(max(tiles, key=lambda w: w.width() * w.height())); settle(1)
    host = window._screens['classify_merged']
    capture('01_classify_home_route')
    selector = host._settings_model._widgets.get('classifier_family')
    if not isinstance(selector, QComboBox) or selector.currentData() != 'cv':
        raise ValueError('Expected the actual initial CV family selector')
    click(selector)
    if not selector.view().isVisible():
        raise ValueError('The actual family choices popup did not open')
    capture('02_family_choices', desktop=True)
    index = selector.findData('ml')
    if index < 0:
        raise ValueError('The actual tabular family choice is missing')
    QTest.keyClick(selector.view(), Qt.Key_Home)
    for _ in range(index):
        QTest.keyClick(selector.view(), Qt.Key_Down)
    QTest.keyClick(selector.view(), Qt.Key_Return); settle(.7)
    if host._settings_model.collect().get('classifier_family') != 'ml':
        raise ValueError('Visible family selection did not reach ML')
    capture('03_tabular_family')
    click(selector); QTest.keyClick(selector.view(), Qt.Key_Home)
    for _ in range(selector.findData('cv')):
        QTest.keyClick(selector.view(), Qt.Key_Down)
    QTest.keyClick(selector.view(), Qt.Key_Return); settle(.7)
    if host._settings_model.collect().get('classifier_family') != 'cv':
        raise ValueError('The original family was not restored')
    capture('04_cv_restored')
    buttons = [w for w in host.findChildren(FoldButton) if w.isVisible()]
    if len(buttons) != 5 or {w.app_key for w in buttons} != set(FOLDED_APPS):
        raise ValueError('The five expected Classify nested routes changed')
    rows = []
    for index, key in enumerate(FOLDED_APPS):
        button = next(w for w in buttons if w.app_key == key)
        QTest.mouseMove(window, QPoint(400, 800)); settle(.15)
        QTest.mouseMove(button, button.rect().center()); settle(1.4)
        # Private Xvfb has no window manager to activate native hover timers.
        # Deliver Qt's real help event to the real button; its production
        # filters, text and popup rendering remain in charge. No popup text
        # is supplied or fabricated by the recorder.
        simulated_help_event = not QToolTip.isVisible() or QToolTip.text() != button.toolTip()
        if simulated_help_event:
            point = button.rect().center()
            app.sendEvent(button, QHelpEvent(QEvent.ToolTip, point, button.mapToGlobal(point)))
            settle(.4)
        if not QToolTip.isVisible() or QToolTip.text() != button.toolTip():
            raise ValueError('The actual hover tooltip did not appear: ' + key)
        capture('05_' + key + '_tooltip', desktop=True)
        rows.append(dict(key=key, tooltip=button.toolTip(),
                         displayed_tooltip=QToolTip.text(),
                         simulated_native_help_event=simulated_help_event,
                         x=button.mapTo(window, QPoint(0, 0)).x()))
    QTest.mouseMove(window, QPoint(400, 800)); settle(.5)
    click(next(w for w in buttons if w.app_key == 'feature_explorer')); settle(.8)
    panels = [w for w in app.allWidgets() if isinstance(w, FeatureExplorerScreen) and w.isVisible()]
    if len(panels) != 1:
        raise ValueError('The actual nested Feature Explorer did not open')
    capture('06_feature_explorer_nested')
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
