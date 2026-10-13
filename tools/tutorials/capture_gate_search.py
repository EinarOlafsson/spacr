"""Exercise the actual bounded gate search, retaining refusals as outcomes."""
from __future__ import annotations


def record_search(app, screen, captures, capture, settle, write_json, fill, choose, file_dialog, work):
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QMessageBox

    QTest.mouseClick(screen.gates._mode_buttons['2D'], Qt.LeftButton)
    choose(screen._x, 'cell_area')
    choose(screen._y, 'cell_channel_1_mean_intensity')
    tree = screen.gates.tree.tree
    # Deleting only the temporary in-memory gates prevents a clustering parent
    # from silently narrowing the populations. The two-gate strategy is saved.
    while not screen.gates.gates.is_empty:
        name = screen.gates.gates.names[0]
        matches = tree.findItems(name, Qt.MatchExactly | Qt.MatchRecursive, 0)
        if len(matches) != 1:
            raise RuntimeError('The current gate is absent from the actual hierarchy')
        QTest.mouseClick(tree.viewport(), Qt.LeftButton, pos=tree.visualItemRect(matches[0]).center())
        QTest.mouseClick(screen.gates.tree._remove, Qt.LeftButton)
        settle(0.3)
        if name in screen.gates.gates.names:
            raise RuntimeError('The temporary gate was not actually removed')
    QTest.mouseClick(screen.side_tabs.tabBar(), Qt.LeftButton,
                     pos=screen.side_tabs.tabBar().tabRect(1).center())
    fill(screen.search._eps.lineEdit(), '0.5')
    fill(screen.search._min_samples.lineEdit(), '20')
    if not screen.search._scale.isChecked():
        QTest.mouseClick(screen.search._scale, Qt.LeftButton)
    if not screen.search._walk.isChecked():
        QTest.mouseClick(screen.search._walk, Qt.LeftButton)
    fill(screen.search._walk_steps.lineEdit(), '5')
    expected = {'cluster_eps': 0.5, 'cluster_min_samples': 20,
                'cluster_scale': True, 'cluster_walk': True, 'cluster_walk_steps': 5}
    if any(getattr(screen.settings(), key) != value for key, value in expected.items()):
        raise RuntimeError('Visible bounded search parameters did not reach the real settings')
    capture('21_bounded_walk_settings')
    outcomes = []

    def run(label):
        events, errors = [], []
        timer = QTimer(screen)
        def observe():
            dialog = app.activeModalWidget()
            try:
                if isinstance(dialog, QMessageBox):
                    events.append({'title': dialog.windowTitle(), 'text': dialog.text()})
                    capture(label + '_outcome')
                    button = dialog.button(QMessageBox.Ok)
                    if button is None:
                        raise RuntimeError('The actual search message has no OK button')
                    QTest.mouseClick(button, Qt.LeftButton)
                elif isinstance(dialog, QDialog) and dialog.windowTitle() == 'Cluster settings':
                    events.append({'title': dialog.windowTitle(), 'eps': dialog.eps(),
                                   'min_samples': dialog.min_samples(), 'walk': dialog.walk(),
                                   'walk_steps': dialog.walk_steps(), 'scale': dialog.scale()})
                    settings = screen.settings()
                    if (dialog.eps(), dialog.min_samples(), dialog.walk(), dialog.walk_steps(), dialog.scale()) != (
                            settings.cluster_eps, settings.cluster_min_samples, settings.cluster_walk,
                            settings.cluster_walk_steps, settings.cluster_scale):
                        raise RuntimeError('Search confirmation does not match the actual inline controls')
                    capture(label + '_confirmation')
                    QTest.mouseClick(dialog.findChild(QDialogButtonBox).button(QDialogButtonBox.Ok), Qt.LeftButton)
            except Exception as exc:
                errors.append(str(exc))
                if dialog is not None:
                    dialog.reject()
        timer.timeout.connect(observe)
        timer.start(350)
        try:
            QTest.mouseClick(screen.search._run, Qt.LeftButton)
            settle(0.7)
        finally:
            timer.stop()
        if errors:
            raise RuntimeError('; '.join(errors))
        capture(label + '_result')
        result = {'label': label, 'dialogs': events,
                  'gates': screen.gates.gates.to_dict(),
                  'status': screen.gates.status(),
                  'populations': {name: int(screen.gates.gates.mask(screen._frame, name).sum())
                                  for name in screen.gates.gates.names}}
        outcomes.append(result)
        write_json(captures / 'cluster_search.json', outcomes)
        return result

    walk = run('22_walk')
    if not walk['gates']['gates'] and not any('nothing to recommend' in event.get('title', '').lower() for event in walk['dialogs']):
        raise RuntimeError('An empty Walk must explain why it refused a recommendation')
    # Keep the actual Walk result, including a refusal. A fixed-radius pass is
    # a different user decision and is labelled as such, never as Walk success.
    if not screen.gates.gates.is_empty:
        file_dialog(screen._save_gates, work / 'walk_gates.json', '23_save_walk_strategy', save=True)
        while not screen.gates.gates.is_empty:
            item = tree.topLevelItem(0)
            QTest.mouseClick(tree.viewport(), Qt.LeftButton, pos=tree.visualItemRect(item).center())
            QTest.mouseClick(screen.gates.tree._remove, Qt.LeftButton)
            settle(0.3)
    QTest.mouseClick(screen.search._walk, Qt.LeftButton)
    if screen.settings().cluster_walk:
        raise RuntimeError('The real Walk toggle did not turn off')
    capture('24_fixed_radius_settings')
    fixed = run('25_fixed_radius')
    if not fixed['gates']['gates']:
        raise RuntimeError('This example did not produce an actual fixed-radius population')
    if any(gate.get('parent') for gate in fixed['gates']['gates']):
        raise RuntimeError('The cluster pass inherited an unintended parent gate')
    file_dialog(screen._save_gates, work / 'cluster_gates.json', '26_save_cluster_strategy', save=True)
    return outcomes
