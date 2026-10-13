"""Drive genuine Pipeline Graph controls against existing post-cleanup evidence."""
import json
from pathlib import Path
import time

from capture_report import snapshot_source, verify_private_sqlite_changes
from manager_data import verify_bind
from pipeline_graph_data import verify_graph, verify_originals


def record_graph(app, window, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt, QTimer, QPoint
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QFileDialog, QDialogButtonBox, QLineEdit, QSplitter

    inputs = json.loads((Path(stage)/'pipeline_graph_state'/f'{Path(captures).name}.json').read_text())
    root = verify_bind(inputs)
    deadline = time.monotonic()+timeout
    proof = {'accepted': False, 'lesson': '52_pipeline_graph', 'inputs': inputs,
             'app_source_modified': False, 'analysis_run': False, 'published': False}
    screen = None
    loads = []

    def wait_for(predicate):
        while not predicate():
            if time.monotonic() > deadline:
                raise TimeoutError('Bounded Pipeline Graph capture timed out')
            settle(.05)

    def click(widget):
        if not widget.isVisible() or not widget.isEnabled():
            raise ValueError('Actual graph control is not available')
        QTest.mouseClick(widget, Qt.LeftButton, pos=widget.visibleRegion().boundingRect().center())
        settle(.25)

    def done():
        wait_for(lambda: not screen.is_busy() and not screen.active_jobs())
        if screen.last_error:
            raise ValueError(screen.last_error)

    def browse(directory, frame):
        accepted, errors = [], []
        timer, watchdog = QTimer(window), QTimer(window)
        timer.setSingleShot(True)
        watchdog.setSingleShot(True)

        def handle():
            dialog = app.activeModalWidget()
            try:
                if not isinstance(dialog, QFileDialog):
                    raise ValueError('Expected actual Choose a project folder picker')
                dialog.accepted.connect(lambda: accepted.append(True))
                edit = dialog.findChild(QLineEdit, 'fileNameEdit')
                click(edit)
                QTest.keyClick(edit, Qt.Key_A, Qt.ControlModifier)
                QTest.keyClicks(edit, str(directory))
                capture(frame)
                box = dialog.findChild(QDialogButtonBox)
                buttons = [b for b in box.buttons() if box.buttonRole(b) == QDialogButtonBox.AcceptRole]
                if len(buttons) != 1:
                    raise ValueError('No unique genuine picker accept button')
                click(buttons[0])
            except Exception as error:
                errors.append(str(error))
                if dialog is not None:
                    dialog.reject()

        def abort():
            errors.append('The actual folder picker timed out')
            dialog = app.activeModalWidget()
            if dialog is not None:
                dialog.reject()

        timer.timeout.connect(handle)
        watchdog.timeout.connect(abort)
        timer.start(300)
        watchdog.start(15000)
        try:
            click(screen._browse_button)
        finally:
            timer.stop()
            watchdog.stop()
            timer.deleteLater()
            watchdog.deleteLater()
        if errors or not accepted:
            raise ValueError('; '.join(errors) or 'Folder was not accepted')
        done()

    def check_graph():
        graph = screen.graph()
        result = verify_graph(graph, root, inputs['artifact_rows'], recorded_edges=inputs['recorded_edges'])
        if sorted(result['states'].values()) != ['current', 'current', 'missing']:
            raise ValueError('Expected two existing artifacts and the genuinely pruned crops')
        proof['independent_graph_checks'] = result
        proof['graph'] = graph.to_dict()
        return graph

    def node(kind, frame):
        graph = check_graph()
        picked = next(n for n in graph.nodes if n.kind == kind)
        rect = screen._canvas.node_rects()[picked.artifact_id]
        QTest.mouseClick(screen._canvas, Qt.LeftButton, pos=rect.center())
        settle(.2)
        detail = screen._details.toPlainText()
        expected = [f'{picked.module} → {kind}', f'state: {picked.state}', f'path: {picked.path}',
                    f'run id: {picked.run_id}', f'produced: {picked.created_utc}',
                    f'settings digest: {picked.settings_hash[:16]}',
                    f'size: {picked.size_bytes} bytes in {picked.n_files} file(s)',
                    'Nothing was derived from this.']
        if screen._canvas.selected != picked.artifact_id or any(value not in detail for value in expected):
            raise ValueError('Actual clicked node detail does not match its raw record')
        proof.setdefault('details', {})[kind] = detail
        capture(frame)

    def filter_state(state, expected, frame):
        click(screen._filters[state])
        wanted = {n.artifact_id for n in screen.graph().nodes if n.state in expected}
        if screen._canvas._visible != wanted or set(screen._canvas.node_rects()) != wanted:
            raise ValueError('Actual state filter failed to preserve the exact node identities')
        proof.setdefault('filters', []).append({'clicked': state, 'visible': sorted(wanted),
                                              'detail_text': screen._details.toPlainText(),
                                              'verdict': screen._verdict.text()})
        capture(frame)

    try:
        help_action = next(a for a in window.menuBar().actions() if a.text().replace('&', '') == 'Help')
        menu = help_action.menu()
        match = [a for a in menu.actions() if a.text().replace('&', '') == 'Pipeline graph']
        if len(match) != 1:
            raise ValueError('Expected unique Help -> Pipeline graph route')
        QTest.mouseClick(window.menuBar(), Qt.LeftButton, pos=window.menuBar().actionGeometry(help_action).center())
        settle(.2)
        capture('01_help_pipeline_graph')
        QTest.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(match[0]).center())
        wait_for(lambda: window._screens.get('pipeline_graph') is not None)
        screen = window._screens['pipeline_graph']
        screen.graph_loaded.connect(lambda g: loads.append(len(g.nodes)))
        done()
        capture('02_empty_graph')
        browse(inputs['conversion_copy'], '03_choose_real_conversion')
        graph = screen.graph()
        proof['unregistered'] = verify_graph(graph, inputs['conversion_copy'], [])
        proof['unregistered_verdict'] = screen._verdict.text()
        capture('04_unregistered_is_not_unprocessed')
        browse(root, '05_choose_cleanup_snapshot')
        check_graph()
        proof['verdict'] = screen._verdict.text()
        proof['declared_order'] = screen._declared.text()
        capture('06_three_actual_artifacts')
        splitter = screen._scroll.parentWidget()
        if not isinstance(splitter, QSplitter):
            raise ValueError('Expected real canvas/details splitter')
        handle = splitter.handle(1)
        start = handle.rect().center()
        QTest.mousePress(handle, Qt.LeftButton, pos=start)
        QTest.mouseMove(handle, start-QPoint(850, 0), delay=120)
        QTest.mouseRelease(handle, Qt.LeftButton, pos=start-QPoint(850, 0))
        settle(.3)
        node('measurements-db', '07_measurement_provenance')
        node('crops', '08_missing_crop_provenance')
        node('resource-log', '09_external_resource_log')
        filter_state('current', {'missing', 'stale'}, '10_missing_only')
        node('crops', '11_missing_detail')
        filter_state('missing', {'stale'}, '12_no_visible_artifacts')
        # Clipboard is a real GUI action, independently checked against SQL IDs.
        click(screen._copy_button)
        dot = app.clipboard().text()
        ids = {r['artifact_id'] for r in inputs['artifact_rows']}
        if not dot.startswith('digraph ') or any(f'"{i}" [' not in dot for i in ids) or ' -> ' in dot:
            raise ValueError('Copied DOT omits actual nodes or invents a recorded edge')
        proof['clipboard_dot'] = dot
        proof['copy_includes_hidden_nodes'] = len(ids)
        capture('13_copy_full_graph_not_filtered_view')
        filter_state('current', {'current', 'stale'}, '14_current_only')
        filter_state('missing', {'current', 'stale', 'missing'}, '15_restore_all_nodes')
        before = screen.graph().to_dict()
        click(screen._reload_button)
        done()
        after = check_graph().to_dict()
        if {k:v for k,v in before.items() if k != 'generated_utc'} != {k:v for k,v in after.items() if k != 'generated_utc'}:
            raise ValueError('Redraw changed actual provenance or node states')
        node('measurements-db', '16_redrawn_unchanged_graph')
        proof['source_preservation'] = verify_originals(inputs)
        proof['private_sqlite_sidecar_changes'] = verify_private_sqlite_changes(root, inputs['clone_files'], snapshot_source(root))
        proof['load_counts'] = loads
        proof['remaining_workers'] = screen.active_jobs()
        proof['accepted'] = True
    finally:
        write_json(Path(captures)/'pipeline_graph_acceptance.json', proof)
        if screen is not None:
            screen.close()
