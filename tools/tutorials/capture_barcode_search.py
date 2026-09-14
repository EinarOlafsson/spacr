"""Drive the genuine live barcode search, then explicitly apply its proposal."""
from dataclasses import asdict
import time

from map_barcodes_data import prepare_references, digest


def record_search(app, screen, stage, captures, capture, settle, write_json, timeout):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QSplitter
    from spacr.qt.widgets.card import Card

    references = prepare_references('/home/olafsson/Documents/barcodes',
                                    stage / 'map_references' / captures.name)
    write_json(captures / 'references.json', references)
    screen._dna_rain.settings_bar.set_opacity(.05)
    model = screen._settings_model
    for role, versions in references.items():
        if not model.set_value_for_key(role + '_csv', versions['plain']['path']):
            raise RuntimeError('Cannot set the visible reference field')
    # This is an explicit demonstration starting value, not a claimed default.
    if not model.set_value_for_key('window_length', 120):
        raise RuntimeError('Cannot configure the visible example window')
    bar = screen._settings_search
    bar.set_level('all')
    bar.set_query('csv')
    settle(.4)
    screen._settings_scroll.verticalScrollBar().setValue(0)
    capture('04_reference_tables')
    bar.set_query('')
    if screen._console.isVisible():
        QTest.mouseClick(screen._console_header, Qt.LeftButton)
    for card in screen.findChildren(Card):
        if (card.folder is not None and card.title_label is not None
                and card.body.isVisible() and card.title_label.text().lstrip('▼▾ ') in {'Console', 'System'}):
            QTest.mouseClick(card.title_label, Qt.LeftButton)
    settle(.3)
    panel = screen._barcode_search
    toggle = screen._barcode_search_toggle
    if not toggle.isVisible() or not toggle.isEnabled():
        raise RuntimeError('The real Find barcodes toggle is unavailable')
    QTest.mouseClick(toggle, Qt.LeftButton)
    settle(.5)
    if not panel.isVisible():
        raise RuntimeError('Find barcodes did not expose its real panel')
    for split in panel.findChildren(QSplitter):
        split.setSizes([650, 430, 300])
    before = model.collect()
    reports = []
    partial_recorded = []
    finished = []
    panel.search_finished.connect(finished.append)

    def updated(report):
        reports.append({'reads': report.reads, 'complete': report.complete})
        if not report.complete and not partial_recorded:
            partial_recorded.append(True)
            capture('06_live_search_progress')

    panel.search_updated.connect(updated)
    capture('05_find_barcodes')
    QTest.mouseClick(panel.search_button, Qt.LeftButton)
    deadline = time.monotonic() + timeout
    while panel.is_searching():
        if time.monotonic() > deadline:
            QTest.mouseClick(panel.cancel_button, Qt.LeftButton)
            raise TimeoutError('The real barcode search did not complete')
        settle(.1)
    settle(1)
    report, proposal = panel.report(), panel.proposal()
    write_json(captures / 'search_outcome.json', {
        'report': asdict(report) if report else None,
        'proposal': asdict(proposal) if proposal else None,
        'status': panel.status.text(), 'finished_signal': bool(finished and finished[-1] is report)})
    # At exact chunk-sized EOF, the current iterator leaves complete=False.
    # Require the genuine completion signal AND every downloaded record, not
    # a fabricated completion flag or a smaller-than-requested result.
    if (report is None or not finished or finished[-1] is not report
            or dict(report.reads_by_file) != {'R1': 10000, 'R2': 10000}
            or proposal is None or proposal.unresolved_roles):
        raise RuntimeError('Search did not establish every requested barcode')
    if not partial_recorded or len(reports) < 2:
        raise RuntimeError('No actual live refinement was observed')
    if model.collect() != before:
        raise RuntimeError('The search silently rewrote settings before Apply')
    capture('07_search_verdicts')
    splits = panel.findChildren(QSplitter)
    for split in splits:
        split.setSizes([300, 900, 200])
    settle(.4)
    capture('08_coloured_reads')
    for split in splits:
        split.setSizes([350, 220, 830])
    settle(.4)
    capture('09_proposed_settings')
    if not panel.apply_button.isEnabled():
        raise RuntimeError('There are no settings to apply in this demonstration')
    proposed_changes = panel.proposed_changes()
    QTest.mouseClick(panel.apply_button, Qt.LeftButton)
    settle(.4)
    after = model.collect()
    for key, _, value in proposed_changes:
        if after.get(key) != value:
            raise RuntimeError('Apply did not write the displayed proposed value')
    capture('10_settings_applied')
    write_json(captures / 'barcode_search.json', {
        'report': asdict(report), 'proposal': asdict(proposal),
        'live_updates': reports, 'settings_before_apply': before,
        'settings_after_apply': after, 'proposed_changes': proposed_changes,
        'no_automatic_settings_change': True,
        'demonstration_initial_window_length': 120,
        'applied_through_visible_button': True,
        'application_functions_replaced': False})
    # The proposal explicitly requires reversed reference copies; Apply does
    # not create those files. Show this separate step without claiming it does.
    QTest.mouseClick(toggle, Qt.LeftButton)
    for role in proposal.reverse_complement_needed:
        if role not in references:
            raise RuntimeError('Unexpected reference role needs reversing')
        value = references[role]['reverse_complement']['path']
        if not model.set_value_for_key(role + '_csv', value):
            raise RuntimeError('Cannot select the pre-existing reversed reference')
    # Arbitrary sets are supported by settings/API; the current form does not
    # render that key. Its separate recorded API example must say so.
    bar.set_query('csv')
    settle(.4)
    capture('11_oriented_reference_copies')
    bar.set_query('')
    if not screen._console.isVisible():
        QTest.mouseClick(screen._console_header, Qt.LeftButton)
    for card in screen.findChildren(Card):
        if (card.folder is not None and card.title_label is not None
                and not card.body.isVisible() and card.title_label.text().lstrip('▶▸ ') == 'Console'):
            QTest.mouseClick(card.title_label, Qt.LeftButton)
    for role, versions in references.items():
        for row in versions.values():
            if digest(row['source']) != row['sha256'] or digest(row['path']) != row['sha256']:
                raise RuntimeError('A reference changed during the recording')
    write_json(captures / 'mapping_reference_selection.json', {
        role: model.collect()[role + '_csv'] for role in references})
