"""Suggest progress, cancellation and split notices use reviewed UI copy."""
import hashlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize('language', ['sv', 'de', 'es', 'pt', 'fr', 'zh_CN', 'hi', 'ko', 'is'])
def test_actual_suggest_controls_and_console_use_reviewed_language(qtbot, monkeypatch, language):
    from spacr.qt import i18n
    from spacr.qt.screens.annotate import AnnotateScreen

    root = Path(__file__).resolve().parents[2]
    evidence = json.loads((root / 'docs/i18n/reviewed/runtime' / language /
                           '2026-10-01-suggest-progress-and-cancellation.json').read_text())
    records = evidence['records']
    assert len(records) == 14
    targets = {r['source']: r['translation'] for r in records}
    for record in records:
        assert record['table'] == 'ui'
        assert record['key'] == record['source']
        assert record['source_sha256'] == hashlib.sha256(record['source'].encode()).hexdigest()
    monkeypatch.setattr(i18n, 'current_language', lambda: language)
    screen = AnnotateScreen()
    qtbot.addWidget(screen)
    tooltip = next(s for s in targets if s.startswith('Stop the suggestion run'))
    assert screen._btn_suggest_cancel.toolTip() == targets[tooltip]
    assert screen._btn_suggest_cancel.isHidden()
    assert 'No new suggestions are written' in tooltip
    assert 'round scores may have been updated' in tooltip
    progress = 'Suggest: step {n} of {total} — {what}…'
    stages = ['clearing the outstanding suggestions', 'reading the measurements',
              'fitting on the labels so far', 'ranking the proposals', 'writing the suggestions']
    for n, (stage, source) in enumerate(zip(['clear', 'features', 'fit', 'rank', 'write'], stages), 1):
        screen._on_suggest_progress(n, 5, stage)
        assert screen._status_label.text() == targets[progress].format(
            n=n, total=5, what=targets[source])

    class ActiveRun:
        interrupted = False

        def requestInterruption(self):  # noqa: N802
            self.interrupted = True

    worker = ActiveRun()
    screen._suggest_worker = worker
    screen._btn_suggest_cancel.show()
    screen._btn_suggest_cancel.click()
    screen._suggest_worker = None
    assert worker.interrupted
    assert not screen._btn_suggest_cancel.isEnabled()
    assert screen._status_label.text() == targets[
        'Cancelling the suggestion run after its current step…']
    screen._on_suggest_cancelled()
    assert screen._status_label.text() == targets['Suggest cancelled.']
    screen._on_suggest_finished()
    assert screen._btn_suggest_cancel.isHidden()
    assert screen._btn_suggest.isEnabled()
    screen._on_suggest_split_relaxed('E42')
    console = screen._console.as_text()
    for source in [
        'Suggest cancelled before writing new suggestions. Earlier suggestions may have been cleared and round scores may have been updated.',
        'This classifier used a random split because too few laboratory wells have labels.',
        'Its accuracy may be overestimated.',
        'The suggestions are unaffected.',
        'Label crops from more wells for validation with independent wells. ({why})',
    ]:
        assert targets[source].format(why='E42') in console
