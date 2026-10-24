"""Reject tutorial demonstrations with partial output, even after GUI success."""
from __future__ import annotations

import time


def assess_pipeline(outcome, console_blocks, figure_count, *, requires_figure=True):
    """Return explicit recording acceptance, keeping ordinary QC warnings visible."""
    reasons = []
    if not outcome.get('finished') or not outcome.get('ok'):
        reasons.append('The pipeline did not report successful completion.')
    if outcome.get('errors'):
        reasons.append('The worker reported errors.')
    output = '\n'.join(console_blocks)
    if 'RUN INCOMPLETE' in output or 'ARTIFACTS FROM THIS RUN ARE INCOMPLETE' in output:
        reasons.append('The console reports partial artifacts despite the GUI completion status.')
    if 'Pipeline worker failed' in output:
        reasons.append('A background worker failed, even though the main pipeline finished.')
    if requires_figure and figure_count < 1:
        reasons.append('Plot was enabled but no inspectable figure was produced.')
    return {'accepted': not reasons, 'reasons': reasons, 'figure_count': figure_count,
            'worker_finished': bool(outcome.get('finished')),
            'worker_ok': bool(outcome.get('ok'))}


def observe_pipeline_run(screen, click_run, settle, timeout, *, after_click=None):
    """Observe the real run even when import is deferred or its worker is gone.

    AppScreen creates its worker only after the asynchronous entry import,
    and clears it after thread completion. Its new run timestamp and rendered
    terminal notice survive both transitions; neither an old notice nor merely
    an idle thread proves that this click ran anything.
    """
    from spacr.qt.i18n import tr

    previous_run = getattr(screen, '_run_started_at', None)
    before = {id(block): text for block, _, text
              in screen._console._pipeline_console_blocks()}
    terminal = {tr('✓ Finished'): True, tr('✗ Failed — see traceback above'): False}
    deadline = time.monotonic() + timeout
    click_run()
    if after_click is not None:
        after_click()
    while True:
        new_run = getattr(screen, '_run_started_at', None) != previous_run
        additions = []
        errors = []
        for block, kind, text in screen._console._pipeline_console_blocks():
            old = before.get(id(block), '')
            # A replaced block is not evidence of an appended terminal line.
            if old and not text.startswith(old):
                continue
            added = text[len(old):]
            additions.extend(added.splitlines())
            if kind == 'traceback' and added.strip():
                errors.append(added)
        endings = [terminal[line.strip()] for line in additions
                   if line.strip() in terminal]
        if (new_run and endings and getattr(screen, '_entry_import', None) is None
                and not screen._worker_thread_is_running()
                and screen._btn_run.isEnabled()):
            return dict(finished=True, ok=endings[-1], errors=errors,
                        completion_observation='new run timestamp, rendered terminal notice, idle thread')
        if time.monotonic() > deadline:
            raise TimeoutError('The bounded pipeline did not report a new completed run')
        settle(.1)


def run_recorded_pipeline(screen, capture, settle, timeout, *, batch_clip=False, running_clip=False):
    """Observe this physical pipeline Run without requiring its worker to survive."""
    import time
    import capture_ui
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    def after_click():
        if batch_clip:
            capture_ui.wait(capture, 1.5, settle)
            capture.end()
        settle(1)
        if running_clip:
            capture_ui.wait(capture, 1.5, settle)
            capture.end()
        else:
            capture('21_batch_running')

    next_progress = time.monotonic() + 20
    def waiting(seconds):
        nonlocal next_progress
        if time.monotonic() >= next_progress:
            capture('22_batch_progress')
            next_progress = time.monotonic() + 30
        settle(seconds)

    try:
        return observe_pipeline_run(screen,
            lambda: QTest.mouseClick(screen._btn_run, Qt.LeftButton),
            waiting, timeout, after_click=after_click)
    except TimeoutError:
        if screen._worker_thread_is_running():
            QTest.mouseClick(screen._btn_stop, Qt.LeftButton)
            settle(3)
        raise
