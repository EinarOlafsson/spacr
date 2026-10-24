"""Recruitment result scenes (lesson 25): the figure charts and the console counts.

Split out of ``capture_refresh`` so each scene clip starts BEFORE its actions
in the GNOME capture host: the pointer clicks the figure in the list, and the
console is brought to its counts by the pointer. Outside the host each scene
does what the inline code did (the actions, then ``capture(name)``).
"""
from __future__ import annotations

import capture_ui


def record_recruitment_results(window, screen, queue, blocks, capture, settle):
    """Overlay and four charts, then the console counts, of a finished Recruitment run.

    The archived masks are NOT asserted to be the postprocessed masks behind
    the stored measurement rows (see capture_refresh).
    """
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    if 'Failed to plot images with outlines' in '\n'.join(blocks):
        raise RuntimeError('The actual Recruitment overlay failed')
    if queue.count() < 5:
        raise RuntimeError('Recruitment did not produce its overlay and four charts')

    def give_results_room():
        width = sum(screen._body_splitter.sizes())
        capture_ui.splitter_sizes(capture, screen._body_splitter, [width // 4, width - width // 4])
        capture_ui.scroll_to_value(capture, screen._settings_scroll, 0, 'horizontal')
        if screen._usage_card.body.isVisible():
            if capture_ui.live(capture):
                capture_ui.press(capture, screen._usage_card.title_label)
            else:
                QTest.mouseClick(screen._usage_card.title_label, Qt.LeftButton)
        capture_ui.splitter_sizes(capture, screen._runtime_splitter, [1200, 450])

    prepared = False
    for index in range(queue.count()):
        # Host: each clip starts before the pointer picks the figure in the list.
        with capture_ui.clip(capture, f'25_recruitment_figure_{index:02d}'):
            if not prepared:
                give_results_room()
                prepared = True
            capture_ui.show_figure(capture, queue, index)
            settle()
            capture_ui.point_at(capture, queue, dwell=1.5)
    with capture_ui.clip(capture, '26_recruitment_console_counts'):
        capture_ui.splitter_sizes(capture, screen._runtime_splitter, [300, 1350])
        capture_ui.splitter_sizes(capture, screen._console._split, [1200, 100])
        if capture_ui.live(capture):
            capture_ui.console_end_all(capture, screen._console, settle)
        else:
            for block, _, _ in screen._console._pipeline_console_blocks():
                block.setFocus()
                QTest.keyClick(block, Qt.Key_End)
            screen._console.jump_to_the_end()
        settle()
        capture_ui.point_at(capture, screen._console, dwell=2.0)
    capture_ui.splitter_sizes(capture, screen._runtime_splitter, [1200, 450])
