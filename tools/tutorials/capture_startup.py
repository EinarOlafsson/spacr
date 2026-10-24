"""The launch clip 01_startup_sequence: the app starting, its logo, the blow-away, Home.

The clip is begun by ``capture_host_mode.run_child`` BEFORE the application
window exists (the proposed change is in the hand-off report: the recording is
started on the human process before ``spacr.qt.run`` and adopted by
``HostRuntime.adopt_scene``). This module only ends it, once the sequence is over:
the loading cover has gone, no blow-away is running, and Home has stayed on screen
for a moment.
"""
from __future__ import annotations

import time

CLIP = '01_startup_sequence'


def end_startup_clip(window, host, capture, settle, captures, *, hold=2.5, timeout=60.0) -> dict:
    """Wait for the sequence to finish, hold Home for ``hold`` seconds, end the adopted clip.

    :returns: the ``frames.json`` entry of the clip.
    """
    if host is None or host.current is None or host.current.get('name') != CLIP:
        raise RuntimeError(
            '--record-startup needs the clip 01_startup_sequence begun before the app launched: '
            'run run_native_capture --host gnome --splash, with the capture_host_mode change that '
            'starts the recording ahead of spacr.qt.run (see the hand-off report)')
    from spacr.qt.widgets.loading_screen import blow_away_running

    deadline = time.monotonic() + timeout
    while getattr(window, '_loading_screen', None) is not None or blow_away_running():
        if time.monotonic() > deadline:
            raise TimeoutError('The startup logo did not finish')
        settle(0.1)
    if not window._startup.isVisible():
        raise RuntimeError('Home is not showing after the startup sequence')
    settle(hold)
    entry = host.end_scene(at_least=0.0)          # no planned minimum: it ends with Home settled
    print(f'captured home/{CLIP} (GNOME capture host)', flush=True)
    return {CLIP: entry}
