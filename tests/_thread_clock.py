"""Thread CPU time of one call, fine enough for a frame on Windows too.

``time.thread_time`` reads ``GetThreadTimes`` on Windows, which advances in
steps of about 15.6 ms: a frame costing a few milliseconds reads as 0, and a
budget test then divides by zero or compares nothing. Off Windows one call
is timed exactly as before. On Windows the call is repeated for a fraction
of a second and only the timed parts are summed; their tick boundaries fall
at random against each start, so the mean converges on the true cost.
"""
from __future__ import annotations

import os
import time

COARSE_THREAD_CLOCK = os.name == "nt"


def thread_ms(call, *, between=None, minimum_s: float = 0.2,
              minimum_cpu_s: float = 0.05, maximum_s: float = 5.0) -> float:
    """Thread CPU milliseconds one ``call()`` costs.

    :param call: the work to time.
    :param between: untimed work done before each ``call``, such as
        advancing an animation clock.
    :param minimum_s: on Windows, how long to keep repeating, in seconds.
    :param minimum_cpu_s: on Windows, keep repeating until the timed calls
        have also collected this much thread time (a few clock steps), so a
        cheap call on a busy machine is not read as free.
    :param maximum_s: on Windows, the wall-clock limit on repeating.
    :returns: the cost of one call, in milliseconds.
    """
    if not COARSE_THREAD_CLOCK:
        if between is not None:
            between()
        started = time.thread_time()
        call()
        return (time.thread_time() - started) * 1000.0
    total, count, began = 0.0, 0, time.perf_counter()
    while count == 0 or (time.perf_counter() - began < maximum_s
                         and (time.perf_counter() - began < minimum_s
                              or total < minimum_cpu_s)):
        if between is not None:
            between()
        started = time.thread_time()
        call()
        total += time.thread_time() - started
        count += 1
    return total * 1000.0 / count
