"""The usage panel must never take the GUI down when nvidia-smi misbehaves.

GPUtil.getGPUs() shells out to nvidia-smi and int()s whatever comes back, so a
driver/library version mismatch surfaces as

    ValueError: invalid literal for int() with base 10:
    'Failed to initialize NVML: Driver/library version mismatch'

update_usage runs on a repeating Tk `after` callback, so that exception is not
a one-off -- it lands in the Tk callback handler once a second for the life of
the session. The panel is decoration; it must not be able to break the GUI.
"""
import ast
import textwrap
import inspect
import pytest


def _source():
    import spacr.gui_core as gc
    return inspect.getsource(gc.setup_usage_panel)


def test_the_gpu_poll_is_guarded():
    src = _source()
    assert "GPUtil.getGPUs()" in src
    # The call must be the first statement of a try block, not merely
    # somewhere downstream of one.
    assert "try:\n                gpus = GPUtil.getGPUs()" in src, (
        "GPUtil.getGPUs() is called unguarded -- any nvidia-smi failure "
        "becomes an uncaught exception in a once-a-second Tk callback")


def test_a_failing_gpu_poll_is_reported_once_not_every_second():
    """A permanently broken driver must not print once per second forever."""
    src = _source()
    assert "_gpu_poll" in src, "no latch: the failure would repeat every tick"
    assert "enabled'] = False" in src or 'enabled"] = False' in src, (
        "the latch is never cleared, so polling continues after the failure")


def test_the_reschedule_is_guarded():
    """A destroyed frame raises TclError from .after() -- same failure class."""
    src = _source()
    tail = src[src.rindex("parent_frame.after"):]
    head = src[:src.rindex("parent_frame.after")]
    assert "try:" in head[-300:], "the reschedule is unguarded"


def test_the_message_tells_the_user_what_to_do():
    """A mismatch is fixed by a reboot; saying so saves a support round trip."""
    src = _source()
    assert "reboot" in src.lower()
    assert "CPU" in src, "the user must be told the run continues on CPU"


def test_ram_and_cpu_bars_survive_a_dead_gpu():
    """The GPU is one of four things this panel shows; it is not all of them.

    Checked on the SYNTAX TREE, not by grepping the source text. The
    substring version asserted ``"return" not in between`` over raw
    characters, so it fired on any prose containing the word -- a docstring
    saying "GPUtil.getGPUs() returned at least one GPU", and then the
    ``:returns:`` field that documenting the function added. Both are
    documentation the function should have, and neither is control flow.

    The thing worth guarding is a `return` STATEMENT between the GPU poll and
    the CPU poll, which is what this now looks for.
    """
    
    src = _source()
    gpu_at = src.index("GPUtil.getGPUs()")
    cpu_at = src.index("psutil.cpu_percent")
    assert cpu_at > gpu_at, "CPU polling must come after, and be reachable"

    gpu_line = src.count("\n", 0, gpu_at) + 1
    cpu_line = src.count("\n", 0, cpu_at) + 1
    tree = ast.parse(textwrap.dedent(src))
    offenders = [
        node.lineno for node in ast.walk(tree)
        if isinstance(node, ast.Return) and gpu_line <= node.lineno < cpu_line
    ]
    assert not offenders, (
        f"an early return on GPU failure would kill the CPU bars too; "
        f"return statement(s) at source line(s) {offenders}")
