"""The trial worker has to start on a kernel that refuses its OOM hint.

``spacr.sweep_child`` volunteers itself for the OOM killer at import, before
anything large is loaded, so a runaway trial is reaped instead of the user's
editor.  Writing ``/proc/<pid>/oom_score_adj`` is not permitted everywhere --
a hardened container, a non-Linux machine, a sandboxed CI runner -- and the
worker that cannot leave the hint must still run the trial it was exec'd for.
"""
from __future__ import annotations

import builtins
import errno
import importlib
import os

import spacr.sweep_child as sweep_child


def test_a_kernel_that_refuses_the_oom_hint_still_leaves_a_usable_worker(
        monkeypatch, capsys):
    """Import survives a refused ``oom_score_adj`` write and ``main`` still runs.

    The write is attempted at module scope, so a raised ``OSError`` there would
    make the whole worker unimportable and every trial in the sweep would come
    back as a failure with an import error rather than a fit.
    """
    real_open = builtins.open
    refused = []

    def refuse_the_proc_file(file, *args, **kwargs):
        if "oom_score_adj" in str(file):
            refused.append(str(file))
            raise OSError(errno.EACCES, "Permission denied")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", refuse_the_proc_file)

    reloaded = importlib.reload(sweep_child)

    assert refused == [f"/proc/{os.getpid()}/oom_score_adj"], (
        "the hint is written for this process, once, at import")
    assert reloaded.main([]) == 2, (
        "the worker still reports a usage error rather than failing to import")
    assert "usage: python -m spacr.sweep_child" in capsys.readouterr().err
