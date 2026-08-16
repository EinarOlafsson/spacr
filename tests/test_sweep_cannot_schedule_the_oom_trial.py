"""The sweep must not schedule a trial that eats the machine.

Instruction 114. The maintainer's report was "it crashed vscode many tinmes",
and there were two independent causes. Both are pinned here because both are
invisible: nothing crashes in a test, so a regression in either would look
exactly like a passing suite.

CAUSE 1 -- the filters could not see the fixed values.

    build_trials applied `space.fixed` AFTER `accept()`, so every filter
    judged a half-built trial. The GUI pins each UNTICKED axis into `fixed`,
    which means the settings a user did not vary were precisely the ones the
    filters were blind to. `permutation_at_cell_level_exhausts_memory` read
    `analysis_unit` as None and passed the combination it exists to stop.

    Unticking one checkbox scheduled the ~57 GiB run.

CAUSE 2 -- nothing told the OOM killer who to pick.

    `be_polite()` dropped CPU and I/O priority, which decides who WAITS. It
    said nothing about who gets KILLED. The kernel scores by resident size,
    so on a box running a sweep it picks the Electron editor holding a
    gigabyte over a worker holding six. The user loses their editor; the
    sweep survives. Exactly backwards.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from spacr.parameter_sweep import SweepSpace, build_trials


# --------------------------------------------------------------------------- #
#  Cause 1: fixed values are part of the trial before it is judged
# --------------------------------------------------------------------------- #

def test_a_pinned_analysis_unit_is_visible_to_the_filters():
    """The exact reproduction: one axis varied, one value pinned.

    This is what the GUI sends when a user unticks 'analysis unit' and leaves
    it on 'cell'.
    """
    space = SweepSpace(
        axes={"inference": ["parametric", "nonparametric"]},
        fixed={"analysis_unit": "cell"},
    )

    trials = build_trials(space)

    offenders = [t for t in trials
                 if t.get("inference") == "nonparametric"
                 and t.get("analysis_unit") == "cell"]
    assert not offenders, (
        "the sweep scheduled a cell-level permutation test -- the ~57 GiB "
        "run that `permutation_at_cell_level_exhausts_memory` exists to "
        "prevent. The filters are judging trials before the fixed values "
        "are merged in.")


def test_the_legitimate_combination_still_runs():
    """The other side, so the fix is not just 'reject everything'.

    A well-level permutation is the normal, cheap case and must survive.
    """
    space = SweepSpace(
        axes={"inference": ["parametric", "nonparametric"]},
        fixed={"analysis_unit": "well"},
    )

    trials = build_trials(space)

    assert len(trials) == 2
    assert {t["inference"] for t in trials} == {"parametric", "nonparametric"}


def test_every_emitted_trial_carries_the_fixed_values():
    """A trial the filters approved must be the trial that runs. If `fixed`
    were merged afterwards the two could differ, which is the whole bug."""
    space = SweepSpace(
        axes={"inference": ["parametric"]},
        fixed={"analysis_unit": "well", "regression_type": "ols"},
    )

    for trial in build_trials(space):
        assert trial["analysis_unit"] == "well"
        assert trial["regression_type"] == "ols"


def test_a_pinned_value_can_reject_every_trial():
    """The strongest form: if the pin makes every combination illegal, the
    sweep emits nothing rather than running them anyway."""
    space = SweepSpace(
        axes={"inference": ["nonparametric"]},
        fixed={"analysis_unit": "cell"},
    )

    assert build_trials(space) == []


# --------------------------------------------------------------------------- #
#  Cause 2: the worker volunteers for the OOM killer
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not os.path.exists("/proc/self/oom_score_adj"),
                    reason="oom_score_adj is Linux-only")
def test_be_polite_volunteers_this_process_for_the_oom_killer():
    """Run in a CHILD, because the call also renices to 19 and would drag
    the test session down with it."""
    code = (
        "import os;"
        "from spacr.parameter_sweep import be_polite;"
        "be_polite();"
        "print(open(f'/proc/{os.getpid()}/oom_score_adj').read().strip())"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)

    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) >= 500, (
        "a sweep worker must be a more attractive OOM target than the user's "
        "editor, or the kernel kills the editor instead")


@pytest.mark.skipif(not os.path.exists("/proc/self/oom_score_adj"),
                    reason="oom_score_adj is Linux-only")
def test_the_contained_child_sets_it_at_import():
    """A contained trial is exec'd into a fresh interpreter and never calls
    be_polite, so the child module has to do it itself."""
    code = (
        "import os, spacr.sweep_child;"
        "print(open(f'/proc/{os.getpid()}/oom_score_adj').read().strip())"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True, timeout=120)

    assert out.returncode == 0, out.stderr
    assert int(out.stdout.strip()) >= 500


def test_be_polite_survives_a_kernel_that_refuses():
    """Best effort, not a requirement: a container or hardened kernel may
    refuse the write, and that is not a reason to fail a sweep."""
    import builtins

    from spacr import parameter_sweep

    real_open = builtins.open

    def _refuse(path, *args, **kwargs):
        if "oom_score_adj" in str(path):
            raise PermissionError("refused")
        return real_open(path, *args, **kwargs)

    builtins.open = _refuse
    try:
        parameter_sweep.be_polite()      # must not raise
    finally:
        builtins.open = real_open
