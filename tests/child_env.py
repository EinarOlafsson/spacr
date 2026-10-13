"""The environment a test's child interpreter should run in.

Several tests answer their question in a subprocess, because what they check
either takes the interpreter down with it or has to be observed from a clean
import graph. Each of them built the child's environment as a dict from
scratch -- ``PATH``, ``HOME``, ``MPLBACKEND`` and little else -- so that the
user's own settings could not decide the answer.

That is the right instinct and the wrong mechanism. A from-scratch dict also
drops the variables the *interpreter* needs to be itself: on a conda install
``CONDA_PREFIX`` is what lets Qt find its platform plugins, and without it the
child dies with ``could not find the Qt platform plugin "offscreen" in ""``
before reaching a single line of spaCR. The test then fails for a reason that
has nothing to do with what it was asserting, on some machines and not others.

So inherit the environment and replace only the parts that would otherwise
leak an answer in: the settings directories, the display, and the module
search path.
"""
from __future__ import annotations

import os
from typing import Optional


def child_env(*, home: Optional[str] = None,
              pythonpath: Optional[str] = None,
              qt: bool = False,
              **extra: str) -> dict:
    """The parent environment with the leaky parts replaced.

    :param home: value for ``HOME`` and ``XDG_CONFIG_HOME``; both are set so a
        child cannot read the real user's saved settings. Defaults to a
        scratch path under ``/tmp``.
    :param pythonpath: value for ``PYTHONPATH``. Pass the repository root to
        run against the checkout; pass ``""`` to force the installed package.
        ``None`` leaves the parent's value alone.
    :param qt: when true, ask for the offscreen platform so the child needs no
        display.
    :param extra: further variables to set, which win over everything above.
    :returns: a fresh dict; the parent's own environment is untouched.
    """
    env = dict(os.environ)
    home = home or "/tmp/spacr-child-home"
    env["HOME"] = home
    env["XDG_CONFIG_HOME"] = home
    env.setdefault("MPLBACKEND", "Agg")
    if qt:
        env["QT_QPA_PLATFORM"] = "offscreen"
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    env.update({str(k): str(v) for k, v in extra.items()})
    return env
