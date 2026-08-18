"""Deprecated alias for :mod:`spacr.parameter_sweep`.

The module was renamed when the sweep became a spaCR module in its own right
("Parameter Sweep") rather than a script beside the regression. Importing this
name still works so nothing written against it breaks; it re-exports the new
API and nothing else.

THE RENAME NEVER SHIPPED, WHICH MEANS THIS SHIM PROTECTS NOBODY. Instruction
127, finding 5, asks the question that decides whether it can go: "check
whether any RELEASED version advertised the name". Checked on 2026-08-18,
against every tag in the repository::

    for tag in v1.3.5 v1.3.6 v1.4.9.8 v1.4.9.9 v1.5.0.1 v1.5.0.4; do
        git cat-file -e $tag:spacr/regression_search.py
    done

Not one of the six carries this file -- and none carries
``spacr/parameter_sweep.py`` either, so BOTH names arrived after the last
release and no installed spaCR has ever exposed either one. There is no
downstream import to keep working, and nothing in ``spacr/`` or ``tests/``
imports it.

IT IS NOT DELETED HERE, and that is a rule rather than a hesitation: which
modules exist is the maintainer's decision (the same rule instruction 06 set
for the GUI apps). The evidence is written down so the decision costs a
sentence rather than the check again. What it costs meanwhile is one
DeprecationWarning per import and nothing else; the re-exports are pinned by
tests/test_the_shim_that_never_shipped.py so it cannot rot into a shim that
lies while it waits.
"""
from __future__ import annotations

import warnings

from .parameter_sweep import *  # noqa: F401,F403
from .parameter_sweep import (  # noqa: F401
    DEFAULT_SWEEP_SPACE as DEFAULT_SEARCH_SPACE,
    SweepSpace as SearchSpace,
    run_sweep as run_search,
    run_sweep_parallel as run_search_parallel,
    summarise_sweep as summarise_search,
)

warnings.warn(
    "spacr.regression_search is now spacr.parameter_sweep; the old names "
    "(SearchSpace, run_search, summarise_search) still work.",
    DeprecationWarning, stacklevel=2)
