"""Deprecated alias for :mod:`spacr.parameter_sweep`.

The module was renamed when the sweep became a spaCR module in its own right
("Parameter Sweep") rather than a script beside the regression. Importing this
name still works so nothing written against it breaks; it re-exports the new
API and nothing else.
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
