"""
Custom per-object feature functions.

Users can drop Python files under ``~/.spacr/features/*.py`` that
export functions:

    def <name>(mask: np.ndarray, image: np.ndarray, **kwargs) -> float | dict

spaCR's measure step auto-discovers them at ``measure_crop`` boot
and includes each one alongside the built-in features (intensity /
morphology / colocalisation / …). The result is written to the
same measurements DB, one column per key returned.

Example ``~/.spacr/features/asymmetry.py``::

    import numpy as np
    def asymmetry(mask, image, **_):
        ys, xs = np.where(mask > 0)
        if len(xs) < 5:
            return 0.0
        # Something the built-ins don't compute
        return float(np.std(xs) / (np.std(ys) + 1e-9))

Every function is called once per (object, image_channel) tuple.
Errors in one custom feature don't stop the others — they're logged
and the offending column simply gets NaN for that object.

Public API::

    from spacr.custom_features import (
        features_dir, discover_features, call_feature,
    )
"""
from __future__ import annotations

import importlib.util
import inspect
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

LOG = logging.getLogger("spacr.custom_features")


def features_dir() -> Path:
    """Return ``~/.spacr/features/`` — created if it doesn't exist."""
    p = Path.home() / ".spacr" / "features"
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class CustomFeature:
    """One discovered feature function.

    :ivar name: label used as the DB column prefix; when the function
        returns a dict, keys are appended to make ``<name>_<key>``.
    :ivar source: the ``.py`` file the function lives in.
    :ivar fn: the callable itself.
    """
    name:   str
    source: Path
    fn:     Callable[..., Any]


def discover_features() -> List[CustomFeature]:
    """Walk :func:`features_dir` and return every public callable found.

    Silently skips files that fail to import — the offending path is
    logged at INFO and users see a warning in the Console.
    """
    hits: List[CustomFeature] = []
    for py in sorted(features_dir().glob("*.py")):
        if py.name.startswith("_"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"spacr._userfeat_{py.stem}", py,
            )
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception as e:
            LOG.info("skipping %s (import failed: %s)", py, e)
            continue
        for name in dir(mod):
            if name.startswith("_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            # Only include functions defined IN the file
            try:
                if fn.__module__ != mod.__name__:
                    continue
            except Exception:
                continue
            # Require at least `mask` + `image` parameters
            try:
                sig = inspect.signature(fn)
                params = list(sig.parameters.values())
                if len(params) < 2:
                    continue
            except Exception:
                continue
            hits.append(CustomFeature(name=name, source=py, fn=fn))
    LOG.info("discovered %d custom feature(s) under %s",
              len(hits), features_dir())
    return hits


def call_feature(cf: CustomFeature, mask, image,
                    **kwargs) -> Dict[str, Any]:
    """Invoke a custom feature safely, coercing the result to a
    ``{column_name: value}`` dict.

    :param cf: discovered feature.
    :param mask: 2-D uint16 mask (background 0).
    :param image: 2-D uint16 image aligned with ``mask``.
    :returns: mapping of DB column name → scalar. Empty dict on
        exception.
    """
    try:
        result = cf.fn(mask, image, **kwargs)
    except Exception as e:
        LOG.info("custom feature %s raised: %s", cf.name, e)
        return {}
    if isinstance(result, dict):
        return {f"{cf.name}_{k}": v for k, v in result.items()}
    return {cf.name: result}
