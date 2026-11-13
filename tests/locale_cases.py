"""Run what used to be a parametrisation as the subtests of one test.

The test-suite reduction of 2026-11-12 (batch 3) folded the per-locale
parametrisations into one test per behaviour.  Each such test loops over
every locale (and every other former parameter) inside pytest's built-in
``subtests`` fixture, so a run still reports EVERY failing case, each named
by its locale and its string, but the suite collects one item instead of
nine, twenty-seven or a hundred and ninety-two.

:func:`cases` takes the same ``(argnames, argvalues[, ids])`` triples that
``pytest.mark.parametrize`` took, crosses them, and yields one
``(label, kwargs)`` pair per case.  ``label`` is what the subtest report
prints; ``kwargs`` are the values the case runs with.  :func:`case_dir`
gives every case a fresh directory of its own under the test's
``tmp_path``, as separate items had.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any, Iterator

import pytest

_SCALAR = (str, int, float, bool, type(None))


def _names(argnames) -> list[str]:
    if isinstance(argnames, str):
        return [name.strip() for name in argnames.split(",") if name.strip()]
    return list(argnames)


def _shown(value: Any) -> Any:
    """A short, readable stand-in for a value in a subtest's report line."""
    if isinstance(value, _SCALAR):
        return value
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if name:
        return name
    text = repr(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _axis(argnames, argvalues, ids=None) -> list[tuple[dict, dict]]:
    names = _names(argnames)
    rows = []
    for index, raw in enumerate(argvalues):
        ident = None
        if hasattr(raw, "values") and hasattr(raw, "marks") and hasattr(raw, "id"):
            if raw.marks:
                raise TypeError(f"case {raw!r} carries marks; keep it a parametrisation")
            ident, values = raw.id, tuple(raw.values)
        elif len(names) == 1:
            values = (raw,)
        else:
            values = tuple(raw)
        if len(values) != len(names):
            raise ValueError(f"{names} got {len(values)} values: {raw!r}")
        if ident is None and ids is not None:
            ident = ids(raw) if callable(ids) else list(ids)[index]
        kwargs = dict(zip(names, values))
        if ident is not None:
            label = {"case": str(ident)}
            # keep a locale visible even when the case has an id of its own
            for name in names:
                if re.search(r"lang|locale|code", name) and isinstance(kwargs[name], str):
                    label[name] = kwargs[name]
        else:
            label = {name: _shown(value) for name, value in kwargs.items()}
        rows.append((label, kwargs))
    return rows


def cases(*axes) -> Iterator[tuple[dict, dict]]:
    """Every combination of the given parametrisation axes.

    Each axis is ``(argnames, argvalues)`` or ``(argnames, argvalues, ids)``,
    exactly as ``pytest.mark.parametrize`` took them.
    """
    expanded = [_axis(*axis) for axis in axes]
    for combination in itertools.product(*expanded):
        label: dict = {}
        kwargs: dict = {}
        for part_label, part_kwargs in combination:
            for key, value in part_label.items():
                label[key] = f"{label[key]}-{value}" if key in label else value
            kwargs.update(part_kwargs)
        yield label, kwargs


def case_dir(tmp_path: Path, label: dict) -> Path:
    """A fresh, empty directory for one case under the test's ``tmp_path``."""
    stem = "-".join(f"{key}={value}" for key, value in label.items())
    stem = re.sub(r"[^\w.=-]+", "_", stem)[:60] or "case"
    path = tmp_path / stem
    suffix = 1
    while path.exists():
        suffix += 1
        path = tmp_path / f"{stem}-{suffix}"
    path.mkdir(parents=True)
    return path


def monkeypatch_scope():
    """A monkeypatch whose changes are undone when its case ends."""
    return pytest.MonkeyPatch.context()
