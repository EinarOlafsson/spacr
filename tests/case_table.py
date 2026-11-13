"""Run a table of behaviour cases as sub-tests of ONE collected test.

The test-suite reduction (2026-11-12) replaces many one-case-per-function
coverage tests of a single unit with one case-table test per unit.  Each case
is still an ordinary function with ordinary pytest fixtures; this helper gives
every case what pytest would have given a test of its own:

* a fresh ``monkeypatch`` (undone at the end of the case),
* a fresh ``tmp_path`` (an empty sub-directory),
* ``capsys`` / ``capfd`` / ``caplog`` with their buffers emptied first,
* module and class fixtures, built per case and torn down per case (generator
  fixtures run their teardown), autouse fixtures of the case module included,
* ``pytest.mark.parametrize`` expanded into one sub-test per value set,
* any other fixture (a conftest one) taken from the outer test.

A failing case is reported on its own by the ``subtests`` fixture, naming the
case, so a table does not hide which row broke.
"""
from __future__ import annotations

import contextlib
import inspect
import itertools

import pytest
from _pytest.fixtures import FixtureFunctionDefinition

_OUTER = ("capsys", "capfd", "caplog", "capsysbinary", "capfdbinary", "request",
          "recwarn")


def _is_fixture(obj):
    return isinstance(obj, FixtureFunctionDefinition)


def _raw(fixture):
    return fixture._get_wrapped_function()


def _autouse(fixture):
    return bool(getattr(fixture._fixture_function_marker, "autouse", False))


def _expand_parametrize(func):
    """Yield dicts of argument values, one per parametrize combination."""
    marks = [m for m in getattr(func, "pytestmark", []) if m.name == "parametrize"]
    unknown = [m.name for m in getattr(func, "pytestmark", [])
               if m.name not in ("parametrize", "usefixtures")]
    if unknown:
        raise RuntimeError(f"case_table cannot run marks {unknown} on {func!r}")
    axes = []
    for mark in marks:
        names, values = mark.args[0], mark.args[1]
        if isinstance(names, str):
            names = [n.strip() for n in names.split(",") if n.strip()]
        axis = []
        for value in values:
            if isinstance(value, type(pytest.param())):
                value = value.values
            elif len(names) == 1:
                value = (value,)
            axis.append(dict(zip(names, value)))
        axes.append(axis)
    if not axes:
        yield {}
        return
    for combo in itertools.product(*axes):
        merged = {}
        for part in combo:
            merged.update(part)
        yield merged


class _Case:
    """One case: a function, or a method of a class instantiated per run."""

    def __init__(self, entry):
        if isinstance(entry, tuple):
            self.owner, name = entry
            self.func = getattr(self.owner, name)
            self.label = f"{self.owner.__name__}.{name}"
        else:
            self.owner = None
            self.func = entry
            self.label = entry.__name__


def run_cases(request, subtests, entries, tmp_path_factory=None):
    """Run every entry in ``entries`` as one sub-test (or several)."""
    base = request.getfixturevalue("tmp_path")
    ran = 0
    for index, entry in enumerate(entries):
        case = _Case(entry)
        for n, params in enumerate(_expand_parametrize(case.func)):
            label = case.label if not params else f"{case.label}[{n}]"
            with subtests.test(msg=label):
                _run_one(request, case, params, base / f"case{index}_{n}")
            ran += 1
    return ran


def _run_one(request, case, params, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    func = case.func
    module_globals = func.__globals__
    instance = case.owner() if case.owner is not None else None
    with contextlib.ExitStack() as stack:
        cache = {}

        def lookup(name):
            if name in cache:
                return cache[name]
            if name in params:
                value = params[name]
            elif name == "monkeypatch":
                value = stack.enter_context(pytest.MonkeyPatch.context())
            elif name == "tmp_path":
                value = workdir
            elif name in _OUTER:
                value = request.getfixturevalue(name)
                if name in ("capsys", "capfd", "capsysbinary", "capfdbinary"):
                    value.readouterr()
                elif name == "caplog":
                    value.clear()
            else:
                fixture = None
                if instance is not None:
                    fixture = getattr(case.owner, name, None)
                    if not _is_fixture(fixture):
                        fixture = None
                if fixture is None:
                    candidate = module_globals.get(name)
                    fixture = candidate if _is_fixture(candidate) else None
                if fixture is None:
                    value = request.getfixturevalue(name)
                else:
                    value = build(fixture, instance if fixture is not None
                                  and instance is not None
                                  and getattr(case.owner, name, None) is fixture
                                  else None)
            cache[name] = value
            return value

        def build(fixture, owner_instance):
            raw = _raw(fixture)
            sig = inspect.signature(raw)
            kwargs = {}
            for pname in sig.parameters:
                if pname == "self":
                    continue
                kwargs[pname] = lookup(pname)
            if owner_instance is not None:
                result = raw(owner_instance, **kwargs)
            else:
                result = raw(**kwargs)
            if inspect.isgenerator(result):
                value = next(result)
                stack.callback(_finish, result)
                return value
            return result

        def _finish(generator):
            for _ in generator:
                pass

        # autouse fixtures of the case module run for every case
        for name, obj in list(module_globals.items()):
            if _is_fixture(obj) and _autouse(obj):
                lookup(name)
        for mark in getattr(func, "pytestmark", []):
            if mark.name == "usefixtures":
                for name in mark.args:
                    lookup(name)

        sig = inspect.signature(func)
        kwargs = {}
        for pname in sig.parameters:
            if pname == "self":
                continue
            kwargs[pname] = lookup(pname)
        if instance is not None:
            getattr(instance, func.__name__)(**kwargs)
        else:
            func(**kwargs)
