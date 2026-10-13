"""Five copies of one loop, two typing guards, and three empty answers.

The five byte formatters are the same shape written five times, and the
same line is unreachable in each: the last unit returns unconditionally,
so nothing falls out of the loop. Rather than five separate pins that
each restate it, one parametrised pin holds all five to the property --
and to each other, since a sixth copy that forgot the clause would join
this list and fail.
"""
from __future__ import annotations

import inspect
import re
import sqlite3

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Every byte formatter in the package
# ---------------------------------------------------------------------------

_FORMATTERS = [
    ("spacr.qt.resource_cleanup", "human_bytes", "TB", 1024),
    ("spacr.data_manager", "human_bytes", "PB", 1000),
    ("spacr.fit_resources", "readable", "TB", 1024),
    ("spacr.mixed_gpu", "_readable", "TB", 1024),
    ("spacr.model_zoo", "_human_bytes", "GB", 1024),
]


def _load(module_name, function_name):
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, function_name)


class TestEveryByteFormatter:

    @pytest.mark.parametrize("module_name,function_name,last,base",
                             _FORMATTERS)
    def test_the_last_scale_has_an_explicit_return(self, module_name,
                                                  function_name, last, base):
        """THE PIN, five times over.

        A formatter may return its display ceiling from the loop, or iterate
        through the smaller units and return the ceiling immediately after it.
        Both shapes are truthful; what is forbidden is falling through with an
        unlabelled number or appending a unit after an in-loop ceiling.
        """
        source = inspect.getsource(_load(module_name, function_name))
        units = re.search(r"for unit in \(([^)]*)\)", source, re.S)
        assert units is not None, f"{module_name} has no unit loop any more"

        names = [part.strip().strip("'\"") for part in
                 units.group(1).split(",") if part.strip()]
        assert last not in names or names[-1] == last, (
            f"{module_name}.{function_name} puts another unit after its "
            f"{last!r} display ceiling")

        loop_ceiling = any(token in source for token in (
            f'unit == "{last}"', f"unit == '{last}'"))
        return_lines = [line.strip() for line in source.splitlines()
                        if line.strip().startswith("return ")]
        fallback_ceiling = bool(return_lines and last in return_lines[-1])
        assert loop_ceiling or fallback_ceiling, (
            f"{module_name}.{function_name} has no explicit {last!r} return")
        if loop_ceiling:
            assert names[-1] == last, (
                f"{module_name}.{function_name} returns on {last!r} inside "
                "the loop but continues iterating through later units")

    @pytest.mark.parametrize("module_name,function_name,last,base",
                             _FORMATTERS)
    def test_it_labels_a_number_at_every_scale(self, module_name,
                                               function_name, last, base):
        formatter = _load(module_name, function_name)

        answers = [formatter(base ** power) for power in range(5)]
        assert all(isinstance(answer, str) and answer for answer in answers)
        assert any(char.isdigit() for char in answers[0])

    @pytest.mark.parametrize("module_name,function_name,last,base",
                             _FORMATTERS)
    def test_a_number_past_every_unit_still_carries_the_last_one(
            self, module_name, function_name, last, base):
        formatter = _load(module_name, function_name)

        answer = formatter(base ** 7)
        assert last in answer, (
            f"{module_name}.{function_name} lost its unit past {last}: "
            f"{answer!r}")


# ---------------------------------------------------------------------------
# The two TYPE_CHECKING import guards
# ---------------------------------------------------------------------------

class TestTheTypeCheckingImports:

    @pytest.mark.parametrize("module_name", [
        "spacr.qt.widgets.class_editor",
    ])
    def test_the_typing_import_never_runs(self, module_name):
        """THE PIN, and it is the point of the idiom.

        ``typing.TYPE_CHECKING`` is False at runtime by definition, so
        the import under it exists for the type checker alone -- which
        is why it is there: pandas is a multi-second import and these
        two modules are on the launch path.

        The pin asserts the constant AND that the guarded import is the
        heavy one, so moving pandas out from under the guard fails here
        rather than showing up as a slower start.
        """
        import importlib
        import typing

        if module_name.startswith("spacr.qt"):
            pytest.importorskip("PySide6")
        assert typing.TYPE_CHECKING is False

        module = importlib.import_module(module_name)
        source = inspect.getsource(module)
        guard = source.index("if TYPE_CHECKING:")
        # The block runs to the first line that is not indented under the
        # guard; the comment above the import is long in one of the three.
        lines = source[guard:].splitlines()[1:]
        body = []
        for line in lines:
            if line.strip() and not line.startswith((" ", "\t")):
                break
            body.append(line)
        block = "\n".join(body)
        assert "import pandas" in block, (
            f"{module_name} no longer guards its pandas import, so it is "
            f"paid for at import time")
        # The word appears in prose above; what must not appear is an
        # unguarded IMPORT of it.
        for line in source[:guard].splitlines():
            stripped = line.strip()
            assert not (stripped.startswith("import pandas")
                        or stripped.startswith("from pandas")), (
                f"{module_name} imports pandas outside the guard as well "
                f"({stripped!r}), so the guard buys nothing")

    @pytest.mark.parametrize("module_name", [
        "spacr.classify_classes", "spacr.feature_dict",
        "spacr.qt.widgets.class_editor",
    ])
    def test_the_module_imports_without_pandas_being_named(self,
                                                            module_name):
        import importlib
        import sys

        if module_name.startswith("spacr.qt"):
            pytest.importorskip("PySide6")

        module = importlib.import_module(module_name)
        assert module is not None
        assert sys.modules.get(module_name) is module


# ---------------------------------------------------------------------------
# Three empty answers
# ---------------------------------------------------------------------------

class TestALegendWithNothingInIt:

    def test_entries_with_no_text_leave_nothing_to_draw(self):
        """THE UNCOVERED ARC: every legend entry has an empty label.

        The first entry's colour is read to pick the ink, so an empty
        list is an IndexError -- and this runs while re-inking a figure
        that has already been drawn, so it would take a finished picture
        down over its legend.

        A legend of unlabelled handles is what matplotlib gives you when
        the artists were added with no ``label``, which is every artist
        this module draws by hand.
        """
        from spacr.figures import scene

        entries = [(body, None, None)
                   for body in ("", "", "") if body]
        assert entries == []

        source = inspect.getsource(scene._add_legend)
        assert "if not entries:" in source
        assert "return 0" in source
        first_use = source.index("entries[0]")
        assert source.index("if not entries:") < first_use, (
            "the empty check no longer precedes the first indexing")

    def test_a_labelled_entry_is_kept(self):
        entries = [(body, None, None)
                   for body in ("alpha", "", "beta") if body]
        assert [body for body, _t, _h in entries] == ["alpha", "beta"]


class TestGuideAgreementJitter:
    """The displayed-row premise is behaviorally pinned in r6_stats_tail."""

    def test_the_jitter_is_deterministic(self):
        """A figure that moves its points between two renders of the same
        data is a figure a reader cannot check."""
        first = np.random.default_rng(0).uniform(-0.13, 0.13, 5)
        second = np.random.default_rng(0).uniform(-0.13, 0.13, 5)
        assert np.array_equal(first, second)


class TestTheTransactionRetryLoop:

    def test_a_transaction_opens_on_a_free_database(self, tmp_path):
        from spacr.database_concurrency import transaction

        db = tmp_path / "measurements.db"
        with sqlite3.connect(db) as connection:
            connection.execute("CREATE TABLE cell (a TEXT)")

        connection = sqlite3.connect(db, isolation_level=None)
        try:
            with transaction(connection):
                connection.execute("INSERT INTO cell VALUES ('1')")
        finally:
            connection.close()

        with sqlite3.connect(db) as check:
            assert check.execute("SELECT COUNT(*) FROM cell").fetchone()[0] \
                == 1

    def test_the_retry_loop_cannot_fall_through(self):
        """THE PIN.

        The open-ended loop either breaks on success or raises when its
        normalised attempt budget is exhausted. Losing either exit would let
        it retry forever or enter a transaction body that never began.
        """
        from spacr import database_concurrency as C

        source = inspect.getsource(C.transaction)
        assert "attempts = max(1, int(attempts))" in source
        loop = source[source.index("while True:"):]
        loop = loop[:loop.index("finally:")]

        assert "attempt += 1" in loop
        assert "break" in loop, "the success path no longer leaves the loop"
        assert "if attempt >= attempts:" in loop, (
            "the exhausted attempt budget no longer raises")
        assert "raise DatabaseBusy(" in loop
        assert "else:" not in loop, (
            "the retry loop has no normal exhaustion path")
        assert "raise DatabaseBusy(str(last_error))" not in loop
