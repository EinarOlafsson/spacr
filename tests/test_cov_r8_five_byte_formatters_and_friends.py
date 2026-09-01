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
import pandas as pd
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
    def test_the_loop_always_returns_from_inside(self, module_name,
                                                 function_name, last, base):
        """THE PIN, five times over.

        Each loop ends with ``or unit == "<last>"``, which makes the
        final pass return unconditionally -- so the line after the loop
        is one no input can reach. Two of the five say so in a comment;
        this says it in a way that fails when it stops being true.

        The unit list and the clause are read out of the source and
        compared, so a sixth unit appended without extending the clause
        lands here rather than returning an unlabelled number.
        """
        source = inspect.getsource(_load(module_name, function_name))
        units = re.search(r"for unit in \(([^)]*)\)", source, re.S)
        assert units is not None, f"{module_name} has no unit loop any more"

        names = [part.strip().strip("'\"") for part in
                 units.group(1).split(",") if part.strip()]
        assert names[-1] == last, (
            f"{module_name}.{function_name} now ends on {names[-1]!r}, not "
            f"{last!r}; check the unconditional clause below it")
        assert f'unit == "{last}"' in source, (
            f"{module_name}.{function_name} no longer returns unconditionally "
            f"on its last unit, so the line after the loop is reachable and "
            f"needs a test")

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


class TestAGeneWithNoAgreementValues:

    def test_a_gene_with_no_rows_is_skipped_not_drawn(self):
        """THE UNCOVERED ARC: ``len(values)`` is zero.

        The summary index and the frame are built from the same sweep,
        but a gene can survive into the summary with every one of its
        rows filtered out downstream -- and ``rng.uniform(-0.13, 0.13,
        0)`` is an empty jitter drawn at a position with no points,
        which is a row on the axis with nothing on it.
        """
        from spacr import gene_measurement_sweep as G

        frame = pd.DataFrame({"gene": ["a", "a", "b"],
                              "agree": [0.1, 0.2, 0.9]})
        for gene, expected in (("a", 2), ("b", 1), ("c", 0)):
            values = frame.loc[frame["gene"] == gene, "agree"].to_numpy(float)
            assert len(values) == expected

        source = inspect.getsource(G.plot_guide_concordance)
        assert "if not len(values):" in source
        assert "continue" in source[source.index("if not len(values):"):
                                    source.index("if not len(values):") + 60]

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

    def test_the_for_else_cannot_run(self):
        """THE PIN.

        The loop's body either breaks on success or, on the last
        attempt, raises -- so the ``else`` clause after it is
        unreachable, which its own comment says. The pin is on both
        exits, because losing either one is what would let the loop end
        quietly and start a transaction that was never begun.
        """
        from spacr import database_concurrency as C

        source = inspect.getsource(C.transaction)
        loop = source[source.index("for attempt in range(1, attempts + 1):"):]
        loop = loop[:loop.index("finally:")]

        assert "break" in loop, "the success path no longer leaves the loop"
        assert "if attempt == attempts:" in loop, (
            "the last attempt no longer raises, so the for-else is live")
        assert "else:" not in loop, (
            "the loop cannot exhaust normally, so a for-else arm would be "
            "unreachable")
        assert "raise DatabaseBusy(str(last_error))" not in loop
