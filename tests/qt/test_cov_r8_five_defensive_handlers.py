"""Five handlers guarding against something that cannot happen yet.

Each one is already marked ``pragma: no cover`` in the source with a
reason. What is missing is a test that fails when the reason stops
holding -- a comment cannot notice that pandas changed its return shape,
or that JobRunner started refusing work.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


class TestTheDefaultPngMapping:

    def test_it_is_read_from_crops_rather_than_copied(self):
        """THE PIN, for a handler over a hard dependency.

        The mapping is IMPORTED rather than written out again, because a
        literal copy is exactly how the preview and the run came to
        disagree in the first place. The import is guarded only because
        this module is loaded to build a screen and an import error
        there would be a blank panel.

        ``spacr.crops`` is a hard dependency, so the handler cannot run
        -- and the fallback it returns is the value to check, since a
        fallback that drifted from the real default would reintroduce
        the disagreement it exists to avoid.
        """
        from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING
        from spacr.qt.widgets import measure_preview as M

        source = inspect.getsource(M)
        assert "from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING" in source

        fallback = {"r": 2, "g": 1, "b": 0}
        assert f'return {fallback}'.replace("'", '"') in source.replace(
            "'", '"'), "the fallback literal changed shape"
        assert dict(DEFAULT_PNG_CHANNEL_MAPPING) == fallback, (
            "the fallback no longer matches spacr.crops' real default, so a "
            "run that hit it would disagree with the pipeline")


class TestTheJoinButtonReset:

    def test_the_button_is_reset_only_if_the_job_did_not_start(self):
        """THE PIN, for ``if not started``.

        ``JobRunner.submit`` always answers True today, so the reset
        cannot run -- but a runner that started refusing work (a queue
        cap, a shutdown in progress) would leave the dialog with its
        join button disabled and no job coming, which is a dialog that
        cannot be used again without being closed.
        """
        from spacr.qt.widgets import measurement_compare_dialog as D

        source = inspect.getsource(D)
        assert "started = self._jobs.submit(work, self._finish_join)" in source
        assert "if not started:" in source
        assert "self._joining = False" in source
        assert "self._reset_the_join_button()" in source

        from spacr.qt.job_runner import JobRunner

        submit = inspect.getsource(JobRunner.submit)
        assert "return True" in submit, (
            "JobRunner.submit no longer returns True unconditionally, so the "
            "reset above is live and needs a test of its own")


class TestClearingTheResultsSelection:

    def test_the_clear_is_blocked_so_it_does_not_emit_a_third_time(self):
        """The reason for the block, which is the substance here.

        Clearing a selection emits, and an emit from a reset re-enters
        the very handler that asked for the reset -- so the clear is
        blocked and the previous block state restored rather than being
        set to False.
        """
        from spacr.qt.widgets import regression_results as R

        source = inspect.getsource(R)
        assert "blocked = self.table.table.blockSignals(True)" in source
        assert "self.table.table.blockSignals(blocked)" in source, (
            "the block state is no longer restored, so a caller that had "
            "already blocked signals gets them back unexpectedly")

    def test_a_table_that_is_gone_does_not_stop_the_rest_of_the_reset(self):
        """THE PIN, for ``except (RuntimeError, AttributeError)``.

        The plots and the histograms below still have highlights to
        clear, and a results panel whose table went during a rebuild
        must not keep them: a highlight left on a plot points at a row
        that is not selected any more.
        """
        from spacr.qt.widgets import regression_results as R

        source = inspect.getsource(R)
        assert "except (RuntimeError, AttributeError):" in source
        handler = source.index("except (RuntimeError, AttributeError):")
        assert "for plot in self._keyed_plots():" in source[handler:
                                                            handler + 400], (
            "the highlight clearing no longer follows the handler, so a "
            "missing table now leaves highlights behind")


class TestTheAggregationTableShape:

    def test_a_list_of_aggregations_gives_a_frame_however_short(self):
        """THE PIN, for ``isinstance(table, pd.Series)``.

        ``SeriesGroupBy.agg`` returns a DataFrame for a LIST of function
        names however short the list is, and the list here is always
        non-empty because ``n`` is always in the spec. So the reshape
        below cannot run.

        Run against pandas rather than quoted, for one name and for
        three, because "however short" is the half that would break.
        """
        frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})
        grouped = frame.groupby("g")

        for wanted in (["mean"], ["mean", "std"], ["count", "mean", "std"]):
            table = grouped["v"].agg(wanted)
            assert isinstance(table, pd.DataFrame), (
                f"agg({wanted}) answered a {type(table).__name__}, so the "
                f"reshape in pivot_spec is live")

    def test_a_bare_name_does_give_a_series_which_is_why_the_list_matters(self):
        frame = pd.DataFrame({"g": ["a", "a", "b"], "v": [1.0, 2.0, 3.0]})

        assert isinstance(frame.groupby("g")["v"].agg("mean"), pd.Series)


class TestChoosingTheScannableColumns:

    def test_a_column_with_one_distinct_value_is_not_offered(self):
        """Nothing to scan across: every object has the same value."""
        column = pd.Series([3.0, 3.0, 3.0])

        assert int(column.nunique(dropna=True)) < 2

    def test_a_column_with_a_spread_is_offered(self):
        column = pd.Series([1.0, 2.0, 3.0])

        assert int(column.nunique(dropna=True)) >= 2

    def test_nunique_cannot_raise_for_a_column_that_got_this_far(self):
        """THE PIN, for ``except TypeError``.

        The two checks above have already refused anything that is not
        numeric and anything boolean, and ``nunique`` over a numeric
        column cannot raise. The handler is for an odd dtype that claims
        to be numeric and is not comparable -- which nothing in the
        package produces.
        """
        from spacr.qt.widgets import measurement_scan_panel as S

        source = inspect.getsource(S)
        assert "if not pd.api.types.is_numeric_dtype(column):" in source
        assert "if pd.api.types.is_bool_dtype(column):" in source
        numeric = source.index("is_numeric_dtype(column)")
        counting = source.index("column.nunique(dropna=True)")
        assert numeric < counting, (
            "the numeric check no longer precedes the count, so a column of "
            "anything at all now reaches nunique")

        for values in ([1, 2, 3], [1.5, np.nan, 2.5], np.arange(4)):
            assert int(pd.Series(values).nunique(dropna=True)) >= 1
