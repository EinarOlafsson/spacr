"""`load` reporting why a montage cannot be built, and an inert husk.

`load` runs on a worker thread and touches no widget, so every failure
has to come back as `MontageLoad.error` rather than as an exception --
the caller is a tab that must stay on screen and say why. These tests
characterise the refusals it can give, which is what makes it safe to
delete the dead block found beside them.

THE HUSK. Between the "no fractions anywhere" refusal and the crop loop
there was:

    try:
        pass
    except MontageError as error:
        return MontageLoad(...)
    except Exception as error:
        return MontageLoad(..., error=f"Could not read the per-well guide
                                       fractions: {error}")

`pass` cannot raise, so neither handler could ever run. The message in
the second one duplicates the real handler twenty lines above it, which
is where the reading actually happens -- the work was moved out and the
shell was left behind. Seven lines that could not execute, removed.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.qt


def _request(**over):
    """A MontageRequest with the two required fields filled in."""
    from spacr.qt.widgets.cell_montage_view import MontageRequest

    fields = {"name": "guide_1", "effect": 0.5}
    fields.update(over)
    return MontageRequest(**fields)


@pytest.fixture()
def a_database(tmp_path):
    """An empty measurements.db, so the earlier refusal is satisfied."""
    import sqlite3

    path = tmp_path / "measurements.db"
    sqlite3.connect(path).close()
    return str(path)


class TestTheRefusalsInOrder:
    """Each says WHICH thing was missing, because a montage that simply
    fails to appear tells the user nothing about what to attach."""

    def test_no_database_is_reported_before_anything_is_read(self):
        from spacr.qt.widgets.cell_montage_view import load

        result = load(_request(results_path="", count_csvs=(),
                               databases=()))
        assert result.unavailable is True
        assert "No measurement database is attached" in result.error
        assert "Attach one to a plate row first" in result.error

    def test_no_guide_fractions_says_either_source_would_do(self,
                                                            a_database):
        """The refusal above the husk, and it names both ways out.

        No run folder holding regression_data.csv and no count CSV.
        Saying "either one is enough" is the difference between a user
        hunting for the right file and knowing they have a choice.
        """
        from spacr.qt.widgets.cell_montage_view import load

        result = load(_request(results_path="", count_csvs=(),
                               databases=(a_database,)))
        assert result.unavailable is True
        assert "No per-well guide fractions are available" in result.error
        assert "Either one is enough" in result.error

    def test_a_folder_without_regression_data_is_reported(self, tmp_path,
                                                          a_database):
        from spacr.qt.widgets.cell_montage_view import load

        empty = tmp_path / "run"
        empty.mkdir()
        result = load(_request(results_path=str(empty), count_csvs=(),
                               databases=(a_database,)))
        assert result.unavailable is True
        assert result.error

    def test_an_unreadable_count_csv_is_reported_not_raised(self, tmp_path,
                                                            a_database):
        """`load` runs on a worker thread; the caller is a tab that must
        stay on screen and say why. Nothing may propagate."""
        from spacr.qt.widgets.cell_montage_view import load

        broken = tmp_path / "counts.csv"
        broken.write_text("this is not a counts table\n")
        result = load(_request(results_path="",
                               count_csvs=(str(broken),),
                               databases=(a_database,)))
        assert result.error
        assert result.unavailable is True


def test_the_inert_try_block_is_gone():
    """The husk removed above must not come back.

    A `try:` whose body is `pass` cannot raise, so its handlers are
    unreachable by construction -- not defensive, just noise that
    coverage reports for ever.
    """
    import inspect

    from spacr.qt.widgets import cell_montage_view

    source = inspect.getsource(cell_montage_view.load)
    assert "try:\n        pass\n" not in source, (
        "an inert try/except husk is back in load()")
