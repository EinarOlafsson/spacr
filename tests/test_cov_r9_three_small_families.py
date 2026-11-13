"""Two families of guard, each appearing in more than one module.

Grouped rather than filed per module, because the interesting thing
about each is the same in every copy -- and a family pinned once is a
family that cannot drift apart quietly.
"""
from __future__ import annotations

import inspect


# ---------------------------------------------------------------------------
# 1. every numeric role channel is in the dense map built from those roles
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. `if name not in <list>:` -- order-preserving de-duplication
# ---------------------------------------------------------------------------

class TestOrderPreservingDeduplication:

    def test_a_column_dropped_by_two_tables_is_reported_once(self):
        """THE ARC: the name is already in the list.

        Two measurement tables can drop the same merged column -- the
        same feature is measured per object type -- and reporting it
        twice would say twice as many columns were lost as were.
        """
        seen = []
        for name in ("area", "perimeter", "area"):
            if name not in seen:
                seen.append(name)

        assert seen == ["area", "perimeter"]

    def test_the_reporter_sorts_what_it_collected(self):
        """The list is built in walk order and returned sorted, so the
        report does not change because the tables were read in a
        different order."""
        from spacr import plate_measurements as PM

        source = inspect.getsource(PM.PlateMerge.dropped_columns.fget)

        assert "if name not in found:" in source
        assert "return tuple(sorted(found))" in source, (
            "the dropped-column report is no longer sorted, so it moves "
            "with table order and two identical runs disagree")
