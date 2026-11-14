"""One uncovered decision each, in four modules that are otherwise complete.

Grouped rather than split into four files because each is a single
branch and the reasoning per module is a paragraph, not a suite. Two
are live and driven; two are dead and pinned to the code that makes
them dead.
"""
from __future__ import annotations

import inspect


# ---------------------------------------------------------------------------
# spacr/qt/fractal_defaults.py -- clamp
# ---------------------------------------------------------------------------

class TestClamp:
    """The whole function was unexecuted: nothing called it under test.

    It is the guard between a preferences field and the fractal's speed
    and depth, so a value that slips past it is a fractal running at a
    number no slider can produce.
    """

    def test_a_value_inside_the_range_comes_back_unchanged(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(2.5, 1.0, 6.0) == 2.5

    def test_below_the_floor_returns_the_floor(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(-100.0, 1.0, 6.0) == 1.0

    def test_above_the_ceiling_returns_the_ceiling(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(1e9, 1.0, 6.0) == 6.0

    def test_the_bounds_themselves_are_inside(self):
        from spacr.qt.fractal_defaults import clamp

        assert clamp(1.0, 1.0, 6.0) == 1.0
        assert clamp(6.0, 1.0, 6.0) == 6.0

    def test_the_defaults_it_exists_to_hold_are_within_their_own_range(self):
        from spacr.qt import fractal_defaults as fd

        assert fd.clamp(fd.DEFAULT_SPEED_MAX,
                        0.0, fd.DEFAULT_SPEED_MAX) == fd.DEFAULT_SPEED_MAX


# ---------------------------------------------------------------------------
# spacr/organelle_types.py -- a preset with no members
# ---------------------------------------------------------------------------

class TestExplainingAPresetThatNamesNothing:

    def test_custom_is_the_preset_with_no_members(self):
        from spacr.organelle_types import ORGANELLE_TYPES

        empty = [name for name, preset in ORGANELLE_TYPES.items()
                 if not preset.members]
        assert empty == ["custom"], (
            "the set of member-less presets changed; the 'covers:' line "
            "below is skipped for exactly these")

    def test_a_member_less_preset_prints_no_covers_line(self, capsys):
        """THE UNCOVERED BRANCH.

        Every preset that names structures carries a 'covers:' line.
        One that names none must omit the line rather than print an
        empty one -- "covers: " reads as a preset that covers nothing,
        which is a different claim from making no claim.
        """
        from spacr.organelle_types import OrganelleType, _explain

        nameless = OrganelleType(
            label="Nameless", members=(), method="log",
            morphology="spots", size_split=None, params={}, caveat="")

        _explain(nameless, 6, {"organelle_method": "log"}, {})
        printed = capsys.readouterr().out

        assert "Nameless" in printed
        assert "set    organelle_method" in printed
        assert "covers:" not in printed


# ---------------------------------------------------------------------------
# spacr/portable_paths.py -- the climb never revisits a folder
# ---------------------------------------------------------------------------

class TestTheClimbNeverRepeatsItself:

    def test_the_duplicate_guard_cannot_fire(self):
        """THE PIN.

        ``here`` is an absolute path, so never empty, and the loop stops
        the moment ``dirname`` returns the same path it was given -- so
        every step is strictly shorter than the last and cannot already
        be in the list. The check is cheap insurance, not a live path.

        This fails if the ``parent == here`` break is removed, which is
        what would make a repeat possible.
        """
        from spacr import portable_paths

        source = inspect.getsource(portable_paths.candidate_roots)
        assert "if parent == here:" in source and "break" in source, (
            "the climb no longer stops at the filesystem root, so it can "
            "revisit a folder and the duplicate guard is live")
