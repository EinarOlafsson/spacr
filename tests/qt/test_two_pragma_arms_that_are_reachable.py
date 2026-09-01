"""Two ``# pragma: no cover`` arms, driven rather than excused.

``.coveragerc`` empties ``exclude_lines`` and ``partial_branches``, so a
pragma marker excludes nothing -- it is a note about why a line looked
hard to reach, and the line is still counted. Instruction 288 counts
these two.

Both are the shape that IS reachable, and the technique is the same one
that closed the import-guard arms elsewhere today: make the thing the
guard guards against actually happen.

* an ``except Exception`` around an import is drivable whether or not the
  package is installed, by making the import raise;
* an ``except (ValueError, NotImplementedError)`` around a matplotlib
  call is drivable by handing it an object whose method raises.

Neither needed a change to the source. They were already correct; they
were merely untested, which the pragma recorded rather than fixed.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets import measure_preview


class TestTheDefaultChannelMapping:
    """``_default_png_mapping`` falls back when ``spacr.crops`` will not
    import."""

    def test_it_reads_the_real_mapping_when_it_can(self):
        """The path taken on every real machine.

        Asserted first so the fallback below means "the guard fired"
        rather than "this function never works".
        """
        from spacr.crops import DEFAULT_PNG_CHANNEL_MAPPING

        assert measure_preview._default_png_mapping() == dict(
            DEFAULT_PNG_CHANNEL_MAPPING)

    def test_it_falls_back_when_crops_will_not_import(self, monkeypatch):
        """THE PRAGMA'D ARM. ``crops`` is a hard dependency, so this
        cannot be reached by uninstalling anything -- but an import can
        always be made to fail.

        The fallback is not arbitrary: r/g/b to 2/1/0 is the mapping the
        preview and the run must agree on, and the comment above it
        records that they once disagreed. So the VALUE is asserted, not
        just that something came back.
        """
        real = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "spacr.crops" or name.startswith("spacr.crops."):
                raise ImportError("blocked for the test")
            return real(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        assert measure_preview._default_png_mapping() == {"r": 2, "g": 1,
                                                          "b": 0}


class TestReleasingADragPatch:
    """``_on_release`` survives a patch that will not remove itself."""

    class _Refuses:
        """A drag patch whose ``remove`` raises, as matplotlib's can.

        A patch already detached from its axes raises ValueError, and
        some artist types raise NotImplementedError -- which is why the
        guard names both.
        """

        def __init__(self, error):
            self._error = error
            self.asked = False

        def remove(self):
            self.asked = True
            raise self._error

    @pytest.mark.parametrize("error", [ValueError("already removed"),
                                       NotImplementedError("cannot remove")])
    def test_a_patch_that_will_not_remove_does_not_break_the_release(
            self, error):
        """THE PRAGMA'D ARM.

        Release must finish: it is what clears the drag state, and a
        release that raises leaves the editor believing a drag is still
        in progress -- so the next click continues a gesture the user
        ended.
        """
        from spacr.qt.widgets.gate_editor import GateCanvas

        # A bare instance with only the state `_on_release` reads before
        # the guard. `_mode` is not a volume mode, so `_in_volume` short-
        # circuits and never asks for axes -- which is what lets this run
        # without a figure.
        editor = GateCanvas.__new__(GateCanvas)
        patch = self._Refuses(error)
        editor._mode = "2D"
        editor._tool = None
        editor._resize = None
        editor._drag_origin = None
        editor._drag_patch = patch

        GateCanvas._on_release(editor, _NoEvent())

        assert patch.asked, "the patch was never asked to remove itself"
        assert editor._drag_patch is None, (
            "the drag patch survived the release, so the next click "
            "continues a gesture the user ended")


class _NoEvent:
    """A matplotlib event that landed on no axes."""

    inaxes = None
    button = 1
    xdata = ydata = None
