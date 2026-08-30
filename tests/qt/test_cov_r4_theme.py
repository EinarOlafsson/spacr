"""The spaceout ink solver: how a contrast band gets spent on colour.

:func:`spacr.qt.theme._hue_ink` is the one place in the module where the
dressing is allowed to make a *text* colour chromatic. Every other role is
moved by :func:`~spacr.qt.theme._hue_shift`, which keeps the luminance it
was handed; the ink roles are handed a whole band of luminances that
:data:`~spacr.qt.theme.CONTRAST_RULES` still allows and told to pick the most
coloured point in it. This file pins that contract:

* the answer is the **most chromatic** colour on the hue whose luminance the
  band admits, and it is more coloured than the plain hue shift it replaces;
* a band **no 8-bit colour lands in** — the bands are solved in the reals and
  the ramps are quantised, so a narrow one really can admit nothing — falls
  back to that plain hue shift rather than to whatever ``best`` happened to
  hold;
* the value ramp it scans is **strictly increasing in chroma**, which is the
  property that makes the scan's "is this better than what I have" guard a
  formality on that ramp (see the guard note in
  ``test_the_value_ramp_never_offers_a_less_coloured_candidate``).

The neighbouring failure paths of the same module — the hopeless scrim, the
ink caught between its surfaces, the undampable drift offset, the widget
whose Qt style has gone, the batched QSS registration — are pinned in
``tests/qt/test_cov_wf_qt_theme.py`` and are deliberately not repeated here.

Nothing here mutates theme state: ``_hue_ink``, ``_hue_shift`` and
``_hue_rgb`` are pure functions of their arguments, and ``_ink_band`` only
reads the palettes (its own dressing context manager restores what it set).
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt import theme                         # noqa: E402

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _value_ramp(hue: float, saturation: float = 1.0):
    """The hue's own colour, darkened — the first ramp ``_hue_ink`` scans."""
    base = theme._hue_rgb(hue, saturation)
    return [(int(round(base[0] * level)),
             int(round(base[1] * level)),
             int(round(base[2] * level)))
            for level in range(256)]


def _tint_ramp(hue: float, saturation: float = 1.0):
    """The hue mixed toward white — the second ramp ``_hue_ink`` scans."""
    base = theme._hue_rgb(hue, saturation)
    out = []
    for step in range(256):
        weight = step / 255.0
        out.append(tuple(int(round(255.0 * (1.0 - weight + weight * channel)))
                         for channel in base))
    return out


def _chroma(rgb) -> int:
    """How coloured a triple is: the spread ``_hue_ink`` maximises."""
    return max(rgb) - min(rgb)


def _reachable_hues():
    """Every hue the dressing can ask an ink role for.

    ``spaceout_palette`` calls ``_hue_ink(seat + drift)``, so the reachable
    set is each seat in :data:`~spacr.qt.theme.SPACEOUT_HUES` offset by each
    step of the drift grid.
    """
    return sorted({(seat + drift) % 360.0
                   for seat in theme.SPACEOUT_HUES.values()
                   for drift in theme._drift_grid()})


@pytest.fixture()
def dark_ink_bands():
    """The solved luminance band of each dark-theme ink role.

    Read through :func:`~spacr.qt.theme._ink_band` rather than written down,
    so the test asks ``_hue_ink`` for the bands the application really hands
    it. Read-only: the band solver restores the dressing it sets.
    """
    bands = {role: theme._ink_band("dark", role)
             for role in theme.SPACEOUT_INK_ROLES}
    assert all(band is not None for band in bands.values()), bands
    return bands


# ---------------------------------------------------------------------------
# What the band buys
# ---------------------------------------------------------------------------

def test_the_ink_is_the_most_coloured_colour_its_band_admits(dark_ink_bands):
    """The point of solving a band at all is that the ink can then be a
    colour instead of a grey. So the answer has to be the *argmax* of chroma
    over the candidates the band admits — not merely some colour inside it —
    and it has to beat the plain hue shift, which is what the role would have
    got for free. A solver that returned the first admissible candidate would
    hand ``fg`` back the white it started as on every hue.
    """
    for role, (low, high) in dark_ink_bands.items():
        fallback = theme.DARK_PALETTE[role]
        for hue in (30.0, 210.0, 300.0):
            ink = theme._hue_ink(hue, low, high, fallback)
            plain = theme._hue_shift(fallback, hue)

            admitted = [rgb for rgb in _value_ramp(hue) + _tint_ramp(hue)
                        if low <= theme._rgb_luminance(rgb) <= high]
            assert admitted, (role, hue)

            # Still readable: the whole reason the band exists.
            assert low <= theme.relative_luminance(ink) <= high, (role, hue)
            # Nothing the band admits is more coloured than what came back.
            assert (_chroma(theme._channels(ink))
                    == max(_chroma(rgb) for rgb in admitted)), (role, hue)
            # And it is a real gain over the shift it replaced.
            assert (_chroma(theme._channels(ink))
                    > _chroma(theme._channels(plain))), (role, hue, ink, plain)


def test_a_band_no_colour_lands_in_leaves_the_role_on_the_plain_hue_shift(
        dark_ink_bands):
    """The bands are solved in the reals and the two ramps are 512 quantised
    8-bit points, so a band can be perfectly valid and still contain none of
    them — near white the ramps step by more than 0.008 of relative
    luminance. The documented answer there is the plain hue shift: unchanged
    and readable. It matters that it is that and not ``best``, which at that
    moment is still the ``None`` it was initialised to.
    """
    hue = 210.0
    fallback = theme.DARK_PALETTE["fg_muted"]
    lumas = sorted(theme._rgb_luminance(rgb)
                   for rgb in _value_ramp(hue) + _tint_ramp(hue))
    gap, below = max((lumas[i + 1] - lumas[i], lumas[i])
                     for i in range(len(lumas) - 1))
    low, high = below + gap * 0.25, below + gap * 0.75

    # The premise, asserted rather than assumed: the band really is empty.
    assert not [luma for luma in lumas if low <= luma <= high]

    assert theme._hue_ink(hue, low, high, fallback) == \
        theme._hue_shift(fallback, hue)

    # The same hue and the same fallback, with the band the theme actually
    # solved for the role: now something IS admitted, and the answer moves
    # off the plain shift and gains colour.
    wide_low, wide_high = dark_ink_bands["fg_muted"]
    coloured = theme._hue_ink(hue, wide_low, wide_high, fallback)

    assert coloured != theme._hue_shift(fallback, hue)
    assert (_chroma(theme._channels(coloured))
            > _chroma(theme._channels(theme._hue_shift(fallback, hue))))


# ---------------------------------------------------------------------------
# Why the value ramp's "is this better" guard never has to say no
# ---------------------------------------------------------------------------

def test_the_value_ramp_never_offers_a_less_coloured_candidate():
    """``_hue_ink`` scans its value ramp keeping the strictest improvement in
    chroma, and on that ramp the guard can never reject: the ramp is
    ``round(base * level)`` on the fully saturated plane, where one channel
    is 1.0 and one is 0.0, so the chroma of the candidate at ``level`` IS
    ``level``. Every step is therefore a strict improvement on every step
    before it, and the scan's result depends only on which levels the band
    admits.

    That is not a fact about ramps in general, and the second half of this
    test drives the case that shows it: at the saturation the *surfaces* are
    dressed with, the same construction plateaus — neighbouring levels land
    on the same chroma — so a scan over that plane would reject candidates
    routinely. It is specifically the full-saturation plane ``_hue_ink`` asks
    for that makes the guard a formality there.
    """
    for hue in _reachable_hues():
        base = theme._hue_rgb(hue)
        assert min(base) == 0.0 and max(base) == 1.0, (hue, base)
        assert [_chroma(rgb) for rgb in _value_ramp(hue)] == list(range(256))

    softened = theme.SPACEOUT_SATURATION["bg"]
    plateaued = [hue for hue in _reachable_hues()
                 if [_chroma(rgb) for rgb in _value_ramp(hue, softened)]
                 != list(range(256))]

    assert plateaued == _reachable_hues()
    spreads = [_chroma(rgb) for rgb in _value_ramp(210.0, softened)]
    assert any(later == earlier
               for earlier, later in zip(spreads, spreads[1:]))
