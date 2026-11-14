"""Where the magic wand's rescues cut, taper and cap a runaway flood.

Three units, each walked as a case table with exact answers:

* ``trim_directional_runaway`` cuts a flood where its width profile steps
  up, in each of the four directions, and reports the cut as the image row
  or column it was made at; a click off the frame or a clean flood cuts
  nothing and hands back a copy;
* ``taper_region_to_intensity`` moves a straight cut onto the nearest
  intensity edge, reads a colour image as its channel mean, and gives back
  a copy of the provisional region when the click is not in it or nothing
  was cut away;
* ``wand_region`` runs the rescues in order and reports what each did:
  the cut directions, whether the intensity border and the taper ran,
  whether the budget capped or rejected the flood, and the pixel counts.

Geometry only -- nothing here needs a Qt widget.
"""
from __future__ import annotations

import numpy as np

from spacr.qt import wand_rescue as W


def _leaks_upward():
    """A narrow bar under the click, opening into the full width above it."""
    region = np.zeros((20, 20), dtype=bool)
    region[6:15, 8:12] = True
    region[0:6, :] = True
    return region


def _box(shape, rows, cols):
    out = np.zeros(shape, dtype=bool)
    out[rows, cols] = True
    return out


_DETECTOR = dict(ratio=2.0, warmup=2, min_baseline=1, confirm=1)


def test_trim_directional_runaway_cuts_each_leak_at_its_image_coordinate(
        subtests):
    """Each leak direction is cut at the reported row/column; no leak cuts nothing."""
    up = _leaks_upward()
    clean = np.pad(np.ones((4, 4), dtype=bool), 3)
    # case, region, seed (row, col), expected cuts, expected kept region
    cases = [
        ("leak upwards", up, (10, 9), {"up": 5},
         _box((20, 20), slice(6, 15), slice(8, 12))),
        ("leak downwards", np.flipud(up), (9, 9), {"down": 14},
         _box((20, 20), slice(5, 14), slice(8, 12))),
        ("leak leftwards", up.T, (9, 10), {"left": 5},
         _box((20, 20), slice(8, 12), slice(6, 15))),
        ("leak rightwards", np.fliplr(up.T), (9, 9), {"right": 14},
         _box((20, 20), slice(8, 12), slice(5, 14))),
        ("a clean flood", clean, (4, 4), {}, clean),
        ("a click below the frame", up, (99, 9), {}, up),
        ("a click right of the frame", up, (9, 99), {}, up),
        ("a click above the frame", up, (-1, 9), {}, up),
    ]
    for case, region, seed, cuts, kept in cases:
        with subtests.test(msg=case):
            before = region.copy()
            out, found = W.trim_directional_runaway(region, seed, **_DETECTOR)
            assert found == cuts
            np.testing.assert_array_equal(out, kept)
            assert out is not region
            np.testing.assert_array_equal(region, before)

    with subtests.test(msg="default guards see no leak in a short bar"):
        # warmup=12 / min_baseline=8 by default: the 9-row bar never gets
        # past the warm-up, so the same leak goes unjudged.
        out, found = W.trim_directional_runaway(up, (10, 9))
        assert found == {}
        np.testing.assert_array_equal(out, up)


def _tapering_case():
    image = np.zeros((24, 24), dtype=np.float32)
    image[6:18, 6:18] = 200.0
    flooded = _box((24, 24), slice(6, 18), slice(6, 18))
    provisional = _box((24, 24), slice(6, 14), slice(6, 18))  # a straight cut
    return image, flooded, provisional


def test_taper_region_to_intensity_moves_the_cut_or_returns_a_copy(subtests):
    """The taper's answer for a grey, a colour, an off-click and a nothing-cut case."""
    image, flooded, provisional = _tapering_case()
    colour = np.repeat(image[:, :, None], 3, axis=2)
    moved = _box((24, 24), slice(6, 15), slice(6, 18))
    # case, image, flooded, provisional, seed, expected
    cases = [
        ("grey image", image, flooded, provisional, (10, 10), moved),
        ("colour image reads as its channel mean", colour, flooded,
         provisional, (10, 10), moved),
        ("click outside the provisional region", image, flooded,
         provisional, (20, 20), provisional),
        ("click outside the frame", image, flooded, provisional, (30, 3),
         provisional),
        ("nothing was cut away", image, provisional, provisional, (10, 10),
         provisional),
    ]
    for case, img, flood, prov, seed, expected in cases:
        with subtests.test(msg=case):
            out = W.taper_region_to_intensity(img, flood, prov, seed,
                                              margin=2)
            np.testing.assert_array_equal(out, expected)
            assert out is not prov


def _leaky_image():
    """An object joined to a bright field by a one-pixel seam."""
    image = np.zeros((64, 64), dtype=np.float32)
    image[20:31, 20:31] = 200.0                # the object
    image[25, 30:41] = 200.0                   # the seam it escapes along
    image[10:51, 40:61] = 200.0                # the field it escapes into
    return image


def _square():
    image = np.zeros((40, 40), dtype=np.float32)
    image[10:31, 10:31] = 200.0                # 441 pixels
    return image


def _report(**changes):
    report = {"flooded_px": 991, "kept_px": 130, "cuts": ["right"],
              "intensity_border": False, "refined_tolerance": 10.0,
              "tapered": False, "capped": False, "rejected": False}
    report.update(changes)
    return report


def test_wand_region_reports_what_each_rescue_did(subtests, monkeypatch):
    """Every rescue combination gives its exact region size and report."""
    leaky = _leaky_image()
    colour = np.repeat(leaky[:, :, None], 3, axis=2)
    cut_object = np.zeros((64, 64), dtype=bool)
    cut_object[20:31, 20:31] = True
    cut_object[25, 30:40] = True               # the seam up to the cut column
    no_cap = dict(intensity_border=False, gradient_taper=False)
    capped = {"flooded_px": 441, "cuts": [], "capped": True}
    # case, image, (x, y), kwargs, expected report, expected region or None
    cases = [
        ("straight cut with both refinements off", leaky, (25, 25), no_cap,
         _report(), cut_object),
        ("colour image answers as its grey", colour, (25, 25), no_cap,
         _report(), cut_object),
        ("intensity border finds no clean tolerance", leaky, (25, 25),
         dict(gradient_taper=False), _report(), cut_object),
        ("taper on, leak cut", leaky, (25, 25), {},
         _report(tapered=True), None),
        ("cap with the taper off keeps exactly the budget", _square(),
         (20, 20), dict(max_pixels=40, gradient_taper=False),
         _report(kept_px=40, **capped), None),
        ("cap with the taper on tapers the arc", _square(), (20, 20),
         dict(max_pixels=40, gradient_taper=True),
         _report(kept_px=40, tapered=True, **capped), None),
        ("over the cap without salvage is rejected", _square(), (20, 20),
         dict(max_pixels=40, salvage_over_cap=False),
         _report(kept_px=0, rejected=True, **capped),
         np.zeros((40, 40), dtype=bool)),
        ("a clean flood is the flood", _square(), (0, 0), {},
         _report(flooded_px=1159, kept_px=1159, cuts=[]),
         _square() == 0),
        ("unknown settings are ignored", _square(), (0, 0),
         dict(not_a_rescue=True),
         _report(flooded_px=1159, kept_px=1159, cuts=[]), _square() == 0),
    ]
    for case, image, (x, y), kwargs, report, region in cases:
        with subtests.test(msg=case):
            out, got = W.wand_region(image, x, y, 10.0, **kwargs)
            assert got == report
            assert int(out.sum()) == report["kept_px"]
            if region is not None:
                np.testing.assert_array_equal(out, region)
            if report["kept_px"]:
                assert out[y, x], "the clicked pixel was thrown away"

    with subtests.test(msg="a seed off the image floods nothing"):
        out, got = W.wand_region(leaky, 99, 99, 10.0)
        assert got == _report(flooded_px=0, kept_px=0, cuts=[])
        assert not out.any()

    with subtests.test(msg="a taper that finds nothing keeps the straight cut"):
        expected, _ = W.wand_region(leaky, 25, 25, 10.0, gradient_taper=False)
        offered = []

        def finds_nothing(_grey, _flood, region, *args, **kwargs):
            offered.append(region.copy())
            return np.zeros(leaky.shape, dtype=bool)

        with monkeypatch.context() as patch:
            patch.setattr(W, "taper_region_to_intensity", finds_nothing)
            out, got = W.wand_region(leaky, 25, 25, 10.0,
                                     max_pixels=100_000, gradient_taper=True)
        assert len(offered) == 1
        np.testing.assert_array_equal(offered[0], expected)
        np.testing.assert_array_equal(out, expected)
        assert got == _report()
