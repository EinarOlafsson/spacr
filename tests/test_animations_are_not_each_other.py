"""Two settings must not be illustrated by the same picture.

Area filters vary size; absolute mean-intensity filters vary brightness.
The latter have fixed numeric object means and an inclusive numeric bound,
not a percentile or a fixed quota of objects to remove. Perimeter merging
still has a surviving pair that fails its shared-boundary criterion.
"""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from spacr.setting_animations import setting_animations

TOOLS = Path(__file__).resolve().parent.parent / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

KINDS = ("cell", "nucleus", "pathogen", "organelle")
#: The kinds that still have a maximum-area and two mean-bound animations.
#: The maintainer retired cell, nucleus and pathogen's on 2026-09-25 (item
#: 511) into object_filters rows, so their animations went with the keys;
#: :func:`test_the_retired_object_bound_animations_are_gone` holds that.
BOUND_KINDS = ("organelle",)

#: The bug measured 0.9-2.0%. Anything this close is one picture with a
#: decoration on it, not two illustrations.
MIN_DISTINCT_INK = 0.15


def _frames(path):
    frames, durations = [], []
    with Image.open(path) as image:
        try:
            while True:
                frames.append(np.asarray(image.convert("RGB"), dtype=np.int16))
                durations.append(image.info.get("duration", 100))
                image.seek(image.tell() + 1)
        except EOFError:
            pass
    edges = np.cumsum([0, *durations], dtype=float)
    return frames, edges / edges[-1]


def _ink_difference(first, second):
    """Differing pixels as a fraction of the area either animation draws in.

    Normalising by the FRAME instead of by the ink is what let this hide: a
    360x360 canvas is mostly black, so two line drawings that share every
    stroke still differ in "only 0.1% of the frame" and look fine.
    """
    a, a_edges = _frames(first)
    b, b_edges = _frames(second)
    # Pillow combines repeated hold frames. Compare at every transition in
    # either normalized timeline, rather than treating a longer hold as a
    # different number of animation steps or ignoring an unmatched frame.
    times = np.unique(np.concatenate((a_edges[:-1], b_edges[:-1])))
    pairs = [
        (a[np.searchsorted(a_edges, t, side="right") - 1],
         b[np.searchsorted(b_edges, t, side="right") - 1])
        for t in times
    ]
    drawn = np.logical_or(
        np.logical_or.reduce([f.sum(2) > 45 for f in a]),
        np.logical_or.reduce([f.sum(2) > 45 for f in b]),
    )
    differs = np.logical_or.reduce(
        [np.abs(x - y).sum(2) > 30 for x, y in pairs])
    return float((differs & drawn).sum()) / float(drawn.sum())


@pytest.fixture(scope="module")
def paths():
    return {animation.slug: animation.path for animation in setting_animations()}


@pytest.mark.parametrize("kind", BOUND_KINDS)
@pytest.mark.parametrize("bound", ("min", "max"))
def test_area_and_mean_intensity_are_different_pictures(kind, bound, paths):
    got = _ink_difference(
        paths[f"{kind}_{bound}_area"], paths[f"{kind}_{bound}_intensity"])
    assert got > MIN_DISTINCT_INK, (
        f"{kind}_{bound}_area and {kind}_{bound}_intensity differ in only "
        f"{got:.1%} of their drawn area; they illustrate different criteria"
    )


@pytest.mark.parametrize("kind", BOUND_KINDS)
def test_lower_and_upper_mean_bounds_remove_different_objects(kind, paths):
    got = _ink_difference(
        paths[f"{kind}_min_intensity"], paths[f"{kind}_max_intensity"])
    assert got > MIN_DISTINCT_INK, f"{kind}: lower/upper bounds look alike ({got:.1%})"


class TestTheSceneDrawsItsCriterion:
    """The shared-perimeter criterion must leave its rejecting pair alone."""

    def _record(self, kind, action):
        gen = pytest.importorskip("generate_setting_animations")
        outlines, lines = [], []
        spec = next(
            s for s in gen._specs()
            if s.slug == f"{kind}_perimeter_fraction"
        )

        class Recorder:
            def rectangle(self, *a, **k):
                pass

            def line(self, points, color=None, width=0.5, **k):
                lines.append((list(points), color, width))

            def __getattr__(self, name):
                return lambda *a, **k: None

        painter = Recorder()
        real = gen._object_outline

        def spy(_painter, kind_, center, size, amount=1.0, phase=0.0, **k):
            outlines.append((kind_, center, size, amount))

        gen._object_outline = spy
        gen_well = gen._well
        gen._well = lambda *a, **k: None
        try:
            gen._generic_merge(painter, spec, action)
        finally:
            gen._object_outline = real
            gen._well = gen_well
        return outlines, lines

    @pytest.mark.parametrize("kind", KINDS)
    def test_a_pair_survives_the_merge(self, kind):
        outlines, lines = self._record(kind, 1.0)
        assert lines == []
        solid = [o for o in outlines if o[3] >= 0.99]
        # one merged object plus the pair that failed the criterion
        assert len(solid) == 3, [(o[1], o[3]) for o in outlines]
        survivors = sorted(solid, key=lambda o: -o[1][1])[:2]
        assert survivors[0][1][1] == survivors[1][1][1], "the pair is not level"
        assert survivors[0][1][0] != survivors[1][1][0], "the pair is one object"

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_perimeter_variant_separates_its_surviving_pair(self, kind):
        """Its criterion is shared boundary length, so the pair that fails it
        must share less boundary than the pair that passes -- which is
        distance, and is checkable."""
        outlines, _ = self._record(kind, 0.0)
        levels = {}
        for _kind, center, _size, amount in outlines:
            if amount >= 0.99:
                levels.setdefault(center[1], []).append(center[0])
        pairs = {y: xs for y, xs in levels.items() if len(xs) == 2}
        assert len(pairs) == 2, f"expected two pairs before the merge: {levels}"
        merging_y = min(pairs)
        surviving_y = max(pairs)
        gap = {y: abs(xs[0] - xs[1]) for y, xs in pairs.items()}
        assert gap[surviving_y] > gap[merging_y] * 1.5, (
            "the pair that survives must overlap visibly less than the pair "
            f"that merges: {gap}"
        )


class TestAbsoluteMeanIntensityScenes:
    def _record(self, kind, bound_name, action, monkeypatch, **params):
        gen = pytest.importorskip("generate_setting_animations")
        spec = next(s for s in gen._specs() if s.slug == f"{kind}_{bound_name}_intensity")
        spec = replace(spec, params={**spec.params, **params})
        outlines, labels = [], []

        class Recorder:
            point = staticmethod(lambda point: point)

            def __init__(self):
                self.draw = self

            def text(self, point, text, **kwargs):
                labels.append((point, text, kwargs["fill"]))

        def outline(_painter, kind_, center, size, amount=1.0, **kwargs):
            outlines.append((kind_, center, size, amount))

        with monkeypatch.context() as patch:
            patch.setattr(gen, "_object_outline", outline)
            patch.setattr(gen, "_well", lambda *_args: None)
            patch.setattr(gen, "_font", lambda *_args: None)
            gen._mean_intensity_scene(Recorder(), spec, action)
        return outlines, labels

    @pytest.mark.parametrize("kind", BOUND_KINDS)
    @pytest.mark.parametrize("bound", ("min", "max"))
    def test_equal_sizes_vary_mean_brightness_not_area(self, kind, bound, monkeypatch):
        outlines, labels = self._record(kind, bound, 0, monkeypatch)
        assert len(outlines) == 4
        assert len({o[2] for o in outlines}) == 1
        assert [o[3] for o in outlines] == pytest.approx([0.55, 0.7, 0.85, 1.0])
        assert [label[1] for label in labels[1:]] == ["μ=20", "μ=40", "μ=60", "μ=80"]
        assert labels[0][1] == "0"

    @pytest.mark.parametrize("kind", BOUND_KINDS)
    @pytest.mark.parametrize("bound,expected,caption", [
        ("min", [0, 0, 0.85, 1.0], "≥ 60"),
        ("max", [0.55, 0.7, 0, 0], "≤ 40"),
    ])
    def test_bound_equality_survives_and_only_the_rejected_masks_fade(
            self, kind, bound, expected, caption, monkeypatch):
        outlines, labels = self._record(kind, bound, 1, monkeypatch)
        assert [o[3] for o in outlines] == pytest.approx(expected)
        assert labels[0][1] == caption

    @pytest.mark.parametrize("kind", BOUND_KINDS)
    @pytest.mark.parametrize("bound", ("min", "max"))
    def test_zero_is_off_even_at_the_filtered_endpoint(self, kind, bound, monkeypatch):
        outlines, labels = self._record(kind, bound, 1, monkeypatch, bound=0)
        assert [o[3] for o in outlines] == pytest.approx([0.55, 0.7, 0.85, 1.0])
        assert labels[0][1] == "0"

    @pytest.mark.parametrize("bound,means,expected", [
        ("min", (60, 60, 60, 60), [0.85] * 4),
        ("min", (59, 59, 59, 59), [0] * 4),
        ("max", (40, 40, 40, 40), [0.7] * 4),
        ("max", (41, 41, 41, 41), [0] * 4),
    ])
    def test_absolute_cutoffs_can_keep_all_or_none_not_a_fixed_quota(
            self, bound, means, expected, monkeypatch):
        outlines, _ = self._record("organelle", bound, 1, monkeypatch, means=means)
        assert [o[3] for o in outlines] == pytest.approx(expected)


def test_the_retired_object_bound_animations_are_gone():
    """Asserted gone rather than quietly dropped from the parametrisation."""
    gen = pytest.importorskip("generate_setting_animations")
    from spacr.settings import RETIRED_OBJECT_BOUNDS

    retired = set(RETIRED_OBJECT_BOUNDS)
    gone = {f"{kind}_{name}" for kind in ("cell", "nucleus", "pathogen")
            for name in ("max_area", "min_intensity", "max_intensity")}
    specs = gen._specs()
    assert not retired & {key for spec in specs for key in spec.settings}
    assert not gone & {spec.slug for spec in specs}
    assert not retired & {key for item in setting_animations()
                          for key in item.settings}
    assets = TOOLS.parent / "spacr" / "resources" / "setting_animations" / "gifs"
    assert not [slug for slug in sorted(gone)
                if (assets / f"{slug}.gif").exists()]
    kept = {spec.slug: spec.settings for spec in specs}
    assert kept["cell_min_area"] == ("cell_min_size", "object_filters")
    assert kept["nucleus_min_area"] == ("nucleus_min_size",)


def test_the_five_retired_control_families_have_no_specs_or_routes():
    gen = pytest.importorskip("generate_setting_animations")
    retired = {
        f"{kind}_{suffix}" for kind in KINDS for suffix in (
            "minimum_area_to_split", "min_watershed_distance", "intensity_threshold",
            "intensity_merge", "intensity_split",
        )
    }
    specs = gen._specs()
    assert not (retired & {key for spec in specs for key in spec.settings})
    assert not (retired & {spec.slug for spec in specs})
    assert not (retired & {key for item in setting_animations() for key in item.settings})
    assets = TOOLS.parent / "spacr" / "resources" / "setting_animations" / "gifs"
    assert not [
        key for key in sorted(retired) if (assets / f"{key}.gif").exists()
    ], "retired merge/split GIFs must not remain packaged"
    assert not any(spec.scene == "split" for spec in specs)
    # Independent organelle watershed segmentation is not the retired repair.
    assert any(spec.slug == "organelle_watershed_spots" for spec in specs)


class TestTheIntensityFilterAnimationsAreGone:
    """The class here asserted that eight animations varied brightness and not
    size. Instruction 391 deleted the settings they documented -- the
    intensity-percentile band at four object roles -- so the animations went
    with them and the tests ran off the end of a generator with StopIteration.

    THE ANIMATIONS ARE ASSERTED GONE RATHER THAN QUIETLY DROPPED. Narrowing
    the parametrisation to the surviving kinds would have turned 24 failures
    into 24 silent non-tests, and nothing would then notice if a future change
    reintroduced an animation for a setting that does not exist -- which is a
    docs row pointing at nothing.

    The `_min_area` half of the old class is KEPT below: its whole purpose was
    to prove the intensity fix had not flattened the family it was
    distinguishing from, and the area filter is still here to be flattened.
    """

    RETIRED = tuple(f"{kind}_{bound}_intensity_percentile"
                    for kind in KINDS for bound in ("min", "max"))

    def test_no_animation_documents_a_removed_intensity_setting(self):
        gen = pytest.importorskip("generate_setting_animations")
        slugs = {s.slug for s in gen._specs()}
        back = sorted(slug for slug in self.RETIRED if slug in slugs)
        assert back == [], (
            f"these animations document settings 391 removed: {back}. An "
            "animation for a setting that does not exist is a docs row "
            "pointing at nothing.")

    def test_the_settings_themselves_are_really_gone(self):
        """So this file fails for the right reason if they ever come back."""
        from spacr.settings import expected_types

        alive = sorted(s for s in self.RETIRED if s in expected_types)
        assert alive == [], alive

    @pytest.mark.parametrize("kind", KINDS)
    def test_the_area_filter_still_varies_size(self, kind):
        """The fix must not flatten the family it was distinguishing from."""
        gen = pytest.importorskip("generate_setting_animations")
        spec = next(s for s in gen._specs() if s.slug == f"{kind}_min_area")
        seen = []
        real, well = gen._object_outline, gen._well
        gen._object_outline = (
            lambda p, k, c, size, amount=1.0, phase=0.0, **kw: seen.append((c, size, amount)))
        gen._well = lambda *a, **k: None
        try:
            gen._filter_scene(object(), spec, 0.0)
        finally:
            gen._object_outline, gen._well = real, well
        sizes = {s for _c, s, _a in seen}
        assert len(sizes) == 4, sizes
