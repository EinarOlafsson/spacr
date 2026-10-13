"""The measured layout policy: read it, or work without it (359).

The artifact is generated -- `tools/measure_the_layout_matrix.py` -- and
this is its reader. Everything here is about the two ways a reader can be
wrong: believing a file it should not, and falling over when there is none.

NOTHING HERE NEEDS A DISPLAY, and that is a requirement of the item rather
than a convenience: the policy must be "loaded without importing heavy
scientific or GPU libraries", so the Qt metrics arrive as arguments and
the module under test imports `json` and `importlib.resources` and nothing
else.
"""

import json

import pytest

from spacr.qt import _layout_policy as lp


@pytest.fixture
def measured():
    """A small artifact with two scales measured."""
    return {
        "schema": 1,
        "matrix": {"apps": ["measure", "regression"], "locales": ["en", "de"],
                   "scales": [1.0, 2.0], "widths": [1000, 2100],
                   "height": 850},
        "minimum_width": {"1.0": 1400, "2.0": 2100},
        "rows": {},
    }


class TestItRefusesWhatItCannotRead:

    def test_a_policy_from_the_future_is_not_read(self, monkeypatch, tmp_path):
        """A schema nobody here understands is worse than no schema.

        Reading it would apply rules that had not been written yet, and
        the failure would be a window size nobody can explain.
        """
        later = {"schema": lp.SUPPORTED_SCHEMA + 1,
                 "minimum_width": {"1.0": 9999}}
        assert lp.minimum_width_for(1.0, later) == 9999, (
            "an explicit policy argument is used as given -- the schema "
            "guard belongs to the READER of the bundled file")
        _bundle(monkeypatch, tmp_path, later)
        assert lp.read_policy(refresh=True) == {}

    @pytest.mark.parametrize("junk", ["[]", "null", "not json at all", ""])
    def test_a_file_that_is_not_a_policy_reads_as_none(self, monkeypatch,
                                                       tmp_path, junk):
        _bundle(monkeypatch, tmp_path, junk, raw=True)
        assert lp.read_policy(refresh=True) == {}

    def test_a_missing_file_is_not_an_error(self, monkeypatch):
        def _explode(*_a, **_k):
            raise FileNotFoundError("no artifact in this wheel")

        monkeypatch.setattr("importlib.resources.files", _explode)
        assert lp.read_policy(refresh=True) == {}
        # And the caller still gets a window.
        assert lp.recommended_window_size((1920, 1080)) == lp.FALLBACK


class TestTheWidthItRecommends:

    def test_it_takes_the_measured_scale(self, measured):
        assert lp.minimum_width_for(1.0, measured) == 1400
        assert lp.minimum_width_for(2.0, measured) == 2100

    def test_between_two_scales_it_rounds_up(self, measured):
        """Rounding down opens a window that fits the smaller scale."""
        assert lp.minimum_width_for(1.5, measured) == 2100

    def test_above_every_measured_scale_it_takes_the_widest(self, measured):
        assert lp.minimum_width_for(4.0, measured) == 2100

    def test_below_every_measured_scale_it_takes_the_smallest(self, measured):
        assert lp.minimum_width_for(0.5, measured) == 1400

    @pytest.mark.parametrize("bad", ["wide", None, object()])
    def test_a_scale_that_is_not_a_number_answers_nothing(self, measured, bad):
        assert lp.minimum_width_for(bad, measured) is None


class TestItNeverOpensAWindowOffTheScreen:

    def test_a_requirement_wider_than_the_display_is_clamped(self, measured):
        """The clamp is the contract, and the ordering is the whole of it.

        A measured requirement wider than the display says this module
        cannot be shown whole here. It is not permission to open a window
        past the edge -- the user can scroll a narrow window, and cannot
        reach the corner of one whose corner is off-screen.
        """
        width, _height = lp.recommended_window_size((1280, 720), 2.0, measured)
        assert width == 1280

    def test_it_uses_the_full_width_when_the_screen_has_it(self, measured):
        width, height = lp.recommended_window_size((3840, 2160), 1.0, measured)
        assert width == 1400
        assert height == 850

    @pytest.mark.parametrize("available", [(0, 0), (-1, 900), "not a size",
                                           None, (1920,)])
    def test_a_geometry_it_cannot_read_falls_back(self, available, measured):
        assert lp.recommended_window_size(available, 1.0, measured) == \
            lp.FALLBACK

    def test_the_floor_is_a_window_a_user_can_still_use(self, measured):
        """A 200x100 display is not a reason to open a 200 px window."""
        assert lp.recommended_window_size((200, 100), 1.0, measured) == \
            (640, 480)


class TestItSaysWhy:

    def test_the_sentence_names_the_evidence(self, measured):
        said = lp.why(1.0, measured)
        assert "1400" in said
        assert "2 modules" in said
        assert "2 locales" in said

    def test_without_a_policy_it_says_that_instead(self):
        said = lp.why(1.0, {})
        assert "no measured layout policy" in said


def _bundle(monkeypatch, tmp_path, content, raw: bool = False):
    """Point the reader at `content` instead of the packaged artifact."""
    path = tmp_path / "layout_policy.json"
    path.write_text(content if raw else json.dumps(content),
                    encoding="utf-8")

    class _Anchor:
        def __truediv__(self, _name):
            return path

    monkeypatch.setattr("importlib.resources.files", lambda _pkg: _Anchor())
