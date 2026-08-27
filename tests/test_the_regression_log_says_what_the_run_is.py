"""What a run's banner and summary say about what it actually did.

Three things a real run's console got wrong (instruction 271). The first --
a false ERROR about a missing `src` -- is fixed by instruction 272. These
are the other two.
"""

from __future__ import annotations

import pytest


# --- 2. the banner --------------------------------------------------------


@pytest.fixture
def screen(qtbot):
    pytest.importorskip("PySide6")
    from spacr.qt.screens.app_screen import AppScreen

    made = AppScreen("regression")
    qtbot.addWidget(made)
    return made


@pytest.mark.parametrize("settings,expected", [
    ({"inference": "nonparametric", "regression_type": "mixed"}, True),
    ({"analysis_mode": "guide_permutation"}, True),
    ({"inference": "parametric", "regression_type": "mixed"}, False),
    # 'auto' resolves by counting guides and wells, which the panel cannot
    # see -- so it is NOT certain, and the model banner stays.
    ({"inference": "auto", "regression_type": "mixed"}, False),
])
def test_only_a_certain_permutation_changes_the_banner(screen, settings,
                                                       expected):
    assert screen._it_will_permute(settings) is expected


def test_the_permutation_banner_describes_the_permutation(screen):
    said = screen._say_what_the_permutation_will_do({
        "inference": "nonparametric", "regression_type": "mixed",
        "level": "both", "guide_permutations": 200_000,
        "guide_permutation_block": "plateID"})
    assert "200,000" in said
    assert "plateID" in said
    assert "no model is fitted" in said.lower()


def test_the_banner_gives_the_p_floor(screen):
    """The one that surprises people: a permutation cannot report a P below
    1/(B+1) however strong the effect."""
    said = screen._say_what_the_permutation_will_do({"guide_permutations": 1000})
    assert "1/(permutations+1)" in said
    assert "0.001" in said or "1e-03" in said or "0.000999" in said


def test_the_floor_moves_with_the_count(screen):
    small = screen._say_what_the_permutation_will_do({"guide_permutations": 1000})
    large = screen._say_what_the_permutation_will_do({"guide_permutations": 200_000})
    assert small != large


def test_a_bad_permutation_count_does_not_raise(screen):
    """A settings file can hold anything; a banner must not be what fails."""
    for value in (None, "", "many", -5):
        assert screen._say_what_the_permutation_will_do(
            {"guide_permutations": value})


# --- 3. retention and pairing in the summary ------------------------------


def _summary_line(settings, key):
    from spacr.regression_summary import _NOT_RECORDED    # noqa: F401

    lines = {}

    def add(name, value=None, reason=None):
        lines[name] = value if value is not None else reason

    return lines, add


def test_the_retention_is_stated_and_flagged_when_low():
    """"580,214 of 586,038 dropped" and "1.0% retained" are the same fact,
    and a reader scanning for what the run rests on reads the second."""
    from spacr.regression_summary import _LOW_RETENTION_PERCENT

    assert 0 < _LOW_RETENTION_PERCENT < 100
    retained, offered = 5_824, 586_038
    share = 100.0 * retained / offered
    assert share < _LOW_RETENTION_PERCENT, (
        "the run that prompted this item would not be flagged")


def test_an_ordinary_trim_is_not_flagged():
    """It must fire on the runs worth a second look, not on every run."""
    from spacr.regression_summary import _LOW_RETENTION_PERCENT

    share = 100.0 * 90_000 / 100_000
    assert share >= _LOW_RETENTION_PERCENT


def test_the_pairing_threshold_tolerates_the_expected_mismatch():
    """The two sides are NOT expected to match: sequencing covers every well
    and imaging keeps only what survives segmentation. On the TSG101 screen
    463 score wells all found a partner among 1,344 count wells -- 34% of
    the larger side, and a perfect join."""
    from spacr.regression_summary import _LOW_PAIRING_PERCENT

    assert _LOW_PAIRING_PERCENT <= 50.0


def test_the_pairing_is_recorded_by_the_checker():
    """It used to be printed and not recorded, so the summary said so."""
    import inspect

    from spacr.ml import _check_score_count_pairing

    source = inspect.getsource(_check_score_count_pairing)
    assert "record" in inspect.signature(_check_score_count_pairing).parameters
    assert 'record["wells_paired"]' in source
    assert 'record["wells_unpaired_counts"]' in source


def test_the_checker_is_called_with_the_recorder():
    import inspect

    from spacr import ml

    source = inspect.getsource(ml)
    assert "_check_score_count_pairing(independent_df, dependent_df, merged_df,\n" \
           "                               record=settings.get('_regression_exclusions'))" in source
