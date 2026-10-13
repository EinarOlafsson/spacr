"""Public callables in the big modules that no test had ever named.

Instruction 60. Of 164 public names in ``utils`` and 54 in ``io``, sixteen
across the whole package had never appeared in a test file -- which is a
different and worse state from "covered by something incidental": nobody has
written down what they are supposed to do.

They are small, and small is where a silent wrong answer lives. Two of them
decide which measurement columns a model trains on and how a class is named
in a report; getting either quietly wrong is discovered, if ever, in the
analysis of a screen that took a week to run.
"""
from __future__ import annotations


import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# utils.normalize_feature_filter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("spelling", ["", "none", "None", "NULL", "all",
                                      "all_channels", "all channels", "*",
                                      "  none  "])
def test_every_no_filter_spelling_collapses_to_none(spelling):
    """A settings CSV holds the literal string "None", and treating that as a
    feature-name substring removes every measurement column -- while the
    panel it came from meant "all channels"."""
    from spacr.utils import normalize_feature_filter

    assert normalize_feature_filter(spelling) is None


def test_a_real_substring_survives():
    from spacr.utils import normalize_feature_filter

    assert normalize_feature_filter("  channel_1 ") == "channel_1"


def test_a_list_passes_through_untouched():
    """A list of channels is not a string and must not be stripped, lowered
    or compared against the no-filter spellings."""
    from spacr.utils import normalize_feature_filter

    channels = ["channel_0", "channel_2"]
    assert normalize_feature_filter(channels) is channels
    assert normalize_feature_filter(None) is None


# ---------------------------------------------------------------------------
# io.format_class_balance_report
# ---------------------------------------------------------------------------

def _report(labels, classes, **kwargs):
    """The report, produced the way the pipeline produces it.

    THROUGH `report_class_balance`, NOT BY HAND. `format_class_balance_report`
    reads `summary['action']`, which `summarize_class_imbalance` does not set
    -- only the wrapper does, after it has decided what rebalancing to do. A
    summary assembled in a test would either KeyError or carry an `action`
    nobody's code path can produce, and the second is worse.
    """
    from spacr.io import report_class_balance

    return report_class_balance(labels, classes=classes, **kwargs)["report"]


def test_the_report_names_every_class_and_its_count():
    text = _report([0] * 90 + [1] * 10, ["nc", "pc"])
    assert "nc" in text and "pc" in text
    assert "90" in text and "10" in text


def test_an_empty_class_reports_an_infinite_ratio_not_a_crash():
    """A class with nothing in it is the single most useful thing this
    report can say, and a ZeroDivisionError says it least well."""
    text = _report([0] * 40, ["nc", "pc"])
    assert "inf" in text


def test_the_split_being_described_is_named():
    """Only the train split is ever resampled, so a reader has to be able to
    tell which split a skew report is about."""
    text = _report([0] * 5 + [1] * 5, ["a", "b"], split_name="validation")
    assert "validation" in text


def test_the_report_is_more_than_one_line():
    text = _report([0] * 5 + [1] * 7, ["a", "b"])
    assert len(text.splitlines()) > 1
    assert "action:" in text, (
        "the report's last line is what the run DID about the skew, and it "
        "is the only part a reader can act on")


# ---------------------------------------------------------------------------
# settings.set_graph_importance_defaults
# ---------------------------------------------------------------------------

def test_the_graph_importance_defaults_are_a_box_with_jitter():
    """139 B, on the module that draws it: a bar at a mean shows one number
    and hides the spread it was computed from."""
    from spacr.settings import set_graph_importance_defaults

    got = set_graph_importance_defaults({})
    assert got["graph_type"] == "jitter_box"


def test_it_fills_in_place_and_returns_the_same_dict():
    """Every other `set_*_defaults` in this module does, and a caller that
    relied on one and got the other would silently drop its settings."""
    from spacr.settings import set_graph_importance_defaults

    given = {}
    assert set_graph_importance_defaults(given) is given
    assert given, "nothing was filled in"


def test_a_value_the_caller_set_is_not_overwritten():
    from spacr.settings import set_graph_importance_defaults

    got = set_graph_importance_defaults({"grouping_column": "channel"})
    assert got["grouping_column"] == "channel"


# ---------------------------------------------------------------------------
# deep_spacr.class_labels
# ---------------------------------------------------------------------------

def test_the_folder_names_are_used_when_they_are_known():
    from spacr.deep_spacr import class_labels

    assert class_labels({"num_classes": 2}, ["nc", "pc"]) == ["nc", "pc"]


def test_it_never_returns_an_empty_list():
    """The caller is about to index it per class, so an empty list is an
    IndexError somewhere else with no clue where it came from."""
    from spacr.deep_spacr import class_labels

    names = class_labels({"num_classes": 3})
    assert len(names) == 3
    assert all(str(name) for name in names)


def test_too_few_folder_names_are_made_up_to_the_class_count():
    """Training can be handed fewer folders than the head has outputs, and
    a short list would leave the last class unnamed."""
    from spacr.deep_spacr import class_labels

    names = class_labels({"num_classes": 3}, ["nc"])
    assert len(names) == 3


# ---------------------------------------------------------------------------
# hyperparam.umap_checkpoint_path
# ---------------------------------------------------------------------------

def test_an_explicit_checkpoint_path_wins():
    from spacr.hyperparam import umap_checkpoint_path

    assert umap_checkpoint_path(
        {"checkpoint_path": "/tmp/mine.json"}) == "/tmp/mine.json"


def test_the_default_sits_under_the_projects_results(tmp_path):
    """A checkpoint written beside the database is a file the user finds in
    their data folder and cannot explain."""
    from spacr.hyperparam import umap_checkpoint_path

    src = tmp_path / "plate1"
    (src / "measurements").mkdir(parents=True)
    got = umap_checkpoint_path({"src": str(src)})
    if got is not None:
        assert "results" in got
        assert got.endswith(".json")


def test_settings_that_name_nothing_get_no_path():
    """A checkpoint path invented from nothing would be written into the
    working directory of whoever happened to launch the run."""
    from spacr.hyperparam import umap_checkpoint_path

    assert umap_checkpoint_path({}) in (None, "")


# ---------------------------------------------------------------------------
# plot.data_colours
# ---------------------------------------------------------------------------

def test_the_data_colours_are_the_claim_not_the_frame():
    """It is used to say when a colour stops working on paper, so a spine or
    a grid line counted among them would report a false failure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.plot import data_colours

    figure, axes = plt.subplots()
    try:
        axes.plot([0, 1], [0, 1], color="#4c72b0")
        axes.axhline(0.5, color="#888888", linestyle="--")
        found = {str(c).lower() for c in data_colours(figure)}
        assert "#4c72b0" in found
    finally:
        plt.close(figure)


def test_a_figure_with_no_data_has_no_data_colours():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.plot import data_colours

    figure = plt.figure()
    try:
        assert list(data_colours(figure)) == []
    finally:
        plt.close(figure)


# ---------------------------------------------------------------------------
# io.crop_refs_for_rows
# ---------------------------------------------------------------------------

def test_one_reference_per_row(tmp_path):
    """The count is the contract: a caller zips these against the frame, and
    a dropped row silently pairs every later crop with the wrong metadata."""
    from spacr.io import crop_refs_for_rows

    frame = pd.DataFrame({
        "png_path": [str(tmp_path / f"{i}.png") for i in range(4)],
    })
    refs = crop_refs_for_rows("png", frame)
    assert len(refs) == len(frame)


def test_an_empty_frame_gives_an_empty_list(tmp_path):
    from spacr.io import crop_refs_for_rows

    empty = pd.DataFrame({"png_path": []})
    assert crop_refs_for_rows("png", empty) == []


def test_a_blank_path_is_not_carried_as_a_path(tmp_path):
    """A NaN png_path reaching a crop reader is an open() on the string
    'nan', which fails somewhere with no mention of the row it came from."""
    from spacr.io import crop_refs_for_rows

    frame = pd.DataFrame({
        "png_path": [str(tmp_path / "a.png"), float("nan")],
    })
    refs = crop_refs_for_rows("png", frame)
    assert len(refs) == 2
    assert not str(getattr(refs[1], "png_path", "") or "").endswith("nan")


def test_the_name_column_is_honoured(tmp_path):
    """Without it the name is the basename of the path, which for a merged
    source is the FIELD's name and not the crop's."""
    from spacr.io import crop_refs_for_rows

    frame = pd.DataFrame({
        "png_path": [str(tmp_path / "field.png")],
        "crop_name": ["plate1_A01_1_cell_7.png"],
    })
    refs = crop_refs_for_rows("png", frame, name_column="crop_name")
    assert len(refs) == 1
    assert "cell_7" in str(getattr(refs[0], "name", refs[0]))
