"""The corners of the analysis submodules: refusals, absences and narration.

Four kinds of branch are pinned here, and each is a place where an assay can
report a confident number instead of admitting it has none:

* a statistic that cannot be computed -- a bimodality coefficient whose
  denominator overflows, a threshold method that refuses the data range --
  must come back ``NaN``, never a value;
* a well id the reader cannot parse must stop the run, because the join it
  feeds merges to nothing silently;
* a control-well selector given one well as a bare string must select that
  well, not each of its characters;
* ``verbose`` must say which column, which key and which comparison were
  used, since that is the only record of a choice the caller did not make.

Everything runs on tiny synthetic frames: CPU-only, offline, sub-second.
"""

from __future__ import annotations

import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io  # noqa: E402,F401


@pytest.fixture(autouse=True)
def _close_figs():
    """No figure leaves a test in this module."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _resolve_invasion_intensity_column: saying which column was chosen
# ---------------------------------------------------------------------------

def _intensity_frame(**columns):
    return pd.DataFrame({key: [float(v)] for key, v in columns.items()})


@pytest.mark.parametrize("column, value, stat, name, printed", [
    # The template is not visible from the settings; the resolved name is.
    ("pathogen_channel_2_max_intensity", 90.0, "max", "max",
     "pathogen_channel_2_max_intensity"),
    # A column spaCR did not name is a choice the log has to record.
    ("my_own_rim_stat", 12.0, "my_own_rim_stat", "custom", "custom column"),
], ids=["named-statistic", "custom-column"])
def test_the_resolved_intensity_column_is_printed(capsys, column, value, stat,
                                                  name, printed):
    from spacr.submodules import _resolve_invasion_intensity_column

    frame = _intensity_frame(**{column: value})
    resolved, label = _resolve_invasion_intensity_column(
        frame, "pathogen", 2, stat, verbose=True)
    assert (resolved, label) == (column, name)
    assert printed in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _ensure_field_key
# ---------------------------------------------------------------------------

def test_building_a_field_key_says_what_it_was_built_from(capsys):
    """``prcf`` is the unit of observation; how it was composed is the record."""
    from spacr.submodules import _ensure_field_key

    frame = pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r1"],
        "columnID": ["c1"], "fieldID": ["f1"],
    })
    out = _ensure_field_key(frame, source="the parasite table", verbose=True)
    assert out["prcf"].tolist() == ["plate1_r1_c1_f1"]
    assert "plate_row_column_field" in capsys.readouterr().out


def test_a_timelapse_field_key_names_the_time_column_it_used(capsys):
    """On a timelapse the key must name one FRAME, not one stack."""
    from spacr.submodules import _ensure_field_key

    frame = pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r1"],
        "columnID": ["c1"], "fieldID": ["f1"], "timeID": ["t3"],
    })
    out = _ensure_field_key(frame, source="the parasite table", verbose=True)
    assert out["prcf"].tolist() == ["plate1_r1_c1_f1_t3"]
    assert "plate_row_column_field_timeID" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# _invasion_compare_conditions
# ---------------------------------------------------------------------------

def _wells_frame():
    """Six wells over two conditions, three each, with a known efficiency."""
    records = []
    for group, efficiencies in (("dmso", (0.20, 0.25, 0.30)),
                                ("drug", (0.70, 0.75, 0.80))):
        for index, efficiency in enumerate(efficiencies):
            total = 100
            invaded = int(round(efficiency * total))
            records.append({
                "prc": f"plate1_r{index + 1}_c{1 if group == 'dmso' else 2}",
                "condition": group,
                "invasion_efficiency": efficiency,
                "n_invaded": invaded,
                "n_attached": total - invaded,
                "n_total": total,
                "qc_pass": True,
            })
    return pd.DataFrame(records)


def test_the_comparison_table_is_printed_when_asked_for(capsys):
    """The unit of replication is the well, and the printout has to say so.

    A reader who sees only a p value assumes the parasites were the sample.
    The heading names the well as the unit, and the table beside it carries
    the well counts that make the number checkable.
    """
    from spacr.submodules import _invasion_compare_conditions

    results = _invasion_compare_conditions(
        _wells_frame(), "condition", verbose=True)
    printed = capsys.readouterr().out
    assert "unit of replication: well" in printed
    assert "dmso" in printed and "drug" in printed
    assert results["n_wells_1"].tolist() == [3]


def test_the_comparison_is_silent_by_default(capsys):
    """A helper that prints unasked makes every caller's output unreadable."""
    from spacr.submodules import _invasion_compare_conditions

    _invasion_compare_conditions(_wells_frame(), "condition")
    assert capsys.readouterr().out == ""


# ---------------------------------------------------------------------------
# Legacy row/column spellings, on a frame that arrived uncanonicalised
# ---------------------------------------------------------------------------
#
# ``tabular.read_table`` renames every legacy spelling on the way in, so a
# frame reaching these helpers normally carries ``rowID``/``columnID`` already.
# The repairs below therefore cannot be reached through the ordinary reader --
# which is exactly why they need a test: they are what a caller reading a file
# EXACTLY AS WRITTEN (``read_table(..., canonicalise=False)``, the documented
# mode for inspecting a file) still depends on, and a silent break there ends
# in a merge that produces zero rows and no error.


# ---------------------------------------------------------------------------
# The invasion figures
# ---------------------------------------------------------------------------

def _classified_parasites():
    """Per-parasite classes over two conditions, with an outside signal."""
    from spacr.submodules import _INVASION_CLASSES

    records = []
    for group, invaded in (("dmso", 3), ("drug", 7)):
        for index in range(10):
            records.append({
                "prc": f"plate1_r1_c{1 if group == 'dmso' else 2}",
                "prcf": f"plate1_r1_c{1 if group == 'dmso' else 2}_f1",
                "condition": group,
                "outside_intensity": 100.0 + index,
                "invasion_class": ("invaded" if index < invaded
                                   else "attached"),
            })
    frame = pd.DataFrame(records)
    frame["invasion_class"] = pd.Categorical(
        frame["invasion_class"], categories=list(_INVASION_CLASSES))
    return frame


def test_a_bar_with_no_recorded_denominator_is_left_unannotated():
    """A proportion without its n is not a result, and a wrong n is worse.

    The denominators are keyed by the bar's own label. A group missing from
    that mapping -- a condition renamed between the count and the plot -- must
    get no annotation rather than the next group's count written above it.
    """
    from spacr.submodules import _invasion_stacked_bars

    parasites = _classified_parasites()
    results, pairwise, fig = _invasion_stacked_bars(
        {"verbose": False}, parasites, "condition", "prc", "well", "viridis",
        "Invasion", denominators={"dmso": 10})

    annotated = [text.get_text() for text in fig.axes[0].texts
                 if text.get_text().startswith("n=")]
    assert annotated == ["n=10"], "only the group with a denominator is named"
    assert results is not None and pairwise is not None


def test_a_well_listed_twice_is_panelled_from_its_first_row():
    """A well table with a repeated key gives ``.loc`` a frame, not a row.

    Reading ``row['threshold_median']`` off a two-row frame yields a Series,
    and ``float()`` of that raises inside the drawing. The panel takes the
    first row so the figure still says which threshold was applied.
    """
    from spacr.submodules import _invasion_threshold_panels

    parasites = _classified_parasites()
    wells = pd.DataFrame([
        {"prc": "plate1_r1_c1", "threshold_median": 105.0,
         "reference_threshold_median": 103.0, "threshold_source": "otsu",
         "bimodality_coefficient": 0.7, "n_total": 10},
        {"prc": "plate1_r1_c1", "threshold_median": 106.0,
         "reference_threshold_median": 103.0, "threshold_source": "otsu",
         "bimodality_coefficient": 0.7, "n_total": 10},
    ])
    fig = _invasion_threshold_panels(parasites, wells)
    axis = fig.axes[0]
    assert "plate1_r1_c1" in axis.get_title()
    applied = [line.get_xdata()[0] for line in axis.lines
               if line.get_label().startswith("threshold")]
    assert applied == [105.0], "the second row's threshold was drawn instead"
