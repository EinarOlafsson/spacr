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

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import spacr.io  # noqa: E402,F401
import spacr.submodules  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figs():
    """No figure leaves a test in this module."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# _style_colour_bar
# ---------------------------------------------------------------------------

def test_a_figure_with_no_colour_bar_is_left_alone():
    """The styler runs on every heatmap figure, drawn bar or not.

    ``sns.heatmap`` adds the bar's axes only when one was asked for. Reaching
    for ``fig.axes[-1]`` on a bar-less figure would restyle the heatmap's own
    axes -- hiding its spines and recolouring its ticks -- so the guard has
    to leave that figure untouched.
    """
    from spacr.submodules import _style_colour_bar

    fig, axes = plt.subplots()
    axes.plot([0, 1], [0, 1])
    before = [spine.get_visible() for spine in axes.spines.values()]
    _style_colour_bar(fig)
    assert [spine.get_visible() for spine in axes.spines.values()] == before
    assert len(fig.axes) == 1


# ---------------------------------------------------------------------------
# _bimodality_coefficient
# ---------------------------------------------------------------------------

def test_a_coefficient_that_overflows_its_moments_is_nan_not_a_number():
    """Two populations at the ends of the double range have no coefficient.

    ``skew`` and ``kurtosis`` are ratios of central moments, and squaring a
    value near the largest finite double overflows both of them to ``NaN``.
    A coefficient built from that would be a number with no meaning, and the
    invasion assay reads it as "are there two populations here?" -- so the
    honest answer is that this sample cannot say.
    """
    from spacr.submodules import _bimodality_coefficient

    values = np.array([1e300, -1e300] * 30)
    assert np.isnan(_bimodality_coefficient(values))


def test_a_clean_two_point_mixture_still_scores_one():
    """The guard must not be swallowing the ordinary answer."""
    from spacr.submodules import _bimodality_coefficient

    values = np.array([1.0] * 30 + [5.0] * 30)
    assert _bimodality_coefficient(values) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _invasion_threshold
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["otsu", "triangle", "yen"])
def test_a_data_range_too_wide_to_histogram_yields_no_threshold(method):
    """One overflowing intensity must not take the whole well's cut with it.

    skimage builds 256 finite-sized bins across the data range; a range that
    spans the double limits cannot be binned and it raises ``ValueError``.
    Letting that escape would end the run on one bad object, so the well
    reports no threshold and the QC downstream sees a ``NaN``.
    """
    from spacr.submodules import _invasion_threshold

    values = np.array([1e308, -1e308, 0.0])
    assert np.isnan(_invasion_threshold(values, method))


def test_the_same_methods_return_a_real_cut_on_ordinary_values():
    """The refusal is about the data, not about the method being broken."""
    from spacr.submodules import _invasion_threshold

    values = np.array([1.0] * 20 + [10.0] * 20)
    for method in ("otsu", "triangle", "yen", "li", "mean"):
        cut = _invasion_threshold(values, method)
        assert 1.0 < cut < 10.0, method


# ---------------------------------------------------------------------------
# _invasion_control_mask
# ---------------------------------------------------------------------------

def _well_frame():
    return pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c12", "plate1_r2_c12"],
        "rowID": ["r1", "r1", "r2"],
        "columnID": ["c1", "c12", "c12"],
    })


def test_one_control_well_named_as_a_bare_string_selects_that_well():
    """Iterating the string would test 'r', '1', '_', 'c'... and match nothing.

    ``control_wells='r1_c12'`` is what a settings CSV carrying a single
    staining control holds. Treating it as a sequence of characters gives an
    all-False mask, and an invasion run with no control wells silently falls
    back to the per-field threshold with nothing to judge it against.
    """
    from spacr.submodules import _invasion_control_mask

    mask = _invasion_control_mask(_well_frame(), "r1_c12")
    assert mask.tolist() == [False, True, False]


def test_an_empty_control_list_selects_nothing():
    """``control_wells=[]`` is "none named", not "everything"."""
    from spacr.submodules import _invasion_control_mask

    mask = _invasion_control_mask(_well_frame(), [])
    assert not mask.any()
    assert list(mask.index) == list(_well_frame().index)


def test_a_whole_column_named_as_a_string_selects_every_row_of_it():
    """The bare-string path has to keep the whole well vocabulary working."""
    from spacr.submodules import _invasion_control_mask

    mask = _invasion_control_mask(_well_frame(), "c12")
    assert mask.tolist() == [False, True, True]


# ---------------------------------------------------------------------------
# _resolve_invasion_intensity_column: saying which column was chosen
# ---------------------------------------------------------------------------

def _intensity_frame(**columns):
    return pd.DataFrame({key: [float(v)] for key, v in columns.items()})


def test_a_named_statistic_prints_the_column_it_resolved_to(capsys):
    """The template is not visible from the settings; the resolved name is."""
    from spacr.submodules import _resolve_invasion_intensity_column

    frame = _intensity_frame(**{"pathogen_channel_2_max_intensity": 90.0})
    column, name = _resolve_invasion_intensity_column(
        frame, "pathogen", 2, "max", verbose=True)
    assert (column, name) == ("pathogen_channel_2_max_intensity", "max")
    assert "pathogen_channel_2_max_intensity" in capsys.readouterr().out


def test_a_custom_column_says_that_it_is_a_custom_column(capsys):
    """A column spaCR did not name is a choice the log has to record."""
    from spacr.submodules import _resolve_invasion_intensity_column

    frame = _intensity_frame(**{"my_own_rim_stat": 12.0})
    column, name = _resolve_invasion_intensity_column(
        frame, "pathogen", 2, "my_own_rim_stat", verbose=True)
    assert (column, name) == ("my_own_rim_stat", "custom")
    assert "custom column" in capsys.readouterr().out


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
# compare_reads_to_scores: one y column, named as a string
# ---------------------------------------------------------------------------

def test_a_single_y_column_given_as_a_string_is_checked_and_plotted(tmp_path):
    """``y_columns='class_1_fraction'`` is a documented call, not a list of one.

    The column-presence check has to accept a bare name, or it would iterate
    the string and test the characters 'c', 'l', 'a'... against the frame's
    columns, decide none of them is there, print the column list and return
    ``None`` where two figures were promised.
    """
    from tests.test_cov_submodules_reads_vs_scores import (
        EMPIRICAL, _reads_frame, _scores_frame)
    from spacr.submodules import compare_reads_to_scores

    reads = tmp_path / "reads.csv"
    scores = tmp_path / "scores.csv"
    _reads_frame().to_csv(reads, index=False)
    _scores_frame().to_csv(scores, index=False)

    figures = compare_reads_to_scores(
        str(reads), str(scores), empirical_dict=EMPIRICAL,
        y_columns="class_1_fraction", save_paths=[None, None])

    assert isinstance(figures, list) and len(figures) == 2
    lines = figures[0].axes[0].lines
    assert len(lines) == 1, "one named column is one line"


def test_a_y_column_that_is_not_in_the_frame_lists_what_is(tmp_path, capsys):
    """The refusal has to name the columns, or seaborn's ValueError is all
    the user gets."""
    from tests.test_cov_submodules_reads_vs_scores import (
        EMPIRICAL, _reads_frame, _scores_frame)
    from spacr.submodules import compare_reads_to_scores

    reads = tmp_path / "reads.csv"
    scores = tmp_path / "scores.csv"
    _reads_frame().to_csv(reads, index=False)
    _scores_frame().to_csv(scores, index=False)

    out = compare_reads_to_scores(
        str(reads), str(scores), empirical_dict=EMPIRICAL,
        y_columns="not_a_column", save_paths=[None, None])

    assert out is None
    printed = capsys.readouterr().out
    assert "columns in dataframe" in printed
    assert "class_1_fraction" in printed


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

def _uncanonicalised_reader(monkeypatch):
    """Point the module's reader at the raw-file mode of the real one."""
    import functools

    import spacr.submodules as submodules
    from spacr.tabular import read_table as real_read_table

    monkeypatch.setattr(
        submodules, "read_table",
        functools.partial(real_read_table, canonicalise=False))


def test_both_legacy_column_spellings_are_repaired_on_the_reads_table(
        tmp_path, monkeypatch):
    """``column`` and ``column_name`` both have to become ``columnID``.

    One plate's counts spelled the well column one way and the next plate's
    the other -- which is what happens when two runs of the sequencing
    pipeline, months apart, are analysed together. Repairing only one
    spelling leaves the other plate's wells unmatched, and the concatenation
    below hides that as NaNs rather than raising.
    """
    from tests.test_cov_submodules_reads_vs_scores import (
        EMPIRICAL, _reads_frame, _scores_frame)
    from spacr.submodules import compare_reads_to_scores

    _uncanonicalised_reader(monkeypatch)

    paths = []
    for index, (row_key, column_key) in enumerate(
            (("row", "column"), ("row_name", "column_name"))):
        reads = _reads_frame(with_plate=False).rename(
            columns={"rowID": row_key, "columnID": column_key})
        scores = _scores_frame(with_plate=False).rename(
            columns={"rowID": "row_name"})
        reads_path = tmp_path / f"reads{index}.csv"
        scores_path = tmp_path / f"scores{index}.csv"
        reads.to_csv(reads_path, index=False)
        scores.to_csv(scores_path, index=False)
        paths.append((str(reads_path), str(scores_path)))

    merged = []
    monkeypatch.setattr(spacr.submodules, "display", merged.append)

    figures = compare_reads_to_scores(
        [p[0] for p in paths], [p[1] for p in paths],
        empirical_dict=EMPIRICAL, save_paths=[None, None])

    assert isinstance(figures, list) and len(figures) == 2
    assert merged, "the joined table was never produced"
    joined = merged[0]
    assert sorted(joined["prc"].str.split("_").str[0].unique()) == \
        ["plate1", "plate2"], "a plate was lost to an unrepaired spelling"
    assert len(joined) == 8, "four wells per plate have to survive the join"


def test_a_scores_table_spelling_its_keys_both_ways_still_joins(tmp_path,
                                                                monkeypatch):
    """``row_name`` wins over ``row``, and ``column_name`` over ``column``.

    A file carrying both is a file that was edited by hand or exported by two
    tools. The more specific spelling is the one that names the well; letting
    the bare ``row`` win silently merged the measurements onto the wrong
    scores and produced a plausible, wrong picture.
    """
    import numpy as np

    import spacr.submodules
    from spacr.submodules import interperate_vision_model
    from tests.test_cov_submodules_vision_model import (
        _install_fake_merge, _measurement_frame, _record_forest, _settings,
    )

    _uncanonicalised_reader(monkeypatch)
    frame, grid, _signal = _measurement_frame(24, seed=3)
    _install_fake_merge(monkeypatch, frame)
    fits = []
    _record_forest(monkeypatch, fits)

    labels = np.arange(len(grid)) % 2
    scores = pd.DataFrame({
        "plateID": [g[0] for g in grid],
        "fieldID": [g[3] for g in grid],
        "object": np.arange(1, len(grid) + 1),
        "score": labels,
        "row": ["r99"] * len(grid),
        "row_name": [g[1] for g in grid],
        "column": ["c99"] * len(grid),
        "column_name": [g[2] for g in grid],
    })
    scores_path = tmp_path / "scores.csv"
    scores.to_csv(scores_path, index=False)
    src = tmp_path / "plateC"
    src.mkdir()

    out = interperate_vision_model(
        _settings(src, scores_path, top_features=3))

    assert fits, "the forest was never fitted, so nothing joined"
    fitted_x, fitted_y = fits[0]
    assert len(fitted_x) == len(grid), (
        "the specific spellings must be the ones that named the wells")
    assert fitted_y.tolist() == labels.tolist()
    assert "feature_importance" in out


# ---------------------------------------------------------------------------
# generate_score_heatmap
# ---------------------------------------------------------------------------

_HEATMAP_ROWS = [f"r{i}" for i in range(1, 7)]
_HEATMAP_COLUMN = "c3"
_HEATMAP_CONTROLS = ["sgA", "sgB"]


def _heatmap_inputs(tmp_path, *, counts_column_key="columnID",
                    extra_score_columns=None):
    """Score, CV and read-count CSVs for one plate of six wells."""
    rng = np.random.default_rng(7)

    folder = tmp_path / "models"
    for model in ("modelA", "modelB"):
        sub = folder / model
        sub.mkdir(parents=True)
        data = {
            "columnID": [_HEATMAP_COLUMN] * len(_HEATMAP_ROWS),
            "rowID": _HEATMAP_ROWS,
            "pred": rng.uniform(0, 1, len(_HEATMAP_ROWS)),
        }
        data.update(extra_score_columns or {})
        pd.DataFrame(data).to_csv(sub / "scores.csv", index=False)

    cv = tmp_path / "cv.csv"
    pd.DataFrame({
        "columnID": [_HEATMAP_COLUMN] * len(_HEATMAP_ROWS),
        "rowID": _HEATMAP_ROWS,
        "pred_cv": rng.uniform(0, 1, len(_HEATMAP_ROWS)),
    }).to_csv(cv, index=False)

    counts = tmp_path / "counts.csv"
    pd.DataFrame([
        {counts_column_key: _HEATMAP_COLUMN, "rowID": row,
         "grna_name": grna, "count": int(rng.integers(10, 500))}
        for row in _HEATMAP_ROWS for grna in _HEATMAP_CONTROLS
    ]).to_csv(counts, index=False)

    return {
        "folders": [str(folder)], "csv_name": "scores.csv",
        "data_column": "pred", "csv": str(counts), "cv_csv": str(cv),
        "data_column_cv": "pred_cv", "plateID": 1,
        "columnID": _HEATMAP_COLUMN, "control_sgrnas": _HEATMAP_CONTROLS,
        "fraction_grna": "sgA", "cmap": "viridis", "dst": None,
    }


def test_a_reads_csv_spelling_the_well_column_the_old_way_still_joins(
        tmp_path, monkeypatch):
    """``column_name`` is what older reads CSVs call the well column.

    The helper groups on ``columnID``, so a frame still carrying the legacy
    spelling has to be repaired before the filter -- or the filter raises and
    the whole heatmap is lost to a file that is perfectly readable.
    """
    from spacr.submodules import generate_score_heatmap

    _uncanonicalised_reader(monkeypatch)
    settings = _heatmap_inputs(tmp_path, counts_column_key="column_name")
    merged = generate_score_heatmap(settings)

    assert not merged.empty, "the reads never joined the scores"
    assert set(merged["prc"]) == {
        f"plate1_{row}_{_HEATMAP_COLUMN}" for row in _HEATMAP_ROWS}


def test_a_score_csv_carrying_a_row_number_column_does_not_reach_the_mae(
        tmp_path):
    """``row_num`` is an index a spreadsheet export leaves behind.

    It is not a measurement, so it must not travel into the MAE table as if
    it were one -- a per-channel error computed against a row counter is a
    number with no meaning and no way to spot it in the output.
    """
    from spacr.submodules import generate_score_heatmap

    settings = _heatmap_inputs(tmp_path, extra_score_columns={
        "row_num": list(range(len(_HEATMAP_ROWS)))})
    merged = generate_score_heatmap(settings)

    assert "row_num" not in merged.columns
    assert not merged.empty


# ---------------------------------------------------------------------------
# analyze_percent_positive: a well id the reader cannot take apart
# ---------------------------------------------------------------------------

def test_a_well_id_that_is_not_a_prc_key_stops_the_run_and_names_it(
        tmp_path, monkeypatch):
    """A short key merges to nothing, and a silent empty join is the failure.

    ``prc`` is plate_row_column and comes apart from the RIGHT. A key with
    fewer than three parts -- what a foreign importer or a hand-edited table
    produces -- used to leave that row's column filled with ``None``, which
    joined against the rename log to zero rows without a word. The refusal
    has to name the count and show an example, because the offending key is
    the only thing that tells the user which table to fix.
    """
    import spacr.io
    from spacr import schema
    from spacr.submodules import analyze_percent_positive

    frame = pd.DataFrame({
        "prc": ["plate1_r1_c1", "plate1_r1_c1", "plate1", "plate1"],
        "cell_area": [1500.0, 1500.0, 1500.0, 1500.0],
        "cell_channel_1_mean_intensity": [3000.0, 500.0, 3000.0, 500.0],
    })
    monkeypatch.setattr(
        spacr.io, "_read_and_merge_data",
        lambda locs, tables, verbose=False, nuclei_limit=None,
        pathogen_limit=None, **kwargs: (frame.copy(), []))

    src = tmp_path / "screen"
    (src / "measurements").mkdir(parents=True)

    with pytest.raises(schema.KeyParseError) as excinfo:
        analyze_percent_positive({
            "src": str(src),
            "tables": ["cell"],
            "filter_1": ["cell_area", 1000],
            "value_col": "cell_channel_1_mean_intensity",
            "threshold": 2000.0,
        })
    message = str(excinfo.value)
    assert "1 well id(s)" in message
    assert "'plate1'" in message
    assert "plate_row_column" in message


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
