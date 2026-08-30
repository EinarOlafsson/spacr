"""The refusals and the seldom-taken spellings in ``spacr.submodules``.

These entry points are the ones a user runs once on a plate that took a week
to make, so the shapes they have to survive are the ones a real plate arrives
in: a scores CSV that spells its wells ``row_name`` instead of ``rowID``, a
measurements table with no area column, a plate where every well is one
condition, an uninfected well with no parasites in it at all, and a run with
``verbose`` on that has nothing to report.

Each test drives the seldom-taken branch beside the ordinary one, so a
difference in the answer -- not merely the absence of a crash -- is what is
asserted.
"""
from __future__ import annotations

import importlib.util
import os
import sqlite3
import sys
import types

import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import seaborn as sns                    # noqa: E402

# The heavy torch/cellpose chain behind spacr.submodules is paid at collection
# time rather than charged to whichever test runs first.
import spacr.io                          # noqa: E402,F401
import spacr.plot                        # noqa: E402,F401
import spacr.settings                    # noqa: E402,F401
import spacr.sp_stats                    # noqa: E402,F401
import spacr.submodules as sm            # noqa: E402


@pytest.fixture(autouse=True)
def _no_blocking_show_and_clean_figs(monkeypatch):
    """Never let a figure window open, never let figures accumulate."""
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _nested(owner, name):
    """Rebuild a closure-free nested helper of ``owner`` as a callable.

    Same technique -- and same reason -- as
    ``tests/test_cov_submodules_vision_model.py::_nested``: the helper's only
    call sites pass fixed arguments, so its other branches are product code no
    caller can reach. The helper closes over nothing, so module globals are
    enough to run it.
    """
    code = next(c for c in owner.__code__.co_consts
                if isinstance(c, types.CodeType) and c.co_name == name)
    assert code.co_freevars == (), f"{name} unexpectedly closes over state"
    return types.FunctionType(code, sm.__dict__, name)


# ---------------------------------------------------------------------------
# Importing the module at all
# ---------------------------------------------------------------------------

def test_a_half_imported_ipython_does_not_block_the_module(tmp_path,
                                                           monkeypatch):
    """``display`` is a notebook convenience and never a reason not to import.

    IPython can be mid-init in another thread, and the ImportError that comes
    back from a partially imported package would otherwise take the whole
    analysis module -- Qt GUI included -- down with it.
    """
    broken = types.ModuleType("IPython.display")   # no `display` attribute
    monkeypatch.setitem(sys.modules, "IPython.display", broken)

    # A second module object over the same source: reloading the real one
    # would rebind names other tests already hold.
    spec = importlib.util.spec_from_file_location(
        "spacr._submodules_without_ipython", sm.__file__)
    fresh = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, fresh)
    spec.loader.exec_module(fresh)

    assert fresh.display is not sm.display
    assert fresh.display("anything", key="value") is None
    assert fresh.analyze_replication is not None, "the module still imported"


# ---------------------------------------------------------------------------
# compare_reads_to_scores: the line plot's unused spellings
# ---------------------------------------------------------------------------

def _line_frame():
    return pd.DataFrame({
        "x": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "g": ["a", "a", "a", "b", "b", "b"],
    })


def test_a_named_palette_wins_over_the_house_series_colours():
    """The house palette is the default, not a lock.

    A caller who deliberately names a seaborn palette gets it; 'deep' means
    "no preference" and gets the published series colours instead, which is
    what keeps a page of panels reading as one figure.
    """
    plot_line = _nested(sm.compare_reads_to_scores, "plot_line")

    figure = plot_line(_line_frame(), "x", ["y"], None, None, None, None,
                       (6, 4), None, "muted")
    muted = sns.color_palette("muted", 100)
    assert figure.axes[0].lines[0].get_color() == pytest.approx(muted[0])

    figure = plot_line(_line_frame(), "x", ["y"], None, None, None, None,
                       (6, 4), None, "deep")
    assert (matplotlib.colors.to_hex(figure.axes[0].lines[0].get_color()).upper()
            == sm.SERIES_COLOURS[0].upper())


def test_a_group_column_draws_one_line_per_group_not_one_per_column():
    """The hue palette is trimmed to the number of groups on purpose --
    seaborn raises when it is handed a palette longer than its hue levels."""
    plot_line = _nested(sm.compare_reads_to_scores, "plot_line")

    grouped = plot_line(_line_frame(), "x", "y", "g", None, None, None,
                        (6, 4), None, "deep")
    colours = {matplotlib.colors.to_hex(line.get_color())
               for line in grouped.axes[0].lines if len(line.get_xdata())}
    assert len(colours) == 2, "the two groups were drawn in one colour"

    single = plot_line(_line_frame(), "x", "y", None, None, None, None,
                       (6, 4), None, "deep")
    solo = {matplotlib.colors.to_hex(line.get_color())
            for line in single.axes[0].lines if len(line.get_xdata())}
    assert len(solo) == 1


# ---------------------------------------------------------------------------
# _assign_vacuole_ids: a parasite with no centroid
# ---------------------------------------------------------------------------

def test_a_parasite_with_no_centroid_becomes_its_own_vacuole():
    """Non-finite centroids cannot be clustered, and joining them into one
    cluster would report a rosette that nothing measured.

    The interesting case is the host cell where NONE of the centroids can be
    read: there is no cluster to number from, so the ids have to start from
    one rather than from a maximum taken over an empty set.
    """
    frame = pd.DataFrame({
        "prcf": ["plate1_r1_c1_f1"] * 2 + ["plate1_r1_c1_f2"] * 2,
        "cell_id": [1, 1, 2, 2],
        "object_label": [1, 2, 3, 4],
        "pathogen_area": [100.0] * 4,
        "pathogen_channel_0_centroid_weighted-0": [np.nan, np.nan, 50.0, 58.0],
        "pathogen_channel_0_centroid_weighted-1": [np.nan, np.nan, 50.0, 50.0],
    })

    out, key, distance = sm._assign_vacuole_ids(frame, link_distance=20.0)
    assert key == "spatial" and distance == 20.0
    unreadable = list(out.loc[out["prcf"] == "plate1_r1_c1_f1", "vacuole_id"])
    assert len(set(unreadable)) == 2, (
        "two unplaceable parasites were counted as one vacuole")
    readable = list(out.loc[out["prcf"] == "plate1_r1_c1_f2", "vacuole_id"])
    assert len(set(readable)) == 1, "a rosette was split"


# ---------------------------------------------------------------------------
# interpret_vision_model: one legacy well spelling at a time
# ---------------------------------------------------------------------------

def _interpret_with_score_keys(tmp_path, monkeypatch, extra, name):
    """Run the legacy explainer against a scores CSV keyed by ``extra``.

    ``read_table`` renames every legacy spelling on the way in, so these
    repairs are only reachable through its documented raw-file mode -- the
    same route ``tests/test_cov_6_submodules.py`` uses for the sibling case
    where a file carries BOTH spellings.
    """
    from spacr.submodules import interperate_vision_model
    from tests.test_cov_6_submodules import _uncanonicalised_reader
    from tests.test_cov_submodules_vision_model import (
        _install_fake_merge, _measurement_frame, _record_forest, _settings)

    _uncanonicalised_reader(monkeypatch)
    frame, grid, _signal = _measurement_frame(24, seed=5)
    _install_fake_merge(monkeypatch, frame)
    fits = []
    _record_forest(monkeypatch, fits)

    labels = np.arange(len(grid)) % 2
    scores = pd.DataFrame({
        "plateID": [g[0] for g in grid],
        "fieldID": [g[3] for g in grid],
        "object": np.arange(1, len(grid) + 1),
        "score": labels,
    })
    for column, values in extra(grid).items():
        scores[column] = values
    path = tmp_path / f"{name}.csv"
    scores.to_csv(path, index=False)
    src = tmp_path / name
    src.mkdir()

    out = interperate_vision_model(_settings(src, path, top_features=3))
    assert fits, "the forest was never fitted, so nothing joined"
    return fits[0][0], labels, fits[0][1], out


def test_a_scores_table_keyed_by_row_name_and_column_still_joins(tmp_path,
                                                                 monkeypatch):
    """Only the more specific row spelling and only the bare column spelling.

    A file carrying one of each is the ordinary case for an export written by
    two different tools, and an unrepaired key merges the measurements onto
    nothing at all -- eight hundred objects becoming zero, with no error.
    """
    fitted_x, labels, fitted_y, out = _interpret_with_score_keys(
        tmp_path, monkeypatch,
        lambda grid: {"row_name": [g[1] for g in grid],
                      "column": [g[2] for g in grid]},
        "row_name_and_column")

    assert len(fitted_x) == len(labels), "the wells were never matched"
    assert fitted_y.tolist() == labels.tolist()
    assert "feature_importance" in out


def test_a_scores_table_keyed_by_row_and_column_name_still_joins(tmp_path,
                                                                 monkeypatch):
    """The other pairing, which takes the other side of both repairs."""
    fitted_x, labels, fitted_y, out = _interpret_with_score_keys(
        tmp_path, monkeypatch,
        lambda grid: {"row": [g[1] for g in grid],
                      "column_name": [g[2] for g in grid]},
        "row_and_column_name")

    assert len(fitted_x) == len(labels), "the wells were never matched"
    assert fitted_y.tolist() == labels.tolist()
    assert "feature_importance" in out


# ---------------------------------------------------------------------------
# The invasion helpers, off the pipeline
# ---------------------------------------------------------------------------

def _invasion_settings(**overrides):
    settings = sm._set_analyze_invasion_defaults({"src": "unused",
                                                  "verbose": False})
    settings.update(overrides)
    return settings


def test_a_well_with_no_parasites_at_all_keeps_its_row_and_its_zero():
    """A well that held host cells and no parasites is a result, not a gap.

    Dropping it would quietly remove the wells where the drug worked best
    from the denominator of every comparison downstream.
    """
    identity = ["plateID", "rowID", "columnID", "prc", "condition"]
    columns = identity + [
        "invasion_class", "no_host_cell", "prcf", "outside_intensity",
        "threshold", "reference_threshold", "threshold_source",
        "is_outside_low_threshold", "is_outside_high_threshold"]
    empty = pd.DataFrame({name: pd.Series(dtype="object") for name in columns})
    seed = pd.DataFrame([{"plateID": "plate1", "rowID": "r1",
                          "columnID": "c1", "prc": "plate1_r1_c1",
                          "condition": "dmso"}])
    fields = pd.DataFrame(columns=["prcf", "qc_flag_unimodal"])

    wells = sm._invasion_well_table(empty, fields, "condition",
                                    _invasion_settings(), seed_wells=seed)
    assert list(wells["prc"]) == ["plate1_r1_c1"]
    assert int(wells.iloc[0]["n_total"]) == 0
    assert np.isnan(wells.iloc[0]["invasion_efficiency"]), (
        "an efficiency was quoted for a well that scored nothing")
    assert wells.iloc[0]["threshold_source"] == "none"
    assert bool(wells.iloc[0]["qc_pass"]) is False
    assert "low_total" in wells.iloc[0]["qc_flags"]

    # The same well, with parasites in it, is counted rather than seeded.
    scored = pd.DataFrame({
        "plateID": ["plate1"] * 2, "rowID": ["r1"] * 2, "columnID": ["c1"] * 2,
        "prc": ["plate1_r1_c1"] * 2, "condition": ["dmso"] * 2,
        "prcf": ["plate1_r1_c1_f1"] * 2,
        "invasion_class": pd.Categorical(
            ["attached", "invaded"],
            categories=["attached", "invaded", "unclassified"]),
        "no_host_cell": [False, False],
        "outside_intensity": [100.0, 10.0],
        "threshold": [50.0, 50.0], "reference_threshold": [50.0, 50.0],
        "threshold_source": ["field", "field"],
        "is_outside_low_threshold": [1.0, 0.0],
        "is_outside_high_threshold": [1.0, 0.0],
    })
    counted = sm._invasion_well_table(scored, fields, "condition",
                                      _invasion_settings(), seed_wells=seed)
    assert int(counted.iloc[0]["n_total"]) == 2
    assert counted.iloc[0]["invasion_efficiency"] == pytest.approx(0.5)


def test_bars_without_a_denominator_carry_no_n_annotation():
    """A proportion without its denominator is not a result, so the caller
    that has the counts passes them and the caller that does not says so
    rather than writing an n it had to invent."""
    settings = _invasion_settings()
    parasites = pd.DataFrame({
        "prc": ["plate1_r1_c1"] * 20 + ["plate1_r1_c2"] * 20,
        "condition": ["dmso"] * 20 + ["drug"] * 20,
        "invasion_class": pd.Categorical(
            ["attached"] * 10 + ["invaded"] * 10
            + ["attached"] * 15 + ["invaded"] * 5,
            categories=["attached", "invaded", "unclassified"]),
    })

    _results, _pairwise, bare = sm._invasion_stacked_bars(
        settings, parasites, "condition", "prc", "object", "viridis",
        "no counts", denominators=None)
    assert [text.get_text() for text in bare.axes[0].texts] == []

    _results, _pairwise, annotated = sm._invasion_stacked_bars(
        settings, parasites, "condition", "prc", "object", "viridis",
        "with counts", denominators={"dmso": 20, "drug": 20})
    assert sorted(text.get_text()
                  for text in annotated.axes[0].texts) == ["n=20", "n=20"]


# ---------------------------------------------------------------------------
# analyze_endodyogeny
# ---------------------------------------------------------------------------

def test_an_object_below_the_first_volume_bin_is_dropped_and_counted():
    """An object smaller than the first doubling has no bin to fall in.

    ``analyze_endodyogeny`` filters on AREA before it bins on VOLUME, so the
    binner's own guard is only reachable by calling it -- and it has to say
    how many rows went, because a silent dropna is how a size distribution
    loses its smallest objects without anyone noticing.
    """
    bin_volumes = _nested(sm.analyze_endodyogeny, "_calculate_volume_bins")

    frame = pd.DataFrame({"pathogen_area": [100.0, 600.0, 1000.0, 1600.0]})
    binned, categories = bin_volumes(frame, "pathogen", 500, None, False)
    assert len(binned) == 3, "the sub-threshold object was binned anyway"
    assert len(categories) == 3
    assert 100.0 not in set(binned["pathogen_area"])

    kept = pd.DataFrame({"pathogen_area": [600.0, 1000.0, 1600.0]})
    binned, _categories = bin_volumes(kept, "pathogen", 500, None, False)
    assert len(binned) == 3, "an object inside the range was dropped"


def test_a_verbose_run_says_how_many_objects_fell_outside_the_bins(capsys):
    """Quiet is the default; verbose has to name the number that went."""
    bin_volumes = _nested(sm.analyze_endodyogeny, "_calculate_volume_bins")

    frame = pd.DataFrame({"pathogen_area": [100.0, 120.0, 600.0, 1600.0]})
    bin_volumes(frame.copy(), "pathogen", 500, None, True)
    assert "Dropped 2 rows outside volume bin range" in capsys.readouterr().out

    bin_volumes(frame.copy(), "pathogen", 500, None, False)
    assert "Dropped" not in capsys.readouterr().out


def test_a_tables_list_that_already_names_png_list_is_not_given_a_second(
        tmp_path):
    """``png_list`` carries the class column, so it is always read.

    Appending it to a list that already has it reads the same table twice and
    the merge fans out; the caller's own settings dict is the thing that has
    to come back with one of them.
    """
    from spacr.submodules import analyze_endodyogeny
    from tests.test_cov_submodules_endodyogeny import (AREAS_C1, AREAS_C2,
                                                       _settings,
                                                       _write_endodyogeny_db)

    root = tmp_path / "plate1"
    _write_endodyogeny_db(root, {"c1": AREAS_C1, "c2": AREAS_C2})

    asked = _settings(str(root), tables=["cell", "pathogen", "png_list"])
    out = analyze_endodyogeny(asked)
    assert asked["tables"].count("png_list") == 1, "png_list was read twice"

    implied = _settings(str(root), tables=["cell", "pathogen"])
    other = analyze_endodyogeny(implied)
    assert implied["tables"] == ["cell", "pathogen", "png_list"]
    assert len(out["data"]) == len(other["data"]), (
        "naming png_list changed how many objects were analysed")


# ---------------------------------------------------------------------------
# generate_score_heatmap
# ---------------------------------------------------------------------------

def test_a_channel_called_row_num_is_not_dropped_with_the_sort_key(tmp_path):
    """The heatmap sorts on a temporary ``row_num`` and drops it again.

    A CV column that is genuinely called ``row_num`` collides with that name,
    and the guard here is what keeps the sort key out of the returned frame
    -- while the MAE table below still has to be computed from what is left.
    """
    from spacr.submodules import generate_score_heatmap
    from tests.test_cov_submodules_heatmap_postreg import (_model_folder,
                                                           _settings,
                                                           _write_mixed_csv,
                                                           _write_scores_csv)

    folder, _truth = _model_folder(tmp_path)
    mixed = tmp_path / "mixed.csv"
    _write_mixed_csv(str(mixed), ("sgA", "sgB", "sgC"))
    cv = tmp_path / "cv.csv"
    _write_scores_csv(str(cv), seed=3, value_column="row_num")

    out = generate_score_heatmap(
        _settings(tmp_path, [str(folder)], mixed, cv, None,
                  data_column_cv="row_num"))
    assert "row_num" not in out.columns, (
        "the sort key came back as a channel")
    assert {"fraction", "prc", "modelA_pred"} <= set(out.columns)

    # The ordinary spelling keeps its column, so the guard is a name clash
    # and not a blanket drop.
    cv_named = tmp_path / "cv_named.csv"
    _write_scores_csv(str(cv_named), seed=3, value_column="pred_cv")
    ordinary = generate_score_heatmap(
        _settings(tmp_path, [str(folder)], mixed, cv_named, None))
    assert "pred_cv" in ordinary.columns


# ---------------------------------------------------------------------------
# analyze_replication
# ---------------------------------------------------------------------------

def _replication_source(tmp_path, spec, name="plate1", drop_area=False,
                        parasite_columns=None):
    """A replication ``measurements.db``, optionally without its area column."""
    from tests.test_replication_assay import write_db

    root = tmp_path / name
    src = write_db(root, spec)
    if not (drop_area or parasite_columns):
        return src
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql("SELECT * FROM pathogen", con)
        if drop_area:
            frame = frame.drop(columns=["pathogen_area"])
        for column, value in (parasite_columns or {}).items():
            frame[column] = value
        frame.to_sql("pathogen", con, index=False, if_exists="replace")
    return src


def _replication_settings(src, **overrides):
    from tests.test_replication_assay import settings_for

    return settings_for(src, **overrides)


TWO_CONDITION_SPEC = [
    {"row": "r1", "column": "c1", "cell": 1, "n": 2},
    {"row": "r1", "column": "c1", "cell": 2, "n": 4},
    {"row": "r2", "column": "c1", "cell": 1, "n": 2},
    {"row": "r1", "column": "c2", "cell": 1, "n": 8},
    {"row": "r2", "column": "c2", "cell": 1, "n": 4},
]


def test_the_assay_runs_before_the_gui_registers_its_defaults(tmp_path,
                                                              monkeypatch):
    """``spacr.settings`` owns the canonical defaults and wins where it has
    them, but the assay is API-callable before those knobs exist -- which is
    the whole reason the local gap-filler is still in the module."""
    from spacr import settings as settings_module
    from spacr.submodules import analyze_replication

    src = _replication_source(tmp_path, TWO_CONDITION_SPEC)
    with_hook = analyze_replication(_replication_settings(src))
    assert hasattr(settings_module, "set_analyze_replication_defaults")

    monkeypatch.delattr(settings_module, "set_analyze_replication_defaults")
    without_hook = analyze_replication(_replication_settings(src))

    pd.testing.assert_frame_equal(without_hook["wells"], with_hook["wells"])
    assert list(without_hook["vacuoles"]["n_parasites"]) == \
        list(with_hook["vacuoles"]["n_parasites"])


def test_a_parasite_table_with_no_area_column_still_counts_its_vacuoles(
        tmp_path):
    """Area is reported when it is there and is not required to count.

    The vacuole is the counting unit; a segmentation that recorded a
    diameter but no area must not cost the assay its whole readout, and it
    must not report a total area it never measured either.
    """
    from spacr.submodules import analyze_replication

    with_area = analyze_replication(_replication_settings(
        _replication_source(tmp_path, TWO_CONDITION_SPEC, name="withArea")))
    assert "total_parasite_area" in with_area["vacuoles"].columns

    without = analyze_replication(_replication_settings(
        _replication_source(tmp_path, TWO_CONDITION_SPEC, name="noArea",
                            drop_area=True)))
    assert "total_parasite_area" not in without["vacuoles"].columns, (
        "an area was reported for a table that carries none")
    assert sorted(without["vacuoles"]["n_parasites"]) == \
        sorted(with_area["vacuoles"]["n_parasites"])


def test_a_group_the_cell_table_does_not_carry_seeds_no_empty_wells(tmp_path):
    """Empty wells are seeded from the host cells, and only when the cells
    can be labelled with the same grouping the vacuoles were grouped by.

    Grouping on a per-parasite annotation the cell table never had would
    otherwise reach `dropna(subset=[...])` with a column that is not there.
    """
    from spacr.submodules import analyze_replication

    src = _replication_source(
        tmp_path, TWO_CONDITION_SPEC, name="strain",
        parasite_columns={"strain": "RH"})
    out = analyze_replication(_replication_settings(src,
                                                    group_column="strain"))
    assert set(out["wells"]["strain"]) == {"RH"}
    assert len(out["wells"]) == 4, "a well was invented or lost"

    # The default grouping IS carried by the cell table, and an uninfected
    # well then appears with a zero denominator.
    seeded = analyze_replication(_replication_settings(
        _replication_source(tmp_path, TWO_CONDITION_SPEC, name="seeded")))
    assert "condition" in seeded["wells"].columns


def test_a_verbose_run_only_names_the_wells_that_are_actually_flagged(
        tmp_path, capsys):
    """The QC line is a list of wells, and an empty list is not news."""
    from spacr.submodules import analyze_replication

    clean = _replication_source(tmp_path, TWO_CONDITION_SPEC, name="clean")
    analyze_replication(_replication_settings(clean, verbose=True))
    printed = capsys.readouterr().out
    assert "QC:" not in printed, "a clean plate was reported as flagged"

    odd = _replication_source(tmp_path, name="odd", spec=[
        {"row": "r1", "column": "c1", "cell": 1, "n": 3},
        {"row": "r1", "column": "c2", "cell": 1, "n": 4},
    ])
    analyze_replication(_replication_settings(odd, verbose=True))
    flagged = capsys.readouterr().out
    assert "QC:" in flagged and "plate1_r1_c1" in flagged


# ---------------------------------------------------------------------------
# analyze_invasion
# ---------------------------------------------------------------------------

def _invasion_source(tmp_path, fields, name="p", drop_area=False,
                     parasite_columns=None):
    """An invasion ``measurements.db``, optionally reshaped after writing."""
    from tests.test_invasion_assay import write_db

    src = write_db(tmp_path / name, fields)
    if not (drop_area or parasite_columns):
        return src
    db = os.path.join(src, "measurements", "measurements.db")
    with sqlite3.connect(db) as con:
        frame = pd.read_sql("SELECT * FROM pathogen", con)
        if drop_area:
            frame = frame.drop(columns=["pathogen_area"])
        for column, value in (parasite_columns or {}).items():
            frame[column] = value
        frame.to_sql("pathogen", con, index=False, if_exists="replace")
    return src


def _clean_plate():
    from tests.test_invasion_assay import split

    return [{"row": "r1", "column": "c1", "outside": split(36, 24)},
            {"row": "r1", "column": "c2", "outside": split(24, 36)}]


def test_the_invasion_assay_runs_before_the_gui_registers_its_defaults(
        tmp_path, monkeypatch):
    """The same gap-filler contract as the replication assay, and the same
    reason: the API has to work before the GUI knobs are registered."""
    from spacr import settings as settings_module
    from spacr.submodules import analyze_invasion
    from tests.test_invasion_assay import settings_for

    src = _invasion_source(tmp_path, _clean_plate())
    with_hook = analyze_invasion(settings_for(src))
    assert hasattr(settings_module, "set_analyze_invasion_defaults")

    monkeypatch.delattr(settings_module, "set_analyze_invasion_defaults")
    without_hook = analyze_invasion(settings_for(src))

    pd.testing.assert_frame_equal(without_hook["wells"], with_hook["wells"])


def test_a_parasite_table_with_no_area_column_is_still_classified(tmp_path):
    """The area filters are optional; the outside stain is the readout.

    A table with no area column must not lose its classification, and it must
    not be silently filtered to nothing by a comparison against a column that
    is not there.
    """
    from spacr.submodules import analyze_invasion
    from tests.test_invasion_assay import settings_for, well

    with_area = analyze_invasion(settings_for(
        _invasion_source(tmp_path, _clean_plate(), name="withArea")))
    without = analyze_invasion(settings_for(
        _invasion_source(tmp_path, _clean_plate(), name="noArea",
                         drop_area=True)))

    assert "pathogen_area" in with_area["parasites"].columns
    assert "pathogen_area" not in without["parasites"].columns
    for prc in ("plate1_r1_c1", "plate1_r1_c2"):
        assert (well(without, prc)["invasion_efficiency"]
                == pytest.approx(well(with_area, prc)["invasion_efficiency"]))
        assert well(without, prc)["n_total"] == well(with_area, prc)["n_total"]


def test_a_plate_nothing_could_be_classified_on_still_reports_both_classes(
        tmp_path):
    """One population cannot be split, and the per-field table has to say so
    with a zero rather than with a missing column.

    A missing ``n_attached`` would make the merge below drop the class
    entirely, and every downstream efficiency would then be computed over a
    denominator that silently lost half its definition.
    """
    from spacr.submodules import analyze_invasion
    from tests.test_invasion_assay import settings_for

    flat = analyze_invasion(settings_for(_invasion_source(tmp_path, [
        {"row": "r1", "column": "c1", "outside": [10.0] * 40},
        {"row": "r1", "column": "c2", "outside": [10.0] * 40},
    ], name="flat")))

    fields = flat["fields"]
    assert set(flat["parasites"]["invasion_class"]) == {"unclassified"}
    assert list(fields["n_attached"]) == [0, 0]
    assert list(fields["n_invaded"]) == [0, 0]
    assert list(fields["n_total"]) == [0, 0]

    split_plate = analyze_invasion(settings_for(
        _invasion_source(tmp_path, _clean_plate(), name="split")))
    assert sorted(split_plate["fields"]["n_attached"]) == [24, 36]


def test_a_group_the_cell_table_does_not_carry_seeds_no_invasion_wells(
        tmp_path):
    """Wells with host cells and no parasites are seeded from the cell table,
    and only when the cells can be labelled the way the parasites were."""
    from spacr.submodules import analyze_invasion
    from tests.test_invasion_assay import settings_for

    out = analyze_invasion(settings_for(
        _invasion_source(tmp_path, _clean_plate(), name="strain",
                         parasite_columns={"strain": "RH"}),
        group_column="strain"))
    assert set(out["wells"]["strain"]) == {"RH"}
    assert len(out["wells"]) == 2, "a well was invented or lost"

    seeded = analyze_invasion(settings_for(
        _invasion_source(tmp_path, _clean_plate(), name="seeded")))
    assert "condition" in seeded["wells"].columns


def test_a_verbose_invasion_run_only_lists_wells_that_failed_qc(tmp_path,
                                                                capsys):
    """The per-field thresholds are always printed under verbose; the QC list
    is printed only when there is one, because an empty table under a "QC:"
    heading reads as a plate nobody checked."""
    from spacr.submodules import analyze_invasion
    from tests.test_invasion_assay import settings_for, split

    clean = analyze_invasion(settings_for(
        _invasion_source(tmp_path, _clean_plate(), name="cleanv"),
        verbose=True))
    printed = capsys.readouterr().out
    assert "Per-field thresholds:" in printed
    assert "QC:" not in printed, "a clean plate was reported as flagged"
    assert all(clean["wells"]["qc_pass"])

    analyze_invasion(settings_for(_invasion_source(tmp_path, [
        {"row": "r1", "column": "c1", "outside": split(6, 4)},
        {"row": "r1", "column": "c2", "outside": split(4, 6)},
    ], name="thinv"), verbose=True))
    flagged = capsys.readouterr().out
    assert "QC:" in flagged and "low_total" in flagged
