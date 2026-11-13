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


def test_a_named_y_vector_does_not_overwrite_an_existing_source_column():
    """A supplied vector wins a name collision without changing source data."""
    plot_line = _nested(sm.compare_reads_to_scores, "plot_line")
    frame = pd.DataFrame({"x": [1, 2, 3], "y": [0.1, 0.2, 0.3]})
    vector = pd.Series([0.9, 0.8, 0.7], name="y")

    figure = plot_line(frame, "x", vector, None, None, None, None,
                       (6, 4), None, "deep")

    assert figure._spacr_spec["y"] == "_spacr_y_vector"
    np.testing.assert_allclose(figure._spacr_data["_spacr_y_vector"], vector)
    np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), vector)
    np.testing.assert_allclose(frame["y"], [0.1, 0.2, 0.3])
    assert "_spacr_y_vector" not in frame


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
