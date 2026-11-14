"""Table-driven tests of drawing, saving and joining in ``gene_measurement_compare``.

The comparison panel has no console, so every unusable input -- a measurement
that is NaN on every cell, a group with one observation, a filter naming an
absent group, a figure format that will not write, a crop that will not
encode, a database that will not open -- must neither raise nor pass silently.
Drawing keeps every group at the position its tick label claims, saving always
leaves the record a reader checks a figure against, and a join reports every
file that contributed nothing.  One general test per unit, each walking a case
table with ``subtests`` and asserting exact values.
"""
from __future__ import annotations

import json
import os

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from spacr import gene_measurement_compare as gmc  # noqa: E402
from spacr.gene_measurement_compare import REST, Comparison  # noqa: E402


def _comparison(values_by_group, *, level="cell", statistics=None):
    """A comparison built by hand, so frames ``build`` never makes can be drawn."""
    rows = [{"group": group, "value": value, "unit": f"{group}-{index}"}
            for group, values in values_by_group.items()
            for index, value in enumerate(values)]
    return Comparison(measurement="pathogen_area", level=level,
                      frame=pd.DataFrame(rows,
                                         columns=["group", "value", "unit"]),
                      statistics=list(statistics or []))


def _empty(level="well"):
    return Comparison(measurement="pathogen_area", level=level,
                      frame=pd.DataFrame(columns=["group", "value", "unit"]))


def _labels(axes):
    return [t.get_text() for t in axes.get_xticklabels()]


@pytest.fixture()
def objects():
    """Twelve cells over four wells of one plate, with two measurements."""
    n = 12
    return pd.DataFrame({
        "pathogen_area": np.arange(n, dtype=float) + 1.0,
        "cell_area": np.arange(n, dtype=float) + 20.0,
        "plateID": ["p1"] * n,
        "rowID": ["r1"] * 6 + ["r2"] * 6,
        "columnID": ["c1", "c2", "c3"] * 4,
    })


@pytest.fixture()
def built(objects):
    return gmc.build(objects, "pathogen_area", level="cell",
                     groups={"g": objects.index[:6]})


ALL_NAN = {"g": [np.nan, np.inf], REST: [np.nan]}
NAN_GROUP = {"g": [np.nan, np.nan], REST: [1.0, 2.0, 3.0, 4.0]}
ONE_POINT_EACH = {"g": [3.0], REST: [5.0]}
SPREAD = {"g": [3.0, 4.0, 9.0], REST: [5.0, 6.0, 11.0]}
MIXED = {"g": [1.0], "h": [2.0, 3.0, 4.0], REST: [5.0, 6.0, 7.0]}


def test_plot_draws_every_group_at_its_labelled_position(subtests, tmp_path):
    """Nothing finite draws nothing; thin groups keep their tick and count."""
    for case_id, comparison in [("all_values_non_finite", _comparison(ALL_NAN)),
                                ("no_rows", _empty())]:
        with subtests.test(msg=case_id):
            assert gmc.plot(comparison) is None

    cases = [
        # (case id, values, kind, labels, n collections or None to skip)
        ("jitter_skips_empty_group", NAN_GROUP, "jitter",
         ["g\n(n=0)", f"{REST}\n(n=4)"], 1),
        ("jitter_box_skips_empty_group", NAN_GROUP, "jitter_box",
         ["g\n(n=0)", f"{REST}\n(n=4)"], 1),
        ("violin_of_single_points_draws_no_body", ONE_POINT_EACH, "violin",
         ["g\n(n=1)", f"{REST}\n(n=1)"], 0),
        ("violin_with_spread_draws_two_bodies", SPREAD, "violin",
         ["g\n(n=3)", f"{REST}\n(n=3)"], 2),
        ("violin_skips_only_the_thin_group", MIXED, "violin",
         ["g\n(n=1)", "h\n(n=3)", f"{REST}\n(n=3)"], 2),
    ]
    for case_id, values, kind, labels, n_collections in cases:
        with subtests.test(msg=case_id):
            figure = gmc.plot(_comparison(values), kind=kind)
            try:
                axes = figure.axes[0]
                assert _labels(axes) == labels
                assert list(axes.get_xticks()) == list(range(len(labels)))
                assert axes.get_ylabel() == "pathogen_area"
                assert axes.get_title() == "cell level"
                assert len(axes.collections) == n_collections
                if case_id == "violin_skips_only_the_thin_group":
                    centres = sorted(
                        round(float(np.mean(body.get_paths()[0]
                                            .vertices[:, 0])))
                        for body in axes.collections)
                    assert centres == [1, 2]
            finally:
                plt.close(figure)

    with subtests.test(msg="small_bar_warns_on_the_figure"):
        figure = gmc.plot(_comparison(SPREAD), kind="bar")
        try:
            assert [t.get_text() for t in figure.axes[0].texts] == [
                "n = 3 in the smallest group: individual points say more "
                "than a bar here"]
        finally:
            plt.close(figure)

    with subtests.test(msg="path_written_through_save_figure"):
        target = tmp_path / "comparison.pdf"
        figure = gmc.plot(_comparison(SPREAD), str(target))
        try:
            assert target.read_bytes()[:5] == b"%PDF-"
        finally:
            plt.close(figure)


def test_render_comparison_refuses_nothing_finite_and_honours_the_filter(
        subtests):
    """No rows, no finite value or an unmatched filter give (None, None)."""
    refusals = [
        ("no_comparison", None, gmc.ComparisonStyle()),
        ("no_rows", _empty(), gmc.ComparisonStyle()),
        ("all_values_non_finite", _comparison(ALL_NAN), gmc.ComparisonStyle()),
        ("violin_all_values_non_finite",
         _comparison({"g": [np.nan], REST: [np.inf]}),
         gmc.ComparisonStyle(kind="violin")),
        ("filter_matches_no_group", _comparison(SPREAD),
         gmc.ComparisonStyle(kind="violin", only="not_a_group")),
    ]
    for case_id, comparison, style in refusals:
        with subtests.test(msg=case_id):
            assert gmc.render_comparison(comparison, style) == (None, None)

    cases = [
        # (case id, values, style, labels, n collections)
        ("styled_violin_keeps_single_point_groups", ONE_POINT_EACH,
         gmc.ComparisonStyle(kind="violin"),
         ["g\n(n=1)", f"{REST}\n(n=1)"], 2),
        ("one_finite_value_per_group_is_enough",
         {"g": [np.nan, 2.0], REST: [np.inf, 7.0]},
         gmc.ComparisonStyle(kind="violin"),
         ["g\n(n=1)", f"{REST}\n(n=1)"], 2),
        ("filter_draws_only_the_named_group", SPREAD,
         gmc.ComparisonStyle(kind="violin", only="g"), ["g\n(n=3)"], 1),
        ("jitter_skips_empty_group", NAN_GROUP,
         gmc.ComparisonStyle(kind="jitter"),
         ["g\n(n=0)", f"{REST}\n(n=4)"], 1),
        ("jitter_box_skips_empty_group", NAN_GROUP,
         gmc.ComparisonStyle(kind="jitter_box"),
         ["g\n(n=0)", f"{REST}\n(n=4)"], 1),
    ]
    for case_id, values, style, labels, n_collections in cases:
        with subtests.test(msg=case_id):
            figure, axes = gmc.render_comparison(_comparison(values), style)
            try:
                assert figure is axes.figure
                assert figure.axes[0] is axes
                assert _labels(axes) == labels
                assert len(axes.collections) == n_collections
            finally:
                plt.close(figure)


def test_save_always_leaves_the_record_and_only_what_was_written(
        subtests, tmp_path, built, monkeypatch):
    """Each artifact is listed only when written; settings.json always is."""
    cases = [
        # (case id, comparison, kind, written keys, files on disk)
        ("drawable_writes_everything",
         _comparison({"g": [1.0, 2.0, 3.0], REST: [4.0, 5.0, 6.0]}),
         "jitter_box", ["data", "pdf", "png", "settings"],
         ["comparison.pdf", "comparison.png", "data.csv", "settings.json"]),
        ("bar_with_statistics_writes_everything",
         _comparison({"g": [1.0, 2.0], REST: [3.0, 4.0]}, level="well",
                     statistics=[{"test": "t-test", "p": 0.5}]),
         "bar", ["data", "pdf", "png", "settings", "statistics"],
         ["comparison.pdf", "comparison.png", "data.csv", "settings.json",
          "statistics.csv"]),
        ("undrawable_keeps_numbers_and_statistics",
         _comparison(ALL_NAN, statistics=[{"test": "t-test", "p": 0.5}]),
         "jitter_box", ["data", "settings", "statistics"],
         ["data.csv", "settings.json", "statistics.csv"]),
        ("empty_writes_only_the_record", _empty(), "bar", ["settings"],
         ["settings.json"]),
    ]
    for case_id, comparison, kind, keys, files in cases:
        with subtests.test(msg=case_id):
            folder = tmp_path / case_id
            written = gmc.save(comparison, str(folder), kind=kind)
            assert sorted(written) == keys
            assert sorted(os.listdir(folder)) == files
            for key, path in written.items():
                assert path == str(folder / {
                    "pdf": "comparison.pdf", "png": "comparison.png",
                    "data": "data.csv", "statistics": "statistics.csv",
                    "settings": "settings.json"}[key])
            if "data" in written:
                pd.testing.assert_frame_equal(
                    pd.read_csv(written["data"]),
                    comparison.frame.reset_index(drop=True),
                    check_dtype=False)
            if "statistics" in written:
                assert pd.read_csv(written["statistics"]).to_dict(
                    "records") == [{"test": "t-test", "p": 0.5}]

    with subtests.test(msg="empty_record_says_nothing_was_there"):
        record = json.loads((tmp_path / "empty_writes_only_the_record" /
                             "settings.json").read_text(encoding="utf-8"))
        assert {k: record[k] for k in ("groups", "n_per_group", "measurement",
                                       "plot", "level")} == {
            "groups": [], "n_per_group": {}, "measurement": "pathogen_area",
            "plot": "bar", "level": "well"}
        assert record["level_means"].startswith("one row per well")
        assert "regression_settings" not in record

    with subtests.test(msg="nested_folder_created_and_settings_made_json"):
        target = tmp_path / "nested" / "run-01"
        written = gmc.save(built, str(target), settings={
            "threshold": np.float64(0.25), "genes": ("a", "b"),
            "channels": [0, 1, 2], "nested": {"alpha": np.int64(3)},
            "backend": np.dtype("float32"), "flag": True, "none": None})
        record = json.loads((target / "settings.json").read_text(
            encoding="utf-8"))
        assert written["settings"] == str(target / "settings.json")
        assert record["regression_settings"] == {
            "threshold": 0.25, "genes": ["a", "b"], "channels": [0, 1, 2],
            "nested": {"alpha": "3"}, "backend": "float32", "flag": True,
            "none": None}
        assert record["groups"] == ["g", REST]
        assert record["n_per_group"] == {"g": 6, REST: 6}

    with subtests.test(msg="private_figure_closed_callers_kept"):
        caller = plt.figure()
        before = set(plt.get_fignums())
        try:
            gmc.save(built, str(tmp_path / "closing"))
            assert set(plt.get_fignums()) == before
        finally:
            plt.close(caller)

    with subtests.test(msg="crops_bad_one_and_empty_well_skipped"):
        grey = np.zeros((4, 4), dtype=np.uint8)
        grey[1, 1] = 200
        colour = np.zeros((3, 3, 4), dtype=np.uint8)
        written = gmc.save(built, str(tmp_path / "crops"), images={
            "p1_r1_c1": [grey, np.zeros((4,), dtype=np.uint8), colour],
            "p1_r1_c2": [], "p1_r1_c3": None})
        cells = tmp_path / "crops" / "cells"
        assert written["cells"] == str(cells)
        assert sorted(os.listdir(cells)) == ["p1_r1_c1"]
        assert sorted(os.listdir(cells / "p1_r1_c1")) == ["0000.png",
                                                          "0002.png"]
        from PIL import Image
        with Image.open(cells / "p1_r1_c1" / "0000.png") as image:
            assert image.mode == "RGB"
            assert np.asarray(image)[1, 1].tolist() == [200, 200, 200]

    with subtests.test(msg="no_crops_makes_no_cells_folder"):
        written = gmc.save(built, str(tmp_path / "nocrops"),
                           images={"p1_r1_c1": []})
        assert "cells" not in written
        assert not (tmp_path / "nocrops" / "cells").exists()

    with subtests.test(msg="refusing_figure_writer_costs_only_formats"):
        from spacr import plot as plot_module

        def _refuse(*_args, **_kwargs):
            raise OSError("no writable font cache")

        monkeypatch.setattr(plot_module, "save_figure", _refuse)
        written = gmc.save(built, str(tmp_path / "noformats"))
        assert sorted(written) == ["data", "settings"]
        assert sorted(os.listdir(tmp_path / "noformats")) == [
            "data.csv", "settings.json"]


@pytest.fixture()
def montage_rows():
    """Three montage object rows carrying a ``prcfo`` identity."""
    return pd.DataFrame({
        "prcfo": ["p1_r1_c1_f1_1", "p1_r1_c1_f1_2", "p1_r1_c1_f2_1"],
        "count": [1, 2, 3],
    })


def test_join_measurements_reports_every_file_that_added_nothing(
        subtests, montage_rows, monkeypatch):
    """Good databases join; failed, anonymous or redundant ones are named."""
    from spacr import io as spacr_io

    wide = pd.DataFrame({"prcfo": montage_rows["prcfo"],
                         "cell_area": [10.0, 20.0, 30.0]})

    def _read(path, **_kwargs):
        path = str(path)
        if "bad" in path:
            raise RuntimeError("database is locked")
        if "empty" in path:
            return pd.DataFrame()
        if "anon" in path:
            return pd.DataFrame({"x": [1, 2, 3]})
        if "same" in path:
            return pd.DataFrame({"prcfo": montage_rows["prcfo"],
                                 "count": [9, 9, 9]})
        return wide

    monkeypatch.setattr(spacr_io, "_read_and_join_tables", _read)

    joined = [
        ("empty_database_stepped_over", ["/empty.db", "/good.db"], ""),
        ("anonymous_rows_named", ["/anon.db", "/good.db"],
         "/anon.db: the joined tables name no object"),
        ("unopenable_database_named_beside_good_join",
         ["/bad.db", "/good.db"], "/bad.db: database is locked"),
    ]
    for case_id, databases, note in joined:
        with subtests.test(msg=case_id):
            out, got = gmc.join_measurements(montage_rows, databases)
            assert got == note
            assert list(out.columns) == ["prcfo", "count", "cell_area"]
            assert list(out["cell_area"]) == [10.0, 20.0, 30.0]
            assert list(out["count"]) == [1, 2, 3]

    untouched = [
        ("nothing_new_to_add", montage_rows, ["/same.db"],
         "the measurement tables add no column these rows do not already "
         "have"),
        ("every_database_failed", montage_rows, ["/bad.db", "/empty.db"],
         "no measurement table could be read: /bad.db: database is locked"),
        ("no_database_given", montage_rows, [],
         "no measurement table could be read"),
    ]
    for case_id, rows, databases, note in untouched:
        with subtests.test(msg=case_id):
            out, got = gmc.join_measurements(rows, databases)
            assert out is rows
            assert got == note
            assert list(out["count"]) == [1, 2, 3]
