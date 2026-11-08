"""Behaviour of spacr.assay_report paths the main report tests leave out.

Held here: one-page fitting and grouped pagination, legend pages, condition
chunking edge cases, long-format and database helpers, well and object QC
measure subsets, grouped panels with outliers and log axes, compact styling,
tile selection fallbacks, plate geometry helpers, legend sentences and the
failure and bin-image paths of build_assay_report.
"""
import os
import sqlite3
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from spacr import assay_report as ar  # noqa: E402
from spacr.assay_panels import axis_layout, outlier_policy, string_parts  # noqa: E402
from tests.test_assay_report import _field, _frame, _report_inputs  # noqa: E402


@pytest.fixture(autouse=True)
def _close_figures(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def _item(h, lo=None, width=1.0):
    return ar.ReportItem("R", "R", lambda fig, rect: None, width, h, min_height_mm=lo)


FIT = replace(ar.COMPACT, fit_one_page=True)


# --------------------------------------------------------------------------- pagination
def test_rows_of_no_items_and_height_of_no_blocks_are_empty():
    """No items give no rows, and no blocks need no height."""
    assert ar._rows([]) == []
    assert ar._one_page_height([], FIT) == 0.0


def test_one_page_fit_keeps_heights_that_already_fit():
    """A figure that fits one page keeps its item heights."""
    pages = ar.paginate([ar.ReportBlock("a", [_item(50, 20)])], FIT)
    assert len(pages) == 1
    item = [pl for pl in pages[0] if pl["kind"] == "item"][0]
    assert item["rect_mm"][3] == 50


def test_one_page_fit_shrinks_rows_towards_their_minimum():
    """A slightly too tall figure is shrunk onto one page, never below the minimum heights."""
    block = ar.ReportBlock("a", [_item(100, 40), _item(100, 40), _item(100, 40)])
    pages = ar.paginate([block], FIT)
    assert len(pages) == 1
    heights = [pl["rect_mm"][3] for pl in pages[0] if pl["kind"] == "item"]
    assert len(heights) == 3 and all(40 <= h < 100 for h in heights)
    bottom = max(pl["rect_mm"][1] + pl["rect_mm"][3] for pl in pages[0])
    assert bottom <= FIT.margin + FIT.usable + 1e-6


def test_one_page_fit_packs_block_groups_page_by_page():
    """Groups that cannot share a page are packed per page; an oversized group flows on its own."""
    blocks = [ar.ReportBlock("big0", [_item(300, 300)]),
              ar.ReportBlock("b1", [_item(150, 150)]),
              ar.ReportBlock("g1", [_item(100, 100)], group="g"),
              ar.ReportBlock("g2", [_item(100, 100)], group="g"),
              ar.ReportBlock("big", [_item(300, 300)]),
              ar.ReportBlock("b5", [_item(50, 50)])]
    pages = ar.paginate(blocks, FIT)
    heads = [[pl["text"] for pl in pg if pl["kind"] == "heading"] for pg in pages]
    assert heads == [["big0"], ["b1"], ["g1", "g2"], ["big"], ["b5"]]
    pages = ar.paginate([ar.ReportBlock("x", [_item(300, 300)]), ar.ReportBlock("y", [_item(300, 300)])],
                        FIT)
    assert [[pl["text"] for pl in pg if pl["kind"] == "heading"] for pg in pages] == [["x"], ["y"]]


def test_paginate_overrides_page_geometry():
    """Explicit margin and gap override the layout geometry."""
    pages = ar.paginate([ar.ReportBlock("", [_item(30, width=0.5), _item(30, width=0.5)])],
                        margin=20.0, gap=10.0)
    xs = sorted(pl["rect_mm"][0] for pl in pages[0] if pl["kind"] == "item")
    w = 0.5 * (ar.A4_MM[0] - 40 + 10) - 10
    assert xs == pytest.approx([20.0, 20.0 + w + 10.0])


def test_fit_axes_leaves_an_invisible_axes_at_the_rect():
    """An axes without a tight box is placed at the rect and not shrunk."""
    fig = ar._new_figure(100, 100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_visible(False)
    ar.fit_axes(fig, ax, (0.1, 0.2, 0.5, 0.6))
    assert tuple(np.round(ax.get_position().bounds, 6)) == (0.1, 0.2, 0.5, 0.6)


# --------------------------------------------------------------------------- legend pages
def test_render_report_puts_the_legend_on_pages_of_its_own(tmp_path):
    """Legend text gets its own pages, numbered after the figure pages, with or without PNGs."""
    pages = ar.paginate([ar.ReportBlock("a", [_item(30)])])
    text = "\n".join(f"Paragraph {i} " + "word " * 30 for i in range(120))
    res = ar.render_report(pages, str(tmp_path / "r.pdf"), legend_text=text, title="t", png=False)
    assert res["png"] == [] and res["n_figure_pages"] == 1
    assert res["n_legend_pages"] == len(ar._legend_pages(text)) > 1
    assert res["n_pages"] == 1 + res["n_legend_pages"]
    res = ar.render_report(pages, str(tmp_path / "s.pdf"), legend_text="short legend", title="t", dpi=30)
    assert [os.path.basename(p) for p in res["png"]] == ["s.png", "s_page_02.png"]
    assert all(os.path.exists(p) for p in res["png"])


def test_legend_pages_keep_empty_paragraphs_as_blank_lines():
    """An empty paragraph becomes one blank line."""
    pages = ar._legend_pages("first\n\nsecond")
    assert pages == [["first", "", "second"]]


# --------------------------------------------------------------------------- condition chunks
def test_condition_chunks_without_a_fitting_width_split_at_full_width():
    """No layout width holds the panel, so conditions are split into full-width chunks."""
    width, chunks = ar.condition_chunks(list("abc"), 100.0, 0.0, 4, layout=ar.SPACIOUS)
    assert width == 1.0 and chunks == [["a"], ["b"], ["c"]]


def test_condition_chunks_too_many_for_half_width_go_full_in_one_chunk():
    """More conditions than half a row allows, but few enough for one row, stay in one full chunk."""
    width, chunks = ar.condition_chunks(list("abc"), 1.0, 1.0, 4)
    assert width == "full" and chunks == [["a", "b", "c"]]


# --------------------------------------------------------------------------- long tables and databases
def test_long_tables_skip_missing_columns_and_can_be_empty():
    """Object types without the column are skipped; none at all gives an empty table with the columns."""
    df = _frame(n_cond=2, objects=("cell",), channels=(0,))
    empty = ar.well_intensity_long(df, "condition", 0, objects=["ghost"])
    assert empty.empty and "mean_intensity" in empty.columns
    long = ar.well_column_long(df, "condition", {"cell": "cell_area", "ghost": "ghost_area"}, "area")
    assert set(long["object_type"]) == {"cell"} and (long["area"] == 100.0).all()
    none = ar.well_column_long(df, "condition", {"ghost": "ghost_area"}, "area")
    assert none.empty and list(none.columns) == ["condition", "plateID", "rowID", "columnID",
                                                 "object_type", "column", "area", "n_objects"]


def test_db_tables_need_well_keys_and_no_tables_count_nothing(tmp_path):
    """Tables without well keys are not object tables; an empty table list gives an empty count."""
    db = tmp_path / "measurements.db"
    with sqlite3.connect(db) as con:
        con.execute("create table cell (plateID text, rowID text, columnID text)")
        con.execute("create table notes (text text)")
        con.execute("create table png_list (plateID text, rowID text, columnID text)")
    assert ar.db_object_tables(str(db)) == ["cell"]
    frame = pd.DataFrame(dict(plateID=["p1"], rowID=["r1"], columnID=["c1"], condition=["a"]))
    counts = ar.db_well_counts(str(db), frame, "condition", tables=[])
    assert counts.empty and list(counts.columns) == ["condition", "plateID", "rowID", "columnID",
                                                     "object_type", "objects_per_well", "count_source"]


# --------------------------------------------------------------------------- QC
def _qc_frame():
    df = _frame(n_cond=2, objects=("cell", "pathogen"), channels=(0,))
    df["organelle_channel_0_mean_intensity"] = 50.0
    return df


def test_well_qc_counts_only_countable_objects_per_plate():
    """An object type without a count column gets no count; without a group the plate is the reference."""
    t = ar.well_qc_table(_qc_frame(), measures=("count",))
    assert "cell_count" in t.columns and "pathogen_count" in t.columns
    assert "organelle_count" not in t.columns
    assert not any(c.endswith("_area_median") or "intensity" in c for c in t.columns)
    assert set(t["qc_group"]) == {""} and set(t["qc_reference"]) == {"plate"}
    assert (t["cell_count"] == 4).all()


def test_well_qc_intensity_skips_channels_without_a_column():
    """Requested channels without a mean-intensity column are skipped."""
    t = ar.well_qc_table(_qc_frame(), measures=("intensity",), channels=[0, 9])
    assert {"cell_ch0_intensity_median", "organelle_ch0_intensity_median"} <= set(t.columns)
    assert not any("ch9" in c for c in t.columns) and "cell_count" not in t.columns


def test_object_qc_measure_subsets():
    """Only the requested measures are checked."""
    df = _qc_frame()
    _, area = ar.object_qc_mask(df, measures=("area",))
    assert set(area["measure"]) == {"area"}
    keep, inten = ar.object_qc_mask(df, measures=("intensity",))
    assert set(inten["measure"]) == {"ch0 intensity"} and keep.all()


# --------------------------------------------------------------------------- grouped panels
def _grouped_long(order=("a", "b"), objects=(("cell", 100.0), ("pathogen", 1.0)), wells=6, spike=None,
                  drop=None):
    rng = np.random.default_rng(0)
    rows = []
    for g in order:
        for obj, base in objects:
            n = wells - 2 if drop == obj else wells
            for w in range(n):
                v = base * (1 + rng.normal(0, 0.03))
                if spike and spike == (g, obj) and w == 0:
                    v = base * 50
                rows.append(dict(condition=g, plateID="p1", rowID=f"r{order.index(g)}", columnID=f"c{w}",
                                 object_type=obj, mean_intensity=v))
    return pd.DataFrame(rows)


def test_grouped_outliers_are_marked_left_out_and_reported():
    """Outliers are marked per object type, left out of the dots, and named in the legend."""
    long = _grouped_long(spike=("a", "cell"))
    with outlier_policy("mad"):
        spec = ar.grouped_spec(long, "condition", ["a", "b"], ["cell", "pathogen"])
    assert int(spec["data"]["outlier"].sum()) == 1
    out = spec["stats"]["cell"]["outliers"]
    assert int(out.loc[out["condition"] == "a", "n_removed"].iloc[0]) == 1
    fig, ax = plt.subplots()
    ar.draw_grouped_panel(ax, spec)
    dots = sum(len(c.get_offsets()) for c in ax.collections if c.get_gid() == "grouped_dots")
    assert dots == len(long) - 1
    text = ar._outlier_text(spec)
    assert "MAD fence" in text and "cell a: 1" in text
    assert "outliers" in set(ar.stats_long(spec)["table"])


def test_grouped_compact_log_axis_and_missing_condition():
    """Values spanning a large ratio draw on a log axis; a condition without data draws no dots."""
    long = _grouped_long()
    spec = ar.grouped_spec(long, "condition", ["a", "b", "z"], ["cell", "pathogen"])
    fig, ax = plt.subplots()
    ar.draw_grouped_panel(ax, spec, compact=True)
    assert ax.get_yscale() == "log" and ax.get_title() == "log axis"
    xs = np.concatenate([c.get_offsets()[:, 0] for c in ax.collections if c.get_gid() == "grouped_dots"])
    assert xs.max() < 1.6
    flat = ar.grouped_spec(_grouped_long(objects=(("cell", 10.0), ("pathogen", 9.0))), "condition",
                           ["a", "b"], ["cell", "pathogen"])
    fig, ax = plt.subplots()
    ar.draw_grouped_panel(ax, flat, compact=True)
    assert ax.get_yscale() == "linear" and ax.get_title() == " "


def test_compact_grouped_spec_items_use_the_compact_width_per_condition():
    """A compact grouped spec takes 0.75 mm per object type plus 1.4 mm per condition."""
    spec = ar.grouped_spec(_grouped_long(), "condition", ["a", "b"], ["cell", "pathogen"])
    assert ar._per_category(spec, ar.COMPACT) == (0.75 * 2 + 1.4, 10)
    items = ar.spec_items("R1", spec, 12, ar.COMPACT)
    assert len(items) == 1 and items[0].frac == ar.COMPACT.widths[0]


def test_grouped_legend_names_n_once_or_per_object_type():
    """Equal n per object type is stated once; differing n is stated per object type."""
    same = ar.grouped_spec(_grouped_long(), "condition", ["a", "b"], ["cell", "pathogen"])
    text = ar._spec_legend("R1", same, "What.", "wells")
    assert "n wells per condition (every object type): a: 6; b: 6." in text
    diff = ar.grouped_spec(_grouped_long(drop="pathogen"), "condition", ["a", "b"], ["cell", "pathogen"])
    text = ar._spec_legend("R2", diff, "What.", "wells")
    assert "every object type" not in text
    assert "cell: n wells per condition a: 6; b: 6;" in text
    assert "pathogen: n wells per condition a: 4; b: 4;" in text


# --------------------------------------------------------------------------- compact styling and axes
def test_compact_style_leaves_other_artists_and_empty_ticks_alone():
    """Fills, circles and empty tick labels are left as they are; bars are slimmed."""
    from matplotlib.patches import Circle
    fig, ax = plt.subplots()
    bars = ax.bar([0, 1], [1, 2], width=0.8)
    fill = ax.fill_between([0, 1], [0, 0], [1, 1], linewidth=2.0)
    circ = Circle((0.5, 0.5), 0.1, linewidth=3.0)
    ax.add_patch(circ)
    ax.set_xticks([])
    ar.compact_style(ax)
    assert bars[0].get_width() == pytest.approx(0.8 * 0.7)
    assert circ.get_linewidth() == 3.0
    assert fill.get_linewidths()[0] == 2.0
    assert len(ax.get_xticks()) == 0


def test_matrix_axis_height_counts_the_matrix_rows():
    """A +/- matrix axis adds one compact row height per factor under the plot."""
    order = ["A_x", "A_y", "B_x"]
    axis = axis_layout(order, {g: string_parts(g) for g in order}, "auto")
    data = pd.DataFrame(dict(condition=np.repeat(order, 4), v=np.arange(12.0)))
    spec = ar.bar_spec(data, "v", "condition", order)
    spec.update(axis=axis)
    labels, ann, matrix = ar._axis_for(spec, ["A_x", "B_x"])
    assert matrix and labels is None and all(set(m) == {"A_x", "B_x"} for m in ann.values())
    h = ar._panel_item_height(spec, order, ar.COMPACT)
    xlab = len(axis["annotations"]) * ar.COMPACT_ANNOT[1] / 72 * 25.4 + 2.0
    head = 1.5 + 3.0
    assert h == pytest.approx(ar.COMPACT.id_strip + head + ar.COMPACT.plot_mm + xlab)


# --------------------------------------------------------------------------- tiles
def test_mask_dims_overrides_replace_only_given_planes(tmp_path):
    """Overrides set the plane of a role; a None override keeps the default plane."""
    dims = ar.resolve_mask_dims(str(tmp_path), ["cell", "pathogen"], 7, {"cell": "2", "pathogen": None})
    assert dims == {"cell": 2, "pathogen": 6}


def test_extract_tile_rejections(tmp_path):
    """Missing labels, oversized pathogens and crops beyond the field are rejected."""
    arr = _field(tmp_path / "f.npy")
    assert ar.extract_tile(arr, 4, 99, 48, center_plane=6) is None
    assert ar.extract_tile(arr, 4, 1, 12, center_plane=6) is None
    assert ar.extract_tile(arr, 4, 1, 120, require_inside=True) is None
    raw, box, padded = ar.extract_tile(arr, 4, 1, 120, center_plane=6, require_inside=False)
    assert padded.shape == (120, 120, 7) and raw.shape[0] < 120


def test_tile_colours_by_name_or_tuple():
    """Colour names resolve, tuples pass through, unknown names are refused and named back."""
    assert ar._colour([1, 0, 0]) == (1.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="unknown tile colour 'purple'"):
        ar._colour("purple")
    assert ar._colour_name("red") == "red"
    assert ar._colour_name((0, 1, 0)) == "green"
    assert ar._colour_name((0.3, 0.3, 0.3)) == "(0.3, 0.3, 0.3)"


def test_build_tiles_skips_unusable_rows_and_reads_ranges_from_fields(tmp_path):
    """Missing stacks, unparsable and absent labels are skipped, short groups are noted, and the
    field display range covers the mono channel and at most ``max_range_fields`` fields."""
    merged = tmp_path / "merged"
    merged.mkdir()
    _field(merged / "f1.npy", seed=0)
    _field(merged / "f2.npy", seed=1)
    df = pd.DataFrame(dict(condition=["a"] * 6,
                           file_name=["missing", "f1", "f1", "f1", "f2", "f2"],
                           object_label=["o1", "nolabel", "o99", "o1", "o2", "o3"],
                           pathogen_distance_to_field_edge=50.0))
    res = ar.build_tiles(df, "condition", ["a", "empty"], str(merged),
                         {"cell": 4, "nucleus": 5, "pathogen": 6}, {2: "red", 1: "green"},
                         str(tmp_path / "out"), n=5, size=64, seed=0, range_source="fields",
                         max_range_fields=1, mono_channel=3)
    assert len(res["table"]) == 3
    assert res["notes"] == ["a: 3 of 5 tiles found", "empty: 0 of 5 tiles found"]
    assert res["n_range_fields"] == 1 and set(res["ranges"]) == {1, 2, 3}
    assert res["mono_channel"] == 3 and res["dtype"] == "uint16"
    assert (res["montages"]["merged"][:64, -64:] == 0).all()


def test_vector_outlines_without_contours_add_nothing():
    """No contours for the drawn conditions add no line collection."""
    fig, ax = plt.subplots()
    ar._draw_vector_outlines(ax, {}, ["a"], 2, 10, 1)
    assert len(ax.collections) == 0


def test_crop_items_split_wide_orders_into_balanced_chunks():
    """More conditions than a row holds are split into balanced chunks labelled as continued."""
    order = list("abcde")
    result = dict(tiles={v: {} for v in ar.IMAGE_VERSIONS}, gap=1, contours=None)
    items = ar.crop_items("R3", result, order, 2, 16, "spacious", max_per_row=2)
    assert [it.label for it in items] == ["R3", "R3 (continued)", "R3 (continued)"]
    fig = ar._new_figure(200, 100)
    items[-1].draw(fig, (0.05, 0.05, 0.9, 0.9))
    assert len(fig.axes) == 3


# --------------------------------------------------------------------------- plate geometry
def test_well_index_and_oversized_plates():
    """Row letters map to numbers, other text gives None, and wells beyond 1536 keep their own shape."""
    assert ar._well_index("B") == 2 and ar._well_index("r12") == 12
    assert ar._well_index("") is None and ar._well_index("#") is None
    assert ar.plate_shape([40], [60]) == (40, 60)
    assert ar.plate_shape([3], [5]) == (8, 12)


# --------------------------------------------------------------------------- legend text
def _bar(outliers=None):
    data = pd.DataFrame(dict(condition=np.repeat(["a", "b", "c"], 6),
                             v=np.r_[np.full(6, 1.0) + np.arange(6) * 0.01,
                                     np.full(6, 5.0) + np.arange(6) * 0.01,
                                     np.full(6, 9.0) + np.arange(6) * 0.01]))
    if outliers:
        data.loc[0, "v"] = 500.0
    return ar.bar_spec(data, "v", "condition", ["a", "b", "c"], outliers=outliers)


def test_bar_outlier_sentence_names_the_removed_points():
    """Removed outliers are listed per condition in the legend sentence."""
    text = ar._outlier_text(_bar(outliers="mad"))
    assert "removed per condition: a: 1." in text


def test_requested_bracket_pairs_are_named_in_the_legend():
    """A list of bracket pairs is described as the requested pairs."""
    spec = _bar()
    spec["bracket_pairs"] = [("a", "c")]
    assert ar._bracket_text(spec).startswith(" Brackets mark the requested pairs")


def test_control_legend_states_n_and_no_test():
    """Control legends give n wells per condition and say that no test was run."""
    text = ar._control_legend("C1", _bar(), "Cell area.")
    assert text.startswith("C1. Cell area. Bars: mean ± SD over wells")
    assert "n wells per condition: a: 6; b: 6; c: 6." in text and "No statistical test (control)." in text
    grouped = ar.grouped_spec(_grouped_long(), "condition", ["a", "b"], ["cell", "pathogen"])
    assert "a: 6; b: 6" in ar._control_legend("C2", grouped, "Intensity.")


def test_filter_steps_drawer_skips_conditions_without_steps():
    """A condition without filter-step rows draws no line; the all-objects line is always drawn."""
    steps = pd.DataFrame(dict(step_index=[0, 1], step=["start", "size_filter"], group=["a", "a"],
                              remaining=[100, 80]))
    table = ar.filter_steps_table(steps, ["a", "b"])
    draw = ar.filter_steps_drawer(table, ["a", "b"])
    fig = ar._new_figure(180, 90)
    draw(fig, (0.1, 0.2, 0.6, 0.7))
    ax = fig.axes[0]
    assert [ln.get_label() for ln in ax.lines] == ["a", "all"]
    assert [t.get_text() for t in ax.texts] == ["−20.0%"]


# --------------------------------------------------------------------------- build_assay_report
def _bins(df):
    df = df.copy()
    df["bin"] = np.where(np.arange(len(df)) % 2 == 0, "low", "high")
    return df


def test_report_records_failed_images_and_unreadable_database(tmp_path):
    """Tile, crop and bin-image failures go into the legend; an unreadable database is ignored;
    an explicit count list drops cytoplasm; heatmaps without well QC mark no wells."""
    df, order, labels, tiles = _report_inputs(tmp_path, 2, channels=(0,))
    df = _bins(df)
    bad = dict(tiles, mask_dims={"pathogen": 6})
    db = tmp_path / "measurements.db"
    db.write_bytes(b"not a database at all, just text" * 10)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                                "condition", order, [0], labels=labels, tiles=bad, crops={"n": 2},
                                db_path=str(db), count_objects=["cell", "cytoplasm"],
                                extra_specs=lambda version, frame: [],
                                bins=dict(column="bin", order=["low", "high"], name="size bin"),
                                heatmaps=True, dpi=40)
    legend = open(res["legend"], encoding="utf-8").read()
    assert "Tiles could not be made: 'cell'." in legend
    assert "crops could not be made: 'cell'." in legend
    assert "Images per bin could not be made: 'cell'." in legend
    assert "red x marks wells that failed well_qc (0 wells)" in legend
    names = os.listdir(tmp_path / "rep" / "panels")
    assert any("cell_objects_per_well" in n for n in names)
    assert not any("cytoplasm_objects_per_well" in n for n in names)
    assert any("heatmap_bin_low_fraction" in n for n in names)


def test_report_bin_images_per_bin(tmp_path):
    """With tiles and bins, one representative object and a crop set per bin are drawn."""
    df, order, labels, tiles = _report_inputs(tmp_path, 2, channels=(0,))
    df = _bins(df)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                                "condition", order, [0], labels=labels, tiles=tiles,
                                bins=dict(column="bin", order=["low", "high"], labels={"low": "small"},
                                          name="size bin"),
                                heatmaps=False, dpi=40)
    legend = open(res["legend"], encoding="utf-8").read()
    assert "One object per size bin closest to the median" in legend
    assert "The 5 objects per size bin closest to the median" in legend
    names = os.listdir(tmp_path / "rep" / "panels")
    rep = [n for n in names if "representative_per_bin_filtered" in n]
    assert len(rep) == 1
    data = pd.read_csv(tmp_path / "rep" / "panels" / rep[0] / "data.csv")
    assert set(data["bin"]) == {"low", "high"} and len(data) == 2


def test_report_without_found_tiles_draws_no_image_panels(tmp_path):
    """Tiles that cannot be found anywhere leave out the tile and bin image panels without an error."""
    df, order, labels, tiles = _report_inputs(tmp_path, 2, channels=(0,))
    df = _bins(df)
    empty = tmp_path / "empty_merged"
    empty.mkdir()
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                                "condition", order, [0], labels=labels,
                                tiles=dict(tiles, merged_dir=str(empty)),
                                bins=dict(column="bin", order=["low", "high"]), heatmaps=False, dpi=40)
    folders = [os.path.basename(f) for f in res["panels"].values()]
    assert not any("tiles_" in f or "per_bin" in f for f in folders)
    assert any("recruitment_filtered_well" in f for f in folders)


def test_report_bins_absent_from_the_data_add_nothing(tmp_path):
    """Bins with no matching objects add no images and no bin sentence."""
    df, order, labels, _ = _report_inputs(tmp_path, 2, channels=(0,))
    df = _bins(df)
    res = ar.build_assay_report({"unfiltered": df, "filtered": df}, str(tmp_path / "rep"), "recruitment",
                                "condition", order, [0], labels=labels, tiles=None,
                                bins=dict(column="bin", order=["huge"], name="size bin"),
                                heatmaps=False, dpi=40)
    legend = open(res["legend"], encoding="utf-8").read()
    assert "size bin" not in legend


def test_panel_folder_without_tables_holds_only_the_figure(tmp_path):
    """A panel folder saved without data or stats holds only the PDF and PNG."""
    def draw(fig, rect):
        fig.add_axes(rect).plot([0, 1], [0, 1])
    folder = ar.write_panel_folder(str(tmp_path), "R09", "line", draw, (60.0, 40.0), dpi=40)
    assert sorted(os.listdir(folder)) == ["R09_line.pdf", "R09_line.png"]
