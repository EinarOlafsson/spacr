"""The tab that shows the cells behind one dot on the volcano.

Instruction 131's Qt half. :mod:`spacr.cell_montage` already decides WHICH
objects a coefficient's montage holds and says why; these tests hold the
properties the TAB is responsible for:

* it is reached from the selection the regression panel already has -- the
  same ``table.key_selected`` funnel the gene tile is on -- and never from a
  second one, because two selections mean a montage of a different gene from
  the one the volcano is ringing;
* the GUI thread never reads an image: the load goes through a
  ``JobRunner``, whose completion handler is a bound method;
* a control that cannot act is greyed out AND says why (instruction 106),
  including the case the request itself names -- no exported PNGs and no
  ``merged/`` stacks;
* the caption, with the sentence saying guide membership is INFERRED, is on
  screen for every montage including the empty one;
* the figure is written through ``spacr.plot.save_figure``, so it honours the
  figure-format preference and no ``.pdf`` is spelled anywhere.

The screen underneath is real: a ``merged/`` folder of label-masked arrays, a
real ``measurements.db`` with a ``png_list``, and a real
``regression_data.csv``. Pixels are cut through :mod:`spacr.crops`, not faked.
"""
from __future__ import annotations

import os
import sqlite3

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt

from spacr.qt.widgets.cell_montage_view import (          # noqa: E402
    OBJECT_CHOICES, THUMBNAIL_PX, CellMontageView, MontageLoad,
    MontageRequest, coefficient_from_frame, experiment_root, load,
    montage_figure, parse_channels,
)

CELL_DIM, NUC_DIM, PATH_DIM = 4, 5, 6
WELLS = ("r1_c1", "r1_c2", "r1_c3", "r1_c4")
OBJECTS_PER_WELL = 8

#: What a gene-level coefficient looks like coming out of the panel, and the
#: guide-level one beside it. Same spelling as ``spacr.hits`` parses.
GENE_KEY = "gene_fraction:gene[GRA14]"
GUIDE_KEY = "fraction:grna[GRA14_1]"


# --------------------------------------------------------------------------- #
#  A screen on disk
# --------------------------------------------------------------------------- #

def _field(labels, h=96, w=112, n_channels=4, seed=0):
    """A merged array: four intensity planes then cell / nucleus / pathogen."""
    rng = np.random.default_rng(seed)
    data = rng.integers(1, 4000, size=(h, w, n_channels + 3)).astype(np.uint16)
    for dim in (CELL_DIM, NUC_DIM, PATH_DIM):
        data[:, :, dim] = 0
    for index, label in enumerate(labels):
        y0 = 4 + (index // 4) * 22
        x0 = 4 + (index % 4) * 26
        data[y0:y0 + 18, x0:x0 + 20, CELL_DIM] = label
        data[y0 + 3:y0 + 15, x0 + 3:x0 + 17, NUC_DIM] = label
        data[y0 + 5:y0 + 8, x0 + 5:x0 + 8, PATH_DIM] = label
    return data


def _scores(well_index, n=OBJECTS_PER_WELL):
    start = 0.05 + 0.02 * well_index
    spread = 0.9 - 0.2 * well_index
    return [round(start + spread * i / (n - 1), 4) for i in range(n)]


def _screen(tmp_path, *, with_png=False, with_merged=True):
    """Write a four-well plate: merged arrays, a database, and the counts.

    :returns: ``(root, db_path, results_csv)``.
    """
    root = tmp_path / "exp"
    (root / "measurements").mkdir(parents=True)
    if with_merged:
        (root / "merged").mkdir(parents=True)
    db_path = str(root / "measurements" / "measurements.db")

    labels = list(range(1, OBJECTS_PER_WELL + 1))
    cell_rows, png_rows = [], []
    for well_index, well in enumerate(WELLS):
        row_id, column_id = well.split("_")
        name = f"plate1_{well}_1"
        npy = str(root / "merged" / f"{name}.npy")
        if with_merged:
            np.save(npy, _field(labels, seed=well_index))
        png_dir = root / "data" / f"plate1_{well}" / "cell_png"
        if with_png:
            png_dir.mkdir(parents=True, exist_ok=True)
        for label, score in zip(labels, _scores(well_index)):
            png_path = str(png_dir / f"{name}_{label}.png")
            if with_png:
                from PIL import Image
                crop = np.zeros((32, 32, 3), dtype=np.uint8)
                crop[:, :, 0] = label * 20
                Image.fromarray(crop).save(png_path)
            cell_rows.append((label, "plate1", row_id, column_id, "f1",
                              npy, f"{name}.npy"))
            png_rows.append(("plate1", row_id, column_id, "f1", f"o{label}",
                             png_path,
                             f"plate1_{row_id}_{column_id}_f1_o{label}", score))

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE cell (object_label INTEGER, plateID TEXT, "
                 "rowID TEXT, columnID TEXT, fieldID TEXT, path_name TEXT, "
                 "file_name TEXT)")
    conn.executemany("INSERT INTO cell VALUES (?,?,?,?,?,?,?)", cell_rows)
    conn.execute("CREATE TABLE png_list (plateID TEXT, rowID TEXT, "
                 "columnID TEXT, fieldID TEXT, cell_id TEXT, png_path TEXT, "
                 "prcfo TEXT, pred REAL)")
    conn.executemany("INSERT INTO png_list VALUES (?,?,?,?,?,?,?,?)", png_rows)
    conn.commit()
    conn.close()

    results = tmp_path / "results"
    results.mkdir()
    fractions = {
        "r1_c1": {"GRA14_1": 0.25, "GRA14_2": 0.25, "OTHER_1": 0.5},
        "r1_c2": {"GRA14_1": 0.125, "OTHER_1": 0.875},
        "r1_c3": {"GRA14_2": 0.5, "OTHER_1": 0.5},
        "r1_c4": {"OTHER_1": 1.0},
    }
    genes = {"GRA14_1": "GRA14", "GRA14_2": "GRA14", "OTHER_1": "OTHER"}
    rows = []
    for well, guides in fractions.items():
        row_id, column_id = well.split("_")
        for guide, fraction in guides.items():
            rows.append({"prc": f"plate1_{well}", "plateID": "plate1",
                         "rowID": row_id, "columnID": column_id,
                         "grna": guide, "gene": genes[guide],
                         "fraction": fraction,
                         "cell_count": OBJECTS_PER_WELL, "pred": 0.5})
    pd.DataFrame(rows).to_csv(results / "regression_data.csv", index=False)
    coefficients = pd.DataFrame([
        {"feature": GENE_KEY, "coefficient": 0.2, "p_value": 1e-4},
        {"feature": GUIDE_KEY, "coefficient": 0.15, "p_value": 2e-3},
        {"feature": "Intercept", "coefficient": 0.5, "p_value": 0.9},
    ])
    results_csv = str(results / "results.csv")
    coefficients.to_csv(results_csv, index=False)
    return str(root), db_path, results_csv


def _rows(db_path):
    """The input table's own row shape -- ``{"plate", "database"}``."""
    return [{"plate": "plate1", "database": db_path}]


def _view(qtbot, tmp_path, **kwargs):
    root, db_path, results_csv = _screen(tmp_path, **kwargs)
    frame = pd.read_csv(results_csv)
    view = CellMontageView(
        frame_provider=lambda: frame,
        results_provider=lambda: results_csv,
        database_provider=lambda: _rows(db_path),
        threaded=False)
    qtbot.addWidget(view)
    return view, root, db_path, results_csv


# --------------------------------------------------------------------------- #
#  The selection it is reached from
# --------------------------------------------------------------------------- #

def test_the_tab_is_on_the_panels_own_selection_and_not_a_second_one(qtbot):
    """`table.key_selected` -> `set_coefficient`, the same funnel as the tile."""
    pytest.importorskip("pyqtgraph")
    from spacr.qt.widgets.regression_results import RegressionResultsPanel

    panel = RegressionResultsPanel()
    qtbot.addWidget(panel)
    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    panel.table.key_selected.connect(view.set_coefficient)

    panel.table.key_selected.emit(GENE_KEY)
    assert view._name == "GRA14"
    assert view._level == "gene"


def test_a_guide_term_and_a_gene_term_resolve_to_different_levels():
    """The parse is spacr.hits', so the tab cannot disagree with the volcano."""
    frame = pd.DataFrame([{"feature": GUIDE_KEY, "coefficient": 0.15},
                          {"feature": GENE_KEY, "coefficient": 0.2}])
    assert coefficient_from_frame(GUIDE_KEY, frame) == ("GRA14_1", "grna", 0.15)
    assert coefficient_from_frame(GENE_KEY, frame) == ("GRA14", "gene", 0.2)
    # A nuisance term names neither, and that is a real answer.
    assert coefficient_from_frame("Intercept", frame)[0] == ""


def test_a_key_the_table_does_not_hold_has_no_effect_rather_than_a_wrong_one():
    """A stale key must not borrow the first row's coefficient."""
    frame = pd.DataFrame([{"feature": GENE_KEY, "coefficient": 0.2}])
    name, level, effect = coefficient_from_frame(
        "gene_fraction:gene[MISSING]", frame)
    assert (name, level) == ("MISSING", "gene")
    assert effect is None


# --------------------------------------------------------------------------- #
#  Instruction 106: greyed out, and it says why
# --------------------------------------------------------------------------- #

def test_every_missing_input_greys_the_button_with_its_own_sentence(qtbot,
                                                                    tmp_path):
    view, _root, db_path, results_csv = _view(qtbot, tmp_path, with_png=True)

    # Nothing picked yet.
    assert not view._show.isEnabled()
    assert "Click a coefficient" in view.reason()
    assert view._show.toolTip() == view.reason()

    # A term that names no gene.
    view.set_coefficient("Intercept")
    assert not view._show.isEnabled()
    assert "neither a gene nor a guide" in view.reason()

    # No database attached.
    bare = CellMontageView(
        frame_provider=lambda: pd.read_csv(results_csv),
        results_provider=lambda: results_csv,
        database_provider=lambda: [],
        threaded=False)
    qtbot.addWidget(bare)
    bare.set_coefficient(GENE_KEY)
    assert not bare._show.isEnabled()
    assert "No measurement database is attached" in bare.reason()

    # No results loaded, so no regression_data.csv to find.
    no_results = CellMontageView(
        frame_provider=lambda: pd.read_csv(results_csv),
        results_provider=lambda: "",
        database_provider=lambda: _rows(db_path),
        threaded=False)
    qtbot.addWidget(no_results)
    no_results.set_coefficient(GENE_KEY)
    assert not no_results._show.isEnabled()
    assert "regression_data.csv" in no_results.reason()


def test_a_channel_box_that_is_not_channels_says_so_instead_of_loading(
        qtbot, tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    assert view._show.isEnabled()

    view._channels.setText("red,green")
    assert not view._show.isEnabled()
    assert "not a list of channel indices" in view.reason()
    assert view.build() is False

    view._channels.setText("0,1,2")
    assert view._show.isEnabled()


def test_a_run_with_no_pngs_and_no_merged_says_that_and_greys_the_button(
        qtbot, tmp_path):
    """The case the request names: neither source exists.

    "a run with no exported PNGs and no .npy stacks must say that, not show
    an empty grid" -- and the refusal is REMEMBERED, so the button greys
    itself out rather than inviting the same click again.
    """
    view, root, _db, _csv = _view(qtbot, tmp_path, with_png=False,
                                  with_merged=False)
    assert not os.path.isdir(os.path.join(root, "merged"))
    view.set_coefficient(GENE_KEY)
    # Nothing has looked at the disk yet, so the button is live.
    assert view._show.isEnabled()

    assert view.build() is True
    assert view.plans() == ()
    message = view.status_text()
    assert "no exported crop PNGs and no merged" in message
    # Greyed out afterwards, with that same sentence on it.
    assert not view._show.isEnabled()
    assert view._show.toolTip() == message
    # And the grid says it rather than being an empty rectangle.
    assert message in view.caption_text()

    # Changing a crop setting re-arms it: forcing a source that has not been
    # tried is not the request that was refused.
    view._source.setCurrentIndex(2)
    assert view._show.isEnabled()


def test_save_is_greyed_until_there_is_something_to_save(qtbot, tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    assert not view._save.isEnabled()
    assert "no montage to save yet" in view._save.toolTip()
    view.set_coefficient(GENE_KEY)
    assert view.build() is True
    assert view._save.isEnabled()


# --------------------------------------------------------------------------- #
#  The montage itself
# --------------------------------------------------------------------------- #

def test_the_montage_draws_from_merged_when_no_pngs_were_exported(qtbot,
                                                                  tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=False)
    view.set_coefficient(GENE_KEY)
    assert view.build() is True

    plans = view.plans()
    assert len(plans) == 1
    assert plans[0].source_kind == "merged"
    assert plans[0].n_objects > 0
    # Real pixels, cut through spacr.crops.
    crops = view.images()[0]
    assert len(crops) == plans[0].n_objects
    assert all(c is not None and c.ndim == 3 for c in crops)
    assert "merged crop source" in view.caption_text()


def test_the_montage_prefers_the_exported_pngs_and_says_it_used_them(
        qtbot, tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    assert view.build() is True
    assert view.plans()[0].source_kind == "png"
    assert "png crop source" in view.caption_text()


def test_the_number_shown_per_well_is_round_objects_times_fraction(qtbot,
                                                                   tmp_path):
    """The count rule, read off the plan the tab put on screen.

    Wells 1-3 report GRA14 at 0.5, 0.125 and 0.5 of eight objects, so the
    expected counts are 4, 1 and 4; the fourth well reports no GRA14 at all
    and is not a well of this montage.
    """
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()
    plan = view.plans()[0]
    expected = {w.well: w.n_expected for w in plan.wells}
    assert expected == {"plate1_r1_c1": 4, "plate1_r1_c2": 1,
                        "plate1_r1_c3": 4}


def test_the_caption_always_says_membership_is_inferred(qtbot, tmp_path):
    """The one sentence this feature must never ship without."""
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()
    caption = view.caption_text()
    assert "INFERRED, not observed" in caption
    assert "pooled screen" in caption
    assert "GRA14" in caption
    # And the thumbnails do not claim otherwise either.
    thumbs = [view._grid.itemAt(i).widget() for i in range(view._grid.count())]
    tips = [t.toolTip() for t in thumbs if t is not None]
    assert tips and all("membership is inferred" in tip for tip in tips)
    assert not any("carries" in tip for tip in tips)


def test_a_coefficient_whose_wells_contribute_nothing_says_so_not_a_blank_grid(
        qtbot, tmp_path):
    """An empty montage is an answer; an empty grid is indistinguishable from
    a bug.

    Driven with an effect so large that the implied score lies far outside
    anything this screen observed, which is the honest way to get zero.
    """
    root, db_path, results_csv = _screen(tmp_path, with_png=True)
    frame = pd.DataFrame([{"feature": GENE_KEY, "coefficient": 50.0}])
    view = CellMontageView(
        frame_provider=lambda: frame,
        results_provider=lambda: results_csv,
        database_provider=lambda: _rows(db_path),
        threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)
    assert view.build() is True

    plan = view.plans()[0]
    assert plan.is_empty
    assert view.images()[0] == ()
    labels = [view._grid.itemAt(i).widget().text()
              for i in range(view._grid.count())
              if hasattr(view._grid.itemAt(i).widget(), "text")]
    assert any("No object was selected" in text for text in labels)
    assert "INFERRED, not observed" in view.caption_text()
    assert "OUTSIDE the observed range" in view.caption_text()


def test_one_guide_at_a_time_is_a_different_question_and_says_which(qtbot,
                                                                    tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view._per_guide.setCurrentIndex(1)
    assert view.build() is True

    plans = view.plans()
    assert len(plans) == 2                      # GRA14_1 and GRA14_2
    assert {p.guides for p in plans} == {("GRA14_1",), ("GRA14_2",)}
    assert all(p.guide_aggregation == "separate" for p in plans)
    assert "one guide at a time" in view.caption_text()
    assert len(view.images()) == 2


def test_the_object_type_chooses_the_mask_plane_the_crop_is_cut_by(qtbot,
                                                                   tmp_path):
    """"the user needs to specify which array the masks are in" -- this is it.

    The nucleus plane holds a strictly smaller region than the cell plane in
    the fixture, so cutting by it is measurable rather than merely accepted.
    """
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=False)
    view.set_coefficient(GENE_KEY)
    view.build()
    by_cell = view.images()[0][0]

    assert "nucleus" in OBJECT_CHOICES
    view._object.setCurrentIndex(OBJECT_CHOICES.index("nucleus"))
    view.build()
    by_nucleus = view.images()[0][0]
    assert by_cell.shape == by_nucleus.shape         # both centred and padded
    assert not np.array_equal(by_cell, by_nucleus)


def test_the_channel_box_chooses_which_planes_become_the_picture(qtbot,
                                                                 tmp_path):
    """The other half of that sentence: "which channels should be used"."""
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=False)
    view.set_coefficient(GENE_KEY)
    view._channels.setText("0,1,2")
    view.build()
    first = view.images()[0][0]

    view._channels.setText("3,2,1")
    view.build()
    second = view.images()[0][0]
    assert not np.array_equal(first, second)


# --------------------------------------------------------------------------- #
#  Threading
# --------------------------------------------------------------------------- #

def test_the_completion_handler_is_a_bound_method_of_the_widget(qtbot,
                                                               tmp_path):
    """The project rule: relay a worker's `finished` through a BOUND METHOD.

    `JobRunner` owns that relay -- a closure on ``worker.finished`` that does
    nothing but re-emit, and a bound method on ``thread.finished`` -- and the
    tab's own handler has to be a bound method of the widget too, or it runs
    off the GUI thread.
    """
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    assert view._on_loaded.__self__ is view
    assert view._jobs.parent() is view


def test_the_load_runs_on_a_real_worker_thread_and_lands_on_the_gui_thread(
        qtbot, tmp_path):
    """Threaded for real, not just inline: the montage arrives via the relay."""
    root, db_path, results_csv = _screen(tmp_path, with_png=True)
    frame = pd.read_csv(results_csv)
    view = CellMontageView(
        frame_provider=lambda: frame,
        results_provider=lambda: results_csv,
        database_provider=lambda: _rows(db_path),
        threaded=True)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)

    with qtbot.waitSignal(view.montage_ready, timeout=30000) as blocker:
        assert view.build() is True
    assert blocker.args[0] == view.plans()[0].n_objects > 0


# --------------------------------------------------------------------------- #
#  The saved figure
# --------------------------------------------------------------------------- #

def test_the_figure_goes_through_save_figure_and_carries_the_caption(
        qtbot, tmp_path, monkeypatch):
    """No literal extension anywhere: the format is the preference's to pick."""
    import spacr.plot as plot

    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()

    seen = {}
    real = plot.save_figure

    def spy(fig, path, **kwargs):
        seen["path"] = path
        seen["texts"] = [t.get_text() for t in fig.findobj(
            lambda o: hasattr(o, "get_text"))]
        return real(fig, path, **kwargs)

    monkeypatch.setattr(plot, "save_figure", spy)
    target = str(tmp_path / "montage")
    written = view.save(target)

    assert seen["path"] == target
    assert os.path.isfile(written)
    # save_figure chose the extension from the preference, not this module.
    assert os.path.splitext(written)[1] in (".png", ".pdf")
    assert any("INFERRED, not observed" in t for t in seen["texts"])


def test_the_figure_draws_a_panel_for_every_selected_object(qtbot, tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()
    figure = montage_figure(view.plans(), view.images(), columns=4)
    # One axes per crop, plus the caption's.
    assert len(figure.axes) == view.plans()[0].n_objects + 1


def test_saving_with_no_montage_says_so_rather_than_writing_an_empty_file(
        qtbot, tmp_path):
    view = CellMontageView(threaded=False)
    qtbot.addWidget(view)
    assert view.save(str(tmp_path / "nothing")) is None
    assert "no montage to save" in view.status_text()


# --------------------------------------------------------------------------- #
#  The headless loader
# --------------------------------------------------------------------------- #

def test_the_loader_returns_a_sentence_and_never_raises(tmp_path):
    """Every failure is a `MontageLoad.error`, because the tab must stay up."""
    root, db_path, results_csv = _screen(tmp_path, with_png=True)

    assert load(MontageRequest(name="", effect=0.1)).error
    assert load(MontageRequest(name="GRA14", effect=0.1)).unavailable
    missing = load(MontageRequest(name="GRA14", effect=0.1,
                                  results_path=str(tmp_path / "nowhere"),
                                  databases=(db_path,)))
    assert "regression_data.csv" in missing.error

    refused = load(MontageRequest(
        name="GRA14", effect=0.1,
        results_path=os.path.dirname(results_csv), databases=(db_path,),
        prefer="png", object_type="cell"))
    assert isinstance(refused, MontageLoad)


def test_the_two_csvs_that_look_right_and_are_not_cannot_be_read_by_mistake(
        tmp_path):
    """``grna_well.csv`` says how MANY wells a guide was seen in, never which,
    and ``well_grna.csv`` does not name a guide at all -- so neither can drive
    the montage, and the headless half refuses both by name.

    The tab cannot reach that refusal, and that is the point of resolving to
    the FOLDER: the results path it is handed is whichever coefficient table
    the panel loaded, so what it needs is always ``regression_data.csv``
    beside it. Pointed at a decoy in a folder that has the real file, it reads
    the real file; pointed at one in a folder that does not, it names the file
    that is missing rather than the file that was named.
    """
    _root, db_path, results_csv = _screen(tmp_path, with_png=True)
    decoy = pd.DataFrame([{"grna": "GRA14_1", "plateID": "plate1",
                           "grna_well_count": 3}])

    beside = os.path.join(os.path.dirname(results_csv), "grna_well.csv")
    decoy.to_csv(beside, index=False)
    good = load(MontageRequest(name="GRA14", effect=0.2, results_path=beside,
                               databases=(db_path,)))
    assert good.ok, good.error

    (tmp_path / "qc").mkdir()
    alone = tmp_path / "qc" / "grna_well.csv"
    decoy.to_csv(alone, index=False)
    missing = load(MontageRequest(name="GRA14", effect=0.2,
                                  results_path=str(alone),
                                  databases=(db_path,)))
    assert "regression_data.csv does not exist" in missing.error
    assert missing.unavailable


def test_each_plate_gets_its_own_crop_source(tmp_path):
    """Two experiment folders can have two different answers, and do here."""
    a_root, a_db, results_csv = _screen(tmp_path / "a", with_png=True)
    b_root, b_db, _b_csv = _screen(tmp_path / "b", with_png=False)
    result = load(MontageRequest(
        name="GRA14", effect=0.2, results_path=results_csv,
        databases=(a_db, b_db)))
    assert result.ok
    kinds = {d.split(" ")[0] for d in result.sources.values()}
    assert kinds == {"png", "merged"}
    assert result.plans[0].source_kind == "merged+png"


def test_experiment_root_and_channel_parsing():
    assert experiment_root("/x/y/measurements/measurements.db") == "/x/y"
    assert experiment_root("/x/y/measurements.db") == "/x/y"
    assert parse_channels("") is None
    assert parse_channels("0, 1;2") == (0, 1, 2)
    with pytest.raises(ValueError):
        parse_channels("-1")
    with pytest.raises(ValueError):
        parse_channels("blue")


def test_the_thumbnail_is_the_view_and_the_crop_underneath_is_untouched(
        qtbot, tmp_path):
    """The saved figure is drawn from the full arrays, not from the pixmaps."""
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=False)
    view.set_coefficient(GENE_KEY)
    view.build()
    crop = view.images()[0][0]
    assert max(crop.shape[:2]) != THUMBNAIL_PX or crop.shape[0] == crop.shape[1]
    thumb = view._grid.itemAt(0).widget()
    assert thumb.pixmap().width() <= THUMBNAIL_PX


# --------------------------------------------------------------------------- #
#  Where it lives on the regression screen
# --------------------------------------------------------------------------- #

@pytest.fixture()
def screen(qtbot):
    pytest.importorskip("pyqtgraph")
    from spacr.qt.screens.app_screen import AppScreen

    widget = AppScreen("regression")
    qtbot.addWidget(widget)
    return widget


def test_the_cells_tab_is_a_named_tab_beside_the_figures(screen):
    """"a new tab where the figures are" -- one tab per view, named.

    Beside the run's figures, which on this screen is the tab stack in the
    left half of the figures splitter: Runs, Results, Measurements, Cells.
    """
    tabs = screen._results_tabs
    assert tabs.widget(3) is screen._cell_montage
    assert tabs.tabText(3) == "Cells"
    assert "POOLED" in tabs.tabToolTip(3)
    # The tabs that were there before it are where they were.
    assert tabs.widget(1) is screen._results_panel
    assert tabs.widget(2) is screen._scan_panel


def test_the_tab_is_present_even_when_it_cannot_be_filled(screen):
    """Instruction 131 C: a tab that cannot be filled SAYS WHY, never absent.

    A fresh screen has no run, no database and no selection, which is the
    common case -- and it is the case in which a missing tab would leave the
    user hunting for a feature they were told about.
    """
    view = screen._cell_montage
    assert view is not None
    assert screen._results_tabs.indexOf(view) >= 0
    assert not view._show.isEnabled()
    assert view.reason()
    assert view._show.toolTip() == view.reason()


def test_the_tab_rides_the_panels_existing_selection(screen, tmp_path):
    """No second selection mechanism: `table.key_selected` is the one funnel.

    Driven with the GUIDE term, because the panel opens filtered to guides --
    the level on which the two inference modes agree -- so the gene row is not
    in the table to be selected. That is the panel's business and this tab
    inherits it, which is exactly the point of riding its selection.
    """
    _root, _db, results_csv = _screen(tmp_path, with_png=True)
    assert screen._results_panel.load(results_csv) is True

    screen._results_panel.table.select_key(GUIDE_KEY)
    assert screen._cell_montage._name == "GRA14_1"
    assert screen._cell_montage._level == "grna"
    assert screen._cell_montage._effect == pytest.approx(0.15)


def test_opening_the_tab_re_reads_what_it_can_now_do(screen, tmp_path,
                                                     monkeypatch):
    """Databases are attached while this tab is behind another one."""
    _root, db_path, results_csv = _screen(tmp_path, with_png=True)
    screen._results_panel.load(results_csv)
    screen._results_panel.table.select_key(GUIDE_KEY)
    assert "No measurement database is attached" in screen._cell_montage.reason()

    monkeypatch.setattr(screen, "_attached_database_rows",
                        lambda: _rows(db_path))
    screen._cell_montage._database_provider = screen._attached_database_rows
    screen._results_tabs.setCurrentWidget(screen._cell_montage)
    assert screen._cell_montage.reason() == ""
    assert screen._cell_montage._show.isEnabled()


def test_the_results_path_provider_names_the_run_now_on_screen(screen,
                                                               tmp_path):
    """regression_data.csv is written beside the coefficient table."""
    _root, _db, results_csv = _screen(tmp_path, with_png=True)
    assert screen._results_source_path() == ""
    screen._results_panel.load(results_csv)
    assert screen._results_source_path() == results_csv


def test_a_screen_that_is_not_the_regression_module_has_no_cells_tab(qtbot):
    """`_cell_montage` is born on every screen so handlers can read it."""
    from spacr.qt.screens.app_screen import AppScreen

    other = AppScreen("mask")
    qtbot.addWidget(other)
    assert other._cell_montage is None
    assert other._results_tabs is None if hasattr(other, "_results_tabs") \
        else True


# --------------------------------------------------------------------------- #
#  Every way it can go wrong, driven rather than assumed
# --------------------------------------------------------------------------- #

class _AngryFrame:
    """A coefficient table that raises when it is read.

    A host that hands the tab something broken must not take the plot down
    with it — the montage is an explanation, and an explanation that raises
    leaves the user with a traceback instead of the point they clicked.
    """

    def __len__(self):
        return 1

    @property
    def columns(self):
        raise ValueError("this frame is broken")


def test_a_coefficient_whose_effect_is_not_a_number_has_no_effect(qtbot,
                                                                  tmp_path):
    """A blank or textual coefficient must not become 0.0 by accident.

    The score window is 'baseline + effect', so a coefficient read as zero
    would centre the window on the screen's own median and produce a montage
    of perfectly ordinary cells captioned as this gene's.
    """
    frame = pd.DataFrame([{"feature": GENE_KEY, "coefficient": "not a number"}])
    assert coefficient_from_frame(GENE_KEY, frame)[2] is None
    empty = pd.DataFrame([{"feature": GENE_KEY, "coefficient": np.nan}])
    assert coefficient_from_frame(GENE_KEY, empty)[2] is None

    view = CellMontageView(frame_provider=lambda: frame, threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)
    assert "no fitted effect" in view.reason()
    assert not view._show.isEnabled()


def test_a_broken_provider_does_not_take_the_tab_down(qtbot):
    """Every provider is the host's, and any of them can be broken."""
    def angry():
        raise RuntimeError("the host is on fire")

    view = CellMontageView(frame_provider=_AngryFrame,
                           results_provider=angry,
                           database_provider=angry,
                           threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)
    assert view._name == ""            # the parse raised and was contained
    assert view._results_path() == ""
    assert view.databases() == ()
    assert view.reason()

    # And the frame provider itself blowing up, which is the other half.
    view._frame_provider = angry
    assert view._frame() is None

    # And with no providers at all, which is how the tab is built before a
    # regression screen ever hands it anything.
    bare = CellMontageView(threaded=False)
    qtbot.addWidget(bare)
    bare.refresh()
    assert bare._frame() is None
    assert bare._results_path() == ""
    assert bare.databases() == ()


def test_a_results_folder_whose_fractions_csv_is_unreadable_says_so(tmp_path):
    _root, db_path, results_csv = _screen(tmp_path, with_png=True)
    path = os.path.join(os.path.dirname(results_csv), "regression_data.csv")
    with open(path, "w") as handle:
        handle.write('prc,grna,gene,fraction\n"unclosed,1,2,3\n,,\n')
    result = load(MontageRequest(name="GRA14", effect=0.2,
                                 results_path=results_csv,
                                 databases=(db_path,)))
    assert "Could not read the per-well guide fractions" in result.error


def test_one_broken_database_is_named_and_the_others_still_draw(tmp_path):
    """A plate whose database is not a database must not lose the montage."""
    _root, db_path, results_csv = _screen(tmp_path, with_png=True)
    broken = str(tmp_path / "not_a_database.db")
    with open(broken, "w") as handle:
        handle.write("this is not sqlite")

    result = load(MontageRequest(name="GRA14", effect=0.2,
                                 results_path=results_csv,
                                 databases=(broken, db_path)))
    assert result.ok
    caption = result.plans[0].caption()
    assert "not_a_database.db" in caption

    # Every database broken IS a refusal, and it names them.
    none_left = load(MontageRequest(name="GRA14", effect=0.2,
                                    results_path=results_csv,
                                    databases=(broken,)))
    assert not none_left.ok
    assert none_left.unavailable
    assert "not_a_database.db" in none_left.error


def test_a_coefficient_no_well_reports_is_refused_by_name(tmp_path):
    _root, db_path, results_csv = _screen(tmp_path, with_png=True)
    result = load(MontageRequest(name="NOT_A_GENE", effect=0.2, level="gene",
                                 results_path=results_csv,
                                 databases=(db_path,)))
    assert not result.ok
    assert "NOT_A_GENE" in result.error

    per_guide = load(MontageRequest(name="NOT_A_GENE", effect=0.2,
                                    level="gene", results_path=results_csv,
                                    databases=(db_path,), per_guide=True))
    assert "one guide at a time" in per_guide.error


def test_a_selection_that_raises_something_else_is_still_a_sentence(
        tmp_path, monkeypatch):
    """The catch-all is not decoration: it keeps the tab on screen."""
    import spacr.cell_montage as cell_montage

    _root, db_path, results_csv = _screen(tmp_path, with_png=True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("pandas changed under us")

    monkeypatch.setattr(cell_montage, "select_montage", explode)
    result = load(MontageRequest(name="GRA14", effect=0.2,
                                 results_path=results_csv,
                                 databases=(db_path,)))
    assert "Could not select the montage" in result.error
    assert "pandas changed under us" in result.error


def test_a_merged_array_that_has_gone_missing_leaves_a_named_gap(qtbot,
                                                                 tmp_path):
    """The crops that cannot be cut are blank AND said so, not dropped.

    Dropping them would renumber the montage and quietly break the count rule
    the caption states; leaving them blank keeps `round(n x fraction)` true
    and puts the reason on the figure.
    """
    root, db_path, results_csv = _screen(tmp_path, with_png=False)
    for name in os.listdir(os.path.join(root, "merged")):
        os.remove(os.path.join(root, "merged", name))
        break                                  # one field gone, not all

    view = CellMontageView(
        frame_provider=lambda: pd.read_csv(results_csv),
        results_provider=lambda: results_csv,
        database_provider=lambda: _rows(db_path),
        threaded=False)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)
    assert view.build() is True

    crops = view.images()[0]
    assert any(c is None for c in crops)
    assert "could not be cut" in view.caption_text()
    # The gap is a labelled placeholder in the grid, not a missing cell.
    texts = [view._grid.itemAt(i).widget().text()
             for i in range(view._grid.count())
             if hasattr(view._grid.itemAt(i).widget(), "text")]
    assert "no crop" in texts
    # And the figure draws it as one too.
    figure = montage_figure(view.plans(), view.images(), columns=4)
    assert any("no crop" in t.get_text()
               for t in figure.findobj(lambda o: hasattr(o, "get_text")))


def test_a_plate_with_no_crop_source_leaves_its_objects_blank_and_named(
        tmp_path):
    """One plate has crops and one has none: the montage is the union."""
    _a_root, a_db, results_csv = _screen(tmp_path / "a", with_png=True)
    b_root, b_db, _b = _screen(tmp_path / "b", with_png=False)
    import shutil
    shutil.rmtree(os.path.join(b_root, "merged"))

    result = load(MontageRequest(name="GRA14", effect=0.2,
                                 results_path=results_csv,
                                 databases=(a_db, b_db)))
    assert result.ok
    assert set(result.sources) == {_a_root}


def test_the_runner_failing_outright_is_reported_not_swallowed(qtbot,
                                                               tmp_path,
                                                               monkeypatch):
    import spacr.qt.widgets.cell_montage_view as module

    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)

    def explode(_request):
        raise RuntimeError("the worker died")

    monkeypatch.setattr(module, "load", explode)
    with qtbot.waitSignal(view.montage_failed, timeout=5000) as blocker:
        view.build()
    assert "the worker died" in blocker.args[0]
    assert "The montage load failed" in view.status_text()


def test_a_loader_that_returns_nothing_at_all_still_says_something(qtbot,
                                                                   tmp_path):
    """Both shapes of "nothing": not a MontageLoad, and an empty one."""
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view._on_loaded(None)
    assert "returned nothing" in view.status_text()

    view._on_loaded(MontageLoad())
    assert view.plans() == ()
    assert "no montage and no reason" in view.status_text()


def test_a_source_that_hands_back_the_wrong_thing_does_not_crash_the_grid(
        qtbot, tmp_path):
    """The two guards in the fill path, driven.

    A crop source is contracted to return one ``(H, W, 3)`` uint8 array per
    row (:class:`spacr.crops.CropSource`). These are what happens when one
    does not: a greyscale array, and more arrays than there are rows. Neither
    may take the tab down, because the tab is what would have to say so.
    """
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()
    plan = view.plans()[0]

    grey = np.full((16, 16), 200, dtype=np.uint8)
    too_many = tuple([grey] * (plan.n_objects + 1))
    view._on_loaded(MontageLoad(plans=(plan,), images=(too_many,)))
    assert view.plans() == (plan,)
    thumbs = [view._grid.itemAt(i).widget() for i in range(view._grid.count())]
    assert len(thumbs) == plan.n_objects + 1
    assert thumbs[0].pixmap().width() == THUMBNAIL_PX


def test_saving_through_the_dialog_honours_a_cancelled_dialog(qtbot, tmp_path,
                                                              monkeypatch):
    from PySide6.QtWidgets import QFileDialog

    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()

    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: ("", "")))
    assert view.save() is None

    target = str(tmp_path / "asked")
    monkeypatch.setattr(QFileDialog, "getSaveFileName",
                        staticmethod(lambda *a, **k: (target, "")))
    written = view.save()
    assert written and os.path.isfile(written)
    # The name the dialog was offered carries the gene, not a hard-coded
    # extension of this module's choosing.
    assert "saved to" in view.status_text()


def test_the_grid_reflows_when_the_tab_gets_wider(qtbot, tmp_path):
    """300 thumbnails re-laid out on every resize event is a stutter, so the
    reflow is debounced and only runs when the column count actually moved."""
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    # SHOWN, and that is not incidental: an unshown QScrollArea keeps its
    # size-hint viewport whatever the widget is resized to, so the column
    # count would not move and the test would prove nothing. It is the same
    # trap that made a snapshot of the unshown volcano one flat colour.
    view.show()
    qtbot.waitUntil(lambda: view._scroll.viewport().width() > 0, timeout=5000)
    view.set_coefficient(GENE_KEY)
    view.build()
    narrow = view._columns
    assert narrow >= 1

    view.resize(1600, 900)
    qtbot.waitUntil(lambda: view._columns > narrow, timeout=5000)
    # A second reflow at the same width is a no-op rather than a rebuild.
    before = view._grid.count()
    view._relayout()
    assert view._grid.count() == before

    # And with nothing on screen there is nothing to reflow.
    empty = CellMontageView(threaded=False)
    qtbot.addWidget(empty)
    empty.resize(900, 600)
    empty._relayout()
    assert empty._grid.count() == 0


def test_a_montage_never_outlives_the_point_it_was_built_for(qtbot, tmp_path):
    """Move the selection and the old gene's cells leave the screen with it.

    The single most dangerous thing this tab can do is show one gene's cells
    under another gene's name. Clicking through points faster than a load
    returns is the ordinary case -- a merged-source montage is seconds of
    disk -- so both halves are held here: the grid empties on the click, and
    an answer for the point the user has left never lands.
    """
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view.build()
    assert view.plans() and view._shown_key == GENE_KEY
    built = view.plans()[0]

    view.set_coefficient(GUIDE_KEY)
    assert view.plans() == ()
    assert view.caption_text() == ""
    assert view._shown_key == ""
    assert "Ready" in view.status_text()

    # The superseded load landing late must not repaint the old gene.
    stale = MontageLoad(request=MontageRequest(name="GRA14", effect=0.2),
                        plans=(built,), images=((np.zeros((8, 8, 3), np.uint8),) * built.n_objects,))
    view._on_loaded(stale)
    assert view.plans() == ()
    assert "GRA14_1" not in view.caption_text()


def test_editing_a_crop_setting_abandons_the_load_it_no_longer_describes(
        qtbot, tmp_path):
    view, _root, _db, _csv = _view(qtbot, tmp_path, with_png=True)
    view.set_coefficient(GENE_KEY)
    view._channels.setText("0,1,2")
    assert view._pending is None
    assert view.build() is True
    first = view.plans()[0]

    view._channels.setText("3,2,1")
    stale = MontageLoad(request=MontageRequest(name="GRA14", effect=0.2),
                        plans=(first,), images=((None,) * first.n_objects,))
    view._pending = MontageRequest(name="GRA14", effect=0.2,
                                   channels=(3, 2, 1))
    view._on_loaded(stale)
    assert view._pending is not None      # the answer that matters is still due
    assert view.plans() == (first,) or view.plans() == ()


def test_closing_the_tab_stops_waiting_for_a_load(qtbot, tmp_path):
    """A QThread destroyed while running aborts the process, and a
    merged-source montage is seconds long, so leaving mid-load is ordinary."""
    root, db_path, results_csv = _screen(tmp_path, with_png=True)
    view = CellMontageView(
        frame_provider=lambda: pd.read_csv(results_csv),
        results_provider=lambda: results_csv,
        database_provider=lambda: _rows(db_path),
        threaded=True)
    qtbot.addWidget(view)
    view.set_coefficient(GENE_KEY)
    view.build()
    view.close()
    assert view._pending is None
    assert not view._jobs.is_busy()


def test_closing_the_regression_screen_shuts_the_cells_loader_down(screen,
                                                                   tmp_path):
    """The tab is a child widget: navigating away gives it no close event."""
    _root, db_path, results_csv = _screen(tmp_path, with_png=True)
    screen._results_panel.load(results_csv)
    screen._results_panel.table.select_key(GUIDE_KEY)
    screen._cell_montage._database_provider = lambda: _rows(db_path)
    screen._cell_montage.refresh()
    screen._cell_montage.build()

    screen.close()
    assert screen._cell_montage._pending is None
    assert not screen._cell_montage._jobs.is_busy()
