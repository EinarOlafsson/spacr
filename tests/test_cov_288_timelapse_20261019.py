"""Timepose tracking in timelapse: input validation, transient removal, plotting, empty frames.

A fake adapter stands in for the model; no GPU, token or network is used.
"""
from __future__ import annotations

import inspect
import re

import numpy as np
import pandas as pd
import pytest

from spacr import model_zoo
from spacr import timelapse as tl


class Adapter:
    def __init__(self, masks, edges):
        self.masks, self.edges, self.calls = masks, edges, []

    def segment_track_movie(self, model, images, invert=False, masks=None):
        self.calls.append(np.asarray(images))
        return {"masks": self.masks, "edges": self.edges}


def _install(monkeypatch, adapter):
    monkeypatch.setattr(model_zoo, "_load_timepose",
                        lambda key, device=None: (adapter, object()))


def _blocks():
    """Object 1 present in all four frames; object 2 only in frames 0 and 1."""
    masks = np.zeros((4, 24, 24), np.int32)
    for t in range(4):
        masks[t, 2:6, 2:6] = 1
    masks[0, 14:18, 14:18] = 2
    masks[1, 14:18, 14:18] = 2
    edges = [[0, 1, 1], [1, 1, 1], [2, 1, 1], [0, 2, 2]]
    return masks, edges


def _scn_images_move_to_channels_first_only_when_channels_are_last():
    _obs = []
    last = np.zeros((3, 16, 16, 2))
    assert tl._timepose_images(last).shape == (3, 2, 16, 16)
    _obs.append(_seen((tl._timepose_images(last).shape), (tl._timepose_images(last).shape == (3, 2, 16, 16))))
    first = np.zeros((3, 2, 16, 16))
    assert tl._timepose_images(first).shape == (3, 2, 16, 16)
    _obs.append(_seen((tl._timepose_images(first).shape), (tl._timepose_images(first).shape == (3, 2, 16, 16))))
    plain = np.zeros((3, 16, 16))
    assert tl._timepose_images(plain).shape == (3, 16, 16)
    _obs.append(_seen((tl._timepose_images(plain).shape), (tl._timepose_images(plain).shape == (3, 16, 16))))
    return _obs


def _scn_a_frame_without_objects_starts_no_track_and_the_next_frame_carries_on():
    _obs = []
    masks = np.zeros((3, 10, 10), np.int32)
    masks[0, 1:4, 1:4] = 1
    masks[2, 1:4, 1:4] = 1
    tracked, parents = tl._edges_to_tracked_stack(masks, [[0, 1, 1]])
    assert tracked[1].max() == 0
    _obs.append(_seen((tracked[1].max()), (tracked[1].max() == 0)))
    assert tracked[0].max() == 1
    _obs.append(_seen((tracked[0].max()), (tracked[0].max() == 1)))
    assert tracked[2].max() == 2
    _obs.append(_seen((tracked[2].max()), (tracked[2].max() == 2)))
    assert parents == {}
    _obs.append(_seen((parents), (parents == {})))
    return _obs


def _scn_a_mask_stack_that_is_not_three_dimensional_is_refused(tmp_path, monkeypatch, capsys):
    _obs = []
    with pytest.raises(ValueError) as caught:
        tl._timepose_track_cells(src=str(tmp_path / "m"), name="f", batch_filenames=None,
                                 object_type="cell", masks=np.zeros((8, 8), np.int32),
                                 images=np.zeros((8, 8)))
    _obs.append(_seen(type(caught.value).__name__ + ": " + str(caught.value), True))
    assert "(8, 8)" in str(caught.value)
    _obs.append(_seen(("(8, 8)" in str(caught.value)), ("(8, 8)" in str(caught.value))))
    return _obs


def _scn_a_single_frame_has_nothing_to_link_and_never_loads_a_model(tmp_path, monkeypatch, capsys):
    _obs = []
    def boom(*args, **kwargs):
        raise AssertionError("the model must not be loaded")
    monkeypatch.setattr(model_zoo, "_load_timepose", boom)
    masks = np.zeros((1, 8, 8), np.int32)
    masks[0, 2:4, 2:4] = 1
    stack = tl._timepose_track_cells(src=str(tmp_path / "m"), name="f", batch_filenames=None,
                                     object_type="nucleus", masks=masks, images=None)
    assert np.asarray(stack).shape == masks.shape
    _obs.append(_seen((np.asarray(stack).shape), (np.asarray(stack).shape == masks.shape)))
    assert np.asarray(stack).max() == 1
    _obs.append(_seen((np.asarray(stack).max()), (np.asarray(stack).max() == 1)))
    assert "nucleus" in capsys.readouterr().out
    _obs.append(_seen(("nucleus" in capsys.readouterr().out), ("nucleus" in capsys.readouterr().out)))
    assert not (tmp_path / "tracks").exists()
    _obs.append(_seen((not (tmp_path / "tracks").exists()), (not (tmp_path / "tracks").exists())))
    return _obs


def _scn_images_that_do_not_match_the_masks_are_refused(tmp_path, monkeypatch, capsys):
    _obs = []
    masks, _ = _blocks()
    for images in (np.zeros((3, 24, 24)), np.zeros((4, 20, 24))):
        with pytest.raises(ValueError) as caught:
            tl._timepose_track_cells(src=str(tmp_path / "m"), name="f",
                                     batch_filenames=None, object_type="cell",
                                     masks=masks, images=images)
        _obs.append(_seen(type(caught.value).__name__ + ": " + str(caught.value), True))
        assert "does not match" in str(caught.value)
        _obs.append(_seen(("does not match" in str(caught.value)), ("does not match" in str(caught.value))))
    return _obs


def _scn_transient_objects_are_dropped_from_the_table_and_the_masks(tmp_path, monkeypatch, capsys):
    _obs = []
    masks, edges = _blocks()
    _install(monkeypatch, Adapter(masks, edges))
    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    stack = tl._timepose_track_cells(src=str(src), name="g", batch_filenames=None,
                                     object_type="cell", masks=masks,
                                     images=np.ones((4, 24, 24), np.float32),
                                     timelapse_remove_transient=True)
    table = pd.read_csv(tmp_path / "run" / "tracks" / "timepose_tracks_cell_g.csv")
    assert table["track_id"].nunique() == 1
    _obs.append(_seen((table["track_id"].nunique()), (table["track_id"].nunique() == 1)))
    assert table.groupby("track_id")["frame"].nunique().iloc[0] == 4
    _obs.append(_seen((table.groupby("track_id")["frame"].nunique().iloc[0]), (table.groupby("track_id")["frame"].nunique().iloc[0] == 4)))
    assert "Removed 2 objects" in capsys.readouterr().out
    _obs.append(_seen(("Removed 2 objects" in capsys.readouterr().out), ("Removed 2 objects" in capsys.readouterr().out)))
    out = np.asarray(stack)
    assert len(np.unique(out[out > 0])) == 1
    _obs.append(_seen((len(np.unique(out[out > 0]))), (len(np.unique(out[out > 0])) == 1)))
    assert out[0, 14:18, 14:18].max() == 0
    _obs.append(_seen((out[0, 14:18, 14:18].max()), (out[0, 14:18, 14:18].max() == 0)))
    return _obs


def _scn_plot_and_save_hand_the_tracked_stack_to_the_track_visualiser(tmp_path, monkeypatch, capsys):
    _obs = []
    masks, edges = _blocks()
    _install(monkeypatch, Adapter(masks, edges))
    seen = []
    import spacr.plot as plot_module
    monkeypatch.setattr(plot_module, "_visualize_and_save_timelapse_stack_with_tracks",
                        lambda *args: seen.append(args))
    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    names = [f"f{t}.tif" for t in range(4)]
    tl._timepose_track_cells(src=str(src), name="g", batch_filenames=names,
                             object_type="cell", masks=masks,
                             images=np.ones((4, 24, 24), np.float32),
                             plot=True, save=True)
    assert len(seen) == 1
    _obs.append(_seen((len(seen)), (len(seen) == 1)))
    stack, table, save, got_src, name, plot, filenames, object_type, mode = seen[0]
    assert stack.shape == masks.shape
    _obs.append(_seen((stack.shape), (stack.shape == masks.shape)))
    assert isinstance(table, pd.DataFrame) and not table.empty
    _obs.append(_seen((isinstance(table, pd.DataFrame) and not table.empty), (isinstance(table, pd.DataFrame) and not table.empty)))
    assert (save, plot, name, object_type, mode) == (True, True, "g", "cell", "timepose")
    _obs.append(_seen(((save, plot, name, object_type, mode)), ((save, plot, name, object_type, mode) == (True, True, "g", "cell", "timepose"))))
    assert filenames == names and got_src == str(src)
    _obs.append(_seen((filenames == names and got_src == str(src)), (filenames == names and got_src == str(src))))
    return _obs


def _scn_an_empty_result_writes_an_empty_table_without_lineage_columns(tmp_path, monkeypatch, capsys):
    _obs = []
    masks = np.zeros((3, 12, 12), np.int32)
    _install(monkeypatch, Adapter(masks, []))
    src = tmp_path / "run" / "masks"
    src.mkdir(parents=True)
    stack = tl._timepose_track_cells(src=str(src), name="e", batch_filenames=None,
                                     object_type="cell", masks=masks,
                                     images=np.ones((3, 12, 12), np.float32),
                                     timelapse_remove_transient=True)
    assert np.asarray(stack).max() == 0
    _obs.append(_seen((np.asarray(stack).max()), (np.asarray(stack).max() == 0)))
    path = tmp_path / "run" / "tracks" / "timepose_tracks_cell_e.csv"
    assert path.exists()
    _obs.append(_seen((path.exists()), (path.exists())))
    assert "parent_track_id" not in pd.read_csv(path).columns
    _obs.append(_seen(("parent_track_id" not in pd.read_csv(path).columns), ("parent_track_id" not in pd.read_csv(path).columns)))
    return _obs


def _plain(value):
    """Return a plain comparable copy of a value, or raise ValueError."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        return re.sub(r"/tmp/pytest-of-[^/]+/pytest-\d+/[^/\s]+", "<tmp>", value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return None if value != value else round(value, 6)
    if isinstance(value, (list, tuple)) and len(value) <= 60:
        return [_plain(v) for v in value]
    if isinstance(value, (set, frozenset)) and len(value) <= 60:
        return sorted((_plain(v) for v in value), key=repr)
    if isinstance(value, dict) and len(value) <= 60:
        return {str(k): _plain(v) for k, v in value.items()}
    raise ValueError(type(value))


def _seen(value, held):
    """Observed value of one check: the value itself when plain, else whether the check held."""
    try:
        return _plain(value)
    except ValueError:
        return bool(held)


def _run_scenario(request, scenario, kwargs):
    """Run one scenario function, resolving its fixtures by name."""
    wanted = [n for n in inspect.signature(scenario).parameters if n not in kwargs]
    fixtures = {n: request.getfixturevalue(n) for n in wanted}
    return scenario(**fixtures, **kwargs)


@pytest.mark.parametrize("scenario, kwargs, expected", [
    pytest.param(_scn_images_move_to_channels_first_only_when_channels_are_last, {}, [[3, 2, 16, 16], [3, 2, 16, 16], [3, 16, 16]],
                 id="images-go-channels-first-only-when-channels-are-last"),
    pytest.param(_scn_a_frame_without_objects_starts_no_track_and_the_next_frame_carries_on, {}, [0, 1, 2, {}],
                 id="empty-frame-starts-no-track-and-next-frame-carries-on"),
])
def test_timepose_helpers_prepare_images_and_link_edges_into_tracked_stacks(request, scenario, kwargs, expected):
    observed = _run_scenario(request, scenario, kwargs)
    assert observed == expected


@pytest.mark.parametrize("scenario, kwargs, expected", [
    pytest.param(_scn_a_mask_stack_that_is_not_three_dimensional_is_refused, {}, ['ValueError: Timepose needs a (T, Y, X) mask stack, got shape (8, 8).', True],
                 id="non-3d-mask-stack-refused"),
    pytest.param(_scn_a_single_frame_has_nothing_to_link_and_never_loads_a_model, {}, [[1, 8, 8], 1, False, True],
                 id="single-frame-skips-the-model"),
    pytest.param(_scn_images_that_do_not_match_the_masks_are_refused, {}, ['ValueError: images shape (3, 24, 24) does not match masks shape (4, 24, 24).', True, 'ValueError: images shape (4, 20, 24) does not match masks shape (4, 24, 24).', True],
                 id="mismatched-images-refused"),
    pytest.param(_scn_transient_objects_are_dropped_from_the_table_and_the_masks, {}, [1, 4, False, 1, 0],
                 id="transient-objects-dropped"),
    pytest.param(_scn_plot_and_save_hand_the_tracked_stack_to_the_track_visualiser, {}, [1, [4, 24, 24], True, [True, True, 'g', 'cell', 'timepose'], True],
                 id="plot-and-save-call-the-visualiser"),
    pytest.param(_scn_an_empty_result_writes_an_empty_table_without_lineage_columns, {}, [0, True, True],
                 id="empty-result-writes-empty-table"),
])
def test_timepose_tracking_validates_inputs_writes_tracks_and_drops_transients(request, scenario, kwargs, expected):
    observed = _run_scenario(request, scenario, kwargs)
    assert observed == expected
