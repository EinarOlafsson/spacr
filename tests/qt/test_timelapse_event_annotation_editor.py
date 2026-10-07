"""An alpha annotation editor exports actual tracked observations for F567."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import tifffile
from PySide6.QtCore import QSettings, Qt

from spacr.qt.widgets.timelapse_preview import (
    _EventAnnotationDialog, _annotation_field_payload, render_frame, track_colour,
)
from spacr.tabular import write_table
from spacr.timelapse import _event_read_annotations


@pytest.fixture
def prefs(tmp_path, monkeypatch):
    """Keep the alpha switch in an isolated per-test settings file."""
    from spacr.qt import preferences

    path = tmp_path / "alpha.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    return preferences


def _field_files(tmp_path):
    """Write real TIFF pages and the on-disk tracker identity they belong to."""
    sequence = tmp_path / "field.tif"
    with tifffile.TiffWriter(sequence) as writer:
        for frame in range(3):
            image = np.zeros((24, 24), dtype=np.uint16)
            image[7 + frame:12 + frame, 8:13] = 100 + frame
            writer.write(image)
    tracks = tmp_path / "tracks" / "trackpy_tracks_cell_field.csv"
    write_table(pd.DataFrame([
        {"track_id": 7, "frame": frame, "x": 10, "y": 9 + frame}
        for frame in range(3)] + [
        {"track_id": 9, "frame": frame, "x": 16, "y": 15}
        for frame in (1, 2)]), tracks, canonicalise=False)
    target = tmp_path / "tracks" / "events" / "annotations.csv"
    return sequence, tracks, target


def _dialog(qtbot, sequence, tracks, target):
    """Open the editor synchronously while retaining its real lazy sequence."""
    dialog = _EventAnnotationDialog(threaded=False)
    qtbot.addWidget(dialog)
    dialog._sequence_path.setText(str(sequence))
    dialog._tracks_path.setText(str(tracks))
    dialog._output_path.setText(str(target))
    dialog._load_field()
    assert dialog._field is not None, dialog._status.text()
    return dialog


def test_real_tracked_frames_round_trip_through_detector_reader(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    assert dialog._field["sequence"].read_count == 1
    assert len(dialog._field["sequence"]) == 3
    dialog._track.setCurrentIndex(dialog._track.findData(7))
    dialog._frame.setValue(1)
    dialog._event.setText("Mitosis")
    dialog._add_event()
    assert dialog._events == []
    assert not target.exists()

    dialog._confirm.setChecked(True)
    dialog._add_event()
    assert dialog._events == [{"track_id": 7, "frame": 1, "event": "mitosis"}]
    dialog._event.setText("mitosis")
    dialog._add_event()
    assert len(dialog._events) == 1
    dialog._track.setCurrentIndex(dialog._track.findData(9))
    dialog._frame.setValue(0)
    dialog._event.setText("egress")
    dialog._add_event()
    assert len(dialog._events) == 1
    dialog._frame.setValue(2)
    dialog._add_event()
    assert len(dialog._events) == 2
    dialog._save()
    assert dialog._saved_path == str(target)
    actual = _event_read_annotations(str(target))
    assert actual[["field", "track_id", "frame", "event", "object"]].to_dict(
        "records") == [
            {"field": "field", "track_id": 7, "frame": 1,
             "event": "mitosis", "object": "cell"},
            {"field": "field", "track_id": 9, "frame": 2,
             "event": "egress", "object": "cell"},
        ]
    assert dialog._field["sequence"].read_count <= 3
    dialog.close()


def test_cancel_does_not_publish_staged_annotations(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("death")
    dialog._add_event()
    assert len(dialog._events) == 1
    dialog.reject()
    assert not target.exists()
    assert dialog._field is None
    assert dialog._preview.pixmap().isNull()


def test_sorted_event_rows_edit_and_remove_the_selected_observation(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._events = [
        {"track_id": 7, "frame": 1, "event": "mitosis"},
        {"track_id": 9, "frame": 2, "event": "death"},
    ]
    dialog._refresh_rows()
    qtbot.wait(25)
    dialog._rows.sortItems(0, Qt.DescendingOrder)
    qtbot.wait(25)
    assert dialog._rows.horizontalHeader().sortIndicatorSection() == 0
    assert dialog._rows.item(0, 0).text() == "9"
    assert dialog._rows.item(0, 1).text() == "2"
    assert dialog._rows.item(0, 2).text() == "death"
    assert dialog._rows.item(0, 0).data(Qt.UserRole) == 1
    dialog._rows.selectRow(0)
    assert dialog._track.currentData() == 9
    assert dialog._frame.value() == 2
    dialog._event.setText("egress")
    dialog._add_event()
    qtbot.wait(25)
    assert dialog._events == [
        {"track_id": 7, "frame": 1, "event": "mitosis"},
        {"track_id": 9, "frame": 2, "event": "egress", "extra": {}},
    ]
    assert dialog._rows.item(0, 0).text() == "9"
    assert dialog._rows.item(0, 1).text() == "2"
    assert dialog._rows.item(0, 2).text() == "egress"
    assert dialog._rows.item(0, 0).data(Qt.UserRole) == 1
    dialog._rows.selectRow(0)
    dialog._remove_event()
    assert dialog._events == [{"track_id": 7, "frame": 1, "event": "mitosis"}]
    dialog._save()
    assert _event_read_annotations(str(target))["event"].tolist() == ["mitosis"]
    dialog.close()


def test_existing_other_field_is_preserved_and_external_change_is_refused(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    write_table(pd.DataFrame([
        {"field": "other", "track_id": 3, "frame": 0,
         "event": "invasion", "object": "pathogen", "reviewer": "A"},
        {"field": "field", "track_id": 7, "frame": 2,
         "event": "death", "object": "pathogen", "reviewer": "B"},
        {"field": "field", "track_id": 7, "frame": 1,
         "event": "mitosis", "object": "cell", "reviewer": "C"},
    ]), target, canonicalise=False)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._rows.selectRow(0)
    dialog._event.setText("egress")
    dialog._add_event()
    dialog._save()
    actual = _event_read_annotations(str(target))
    assert set(zip(actual["field"], actual["event"])) == {
        ("other", "invasion"), ("field", "death"), ("field", "egress")}
    assert dict(zip(actual["event"], actual["reviewer"])) == {
        "invasion": "A", "death": "B", "egress": "C"}

    changed = target.read_bytes() + b"\n"
    target.write_bytes(changed)
    dialog._event.setText("mitosis")
    dialog._add_event()
    dialog._save()
    assert target.read_bytes() == changed
    assert "changed" in dialog._status.text().lower()
    dialog.close()


def test_source_change_after_confirmation_cannot_publish_stale_track_ids(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()
    tracks.write_bytes(tracks.read_bytes() + b"\n")
    dialog._save()
    assert not target.exists()
    assert "changed" in dialog._status.text().lower()
    dialog._sequence_path.setText(str(tmp_path / "different.tif"))
    assert not dialog._confirm.isChecked()
    dialog.close()


def test_selected_event_can_be_removed_before_atomic_save(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()
    dialog._rows.selectRow(0)
    dialog._remove_event()
    assert dialog._events == []
    dialog._save()
    assert target.exists()
    assert _event_read_annotations(str(target)).empty
    dialog.close()


def test_later_tracked_frame_is_available_without_loading_the_whole_movie(
        qtbot, tmp_path):
    sequence = tmp_path / "long.tif"
    with tifffile.TiffWriter(sequence) as writer:
        for frame in range(15):
            writer.write(np.full((24, 24), frame, dtype=np.uint16))
    tracks = tmp_path / "tracks" / "trackpy_tracks_cell_long.csv"
    write_table(pd.DataFrame([{"track_id": 5, "frame": frame,
                              "x": 10, "y": 10} for frame in range(15)]),
                tracks, canonicalise=False)
    target = tmp_path / "annotations.csv"
    dialog = _dialog(qtbot, sequence, tracks, target)
    assert len(dialog._field["sequence"]) == 15
    assert dialog._field["sequence"].read_count == 1
    dialog._confirm.setChecked(True)
    dialog._frame.setValue(14)
    dialog._event.setText("mitosis")
    dialog._add_event()
    dialog._save()
    assert _event_read_annotations(str(target))["frame"].tolist() == [14]
    for frame in range(15):
        dialog._frame.setValue(frame)
    assert len(dialog._field["sequence"]._cache) <= 6
    dialog.close()


@pytest.mark.parametrize("bad_rows", [
    [{"track_id": 7, "frame": 0}, {"track_id": 7, "frame": 0}],
    [{"track_id": 7, "frame": 1.5}],
    [{"track_id": 7, "frame": 3}],
])
def test_invalid_or_nonunique_track_observations_are_refused(tmp_path, bad_rows):
    sequence, tracks, target = _field_files(tmp_path)
    write_table(pd.DataFrame([{**row, "x": 10, "y": 9} for row in bad_rows]),
                tracks, canonicalise=False)
    with pytest.raises(ValueError):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert not target.exists()


def test_alpha_gate_hides_editor_entry_until_enabled(qtbot, prefs):
    from spacr.qt.preferences import _apply_alpha_widgets
    from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
    from spacr.settings import ALPHA_FEATURES

    panel = TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.show()
    assert panel._event_annotation_btn.objectName() in ALPHA_FEATURES[567]["widgets"]
    assert panel._event_annotation_btn.isHidden()
    prefs._set_show_alpha_features(True)
    _apply_alpha_widgets(panel)
    assert not panel._event_annotation_btn.isHidden()
    prefs._set_show_alpha_features(False)
    _apply_alpha_widgets(panel)
    assert panel._event_annotation_btn.isHidden()
    panel.close()


@pytest.mark.parametrize("change,reason", [
    (lambda table: table.drop(columns=["event"]), "lacks"),
    (lambda table: table.assign(event=None), "incomplete"),
    (lambda table: table.assign(frame=1.5), "integers"),
    (lambda table: table.assign(event="background"), "invalid event"),
    (lambda table: table.assign(field=""), "incomplete"),
    (lambda table: table.drop(columns=["object"]), "object column"),
    (lambda table: table.assign(object=None), "object identity"),
    (lambda table: table.assign(track_id=9, frame=0), "observed track frame"),
    (lambda table: pd.concat([table, table.assign(event="death")]),
     "labels one track frame"),
])
def test_bad_existing_event_tables_are_refused_before_any_edit(
        tmp_path, change, reason):
    sequence, tracks, target = _field_files(tmp_path)
    good = pd.DataFrame([{"field": "field", "track_id": 7,
                          "frame": 1, "event": "mitosis", "object": "cell"}])
    write_table(change(good), target, canonicalise=False)
    before = target.read_bytes()
    with pytest.raises(ValueError, match=reason):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert target.read_bytes() == before


@pytest.mark.parametrize("shape", [(3, 24, 24, 2), (3, 2, 24, 24)])
def test_channel_order_is_visible_without_loading_the_stack(tmp_path, shape):
    _sequence, tracks, target = _field_files(tmp_path)
    sequence = tmp_path / "field.npy"
    np.save(sequence, np.zeros(shape, dtype=np.uint16))
    payload = _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert payload["channels"] == 2
    assert payload["sequence"].read_count == 1
    assert len(payload["sequence"]._cache) == 1


def test_detector_reader_failure_cleans_stage_without_publishing(
        qtbot, tmp_path, monkeypatch):
    from spacr import timelapse

    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()

    def reject_stage(_path):
        raise ValueError("reader rejected staged rows")

    monkeypatch.setattr(timelapse, "_event_read_annotations", reject_stage)
    dialog._save()
    assert "reader rejected" in dialog._status.text()
    assert not target.exists()
    assert not list(target.parent.glob(".event-annotations-*.csv"))
    dialog.close()


def test_browse_and_cancel_keep_the_three_source_paths_distinct(
        qtbot, tmp_path, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as module

    sequence, tracks, target = _field_files(tmp_path)
    dialog = _EventAnnotationDialog(threaded=False)
    qtbot.addWidget(dialog)
    opened = iter([(str(tracks), ""), (str(sequence), "")])
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName",
                        lambda *_args: next(opened))
    monkeypatch.setattr(module.QFileDialog, "getExistingDirectory",
                        lambda *_args: str(tmp_path))
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        lambda *_args: (str(target), ""))
    dialog._pick_tracks()
    dialog._pick_sequence()
    assert dialog._tracks_path.text() == str(tracks)
    assert dialog._sequence_path.text() == str(sequence)
    dialog._pick_sequence_folder()
    assert dialog._sequence_path.text() == str(tmp_path)
    dialog._pick_output()
    assert dialog._output_path.text() == str(target)
    monkeypatch.setattr(module.QFileDialog, "getOpenFileName",
                        lambda *_args: ("", ""))
    monkeypatch.setattr(module.QFileDialog, "getExistingDirectory",
                        lambda *_args: "")
    monkeypatch.setattr(module.QFileDialog, "getSaveFileName",
                        lambda *_args: ("", ""))
    dialog._pick_tracks()
    dialog._pick_sequence()
    dialog._pick_sequence_folder()
    dialog._pick_output()
    assert (dialog._tracks_path.text(), dialog._sequence_path.text(),
            dialog._output_path.text()) == (str(tracks), str(tmp_path), str(target))
    dialog.close()


def test_failed_open_cannot_reuse_a_confirmed_field(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._sequence_path.setText(str(tmp_path / "missing.tif"))
    dialog._load_field()
    assert "Could not open" in dialog._status.text()
    assert not dialog._confirm.isChecked()
    dialog._event.setText("mitosis")
    dialog._add_event()
    assert dialog._events == []
    assert not target.exists()
    dialog.close()


def test_missing_paths_and_non_csv_output_refuse_before_file_read(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _EventAnnotationDialog(threaded=False)
    qtbot.addWidget(dialog)
    dialog._load_field()
    assert "both" in dialog._status.text()
    dialog._tracks_path.setText(str(tracks))
    dialog._sequence_path.setText(str(sequence))
    dialog._output_path.setText(str(target.with_suffix(".txt")))
    dialog._load_field()
    assert "CSV" in dialog._status.text()
    assert dialog._field is None
    dialog._output_path.clear()
    dialog._load_field()
    assert dialog._field is not None
    assert dialog._output_path.text().endswith("tracks/events/annotations.csv")
    dialog.close()


def test_saved_annotation_path_reaches_the_timelapse_setting_callback(
        qtbot, prefs, monkeypatch, tmp_path):
    from spacr.qt.widgets import timelapse_preview as module

    panel = module.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.show()
    callback = []
    panel.set_propagate_callback(callback.append)
    panel._open_event_annotations()
    assert callback == []
    prefs._set_show_alpha_features(True)
    saved = str(tmp_path / "annotations.csv")

    def save_then_close(dialog):
        dialog._saved_path = saved
        return 0

    monkeypatch.setattr(module._EventAnnotationDialog, "exec", save_then_close)
    panel._open_event_annotations()
    assert callback == [{"timelapse_events_annotations": saved}]
    panel.close()


def test_editor_opened_with_alpha_on_can_close_without_changing_settings(
        qtbot, prefs, monkeypatch):
    from spacr.qt.widgets import timelapse_preview as module

    prefs._set_show_alpha_features(True)
    panel = module.TimelapsePreviewPanel(threaded=False)
    qtbot.addWidget(panel)
    panel.show()
    assert not panel._event_annotation_btn.isHidden()
    callback = []
    panel.set_propagate_callback(callback.append)
    monkeypatch.setattr(module._EventAnnotationDialog, "exec", lambda self: 0)
    panel._open_event_annotations()
    assert callback == []
    panel.close()


def test_invalid_source_and_empty_tracks_fail_before_annotation_file(tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    invalid = tracks.with_name("unlabelled.csv")
    invalid.write_bytes(tracks.read_bytes())
    with pytest.raises(ValueError, match="spaCR tracks CSV"):
        _annotation_field_payload(str(invalid), str(sequence), str(target))
    write_table(pd.DataFrame(columns=["track_id", "frame"]),
                tracks, canonicalise=False)
    with pytest.raises(ValueError, match="needs frame and track_id"):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert not target.exists()


@pytest.mark.parametrize("change", [
    lambda table: table.drop(columns=["x"]),
    lambda table: table.assign(x=np.nan),
    lambda table: table.assign(y=np.inf),
])
def test_unlocatable_tracks_are_refused_instead_of_hiding_the_track_marker(
        tmp_path, change):
    sequence, tracks, target = _field_files(tmp_path)
    from spacr.tabular import read_table

    original = read_table(str(tracks), canonicalise=False, report=None)
    image = tifffile.imread(sequence, key=0)
    rgb = render_frame(image, tracks=original, frame=0)
    assert tuple(rgb[9, 10]) == tuple(track_colour(7))
    write_table(change(original), tracks, canonicalise=False)
    with pytest.raises(ValueError, match="positions"):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert not target.exists()


def test_other_field_without_object_column_can_be_retained(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    write_table(pd.DataFrame([{"field": "other", "track_id": 3,
                              "frame": 1, "event": "invasion"}]),
                target, canonicalise=False)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()
    dialog._save()
    rows = _event_read_annotations(str(target))
    assert set(zip(rows["field"], rows["event"])) == {
        ("other", "invasion"), ("field", "mitosis")}
    assert rows.loc[rows["field"] == "field", "object"].tolist() == ["cell"]
    dialog.close()


def test_stale_loader_result_cannot_replace_a_newer_field(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    original = dialog._field
    dialog._load_token += 1
    dialog._adopt_field(dialog._load_token - 1, {"field": "stale"})
    assert dialog._field is original
    dialog.close()


def test_saved_file_created_during_staging_is_not_overwritten(
        qtbot, tmp_path, monkeypatch):
    from spacr import timelapse

    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()
    original = timelapse._event_read_annotations

    def concurrent_create(path):
        verified = original(path)
        target.write_text("another writer owns this file\n")
        return verified

    monkeypatch.setattr(timelapse, "_event_read_annotations", concurrent_create)
    dialog._save()
    assert target.read_text() == "another writer owns this file\n"
    assert "changed during save" in dialog._status.text()
    assert not list(target.parent.glob(".event-annotations-*.csv"))
    dialog.close()


def test_reader_that_drops_a_row_blocks_publication(qtbot, tmp_path, monkeypatch):
    from spacr import timelapse

    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()
    monkeypatch.setattr(timelapse, "_event_read_annotations",
                        lambda _path: pd.DataFrame())
    dialog._save()
    assert "retain every event" in dialog._status.text()
    assert not target.exists()
    assert not list(target.parent.glob(".event-annotations-*.csv"))
    dialog.close()


def test_one_observation_cannot_acquire_two_classes_or_two_staged_rows(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._add_event()
    dialog._event.setText("death")
    dialog._add_event()
    assert [row["event"] for row in dialog._events] == ["mitosis"]
    assert "already has" in dialog._status.text()
    dialog._rows.selectRow(0)
    dialog._event.setText("death")
    dialog._add_event()
    assert [row["event"] for row in dialog._events] == ["death"]
    dialog._events.append(dict(dialog._events[0], event="mitosis"))
    dialog._save()
    assert "multiple event labels" in dialog._status.text()
    assert not target.exists()
    dialog.close()


def test_existing_other_tracker_provenance_is_refused_before_edit(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    write_table(pd.DataFrame([{
        "field": "field", "track_id": 7, "frame": 1,
        "event": "mitosis", "object": "cell", "tracker_backend": "trackastra",
        "track_source_sha256": "0" * 64,
    }]), target, canonicalise=False)
    before = target.read_bytes()
    with pytest.raises(ValueError, match="another tracker CSV"):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert target.read_bytes() == before


def test_legacy_rows_require_explicit_confirmation_then_gain_provenance(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    write_table(pd.DataFrame([{
        "field": "field", "track_id": 7, "frame": 1,
        "event": "mitosis", "object": "cell",
    }]), target, canonicalise=False)
    dialog = _dialog(qtbot, sequence, tracks, target)
    assert dialog._field["legacy_rows"]
    assert "no tracker identity" in dialog._status.text()
    dialog._save()
    assert "confirm" in dialog._status.text()
    dialog._confirm.setChecked(True)
    dialog._save()
    from spacr.tabular import read_table
    rows = read_table(str(target), canonicalise=False, report=None)
    assert rows["tracker_backend"].tolist() == ["trackpy"]
    assert rows["track_source_sha256"].tolist() == [dialog._field["track_digest"]]
    dialog.close()


def test_two_existing_labels_bound_to_current_tracker_reopen_without_guessing(
        tmp_path):
    import hashlib

    sequence, tracks, target = _field_files(tmp_path)
    digest = hashlib.sha256(tracks.read_bytes()).hexdigest()
    write_table(pd.DataFrame([{
        "field": "field", "track_id": 7, "frame": frame,
        "event": event, "object": "cell", "tracker_backend": "trackpy",
        "track_source_sha256": digest,
    } for frame, event in ((1, "mitosis"), (2, "death"))]),
        target, canonicalise=False)
    payload = _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert not payload["legacy_rows"]
    assert [(row["frame"], row["event"]) for row in payload["events"]] == [
        (1, "mitosis"), (2, "death")]


def test_tracker_modified_during_load_cannot_bind_stale_observations(
        tmp_path, monkeypatch):
    from spacr import tabular

    sequence, tracks, target = _field_files(tmp_path)
    original = tabular.read_table

    def mutate_after_parse(path, **kwargs):
        rows = original(path, **kwargs)
        if path == str(tracks):
            tracks.write_bytes(tracks.read_bytes() + b"\n")
        return rows

    monkeypatch.setattr(tabular, "read_table", mutate_after_parse)
    with pytest.raises(ValueError, match="changed while it was being read"):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert not target.exists()


def test_annotations_modified_during_load_cannot_pair_old_rows_with_new_digest(
        tmp_path, monkeypatch):
    from spacr import tabular

    sequence, tracks, target = _field_files(tmp_path)
    write_table(pd.DataFrame([{
        "field": "field", "track_id": 7, "frame": 1,
        "event": "mitosis", "object": "cell",
    }]), target, canonicalise=False)
    original = tabular.read_table
    newer = b"field,track_id,frame,event,object\nfield,7,2,death,cell\n"

    def replace_after_parse(path, **kwargs):
        rows = original(path, **kwargs)
        if path == str(target):
            target.write_bytes(newer)
        return rows

    monkeypatch.setattr(tabular, "read_table", replace_after_parse)
    with pytest.raises(ValueError, match="Annotations changed while"):
        _annotation_field_payload(str(tracks), str(sequence), str(target))
    assert target.read_bytes() == newer


def test_annotation_output_cannot_alias_tracker_or_image_source(tmp_path):
    sequence, tracks, _target = _field_files(tmp_path)
    from spacr.tabular import read_table

    original = read_table(str(tracks), canonicalise=False, report=None)
    original["field"] = "field"
    original["event"] = "mitosis"
    original["object"] = "cell"
    write_table(original, tracks, canonicalise=False)
    tracks_before = tracks.read_bytes()
    image_before = sequence.read_bytes()
    tracker_alias = tmp_path / "annotations.csv"
    tracker_alias.symlink_to(tracks)
    for output in (tracks, tracker_alias, sequence):
        with pytest.raises(ValueError, match="separate from"):
            _annotation_field_payload(str(tracks), str(sequence), str(output))
    assert tracks.read_bytes() == tracks_before
    assert sequence.read_bytes() == image_before


def test_prefilled_sequence_and_unopened_controls_do_not_write(qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _EventAnnotationDialog(sequence_path=sequence, threaded=False)
    qtbot.addWidget(dialog)
    assert dialog._sequence_path.text() == str(sequence)
    dialog._tracks_path.setText(str(tracks))
    dialog._output_path.setText(str(target))
    dialog._show_frame()
    dialog._event.setText("mitosis")
    dialog._add_event()
    dialog._remove_event()
    dialog._save()
    assert dialog._events == []
    assert "confirm" in dialog._status.text().lower()
    assert not target.exists()
    dialog.close()


def test_display_failure_clears_stale_frame_and_refuses_annotation(
        qtbot, tmp_path, monkeypatch):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    assert not dialog._preview.pixmap().isNull()
    source = dialog._field["sequence"]
    original = source.frame

    def fail_second_frame(frame):
        if frame == 1:
            raise OSError("frame became unreadable")
        return original(frame)

    monkeypatch.setattr(source, "frame", fail_second_frame)
    dialog._confirm.setChecked(True)
    dialog._frame.setValue(1)
    assert "frame became unreadable" in dialog._status.text()
    assert dialog._preview.pixmap().isNull()
    dialog._event.setText("mitosis")
    dialog._add_event()
    assert dialog._events == []
    assert not target.exists()
    dialog.close()


def test_cleared_track_selection_does_not_show_or_save_a_wrong_track(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._track.clear()
    dialog._show_frame()
    dialog._event.setText("mitosis")
    dialog._add_event()
    assert dialog._events == []
    assert not target.exists()
    dialog.close()


@pytest.mark.parametrize("name", ["", "background", "a\nname"])
def test_invalid_event_names_do_not_stage_rows(qtbot, tmp_path, name):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText(name)
    dialog._add_event()
    assert dialog._events == []
    assert "nonempty event" in dialog._status.text()
    dialog.close()


def test_source_and_output_changed_after_load_refuse_staged_save(
        qtbot, tmp_path):
    sequence, tracks, target = _field_files(tmp_path)
    dialog = _dialog(qtbot, sequence, tracks, target)
    dialog._confirm.setChecked(True)
    dialog._event.setText("mitosis")
    dialog._remove_event()
    dialog._tracks_path.setText(str(tmp_path / "another_tracks.csv"))
    dialog._confirm.setChecked(True)
    dialog._add_event()
    assert "source paths changed" in dialog._status.text()
    assert dialog._events == []
    dialog._save()
    assert "source paths changed" in dialog._status.text()
    dialog._tracks_path.setText(str(tracks))
    dialog._confirm.setChecked(True)
    dialog._output_path.setText(str(target.with_suffix(".txt")))
    dialog._save()
    assert "CSV" in dialog._status.text()
    assert not target.exists()
    dialog.close()
