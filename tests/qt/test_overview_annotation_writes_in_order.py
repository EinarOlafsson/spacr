"""Image annotations are saved one at a time, in order, without hiding newer messages.

Two annotations in a row used to be written by two concurrent jobs: the
earlier one could land last in measurements.db, and its late "Saved ..."
message replaced a later annotation-file report -- under load, a refused
annotation import's message (the intermittent
test_condition_tables_are_plate_qualified_and_never_modify_input failure).
"""
import threading

import numpy as np
import tifffile

from spacr import measure
from spacr.qt.widgets import qc_field_browser as browser


def test_queued_annotations_are_written_in_order_and_keep_a_later_file_report(tmp_path, qtbot, monkeypatch):
    for well in ("A01", "A02"):
        tifffile.imwrite(tmp_path / f"plate1_{well}_F1.tif", np.arange(120, dtype=np.uint16).reshape(10, 12))
    view = browser._FieldOverview(settings={"source": str(tmp_path), "channels": (0,)})
    qtbot.addWidget(view, before_close_func=lambda widget: widget._retire())
    view._indexed(browser._overview_index(tmp_path))
    qtbot.waitUntil(lambda: view._normalization_ready and not view._index_jobs.is_busy(), timeout=15000)
    view._metadata_source = str(tmp_path)
    release, calls = threading.Event(), []

    def written(path, fields, column, value, *, source=None, wells=None):
        if not calls:
            assert release.wait(10)
        calls.append(value)
        return {"operation": len(calls), "count": len(fields)}

    monkeypatch.setattr(measure, "_set_image_annotations", written)
    address = (view._fields[0].plate, view._fields[0].well)
    view._store_image_annotation({address}, "condition", "first")
    busy = view._metadata_jobs.active_jobs()
    view._store_image_annotation({address}, "condition", "second")
    assert view._metadata_jobs.active_jobs() == busy
    view._annotation_file_actions += 1
    view.status.setText("Annotation table identities are outside this source.")
    release.set()
    qtbot.waitUntil(lambda: calls == ["first", "second"] and not view._metadata_jobs.is_busy(), timeout=15000)
    assert view.status.text() == "Annotation table identities are outside this source."
    view._store_image_annotation({address}, "condition", "third")
    qtbot.waitUntil(lambda: calls[-1:] == ["third"] and not view._metadata_jobs.is_busy(), timeout=15000)
    assert view.status.text().startswith("Saved condition for 1 images")
