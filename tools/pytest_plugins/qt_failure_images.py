"""Capture allowlisted original QImage locals only after an assertion fails."""
from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_exception_interact(node, call, report):
    item = node
    if call.when != "call" or not report.failed or call.excinfo is None:
        return
    if not isinstance(call.excinfo.value, AssertionError):
        return
    if item.nodeid != "tests/qt/test_field_paints_opaquely_20261012.py::test_home_and_mask_are_identical_to_the_classic_paint[dark]":
        return
    try:
        _capture_original_failure(item, call)
    except Exception as error:
        print("POST-FAILURE IMAGE CAPTURE ERROR TYPE: " + type(error).__name__)


def _capture_original_failure(item, call):
    import datetime
    import gzip
    import hashlib
    import json
    import os
    from pathlib import Path
    import subprocess

    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QImage

    expected_file = (item.config.rootpath / "tests/qt/test_field_paints_opaquely_20261012.py").resolve()
    frame = None
    for entry in call.excinfo.traceback:
        candidate = entry.frame.raw
        if candidate.f_code.co_name == "_compare" and Path(candidate.f_code.co_filename).resolve() == expected_file:
            frame = candidate
    if frame is None:
        return
    values = frame.f_locals
    dest_root = Path(os.environ["SPACR_QT_PIXEL_FAILURE_DIR"])
    token = hashlib.sha256(item.nodeid.encode()).hexdigest()[:20]
    dest = dest_root / (str(os.getpid()) + "-" + token)
    dest.mkdir(parents=True, exist_ok=False)
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=item.config.rootpath).decode().strip()
    metadata = {"schema": "spacr.qt.post-assertion-images/v1", "nodeid": item.nodeid,
                "source_sha": source_sha, "captured_after_original_assertion": True,
                "original_report_failed": True, "pid": os.getpid(),
                "worker": os.environ.get("PYTEST_XDIST_WORKER", "controller"),
                "finished_assertion_line": frame.f_lineno,
                "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "original_assertion_detail": call.excinfo.value.args,
                "label": values.get("label"), "attempt": values.get("_attempt"),
                "images": {}, "pairs": [],
                "limits": {"image_pixels": 16000000, "native_bytes_per_image": 67108864,
                           "coordinate_rows_per_pair": 10000, "widget_chains_per_pair": 256,
                           "widget_chain_depth": 64},
                "full_raw_diff_encoding": "byte-wise XOR of original same-format native image bytes, streamed and gzip-compressed; metadata preserves format/stride/dimensions"}
    image_names = ("first", "flushed", "grabbed", "classic_flushed", "classic_grabbed")
    for name in image_names:
        image = values.get(name)
        if not isinstance(image, QImage):
            continue
        if image.width() * image.height() > 16000000 or image.sizeInBytes() > 67108864:
            metadata["images"][name] = {"oversize_not_captured": True, "width": image.width(), "height": image.height(), "native_bytes": image.sizeInBytes()}
            continue
        path = dest / (name + ".png")
        assert image.save(str(path), "PNG"), name
        raw = bytes(image.constBits())
        metadata["images"][name] = {"width": image.width(), "height": image.height(),
                                    "format": str(image.format()), "dpr": image.devicePixelRatioF(),
                                    "bytes_per_line": image.bytesPerLine(), "native_bytes": len(raw),
                                    "native_sha256": hashlib.sha256(raw).hexdigest(),
                                    "png_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        with gzip.open(dest / (name + ".native.bin.gz"), "wb") as output:
            output.write(raw)
    window = values.get("window")

    def chain(x, y):
        result = []
        widget = window.childAt(QPoint(x, y))
        seen = set()
        while widget is not None and widget is not window and id(widget) not in seen and len(result) < 64:
            seen.add(id(widget))
            point = widget.mapTo(window, QPoint(0, 0))
            result.append({"class": type(widget).__name__, "object_name": widget.objectName(),
                           "geometry": list(widget.geometry().getRect()),
                           "window_origin": [point.x(), point.y()],
                           "opaque_attribute": widget.testAttribute(Qt.WA_OpaquePaintEvent)})
            widget = widget.parentWidget()
        return result

    for left_name, right_name in (("flushed", "classic_flushed"), ("grabbed", "classic_grabbed")):
        left, right = values.get(left_name), values.get(right_name)
        if not isinstance(left, QImage) or not isinstance(right, QImage):
            continue
        if left_name not in metadata["images"] or right_name not in metadata["images"] or metadata["images"][left_name].get("oversize_not_captured") or metadata["images"][right_name].get("oversize_not_captured"):
            metadata["pairs"].append({"opaque": left_name, "classic": right_name, "oversize_pair_not_scanned": True})
            continue
        raw_diff_name = left_name + "-vs-" + right_name + ".native-xor.bin.gz"
        raw_diff_complete = left.format() == right.format() and left.size() == right.size() and left.bytesPerLine() == right.bytesPerLine()
        if raw_diff_complete:
            left_view, right_view = left.constBits(), right.constBits()
            with gzip.open(dest / raw_diff_name, "wb") as output:
                for offset in range(0, left.sizeInBytes(), 65536):
                    output.write(bytes(a ^ b for a, b in zip(left_view[offset:offset+65536], right_view[offset:offset+65536])))
        count = sampled = coordinate_rows = chain_rows = 0
        xmin = ymin = None
        xmax = ymax = None
        first_sample = None
        path = dest / (left_name + "-vs-" + right_name + ".all-pixel-differences.jsonl.gz")
        with gzip.open(path, "wt", encoding="utf-8") as output:
            for y in range(min(left.height(), right.height())):
                for x in range(min(left.width(), right.width())):
                    if left.pixel(x, y) == right.pixel(x, y):
                        continue
                    row = {"xy": [x, y], "opaque_argb32": left.pixel(x, y),
                           "classic_argb32": right.pixel(x, y),
                           "opaque_rgba": list(left.pixelColor(x, y).getRgb()),
                           "classic_rgba": list(right.pixelColor(x, y).getRgb())}
                    if coordinate_rows < 10000:
                        if chain_rows < 256:
                            row["widget_chain_after_assertion"] = chain(x, y)
                            chain_rows += 1
                        output.write(json.dumps(row, separators=(",", ":")) + "\n")
                        coordinate_rows += 1
                    count += 1
                    xmin = x if xmin is None else min(xmin, x)
                    ymin = y if ymin is None else min(ymin, y)
                    xmax = x if xmax is None else max(xmax, x)
                    ymax = y if ymax is None else max(ymax, y)
                    if x % 4 == 0 and y % 4 == 0:
                        sampled += 1
                        if first_sample is None:
                            first_sample = {k: v for k, v in row.items() if k != "widget_chain_after_assertion"}
        metadata["pairs"].append({"opaque": left_name, "classic": right_name,
                                  "all_differing_pixels": count, "every_fourth_pixel_differences": sampled,
                                  "coordinate_rows_written": coordinate_rows, "coordinate_rows_truncated": count > coordinate_rows,
                                  "widget_chain_rows_written": chain_rows, "widget_chain_rows_truncated": count > chain_rows,
                                  "native_xor_diff_path": raw_diff_name if raw_diff_complete else None,
                                  "full_native_xor_diff_complete": raw_diff_complete,
                                  "all_pixel_bounds": [xmin, ymin, xmax, ymax],
                                  "first_fourth_pixel_sample": first_sample,
                                  "diff_path": path.name,
                                  "dimension_match": left.size() == right.size(),
                                  "historical_box_centre_chain_after_assertion": chain(20, 488)})
    field = values.get("field")
    if field is not None:
        point = field.mapTo(window, QPoint(0, 0))
        metadata["field_after_assertion"] = {"class": type(field).__name__,
            "geometry": list(field.geometry().getRect()), "window_origin": [point.x(), point.y()],
            "opaque_attribute": field.testAttribute(Qt.WA_OpaquePaintEvent)}
    (dest / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
