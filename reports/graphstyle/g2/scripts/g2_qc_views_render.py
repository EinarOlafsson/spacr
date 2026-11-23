"""Root-scheduled tiny native QC before/after evidence using actual selected-tree owners."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch


def sha(value):
    """Hash bytes without conflating numerical data and rendered appearance."""
    return hashlib.sha256(value).hexdigest()


def main():
    """Capture supplied crops, field composites, histograms and plans without analysis jobs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    choices = ("measure-grid", "qc-field", "histogram-loaded", "histogram-missing",
               "histogram-zero", "cell-picker", "align-180", "align-400", "overview-export")
    parser.add_argument("--case", choices=choices, action="append")
    parser.add_argument("--theme", choices=("light", "dark"))
    args = parser.parse_args()
    selected_cases = set(args.case or choices)
    selected_themes = (args.theme,) if args.theme else ("light", "dark")
    requested = {name + "/" + case for name in selected_themes for case in selected_cases}
    expected_scientific = requested | {"inputs"}
    baseline = json.loads(args.compare.read_text()) if args.compare else None
    if baseline is not None:
        assert baseline["synthetic_only"] and baseline["input_arrays_unchanged"]
        assert requested <= set(baseline["cases"]), sorted(requested - set(baseline["cases"]))
        assert expected_scientific <= set(baseline["scientific"]), sorted(
            expected_scientific - set(baseline["scientific"]))
    source = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", SPACR_DEVICE="cpu", MPLBACKEND="Agg",
        QT_QPA_PLATFORM="offscreen", XDG_CONFIG_HOME=str(output / "config"),
        XDG_CACHE_HOME=str(output / "cache"), MPLCONFIGDIR=str(output / "mpl"),
        SPACR_OVERVIEW_DISK_CACHE="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    os.environ.pop("SPACR_PIPELINE_FIGURES", None)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, QRectF, QSettings, Qt
    from PySide6.QtGui import QFontInfo, QImage
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QCheckBox, QLabel

    QSettings.setPath(QSettings.NativeFormat, QSettings.UserScope, str(output / "config"))
    QSettings.setPath(QSettings.IniFormat, QSettings.UserScope, str(output / "config"))
    app = QApplication.instance() or QApplication([])
    app.setStyle("Fusion")
    from spacr import align, plot
    from spacr.figures import style
    from spacr.qt import preferences, theme
    from spacr.qt.screens import align as align_view
    from spacr.qt.widgets import measure_preview as preview, qc_field_browser as browser

    modules = (align, plot, style, preferences, theme, align_view, preview, browser)
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
    protected = {str(Path(module.__file__).resolve()): sha(Path(module.__file__).read_bytes())
                 for module in modules}
    protected[str(Path(__file__).resolve())] = sha(Path(__file__).read_bytes())
    report = {"source_root": str(source), "source_head": subprocess.check_output(
        ["git", "-c", "gc.auto=0", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "source_inputs": protected, "synthetic_only": True, "requested_cases": sorted(requested), "cases": {},
        "limitations": ["Native widget evidence is not a tutorial recording or full GUI acceptance.",
            "Missing baseline style observations remain unavailable, never a presentation PASS.",
            "Display coordinates, fonts and themes may change; raw pixels, counts and source identities must not.",
            "Overview reads one tiny generated NPY; no model, cropper, segmentation or plate computation runs."]}
    scientific = {}
    arrays = output / "arrays"
    arrays.mkdir()

    def scalar(value):
        """Unbox NumPy scalars exactly and refuse unsupported values."""
        if isinstance(value, np.generic):
            result = value.item()
            if result is None or isinstance(result, (bool, int, float, str)):
                return result
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    def record_array(name, value):
        """Preserve actual dtype, shape, data and mask bytes in standalone arrays."""
        original = np.asanyarray(value)
        data = np.ascontiguousarray(np.ma.getdata(original))
        mask = np.ascontiguousarray(np.ma.getmaskarray(original))
        assert not data.dtype.hasobject
        np.save(arrays / (name + ".npy"), data, allow_pickle=False)
        np.save(arrays / (name + "-mask.npy"), mask, allow_pickle=False)
        return dict(dtype=data.dtype.str, shape=list(data.shape), data_sha256=sha(data.tobytes()),
                    mask_sha256=sha(mask.tobytes()))

    def image_record(image, name):
        """Retain real QImage pixels independent of row padding and PNG metadata."""
        assert not image.isNull(), name
        rgba = image.convertToFormat(QImage.Format_RGBA8888)
        data = np.frombuffer(rgba.constBits(), np.uint8).reshape(rgba.height(), rgba.bytesPerLine())
        data = data[:, :rgba.width() * 4].reshape(rgba.height(), rgba.width(), 4)
        path = output / (name + ".png")
        path.parent.mkdir(parents=True, exist_ok=True)
        assert image.save(str(path), "PNG"), str(path)
        return dict(file=str(path.relative_to(output)), file_sha256=sha(path.read_bytes()),
                    rgba=record_array(name.replace("/", "-") + "-rgba", data),
                    device_pixel_ratio=float(image.devicePixelRatio()))

    def capture(widget, name):
        """Capture actual Qt paint and measured QLabel text, without repainting screenshot pixels."""
        widget.show()
        app.processEvents()
        assert 0 < widget.width() <= 1500 and 0 < widget.height() <= 1200
        result = image_record(widget.grab().toImage(), name)
        texts = []
        for label in widget.findChildren(QLabel):
            if not label.isVisible() or not label.text():
                continue
            point = label.mapTo(widget, QPoint(0, 0))
            rect = QRectF(label.rect()).translated(point.x(), point.y())
            if not rect.intersects(QRectF(widget.rect())):
                continue
            font = QFontInfo(label.font())
            flags = label.alignment().value | (Qt.TextWordWrap.value if label.wordWrap() else 0)
            bounds = QRectF(label.fontMetrics().boundingRect(label.contentsRect(), flags, label.text()))
            bounds.translate(point.x(), point.y())
            texts.append(dict(text=label.text(), rectangle=list(rect.getRect()),
                text_bounds=list(bounds.getRect()), inside_widget=QRectF(widget.rect()).contains(bounds),
                font_family=font.family(), font_points=font.pointSizeF()))
        result["label_text"] = texts
        result["widget_size"] = [widget.width(), widget.height()]
        observation = getattr(widget, "_spacr_style_observation", None)
        if callable(observation):
            result["paint_observation"] = observation()
            result["finalizer"] = asdict(style.finalize_figure(widget))
        else:
            result["paint_observation"] = observation
            result["finalizer"] = {"available": False,
                "reason": "Creator has no callable native observation; no compliance claim."}
        specification = getattr(widget, "_spacr_style_spec", None)
        if specification is not None:
            result["specification"] = asdict(specification)
        return result

    def close(widget):
        """Release each actual widget and its existing cleanup hooks before the next fixture."""
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()

    def portable(value):
        """Normalize only this run's owned output prefix in actual export provenance."""
        if isinstance(value, dict):
            return {key: portable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str) and value.startswith(str(output) + os.sep):
            return "<OUTPUT>/" + value[len(str(output)) + 1:]
        return value

    rows, columns = np.indices((36, 48))
    rgb = np.stack(((columns * 5) % 256, (rows * 7) % 256,
                    ((rows + columns) * 3) % 256), axis=-1).astype(np.uint8)
    intensities = np.stack((rows * 11 + columns, rows + columns * 17), axis=-1).astype(np.uint16)
    nucleus = np.zeros(rows.shape, np.uint16)
    cells = np.zeros(rows.shape, np.uint16)
    nucleus[5:13, 7:15], nucleus[22:28, 31:37] = 3, 8
    cells[2:17, 4:20], cells[19:32, 27:42] = 3, 8
    inputs = {"rgb": rgb, "intensities": intensities, "nucleus": nucleus, "cells": cells}
    scientific["inputs"] = {name: record_array(name, value) for name, value in inputs.items()}
    input_bytes = {name: value.tobytes() for name, value in inputs.items()}
    preferences.set_language("en")
    preferences.set_font_scale(1.)

    for name in selected_themes:
        (output / name).mkdir()
        preferences.set_theme(name)
        palette = theme.palette_for(name)
        preferences.set_figure_style({"background": palette["bg"], "foreground": palette["fg"]})
        preferences.set_figure_style_per_graph({})
        preferences.set_figure_colors(palette["bg"], palette["fg"])
        theme.apply_qpalette(app, name)
        theme._forget_painting_theme()
        app.setStyleSheet(theme.stylesheet(name))
        assert preferences.resolve_effective_theme() == name
        report.setdefault("themes", {})[name] = dict(name=name, palette=palette,
            settings_file=preferences._settings().fileName())
        assert Path(preferences._settings().fileName()).resolve().is_relative_to(output)

        if "measure-grid" in selected_cases:
            panel = preview.MeasurePreviewPanel(threaded=False)
            panel.resize(740, 520)
            panel.show()
            app.processEvents()
            panel._object_box.blockSignals(True)
            panel._object_box.setCurrentText("cell")
            panel._object_box.blockSignals(False)
            panel._group_cells.blockSignals(True)
            panel._group_cells.setChecked(True)
            panel._group_cells.blockSignals(False)
            crops = [dict(crop=rgb.copy(), category="Nucleated", included=True, label=3, area=240,
                          source_path="synthetic_field.npy"),
                     dict(crop=np.flip(rgb, axis=1).copy(), category="Nucleated", included=False,
                          label=8, area=300, source_path="synthetic_field.npy"),
                     dict(crop=np.roll(rgb, 5, axis=0), category="Unnucleated", included=True,
                          label=12, area=180, source_path="synthetic_field.npy")]
            panel._crops = crops
            panel._render_grid()
            key = name + "/measure-grid"
            report["cases"][key] = capture(panel._grid_scroll, key)
            grid, thumbnails = [], []
            for index in range(panel._grid.count()):
                item = panel._grid.itemAt(index)
                widget = item.widget()
                if isinstance(widget, preview._CropThumb):
                    thumbnails.append(image_record(widget.pixmap().toImage(), name + f"/crop-{widget._index}"))
                grid.append(dict(position=list(panel._grid.getItemPosition(index)),
                                 label=widget.text() if isinstance(widget, QLabel) else "",
                                 index=getattr(widget, "_index", None)))
            scientific[key] = dict(crops=[record_array(name + f"-crop{index}", entry["crop"])
                for index, entry in enumerate(crops)],
                entries=[{field: value for field, value in entry.items() if field != "crop"} for entry in crops],
                thumbnails=[item["rgba"] for item in thumbnails])
            report["cases"][key]["grid_positions"] = grid
            report["cases"][key]["thumbnails"] = thumbnails
            report["cases"][key]["owner_specification"] = (asdict(panel._spacr_style_spec)
                if hasattr(panel, "_spacr_style_spec") else None)
            close(panel)

        if "qc-field" in selected_cases:
            field = browser.QCFieldBrowser((), threaded=False)
            field._run_timer.stop()
            field._recheck_timer.stop()
            field.resize(740, 600)
            payload = browser.QCFieldImage(intensities=intensities, masks={"nucleus": nucleus, "cell": cells})
            field._payload = payload
            field._channel.blockSignals(True)
            field._channel.clear()
            field._channel.addItem("Synthetic channel 1", 1)
            field._channel.blockSignals(False)
            for role in payload.masks:
                checkbox = QCheckBox(role, field)
                checkbox.setChecked(True)
                field._layer_checks[role] = checkbox
            field.show()
            app.processEvents()
            field._render()
            app.processEvents()
            assert field._view._item is not None
            key = name + "/qc-field"
            report["cases"][key] = capture(field._view, key)
            composite = browser.render_qc_field(payload, 1, tuple(payload.masks))
            rendered = image_record(field._view._item.pixmap().toImage(), name + "/qc-composite")
            scientific[key] = dict(channel=1, masks=list(payload.masks),
                composite=record_array(name + "-composite", composite), qimage=rendered["rgba"])
            report["cases"][key]["raw_composite"] = rendered
            report["cases"][key]["owner_specification"] = (asdict(field._spacr_style_spec)
                if hasattr(field, "_spacr_style_spec") else None)
            close(field)

        counts = np.zeros(256, np.float64)
        counts[[4, 32, 100, 224]] = [1., 4., 9., 16.]
        for state, values in (("loaded", counts), ("missing", None), ("zero", np.zeros(256))):
            if "histogram-" + state not in selected_cases:
                continue
            histogram = browser._OverviewLevelsHistogram()
            histogram.resize(400, 600)
            histogram.set_counts(values)
            histogram.set_levels(.1, .9, 1.5)
            key = name + "/histogram-" + state
            report["cases"][key] = capture(histogram, key)
            levels = np.array([0., .1, .5, .9, 1.])
            coordinates = [histogram._x(float(level)) for level in levels]
            inverse = [histogram._level(value) for value in coordinates]
            np.testing.assert_allclose(inverse, levels, rtol=0., atol=1e-14)
            scientific[key] = dict(counts=None if histogram.counts is None else
                record_array(name + "-histogram-" + state, histogram.counts),
                levels=[histogram.black, histogram.white, histogram.gamma],
                handles=list(histogram._handles()))
            report["cases"][key]["coordinate_mapping"] = dict(
                plot=list(QRectF(histogram._plot()).getRect()), levels=levels.tolist(),
                x=coordinates, inverse=inverse)
            close(histogram)

        if "cell-picker" in selected_cases:
            picker = browser._OverviewCellPicker()
            picker.resize(280, 360)
            picker.configure(3, 2, 595.2756 / 841.8898, {(0, 0): "Well A01"})
            picker.selection = (1, 0, 1, 2)
            key = name + "/cell-picker"
            report["cases"][key] = capture(picker, key)
            paper = picker._paper()
            mapping = [picker._cell_at(QPointF(paper.left() + (column + .5) * paper.width() / 2,
                        paper.top() + (row + .5) * paper.height() / 3))
                       for row in range(3) for column in range(2)]
            assert mapping == [(row, column) for row in range(3) for column in range(2)]
            scientific[key] = dict(rows=picker.rows, columns=picker.columns,
                occupied=[list(item) for item in picker.occupied.items()],
                selection=list(picker.selection), cell_mapping=mapping)
            report["cases"][key]["paper_rect"] = list(paper.getRect())
            close(picker)

        tiles = [align.Tile(path=f"synthetic-{index}.npy", index=index, field=index + 1,
                            shape=(36, 48, 1)) for index in range(4)]
        methods = [align.METHOD_REGISTRATION, align.METHOD_REGISTRATION,
                   align.METHOD_NOMINAL, align.METHOD_UNREADABLE]
        placements = [align.Placement(tile, y=0., x=float(index * 60), confidence=confidence,
                                     method=method)
                      for index, (tile, confidence, method) in enumerate(zip(tiles, [.3, 1., 0., 0.], methods))]
        plan = align.AlignPlan(tiles=tiles, placements=placements, canvas_shape=(36, 228, 1))
        for width in (180, 400):
            if f"align-{width}" not in selected_cases:
                continue
            widget = align_view.TileLayoutWidget()
            widget.resize(width, 360)
            widget.set_plan(plan)
            before = asdict(plan)
            key = name + f"/align-{width}"
            report["cases"][key] = capture(widget, key)
            clicks = []
            widget.tile_clicked.connect(clicks.append)
            rectangles = widget.tile_rects()
            for index, rect in rectangles:
                QTest.mouseClick(widget, Qt.LeftButton, pos=rect.center().toPoint())
                assert clicks[-1] == index
            assert asdict(plan) == before
            scientific[key] = dict(plan=before, clicked_indices=clicks)
            report["cases"][key]["tile_rects"] = [(index, list(rect.getRect())) for index, rect in rectangles]
            report["cases"][key]["key"] = getattr(widget, "_spacr_layout_key", None)
            close(widget)

        if "overview-export" in selected_cases:
            fixture = output / name / "overview-field.npy"
            np.save(fixture, intensities, allow_pickle=False)
            original_file = fixture.read_bytes()
            selected = browser._OverviewField("synthetic", "A01", "A", 1, "4", (("", str(fixture)),))
            captured = []
            writer = plot.save_figure

            def observed_writer(figure, path, **kwargs):
                """Record actual export arrays and forward both formats through the normal writer."""
                record = {"images": [record_array(name + f"-overview-{len(captured)}-{index}", image.get_array())
                           for index, image in enumerate(image for axis in figure.axes for image in axis.images)]}
                png = str(Path(path).with_suffix(".png"))
                writer(figure, png, fmt="png", dpi=96, save_mode="print", close=False)
                result = writer(figure, path, **kwargs)
                record["diagnostics"] = asdict(style.finalize_figure(figure))
                captured.append(record)
                return result

            with patch.object(plot, "save_figure", observed_writer):
                paths, errors = browser._overview_export((selected,), output / name / "overview",
                    (0, 1), (1, 99), seed=17, quality_edge=None, dpi=96)
            assert not errors and len(paths) == len(captured) == 1
            assert fixture.read_bytes() == original_file
            metadata = json.loads((output / name / "overview" / "overview_selection.json").read_text())
            key = name + "/overview-export"
            scientific[key] = dict(images=captured[0]["images"], channels=[0, 1], percentiles=[1, 99],
                                   seed=17, fields=[[selected.plate, selected.well, selected.field]],
                                   recorded_selection=portable(metadata))
            report["cases"][key] = dict(figures=captured, requested_dpi=96,
                limitation="Bounded native-view evidence, not publication resolution or tutorial acceptance.")

    assert all(value.tobytes() == input_bytes[name] for name, value in inputs.items())
    report["input_arrays_unchanged"] = True
    report["source_inputs_unchanged"] = all(sha(Path(path).read_bytes()) == value for path, value in protected.items())
    assert report["source_inputs_unchanged"]
    report["scientific"] = scientific
    assert set(report["cases"]) == requested, sorted(requested - set(report["cases"]))
    assert set(scientific) == expected_scientific, sorted(expected_scientific - set(scientific))
    differences = []
    if baseline is not None:
        current = json.loads(json.dumps(scientific, default=scalar))
        differences = sorted(key for key in expected_scientific
                             if current[key] != baseline["scientific"][key])
        report["baseline_comparison"] = dict(report_sha256=sha(args.compare.read_bytes()),
            compared_cases=sorted(requested), scientific_inputs_compared=True,
            scientific_identity=not differences, different_cases=differences)
    report["output_hashes"] = {str(path.relative_to(output)): sha(path.read_bytes())
        for path in sorted(output.rglob("*")) if path.suffix in (".png", ".pdf")}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=scalar) + "\n")
    assert not differences, differences


if __name__ == "__main__":
    main()
