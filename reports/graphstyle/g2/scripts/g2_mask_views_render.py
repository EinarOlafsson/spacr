"""Selected-tree native creator evidence without detector, model or volume work."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(value):
    """Hash exact supplied bytes."""
    return hashlib.sha256(value).hexdigest()


def main():
    """Render an explicit source tree and compare only preserved scientific data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare", type=Path)
    args = parser.parse_args()
    source = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", SPACR_NO_GPU_RENDER="1",
        QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg",
        XDG_CONFIG_HOME=str(output / "config"), XDG_CACHE_HOME=str(output / "cache"),
        MPLCONFIGDIR=str(output / "mpl"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    sys.path.insert(0, str(source))

    import numpy as np
    from PySide6.QtCore import QCoreApplication, QEvent, QPointF
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication
    from spacr.figures import style
    from spacr.qt import preferences, theme
    from spacr.qt.screens import annotate, make_masks
    from spacr.qt.widgets import segmentation_views as views

    modules = (style, preferences, theme, annotate, make_masks, make_masks.engine, views)
    for module in modules:
        assert Path(module.__file__).resolve().is_relative_to(source), module.__file__
    protected = {str(Path(module.__file__).resolve()): sha(Path(module.__file__).read_bytes())
                 for module in modules}
    app = QApplication.instance() or QApplication([])
    array_directory = output / "arrays"
    array_directory.mkdir()

    def array(value, name):
        """Retain dtype, shape and bytes of the actual scientific/display array."""
        values = np.asarray(value)
        np.save(array_directory / (name + ".npy"), values, allow_pickle=False)
        return {"shape": list(values.shape), "dtype": values.dtype.str, "sha256": sha(values.tobytes())}

    def pixmap(value):
        """Observe raw pixmap pixels independently of widget chrome."""
        image = value.toImage().convertToFormat(QImage.Format_RGBA8888)
        return {"size": [image.width(), image.height()], "rgba_sha256": sha(bytes(image.constBits()))}

    def destroy(widget):
        """Close a native widget and complete its queued deletion before the next case."""
        widget.close()
        widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        app.processEvents()

    def snapshot(widget, destination):
        """Save the real widget's normal paint without injecting baseline style."""
        widget.show()
        app.processEvents()
        assert widget.grab().save(str(destination))

    rows, columns = np.indices((48, 64))
    pixels = np.stack([(columns * 4) % 256, (rows * 5) % 256,
                       ((rows + columns) * 3) % 256], axis=-1).astype(np.uint8)
    labels = np.zeros((48, 64), dtype=np.uint16)
    labels[8:23, 7:26] = 12
    labels[26:42, 35:56] = 39
    flows = np.roll(pixels, 1, axis=2)
    logits = np.linspace(-4, 4, 48 * 64, dtype=np.float32).reshape(48, 64)
    field = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    mask = np.zeros((64, 64), dtype=np.uint16)
    mask[27:35, 27:35] = 39
    inputs = dict(pixels=pixels, labels=labels, flows=flows, logits=logits, field=field, mask=mask)
    original = {name: array(values, "input-" + name) for name, values in inputs.items()}
    report = {"purpose": "actual creator evidence; not native tutorial acceptance",
        "source_root": str(source), "source_inputs": protected,
        "source_head": subprocess.check_output(["git", "-c", "gc.auto=0", "-C", str(source),
                                                "rev-parse", "HEAD"], text=True).strip(),
        "inputs": original, "cases": {}, "themes": {},
        "scientific_pipeline_or_inference_run": False}
    for name in ("light", "dark"):
        preferences.set_theme(name)
        theme.apply_qpalette(app, name)
        assert preferences.resolve_effective_theme() == name
        report["themes"][name] = {"resolved": preferences.resolve_effective_theme(),
                                  "palette": theme.active_palette()}
        destination = output / name
        destination.mkdir()
        native = annotate._ProbabilityHistogram()
        native.set_distribution([1, 2], {"1": [0, .05, .1, .11, .6, .8, .9, 1],
                                        "2": [.1, .25, .3, .7, .75, .95]}, "Probability")
        native.set_lines([.2, .8])
        nine = annotate._ProbabilityHistogram()
        nine.set_distribution(list(range(1, 10)),
            {str(value): [.05, .25, .55, .95] for value in range(1, 10)}, "Probability")
        nine.set_lines([.2, .8])
        otsu = make_masks._OtsuHistogramPlot([2, 5, 8, 7, 4, 1], [0, 1, 2, 3, 4, 5, 6], [1.3, 6.8])
        normalization = make_masks._LevelsPlot([2, 5, 8, 7, 4, 1], [0, 1, 2, 3, 4, 5, 6], [1, 5])
        for stem, widget in (("probability", native), ("probability_nine_classes", nine),
                             ("otsu", otsu), ("normalization", normalization)):
            if widget is nine:
                widget.resize(640, 900)
            else:
                widget.resize(560, 640)
            snapshot(widget, destination / (stem + ".png"))
            if isinstance(widget, annotate._ProbabilityHistogram):
                numerical = {"classes": list(widget._classes),
                             "counts": widget.counts(), "thresholds": widget.lines(),
                             "edges": [index / widget.BINS for index in range(widget.BINS + 1)]}
            else:
                numerical = {"counts": widget.counts.tolist(), "thresholds": list(widget.levels),
                             "edges": widget.edges.tolist()}
            observe = getattr(widget, "_spacr_style_observation", None)
            if callable(observe):
                appearance = {"observation": observe(), "diagnostics": asdict(style.finalize_figure(widget))}
            else:
                appearance = {"observation_available": False,
                              "reason": "Selected baseline creator has no native style observation API."}
            report["cases"][name + "-" + stem] = {"numerical": numerical, "appearance": appearance}
            destroy(widget)

        segmentation = views.SegmentationViews()
        segmentation.resize(560, 460)
        segmentation.set_arrays(image=pixels, labels={"cells": labels}, flows=flows, cellprob=logits)
        for view in views.VIEWS:
            segmentation.set_view(view)
            stem = views.picture_name_of(view)
            snapshot(segmentation, destination / (stem + ".png"))
            report["cases"][name + "-" + stem] = {
                "numerical": array(segmentation.rendered(), name + "-" + stem),
                "appearance": {"display_metadata_available": hasattr(segmentation, "_image_display_metadata"),
                    "display": getattr(segmentation, "_image_display_metadata", None), "certified": False}}
        destroy(segmentation)

        thumbnail = annotate._Thumbnail(0)
        thumbnail.resize(240, 240)
        thumbnail.setPixmap(views.to_qpixmap(pixels))
        thumbnail.set_occupied(True)
        thumbnail.set_count_text("39")
        thumbnail.set_current(True)
        snapshot(thumbnail, destination / "annotation_count.png")
        report["cases"][name + "-annotation_count"] = {
            "numerical": {"crop": pixmap(thumbnail.pixmap()), "count": thumbnail.count_text()},
            "appearance": {"certified": False}}
        destroy(thumbnail)

        canvas = make_masks._MaskCanvas()
        canvas.resize(600, 400)
        canvas.set_image_and_mask(field, mask)
        canvas.zoom_at(32, 32, 4.)
        x0, y0, _, _ = canvas._viewport_bounds()
        rectangles = [(x0 + 3, y0 + 3, x0 + 7, y0 + 7),
                      (x0 - 8, y0 - 8, x0 - 4, y0 - 4),
                      (x0 - 3, y0 + 8, x0 + 3, y0 + 12)]
        names = ["Inside annotation", "Offscreen annotation", "Partial annotation"]
        canvas.box_classes = names
        canvas.boxes = [(index, *rectangle) for index, rectangle in enumerate(rectangles)]
        snapshot(canvas, destination / "boxes_zoomed.png")
        report["cases"][name + "-boxes_zoomed"] = {"numerical": {
            "base_pixels": pixmap(canvas.pixmap()), "boxes": canvas.boxes,
            "class_names": names, "viewport": canvas._viewport_bounds()}, "appearance": {"certified": False}}
        canvas.boxes = []
        canvas.recrop_boxes = [(*rectangle, label) for rectangle, label in zip(rectangles, names)]
        snapshot(canvas, destination / "recrops_zoomed.png")
        report["cases"][name + "-recrops_zoomed"] = {"numerical": {
            "base_pixels": pixmap(canvas.pixmap()), "recrops": canvas.recrop_boxes,
            "viewport": canvas._viewport_bounds()}, "appearance": {"certified": False}}
        canvas.recrop_boxes = []

        magnifier = make_masks._LiveMagnifier(canvas, parent=canvas)
        magnifier._worker.close()
        magnifier._image_worker.close()
        magnifier.enabled = True
        magnifier.locked = True
        magnifier.size = 8
        magnifier.zoom = 1.5
        magnifier._cursor = (32, 32)
        magnifier._anchor = QPointF(canvas._image_to_canvas(32.5, 32.5))
        canvas.magnifier = magnifier
        snapshot(canvas, destination / "magnifier_locked_updating.png")
        box, lens, scale = magnifier.lens_geometry()
        part, area = magnifier._visible_part(box, lens, scale)
        assert part is not None
        px0, py0, px1, py1 = part
        step = magnifier._picture_step(scale)
        displayed = make_masks._box_picture(magnifier.detector_field(), box, part,
            canvas.mask[py0:py1, px0:px1], canvas.norm_lo, canvas.norm_hi, step)
        assert magnifier._worker._thread is None and magnifier._image_worker._thread is None
        report["cases"][name + "-magnifier_locked_updating"] = {
            "numerical": {"base_pixels": pixmap(canvas.pixmap()),
                "magnified_pixels": array(displayed, name + "-magnifier-pixels"),
                "source_box": box, "visible_part": part, "lens": list(lens.getRect()),
                "scale": scale, "sampling_step": step},
            "appearance": {"state": "Prepared locked/updating preview; both detector queues closed before enabling.",
                           "detector_threads_started": False, "certified": False}}
        magnifier.close()
        canvas.magnifier = None
        destroy(canvas)

    report["inputs_after"] = {name: array(values, "input-after-" + name) for name, values in inputs.items()}
    assert report["inputs_after"] == original
    assert {path: sha(Path(path).read_bytes()) for path in protected} == protected
    differences = []
    if args.compare:
        before = json.loads(args.compare.read_text())
        if before["inputs"] != report["inputs"]:
            differences.append({"case": "inputs", "changed": "source arrays"})
        if set(before["cases"]) != set(report["cases"]):
            differences.append({"case": "case inventory", "changed": "case names"})
        for name, case in report["cases"].items():
            numerical = json.loads(json.dumps(case["numerical"]))
            if numerical != before["cases"].get(name, {}).get("numerical"):
                differences.append({"case": name, "changed": "scientific/display data"})
        report["baseline_comparison"] = {"report_sha256": sha(args.compare.read_bytes()),
            "numerical_identity": not differences, "differences": differences}
    report["outputs"] = {str(path.relative_to(output)): sha(path.read_bytes())
                         for path in sorted(output.rglob("*.png"))}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not differences, differences


if __name__ == "__main__":
    main()
