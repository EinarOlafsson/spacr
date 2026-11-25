"""Render actual QC creators from one selected tree and retain exact synthetic evidence."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch


def digest(value):
    """Hash bytes without coercing numerical payloads to a lossy text representation."""
    return hashlib.sha256(value).hexdigest()


def main():
    """Run only tiny supplied QC inputs; never fit illumination or process sequencing data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--case", choices=("barcode", "sweep", "illumination"), action="append")
    parser.add_argument("--theme", choices=("light", "dark"))
    args = parser.parse_args()
    selected_cases = set(args.case or ("barcode", "sweep", "illumination"))
    selected_themes = (args.theme,) if args.theme else ("light", "dark")
    requested = {theme + "-" + case for theme in selected_themes for case in selected_cases}
    before = json.loads(args.compare.read_text()) if args.compare else None
    if before is not None:
        assert requested <= set(before["cases"]), sorted(requested - set(before["cases"]))
    source = args.source_root.resolve(strict=True)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg", QT_QPA_PLATFORM="offscreen",
                      MPLCONFIGDIR=str(output / "mpl"), XDG_CONFIG_HOME=str(output / "config"),
                      XDG_CACHE_HOME=str(output / "cache"), OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    os.environ.pop("SPACR_PIPELINE_FIGURES", None)
    sys.path.insert(0, str(source))
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_rgba
    from PIL import Image
    from spacr import illumination, plot, sequencing_qc
    from spacr.figures import style
    from spacr.qt import preferences

    for module in (illumination, plot, sequencing_qc, style):
        assert Path(module.__file__).resolve().is_relative_to(source)
    protected = [source / relative for relative in (
        "spacr/sequencing_qc.py", "spacr/illumination.py", "spacr/figures/style.py",
        "spacr/figures/bundle.py", "spacr/figure_sink.py", "spacr/figure_font.py", "spacr/plot.py")]
    original = {str(path): digest(path.read_bytes()) for path in protected}
    report = {"source_root": str(source), "source_head": subprocess.check_output(
        ["git", "-c", "gc.auto=0", "-C", str(source), "rev-parse", "HEAD"], text=True).strip(),
        "source_inputs": original, "synthetic_only": True, "numerical_fit_performed": False,
        "requested_cases": sorted(requested), "cases": {},
        "limitations": ["Synthetic presentation evidence; no plate, fit, segmentation or GPU execution.",
                        "Saved PNG corner checks are separate from plot metadata and do not replace visual inspection."]}
    arrays = output / "arrays"
    arrays.mkdir()

    def json_scalar(value):
        """Unbox NumPy scalars exactly; refuse arrays or unsupported scalar representations."""
        if isinstance(value, np.generic):
            scalar = value.item()
            if scalar is None or isinstance(scalar, (bool, int, float, str)):
                return scalar
        raise TypeError(f"Unsupported JSON value: {type(value).__name__}")

    def array_record(value, key):
        """Retain dtype, shape, raw values and mask independently of rendered appearance."""
        values = np.asanyarray(value)
        data = np.asarray(np.ma.getdata(values))
        mask = np.ma.getmaskarray(values)
        np.save(arrays / (key + ".npy"), data, allow_pickle=False)
        np.save(arrays / (key + "-mask.npy"), mask, allow_pickle=False)
        return {"dtype": data.dtype.str, "shape": list(data.shape),
                "data_sha256": digest(data.tobytes()), "mask_sha256": digest(mask.tobytes())}

    def table_record(frame):
        """Capture exact table serialization, dtypes and attrs before and after plotting."""
        return {"csv": frame.to_csv(index=True), "dtypes": [str(value) for value in frame.dtypes],
                "attrs": dict(frame.attrs)}

    def figure_record(figure, key):
        """Keep numerical artists distinct from appearance diagnostics and changed layout."""
        panels = []
        appearance = []
        for index, axis in enumerate(ax for ax in figure.axes if getattr(ax, "_colorbar", None) is None):
            panel = {"lines": [], "images": [], "bars": []}
            for number, line in enumerate(axis.lines):
                panel["lines"].append({"x": array_record(line.get_xdata(), f"{key}-{index}-line{number}-x"),
                                       "y": array_record(line.get_ydata(), f"{key}-{index}-line{number}-y")})
            for number, image in enumerate(axis.images):
                panel["images"].append({"array": array_record(image.get_array(), f"{key}-{index}-image{number}"),
                                        "clim": list(image.get_clim()), "extent": list(image.get_extent())})
            for bar in axis.patches:
                if hasattr(bar, "get_height"):
                    panel["bars"].append({"centre": bar.get_x() + bar.get_width()/2,
                                           "height": bar.get_height(), "bottom": bar.get_y()})
            if key.endswith("barcode") and index == 0:
                panel["histogram_edges"] = [bar.get_x() for bar in axis.patches]
                panel["histogram_widths"] = [bar.get_width() for bar in axis.patches]
            panels.append(panel)
            spec = getattr(axis, "_spacr_style_spec", None)
            appearance.append({"kind": None if spec is None else spec.kind,
                "axes_inches": [axis.get_position().width * figure.get_figwidth(),
                                axis.get_position().height * figure.get_figheight()],
                "spec": None if spec is None else {
                    "target": spec.target, "comparison": dict(spec.comparison), "legend": dict(spec.legend)},
                "xlabel": axis.get_xlabel(), "ylabel": axis.get_ylabel(), "title": axis.get_title(),
                "legend": None if axis.get_legend() is None else {
                    "title": axis.get_legend().get_title().get_text(),
                    "entries": [text.get_text() for text in axis.get_legend().get_texts()]},
                "display_normalization": [{"type": type(image.norm).__name__,
                    "clim": list(image.get_clim()), "centre": getattr(image.norm, "vcenter", None),
                    "cmap": image.cmap.name} for image in axis.images]})
        diagnostics = style.finalize_figure(figure)
        return {"numerical": panels, "appearance": appearance,
                "diagnostics": {"problems": list(diagnostics.problems), "checks": list(diagnostics.checks)}}

    writer = plot.save_figure
    canvas_mismatches = []
    for theme, ground, ink in (("light", "#FFFFFF", "#000000"), ("dark", "#181818", "#F0F0F0")):
        if theme not in selected_themes:
            continue
        preferences.set_figure_style({"background": ground, "foreground": ink})
        preferences.set_figure_colors(ground, ink)
        preferences.set_figure_format("png")
        destination = output / theme
        destination.mkdir()
        captured = []
        saved_png = []

        def save(figure, path, **kwargs):
            """Use the normal writer for PNG and PDF, retaining the requested screen theme."""
            captured.append(figure)
            kwargs.update(fmt="png", dpi=120, save_mode="screen", announce_colours=False)
            saving_ground = plt.rcParams["savefig.facecolor"]
            result = writer(figure, path, **kwargs)
            with Image.open(result) as image:
                pixels = np.asarray(image.convert("RGBA"))
            expected = np.rint(np.asarray(to_rgba(ground)) * 255).astype(np.uint8)
            corners = {}
            for vertical, rows in (("top", slice(0, 3)), ("bottom", slice(-3, None))):
                for horizontal, columns in (("left", slice(0, 3)), ("right", slice(-3, None))):
                    sample = pixels[rows, columns]
                    corners[vertical + "_" + horizontal] = {
                        "rgba_values": np.unique(sample.reshape(-1, 4), axis=0).tolist(),
                        "opaque": bool(np.all(sample[..., 3] == 255)),
                        "matches_requested_ground": bool(np.all(sample == expected))}
            saved_png.append({"file": str(Path(result).relative_to(output)),
                "file_sha256": digest(Path(result).read_bytes()),
                "size": [pixels.shape[1], pixels.shape[0]], "sample_size": [3, 3],
                "requested_ground_rgba": expected.tolist(),
                "figure_ground_after_writer": list(figure.get_facecolor()),
                "savefig_facecolor_before_writer": saving_ground, "corners": corners})
            if not all(item["opaque"] and item["matches_requested_ground"] for item in corners.values()):
                canvas_mismatches.append(str(Path(result).relative_to(output)))
            kwargs["fmt"] = "pdf"
            writer(figure, str(Path(path).with_suffix(".pdf")), **kwargs)
            return result

        with patch.object(plot, "save_figure", save):
            if "sweep" in selected_cases:
                sweep = pd.DataFrame({"threshold": [.001, .003, .01, .03, .1],
                    "grnas_per_well": [30., 10., 5., 3., 1.], "grnas_per_well_retained": [32., 11., 6., 4., 2.],
                    "well_retention": [1., .95, .8, .7, .5], "collision_rate": [.9, .6, .2, .1, 0.]})
                sweep.attrs["units"] = "fractions before percent conversion"
                choice = sequencing_qc.ThresholdChoice(.01, 5., 5., "median", 6, True, 5, .005, .02)
                original_sweep = table_record(sweep)
                figure = sequencing_qc.plot_threshold_sweep(sweep, choice, str(destination))
                key = theme + "-sweep"
                report["cases"][key] = figure_record(figure, key)
                assert table_record(sweep) == original_sweep
                report["cases"][key]["inputs"] = original_sweep
                report["cases"][key]["saved_png"] = saved_png[-1]
            if "barcode" in selected_cases:
                counts = pd.DataFrame({"count": [1, 2, 3, 4, 8, 20]})
                per_well = pd.DataFrame({"reads": [1., 2., 3., 4., 8., 20.]})
                per_well.attrs["plate"] = "synthetic plate"
                starved = per_well.iloc[:1].copy()
                starved.attrs["cutoff"] = 1.5
                positions = pd.DataFrame({"axis": ["column"]*3, "label": ["c10", "c2", "c1"],
                                         "ratio_to_plate": [1.4, .8, 1.], "flagged": [True, False, False]})
                depth = {"reads_per_grna": pd.Series([0., 2., 4., 8.]), "gini": .45,
                         "skew_ratio": 4., "dropout_fraction": .25}
                unmapped = {"per_field": {"columnID": .2, "rowID": .3, "grna_name": .1},
                            "unmapped_fraction": .4, "total_reads": 100}
                tables = [counts, per_well, starved, positions]
                originals = [table_record(frame) for frame in tables]
                depth_values = array_record(depth["reads_per_grna"].to_numpy(), theme + "-depth-source")
                depth_statistics = {name: value for name, value in depth.items() if name != "reads_per_grna"}
                unmapped_snapshot = json.loads(json.dumps(unmapped, default=json_scalar))
                figure = sequencing_qc.plot_barcode_qc(counts, per_well=per_well, starved=starved,
                    positions=positions, depth=depth, unmapped=unmapped, dst=str(destination))
                key = theme + "-barcode"
                report["cases"][key] = figure_record(figure, key)
                assert [table_record(frame) for frame in tables] == originals
                assert array_record(depth["reads_per_grna"].to_numpy(), theme + "-depth-after") == depth_values
                assert {name: value for name, value in depth.items() if name != "reads_per_grna"} == depth_statistics
                assert unmapped == unmapped_snapshot
                report["cases"][key]["inputs"] = {"tables": originals, "depth_values": depth_values,
                    "depth_statistics": depth_statistics, "unmapped": unmapped_snapshot}
                report["cases"][key]["saved_png"] = saved_png[-1]
            if "illumination" in selected_cases:
                observed = np.linspace(.6, 1.4, 48, dtype=np.float32).reshape(6, 8)
                plane = np.linspace(.8, 1.2, 48, dtype=np.float32).reshape(6, 8)
                corrected = observed / plane
                panels = {0: (plane, observed, corrected)}
                metrics = {0: {"gain_min": float(plane.min()), "gain_max": float(plane.max()),
                    "nonuniformity_pct": 30., "slope_before": .3, "slope_after": .1,
                    "bias_removed_pct": 200/3, "n_fields": 4}}
                original_arrays = [array_record(value, theme + "-source-" + str(index))
                                   for index, value in enumerate(panels[0])]
                original_metrics = json.dumps(metrics, sort_keys=True)
                illumination._write_qc_figure("p1", SimpleNamespace(shape=(6, 8)), [0], panels,
                    metrics, str(destination), 1, stage="measurement_qc")
                key = theme + "-illumination"
                report["cases"][key] = figure_record(captured[-1], key)
                assert [array_record(value, theme + "-source-after-" + str(index))
                        for index, value in enumerate(panels[0])] == original_arrays
                assert json.dumps(metrics, sort_keys=True) == original_metrics
                report["cases"][key]["inputs"] = {"arrays": original_arrays,
                                                   "metrics": json.loads(original_metrics)}
                report["cases"][key]["saved_png"] = saved_png[-1]
        plt.close("all")
    assert {str(path): digest(path.read_bytes()) for path in protected} == original
    assert set(report["cases"]) == requested, sorted(requested - set(report["cases"]))
    differences = []
    if before is not None:
        for name, case in report["cases"].items():
            for field in ("numerical", "inputs"):
                if case[field] != before["cases"][name][field]:
                    differences.append({"case": name, "changed": field})
            case["baseline_appearance"] = before["cases"][name]["appearance"]
        report["baseline_comparison"] = {"report_sha256": digest(args.compare.read_bytes()),
            "compared_cases": sorted(requested),
            "numerical_and_input_identity": not differences, "differences": differences,
            "display_changes": "Shared palette, alpha, labels, legends and geometry may differ; raw data, masks, histogram bins and colour limits must agree exactly."}
    report["outputs"] = {str(path.relative_to(output)): digest(path.read_bytes())
                          for path in sorted(output.glob("*/*")) if path.suffix in (".png", ".pdf")}
    report["saved_canvas_mismatches"] = canvas_mismatches
    (output / "report.json").write_text(json.dumps(report, indent=2, default=json_scalar) + "\n")
    assert not differences, differences
    if before is not None:
        assert not canvas_mismatches, canvas_mismatches


if __name__ == "__main__":
    main()
