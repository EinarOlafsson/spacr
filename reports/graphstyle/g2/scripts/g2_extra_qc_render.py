"""Compare extra QC creators on tiny supplied data without running scientific pipelines."""

import argparse
import ast
import base64
import hashlib
from html import unescape
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from unittest.mock import patch


def sha(value):
    """Return the exact SHA-256 of supplied bytes."""
    return hashlib.sha256(value).hexdigest()


def main():
    """Render one source tree's actual nested threshold, heatmap and HTML gallery creators."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    choices = ("threshold", "robustness", "gallery", "gallery-all_nan", "gallery-mixed_nonfinite")
    parser.add_argument("--case", choices=choices, action="append")
    parser.add_argument("--theme", choices=("light", "dark"))
    args = parser.parse_args()
    selected_cases = set(args.case or choices)
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
        XDG_CACHE_HOME=str(output / "cache"), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    os.environ.pop("SPACR_PIPELINE_FIGURES", None)
    sys.path.insert(0, str(source))
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from PIL import Image
    from spacr import image_quality, plot, seg_qc, sequencing
    from spacr.figures import style
    from spacr.qt import preferences

    for module in (image_quality, plot, seg_qc, sequencing, style):
        assert Path(module.__file__).resolve().is_relative_to(source)
    protected = {str(Path(module.__file__).resolve()): sha(Path(module.__file__).read_bytes())
                 for module in (image_quality, plot, seg_qc, sequencing, style)}
    report = {"source_root": str(source), "source_inputs": protected,
        "source_head": subprocess.check_output(["git", "-c", "gc.auto=0", "-C", str(source),
                                                "rev-parse", "HEAD"], text=True).strip(),
        "synthetic_only": True, "requested_cases": sorted(requested), "cases": {}, "writes": [], "themes": {},
        "scientific_pipeline_or_inference_run": False}
    arrays = output / "arrays"
    arrays.mkdir()

    def scalar(value):
        """Serialize only NumPy scalars that unbox to exact native JSON scalar types."""
        if isinstance(value, np.generic):
            result = value.item()
            if result is None or isinstance(result, (bool, int, float, str)):
                return result
        raise TypeError(type(value).__name__)

    def array(value, name):
        """Store exact array and mask bytes alongside their dtype and shape."""
        data = np.asarray(np.ma.getdata(value))
        mask = np.ma.getmaskarray(value)
        np.save(arrays / (name + ".npy"), data, allow_pickle=False)
        np.save(arrays / (name + "-mask.npy"), mask, allow_pickle=False)
        return {"dtype": data.dtype.str, "shape": list(data.shape),
                "data_sha256": sha(data.tobytes()), "mask_sha256": sha(mask.tobytes())}

    def capture(figure, name):
        """Separate raw artist identities from permitted palette, text and geometry changes."""
        numerical = []
        appearance = []
        for index, axis in enumerate(ax for ax in figure.axes if getattr(ax, "_colorbar", None) is None):
            image_numbers, image_appearance = [], []
            for number, image in enumerate(axis.images):
                stem = f"{name}-{index}-image{number}"
                low, high = map(float, image.get_clim())
                span = high - low or 1.0
                probes = np.asarray([low - span, low, low + span / 4.,
                    low + span / 2., high, high + span, np.nan, np.inf, -np.inf], dtype=np.float64)
                image_numbers.append({"array": array(image.get_array(), stem),
                    "clim": image.get_clim(), "extent": image.get_extent(), "origin": image.origin,
                    "norm": {"class": type(image.norm).__module__ + "." + type(image.norm).__qualname__,
                        "vmin": image.norm.vmin, "vmax": image.norm.vmax,
                        "clip": image.norm.clip, "vcenter": getattr(image.norm, "vcenter", None),
                        "gamma": getattr(image.norm, "gamma", None),
                        "probe_values": array(probes, stem + "-norm-probes"),
                        "mapped_values": array(image.norm(probes), stem + "-norm-mapped")}})
                ramp = image.get_cmap()
                image_appearance.append({"alpha": image.get_alpha(),
                    "interpolation": image.get_interpolation(),
                    "colormap": {"class": type(ramp).__module__ + "." + type(ramp).__qualname__,
                        "name": ramp.name, "N": ramp.N,
                        "sample_positions": array(np.linspace(0., 1., 257), stem + "-lut-positions"),
                        "sampled_rgba": array(ramp(np.linspace(0., 1., 257)), stem + "-lut-rgba"),
                        "bad": list(ramp.get_bad()), "under": list(ramp.get_under()),
                        "over": list(ramp.get_over())}})
            numerical.append({
                "xlim": axis.get_xlim(), "ylim": axis.get_ylim(),
                "xscale": axis.get_xscale(), "yscale": axis.get_yscale(),
                "lines": [{"x": array(line.get_xdata(), f"{name}-{index}-{number}-x"),
                           "y": array(line.get_ydata(), f"{name}-{index}-{number}-y")}
                          for number, line in enumerate(axis.lines)],
                "images": image_numbers})
            appearance.append({"title": axis.get_title(), "xlabel": axis.get_xlabel(),
                "ylabel": axis.get_ylabel(), "texts": [text.get_text() for text in axis.texts],
                "images": image_appearance, "axes_background": axis.get_facecolor(),
                "size_inches": [axis.get_position().width * figure.get_figwidth(),
                                axis.get_position().height * figure.get_figheight()]})
        result = style.finalize_figure(figure)
        return {"numerical": numerical, "appearance": appearance,
                "legend": [text.get_text() for legend in figure.legends for text in legend.get_texts()],
                "diagnostics": {"problems": list(result.problems), "checks": list(result.checks)}}

    tree = ast.parse(Path(sequencing.__file__).read_text())
    owner = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "graph_sequencing_stats")
    function = next(node for node in owner.body if isinstance(node, ast.FunctionDef)
                    and node.name == "find_and_visualize_fraction_threshold")
    writer = plot.save_figure
    for theme, background, foreground in (("light", "#FFFFFF", "#000000"),
                                          ("dark", "#181818", "#F0F0F0")):
        if theme not in selected_themes:
            continue
        preferences.set_figure_colors(background, foreground)
        preferences.set_figure_style({"background": background, "foreground": foreground})
        report["themes"][theme] = {"requested": {"background": background, "foreground": foreground},
            "stored_colors": preferences.get_figure_colors(), "creator_target": style.theme_target()}
        destination = output / theme
        destination.mkdir()
        captured = []

        def save(figure, path, **kwargs):
            """Keep normal PNG/PDF writer behavior in the selected theme."""
            captured.append(figure)
            kwargs.update(fmt="png", dpi=120, save_mode="screen", announce_colours=False)
            result = writer(figure, path, **kwargs)
            with Image.open(result) as saved:
                report["writes"].append({"theme": theme, "path": str(Path(result).relative_to(output)),
                    "normal_writer": "spacr.plot.save_figure", "save_mode": "screen",
                    "requested_dpi": 120, "image_size": saved.size,
                    "stored_dpi": saved.info.get("dpi"),
                    "top_left_rgba": saved.convert("RGBA").getpixel((0, 0))})
            kwargs["fmt"] = "pdf"
            writer(figure, str(Path(path).with_suffix(".pdf")), **kwargs)
            return result

        namespace = vars(sequencing).copy()
        namespace["_show_if_interactive"] = lambda: None
        exec(compile(ast.Module(body=[function], type_ignores=[]), sequencing.__file__, "exec"), namespace)
        with patch.object(plot, "save_figure", save):
            if "threshold" in selected_cases:
                frame = pd.DataFrame({"plateID": ["p", "p"], "rowID": ["r1", "r1"],
                    "columnID": ["c1", "c1"], "grna": ["a", "b"], "fraction": [.05, .95]})
                frame.attrs["origin"] = "supplied synthetic fractions"
                original = frame.to_csv(index=True), dict(frame.attrs)
                threshold = namespace[function.name](frame, target_unique_count=1, dst=str(destination))
                name = theme + "-threshold"
                report["cases"][name] = capture(captured[-1], name)
                assert (frame.to_csv(index=True), frame.attrs) == original
                report["cases"][name]["inputs"] = {"csv": original[0], "attrs": original[1]}
                report["cases"][name]["selected_threshold"] = float(threshold)

            if "robustness" in selected_cases:
                summary = pd.DataFrame({"parameter": ["baseline", "flow_threshold", "diameter"],
                    "value": ["run", "0.8", "40"], "fragile": [False, True, True],
                    "object_count_change": [0., -.5, np.nan], "median_area_change": [0., .1, np.inf],
                    "mean_intensity_change": [0., -.2, -.3], "matched_fraction": [1., .75, .8]})
                summary.attrs["settings"] = {"tolerance": .2}
                original = summary.to_csv(index=True), dict(summary.attrs)
                figure = seg_qc._robustness_figure(summary, .2, "cell")
                save(figure, destination / "robustness.png")
                name = theme + "-robustness"
                report["cases"][name] = capture(figure, name)
                assert (summary.to_csv(index=True), summary.attrs) == original
                report["cases"][name]["inputs"] = {"csv": original[0], "attrs": original[1], "tolerance": .2}

        if "gallery" in selected_cases:
            volume = np.arange(2*16*20*2, dtype=np.uint16).reshape(2, 16, 20, 2)
            paths = [destination / "plain.npy", destination / "flagged.npy"]
            np.save(paths[0], volume)
            np.save(paths[1], volume[:, ::-1].copy())
            source_bytes = {path.name: sha(path.read_bytes()) for path in paths}
            fields = [{"field": path.name, "status": "flagged" if index else "accepted",
                       "reasons": ["focus_below_threshold"] if index else [],
                       "channels": [{"channel": 7, "reasons": ["focus_below_threshold"] if index else [],
                                     "focus_variance": 1.25, "saturation_fraction": .01}]}
                      for index, path in enumerate(paths)]
            before_fields = json.dumps(fields, sort_keys=True)
            image_quality._write_gallery(destination / "gallery", fields, paths, [3, 7])
            html = (destination / "gallery.html").read_text()
            previews = [base64.b64decode(value) for value in re.findall(r'data:image/png;base64,([^\"]+)', html)]
            assert len(previews) == 2
            for index, preview in enumerate(previews):
                (destination / f"gallery-preview-{index}.png").write_bytes(preview)
            assert {path.name: sha(path.read_bytes()) for path in paths} == source_bytes
            assert json.dumps(fields, sort_keys=True) == before_fields
            report["cases"][theme + "-gallery"] = {"numerical": {"thumbnail_sha256": [sha(value) for value in previews]},
                "inputs": {"source_npy": source_bytes, "fields": fields, "channel_ids": [3, 7]},
                "html_sha256": sha(html.encode()), "appearance_review_required": True}
        for stem, values in (("all_nan", [np.nan, np.nan, np.nan]),
                             ("mixed_nonfinite", [np.nan, np.inf, -np.inf])):
            if "gallery-" + stem not in selected_cases:
                continue
            source_array = np.asarray([values, values], dtype=np.float32)
            path = destination / (stem + ".npy")
            np.save(path, source_array)
            source_hash = sha(path.read_bytes())
            fields = [{"field": path.name, "status": "flagged", "reasons": ["nonfinite"],
                       "channels": [{"channel": 0, "reasons": ["nonfinite"],
                                     "focus_variance": 0., "saturation_fraction": None}]}]
            fields_before = json.dumps(fields, sort_keys=True)
            image_quality._write_gallery(destination / (stem + "-gallery"), fields, [path], None)
            html = (destination / (stem + "-gallery.html")).read_text()
            previews = [base64.b64decode(value) for value in re.findall(r'data:image/png;base64,([^\"]+)', html)]
            assert len(previews) == 1
            png = destination / (stem + "-preview.png")
            png.write_bytes(previews[0])
            metadata = [json.loads(unescape(value)) for value in re.findall(r'data-display="([^\"]+)"', html)]
            assert sha(path.read_bytes()) == source_hash
            assert json.dumps(fields, sort_keys=True) == fields_before
            report["cases"][theme + "-gallery-" + stem] = {
                "numerical": {"thumbnail_sha256": [sha(value) for value in previews]},
                "inputs": {"source_npy": {path.name: source_hash}, "fields": fields, "channel_ids": None,
                           "array": array(source_array, theme + "-" + stem)},
                "appearance": {"display_metadata": metadata,
                               "fallback_explanation_present": "no finite values; no percentiles were measured" in html},
                "html_sha256": sha(html.encode()), "appearance_review_required": True}
        plt.close("all")

    assert {path: sha(Path(path).read_bytes()) for path in protected} == protected
    assert set(report["cases"]) == requested, sorted(requested - set(report["cases"]))
    differences = []
    if before is not None:
        for name, case in report["cases"].items():
            for key in ("numerical", "inputs", "selected_threshold"):
                if json.loads(json.dumps(case.get(key), default=scalar)) != before["cases"][name].get(key):
                    differences.append({"case": name, "changed": key})
        report["baseline_comparison"] = {"report_sha256": sha(args.compare.read_bytes()),
            "compared_cases": sorted(requested),
            "numerical_and_input_identity": not differences, "differences": differences}
    report["outputs"] = {str(path.relative_to(output)): sha(path.read_bytes())
                          for path in sorted(output.rglob("*")) if path.suffix in (".png", ".pdf", ".html")}
    (output / "report.json").write_text(json.dumps(report, indent=2, default=scalar) + "\n")
    assert not differences, differences


if __name__ == "__main__":
    main()
