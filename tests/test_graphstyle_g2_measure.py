"""Measure presentation contracts on prepared inputs, without scientific reruns."""

from dataclasses import fields, is_dataclass
from hashlib import sha256
import base64
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


class FrozenDoseResult:
    """Prepared fitted-dose output shared by the tests and evidence renderer."""

    def __init__(self):
        """Retain fixed observations and a supplied curve, without fitting a model."""
        self.dose = np.array([0.1, 1.0, 10.0, 100.0])
        self.response = np.array([98.0, 82.0, 48.0, 17.0])
        self.ec50 = 8.0
        self.unit = "µM"
        self.curve_x = np.array([0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0])
        self.curve_y = np.array([99.0, 94.0, 83.0, 66.0, 47.0, 29.0, 16.0])

    def curve(self):
        """Return the exact previously prepared display curve."""
        return self.curve_x, self.curve_y


def input_record(value):
    """Record ordered input bytes, dtypes, masks and metadata without fitting anything."""
    if isinstance(value, pd.DataFrame):
        return {"kind": "dataframe", "csv": value.to_csv(index=True, lineterminator="\n"),
                "columns": [str(item) for item in value.columns],
                "dtypes": [str(item) for item in value.dtypes],
                "arrays": [input_record(value.iloc[:, i].to_numpy()) for i in range(len(value.columns))]}
    if isinstance(value, pd.Series):
        return {"name": str(value.name), "index": input_record(value.index.to_numpy()),
                "values": input_record(value.to_numpy())}
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        payload = (json.dumps(array.tolist(), default=str, ensure_ascii=False).encode()
                   if array.dtype.hasobject else array.tobytes(order="C"))
        missing = pd.isna(array).tobytes()
        return {"dtype": str(array.dtype), "shape": list(array.shape), "sha256": sha256(payload).hexdigest(),
                "data_base64": base64.b64encode(payload).decode('ascii'),
                "missing": sha256(missing).hexdigest(),
                "missing_base64": base64.b64encode(missing).decode('ascii')}
    if is_dataclass(value):
        return {field.name: input_record(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, dict):
        return [[str(key), input_record(item)] for key, item in value.items()]
    if isinstance(value, (tuple, list)):
        return [input_record(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "__dict__"):
        return input_record(vars(value))
    return value


def prepared_measure_cases(measure):
    """Supply every local creator with fixed normal-boundary objects and fitted results."""
    plane = np.arange(256, dtype=np.uint16).reshape(16, 16)
    covered = plane >= 128
    rgb = np.stack([plane.astype(np.uint8), np.full((16, 16), 73, np.uint8),
                    np.full((16, 16), 19, np.uint8)], axis=-1)
    dna = np.array([0.8, 1.7, 1.9, 2.0, 2.1, 2.4, 2.9, 3.4, 3.9, 4.1, 7.0, np.nan])
    fit = measure._DnaFit(100.0, 200.0, 8.0, 12.0, (0.5, 0.2, 0.25, 0.05),
                          (125.0, 175.0), 11, 0)
    condition = pd.DataFrame({
        "condition": ["slow"] * 3 + ["fast"] * 3, "time": [0., 2., 4.] * 2,
        "relative_open_area": [1., .8, .6, 1., .5, .2],
        "relative_open_area_sd": [.04, np.nan, .08, .03, .06, .05],
        "n_wells": [2] * 6, "time_unit": ["h"] * 6})
    well_curves = pd.concat([condition.assign(plateID="p", rowID="r1", columnID=f"c{i}")
                            for i in (1, 2)], ignore_index=True)
    summary = pd.DataFrame({"condition": ["slow", "slow", "fast", "fast"],
                            "wound_ok": [1, 1, 1, 0], "half_closure_time": [5., np.nan, 2., 3.],
                            "time_unit": ["h"] * 4, "plateID": ["p"] * 4,
                            "rowID": ["r1"] * 4, "columnID": ["c1", "c2", "c3", "c4"]})
    signals = pd.DataFrame({"plateID": ["p"] * 6, "viability_method": ["stain"] * 6,
                            "dead_signal": [1., 4., 20., 70., 100., np.nan],
                            "live_signal": [100., 90., 60., 8., 2., 5.],
                            "viability_state": ["live", "live", "live", "dead", "dead", "unstained"]})
    cuts = {stain: {"p": measure._PopulationCut(25., "manual", 3., .4, 6)}
            for stain in ("dead", "live")}
    wells = pd.DataFrame({"plate_key": ["p"] * 7,
                          "role": ["negative", "negative", "positive", "positive", "sample", "sample", "sample"],
                          "viability": [1., .9, .1, np.nan, .5, .6, .7],
                          "cytotoxicity_index": [0., 10., 90., 95., 50., 40., 30.]})
    qc = pd.DataFrame({"plate_key": ["p"], "zprime_viability": [.65], "zprime_cytotoxicity_index": [.7]})
    fits = {"viability": SimpleNamespace(fits=[SimpleNamespace(group="A", result=FrozenDoseResult()),
                                             SimpleNamespace(group="B", result=None)])}
    survival = pd.DataFrame({"condition": ["reference"] * 3 + ["treated"] * 3,
                             "time": [0., 1., 2.] * 2, "survival": [1., .8, .6, 1., .7, .4],
                             "ci_lower": [1., .7, .4, 1., .5, .2], "ci_upper": [1., .9, .8, 1., .9, .6],
                             "censored": [0, 0, 1, 0, 1, 0], "time_unit": ["h"] * 6})
    survival_summary = pd.DataFrame({"level": ["condition"] * 2, "group": ["reference", "treated"],
                                     "n": [10, 10], "events": [4, 6]})
    logrank = pd.DataFrame({"p_value": [.031], "comparison": ["all conditions"]})
    cox = pd.DataFrame({"covariate": ["treated", "size"], "hazard_ratio": [1.8, .7],
                        "hr_lower": [1.2, .5], "hr_upper": [2.4, .9]})
    return {
        "montage": ("img_list_to_grid", ([rgb, rgb.copy()], ["first", "second"]), {}),
        "confluency": ("_confluency_figure", (plane, measure._ConfluencyResult(covered, .5, "masks"), "p_r1_c1"), {}),
        "dna": ("_dna_histogram_figure", (dna, fit, "p", ["G1"] * 6 + ["S"] * 3 + ["G2"] * 2), {}),
        "wound_edges": ("_wound_edge_figure", ("p_r1_c1", [("0 h", plane, covered), ("4 h", plane, ~covered)]), {}),
        "wound_curve": ("_wound_curve_figure", (condition, well_curves), {}),
        "wound_half": ("_wound_half_closure_figure", (summary,), {}),
        "wound_plate": ("_wound_plate_figure", (summary, "p"), {}),
        "viability_threshold": ("_viability_threshold_figure", (signals, cuts, "p", "p"), {}),
        "viability_controls": ("_viability_controls_figure", (wells, qc), {}),
        "viability_dose": ("_viability_dose_figure", (fits,), {}),
        "survival": ("_time_to_event_figure", (survival, survival_summary, logrank, "event"), {}),
        "hazard": ("_hazard_ratio_figure", (cox, "event"), {}),
    }


@pytest.mark.parametrize("case", ["montage", "confluency", "dna", "wound_edges", "wound_curve",
                                  "wound_half", "wound_plate", "viability_threshold", "viability_controls",
                                  "viability_dose", "survival", "hazard"])
def test_measure_creators_preserve_prepared_input_bytes_and_declare_semantics(case):
    """Every converted creator retains input bytes and has meaningful panel declarations."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure

    name, args, kwargs = prepared_measure_cases(measure)[case]
    before = input_record(args)
    figure = getattr(measure, name)(*args, **kwargs)
    try:
        assert input_record(args) == before
        assert figure._spacr_caption
        for ax in figure.axes:
            if getattr(ax, "_colorbar", None) is not None:
                continue
            specification = ax._spacr_style_spec
            assert dict(specification.legend)["caption"]
            if specification.kind not in ("image", "schematic"):
                assert dict(specification.legend)["quantities"]
                assert dict(specification.legend).get("reference") or specification.controls
    finally:
        plt.close(figure)


def test_wound_original_mean_and_offsets_survive_the_bar_conversion():
    """Nonfinite values cannot silently change an existing mean or finite point offsets."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure

    table = pd.DataFrame({"condition": ["a"] * 4, "wound_ok": [1] * 4,
                          "half_closure_time": [1., np.inf, 3., np.nan], "time_unit": ["h"] * 4})
    figure = measure._wound_half_closure_figure(table)
    try:
        ax = figure.axes[0]
        assert np.isposinf(ax._spacr_measure_means[0])
        assert np.isposinf(ax.patches[0].get_height())
        points = next(mark for mark in ax.collections if mark.get_gid() == "spacr-condition:0:points")
        np.testing.assert_array_equal(points.get_offsets(), [[-.12, 1.], [.12, 3.]])
        pd.testing.assert_frame_equal(figure._spacr_data, table)
    finally:
        plt.close(figure)


def test_viability_jitter_advances_over_missing_values_and_both_readouts():
    """Keep the original local RNG stream while leaving global random state untouched."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure

    _, args, _ = prepared_measure_cases(measure)["viability_controls"]
    wells, qc = args
    original_global = np.random.get_state()
    figure = measure._viability_controls_figure(wells, qc)
    try:
        expected_rng = np.random.default_rng(0)
        for ax, readout in zip(figure.axes, ("viability", "cytotoxicity_index")):
            for position, role in enumerate(measure._VIABILITY_ROLES):
                values = wells.loc[wells.role == role, readout].to_numpy(dtype=float)
                x = position + expected_rng.uniform(-.15, .15, len(values))
                finite = np.isfinite(values)
                actual = next(mark for mark in ax.collections if mark.get_gid() == f"spacr-condition:{position}:points")
                np.testing.assert_array_equal(actual.get_offsets(), np.column_stack([x[finite], values[finite]]))
        current_global = np.random.get_state()
        assert original_global[0] == current_global[0]
        np.testing.assert_array_equal(original_global[1], current_global[1])
        assert original_global[2:] == current_global[2:]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("constant", [False, True])
def test_dense_wound_values_use_a_violin_without_points(constant):
    """The newly reached shared dense branch must also accept constant observations."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure

    values = np.ones(1000) if constant else np.linspace(1., 3., 1000)
    table = pd.DataFrame({"condition": ["a"] * len(values), "wound_ok": 1,
                          "half_closure_time": values, "time_unit": "h"})
    figure = measure._wound_half_closure_figure(table)
    try:
        ax = figure.axes[0]
        assert ax._spacr_condition_kinds == ("violin",)
        assert not any(str(mark.get_gid()).endswith(":points") for mark in ax.collections)
        np.testing.assert_array_equal(figure._spacr_data.half_closure_time, values)
        assert ax._spacr_measure_means == (table.half_closure_time.mean(),)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("ratios", [(1.8, .7), (.1, 1e-30), (1e20, .01)])
def test_hazard_bars_keep_estimates_intervals_order_and_no_effect_baseline(ratios):
    """Vertical presentation must retain exact Cox endpoints and the original reversed order."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure
    from matplotlib.container import ErrorbarContainer

    _, args, _ = prepared_measure_cases(measure)["hazard"]
    values = np.asarray(ratios)
    cox = args[0].assign(hazard_ratio=values, hr_lower=values * .75, hr_upper=values * 1.25)
    ordered = cox.iloc[::-1]
    figure = measure._hazard_ratio_figure(cox, args[1])
    try:
        ax = figure.axes[0]
        assert ax.get_yscale() == "log"
        assert [label.get_text() for label in ax.get_xticklabels()] == ordered.covariate.tolist()
        np.testing.assert_array_equal([bar.get_y() for bar in ax.patches], np.minimum(ordered.hazard_ratio, 1.))
        np.testing.assert_array_equal([bar.get_height() for bar in ax.patches], np.abs(ordered.hazard_ratio - 1.))
        for bar, ratio in zip(ax.patches, ordered.hazard_ratio):
            if ratio < 1.:
                assert bar.get_bbox().y0 == ratio
                assert bar.get_bbox().y0 > 0.
        error = next(item for item in ax.containers if isinstance(item, ErrorbarContainer))
        np.testing.assert_array_equal(error.lines[0].get_ydata(), ordered.hazard_ratio)
        assert all(hasattr(artist, '_spacr_reference_original_effects')
                   for artist in (error.lines[0], *error.lines[1], *error.lines[2]))
        for index, segment in enumerate(error.lines[2][0].get_segments()):
            np.testing.assert_allclose(segment[:, 1], ordered.iloc[index][["hr_lower", "hr_upper"]].to_numpy(dtype=float),
                                       rtol=5 * np.finfo(float).eps, atol=0)
        pd.testing.assert_frame_equal(figure._spacr_data, ordered)
    finally:
        plt.close(figure)


def test_dna_histogram_preserves_bins_density_model_components_and_gates():
    """Compare numerical artists with the supplied fit, without rerunning its estimator."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure

    _, args, _ = prepared_measure_cases(measure)["dna"]
    dna, fit, _, _ = args
    values = dna[np.isfinite(dna)]
    upper = max(6., float(np.percentile(values, 99.5)))
    heights, edges = np.histogram(values, bins=120, range=(0., upper), density=True)
    grid = np.linspace(0., upper, 600)
    densities = fit.densities(grid * fit.g1 / 2.) * fit.g1 / 2.
    figure = measure._dna_histogram_figure(*args)
    try:
        ax = figure.axes[0]
        np.testing.assert_array_equal([bar.get_height() for bar in ax.patches], heights)
        np.testing.assert_allclose([bar.get_x() for bar in ax.patches], edges[:-1], rtol=0, atol=1e-15)
        for index, line in enumerate(ax.lines[:3]):
            np.testing.assert_array_equal(line.get_xdata(), grid)
            np.testing.assert_array_equal(line.get_ydata(), densities[:, index])
        np.testing.assert_array_equal(ax.lines[3].get_ydata(), densities.sum(axis=1))
        np.testing.assert_array_equal([line.get_xdata()[0] for line in ax.lines[4:]], fit.c_value(fit.gates))
    finally:
        plt.close(figure)


def test_wound_curve_retains_mean_sd_and_every_well():
    """Existing SD bands and all well traces remain unchanged by appearance defaults."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure

    _, args, _ = prepared_measure_cases(measure)["wound_curve"]
    condition, wells = args
    figure = measure._wound_curve_figure(*args)
    try:
        ax = figure.axes[0]
        assert len(ax.lines) == 7
        for index, (name, block) in enumerate(condition.groupby('condition', sort=True)):
            mean_line = next(line for line in ax.lines if line.get_label().startswith(f"{name} (n="))
            np.testing.assert_array_equal(mean_line.get_xdata(), block.time)
            np.testing.assert_array_equal(mean_line.get_ydata(), 100 * block.relative_open_area)
            vertices = ax.collections[index].get_paths()[0].vertices
            spread = 100 * block.relative_open_area_sd.fillna(0)
            for time, mean, sd in zip(block.time, 100 * block.relative_open_area, spread):
                assert any(np.array_equal(point, [time, mean - sd]) for point in vertices)
                assert any(np.array_equal(point, [time, mean + sd]) for point in vertices)
        pd.testing.assert_frame_equal(figure._spacr_data, condition)
    finally:
        plt.close(figure)


def test_survival_retains_group_cap_all_source_rows_and_explicit_reference(monkeypatch):
    """The styling path cannot enlarge the chosen cohort or infer a control from labels."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure
    from spacr.figures.style import Palette

    monkeypatch.setattr(measure, '_TTE_FIGURE_GROUPS', 1)
    _, args, _ = prepared_measure_cases(measure)["survival"]
    figure = measure._time_to_event_figure(*args, reference='reference')
    try:
        ax = figure.axes[0]
        assert len(ax.lines) == 2
        assert ax.lines[0].get_color() == Palette.CONTROL
        np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1., .8, .6])
        np.testing.assert_array_equal(ax.lines[1].get_xdata(), [2.])
        pd.testing.assert_frame_equal(figure._spacr_data, args[0])
        assert 'first 1 shown' in ax.get_title()
    finally:
        plt.close(figure)


@pytest.fixture
def measure_dark_preferences(qapp, tmp_path, monkeypatch):
    """Exercise real saved dark preferences in a temporary settings store."""
    from PySide6.QtCore import QSettings
    from spacr.qt import preferences

    def temporary_settings():
        """Confine the public preference setters to this test's store."""
        return QSettings(str(tmp_path / "measure-figures.ini"), QSettings.Format.IniFormat)

    monkeypatch.setattr(preferences, "_settings", temporary_settings)
    preferences.set_figure_style({"background": "#181818", "foreground": "#F0F0F0"})
    preferences.set_figure_colors("#181818", "#F0F0F0")


@pytest.mark.parametrize("case", ["viability_controls", "survival", "hazard"])
def test_measure_labels_fit_without_overlap(case, measure_dark_preferences):
    """Measure the actual crowded labels identified in the saved-image review."""
    import matplotlib.pyplot as plt
    import spacr.measure as measure
    from spacr.figures.style import figure_problems

    name, args, kwargs = prepared_measure_cases(measure)[case]
    figure = getattr(measure, name)(*args, **kwargs)
    try:
        problems = figure_problems(figure)
        assert not [problem for problem in problems if 'overlap' in problem or 'cut off' in problem]
    finally:
        plt.close(figure)


@pytest.mark.parametrize("case", ["dna", "wound_half", "wound_plate", "viability_dose"])
def test_measure_dark_model_cores_and_keys_remain_readable(case, measure_dark_preferences):
    """Keep black scientific summaries backed and key chrome on the saved ground."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PathCollection
    from matplotlib.colors import to_rgba
    from matplotlib.text import Text
    import spacr.measure as measure
    from spacr.figures.style import _axis_colourbars

    name, args, kwargs = prepared_measure_cases(measure)[case]
    figure = getattr(measure, name)(*args, **kwargs)
    try:
        figure.canvas.draw()
        ax = figure.axes[0]
        assert to_rgba(figure.get_facecolor()) == to_rgba('#181818')
        assert figure.legends
        for legend in figure.legends:
            assert to_rgba(legend.get_frame().get_facecolor()) == to_rgba('#181818')
            assert all(to_rgba(text.get_color()) == to_rgba('#F0F0F0') for text in legend.get_texts())
        if case == 'wound_plate':
            keys = _axis_colourbars(ax)
            assert len(keys) == 1
            assert keys[0]._spacr_colourbar_source is ax.images[0]
            assert keys[0].get_ylabel() == 'Half-closure time (h)'
            texts = [text for text in keys[0].findobj(Text) if text.get_visible() and text.get_text().strip()]
            assert texts and all(to_rgba(text.get_color()) == to_rgba('#F0F0F0') for text in texts)
        else:
            if case == 'dna':
                summaries = [line for line in ax.lines if line.get_label() == 'model']
            elif case == 'wound_half':
                summaries = [item for item in ax.collections if isinstance(item, LineCollection)]
            else:
                summaries = [ax.lines[0]]
                observations = next(item for item in ax.collections if isinstance(item, PathCollection))
                supplied = args[0]['viability'].fits[0].result
                assert observations.get_zorder() > summaries[0].get_zorder()
                np.testing.assert_array_equal(observations.get_offsets(),
                                              np.column_stack((supplied.dose, supplied.response)))
                np.testing.assert_array_equal(observations.get_sizes(), [4])
                assert observations.get_alpha() == 0.6
            assert summaries
            for artist in summaries:
                colours = artist.get_colors() if isinstance(artist, LineCollection) else [to_rgba(artist.get_color())]
                assert all(tuple(colour[:3]) == (0., 0., 0.) for colour in colours)
                assert artist.get_path_effects()
    finally:
        plt.close(figure)
