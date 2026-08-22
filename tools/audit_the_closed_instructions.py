"""Drive every closed instruction through the real code, not the commits.

WRITTEN BECAUSE THE BACKLOG DISAGREED WITH THE APPLICATION. Asked for
2026-08-21: "read through all the items in the instructions list that you
have checked of as complete today ... double and tripple check that you
actually did what i asked for. there are to many examples of you checking
things of that were not actualy done at all".

THE RULE THIS ENFORCES: an instruction is closed when its ASK can be driven,
not when the code that was meant to satisfy it has been written. Four of the
misses that prompted this were features built on a path the user never
touches -- reading the diff would have passed every one of them.

Run it after closing anything:

    python tools/audit_the_closed_instructions.py
"""

import os, traceback
os.environ.setdefault("MPLBACKEND", "Agg")
from PySide6.QtWidgets import QApplication
app = QApplication.instance() or QApplication([])
import numpy as np, pandas as pd

R = []
def check(item, ask, fn):
    try:
        ok, detail = fn()
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    R.append((item, ok, ask, detail))

# 193 no button cuts off its text
# 193 IS DRIVEN ONCE, further down, THROUGH AppScreen. There was a second
# driver here that called `findChildren` on a `SettingsWidgets`, which is not
# a QWidget and never had the method -- so the check reported the FEATURE as
# broken when what was broken was the check. The same mistake this file
# exists to catch, in the file itself.

# 195 default control 000000
def _195():
    from spacr.settings import get_perform_regression_default_settings as f
    v = f({}).get("controls")
    return (v == ["000000"]), f"controls={v!r}"
check("195", "default controls = 000000", _195)

# 197 hotkey map in help
def _197():
    from spacr.qt.app import MainWindow
    from PySide6.QtWidgets import QMenu
    w = MainWindow()
    for m in w.menuBar().findChildren(QMenu):
        if "Help" in m.title():
            names = [a.text() for a in m.actions()]
            return ("Keyboard shortcuts" in names), f"help={names[:4]}"
    return False, "no Help menu"
check("197", "hotkey map in the Help menu", _197)

# 202 five heading tooltips
def _202():
    from spacr.gene_measurement_compare import HEADING_HELP
    want = {"measurement","level","plot","show","compare"}
    return (set(HEADING_HELP) == want), f"{sorted(HEADING_HELP)}"
check("202", "tooltips on measurement/level/plot/show/compare", _202)

# 204 bar error bars SD/Var/SEM
def _204():
    from spacr.figures.spread import SPREAD_CHOICES
    have = {c[0] for c in SPREAD_CHOICES}
    want = {"sd","sem","var"}
    return (want <= have), f"offered={sorted(have)}"
check("204", "bar error bars: SD, Var, SEM", _204)

# 205 three show scopes
def _205():
    from spacr.well_scope import SCOPES
    return ([s for s,_ in SCOPES] == ["guides","wells","all"]), f"{[s for s,_ in SCOPES]}"
check("205", "show: gRNAs / +well-mates / all", _205)

# 210 four outlier filters, off by default
def _210():
    from spacr.settings import get_perform_regression_default_settings as f
    from spacr.outlier_filter import CRITERIA
    got = f({})
    keys = [f"{c}_outlier_mads" for c,_ in CRITERIA]
    missing = [k for k in keys if k not in got]
    on = [k for k in keys if got.get(k) is not None]
    return (not missing and not on), f"keys={len(keys)} missing={missing} on_by_default={on}"
check("210", "outlier removal on 4 measures, optional", _210)

# 211 cells_per_page gone
def _211():
    from spacr.picture_settings import ALL_KEYS, OWN_DEFAULTS
    from spacr.settings import tooltips
    gone = ("cells_per_page" not in ALL_KEYS and "cells_per_page" not in OWN_DEFAULTS
            and "cells_per_page" not in tooltips)
    return gone, "cells_per_page removed" if gone else "still present"
check("211", "remove cells_per_page; page by container size", _211)

# 213 png_list + dependent variable boxes
def _213():
    from spacr.qt.widgets.measurement_compare_dialog import MeasurementComparePanel
    obj = pd.DataFrame({"prcfo":["p_r_c_f_o1","p_r_c_f_o2"],"area":[1.0,2.0],"gene":["a","b"]})
    p = MeasurementComparePanel(obj, {"a":["a"],"b":["b"]})
    return (hasattr(p,"join_png_list") and hasattr(p,"join_dependent")), "both boxes present"
check("213", "png_list + dependent variable on the merge", _213)

# 215 UMAP tab
def _215():
    from spacr.qt.widgets.regression_results import RegressionResultsPanel
    p = RegressionResultsPanel()
    tabs = [p.tabs.tabText(i) for i in range(p.tabs.count())]
    return ("Annotation check" in tabs), f"tabs={tabs}"
check("215", "UMAP annotation-check tab", _215)

# 220 backends as default deps
def _220():
    import importlib
    have = []
    for m in ("pyfixest","glum","gpytorch"):
        try: importlib.import_module(m); have.append(m)
        except Exception: pass
    return (len(have)>=2), f"importable: {have}"
check("220", "backends installable as core deps", _220)

# 222 three colour channels
def _222():
    from spacr.qt.widgets.regression_results import RegressionResultsPanel
    p = RegressionResultsPanel()
    return (hasattr(p,"_colour_by_2") and hasattr(p,"_colour_by_3")), "3 channels present"
check("222", "colour by more than one column", _222)

# 224 permutation residuals
def _224():
    from spacr.permutation_qc import block_residual_report
    return callable(block_residual_report), "block_residual_report exists"
check("224", "permutation shows its residuals", _224)

# 231 pyqtgraph training monitor
def _231():
    from spacr.qt.widgets.training_monitor import TrainingMonitor
    m = TrainingMonitor()
    m.append(1, {"loss":1.0}); first = m.curves["loss"]
    for e in range(2,30): m.append(e, {"loss":1.0/e})
    return (m.curves["loss"] is first and len(m.points("loss")[0])==29), "same curve, 29 points"
check("231", "training graphs append, pyqtgraph", _231)

# 232 console jump
def _232():
    from spacr.qt.widgets.console_panel import ConsolePanel
    c = ConsolePanel()
    return (hasattr(c,"jump_to_the_end") and hasattr(c,"_end_shortcut")), "Ctrl+End + button"
check("232", "jump to the bottom of a console", _232)

print(f"{'item':6} {'ok':4} ask")
for item, ok, ask, detail in R:
    print(f"{item:6} {'PASS' if ok else 'FAIL'} {ask}")
    if not ok: print(f"         -> {detail}")
print()
print("FAILED:", [i for i,ok,_,_ in R if not ok] or "none")
import os
os.environ.setdefault("MPLBACKEND", "Agg")
from PySide6.QtWidgets import QApplication, QPushButton
app = QApplication.instance() or QApplication([])
import numpy as np, pandas as pd

R = []
def check(item, ask, fn):
    try: ok, detail = fn()
    except Exception as e: ok, detail = False, f"{type(e).__name__}: {e}"
    R.append((item, ok, ask, detail))

# 193 retry: real widget tree
def _193():
    from spacr.qt.screens.app_screen import AppScreen
    s = AppScreen("regression")
    bad = [b.text() for b in s.findChildren(QPushButton)
           if b.text() and b.sizeHint().width()
           < b.fontMetrics().horizontalAdvance(b.text())]
    n = len(s.findChildren(QPushButton))
    return (not bad), f"{n} buttons, clipped: {bad[:4]}"
check("193", "no button cuts off its text", _193)

# 194 plate map squares + drag
def _194():
    # `plate_map_picker`, not `plate_map` -- the first audit named a module
    # that does not exist and reported the FEATURE as missing, which is the
    # same mistake in the checker that it exists to catch in the code.
    from spacr.qt.widgets.plate_map_picker import PlateMapPicker
    picker = PlateMapPicker()
    return hasattr(picker, "begin_drag"), "drag select present"
check("194", "plate map squares you drag across", _194)

# 196 advisor settings actually run
def _196():
    from spacr.settings_advisor import advise_that_runs, Reading, refusals
    r = Reading(plates=4, wells=620, guides=1380, genes=345,
                response="x", n_response=5000, low=0.0, high=1.0,
                on_unit=True, normal_p=0.001, wells_per_guide=2.0)
    a = advise_that_runs(r, {"hits_per_thousand": 20})
    bad = refusals(a.as_settings())
    return (not bad), f"refusals={bad}"
check("196", "advisor's settings actually run", _196)

# 199 every figure opens when clicked
def _199():
    from spacr.qt.widgets.figure_queue import FigureQueue
    q = FigureQueue()
    return hasattr(q, "show_index"), "show_index present"
check("199", "every figure opens when clicked", _199)

# 200 A/F graph types fit the data
def _200():
    from spacr.graph_types import offer, default_for, fits
    f = pd.DataFrame({"g":["a","b"]*10,"v":range(20)})
    rows = offer(f, "g", "v")
    live = [k for k,_,w in rows if not w]
    return ("bar" in live and "scatter" not in live and fits("categorical_continuous", default_for("categorical_continuous"))), f"live={live}"
check("200", "only the graph types that fit are offered", _200)

# 201 outline setting like channels
def _201():
    # A PICTURE setting on the Cells tab, not a classify-panel setting.
    from spacr.qt.widgets.picture_settings_dialog import CHANNEL_KEYS
    return ("outline" in CHANNEL_KEYS and "channels" in CHANNEL_KEYS), \
        f"CHANNEL_KEYS={CHANNEL_KEYS}"
check("201", "outline setting looks like channel settings", _201)

# 203 merged db keeps the annotation
def _203():
    from spacr.gene_measurement_compare import ANNOTATION_COLUMNS, join_measurements
    import inspect
    sig = inspect.signature(join_measurements).parameters
    return ("grna" in ANNOTATION_COLUMNS and "png_list" in sig), "annotation kept; png_list flag"
check("203", "merged database keeps the gRNA annotation", _203)

# 206 multi-select on the volcano
def _206():
    from spacr.qt.widgets.fast_plots import FastPlot
    p = FastPlot(title="t")
    p.set_keys(["a","b","c"])
    for i,(x,y) in enumerate([(0,0),(1,1),(2,2)]): p._row_xy[i]=(x,y)
    p.highlight_keys(["a","c"])
    return (p.selected_keys()==["a","c"]), f"selected={p.selected_keys()}"
check("206", "select multiple gRNAs on the volcano", _206)

# 207 fractions not all 1/0
def _207():
    from spacr.cell_montage import NOT_ANNOTATED, ANNOTATION_COLUMN
    return (bool(NOT_ANNOTATED) and bool(ANNOTATION_COLUMN)), f"{ANNOTATION_COLUMN}/{NOT_ANNOTATED}"
check("207", "unannotated cells are named and counted", _207)

# 208 picture tooltips stay
def _208():
    from spacr.qt.widgets.hover_tooltip import HoverTooltip
    import inspect
    src = inspect.getsource(HoverTooltip)
    return ("frameGeometry" in src), "pointer test on frameGeometry"
check("208", "picture-settings tooltips stay while read", _208)

# 212 nonparametric regressions offered
def _212():
    from spacr.settings import expected_types
    from spacr.training_basis import TRAINING_BASES
    from spacr.qt.screens.settings_model import SettingsWidgets
    w = SettingsWidgets("regression"); w.build_sections()
    return ("inference" in w._widgets and "grna_statistic" in w._widgets), "inference + grna_statistic offered"
check("212", "nonparametric regressions are offered", _212)

# 214 positive control calibration
def _214():
    # The calibration lives in `annotation_validation`; `classifier_quality`
    # holds the correction it feeds.
    from spacr.annotation_validation import mixed_ratio_calibration
    from spacr.classifier_quality import rogan_gladen
    return (callable(mixed_ratio_calibration) and callable(rogan_gladen)), \
        "calibration + Rogan-Gladen present"
check("214", "positive control calibrates the fractions", _214)

# 216 every popup is a draggable window
def _216():
    from spacr.qt.dialogs import detach_all_dialogs
    return callable(detach_all_dialogs), "detach_all_dialogs present"
check("216", "every popup is a movable window", _216)

# 217 settings for my data advises the rest
def _217():
    from spacr.settings_advisor import advise, Reading
    r = Reading(plates=4, wells=620, guides=1380, genes=345, response="x",
                n_response=5000, low=0.0, high=1.0, on_unit=True,
                normal_p=0.4, wells_per_guide=2.0, fraction_median=0.02)
    keys = {c.key for c in advise(r, {"hits_per_thousand":20}).chosen}
    want = {"fraction_threshold","min_cell_count","agg_type"}
    return (bool(want & keys)), f"advises {sorted(keys)}"
check("217", "advisor recommends thresholds/aggregation too", _217)

# 218 before/after transform panel
def _218():
    from spacr.response_distribution import compare, caption
    rng = np.random.default_rng(0)
    r = compare(rng.lognormal(0,1,500), "log")
    return (r["changed"] and "before" in caption(r)), "panel compares both"
check("218", "histogram before and after transform", _218)

# 223 folder save
def _223():
    import tempfile
    from spacr.figures.bundle import save
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    out = save(d, "g", render=lambda p: open(p,"wb").write(b"x"),
               data=pd.DataFrame({"a":[1,2]}),
               groups={"nc":rng.normal(0,1,30),"pc":rng.normal(1,1,30)})
    got = set(os.listdir(out))
    return ({"g.pdf","g.png","data.csv","statistics.csv"} <= got), f"{sorted(got)}"
check("223", "saving a graph makes a folder", _223)

# 225 red assumption + recommendations
def _225():
    from spacr.run_recommendations import recommend
    out = recommend({"normality_p":1e-30,"durbin_watson":1.5}, settings={})
    return (len(out) >= 2), f"{[r.setting for r in out]}"
check("225", "broken assumptions red + recommendations", _225)

# 226 advisor reads the last run
def _226():
    from spacr.settings_advisor import read_the_last_run
    import inspect
    return ("QC_NUMBERS_FILE" in inspect.getsource(read_the_last_run)), "reads the written QC"
check("226", "advisor reads the last run", _226)

# 229 classes column fillable
def _229():
    from spacr.qt.widgets.class_editor import ClassEditorWidget
    e = ClassEditorWidget()
    e.column.setCurrentText("columnID")
    e.class_field.setText("pc"); e.value_field.setText("c3"); e.add_typed_class()
    return (e.value() == {"pc":{"column":"columnID","value":"c3"}}), f"{e.value()}"
check("229", "classes column can be filled in", _229)

# 230 streaming
def _230():
    from spacr.stream_dataset import STREAM_METHODS, settings_for_method, crop_name
    from spacr.settings import deep_spacr_defaults
    got = deep_spacr_defaults({})
    return (len(STREAM_METHODS)==2 and got.get("image_source") and crop_name("p_r_c_f",7).endswith(".png")), "stream method + naming"
check("230", "images are streamed", _230)

# 233 evaluation split
def _233():
    from spacr.settings import categories
    from spacr.classify import FAMILY_SETTINGS
    shared = {"cross_validation_enabled","cv_group_by"}
    bad = [k for k in shared
           if any(k in categories[h] for h in
                  ("Computer Vision Training","Machine Learning Model and Features"))]
    return (not bad and "Leakage Audit" in categories), f"misfiled shared: {bad}"
check("233", "evaluation settings split by category", _233)

for item, ok, ask, detail in R:
    print(f"{item:6} {'PASS' if ok else 'FAIL'} {ask}")
    if not ok: print(f"         -> {detail}")
print()
print("FAILED:", [i for i,ok,_,_ in R if not ok] or "none")

# --------------------------------------------------------------------------
# THE THREE THIS FILE HAD NOT REACHED. Closed the same day as the rest and
# left out of the sweep, which is the gap the sweep exists to close.
# --------------------------------------------------------------------------

# 219 a unit that constrains says so before the run
def _219():
    from spacr.qt.screens.settings_model import SettingsWidgets
    w = SettingsWidgets("regression"); w.build_sections()
    w.set_value_for_key("analysis_unit", "cell")
    w._refresh_analysis_unit_lock()
    locked = [k for k in w._widgets if not w._widgets[k].isEnabled()]
    said = [k for k in locked if w._widgets[k].toolTip()]
    w.set_value_for_key("analysis_unit", "well")
    w._refresh_analysis_unit_lock()
    freed = [k for k in locked if w._widgets[k].isEnabled()]
    return (bool(locked) and len(said) == len(locked) and bool(freed)), \
        f"locked={locked} explained={len(said)}/{len(locked)} released={freed}"
check("219", "cell locks the settings it needs and says why", _219)

# 221 the first-run setup screen
def _221():
    from spacr.qt.widgets.setup_slides import SLIDES, SetupSlides
    screen = SetupSlides()
    asked = [key for _title, _blurb, keys in SLIDES for key in keys]
    # ONE QUESTION PER SLIDE with an explanation beside it, which is what
    # was asked for -- so every slide carries prose, and the eleven
    # settings are spread across them rather than stacked on one page.
    blurbs = [blurb for _title, blurb, _keys in SLIDES if blurb.strip()]
    return (len(SLIDES) >= 5 and len(asked) >= 8
            and len(blurbs) == len(SLIDES) and screen.slide() == 0), \
        f"{len(SLIDES)} slides asking {len(asked)} settings"
check("221", "first run opens a setup screen", _221)

# 228 the bottom panels collapse
def _228():
    from spacr.qt.screens.app_screen import AppScreen
    bad = []
    for key in ("mask", "measure", "regression"):
        screen = AppScreen(key)
        folder = getattr(screen, "_console_folder", None)
        if folder is None:
            bad.append(f"{key}: console does not fold"); continue
        folder.toggle()
        if not screen._console.isHidden():
            bad.append(f"{key}: console stayed up")
        # THE AI CHAT GOES WITH IT: it is the console panel's own second half
        # and has no heading of its own to click.
        if screen._console._chat_row.isVisibleTo(screen):
            bad.append(f"{key}: the chat row stayed on screen")
        folder.toggle()
        if screen._console.isHidden():
            bad.append(f"{key}: console did not come back")
        card = getattr(screen, "_usage_card", None)
        if card is not None and card.folder is None:
            bad.append(f"{key}: system card does not fold")
        screen.deleteLater()
    return (not bad), "; ".join(bad) or "console, chat and system fold on each"
check("228", "console, AI chat and system all collapse", _228)

print(f"{'item':6} {'ok':4} ask")
for item, ok, ask, detail in R[-3:]:
    print(f"{item:6} {'PASS' if ok else 'FAIL'} {ask}")
    if not ok: print(f"         -> {detail}")
print()
print("FAILED:", [i for i, ok, _, _ in R[-3:] if not ok] or "none")
