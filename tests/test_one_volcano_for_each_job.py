"""Count the volcano renderers, so an eighth cannot appear unnoticed.

Instruction 127's strongest finding and the first line of its acceptance:
"One volcano renderer for saved figures, one for the interactive plot, and a
test that counts them so an eighth cannot appear."

WHY A COUNT IS THE RIGHT TEST. "hide my old volcano behind a boolean" was
reported as still broken on 2026-08-17 after being fixed on 2026-08-16,
because the fix gated one of three call sites and nobody had counted them.
An inventory that a test enforces is the thing that was missing; a
recommendation in a document is not.

TWO CORRECTIONS TO 127'S OWN INVENTORY, both measured on 2026-08-18 and both
recorded here rather than in prose nobody re-reads:

  * `regression_qc._panel_volcano_reference` is NOT a renderer. It draws no
    volcano at all -- ``set_axis_off`` and three lines of text pointing at
    the file the real one was written to. Counting it as one of seven made
    the duplication look worse than it is.
  * `volcano_style.render_volcano` is NOT unimported. 127 called it "the
    strongest single deletion candidate in the package" on the evidence that
    nothing in spacr/ imports it; `spacr/qt/widgets/volcano_explorer.py`
    does, and `spacr/qt/screens/volcano.py` names it in its own help text.
    Deleting it on the old evidence would have removed a live screen.
"""
from __future__ import annotations

import ast
import os

import pytest

SPACR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "spacr")

#: Every definition in spacr/ whose name says "volcano", with what it IS.
#:
#: A renderer draws the picture. Everything else either hosts one, names one,
#: or describes one -- and the difference is the entire point of the count.
INVENTORY = {
    ("figures/panels.py", "volcano"): "renderer: saved, house style — SURVIVOR",
    ("qt/widgets/fast_plots.py", "VolcanoPlot"):
        "renderer: interactive, pyqtgraph — SURVIVOR",
    ("plot.py", "volcano_plot"): "renderer: the legacy saved volcano",
    ("toxo.py", "custom_volcano_plot"):
        "renderer: also RETURNS the hit list the GT1 phenotype plot and the "
        "ME49 heatmap are built from, so it cannot simply go",
    ("volcano_style.py", "render_volcano"):
        "renderer: the volcano explorer's, driven by VolcanoStyle",
    ("guide_permutation.py", "plot_guide_permutation_volcano"):
        "renderer: a permutation null is not the screen's volcano",
    ("volcano_style.py", "VolcanoStyle"): "not a renderer: a style object",
    ("ml.py", "create_volcano_filename"): "not a renderer: a file name",
    ("qt/screens/volcano.py", "VolcanoScreen"): "not a renderer: hosts one",
    ("qt/widgets/volcano_explorer.py", "VolcanoExplorer"):
        "not a renderer: hosts render_volcano",
    ("qt/widgets/regression_results.py", "_redraw_volcano"):
        "not a renderer: asks a FastPlot to redraw",
    ("regression_qc.py", "_panel_volcano_reference"):
        "not a renderer: a text signpost pointing at the real one",
}

#: The two the maintainer asked for, one per job.
SURVIVORS = {
    ("figures/panels.py", "volcano"),
    ("qt/widgets/fast_plots.py", "VolcanoPlot"),
}


def _definitions():
    """Every top-level or nested def/class in spacr/ that says "volcano"."""
    found = set()
    for folder, subdirs, files in os.walk(SPACR):
        subdirs[:] = [d for d in subdirs if d != "__pycache__"]
        for name in files:
            if not name.endswith(".py"):
                continue
            path = os.path.join(folder, name)
            with open(path, encoding="utf-8") as handle:
                try:
                    tree = ast.parse(handle.read())
                except SyntaxError:                    # pragma: no cover
                    continue
            relative = os.path.relpath(path, SPACR).replace(os.sep, "/")
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                     ast.ClassDef)):
                    if "volcano" in node.name.lower():
                        found.add((relative, node.name))
    return found


def test_the_volcano_inventory_is_exactly_what_it_was_counted_to_be():
    """A new one is not forbidden; an UNRECORDED new one is.

    Adding a row here costs a sentence saying which job it does, which is the
    step that was skipped every previous time.
    """
    found = _definitions()
    added = found - set(INVENTORY)
    removed = set(INVENTORY) - found
    assert not added, (
        "a volcano definition appeared that nobody counted: "
        + ", ".join(f"{f}::{n}" for f, n in sorted(added)))
    assert not removed, (
        "a counted volcano definition is gone; say so in the inventory: "
        + ", ".join(f"{f}::{n}" for f, n in sorted(removed)))


def test_there_are_six_renderers_and_two_of_them_are_the_survivors():
    """The honest number. 127 said seven; one of the seven draws nothing."""
    renderers = {key for key, role in INVENTORY.items()
                 if role.startswith("renderer")}
    assert len(renderers) == 6
    assert SURVIVORS <= renderers
    survivors = {key for key, role in INVENTORY.items()
                 if role.endswith("SURVIVOR")}
    assert survivors == SURVIVORS, (
        "there is one saved volcano and one interactive volcano; a third "
        "survivor means the consolidation went backwards")


def test_the_qc_panel_draws_no_volcano():
    """It is a signpost. If it ever grows marks it becomes a seventh
    renderer, and the inventory above stops being true."""
    import inspect

    from spacr import regression_qc

    source = inspect.getsource(regression_qc._panel_volcano_reference)
    assert "set_axis_off" in source
    for drawing in ("scatter(", "ax.plot(", "hexbin(", "hist("):
        assert drawing not in source, (
            f"the volcano signpost started drawing ({drawing})")


def test_the_volcano_explorer_still_imports_the_renderer_127_called_dead():
    """The evidence that retired 127's strongest deletion recommendation.

    Kept as a test rather than a note because the recommendation is still in
    the instruction file, and whoever acts on it should fail here first.
    """
    import inspect

    explorer = pytest.importorskip("spacr.qt.widgets.volcano_explorer")
    source = inspect.getsource(explorer)
    assert "render_volcano" in source


def test_the_saved_survivor_and_the_interactive_one_agree_on_what_is_called():
    """One picture, two renderers, and they must call the same points.

    This is the failure 127 is really about -- "the two can and do disagree,
    which is how a figure in a paper stops matching the figure on screen".
    Both are handed the same table and the same alpha, and the set of genes
    each one calls is compared.
    """
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from spacr.figures.panels import volcano

    rng = np.random.default_rng(9)
    n = 150
    frame = pd.DataFrame({
        "gene": [f"g{i}" for i in range(n)],
        "coefficient": rng.normal(0, 0.5, n),
        "p_value": np.clip(rng.beta(0.4, 5, n), 1e-9, 1),
    })
    frame["adjusted_p_value"] = np.clip(frame["p_value"] * 3, 0, 1)

    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    from spacr.figures.style import ROLES

    figure, ax = plt.subplots()
    panel = volcano(ax, frame, alpha=0.05, effect_threshold=None)
    assert panel.drawn

    # COUNTED OFF THE PICTURE, not off the caption. The prose is generated
    # from the same numbers, so agreeing with it proves nothing; the marks
    # are what a reader sees and what a paper reproduces.
    called = 0
    for collection in ax.collections:
        colour = tuple(collection.get_facecolor()[0])
        if colour in (to_rgba(ROLES["up"]), to_rgba(ROLES["down"])):
            called += collection.get_offsets().shape[0]
    plt.close(figure)

    # The house rule, stated here so BOTH renderers can be checked against it
    # rather than against each other's implementation: a point is called when
    # its adjusted p is at or below alpha.
    expected = int((frame["adjusted_p_value"] <= 0.05).sum())
    assert called == expected, (
        f"the saved volcano called {called} of {expected}; its hit rule is "
        f"not 'adjusted p <= alpha'")
    assert f"{expected} called" in panel.caption, (
        "the caption and the marks disagree about how many were called")
