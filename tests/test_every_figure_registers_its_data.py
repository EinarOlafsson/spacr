"""Every spaCR figure registers what it was drawn from.

The right-click figure menu changes the graph type, runs the statistics
and saves one zip with the data, the tests and a re-create script. All of
that needs the tidy data and plot spec the figure was drawn from, and a
picture needs its arrays and metadata. Both arrive through ONE helper,
``spacr.figures.bundle._register_figure_data``, called where the figure is
made.

This guard finds every place that creates a figure (``plt.subplots``,
``plt.figure``, ``Figure(...)``, seaborn's figure-level functions and the
house ``_figure_axes``) and fails when neither the function making it, a
function around it, nor (for a method) its class calls that helper. The
only way past it is the short list below.
"""
from __future__ import annotations

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "spacr"

HELPER = "_register_figure_data"

#: Functions that hand an empty figure to a caller that registers it, or
#: whose figure is never shown: a scratch figure drawn to measure something,
#: or one rasterised straight to an array.
EXEMPT = {
    ("figures/style.py", "_figure_axes"):
        "the house subplots factory; every caller is checked instead",
    ("utils.py", "setup_plot"):
        "returns an empty figure and axes to the caller, which draws",
    ("figures/panels.py", "available"):
        "a scratch figure each panel is test-drawn on and closed, never shown",
    ("qt/widgets/motility_preview.py", "render_motility_figure"):
        "rasterised to an RGB array for a preview label, never a figure",
    ("qt/widgets/regression_results.py", "judge_homogeneity"):
        "a scratch figure the scale-location statistics are read from",
    ("report.py", "write_pdf"):
        "PDF report pages assembled from images already saved, never shown",
    ("timelapse.py", "create_results_figure"):
        "returns an empty figure and axes to the caller, which draws",
}

_FACTORY_NAMES = ("Figure", "_figure_axes", "setup_plot",
                  "create_results_figure")

_SEABORN_FIGURES = {"clustermap", "jointplot", "pairplot", "catplot",
                    "relplot", "displot", "lmplot"}


def _figure_names(tree) -> set:
    """Names a module binds to Matplotlib's ``Figure`` class, aliases included."""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module in (
                "matplotlib.figure", "matplotlib.pyplot"):
            names.update(alias.asname or alias.name for alias in node.names
                         if alias.name == "Figure")
    return names


def _is_creation(call: ast.Call, figure_names=("Figure",)) -> bool:
    """Whether ``call`` makes a new Matplotlib figure.

    ``plt.figure(number)`` with a positional argument re-opens a figure that
    already exists, and a bare ``Figure(...)`` counts only where the name is
    Matplotlib's (another module's ``Figure`` is a record of a file).
    """
    func = call.func
    if isinstance(func, ast.Name):
        return func.id in figure_names or (
            func.id in _FACTORY_NAMES and func.id != "Figure")
    if not isinstance(func, ast.Attribute):
        return False
    base = func.value.id if isinstance(func.value, ast.Name) else ""
    if base == "plt":
        reopens = (func.attr == "figure" and call.args
                   and not isinstance(call.args[0], ast.Constant))
        return (func.attr in ("subplots", "figure", "subplot_mosaic")
                and not reopens)
    if base == "sns":
        return func.attr in _SEABORN_FIGURES
    return func.attr in _FACTORY_NAMES and base != "self"


def _calls_helper(node) -> bool:
    """Whether ``node``'s body calls the registration helper anywhere."""
    for inner in ast.walk(node):
        if isinstance(inner, ast.Call):
            func = inner.func
            name = (func.id if isinstance(func, ast.Name)
                    else getattr(func, "attr", ""))
            if name == HELPER:
                return True
    return False


def unregistered_sites(root: pathlib.Path = ROOT) -> list:
    """``(module, function, line)`` for each figure made without registering.

    :param root: the package directory to scan.
    :returns: the offending sites, sorted.
    """
    missing = []
    for path in sorted(root.rglob("*.py")):
        module = path.relative_to(root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        figure_names = _figure_names(tree)

        def visit(node, stack):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                      ast.ClassDef)):
                    visit(child, stack + [child])
                    continue
                if isinstance(child, ast.Call) and _is_creation(
                        child, figure_names):
                    name = ".".join(s.name for s in stack)
                    inner = stack[-1].name if stack else ""
                    if (module, inner) not in EXEMPT and not any(
                            _calls_helper(scope) for scope in stack):
                        missing.append((module, name, child.lineno))
                visit(child, stack)

        visit(tree, [])
    return sorted(set(missing))


def test_every_figure_registers_its_data_or_image():
    """No figure-producing call site skips the registration helper."""
    missing = unregistered_sites()
    assert not missing, (
        "These make a figure without calling "
        f"spacr.figures.bundle.{HELPER} (tidy data and spec for a plot, the "
        "arrays for a picture), so the figure menu cannot change its type, "
        "test it or save its data:\n"
        + "\n".join(f"  spacr/{m}:{line} {name}" for m, name, line in missing))


def test_the_guard_sees_an_unregistered_figure(tmp_path):
    """The scan flags a bare figure and passes a registered one."""
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "bare.py").write_text(
        "import matplotlib.pyplot as plt\n"
        "def draw(df):\n"
        "    fig, ax = plt.subplots()\n"
        "    ax.plot(df.x, df.y)\n"
        "    return fig\n")
    (package / "good.py").write_text(
        "import matplotlib.pyplot as plt\n"
        "from spacr.figures.bundle import _register_figure_data\n"
        "class Panel:\n"
        "    def __init__(self):\n"
        "        from matplotlib.figure import Figure\n"
        "        self.fig = Figure()\n"
        "    def draw(self, df):\n"
        "        _register_figure_data(self.fig, df, x='x', y='y')\n")
    assert unregistered_sites(package) == [("bare.py", "draw", 3)]


def test_every_factory_exemption_still_exists():
    """An exemption naming a function that is gone is removed, not kept."""
    for module, function in EXEMPT:
        source = (ROOT / module).read_text(encoding="utf-8")
        assert f"def {function}(" in source, (module, function)
