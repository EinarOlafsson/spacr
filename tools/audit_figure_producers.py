"""Audit every spaCR figure producer for N658 parity.

A *producer* is a place in ``spacr/`` that creates a Matplotlib figure:
``plt.figure``, ``plt.subplots``, ``plt.subplot_mosaic``, a Matplotlib
``Figure(...)`` (aliases followed), a seaborn figure-level call, the house
factories, and the implicit ones -- a seaborn axes-level plot or a
``plt.subplot`` call with no ``ax=`` in a function that makes no figure of
its own, which draws on (and if needed creates) pyplot's current figure.

For each producer the audit records four things (item 658):

* ``registers``  -- whether its scope registers tidy data and a plot spec
  (or a picture's arrays) through
  ``spacr.figures.bundle._register_figure_data``;
* ``styled_by``  -- how the user's Preferences figure settings reach it
  (``spacr.figures.style._apply_user_style``, applied by ``save_figure``,
  by the figure queue for every pyplot figure, by every menu canvas, or by
  ``figure_style``'s rcParams);
* ``saved_by``   -- the writer its files go through (``save_figure``, F572's
  ``_checked_savefig``, or a writer named in the one-writer guard);
* ``menu``       -- where the right-click menu (Edit, Change graph type,
  Statistics, Save zip) reaches it: the figure queue for pyplot figures,
  the shared menu canvas for embedded ones.

The routes are followed through the scope chain (a method's class
included) and, when the figure is returned, up to three levels of callers
matched by function name. A producer has parity when it registers, is
styled and is reachable by the menu (or is a figure written only to file, or
returned to an API caller); :data:`EXEMPT` names the figures that never
reach a user as a graph. ``tests/test_every_figure_producer_has_parity.py``
fails when a producer lacks parity.

Usage::

    python tools/audit_figure_producers.py [--out producers.csv]
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import pathlib
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[1]
ROOT = REPO / "spacr"
CSV_PATH = (REPO / "features" / "data" / "658_producer_parity_2026-10-10"
            / "producers.csv")

HELPER = "_register_figure_data"

FIELDS = ("module", "function", "line", "call", "family", "registers",
          "styled_by", "saved_by", "menu", "parity", "reason")

#: Producers whose figure never reaches a user as a graph: a factory that
#: hands an empty figure to a caller (the caller is audited instead), a
#: scratch figure drawn to measure something, or a figure rasterised
#: straight to pixels. Shared with tests/test_every_figure_registers_its_data.
EXEMPT = {
    ("figures/style.py", "_figure_axes"):
        "the house subplots factory; every caller is audited instead",
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
    ("qt/widgets/picture_export.py", "save_picture"):
        "a scratch figure wrapping an already rendered picture so the "
        "figure-integrity check can read it, never shown",
}

_FACTORY_NAMES = ("Figure", "_figure_axes", "setup_plot",
                  "create_results_figure")
_SEABORN_FIGURES = {"clustermap", "jointplot", "pairplot", "catplot",
                    "relplot", "displot", "lmplot"}
_SEABORN_AXES = {"barplot", "boxplot", "violinplot", "stripplot",
                 "swarmplot", "pointplot", "boxenplot", "countplot",
                 "histplot", "kdeplot", "ecdfplot", "rugplot", "scatterplot",
                 "lineplot", "regplot", "residplot", "heatmap"}

_CANVAS_NAMES = {"_canvas_class", "panel_canvas_class",
                 "_OwnedTimerFigureCanvas", "FigureCanvasQTAgg"}
_SAVE_NAMES = {"save_figure"}
_RASTER_NAMES = {"figure_to_pixmap", "render_figure_to_png"}
_PICTURE_MENU = "_show_rendered_figure_menu"
_GENERIC = {"__init__", "run", "main", "draw", "render", "plot", "save",
            "write", "_write", "show", "build", "update", "refresh",
            "panel", "_draw", "_render", "_plot", "_build_ui", "export"}


def _call_name(call: ast.Call) -> str:
    """The bare name a call is made by (``a.b.c()`` gives ``c``)."""
    func = call.func
    if isinstance(func, ast.Name):
        return func.id
    return getattr(func, "attr", "")


def _base(call: ast.Call) -> str:
    """The module alias a call is made on (``plt`` for ``plt.figure()``)."""
    func = call.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id
    return ""


def explicit_family(call: ast.Call, figure_names) -> str:
    """The kind of explicit figure creation ``call`` is, or ``""``."""
    func = call.func
    if isinstance(func, ast.Name):
        if func.id in figure_names:
            return "Figure"
        if func.id in _FACTORY_NAMES and func.id != "Figure":
            return "factory"
        return ""
    if not isinstance(func, ast.Attribute):
        return ""
    base = _base(call)
    if base == "plt":
        reopens = (func.attr == "figure" and call.args
                   and not isinstance(call.args[0], ast.Constant))
        if func.attr in ("subplots", "figure", "subplot_mosaic") and not reopens:
            return "pyplot"
        return ""
    if base == "sns":
        return "seaborn-figure" if func.attr in _SEABORN_FIGURES else ""
    if func.attr in _FACTORY_NAMES and base != "self":
        return "factory"
    return ""


def implicit_family(call: ast.Call) -> str:
    """An axes-level call that draws on pyplot's current figure, or ``""``."""
    keywords = {kw.arg for kw in call.keywords}
    base = _base(call)
    func = call.func
    if not isinstance(func, ast.Attribute):
        return ""
    if None in keywords:
        return ""
    if base == "sns" and func.attr in _SEABORN_AXES and "ax" not in keywords:
        return "seaborn-implicit"
    if base == "plt" and func.attr == "subplot":
        return "pyplot-implicit"
    return ""


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


class _Scope:
    """What one function, class or module does: its calls and its returns.

    ``calls`` includes the calls of every scope nested inside it, and
    ``.savefig`` when it calls ``<something>.savefig`` itself.
    """

    def __init__(self, node, module):
        self.node = node
        self.module = module
        self.name = getattr(node, "name", "")
        self.calls = set()
        self.returns = False
        self.explicit = False


class _Module:
    """One parsed module, indexed in a single pass over its tree."""

    def __init__(self, path: pathlib.Path, root: pathlib.Path = ROOT):
        self.path = path
        self.name = path.relative_to(root).as_posix()
        self.tree = ast.parse(path.read_text(encoding="utf-8"))
        self.figure_names = set()
        self.aliases = {}
        self.top = _Scope(self.tree, self)
        self.scopes = {}
        self.direct = []
        self.candidates = []
        self._walk(self.tree, (), self.top)
        self.calls = self.top.calls
        for call, stack in self.candidates:
            if explicit_family(call, self.figure_names):
                for scope in stack:
                    self.scopes[id(scope)].explicit = True

    def _walk(self, node, stack, info) -> None:
        """Index ``node``'s children into ``info``, the scope they sit in."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ImportFrom):
                for alias in child.names:
                    if (child.module in ("matplotlib.figure",
                                         "matplotlib.pyplot")
                            and alias.name == "Figure"):
                        self.figure_names.add(alias.asname or alias.name)
                    if alias.asname and alias.asname != alias.name:
                        self.aliases[alias.asname] = alias.name
            if isinstance(child, _SCOPES):
                record = _Scope(child, self)
                self.scopes[id(child)] = record
                self._walk(child, stack + (child,), record)
                info.calls |= record.calls
                continue
            if isinstance(child, ast.Call):
                name = _call_name(child)
                info.calls.add(name)
                if (isinstance(child.func, ast.Attribute)
                        and child.func.attr == "savefig"):
                    info.calls.add(".savefig")
                self.candidates.append((child, stack))
                if stack:
                    self.direct.append((name, info))
            elif (isinstance(child, ast.Return) and child.value is not None
                  and not isinstance(info.node, ast.ClassDef)):
                info.returns = True
            self._walk(child, stack, info)


def scan(root: pathlib.Path = ROOT) -> list:
    """Every producer under ``root`` with its parity columns.

    :param root: the package directory to scan.
    :returns: one dict per producer, keyed by :data:`FIELDS`.
    """
    modules = [_Module(path, root) for path in sorted(root.rglob("*.py"))
               if "i18n_catalogs" not in str(path)]
    callers = defaultdict(list)
    functions = defaultdict(list)
    for module in modules:
        for name, info in module.direct:
            callers[name].append(info)
        for record in module.scopes.values():
            if not isinstance(record.node, ast.ClassDef):
                functions[record.name].append(record)

    saving = {"save_figure"}
    for _round in range(2):
        for name, records in functions.items():
            if (name in _GENERIC or name in saving
                    or not ("save" in name or "write" in name)):
                continue
            if any(record.calls & saving for record in records):
                saving.add(name)
    wrappers = {"save_figure": saving,
                "_apply_user_style": {"_apply_user_style",
                                      "render_figure_to_png"} | saving,
                "_checked_savefig": {"_checked_savefig"} | {
                    name for name, records in functions.items()
                    if any("_checked_savefig" in record.calls
                           for record in records)}}

    def reach(scopes, depth=3, seen=None) -> set:
        """Call names reached from ``scopes`` and, for a returned figure,
        from the functions that call them."""
        seen = set() if seen is None else seen
        names = set()
        for scope in scopes:
            if id(scope) in seen:
                continue
            seen.add(id(scope))
            aliases = scope.module.aliases
            calls = scope.calls | {aliases[name] for name in scope.calls
                                   if name in aliases}
            names |= calls
            for target, known in wrappers.items():
                if calls & known:
                    names.add(target)
            if depth and scope.returns and scope.name not in _GENERIC:
                up = [c for c in callers.get(scope.name, [])
                      if c is not scope]
                names |= reach(up, depth - 1, seen)
        return names

    shared_canvas_styles = any(
        module.name == "qt/widgets/graph_builder.py"
        and "_style_new_content" in module.calls for module in modules)
    unique = {}
    for module in modules:
        for call, stack in module.candidates:
            family = explicit_family(call, module.figure_names)
            if not family and not any(module.scopes[id(s)].explicit
                                      for s in stack):
                family = implicit_family(call)
            if not family:
                continue
            row = _row(module, [module.scopes[id(s)] for s in stack], call,
                       family, reach, shared_canvas_styles)
            implicit = family.endswith("implicit")
            key = (row["module"], row["function"], row["family"],
                   "" if implicit else row["line"])
            unique.setdefault(key, row)
    return sorted(unique.values(),
                  key=lambda r: (r["module"], int(r["line"])))


def _row(module, stack, call, family, reach, shared_canvas_styles) -> dict:
    """The parity row for one producer call."""
    qualname = ".".join(s.name for s in stack)
    inner = stack[-1].name if stack else ""
    functions_only = [s for s in stack
                      if not isinstance(s.node, ast.ClassDef)]
    names = reach(functions_only) if functions_only else set(module.calls)
    owner = set()
    for scope in stack:
        if isinstance(scope.node, ast.ClassDef):
            owner |= scope.calls
    names = names | (owner & {HELPER, "_apply_user_style"})
    if owner & _CANVAS_NAMES and not names & _SAVE_NAMES:
        names = names | (owner & _CANVAS_NAMES)
    func = call.func
    label = (func.id if isinstance(func, ast.Name)
             else f"{_base(call) or '?'}.{func.attr}")
    pyplot = family in ("pyplot", "seaborn-figure", "pyplot-implicit",
                        "seaborn-implicit")
    exempt = EXEMPT.get((module.name, inner))
    registered = HELPER in names
    registers = ("yes" if registered else
                 "exempt" if exempt else "no")

    saved = []
    if "save_figure" in names:
        saved.append("save_figure")
    if "_checked_savefig" in names:
        saved.append("_checked_savefig")
    if ".savefig" in names:
        saved.append("savefig (one-writer guard)")
    if not saved:
        saved.append("not written; menu Save zip")

    rendered = bool(names & _RASTER_NAMES) and module.name.startswith("qt/")
    menu = []
    if names & _CANVAS_NAMES:
        menu.append("menu canvas")
    elif rendered and _PICTURE_MENU in module.calls:
        menu.append("rendered-picture menu")
    elif rendered:
        pass
    elif pyplot or "publish" in names:
        menu.append("figure queue")
    elif "save_figure" in names:
        menu.append("file only")
    elif stack and not inner.startswith("_") and stack[-1].returns:
        menu.append("returned to the API caller")

    styled = []
    if "save_figure" in names:
        styled.append("save_figure")
    if "_apply_user_style" in names:
        styled.append("_apply_user_style")
    if "render_figure_to_png" in names or "figure queue" in menu:
        styled.append("figure queue render")
    if "figure_style" in names:
        styled.append("figure_style rcParams")
    if "_canvas_class" in names and shared_canvas_styles:
        styled.append("menu canvas draw")

    reason = ""
    if exempt:
        parity, reason = "exempt", exempt
        styled, saved, menu = ["n/a"], ["n/a"], ["n/a"]
    elif registered and styled and menu:
        parity = "ok"
    else:
        parity = "missing"
    return {"module": module.name, "function": qualname or "<module>",
            "line": str(call.lineno), "call": label, "family": family,
            "registers": registers,
            "styled_by": "; ".join(dict.fromkeys(styled)),
            "saved_by": "; ".join(dict.fromkeys(saved)),
            "menu": "; ".join(dict.fromkeys(menu)),
            "parity": parity, "reason": reason}


def render(rows) -> str:
    """The CSV text for ``rows``."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main(argv=None) -> int:
    """Write the producer CSV and print the parity counts."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(CSV_PATH),
                        help="where the CSV is written")
    args = parser.parse_args(argv)
    rows = scan()
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")
    counts = defaultdict(int)
    for row in rows:
        counts[row["parity"]] += 1
    print(f"wrote {out}: {len(rows)} producers, "
          + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    return 1 if counts.get("missing") else 0


if __name__ == "__main__":
    sys.exit(main())
