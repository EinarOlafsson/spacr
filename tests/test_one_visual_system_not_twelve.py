"""Instruction 136 — the old matplotlib figures get the house style.

    "also overhaul the old matplotlib figures to look better in accordance
     with the figure making skill"

Two checks this instruction asks for BY NAME, and one it implies.

THE COUNT HAS FLIPPED. When 136 was filed it was 133 raw `plt.subplots(` /
`plt.figure(` against 4 uses of `figure_style`. It is the other way round now,
and this holds the direction — the same argument as 145's reader ratchet: a
partial conversion silently un-converts itself as new code is written, and a
figure drawn outside the house style does not FAIL, it just looks like a
different application.

NO MODULE WRITES rcParams GLOBALLY. `plt.rcParams` is process-wide, so a
module that themes one figure through it themes every LATER figure in the
process — including one being saved for paper. `spacr.figure_style` is the one
place allowed to, because applying the house style globally is its job.
"""
from __future__ import annotations

import collections
import pathlib
import re

import pytest

RAW_FIGURE = re.compile(r"plt\.subplots\(|plt\.figure\(")
STYLE_USE = re.compile(r"figure_style|from \.figures\.style|rc_params\(")

#: Writes that reach process-wide state. `rc_context` is the scoped form and
#: is not one.
GLOBAL_RC = re.compile(r"(?:plt|mpl|matplotlib)\.rcParams\s*(?:\.update\(|\[)")

#: The ceiling on raw figure creation, measured 2026-08-20. This counts every
#: `plt.subplots(` / `plt.figure(` in the tree, styled or not, so it caps how
#: many figures spaCR draws by hand -- it does NOT say whether they are in the
#: house style. `UNSTYLED_CEILING` below is the one that says that.
#:
#: 145 -> 146 on 2026-08-20, for `gene_measurement_compare.render_comparison`
#: -- the second implementation of instruction 108's renderer contract, and
#: the thing that proves the shared style base has more than one user. It
#: draws INSIDE `figure_style`, so `UNSTYLED_CEILING` is untouched at 0.
#: Raising THIS number is allowed for a new figure that is properly styled;
#: raising the other one is not.
# 146 -> 147 for ``ml._show_response_distribution``. The diagnostic panel
# creates its axes inside
# ``figure_style(theme_target())``; the unstyled ceiling remains zero.
CEILING = 147

#: The ceiling on figure creations that are NOT inside the house style.
#:
#: ZERO, as of 2026-08-20: 136's "STILL OPEN: the 145 remaining raw call
#: sites" is closed. Measured with the AST walk in `_unstyled`, not a regex,
#: because the regex above cannot see a `with` block and counted the
#: converted sites as raw -- which is why it stood at 145 while the real
#: number fell to 48 and then to 0.
#:
#: A new figure has to open `figure_style(theme_target())` before it is
#: created. If one genuinely cannot -- `utils.setup_plot` is the case, since
#: it exists to draw in the GUI THEME's colours and a house-style context
#: nested inside would win and repaint it for print -- say so where the
#: exception is taken and raise this deliberately.
UNSTYLED_CEILING = 0

#: Context managers that ARE the house style, for :func:`_unstyled`.
STYLE_CONTEXTS = {"figure_style", "rc_context"}

#: Allowed to write rcParams globally, and why.
MAY_SET_RCPARAMS = {
    # Applying the house style process-wide IS this module's job -- it is
    # what `figure_style.apply()` means.
    "figure_style.py",
}


def _spacr() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "spacr"


def _files():
    for path in sorted(_spacr().rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        yield path, path.read_text(encoding="utf-8")


def test_the_checker_is_reading_the_tree_it_thinks_it_is():
    assert (_spacr() / "figure_style.py").is_file()
    assert sum(len(RAW_FIGURE.findall(text)) for _p, text in _files()) > 0


def test_raw_figure_creation_does_not_go_up():
    counts = collections.Counter()
    for path, text in _files():
        found = len(RAW_FIGURE.findall(text))
        if found:
            counts[path.name] = found
    total = sum(counts.values())
    assert total <= CEILING, (
        f"{total} raw figure creations, up from {CEILING}. A figure drawn "
        f"outside the house style does not fail -- it just looks like a "
        f"different application. Route it through spacr.figure_style, or "
        f"lower this ceiling deliberately.\n"
        + "\n".join(f"  {n:3}  {name}" for name, n in counts.most_common(8)))


def _named(call) -> str:
    return getattr(call.func, "attr", getattr(call.func, "id", ""))


def _style_wrappers(tree) -> set:
    """Same-module context managers that open the house style themselves.

    `toxo._house` is one: it opens `plt.rc_context` with the house rc and
    yields. Its six figures are styled, and a checker that only looked for
    the literal name would report every one of them as raw.
    """
    import ast

    found = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(getattr(d, "attr", getattr(d, "id", "")) == "contextmanager"
                   for d in node.decorator_list):
            continue
        if any(isinstance(inner, ast.With)
               and any(isinstance(i.context_expr, ast.Call)
                       and _named(i.context_expr) in STYLE_CONTEXTS
                       for i in inner.items)
               for inner in ast.walk(node)):
            found.add(node.name)
    return found


def _unstyled(text: str) -> list:
    """Line numbers where a figure is created outside the house style.

    AST, NOT A REGEX. rcParams reach an artist when it is CREATED, so what
    matters is whether the creation is lexically inside a style context --
    which is a question about the tree and not about the text.
    """
    import ast

    tree = ast.parse(text)
    allowed = STYLE_CONTEXTS | _style_wrappers(tree)
    raw, spans = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.With) and any(
                isinstance(i.context_expr, ast.Call)
                and _named(i.context_expr) in allowed for i in node.items):
            spans.append((node.lineno, node.end_lineno))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("subplots", "figure", "subplot_mosaic")
                and getattr(node.func.value, "id", "") == "plt"
                and not any(a <= node.lineno <= b for a, b in spans)):
            raw.append(node.lineno)
    return sorted(raw)


def test_the_unstyled_checker_can_tell_the_two_apart():
    """Guard the guard: a checker that reported zero because it found
    nothing would pass this file's real test silently."""
    inside = "import matplotlib.pyplot as plt\nwith figure_style('screen'):\n    f = plt.subplots()\n"
    outside = "import matplotlib.pyplot as plt\nf = plt.subplots()\n"

    assert _unstyled(inside) == []
    assert _unstyled(outside) == [2]


def test_the_checker_follows_a_module_s_own_context_manager():
    """`toxo._house` opens `rc_context` and yields; its figures are styled."""
    text = (
        "import contextlib\n"
        "import matplotlib.pyplot as plt\n"
        "@contextlib.contextmanager\n"
        "def _house(w):\n"
        "    with plt.rc_context({}):\n"
        "        yield\n"
        "def draw():\n"
        "    with _house(6):\n"
        "        return plt.subplots()\n")

    assert _unstyled(text) == []


def test_no_figure_is_created_outside_the_house_style():
    """136's last open item. 133 to 4 when it was filed; 0 raw now."""
    found = {}
    for path, text in _files():
        try:
            lines = _unstyled(text)
        except SyntaxError:                      # pragma: no cover - defensive
            continue
        if lines:
            found[str(path.relative_to(_spacr().parent))] = lines
    total = sum(len(v) for v in found.values())
    assert total <= UNSTYLED_CEILING, (
        f"{total} figure(s) created outside `figure_style`, up from "
        f"{UNSTYLED_CEILING}. rcParams reach an artist when it is CREATED, "
        f"so the context has to be open BEFORE plt.subplots -- a context "
        f"opened after it leaves the spines, ticks and labels at whatever "
        f"the caller's globals happened to be.\n"
        + "\n".join(f"  {name}: {lines}" for name, lines in found.items()))


def test_the_house_style_is_used_more_than_it_is_bypassed():
    """133 to 4 when this was filed. The direction is the whole point."""
    raw = sum(len(RAW_FIGURE.findall(text)) for _p, text in _files())
    styled = sum(len(STYLE_USE.findall(text)) for _p, text in _files())
    assert styled > raw, f"{styled} style uses against {raw} raw figures"


def test_no_module_writes_rcParams_globally():
    offenders = {}
    for path, text in _files():
        if path.name in MAY_SET_RCPARAMS:
            continue
        for line in text.splitlines():
            if GLOBAL_RC.search(line) and "rcParams[" in line and "=" not in line.split("rcParams[")[1][:40]:
                continue        # a READ, e.g. plt.rcParams["axes.prop_cycle"]
            if GLOBAL_RC.search(line) and (".update(" in line or "] =" in line):
                offenders.setdefault(path.name, []).append(line.strip()[:70])
    assert not offenders, (
        "these write matplotlib's PROCESS-WIDE state, so every later figure "
        "in the process inherits it -- including one saved for paper. Use "
        f"`matplotlib.rc_context` instead:\n{offenders}")


@pytest.mark.parametrize("black", [True, False])
def test_setup_plot_leaves_no_colour_behind_it(black):
    """The one that was leaking, held directly."""
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from spacr.utils import setup_plot

    watched = ("figure.facecolor", "axes.facecolor", "text.color",
               "xtick.color", "ytick.color", "axes.labelcolor",
               "axes.edgecolor")
    before = {key: plt.rcParams[key] for key in watched}

    figure, axes = setup_plot(4, black_background=black)
    try:
        assert {key: plt.rcParams[key] for key in watched} == before
        # And the figure itself still carries the theme, which is the point
        # of the function.
        assert axes.get_facecolor() is not None
    finally:
        plt.close(figure)
