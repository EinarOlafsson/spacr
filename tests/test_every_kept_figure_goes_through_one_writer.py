"""108 point 6: the format and resolution preferences reach every kept figure.

    "EVERY SAVE GOES THROUGH `spacr.plot.save_figure`, which honours the
     user's figure-format preference. A literal '.pdf' in a savefig call is a
     complaint this project has already had twice."

WHAT A LEAK COSTS is not obvious from the call site, which is why they
accumulated: `fig.savefig(path, format='pdf', dpi=600)` looks careful. It is a
preference written into a line of code — a user who chose PNG at 300 gets
neither, and a figure saved from a dark-themed session is white ink on a white
page, because `print_ready` never ran on it.

THE COUNT IS A CEILING, and the exceptions are NAMED rather than counted. A
bare number would be satisfied by deleting a figure; naming them means a new
`savefig` has to argue for itself in the same file it is written in.
"""
from __future__ import annotations

import ast
import pathlib

import pytest


def _spacr() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "spacr"


#: Every `savefig` that is NOT a leak, and the reason. Keyed by module path
#: relative to `spacr/`; the value is how many that file is allowed and why.
#:
#: ADD TO THIS ONLY WITH THE REASON WRITTEN AT THE CALL SITE TOO. A dict entry
#: is a promise that a comment beside the code makes the same argument.
ALLOWED = {
    "plot.py": (1, "`save_figure` itself -- the one writer"),
    "report.py": (2, "PdfPages.savefig: pages of a multi-page book, not a "
                     "file a format preference can name"),
    "gene_measurement_sweep.py": (
        1, "`_write` IS the sweep's export rule and has already flipped the "
           "figure ground, each axes ground and every piece of chrome; the "
           "shared writer would repaint a second time"),
    "qt/widgets/save_figure_dialog.py": (
        1, "the user chose the format, the DPI and the page in this dialog "
           "and is looking at the preview; overriding them would write "
           "something other than what was previewed"),
    "qt/widgets/figure_settings.py": (
        1, "SVG and EPS are not in plot.FIGURE_FORMATS and would have their "
           "extension rewritten under the user; png/pdf already route"),
    "qt/widgets/figure_queue.py": (
        4, "the vector source the queue rasterises, the screen raster into a "
           "temp dir, and a BytesIO drag preview -- none is a file a user "
           "keeps, and the print rule would flash every figure to a light "
           "page a moment after it appeared"),
    "qt/screens/app_screen.py": (1, "a BytesIO thumbnail; never a file"),
    "qt/screens/hyperparam.py": (1, "a BytesIO thumbnail; never a file"),
}

#: Files with no entry above are allowed none.
CEILING = sum(count for count, _why in ALLOWED.values())


def _savefig_calls(text: str) -> list:
    """Line numbers of `*.savefig(...)`, from the TREE.

    Not a regex: this instruction's own count was taken by eye twice and was
    wrong both times -- it said 21 when there were 23 -- because a `savefig`
    inside a docstring reads exactly like one in code.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:                          # pragma: no cover
        return []
    return sorted(node.lineno for node in ast.walk(tree)
                  if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute)
                  and node.func.attr == "savefig")


def _found() -> dict:
    out = {}
    for path in sorted(_spacr().rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        lines = _savefig_calls(path.read_text(encoding="utf-8"))
        if lines:
            out[str(path.relative_to(_spacr()))] = lines
    return out


def test_the_checker_can_see_a_savefig():
    """Guard the guard: a checker finding nothing would pass silently."""
    assert _savefig_calls("fig.savefig('a.png')") == [1]
    assert _savefig_calls('"""fig.savefig(x)"""') == [], "prose was counted"


def test_no_new_savefig_bypasses_the_one_writer():
    found = _found()
    strays = {name: lines for name, lines in found.items()
              if name not in ALLOWED}

    assert not strays, (
        "these `savefig` calls bypass `spacr.plot.save_figure`, so the "
        "user's figure-format and resolution preferences do not reach them "
        "and nothing repaints them for paper:\n"
        + "\n".join(f"  {name}: {lines}" for name, lines in strays.items())
        + "\n\nRoute them through `save_figure`, or add the file to ALLOWED "
          "with the reason -- and write the same reason at the call site.")


def test_no_allowed_file_grows_a_second_one_quietly():
    found = _found()
    for name, (count, why) in ALLOWED.items():
        lines = found.get(name, [])
        assert len(lines) <= count, (
            f"{name} has {len(lines)} savefig call(s), up from {count}. The "
            f"ones already there are allowed because: {why}. A new one is "
            f"not covered by that reason.")


def test_the_total_does_not_go_up():
    total = sum(len(lines) for lines in _found().values())

    assert total <= CEILING, f"{total} savefig calls, up from {CEILING}"


def test_every_exception_says_why():
    for name, (_count, why) in ALLOWED.items():
        assert len(why) > 30, name


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_every_exception_still_exists(name):
    """An entry for a file that has stopped calling savefig is a ceiling
    nobody is under, and it would hide the next one added there."""
    assert name in _found(), (
        f"{name} no longer calls savefig; remove it from ALLOWED")
