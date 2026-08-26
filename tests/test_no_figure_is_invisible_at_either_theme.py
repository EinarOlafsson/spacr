"""178's first checklist item, as a check rather than as an eye test.

    "Switch to the light theme and back: no figure has invisible text at
     either setting, and nothing is hard-coded black or white."

A HARD-CODED INK COLOUR DOES NOT FAIL. It draws, the figure appears, and the
label is simply not there to read -- the worst kind of fault, because a
screenshot taken on the theme it happens to suit looks perfect. The two found
by this check are exactly that shape: a verdict box hard-coded white whose
text follows the theme (so on the dark theme it was white on white), and a
`black` threshold line invisible on the same theme.

WHAT IS ALLOWED IS NAMED, NOT COUNTED. Text drawn ON a microscopy image is
white because the image is dark, and that has nothing to do with the theme; a
contact sheet is a black page on purpose. A bare number would be satisfied by
deleting a figure, and would not say which colours were the deliberate ones.

AN EXCEPTION MUST STILL EXIST. `test_every_exception_still_exists` is what
keeps this list honest, and it is the reason the two Tk entries are gone: the
Tk front end was deleted, so `gui_core.py` and `gui_elements.py` hold no ink
at all any more and the allowance for them was describing files that are not
there.
"""
from __future__ import annotations

import ast
import collections
import pathlib

import pytest

#: The spellings of black and white matplotlib accepts.
BLACK_OR_WHITE = frozenset({"k", "w", "black", "white", "#000", "#000000",
                            "#fff", "#ffffff", "#FFF", "#FFFFFF"})

#: The keyword arguments that set ink.
INK_ARGUMENTS = frozenset({"color", "c", "edgecolor", "ec", "facecolor",
                           "fc", "labelcolor", "textcolor", "foreground"})

#: Files allowed hard-coded black or white, how many, and WHY.
#:
#: Every entry here is a colour that is not about the theme. Add one only
#: with the reason, and only when the colour is genuinely fixed by what it is
#: drawn on rather than by where the figure is going.
ALLOWED = {
    "plot.py": (7, "object labels and a scale bar drawn ON a microscopy "
                   "image, plus a montage grid that is deliberately black; "
                   "an image is not the theme's ground"),
    "io.py": (4, "the montage sheet is a black page with white filenames on "
                 "it, by design -- it is a contact sheet, not a plot"),
    "utils.py": (4, "predicted-class labels drawn ON the image they "
                    "describe"),
    "measure.py": (2, "a black figure ground behind image panels"),
    "qt/annotate_engine.py": (1, "drawn onto the annotated microscopy image itself, not onto the page"),
    "qt/widgets/motility_preview.py": (1, "an overlay drawn onto a video frame, not onto a figure"),
}

CEILING = sum(count for count, _why in ALLOWED.values())


def _spacr() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parent.parent / "spacr"


def _hard_coded(text: str) -> list:
    """Line numbers where an ink argument is literally black or white."""
    try:
        tree = ast.parse(text)
    except SyntaxError:                          # pragma: no cover
        return []
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for keyword in node.keywords:
            if (keyword.arg in INK_ARGUMENTS
                    and isinstance(keyword.value, ast.Constant)
                    and str(keyword.value.value) in BLACK_OR_WHITE):
                found.append(keyword.value.lineno)
    return sorted(found)


def _found() -> dict:
    out = {}
    for path in sorted(_spacr().rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        lines = _hard_coded(path.read_text(encoding="utf-8"))
        if lines:
            out[str(path.relative_to(_spacr()))] = lines
    return out


def test_the_checker_can_see_one():
    """Guard the guard: a checker finding nothing would pass silently."""
    assert _hard_coded("ax.plot(x, y, color='black')") == [1]
    assert _hard_coded("ax.plot(x, y, color=ROLES['reference'])") == []
    assert _hard_coded("# ax.plot(x, y, color='black')") == []


def test_no_new_figure_hard_codes_its_ink():
    strays = {name: lines for name, lines in _found().items()
              if name not in ALLOWED}

    assert not strays, (
        "these set ink to a literal black or white, so the figure has "
        "invisible text at one of the two theme settings:\n"
        + "\n".join(f"  {name}: {lines}" for name, lines in strays.items())
        + "\n\nUse ROLES['reference'] for a threshold or guide line, "
          "resolve_ink(theme_target()) for text, and "
          "resolve_label_ground(theme_target()) for what goes behind it. "
          "Or add the file to ALLOWED with the reason it is not about the "
          "theme.")


def test_no_allowed_file_grows_another_quietly():
    found = _found()
    for name, (count, why) in ALLOWED.items():
        lines = found.get(name, [])
        assert len(lines) <= count, (
            f"{name} hard-codes ink {len(lines)} time(s), up from {count}. "
            f"The ones already there are allowed because: {why}")


def test_the_total_does_not_go_up():
    total = sum(len(lines) for lines in _found().values())

    assert total <= CEILING, f"{total} hard-coded ink colours, up from {CEILING}"


def test_every_exception_says_why():
    for name, (_count, why) in ALLOWED.items():
        assert len(why) > 30, name


@pytest.mark.parametrize("name", sorted(ALLOWED))
def test_every_exception_still_exists(name):
    assert name in _found(), f"{name} no longer hard-codes ink; drop it"


class TestTheTwoFaultsThisFound:

    def test_a_verdict_box_follows_the_theme(self):
        """It was white while its text is the theme ink -- which on the dark
        theme IS white. The verdict was drawn, was there, unreadable."""
        from spacr.figures.style import resolve_label_ground

        assert resolve_label_ground("print") != resolve_label_ground("screen")

    def test_the_label_ground_is_the_inks_opposite_number(self):
        from spacr.figures.style import resolve_ink, resolve_label_ground

        for target in ("screen", "print"):
            ink = resolve_ink(target).lstrip("#")
            ground = resolve_label_ground(target).lstrip("#")
            light = lambda h: sum(int(h[i:i + 2], 16) for i in (0, 2, 4)) / 3
            assert abs(light(ink) - light(ground)) > 100, target

    def test_an_explicit_ground_still_wins(self):
        from spacr.figures.style import resolve_label_ground

        assert resolve_label_ground("screen", "#123456") == "#123456"

    def test_a_threshold_line_is_the_reference_role(self):
        """Grey reads on both themes; black reads on one."""
        from spacr.figures.style import ROLES

        assert ROLES["reference"] not in ("black", "#000000", "white",
                                          "#FFFFFF")
