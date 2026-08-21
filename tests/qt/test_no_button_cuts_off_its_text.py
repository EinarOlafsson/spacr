"""193: it is impossible for a button to cut off its own text.

    "some buttons like the plate button cut off the text. make it impossible
    for any button to cut of anny text."

A RULE ABOUT THE CLASS, not about the button that prompted it -- and a rule
is worth having because the failure is INVISIBLE TO WHOEVER WROTE IT. The
author sees their own label fit; the button is only wrong in a language they
do not read, or at a font size they do not use.

The reported one: "Plate…" was `setFixedWidth(58)`, measured by eye against
the English. The German for it is "Teller..." -- a character longer -- and
every other translation is longer than the English too. spaCR ships nine.
"""
from __future__ import annotations

import ast
import pathlib

import pytest

pytest.importorskip("PySide6")

from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QApplication, QPushButton

pytestmark = pytest.mark.qt

#: The calls that make a width an INPUT rather than a consequence.
PINNING = ("setFixedWidth", "setMaximumWidth", "setFixedSize")


def _qt_root() -> pathlib.Path:
    import spacr.qt

    return pathlib.Path(spacr.qt.__file__).parent


#: How a button is constructed. A width pinned on anything else -- a panel,
#: a progress bar, a spin box -- is not this instruction's business.
BUTTON_TYPES = ("QPushButton", "QToolButton")


def _button_names(tree) -> set:
    """Every name in ``tree`` that is assigned a button.

    `self._plate_button = QPushButton(...)` and `button = QPushButton(...)`
    both count, and the attribute is remembered by its last component so
    `self._x` and `_x` are the same button.
    """
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not (isinstance(value, ast.Call)
                and getattr(value.func, "id",
                            getattr(value.func, "attr", "")) in BUTTON_TYPES):
            continue
        # A BUTTON WITH NO TEXT HAS NOTHING TO CUT. `QPushButton()` with no
        # label is a swatch or an icon; pinning its size is describing a
        # picture, which is what a fixed size is for.
        first = value.args[0] if value.args else None
        if not (isinstance(first, ast.Constant) and str(first.value or "")):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def _receiver(node) -> str:
    """The last name in the chain a call is made on."""
    value = node.func.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return ""


def _pinned_widths(text: str) -> list:
    """Line numbers where a BUTTON's width is pinned to a literal.

    Two things are deliberately not flagged.

    A CAP COMPUTED FROM `sizeHint()` is not pinning: it is a preferred size
    with the text as its floor, which is the rule rather than a breach of
    it. The list-reorder arrows are that case -- 30 px keeps them compact
    and the hint keeps them readable.

    AND A WIDTH ON ANYTHING THAT IS NOT A BUTTON. A panel, a progress bar
    and a spin box all pin widths in files that also build buttons, and
    flagging them would make this check about layout in general rather than
    about text being cut off.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:                              # pragma: no cover
        return []
    buttons = _button_names(tree)
    found = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in PINNING):
            continue
        if _receiver(node) not in buttons:
            continue
        argument = ast.dump(ast.Module(body=[ast.Expr(a) for a in node.args],
                                       type_ignores=[]))
        if "sizeHint" in argument:
            continue
        found.append(node.lineno)
    return sorted(found)


#: Files allowed to pin a width, and why. Every entry is a widget that is not
#: a button carrying a label -- a swatch, a rule, a fixed-size cell.
ALLOWED = {
    "widgets/plate_map_picker.py": "the wells of a plate map are squares "
                                   "standing for physical wells (194); their "
                                   "size is the picture, not a label",
    "screens/experiment_design.py": "the same, for the design screen's map",
}


def test_a_textless_button_is_not_flagged():
    """A swatch or an icon button has nothing to cut, and pinning its size
    is describing a picture."""
    assert _pinned_widths("b = QPushButton()\nb.setFixedSize(44, 20)") == []
    assert _pinned_widths("b = QPushButton('')\nb.setFixedWidth(20)") == []


def test_the_checker_can_tell_a_literal_from_a_hint():
    """Guard the guard: a checker finding nothing would pass silently."""
    pinned = "b = QPushButton('x')\nb.setFixedWidth(58)\n"
    assert _pinned_widths(pinned) == [2]

    hinted = ("b = QPushButton('x')\n"
              "b.setMaximumWidth(max(30, b.sizeHint().width()))\n")
    assert _pinned_widths(hinted) == []

    assert _pinned_widths("b = QPushButton('x')\n# b.setFixedWidth(58)") == []


def test_the_checker_ignores_what_is_not_a_button():
    """A panel, a progress bar and a spin box all pin widths in files that
    also build buttons."""
    assert _pinned_widths("panel = QWidget()\npanel.setMaximumWidth(330)") == []
    assert _pinned_widths("self.setFixedWidth(200)") == []


def test_it_finds_a_button_kept_on_self():
    text = ("self._plate = QPushButton('Plate')\n"
            "self._plate.setFixedWidth(58)\n")

    assert _pinned_widths(text) == [2]


def test_no_button_file_pins_a_width_to_a_number():
    strays = {}
    for path in sorted(_qt_root().rglob("*.py")):
        if "i18n_catalogs" in str(path):
            continue
        name = str(path.relative_to(_qt_root()))
        if name in ALLOWED:
            continue
        text = path.read_text(encoding="utf-8")
        if "QPushButton" not in text and "QToolButton" not in text:
            continue
        lines = _pinned_widths(text)
        if lines:
            strays[name] = lines

    assert not strays, (
        "these pin a width to a literal in a file that builds buttons, so a "
        "longer label -- a translation, a larger font -- is cut off:\n"
        + "\n".join(f"  {n}: {ls}" for n, ls in strays.items())
        + "\n\nA button's width is a CONSEQUENCE of its text. Let the layout "
          "have its sizeHint, or cap it with max(n, sizeHint().width()).")


def test_every_exception_says_why():
    for name, why in ALLOWED.items():
        assert len(why) > 30, name


# --------------------------------------------------------------------------- #
#  And on the built screens, in the longest language spaCR ships
# --------------------------------------------------------------------------- #

#: The screens checked. Not every module -- that is the smoke test's job --
#: but the ones that carry the buttons this was reported against, plus a
#: broad one.
SCREENS = ("regression", "measure", "mask", "illumination")

#: Every language with a catalog. The point of checking them all is that the
#: author reads one of them.
LANGUAGES = ("en", "de", "fr", "es", "is", "sv", "pt", "hi", "ko", "zh_CN")


def _longest(label: str) -> tuple:
    """The longest translation of ``label``, and the language it is in."""
    from spacr.qt.i18n import tr

    best, where = label, "en"
    for language in LANGUAGES:
        try:
            got = str(tr(label, language=language))
        except Exception:                            # noqa: BLE001
            continue
        if len(got) > len(best):
            best, where = got, language
    return best, where


@pytest.mark.parametrize("app_key", SCREENS)
def test_no_button_is_narrower_than_its_own_text(qtbot, app_key):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    QApplication.processEvents()

    clipped = []
    for button in screen.findChildren(QPushButton):
        if not button.text() or not button.isVisible():
            continue
        if button.width() < button.sizeHint().width():
            clipped.append(f"{button.text()!r} "
                           f"{button.width()} < {button.sizeHint().width()}")

    assert not clipped, f"{app_key}: " + "; ".join(clipped)


@pytest.mark.parametrize("app_key", SCREENS)
def test_no_button_paints_elided_text(qtbot, app_key):
    """The direct check, and it catches what a width comparison misses when
    a layout has already squeezed the button."""
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen(app_key)
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    QApplication.processEvents()

    from PySide6.QtCore import Qt

    elided = []
    for button in screen.findChildren(QPushButton):
        text = button.text()
        if not text or not button.isVisible():
            continue
        # The room the label actually has, after the frame's own padding.
        room = button.width() - 12
        painted = QFontMetrics(button.font()).elidedText(
            text, Qt.ElideRight, max(room, 0))
        if painted != text:
            elided.append(f"{text!r} -> {painted!r}")

    assert not elided, f"{app_key}: " + "; ".join(elided)


def test_the_reported_button_fits_its_longest_translation(qtbot):
    """"Plate…" is "Teller..." in German -- a character longer than the
    English the 58 px was measured against."""
    from PySide6.QtCore import Qt

    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    screen.resize(1400, 900)
    screen.show()
    QApplication.processEvents()

    button = next((b for b in screen.findChildren(QPushButton)
                   if b.text().startswith("Plate")), None)
    if button is None:
        pytest.skip("no plate button on this screen")

    longest, language = _longest(button.text())
    metrics = QFontMetrics(button.font())
    assert metrics.horizontalAdvance(longest) <= button.width() - 8, (
        f"the {language} label {longest!r} needs "
        f"{metrics.horizontalAdvance(longest)} px and the button is "
        f"{button.width()}")
