"""Dragging a settings splitter and Ctrl + wheel over a column stay smooth.

The tutorial capture of lesson 05 (clip 14_resize_panels, Mask Generation)
showed the app stop painting while the settings splitter was dragged and the
right-hand column's text was sized with Ctrl + wheel. Measured offscreen at
2560x1440 with the field animating, on 2026-10-13:

* every move of a drag rewrote the whole preferences file to remember the
  pane sizes, 7.3 ms of the 11.5 ms the move held the GUI thread;
* every notch of the wheel restyled the whole column, 130-180 ms a notch,
  about half of it Qt looking up fallback fonts for the two glyphs Open Sans
  lacks (``▾`` and ``↓``) at each new size.

Now a drag writes the sizes once, when the handle is let go or rests (a move
made in code still writes at once), and a turn of the wheel restyles the
column once, when the wheel rests. The end state is the same: the same sizes,
the same sheet, the same pixels, and both are saved.

Timing is compared with a reference measured in the same test on the same
machine (the write a move used to make, a restyle a notch used to make), so a
loaded runner slows both sides alike.
"""
from __future__ import annotations

import statistics
import time

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QEvent, QPoint, QPointF, QSettings, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent, QWheelEvent                 # noqa: E402
from PySide6.QtWidgets import (QApplication, QLabel, QPushButton,  # noqa: E402
                               QVBoxLayout, QWidget)

from spacr.qt.widgets import collapsible_splitter as cs            # noqa: E402


def _pump(n: int = 6) -> None:
    for _ in range(n):
        QApplication.processEvents()


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A private preferences file the size of a real one."""
    from spacr.qt import preferences

    path = tmp_path / "spacr-smooth.ini"
    monkeypatch.setattr(preferences, "_settings",
                        lambda: QSettings(str(path), QSettings.IniFormat))
    settings = preferences._settings()
    for i in range(400):
        settings.setValue(f"filler/key{i:03d}", "x" * 40)
    settings.sync()
    return preferences


@pytest.fixture
def writes(monkeypatch):
    """Every pane-size write, as ``(key, extents)``."""
    seen = []
    real = cs.set_pane_extents

    def spy(key, extents):
        seen.append((key, dict(extents)))
        real(key, extents)

    monkeypatch.setattr(cs, "set_pane_extents", spy)
    return seen


def _labels(parent, count, text):
    column = QVBoxLayout(parent)
    for i in range(count):
        column.addWidget(QLabel(f"{text} {i} ▾ …", parent))
    return parent


def _splitter(qtbot, key):
    split = cs.CollapsibleSplitter(Qt.Horizontal, persist_key=key)
    split.add_pane(_labels(QWidget(), 12, "Setting"), "Settings",
                   mode=cs.EDGE, extent=300)
    split.add_pane(_labels(QWidget(), 12, "Runtime"), "Runtime", stretch=2)
    qtbot.addWidget(split)
    split.resize(1200, 400)
    split.show()
    qtbot.waitExposed(split)
    _pump()
    return split


class _Drag:
    """A press on a handle, moves from a fixed point, and the release."""

    def __init__(self, handle):
        self.handle = handle
        self.start = handle.rect().center()
        self.origin = handle.mapToGlobal(self.start)
        self._send(QEvent.MouseButtonPress, 0, Qt.LeftButton, Qt.LeftButton)

    def _send(self, kind, dx, button, buttons):
        glob = self.origin + QPoint(dx, 0)
        local = self.handle.mapFromGlobal(glob)
        QApplication.sendEvent(self.handle, QMouseEvent(
            kind, QPointF(local), QPointF(glob), button, buttons,
            Qt.NoModifier))

    def move(self, dx):
        self._send(QEvent.MouseMove, dx, Qt.NoButton, Qt.LeftButton)

    def release(self, dx):
        self._send(QEvent.MouseButtonRelease, dx, Qt.LeftButton, Qt.NoButton)


def _drag(split, steps=30, step=8):
    drag = _Drag(split.handle(1))
    for i in range(steps):
        drag.move(step * (i + 1))
        _pump(1)
    return drag, step * steps


# --------------------------------------------------------------------------
# the drag
# --------------------------------------------------------------------------

def test_a_drag_writes_its_sizes_once_when_the_handle_is_let_go(
        qtbot, store, writes):
    split = _splitter(qtbot, "smooth::drag")
    before = split.sizes()[0]
    drag, travelled = _drag(split)
    assert writes == [], "a move of a drag wrote the preferences file"
    assert split.pane("Settings").extent == split.sizes()[0]
    assert split.sizes()[0] > before
    drag.release(travelled)
    assert writes == [("smooth::drag", split.extents())]
    assert cs.get_pane_extents("smooth::drag")["Settings"] == split.sizes()[0]
    qtbot.wait(cs._EXTENTS_SAVE_MS + 100)
    assert len(writes) == 1, "the release left a second write waiting"


def test_a_held_handle_that_rests_is_written_once(qtbot, store, writes):
    split = _splitter(qtbot, "smooth::rest")
    _drag(split, steps=5)
    assert writes == []
    qtbot.waitUntil(lambda: len(writes) == 1, timeout=3000)
    assert cs.get_pane_extents("smooth::rest")["Settings"] == split.sizes()[0]


def test_a_move_made_in_code_is_written_at_once(qtbot, store, writes):
    split = _splitter(qtbot, "smooth::code")
    split.moveSplitter(350, 1)
    assert writes == [("smooth::code", split.extents())]


def test_hiding_the_splitter_writes_a_size_that_is_still_waiting(
        qtbot, store, writes):
    split = _splitter(qtbot, "smooth::hide")
    _drag(split, steps=5)
    assert writes == []
    split.hide()
    assert writes == [("smooth::hide", split.extents())]
    assert cs.get_pane_extents("smooth::hide") == split.extents()
    assert not split._extents_timer.isActive()
    split.show()
    _pump()
    split.hide()
    assert len(writes) == 1, "a hide with nothing waiting wrote again"


def test_a_drag_ends_where_it_did_when_every_move_wrote(
        qtbot, store, writes, monkeypatch):
    """The same drag, with the old write on every move, ends identically."""
    split = _splitter(qtbot, "smooth::same")
    start = split.sizes()
    drag, travelled = _drag(split)
    drag.release(travelled)
    _pump()
    new = (split.sizes(), split.extents(), split.grab().toImage())

    split.setSizes(start)
    _pump()
    monkeypatch.setattr(split._extents_timer, "start", split._save_extents)
    drag, travelled = _drag(split)
    drag.release(travelled)
    _pump()
    old = (split.sizes(), split.extents(), split.grab().toImage())
    assert new[0] == old[0] and new[1] == old[1]
    assert new[2] == old[2], "the drag's end state is not pixel-identical"


def test_a_drag_move_costs_a_fraction_of_the_write_it_used_to_make(
        qtbot, store):
    split = _splitter(qtbot, "smooth::time")
    drag = _Drag(split.handle(1))
    moves = []
    for i in range(40):
        started = time.perf_counter()
        drag.move(4 * (i % 20 + 1))
        moves.append(time.perf_counter() - started)
    drag.release(4)
    _pump()
    old = []
    for _ in range(15):
        started = time.perf_counter()
        cs.set_pane_extents("smooth::time", split.extents())
        old.append(time.perf_counter() - started)
    move, write = statistics.median(moves), statistics.median(old)
    assert move < 0.5 * write, (
        f"a drag move took {move * 1000:.2f} ms against "
        f"{write * 1000:.2f} ms for the write each move used to make")


# --------------------------------------------------------------------------
# Ctrl + wheel over a column
# --------------------------------------------------------------------------

class _Column(QWidget):
    """A column of labels that counts the sheets it is given."""

    def __init__(self, rows=60):
        super().__init__()
        self.sheets = 0
        layout = QVBoxLayout(self)
        self.labels = [QLabel(f"Console line {i} ▾ ↓ …", self)
                       for i in range(rows)]
        for label in self.labels:
            layout.addWidget(label)
        self.button = QPushButton("↓ jump to the end", self)
        layout.addWidget(self.button)

    def setStyleSheet(self, sheet):                     # noqa: N802
        self.sheets += 1
        super().setStyleSheet(sheet)

    def state(self):
        return (self.styleSheet(),
                [w.font().pixelSize() for w in self.labels + [self.button]])


@pytest.fixture
def column(qtbot, qt_theme_applied, store):
    from spacr.qt.live_zoom import ColumnTextScale

    store.set_runtime_text_scale(1.0)
    root = _Column()
    qtbot.addWidget(root)
    root.resize(700, 1400)
    root.show()
    qtbot.waitExposed(root)
    scale = ColumnTextScale()
    scale.register(root)
    _pump()
    root.sheets = 0
    yield scale, root
    scale.deleteLater()


def _notch(scale, widget, notches=1):
    event = QWheelEvent(QPointF(5, 5), QPointF(widget.mapToGlobal(QPoint(5, 5))),
                        QPoint(0, 0), QPoint(0, int(120 * notches)),
                        Qt.NoButton, Qt.ControlModifier, Qt.NoScrollPhase,
                        False)
    return scale.eventFilter(widget, event)


def test_ten_notches_restyle_the_column_once_when_the_wheel_rests(
        qtbot, column):
    scale, root = column
    for _ in range(10):
        assert _notch(scale, root.labels[5])
        _pump(2)
    assert scale.scale() == pytest.approx(2.0)
    assert root.sheets == 0, "a notch restyled the column"
    qtbot.waitUntil(lambda: root.sheets == 1, timeout=3000)
    qtbot.wait(300)
    assert root.sheets == 1


def test_the_size_a_turn_of_the_wheel_leaves_is_saved(qtbot, column, store):
    from spacr.qt.live_zoom import ColumnTextScale

    scale, root = column
    for _ in range(4):
        _notch(scale, root.button)
    qtbot.waitUntil(
        lambda: store.get_runtime_text_scale() == pytest.approx(1.4),
        timeout=3000)
    assert ColumnTextScale().scale() == pytest.approx(1.4)


def test_the_wheel_ends_where_a_restyle_a_notch_did(qtbot, column):
    """Same sheet, same font sizes, same pixels as the old path."""
    scale, root = column
    for _ in range(6):
        _notch(scale, root.labels[3])
    qtbot.waitUntil(lambda: root.sheets == 1, timeout=3000)
    _pump()
    new = root.state(), root.grab().toImage()

    scale.reset()
    _pump()
    for step in range(1, 7):
        scale.set_scale(1.0 + 0.1 * step)
        scale.restyle_all()
        _pump(2)
    _pump()
    old = root.state(), root.grab().toImage()
    assert new[0] == old[0]
    assert new[1] == old[1], "the zoom's end state is not pixel-identical"


def test_a_notch_costs_a_fraction_of_the_restyle_it_used_to_make(
        qtbot, column):
    scale, root = column
    restyles = []
    for step in range(4):
        scale.set_scale(1.05 + 0.1 * step)
        started = time.perf_counter()
        scale.restyle_all()
        restyles.append(time.perf_counter() - started)
    scale.reset()
    _pump()
    notches = []
    for i in range(10):
        started = time.perf_counter()
        _notch(scale, root.labels[i], 1 if i < 5 else -1)
        _pump(1)
        notches.append(time.perf_counter() - started)
    notch, restyle = statistics.median(notches), statistics.median(restyles)
    assert notch < 0.25 * restyle, (
        f"a Ctrl + wheel notch took {notch * 1000:.2f} ms against "
        f"{restyle * 1000:.2f} ms for the restyle each notch used to make")


# --------------------------------------------------------------------------
# set_scale: the symbols Open Sans lacks, the fonts opened ahead, the hub
# --------------------------------------------------------------------------
#
# After the wheel rested, the one restyle left still held the GUI thread
# 105-150 ms on Mask Generation: Qt searched the system's fonts for "▾" and
# "↓" at every new size (Open Sans has neither), opened every font the new
# size needed, and asked the application event hub about ~1,000 events no
# watcher acts on. Now the symbols come from fonts spaCR ships, the fonts
# are opened a few per turn before the sheet goes on, and the hub steps off
# while it does. Measured offscreen at 2560x1440 with the field animating,
# the longest single step of set_scale(1.25) on the Mask column went from
# 105 ms to 25 ms.

import contextlib                                                   # noqa: E402
import importlib.util                                               # noqa: E402
import json                                                         # noqa: E402
import os                                                           # noqa: E402
import subprocess                                                   # noqa: E402
import sys                                                          # noqa: E402
from pathlib import Path                                            # noqa: E402

from PySide6.QtCore import QEventLoop, QObject                      # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase, QTextLayout         # noqa: E402

from spacr.qt.theme import _qss_font_families                       # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_QSS_FAMILIES = list(_qss_font_families())


def _glyph_tool():
    pytest.importorskip("fontTools")
    spec = importlib.util.spec_from_file_location(
        "spacr_build_glyph_fonts", _REPO / "tools" / "build_glyph_fonts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def symbols(qapp):
    from spacr.qt import app as app_mod

    app_mod._load_bundled_fonts()
    return app_mod


def _families_drawing(text, weight=400, px=13):
    font = QFont()
    font.setFamilies(_QSS_FAMILIES)
    font.setPixelSize(px)
    font.setWeight(QFont.Weight(weight))
    layout = QTextLayout(text, font)
    layout.beginLayout()
    layout.createLine()
    layout.endLayout()
    return {run.rawFont().familyName() for run in layout.glyphRuns()}


def test_every_symbol_the_interface_draws_is_in_a_font_spacr_ships():
    """A string that draws a glyph no shipped font has fails here.

    Fix it by running ``python tools/build_glyph_fonts.py``.
    """
    tool = _glyph_tool()
    missing = tool.ui_glyphs(have=tool.shipped_cmap())
    assert missing == {}, "drawn but in no shipped font: " + ", ".join(
        f"{c} U+{ord(c):04X} ({where[0]})" for c, where in missing.items())
    assert tool.main(["--check"]) == 0


def test_the_scan_reads_strings_not_docstrings_catalogs_or_locale_text(
        tmp_path):
    tool = _glyph_tool()
    package = tmp_path / "qt"
    (package / "i18n_catalogs").mkdir(parents=True)
    (package / "screen.py").write_text(
        '"""Module ▾ docstring."""\n'
        'def f():\n'
        '    """Function ✓ docstring."""\n'
        '    "statement ■ string"\n'
        '    return [tr("Next →"), "x\\u29c9", "x\\u1d62", "한국어 ←",\n'
        '            "\\u00e9", "tab\\t", 3]\n', encoding="utf-8")
    (package / "i18n_extra.py").write_text('X = "✗"\n', encoding="utf-8")
    (package / "i18n_catalogs" / "ko.py").write_text('X = "✕"\n',
                                                      encoding="utf-8")
    old = tool.REPO
    tool.REPO = tmp_path
    try:
        found = tool.ui_glyphs(root=package, have=set(range(0x100)))
    finally:
        tool.REPO = old
    assert sorted(found) == sorted("→⧉ᵢ")
    assert found["→"] == ["qt/screen.py:5"]


def test_each_symbol_goes_to_the_font_that_drew_it_and_a_gap_is_refused(
        monkeypatch):
    tool = _glyph_tool()
    have = {"NotoSansMath-Regular.ttf": {0x2192, 0x25BE},
            "NotoSans-Regular.ttf": {0x1D62, 0x2192},
            "DejaVuSans.ttf": {0x25BE, 0x2192, 0x2713},
            "NotoSansSymbols-Regular.ttf": {0x2317},
            "NotoSansCJK-Regular.ttc": {0xFF0B}}
    monkeypatch.setattr(tool, "source", lambda name: name)
    monkeypatch.setattr(tool, "_cmap", lambda path: have[path])
    plan = tool.assign("→▾ᵢ✓⌗＋")
    assert plan == {"spaCR Glyphs Math": "→", "spaCR Glyphs Latin": "ᵢ",
                    "spaCR Glyphs Sans": "▾✓",
                    "spaCR Glyphs Symbols": "⌗",
                    "spaCR Glyphs Noto Sans CJK JP": "＋"}
    with pytest.raises(SystemExit, match="U\\+2603"):
        tool.assign("☃")


def test_the_check_names_a_glyph_no_shipped_font_has(monkeypatch, capsys):
    tool = _glyph_tool()
    monkeypatch.setattr(tool, "ui_glyphs",
                        lambda have=None: {"☃": ["spacr/qt/x.py:1"]})
    assert tool.main(["--check"]) == 1
    assert "☃ U+2603 spacr/qt/x.py:1" in capsys.readouterr().out


def test_the_symbol_fonts_stand_in_for_every_family_the_sheets_ask_for(
        symbols):
    tool = _glyph_tool()
    assert [family for family, _faces, _takes in tool.FAMILIES] == list(
        symbols.SYMBOL_FONT_FAMILIES)
    assert set(symbols.SYMBOL_FONT_FAMILIES) <= set(QFontDatabase.families())
    symbols._load_bundled_fonts()
    for family in symbols._SYMBOLS_STAND_IN_FOR:
        assert QFont.substitutes(family) == [
            name.lower() for name in symbols.SYMBOL_FONT_FAMILIES], family


def test_qt_finds_every_symbol_in_a_font_spacr_ships(symbols):
    """No glyph the interface draws sends Qt to the system's fonts."""
    tool = _glyph_tool()
    shipped = {"Open Sans", *symbols.SYMBOL_FONT_FAMILIES}
    for weight in (300, 400, 600, 700):
        for char in sorted(tool.ui_glyphs()) + list("▾↓"):
            drawn = _families_drawing(f"a{char}b", weight)
            assert drawn <= shipped, (char, weight, drawn)
    assert _families_drawing("▾", 600) == {"spaCR Glyphs Sans"}
    assert _families_drawing("↓") == {"spaCR Glyphs Math"}


#: The family each platform installs that the style sheets name.
_INSTALLED_SYSTEM_FAMILY = {"win32": "Segoe UI", "darwin": "Helvetica Neue"}


def _shipped_cmaps(tool):
    """``{family: code points}`` of Open Sans and each glyph family."""
    cmaps = {"Open Sans": tool.open_sans_cmap()}
    for family, faces, _takes in tool.FAMILIES:
        cmaps[family] = set().union(*(tool._cmap(tool.OUT / out)
                                      for _style, _src, out in faces))
    return cmaps


def _first_family_drawing(char, stack, installed, substitutes):
    """The family Qt draws ``char`` from: the first of ``stack`` that is
    installed and has it, a family that is NOT installed standing for its
    substitutes."""
    for family in stack:
        for name in ([family] if family in installed
                     else substitutes.get(family, ())):
            if ord(char) in installed.get(name, ()):
                return name
    return None


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_a_symbol_open_sans_lacks_comes_from_a_shipped_font_everywhere(
        symbols, platform):
    """2026-10-24, Windows: "ᵢ" at weight 300 was drawn by Segoe UI.

    Qt takes a family's substitutes only when the family is not installed,
    and Segoe UI is installed on every Windows (Helvetica Neue on every
    Mac), so the glyph fonts named as its substitutes were never asked.
    Modelled with the shipped fonts' real character maps and the platform's
    system family installed, drawing every glyph the interface draws except
    the fullwidth forms neither Segoe UI nor Helvetica Neue has.
    """
    tool = _glyph_tool()
    chars = sorted(set(tool.ui_glyphs()) | set("▾↓"))
    cmaps = _shipped_cmaps(tool)
    *glyphs, cjk = symbols.SYMBOL_FONT_FAMILIES
    installed = dict(cmaps)
    system = _INSTALLED_SYSTEM_FAMILY.get(platform)
    if system:
        installed[system] = {ord(c) for c in chars} - cmaps[cjk]
    substitutes = {family: symbols.SYMBOL_FONT_FAMILIES
                   for family in symbols._SYMBOLS_STAND_IN_FOR}
    stack = _qss_font_families(platform)
    before_cjk = set().union(cmaps["Open Sans"],
                             *(cmaps[family] for family in glyphs))
    for char in chars:
        drawn = _first_family_drawing(char, stack, installed, substitutes)
        assert drawn in cmaps, (char, f"U+{ord(char):04X}", platform, drawn)
        if ord(char) not in before_cjk:
            assert drawn == cjk, (char, platform, drawn)


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_the_cjk_subset_follows_the_system_family_and_the_rest_precede_it(
        symbols, platform):
    """Chinese and Korean text keeps the system family's font linking
    (Microsoft YaHei, Malgun Gothic; PingFang): the CJK subset is asked only
    after the system family, every other glyph family before it."""
    stack = list(_qss_font_families(platform))
    system = stack.index(_INSTALLED_SYSTEM_FAMILY[platform])
    *glyphs, cjk = symbols.SYMBOL_FONT_FAMILIES
    assert stack[:system] == ["Open Sans", *glyphs]
    assert stack[system + 1] == cjk


def test_linux_asks_for_the_families_it_always_did():
    """Linux installs none of the named system families, so there the
    substitutes take every symbol and the list is unchanged."""
    assert _qss_font_families("linux") == (
        "Open Sans", "Helvetica Neue", "sans-serif")


_WITH_AN_INSTALLED_SYSTEM_FAMILY = r'''
import json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont, QFontDatabase, QTextLayout
app = QApplication([])
from spacr.qt.app import _load_bundled_fonts
from spacr.qt.theme import _qss_font_families
_load_bundled_fonts()
assert sys.argv[2] in QFontDatabase.applicationFontFamilies(
    QFontDatabase.addApplicationFont(sys.argv[3]))
out = {}
for char in sys.argv[4]:
    for weight in (300, 400, 600, 700):
        font = QFont()
        font.setFamilies(list(_qss_font_families(sys.argv[1])))
        font.setPixelSize(13)
        font.setWeight(QFont.Weight(weight))
        layout = QTextLayout("a" + char + "b", font)
        layout.beginLayout(); layout.createLine(); layout.endLayout()
        out[f"{char}|{weight}"] = sorted(
            {r.rawFont().familyName() for r in layout.glyphRuns()})
print(json.dumps(out))
'''


def _stand_in_system_font(tmp_path, family):
    """DejaVu Sans renamed ``family``: a system face with most symbols."""
    from fontTools.ttLib import TTFont

    found = [path for path in (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/usr/share/fonts/dejavu/DejaVuSans.ttf"))
        if path.is_file()]
    if not found:
        pytest.skip("DejaVu Sans is not installed here")
    font = TTFont(str(found[0]))
    for record in font["name"].names:
        if record.nameID in (1, 4, 16):
            record.string = family
        elif record.nameID == 6:
            record.string = family.replace(" ", "")
    out = tmp_path / "stand-in.ttf"
    font.save(str(out))
    return out


@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_qt_skips_an_installed_system_family_for_the_shipped_symbols(
        symbols, tmp_path, platform):
    """The Windows failure reproduced anywhere: with a face named Segoe UI
    (or Helvetica Neue) installed, Qt still draws every symbol from Open
    Sans or a glyph font."""
    tool = _glyph_tool()
    system = _INSTALLED_SYSTEM_FAMILY[platform]
    face = _stand_in_system_font(tmp_path, system)
    chars = "".join(sorted(set(tool.ui_glyphs()) | set("▾↓")))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONPATH=str(_REPO) + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run(
        [sys.executable, "-c", _WITH_AN_INSTALLED_SYSTEM_FAMILY, platform,
         system, str(face), chars],
        capture_output=True, text=True, env=env, timeout=300, check=True)
    drawn = json.loads(done.stdout.strip().splitlines()[-1])
    shipped = {"Open Sans", *symbols.SYMBOL_FONT_FAMILIES}
    wrong = {key: families for key, families in drawn.items()
             if not set(families) <= shipped}
    assert wrong == {}


def test_without_the_symbol_fonts_open_sans_is_registered_alone(
        monkeypatch):
    import PySide6.QtGui as qtgui

    from spacr.qt import app as app_mod

    added, substituted = [], []

    class _SpyDatabase:
        @staticmethod
        def addApplicationFont(path):                    # noqa: N802
            added.append(path)
            return len(added)

    class _SpyFont:
        @staticmethod
        def substitutes(family):
            return []

        @staticmethod
        def insertSubstitutions(family, names):          # noqa: N802
            substituted.append(family)

    monkeypatch.setattr(qtgui, "QFontDatabase", _SpyDatabase)
    monkeypatch.setattr(qtgui, "QFont", _SpyFont)
    real_isdir = os.path.isdir
    hidden = os.path.join("font", "glyphs")
    monkeypatch.setattr(
        os.path, "isdir",
        lambda p: False if str(p).endswith(hidden) else real_isdir(p))
    app_mod._load_bundled_fonts()
    assert added and all("open_sans" in path for path in added)
    assert substituted == []


_RENDER = r'''
import hashlib, json, os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QImage, QPainter,
                           QTextLayout)
app = QApplication([])
if sys.argv[1] == "shipped":
    from spacr.qt.app import _load_bundled_fonts
    _load_bundled_fonts()
else:
    import spacr.qt.app as m
    folder = os.path.join(os.path.dirname(os.path.dirname(m.__file__)),
                          "resources", "font", "open_sans", "static")
    for name in os.listdir(folder):
        QFontDatabase.addApplicationFont(os.path.join(folder, name))
from spacr.qt.theme import _qss_font_families
out = {}
for char in sys.argv[2]:
    for weight in (300, 400, 600, 700):
        for px in (11, 13, 16, 22):
            font = QFont()
            font.setFamilies(list(_qss_font_families()))
            font.setPixelSize(px)
            font.setWeight(QFont.Weight(weight))
            layout = QTextLayout(char, font)
            layout.beginLayout(); layout.createLine(); layout.endLayout()
            family = [r.rawFont().familyName() for r in layout.glyphRuns()]
            image = QImage(60, 50, QImage.Format_ARGB32)
            image.fill(QColor("white"))
            painter = QPainter(image)
            painter.setFont(font)
            painter.setPen(QColor("black"))
            painter.drawText(5, 35, "a" + char + "b")
            painter.end()
            out[f"{char}|{weight}|{px}"] = [
                family, hashlib.md5(bytes(image.constBits())).hexdigest()]
print(json.dumps(out))
'''

#: The system family each symbol font was cut from.
_CUT_FROM = {"spaCR Glyphs Math": "Noto Sans Math",
             "spaCR Glyphs Latin": "Noto Sans",
             "spaCR Glyphs Sans": "DejaVu Sans",
             "spaCR Glyphs Symbols": "Noto Sans Symbols",
             "spaCR Glyphs Noto Sans CJK JP": "Noto Sans CJK JP"}


def _render(kind, chars):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONPATH=str(_REPO) + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run([sys.executable, "-c", _RENDER, kind, chars],
                          capture_output=True, text=True, env=env,
                          timeout=300, check=True)
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_the_symbol_fonts_draw_what_the_system_fonts_drew():
    """Pixel for pixel, wherever the system drew a symbol from the font it
    was cut from (DejaVu Sans, Noto Sans, Noto Sans Math, ... installed)."""
    tool = _glyph_tool()
    chars = "".join(sorted(tool.ui_glyphs()))
    before, after = _render("system", chars), _render("shipped", chars)
    compared = 0
    for key, (families, picture) in after.items():
        assert len(families) == 1, (key, families)
        if before[key][0] != [_CUT_FROM[families[0]]]:
            continue
        compared += 1
        assert before[key][1] == picture, f"{key} draws differently"
    if not compared:
        pytest.skip("none of the source fonts is installed here")


def _longest_turn(seconds):
    """The longest turn of the event loop in the next ``seconds``, in s."""
    app = QApplication.instance()
    longest = 0.0
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        started = time.perf_counter()
        app.processEvents(QEventLoop.AllEvents)
        longest = max(longest, time.perf_counter() - started)
        time.sleep(0.001)
    return longest


@pytest.fixture
def hub(qapp):
    """A watcher on the application event hub, so the hub exists.

    It records each object by its C++ pointer, never by the wrapper it was
    handed. Some of these events arrive while a widget is still being built,
    as a base-class wrapper; holding that wrapper keeps it in shiboken's
    table, and the widget's own wrapper is never returned again --
    ``findChildren`` hands back the plain QWidget instead (seen on an Apple
    M2, where every replayed label then went unheard by identity).
    """
    from shiboken6 import getCppPointer

    from spacr.qt.gil_priority import (_stop_watching_application_events,
                                       _watch_application_events)

    heard = []

    class _Watcher(QObject):
        def eventFilter(self, watched, event):           # noqa: N802
            heard.append((getCppPointer(watched)[0], event.type()))
            return False

    watcher = _Watcher()
    _watch_application_events(qapp, watcher, (
        QEvent.StyleChange, QEvent.PaletteChange, QEvent.Polish,
        QEvent.Show, QEvent.Hide, QEvent.Move, QEvent.Resize,
        QEvent.EnabledChange, QEvent.ParentChange))
    yield heard
    _stop_watching_application_events(qapp, watcher)


_ZOOM = r'''
import contextlib, os, sys, time
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from PySide6.QtCore import QEvent, QObject, QSettings
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope,
                  sys.argv[2])
QSettings.setPath(QSettings.Format.NativeFormat, QSettings.Scope.UserScope,
                  sys.argv[2])
app = QApplication([])
from spacr.qt import gil_priority, live_zoom
from spacr.qt.app import _load_bundled_fonts, _use_open_sans
from spacr.qt.theme import apply_qpalette, stylesheet
kind = sys.argv[1]
old = kind in ("old", "plain")
if old:
    import spacr.qt.app as m
    from PySide6.QtGui import QFontDatabase
    folder = os.path.join(os.path.dirname(os.path.dirname(m.__file__)),
                          "resources", "font", "open_sans", "static")
    for name in os.listdir(folder):
        QFontDatabase.addApplicationFont(os.path.join(folder, name))
    gil_priority._restyle_unheard = lambda a, r: contextlib.nullcontext()
else:
    _load_bundled_fonts()
_use_open_sans(app, "regular")
apply_qpalette(app)
app.setStyleSheet(stylesheet())


class Watcher(QObject):
    def eventFilter(self, watched, event):
        return False


watcher = Watcher()
gil_priority._watch_application_events(
    app, watcher, (QEvent.StyleChange, QEvent.PaletteChange))
roles = ("CardTitle", "CardSubtitle", "TitleHeading", "Subtitle",
         "SubtitleSmall", "Caption", "SectionHeading", "FoldHeading",
         "TourTitle", "HTileName", "DisplayHeading", "Muted")
root = QWidget()
column = QVBoxLayout(root)
for i in range(96):
    text = (f"Step {i} v d r c" if kind == "plain"
            else f"Step {i} \u25be \u2193 \u2192 \u2713")
    label = QLabel(text, root)
    label.setObjectName(roles[i % len(roles)])
    column.addWidget(label)
root.resize(700, 2400)
root.show()
for _ in range(20):
    app.processEvents()
scale = live_zoom.ColumnTextScale()
scale.register(root)
steps = []
real = scale.restyle_all


def timed():
    started = time.perf_counter()
    try:
        return real()
    finally:
        steps.append(time.perf_counter() - started)


scale.restyle_all = timed
slow = scale._warm_some


def warm():
    started = time.perf_counter()
    try:
        return slow()
    finally:
        steps.append(time.perf_counter() - started)


scale._warm_timer.timeout.disconnect()
scale._warm_timer.timeout.connect(warm)
if old:
    scale._scale = 1.37
    timed()
else:
    scale.set_scale(1.37, remember=False)
end = time.perf_counter() + 1.5
while time.perf_counter() < end:
    started = time.perf_counter()
    app.processEvents()
    steps.append(time.perf_counter() - started)
    time.sleep(0.001)
print("LONGEST", max(steps), bool(root.styleSheet()))
'''


def _zoom_step(kind, tmp_path):
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen",
               PYTHONPATH=str(_REPO) + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    done = subprocess.run(
        [sys.executable, "-c", _ZOOM, kind, str(tmp_path / kind)],
        capture_output=True, text=True, env=env, timeout=300, check=True)
    line = [row for row in done.stdout.splitlines()
            if row.startswith("LONGEST")][-1]
    _label, seconds, sheeted = line.split()
    assert sheeted == "True"
    return float(seconds)


#: The old restyle pays the symbol search where it costs at least this many
#: times the old restyle of the same column with plain letters. Medians
#: measured 2026-10-13: 287 ms against 16.1 ms with a desktop's 2464 system
#: fonts, 16.0 ms against 14.1 ms with only the 22 DejaVu fonts a CI runner
#: has.
SEARCH_PAID = 2.0

#: Most set_scale's longest step may cost, in old restyles, where the
#: search is free: 0.91-1.13 per interleaved pair over twelve pairs with
#: only the 22 DejaVu fonts (medians 1.02 and 0.96, 2026-10-13), 0.89 and
#: 0.94 on CI.
NO_WORSE = 1.25


def test_set_scale_holds_the_gui_thread_far_less_than_the_old_restyle(
        tmp_path):
    """Self-calibrated, each side in a fresh process so neither finds the
    other's fonts open: the restyle the column used to make at a new size
    (the system searched for the symbols at every new size, every font
    opened inside it or the layout after it, the hub asked about every
    event) against the longest step set_scale makes now. Every turn of the
    event loop until the column has settled counts, the layout and paint
    after the sheet included.

    The search is what set_scale takes out, and it costs what the system's
    font list makes it cost: about 270 ms with a desktop's 2464 fonts and
    about 2 ms with the 22 DejaVu fonts of a CI runner, where both sides
    are the layout and paint of the column and the 0.7 ratio measured noise
    (18.7 against 19.9 ms and 15.1 against 17.0 ms on CI; 16.2 against
    15.8 ms here with only those fonts). A third side, the old restyle of
    the same column with plain letters for the symbols, says which machine
    this is: where the search is paid, set_scale must take most of it out;
    where it is free, set_scale must be no slower than the old restyle."""
    olds, news, plains = [], [], []
    for _ in range(3):
        olds.append(_zoom_step("old", tmp_path))
        news.append(_zoom_step("new", tmp_path))
        plains.append(_zoom_step("plain", tmp_path))
    old, new = statistics.median(olds), statistics.median(news)
    plain = statistics.median(plains)
    print(f"old restyle {old * 1000:.1f} ms, set_scale's longest step "
          f"{new * 1000:.1f} ms, old restyle without the symbols "
          f"{plain * 1000:.1f} ms")
    if old >= SEARCH_PAID * plain:
        assert new < 0.7 * old, (
            f"set_scale held the GUI thread {new * 1000:.1f} ms at most "
            f"against {old * 1000:.1f} ms for the restyle it replaced")
    else:
        assert new < NO_WORSE * old, (
            f"set_scale held the GUI thread {new * 1000:.1f} ms at most "
            f"against {old * 1000:.1f} ms for the restyle it replaced, on a "
            f"machine whose font list makes the symbol search free "
            f"({plain * 1000:.1f} ms without the symbols)")


def test_set_scale_ends_where_the_old_restyle_did(qtbot, symbols, hub,
                                                  column, monkeypatch):
    """Same sheet, fonts and pixels as one synchronous restyle with the hub
    hearing everything, and the hub's watchers hear the same changes."""
    from spacr.qt import gil_priority

    scale, root = column
    scale.set_scale(1.3, remember=False)
    qtbot.waitUntil(lambda: root.sheets == 1, timeout=3000)
    _pump()
    new = root.state(), root.grab().toImage()
    told = {(watched, kind) for watched, kind in hub}

    scale.reset()
    _pump()
    hub.clear()
    with monkeypatch.context() as patch:
        patch.setattr(gil_priority, "_restyle_unheard",
                      lambda app, widget: contextlib.nullcontext())
        scale._scale = 1.3
        scale.restyle_all()
    _pump()
    old = root.state(), root.grab().toImage()
    heard = {(watched, kind) for watched, kind in hub}
    assert new[0] == old[0]
    assert new[1] == old[1], "set_scale's end state is not pixel-identical"
    from shiboken6 import getCppPointer

    family = {getCppPointer(root)[0]} | {getCppPointer(child)[0]
                                         for child in root.findChildren(QWidget)}
    restyles = {(w, k) for w, k in heard
                if k in (QEvent.StyleChange, QEvent.PaletteChange)
                and w in family}
    assert restyles and restyles <= told


def test_the_fonts_are_opened_a_few_a_turn_before_the_sheet(
        qtbot, symbols, column, monkeypatch):
    from spacr.qt import live_zoom

    scale, root = column
    monkeypatch.setattr(live_zoom, "_WARM_BUDGET_MS", 0.0)
    turns = []
    real = scale.restyle_all

    def restyle_all():
        turns.append(("sheet", len(scale._warming)))
        return real()

    monkeypatch.setattr(scale, "restyle_all", restyle_all)
    fonts = scale._fonts_to_open(root)
    assert fonts == [], "nothing to open at the size the column has"
    scale._scale = 1.7
    fonts = scale._fonts_to_open(root)
    assert len(fonts) >= 2
    assert all(font.pixelSize() > 13 and "▾" in sample and "↓" in sample
               for font, sample in fonts)
    scale._prepare()
    assert len(scale._warming) == len(fonts) - 1
    qtbot.waitUntil(lambda: bool(turns), timeout=3000)
    assert turns == [("sheet", 0)]
    assert root.sheets == 1
    scale.set_scale(1.2, remember=False)
    scale.restyle_all()
    assert scale._warming == [] and not scale._warm_timer.isActive()


def test_a_column_that_is_gone_has_no_fonts_to_open(qapp):
    from shiboken6 import delete

    from spacr.qt.live_zoom import ColumnTextScale

    gone = QWidget()
    delete(gone)
    assert ColumnTextScale()._fonts_to_open(gone) == []


# --------------------------------------------------------------------------
# the hub stepping off while a sheet goes on
# --------------------------------------------------------------------------

def test_the_hub_hears_after_the_sheet_what_it_did_not_hear_during_it(
        qtbot, hub):
    from spacr.qt.gil_priority import _restyle_unheard

    root = QWidget()
    qtbot.addWidget(root)
    layout = QVBoxLayout(root)
    shown, hidden, moved, disabled = (QLabel(t, root) for t in "abcd")
    for label in (shown, hidden, moved, disabled):
        layout.addWidget(label)
    shown.hide()
    stranger = QLabel("e")
    qtbot.addWidget(stranger)
    root.show()
    qtbot.waitExposed(root)
    _pump()
    hub.clear()
    with _restyle_unheard(QApplication.instance(), root):
        root.setStyleSheet("QLabel { font-size: 21px; }")
        shown.show()
        hidden.hide()
        moved.setGeometry(moved.geometry().translated(3, 4))
        moved.resize(moved.width() + 5, moved.height())
        disabled.setEnabled(False)
        stranger.setParent(root)
        born = QLabel("f", root)
        born.ensurePolished()
        during = list(hub)
    assert during == []
    from shiboken6 import getCppPointer

    by_pointer = {}
    for watched, kind in hub:
        by_pointer.setdefault(watched, set()).add(kind)
    heard = {w: by_pointer.get(getCppPointer(w)[0], set())
             for w in (root, shown, hidden, moved, disabled, stranger, born)}
    restyle = {QEvent.StyleChange, QEvent.PaletteChange}
    assert all(restyle <= heard[w] for w in (root, shown, hidden, moved,
                                             disabled, stranger, born))
    assert QEvent.Show in heard[shown] and QEvent.Hide in heard[hidden]
    assert {QEvent.Move, QEvent.Resize} <= heard[moved]
    assert QEvent.EnabledChange in heard[disabled]
    assert QEvent.ParentChange in heard[stranger]
    assert QEvent.Polish in heard[born]
    assert QEvent.Show not in heard[root]


def test_without_a_hub_or_a_root_the_sheet_simply_goes_on(qapp,
                                                          monkeypatch):
    from spacr.qt import gil_priority

    ran = []
    monkeypatch.setattr(gil_priority, "_application_event_hub",
                        lambda app, create=True: None)
    with gil_priority._restyle_unheard(qapp, QWidget()):
        ran.append(1)
    monkeypatch.undo()
    with gil_priority._restyle_unheard(qapp, None):
        ran.append(2)
    assert ran == [1, 2]


def test_the_hub_is_told_nothing_about_a_root_that_is_gone_or_unwatched(
        qapp):
    from shiboken6 import delete

    from spacr.qt.gil_priority import (_application_event_hub_class,
                                       _tell_the_watchers)

    asked = []

    class _Hub:
        _by_kind = {}

        def eventFilter(self, watched, event):           # noqa: N802
            asked.append(event.type())
            return False

    root = QWidget()
    _tell_the_watchers(_Hub(), root, {})
    assert asked == []
    _Hub._by_kind = {QEvent.StyleChange: ()}
    delete(root)
    _tell_the_watchers(_Hub(), root, {})
    assert asked == []
    alive = QWidget()
    _tell_the_watchers(_Hub(), alive, {})
    assert asked == [QEvent.StyleChange]
    assert _application_event_hub_class() is not None


def test_a_native_parent_can_paint_and_close_after_the_python_owner_is_released(
        qtbot, store):
    import gc

    from PySide6.QtCore import QCoreApplication
    from shiboken6 import isValid

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    split = cs.CollapsibleSplitter(Qt.Horizontal, host)
    split.add_pane(_labels(QWidget(), 3, "Settings"), "Settings",
                   mode=cs.EDGE, extent=240)
    split.add_pane(_labels(QWidget(), 3, "Runtime"), "Runtime", stretch=2)
    layout.addWidget(split)
    host.resize(800, 300)
    host.show()
    qtbot.waitExposed(host)
    handle = split.handle(1)
    assert handle.edge_pane() is split.pane("Settings")
    assert not handle.grab().isNull()
    del split
    gc.collect()
    host.repaint()
    _pump()
    host.close()
    host.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    _pump()
    assert not isValid(handle)
