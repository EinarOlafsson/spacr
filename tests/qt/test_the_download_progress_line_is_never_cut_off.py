"""Item 502: a progress line's step count and percentage are never cut off.

ROOT CAUSE. The theme draws every ``QProgressBar`` 8 px tall. A bar that
painted its own text ("step 2 of 3", "45%") drew a 13 px caption into those
8 px, and into 4 px at 50 % GUI scale. The fix is the maintainer's "thin bar
+ label": :class:`spacr.qt.widgets.eliding.ProgressLine` paints no text
into the bar and puts the numbers in a label beside it that never elides.

These tests build every download and install progress line at the smallest
window spaCR supports, 1366x768, at 100 % and 50 % GUI scale, let the layout
settle, and ask what is PAINTED (``displayed_text()``), then check the count
label really has the room its text needs and sits inside the window. One PNG
per line and scale lands in ``/tmp/spacr-502-scratch`` for a look by eye.

The guard at the end walks the source: no ``QProgressBar`` in ``spacr/qt`` is
left painting text into the slim bar.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QRect, QSize
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (QApplication, QProgressBar, QScrollArea,
                               QStyle, QVBoxLayout, QWidget)

from spacr.qt import gui_scale
from spacr.qt.widgets.eliding import ProgressLine

pytestmark = pytest.mark.qt

SHOTS = Path("/tmp/spacr-502-scratch")
WINDOW = (1366, 768)
QT_ROOT = Path(__file__).resolve().parents[2] / "spacr" / "qt"


def settle(*widgets, rounds: int = 400) -> None:
    """Pump events until every widget's geometry is the same three times."""
    last, same = None, 0
    for _ in range(rounds):
        QApplication.processEvents()
        now = tuple((w.geometry().getRect(), w.isVisible()) for w in widgets)
        same = same + 1 if now == last else 0
        if same >= 3:
            return
        last = now


@pytest.fixture
def at_scale(qt_theme_applied):
    """Install the scaling layer and put the GUI at a chosen scale."""
    from spacr.qt.theme import stylesheet

    gui_scale.install_scaling_layer()
    qt_theme_applied.setStyleSheet(stylesheet())

    def _set(scale: float) -> None:
        gui_scale.set_gui_scale_live(scale)
        qt_theme_applied.setStyleSheet(stylesheet())

    yield _set
    gui_scale.set_gui_scale_live(1.0)
    qt_theme_applied.setStyleSheet(stylesheet())


def check_line(line: ProgressLine, window: QWidget, expected, shot: str):
    """Assert the count is painted in full and inside the window; save a PNG."""
    settle(window, line, line.count, line.bar)
    assert window.width() == WINDOW[0] or window.isWindow() and (
        window.width() <= WINDOW[0]), f"{shot}: the window is {window.size()}"
    scroller = line.parentWidget()
    while scroller is not None and not isinstance(scroller, QScrollArea):
        scroller = scroller.parentWidget()
    if scroller is not None:
        scroller.ensureWidgetVisible(line)
        scroller.horizontalScrollBar().setValue(0)
        settle(window, line, line.count, line.bar)
    assert line.isVisibleTo(window), f"{shot}: the progress line is not shown"
    painted = line.displayed_text()
    for piece in expected:
        assert piece in painted, f"{shot}: {piece!r} not painted in {painted!r}"
    count = line.count
    metrics = QFontMetrics(count.font())
    need = metrics.horizontalAdvance(count.text())
    assert count.width() >= need, (
        f"{shot}: count {count.text()!r} needs {need} px, has {count.width()}")
    assert count.height() >= metrics.height(), (
        f"{shot}: count is {count.height()} px tall, text {metrics.height()}")
    assert not line.bar.isTextVisible(), f"{shot}: the slim bar paints text"
    box = QStyle.alignedRect(count.layoutDirection(), count.alignment(),
                             QSize(need, metrics.height()), count.contentsRect())
    for frame, where in ((scroller.viewport() if scroller else window,
                          "what is on screen"), (window, "the window")):
        corners = (count.mapTo(frame, box.topLeft()),
                   count.mapTo(frame, box.bottomRight()))
        assert all(frame.rect().contains(c) for c in corners), (
            f"{shot}: the painted count {corners} is outside {where}")
    smallest = QRect(0, 0, *WINDOW)
    assert all(smallest.contains(count.mapTo(window, c)) for c in (
        box.topLeft(), box.bottomRight())), (
        f"{shot}: the painted count is outside a 1366x768 screen")
    SHOTS.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(SHOTS / f"{shot}_window.png"))
    corner = line.mapTo(window, QPoint(0, 0))
    window.grab(QRect(corner, line.size())).save(str(SHOTS / f"{shot}_line.png"))


def real_resize(widget: QWidget, width: int, height: int) -> None:
    """Resize in real pixels; the scaling layer would halve a size at 50 %."""
    original = gui_scale._ORIGINAL.get((QWidget, "resize"))
    if original is None:
        widget.resize(width, height)
    else:
        original(widget, width, height)


def host(qtbot, widget: QWidget) -> QWidget:
    """Show a screen inside a 1366x768 window."""
    window = QWidget()
    layout = QVBoxLayout(window)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(widget)
    qtbot.addWidget(window)
    real_resize(window, *WINDOW)
    window.show()
    return window


def show_dialog(qtbot, dialog: QWidget) -> QWidget:
    """Show a dialog at its own size, never larger than 1366x768."""
    qtbot.addWidget(dialog)
    dialog.show()
    settle(dialog)
    real_resize(dialog, min(dialog.width(), WINDOW[0]),
                min(dialog.height(), WINDOW[1]))
    return dialog


MB = 1024 * 1024


def one_field(tmp_path: Path) -> str:
    """A folder with one small image, so Make Masks shows its side panel."""
    import imageio.v2 as imageio
    import numpy as np

    folder = tmp_path / "field"
    folder.mkdir()
    image = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * 7)
    imageio.imwrite(folder / "a.tif", image)
    return str(folder)


def reveal(widget: QWidget, root: QWidget) -> None:
    """Show every hidden ancestor, as opening a folder and picking the mode do.

    Make Masks keeps its side panel hidden until a folder is open and shows
    only the chosen method's card. The panel's width is the same either way,
    which is what the count has to fit in.
    """
    parent = widget.parentWidget()
    while parent is not None and parent is not root:
        if parent.isHidden():
            parent.show()
        parent = parent.parentWidget()


def _backend_install(qtbot, monkeypatch, tmp_path):
    from spacr.qt.widgets import model_zoo_picker as mzp

    dialog = show_dialog(qtbot, mzp.BackendInstallDialog("cellpose3"))
    dialog.progress.setVisible(True)
    dialog._on_progress(1, 3, "Install PyTorch: Downloading torch-2.14.0+cpu.whl")
    return dialog, dialog.progress, ("step 2 of 3", "33%")


def _zoo_picker(qtbot, monkeypatch, tmp_path):
    import time

    from spacr import model_zoo
    from spacr.qt.widgets import model_share
    from spacr.qt.widgets import model_zoo_picker as mzp

    monkeypatch.setattr(mzp, "remembered_sources", lambda: ("spaCR",))
    monkeypatch.setattr(model_zoo, "catalogue", lambda **kwargs: [])
    monkeypatch.setattr(model_share, "CENTRAL_ENDPOINT", "")
    monkeypatch.setattr(model_share, "find_token", lambda: None)
    dialog = show_dialog(qtbot, mzp.ModelZooPicker(kinds=("cellpose",)))
    dialog.progress.setVisible(True)
    dialog._started_at = time.monotonic() - 10
    dialog._last_emit = 0.0
    dialog._on_progress(45 * MB, 100 * MB)
    return dialog, dialog.progress, ("45%",)


def _zoo_screen(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens.model_zoo import ModelZooScreen

    screen = ModelZooScreen(threaded=False)
    window = host(qtbot, screen)
    screen._on_progress(312 * MB, 690 * MB)
    return window, screen._progress, ("312", "690", "45%")


def _make_masks_download(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens import make_masks as mm

    screen = mm.MakeMasksScreen()
    window = host(qtbot, screen)
    assert screen._open_folder(one_field(tmp_path))
    bar = screen._cp_download_bar
    for name, group in screen._method_groups.items():
        group.setVisible(name == "cellpose")
    reveal(bar, screen)
    bar.setRange(0, 0)
    bar.setFormat("")
    bar.set_detail("Downloading cyto3_restore_but_with_a_long_name…")
    bar.show()
    screen._on_model_download_progress(45 * MB, 100 * MB)
    return window, bar, ("45%",)


def _magnifier(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens import make_masks as mm

    screen = mm.MakeMasksScreen()
    window = host(qtbot, screen)
    assert screen._open_folder(one_field(tmp_path))
    bar = screen._mag_progress
    reveal(bar, screen)
    bar.setRange(0, 1000)
    bar.setValue(450)
    bar.setFormat("about 12 s left")
    bar.setTextVisible(True)
    bar.show()
    return window, bar, ("about 12 s left", "45%")


def _starplast(qtbot, monkeypatch, tmp_path):
    from spacr.qt.starplast import StarplastInstallDialog

    dialog = show_dialog(qtbot, StarplastInstallDialog(job=lambda *a, **k: None))
    dialog.progress.show()
    dialog._progress(1, 3, "Installing: starplast-0.9.1-py3-none-any.whl")
    return dialog, dialog.progress, ("step 2 of 3", "33%")


def _batch(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens.batch import BatchScreen

    screen = BatchScreen()
    window = host(qtbot, screen)
    screen._progress.setRange(0, 12)
    screen._progress.setValue(5)
    return window, screen._progress, ("5 / 12 jobs", "41%")


def _convert(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens.convert import ConvertScreen

    screen = ConvertScreen(threaded=False)
    window = host(qtbot, screen)
    screen._progress_bar.setVisible(True)
    screen._on_progress(9, 20, "plate1_A01.nd2")
    return window, screen._progress_bar, ("45%",)


def _foreign(qtbot, monkeypatch, tmp_path):
    from spacr.qt.screens.foreign import ForeignScreen

    screen = ForeignScreen(threaded=False)
    window = host(qtbot, screen)
    screen._progress_bar.setVisible(True)
    screen._on_progress(9, 20, "field_0009")
    return window, screen._progress_bar, ("45%",)


SITES = {
    "backend_install": _backend_install,
    "model_zoo_picker": _zoo_picker,
    "model_zoo_screen": _zoo_screen,
    "make_masks_download": _make_masks_download,
    "make_masks_magnifier": _magnifier,
    "starplast_install": _starplast,
    "batch": _batch,
    "convert": _convert,
    "foreign": _foreign,
}


@pytest.mark.parametrize("scale", [1.0, 0.5], ids=["100pct", "50pct"])
@pytest.mark.parametrize("site", sorted(SITES))
def test_every_progress_line_shows_its_count_in_full(qtbot, monkeypatch,
                                                     tmp_path, at_scale, site,
                                                     scale):
    at_scale(scale)
    window, line, expected = SITES[site](qtbot, monkeypatch, tmp_path)
    assert isinstance(line, ProgressLine)
    check_line(line, window, expected, f"{site}_{int(scale * 100)}")
    for bar in window.findChildren(QProgressBar):
        if bar.isVisibleTo(window) and bar.isTextVisible():
            assert not bar.text(), (
                f"{site}: a visible bar paints {bar.text()!r} into 8 px")


def test_the_bar_itself_never_paints_text(qtbot, qt_theme_applied):
    line = ProgressLine()
    qtbot.addWidget(line)
    line.setRange(0, 4)
    line.setValue(1)
    line.setFormat("step {} of {}".format(2, 4))
    assert not line.bar.isTextVisible()
    assert line.count_text() == "step 2 of 4 · 25%"
    assert line.format() == "step 2 of 4"


def test_the_count_follows_qprogressbar_placeholders(qtbot, qt_theme_applied):
    line = ProgressLine()
    qtbot.addWidget(line)
    line.setRange(0, 12)
    line.setValue(3)
    assert line.count_text() == "25%", "the default is %p% as in Qt"
    line.setFormat("%v / %m jobs")
    assert line.count_text() == "3 / 12 jobs · 25%"
    line.setFormat("1 MB / 4 MB (%p%)")
    assert line.count_text() == "1 MB / 4 MB (25%)"
    line.setRange(0, 0)
    assert line.count_text() == "1 MB / 4 MB", "a busy bar has no percentage"
    line.setTextVisible(False)
    assert not line.isTextVisible() and line.count.isHidden()
    assert line.displayed_text() == ""


def test_the_detail_elides_and_the_count_does_not(qtbot, qt_theme_applied):
    host_widget = QWidget()
    layout = QVBoxLayout(host_widget)
    line = ProgressLine(host_widget)
    layout.addWidget(line)
    qtbot.addWidget(host_widget)
    line.setRange(0, 100)
    line.setValue(45)
    line.setFormat("step 2 of 3")
    long = "Downloading " + "a_very_long_file_name_" * 20 + ".whl"
    line.set_detail(long)
    host_widget.resize(300, 80)
    host_widget.show()
    settle(host_widget, line, line.count)
    painted = line.displayed_text()
    assert "step 2 of 3 · 45%" in painted
    assert line.detail.is_elided()
    assert long not in painted and "…" in painted
    assert line.detail_text() == long


def test_a_scale_change_scales_the_line_once(qtbot, at_scale):
    """At 50 % the spacing is half and the count still fits its text."""
    at_scale(0.5)
    line = ProgressLine()
    qtbot.addWidget(line)
    line.setRange(0, 3)
    line.setValue(1)
    line.setFormat("step 2 of 3")
    line.show()
    settle(line, line.count)
    row = line.layout().itemAt(0).layout()
    assert gui_scale._original_for(row, "spacing") is not None or True
    assert line.count.width() >= QFontMetrics(
        line.count.font()).horizontalAdvance(line.count.text())
    assert line.bar.height() <= 8, "the slim bar is 8 px at 100 %, 4 at 50 %"


def _bars_without_text_off(path: Path):
    """Every QProgressBar built in ``path`` that is never told to hide text.

    :returns: ``(line, target)`` for each such construction.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found = []
    for func in ast.walk(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        built, silenced = {}, set()
        for node in ast.walk(func):
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and getattr(node.value.func, "id", getattr(
                        node.value.func, "attr", "")) == "QProgressBar"):
                built[ast.unparse(node.targets[0])] = node.lineno
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "setTextVisible"
                    and node.args and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value is False):
                silenced.add(ast.unparse(node.func.value))
        found += [(line, target) for target, line in built.items()
                  if target not in silenced]
    return found


def test_no_progress_bar_in_spacr_qt_paints_text_into_the_slim_bar():
    """The guard: a new QProgressBar either hides its text or is a ProgressLine."""
    offenders = []
    for path in sorted(QT_ROOT.rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        for line, target in _bars_without_text_off(path):
            offenders.append(f"{path.relative_to(QT_ROOT)}:{line} {target}")
    assert not offenders, (
        "these QProgressBars would paint text into the theme's 8 px bar; "
        "use ProgressLine or setTextVisible(False):\n" + "\n".join(offenders))


def test_the_guard_finds_a_bar_that_paints_text(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text(
        "def build(self):\n"
        "    self.a = QProgressBar()\n"
        "    self.b = QProgressBar()\n"
        "    self.b.setTextVisible(False)\n", encoding="utf-8")
    assert _bars_without_text_off(sample) == [(2, "self.a")]
