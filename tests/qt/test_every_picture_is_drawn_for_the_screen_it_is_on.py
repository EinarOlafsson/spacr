"""Every picture is rendered at the density of the panel it lands on.

``QPixmap.scaled(w, h, ...)`` counts DEVICE pixels while every caller in a
GUI means LOGICAL ones. On a display with a device pixel ratio of 2 -- every
retina Mac, and plenty of Windows and Linux machines -- Qt stretches the
logical bitmap across twice as many real pixels in each direction, so a 72 px
render of a 3334 px logo is blown up to 144 px of blur. That is the whole of
"the spacr logo looks super low res on the mac": not the artwork, the
scaling.

WHAT IS MEASURED HERE, and why it is measured this way
------------------------------------------------------
The device pixel ratio is not something a widget can be talked into. So the
headline numbers come from a REAL Qt process running at a REAL ratio:
:data:`_BATTERY` is executed twice under ``QT_SCALE_FACTOR=1`` and
``QT_SCALE_FACTOR=2``, builds the actual :class:`~spacr.qt.widgets.home.HomePage`,
the actual mask canvas, the actual figure strip, and reports what is on the
widgets. Nothing in that subprocess is patched.

The two ratios answer the two halves of the claim: at 2 the pictures carry
twice the pixels and still occupy the same space, and at 1 every number is
what it was before any of this was written, so no ordinary display regresses.

The rest of the file drives the same code in-process, where a stand-in ratio
is enough because the seam under test is what the code does WITH the ratio,
not how it obtained it.

WHAT IN-PROCESS CANNOT SHOW, measured while writing this: ``QLabel.setPixmap``
NORMALISES the picture to the widget's REAL device pixel ratio. Hand a
144 px 2x pixmap to a label on an ordinary screen and ``label.pixmap()``
gives back 72 px at 1x. So a stand-in ratio proves what the render produced,
never what the label kept -- which is why the widget measurements are the
subprocess's job and the in-process tests read the render, the geometry, or
the fact that a redraw happened at all.

Also measured, and the reason none of this is already handled by Qt: on a
REAL 2x screen, a label given the old 72 px 1x render keeps it at 72 px 1x
and paints it across 144 device pixels. Qt does not compensate. The stretch
is real.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel

from spacr.qt import hidpi
from spacr.qt.hidpi import (
    MAX_RATIO, device_ratio, follow_device_ratio, logical_size, scaled_for,
)

QT_ROOT = Path(__file__).resolve().parents[2] / "spacr" / "qt"


# --------------------------------------------------------------------------
# A real Qt process, at a real device pixel ratio
# --------------------------------------------------------------------------

_BATTERY = r'''
import json
import tempfile

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QApplication, QLabel

app = QApplication([])
from spacr.qt.hidpi import device_ratio, logical_size


def lg(pixmap):
    size = logical_size(pixmap)
    return [size.width(), size.height()]


out = {"ratio": device_ratio(QLabel())}

# The masthead -- the picture the complaint was about.
from spacr.qt.widgets.home import HomePage
page = HomePage([("mask", "Mask", "d", "Core")], lambda key: None)
mark = page._hero_mark
pixmap = mark.pixmap()
out["home"] = {
    "label": [mark.width(), mark.height()],
    "hint": [mark.sizeHint().width(), mark.sizeHint().height()],
    "device": [pixmap.width(), pixmap.height()],
    "dpr": pixmap.devicePixelRatio(),
    "logical": lg(pixmap),
}

# The mask canvas: a picture AND a coordinate system laid over it.
from spacr.qt.screens.make_masks import _MaskCanvas
canvas = _MaskCanvas()
canvas.resize(400, 300)
rng = np.random.default_rng(0)
canvas.image = (rng.random((200, 200)) * 60000).astype(np.uint16)
canvas.mask = np.zeros((200, 200), np.uint8)
canvas.mask[50:150, 50:150] = 1
canvas.refresh()
shown = canvas.pixmap()
back = canvas._image_to_canvas(100, 100)
out["canvas"] = {
    "device": [shown.width(), shown.height()],
    "dpr": shown.devicePixelRatio(),
    "logical": lg(shown),
    "widget": [canvas.width(), canvas.height()],
    "click_maps_to": list(canvas._canvas_to_image(200, 150)),
    "pixel_maps_to": [back.x(), back.y()],
    "brush": canvas._mask_radius_for_brush(),
    "drag": list(canvas._image_delta(10, 10)),
}

# The figures strip: a QIcon, which has its own idea of density.
from spacr.qt.widgets.figure_queue import FigureQueue, THUMB_SIZE
queue = FigureQueue()
source = QPixmap(1400, 900)
source.fill(Qt.red)
icon = queue._thumb_icon(source)
served = icon.pixmap(THUMB_SIZE, out["ratio"])
out["strip"] = {
    "device": [served.width(), served.height()],
    "dpr": served.devicePixelRatio(),
    "logical": lg(served),
    "icon_size": [THUMB_SIZE.width(), THUMB_SIZE.height()],
}

# The crop preview, which rescales itself on every resize.
from spacr.qt.widgets.umap_explorer import _ScaledPreview
preview = _ScaledPreview()
preview.resize(200, 200)
crop = QPixmap(600, 600)
crop.fill(Qt.blue)
preview.setPixmap(crop)
out["preview"] = {
    "device": [preview.pixmap().width(), preview.pixmap().height()],
    "dpr": preview.pixmap().devicePixelRatio(),
    "logical": lg(preview.pixmap()),
    "widget": [preview.width(), preview.height()],
}

# A figure grid cell, which sizes its own box round the picture.
from spacr.qt.widgets.figure_grid_view import _FigureCell
figure = QPixmap(1000, 500)
figure.fill(Qt.green)
cell = _FigureCell(0, figure, "t")
cell.fit_to(300)
drawn = cell._image.pixmap()
out["grid"] = {
    "device": [drawn.width(), drawn.height()],
    "dpr": drawn.devicePixelRatio(),
    "logical": lg(drawn),
    "reserved_height": cell._image.height(),
}

# The splash, which scales its logo inside paintEvent. Measured as PAINTED
# PIXELS: where the mark actually lands, in the widget's own coordinates.
from spacr.qt.widgets.loading_screen import LoadingScreen
splash = LoadingScreen(3)
splash.resize(700, 300)
shot = splash.grab()
image = shot.toImage()
background = image.pixelColor(2, 2)
box = [None, None, None, None]
for y in range(image.height()):
    for x in range(int(image.width() * 0.30)):   # the mark, left of the words
        if image.pixelColor(x, y) != background:
            box[0] = x if box[0] is None else min(box[0], x)
            box[1] = y if box[1] is None else box[1]
            box[2] = x if box[2] is None else max(box[2], x)
            box[3] = y
out["splash"] = {
    "grab": [shot.width(), shot.height()],
    "grab_dpr": shot.devicePixelRatio(),
    "mark_box_logical": [round(v / out["ratio"]) for v in box],
}

# The Cellpose flow pane, composited once and then left up.
from spacr.qt.screens.make_masks import _FlowPane
pane = _FlowPane()
pane.resize(400, 300)
pane.show_rgb(np.full((240, 240, 3), 120, np.uint8))
out["flow"] = {
    "device": [pane.pixmap().width(), pane.pixmap().height()],
    "dpr": pane.pixmap().devicePixelRatio(),
    "logical": lg(pane.pixmap()),
}

# A tutorial video frame, which is measured in FILE pixels and must not move.
from spacr.qt.tutorial.engine import Recorder
window = QLabel("x")
window.resize(400, 300)
window.show()
recorder = Recorder(window, tempfile.mkdtemp(), size=(320, 240))
recorder.refresh_base()
frame = recorder._base_frame
out["video"] = {
    "device": [frame.width(), frame.height()],
    "dpr": frame.devicePixelRatio(),
    "logical": lg(frame),
    "asked": [320, 240],
}

# The icons. `iconset` draws vector glyphs, which are resolution-free, but
# the bundled PNGs were the other suspect: they go to QIcon WHOLE, never
# pre-scaled, so QIcon serves the density it is asked for. Measured, not
# assumed.
from PySide6.QtCore import QSize
from spacr.qt import iconset
out["icons"] = {}
for name in ("mask", "measure", "classify"):
    served = iconset.app_icon(name).pixmap(QSize(24, 24), out["ratio"])
    out["icons"][name] = [served.width(), served.height(),
                          served.devicePixelRatio()]
glyph = iconset.icon("open", size=24).pixmap(QSize(24, 24), out["ratio"])
out["icons"]["glyph"] = [glyph.width(), glyph.height(),
                         glyph.devicePixelRatio()]

print("BATTERY " + json.dumps(out))
'''


def _run_battery(ratio: int) -> dict:
    """Run :data:`_BATTERY` in a Qt process whose real ratio is ``ratio``."""
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["QT_SCALE_FACTOR"] = str(ratio)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run([sys.executable, "-c", _BATTERY],
                          capture_output=True, text=True, timeout=240,
                          env=env, cwd=str(QT_ROOT.parents[1]))
    line = next((ln for ln in proc.stdout.splitlines()
                 if ln.startswith("BATTERY ")), None)
    assert line is not None, (
        f"the measuring process said nothing usable.\n"
        f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}")
    return json.loads(line[len("BATTERY "):])


@pytest.fixture(scope="module")
def measured() -> dict:
    """What a real Qt process reports at a ratio of 1 and of 2."""
    return {1: _run_battery(1), 2: _run_battery(2)}


# --------------------------------------------------------------------------
# The headline: the logo, on a display that can show it
# --------------------------------------------------------------------------

def test_the_masthead_logo_is_drawn_at_the_screens_own_resolution(measured):
    """Twice the pixels on a 2x screen, in a label that has not moved."""
    hero = measured[2]["home"]
    assert hero["dpr"] == 2.0, "the pixmap does not say it is a dense one"
    assert hero["device"] == [144, 144], (
        f"rendered {hero['device']} device px for a 72 px mark -- that is the "
        "old logical render being stretched, which is the whole complaint")
    assert hero["logical"] == [72, 72]
    # "the label it sits in is still the logical size, so nothing moved"
    assert hero["label"] == [72, 72]
    assert hero["hint"] == [72, 72], (
        "the label asks the layout for the pixel count instead of the size --"
        " that is setDevicePixelRatio missing, and it makes the masthead"
        " twice as wide")


def test_an_ordinary_display_renders_exactly_what_it_did_before(measured):
    """At a ratio of 1 every measurement is the pre-existing one."""
    plain = measured[1]
    assert plain["ratio"] == 1.0
    assert plain["home"]["device"] == [72, 72]
    assert plain["home"]["dpr"] == 1.0
    assert plain["strip"]["device"] == [140, 90]
    assert plain["preview"]["device"] == [200, 200]
    assert plain["grid"]["device"] == [300, 150]
    assert plain["video"]["device"] == [320, 240]
    # Same pixel dimensions as today, whichever ratio the picture came from.
    for key in ("home", "strip", "preview", "grid"):
        assert plain[key]["device"] == [d // 2 for d in
                                        measured[2][key]["device"]], key


def test_the_pictures_that_fill_a_widget_follow_it(measured):
    """A crop, a figure cell and a strip icon all double, none of them grow."""
    for key, logical in (("strip", [140, 90]), ("preview", [200, 200]),
                         ("grid", [300, 150]), ("flow", [300, 300])):
        dense = measured[2][key]
        assert dense["dpr"] == 2.0, key
        assert dense["logical"] == logical, key
        assert dense["device"] == [v * 2 for v in logical], key
    # The cell reserves the height the figure OCCUPIES, not its pixel count.
    assert measured[2]["grid"]["reserved_height"] == 150
    assert measured[1]["grid"]["reserved_height"] == 150


def test_the_icons_were_already_right_and_stay_that_way(measured):
    """Neither icon path pre-scales, so QIcon serves the density asked for.

    Worth measuring rather than assuming: a bundled PNG handed to ``QIcon``
    whole is served at whatever density the painter requests, while the same
    PNG pre-scaled to a logical size -- which is what every other picture in
    the application was doing -- is not. This is the one family that was
    already correct, and this is what keeps it correct.
    """
    for name in ("mask", "measure", "classify", "glyph"):
        plain = measured[1]["icons"][name]
        dense = measured[2]["icons"][name]
        assert plain == [24, 24, 1.0], name
        assert dense == [48, 48, 2.0], (
            f"{name} is served at {dense} on a 2x screen -- something on the"
            " icon path has started pre-scaling to a logical size")


# --------------------------------------------------------------------------
# A picture with a coordinate system on it
# --------------------------------------------------------------------------

def test_a_click_on_the_mask_canvas_lands_on_the_same_object_at_either_ratio(
        measured):
    """The canvas is denser; the geometry it is read through is not.

    ``QPixmap.width()`` counts device pixels, so a canvas that renders at
    the screen's density reports twice its on-screen width. Everything that
    maps a mouse position onto the picture has to keep speaking in widget
    coordinates or every stroke lands on the wrong object.
    """
    plain, dense = measured[1]["canvas"], measured[2]["canvas"]
    assert dense["dpr"] == 2.0
    assert dense["device"] == [v * 2 for v in plain["device"]]
    assert dense["logical"] == plain["logical"]
    assert dense["click_maps_to"] == plain["click_maps_to"], (
        "the same click reaches a different image pixel on a HiDPI screen")
    assert dense["pixel_maps_to"] == plain["pixel_maps_to"]
    assert dense["brush"] == plain["brush"]
    assert dense["drag"] == plain["drag"]


def test_the_splash_paints_its_mark_in_the_same_place_on_any_screen(measured):
    """Scaled inside a paint, and centred on what it OCCUPIES.

    The mark is centred by subtracting half the scaled picture's height.
    ``QPixmap.height()`` is a device-pixel count, so on a 2x screen that
    subtracts twice as much and the logo climbs out of the strapline it is
    meant to sit beside -- measured at 70 logical px too high before the
    centring was read through :func:`logical_size`.
    """
    plain, dense = measured[1]["splash"], measured[2]["splash"]
    assert dense["grab"] == [v * 2 for v in plain["grab"]], (
        "the splash is not being captured at the screen's density at all")
    assert dense["grab_dpr"] == 2.0
    assert plain["mark_box_logical"][0] is not None, "nothing was painted"
    for painted, expected in zip(dense["mark_box_logical"],
                                 plain["mark_box_logical"]):
        assert abs(painted - expected) <= 1, (
            f"the mark is painted at {dense['mark_box_logical']} on a 2x"
            f" screen and {plain['mark_box_logical']} on a 1x one")


def test_a_video_frame_is_the_same_recording_on_every_machine(measured):
    """The one picture measured in file pixels, not screen ones.

    ``window.grab()`` comes back carrying the window's ratio. A frame that
    still claims to be dense draws at a fraction of its size on the plain
    canvas the recorder composites onto -- a quarter-size picture in the
    corner of every frame on a retina display.
    """
    for ratio in (1, 2):
        frame = measured[ratio]["video"]
        assert frame["device"] == [320, 240], ratio
        assert frame["dpr"] == 1.0, ratio
        assert frame["logical"] == [320, 240], ratio


# --------------------------------------------------------------------------
# The helper itself
# --------------------------------------------------------------------------

class _AtRatio:
    """Something that answers ``devicePixelRatioF``, and nothing else."""

    def __init__(self, ratio: float):
        self.ratio = float(ratio)

    def devicePixelRatioF(self) -> float:              # noqa: N802 - Qt name
        return self.ratio


@pytest.mark.parametrize("side", [1, 16, 72, 96, 140, 519])
def test_the_helper_is_what_scaled_already_did_on_a_plain_display(qapp, side):
    """Byte-for-byte the old render at a ratio of 1."""
    source = QPixmap(1000, 800)
    source.fill(Qt.red)
    before = source.scaled(side, side, Qt.KeepAspectRatio,
                           Qt.SmoothTransformation)
    after = scaled_for(source, _AtRatio(1.0), side)
    assert (after.width(), after.height()) == (before.width(), before.height())
    assert after.devicePixelRatio() == 1.0
    assert after.toImage() == before.toImage(), (
        "the same pixels, or an ordinary display has regressed")


def test_a_dense_screen_gets_dense_pixels_and_the_same_layout(qapp):
    """The two lines, and what each one is for."""
    source = QPixmap(3334, 3334)
    source.fill(Qt.blue)
    dense = scaled_for(source, _AtRatio(2.0), 72)
    assert (dense.width(), dense.height()) == (144, 144)
    assert dense.devicePixelRatio() == 2.0
    assert logical_size(dense) == QSize(72, 72)

    # MISSING THE SECOND LINE IS WORSE THAN MISSING BOTH: without it the
    # picture is right in pixels and twice the intended size, because Qt has
    # no way to know those pixels are dense ones.
    label = QLabel()
    label.setPixmap(dense)
    assert label.sizeHint() == QSize(72, 72)

    naive = source.scaled(144, 144, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    naive.setDevicePixelRatio(1.0)
    other = QLabel()
    other.setPixmap(naive)
    assert other.sizeHint() == QSize(144, 144), (
        "this is the failure the second line prevents; if it stops happening"
        " the guard above no longer guards anything")


def test_a_fractional_ratio_still_lands_on_whole_pixels(qapp):
    """1.25 and 1.5 displays are ordinary Windows machines, not edge cases."""
    source = QPixmap(400, 400)
    source.fill(Qt.green)
    for ratio, expected in ((1.25, 90), (1.5, 108), (3.0, 216)):
        out = scaled_for(source, _AtRatio(ratio), 72)
        assert out.width() == expected, ratio
        assert out.devicePixelRatio() == ratio
        assert logical_size(out) == QSize(72, 72), ratio


def test_a_size_can_be_given_in_any_of_the_shapes_a_call_site_has(qapp):
    source = QPixmap(400, 200)
    source.fill(Qt.white)
    target = _AtRatio(2.0)
    square = scaled_for(source, target, 50)
    pair = scaled_for(source, target, 50, 50)
    size = scaled_for(source, target, QSize(50, 50))
    tup = scaled_for(source, target, (50, 50))
    sizes = {(p.width(), p.height()) for p in (square, pair, size, tup)}
    assert sizes == {(100, 50)}


def test_the_ratio_is_found_when_the_widget_will_not_say(qapp):
    """A missing answer falls through to the screen, never to nonsense."""
    assert device_ratio(None) == float(
        qapp.primaryScreen().devicePixelRatio())
    assert device_ratio(object()) == float(
        qapp.primaryScreen().devicePixelRatio())

    class _ViaScreen:
        def screen(self):
            return _AtRatio(2.0)

    assert device_ratio(_ViaScreen()) == 2.0

    class _Hostile:
        def devicePixelRatioF(self):
            return 4000.0

    assert device_ratio(_Hostile()) == MAX_RATIO, (
        "a broken QT_SCALE_FACTOR would render a 72 px logo at 288000 px,"
        " which is an out-of-memory crash rather than a sharper logo")

    class _Broken:
        def devicePixelRatioF(self):
            raise RuntimeError("no screen")

    assert device_ratio(_Broken()) == float(
        qapp.primaryScreen().devicePixelRatio())


def test_a_picture_reads_back_the_size_it_occupies(qapp):
    dense = QPixmap(300, 200)
    dense.setDevicePixelRatio(2.0)
    assert logical_size(dense) == QSize(150, 100)
    assert logical_size(QPixmap()) == QSize(0, 0)
    assert logical_size(None) == QSize(0, 0)


def test_a_null_source_comes_back_untouched(qapp):
    empty = QPixmap()
    assert scaled_for(empty, _AtRatio(2.0), 40) is empty
    assert scaled_for(None, _AtRatio(2.0), 40) is None


def test_a_qimage_takes_the_same_route(qapp):
    """Anything drawn by hand into a buffer asks the same helper."""
    from PySide6.QtGui import QImage

    source = QImage(400, 400, QImage.Format_RGB888)
    source.fill(Qt.black)
    out = scaled_for(source, _AtRatio(2.0), 50)
    assert (out.width(), out.height()) == (100, 100)
    assert out.devicePixelRatio() == 2.0


# --------------------------------------------------------------------------
# Dragging a window between screens
# --------------------------------------------------------------------------

def test_a_move_to_a_denser_screen_redraws_from_the_master(qapp, qtbot):
    """Re-rendered from the source, not re-scaled from what is on screen.

    A picture rendered at 2x and moved to a 1x screen is merely wasteful;
    the reverse is blurry, so the reverse is what has to be noticed. Qt
    sends ``DevicePixelRatioChange`` for both.
    """
    # A source with real detail in it: a resample of a resample loses
    # detail a resample of the master keeps, so the two are distinguishable.
    source = QPixmap(288, 288)
    painter_image = source.toImage()
    source.fill(Qt.black)
    del painter_image
    from PySide6.QtGui import QPainter

    painter = QPainter(source)
    for step in range(0, 288, 3):
        painter.fillRect(step, 0, 1, 288, Qt.white)
    painter.end()

    label = QLabel()
    qtbot.addWidget(label)
    drawn = []

    def redraw():
        # What the RENDER produced. The label would normalise it back to the
        # process's real ratio -- see the module docstring.
        picture = scaled_for(source, label, 72)
        drawn.append(picture)
        label.setPixmap(picture)

    redraw()
    watcher = follow_device_ratio(label, redraw)
    assert watcher is not None
    assert drawn[0].width() == 72

    # A ratio-change event that changes nothing must not cost a redraw.
    qapp.sendEvent(label, QEvent(QEvent.Type.DevicePixelRatioChange))
    assert len(drawn) == 1

    label.devicePixelRatioF = lambda: 2.0
    qapp.sendEvent(label, QEvent(QEvent.Type.DevicePixelRatioChange))
    assert len(drawn) == 2, "the move onto a denser screen was not noticed"
    assert drawn[1].width() == 144
    assert drawn[1].devicePixelRatio() == 2.0
    assert drawn[1].toImage() == scaled_for(source, label, 72).toImage(), (
        "the redraw went through the picture already on the label rather than"
        " the master, so every screen the window crosses costs a resample")
    # A resample of a resample is measurably not the master's answer, which
    # is what makes the line above an assertion rather than a tautology.
    twice = scaled_for(drawn[0], label, 72)
    assert twice.toImage() != drawn[1].toImage()

    label.devicePixelRatioF = lambda: 1.0
    qapp.sendEvent(label, QEvent(QEvent.Type.DevicePixelRatioChange))
    assert len(drawn) == 3, "the move back is not noticed either"
    assert drawn[2].width() == 72


def test_a_redraw_that_throws_does_not_take_the_window_with_it(qapp, qtbot):
    """A picture that cannot be drawn again is a soft picture, not a crash."""
    label = QLabel()
    qtbot.addWidget(label)

    def redraw():
        raise RuntimeError("the picture is gone")

    follow_device_ratio(label, redraw)
    label.devicePixelRatioF = lambda: 2.0
    qapp.sendEvent(label, QEvent(QEvent.Type.DevicePixelRatioChange))


# --------------------------------------------------------------------------
# The sweep: every site, or a reason
# --------------------------------------------------------------------------

#: The only bare ``.scaled()`` calls allowed under ``spacr/qt``, each with
#: the reason it is not a picture being put on a screen.
SCALED_EXEMPT = {
    "hidpi.py":
        "the helper itself -- this is the one place the raw call belongs",
    "widgets/ambient.py":
        "the blur, an internal downscale of the ambient buffer, which "
        "already carries the ratio it was allocated at and is blitted back "
        "up at the same density",
    "tutorial/engine.py":
        "a video frame, measured in FILE pixels: a recording must come out "
        "the same size on every machine",
}


def _scaled_call_sites() -> dict:
    """``relative path -> [line numbers]`` for every ``.scaled(`` under qt.

    Parsed rather than grepped: a module docstring that NAMES the method --
    ``ambient`` explains at length why its blur is one -- is documentation,
    not a call site, and a sweep that counts it teaches people to stop
    explaining themselves.
    """
    import ast

    found: dict = {}
    unreadable = []
    for path in sorted(QT_ROOT.rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8",
                                            errors="replace"))
        except SyntaxError as exc:
            unreadable.append(f"{path.relative_to(QT_ROOT)}: {exc}")
            continue
        lines = [node.lineno for node in ast.walk(tree)
                 if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "scaled"]
        if lines:
            found[str(path.relative_to(QT_ROOT))] = sorted(lines)
    assert not unreadable, (
        "a file under spacr/qt does not parse, so this sweep cannot say"
        " whether it scales a picture: " + "; ".join(unreadable))
    return found


def test_every_scaled_site_goes_through_the_helper_or_is_named():
    """A rule applied by hand holds until the next site is written.

    This is the rule applied by the suite instead: a new ``.scaled()``
    anywhere under ``spacr/qt`` fails here until its author either routes it
    through :func:`spacr.qt.hidpi.scaled_for` or writes down, in
    :data:`SCALED_EXEMPT`, why the picture is not one being shown on a
    screen.
    """
    sites = _scaled_call_sites()
    unexplained = {path: lines for path, lines in sites.items()
                   if path not in SCALED_EXEMPT}
    assert not unexplained, (
        "these render a picture at logical pixels and let Qt stretch it:\n"
        + "\n".join(f"  spacr/qt/{path}:{lines}"
                    for path, lines in sorted(unexplained.items()))
        + "\nUse spacr.qt.hidpi.scaled_for(source, widget, size), or add the"
          " file to SCALED_EXEMPT with the reason it does not need to.")
    # And the exemptions are real files, so the list cannot rot quietly.
    for path in SCALED_EXEMPT:
        assert (QT_ROOT / path).is_file(), path
        assert path in sites, (
            f"{path} is exempted from a rule it no longer breaks -- drop it")


def test_the_pictures_are_asked_for_by_the_widget_they_land_on():
    """Every call names a target; a picture scaled for nothing is the bug."""
    import ast

    offenders = []
    for path in sorted(QT_ROOT.rglob("*.py")):
        if "i18n_catalogs" in path.parts or path.name == "hidpi.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", None)
            if name != "scaled_for":
                continue
            if len(node.args) < 3:
                offenders.append(f"{path.relative_to(QT_ROOT)}:{node.lineno}")
    assert not offenders, (
        "scaled_for needs a source, a target and a logical size: " +
        ", ".join(offenders))


# --------------------------------------------------------------------------
# The one hand-drawn buffer that had to learn the same lesson
# --------------------------------------------------------------------------

def test_a_rounded_crop_keeps_the_whole_crop_on_a_dense_screen(qapp):
    """Rounding the corners must not crop the picture into one of them.

    The rounding canvas is allocated at the source's PIXEL size. Left at a
    ratio of 1 it paints a dense thumbnail at half scale into the top-left
    quarter of a box twice as large -- the crop still there, three quarters
    of the tile empty.
    """
    from spacr.qt.widgets.measure_preview import _rounded_pixmap

    dense = QPixmap(200, 200)
    dense.fill(Qt.red)
    dense.setDevicePixelRatio(2.0)

    out = _rounded_pixmap(dense, radius=8)
    assert out.devicePixelRatio() == 2.0
    assert (out.width(), out.height()) == (200, 200)
    image = out.toImage()
    # The middle of the tile, and a point well inside the far corner, are
    # both the crop. Only the rounded corners themselves are transparent.
    assert image.pixelColor(100, 100).alpha() == 255
    assert image.pixelColor(180, 180).alpha() == 255, (
        "three quarters of the tile is empty: the crop was drawn at half"
        " scale into the corner")
    assert image.pixelColor(0, 0).alpha() == 0, "the corner is still rounded"


def test_the_crop_thumbnail_is_rounded_at_a_constant_size(qapp):
    """8 logical px of radius on every display, not 8 device px."""
    from spacr.qt.widgets.measure_preview import _rounded_pixmap

    plain = QPixmap(100, 100)
    plain.fill(Qt.red)
    dense = QPixmap(200, 200)
    dense.fill(Qt.red)
    dense.setDevicePixelRatio(2.0)

    def corner_depth(pixmap, ratio):
        """How far in from the corner, in LOGICAL px, the picture starts."""
        image = pixmap.toImage()
        for step in range(pixmap.width()):
            if image.pixelColor(step, step).alpha() > 0:
                return step / ratio
        raise AssertionError("the whole tile was clipped away")

    assert corner_depth(_rounded_pixmap(plain), 1.0) == pytest.approx(
        corner_depth(_rounded_pixmap(dense), 2.0), abs=0.6)


# --------------------------------------------------------------------------
# In-process, through the widget's own route
# --------------------------------------------------------------------------

def _mask_canvas(qtbot):
    from spacr.qt.screens.make_masks import _MaskCanvas

    canvas = _MaskCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(400, 300)
    rng = np.random.default_rng(1)
    canvas.image = (rng.random((120, 120)) * 60000).astype(np.uint16)
    canvas.mask = np.zeros((120, 120), np.uint8)
    canvas.mask[20:80, 20:80] = 1
    canvas.refresh()
    return canvas


def test_the_mask_canvas_reads_its_picture_in_widget_coordinates(qapp, qtbot):
    """A dense canvas must not move the objects under the cursor.

    Every gesture on this canvas -- brush, wand, draw, divide, recrop, pan --
    is mapped through the displayed pixmap. ``QPixmap.width()`` counts DEVICE
    pixels, so once the canvas composites at the panel's density that number
    doubles while the widget does not, and a mapping that believes it lands
    on the wrong object by the same factor.
    """
    canvas = _mask_canvas(qtbot)
    plain = canvas.pixmap()
    answers = (canvas._canvas_to_image(150, 120),
               canvas._image_to_canvas(60, 60).toTuple(),
               canvas._mask_radius_for_brush(),
               canvas._image_delta(10, 10))

    # The same picture, composited for a 2x panel: twice the pixels, the
    # same size on screen. Handed in directly because a QLabel normalises a
    # pixmap to the ratio the running process really has.
    dense = plain.scaled(plain.width() * 2, plain.height() * 2,
                         Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    dense.setDevicePixelRatio(2.0)
    assert logical_size(dense) == logical_size(plain)
    canvas.pixmap = lambda: dense

    assert (canvas._canvas_to_image(150, 120),
            canvas._image_to_canvas(60, 60).toTuple(),
            canvas._mask_radius_for_brush(),
            canvas._image_delta(10, 10)) == answers, (
        "the canvas is reading device pixels as widget coordinates, so every"
        " stroke on a HiDPI screen lands on a different object")


@pytest.mark.parametrize("build,redrawn", [
    ("mask_canvas", "the field is composited once and stays up between edits"),
    ("flow_pane", "the flow picture is composited once per run"),
    ("crop_preview", "the crop is rescaled on resize, and a screen change"
                     " is not a resize"),
    ("figure_cell", "the grid keeps its cell widths across a screen change"),
])
def test_a_picture_that_stays_up_is_redrawn_when_the_screen_changes(
        qapp, qtbot, build, redrawn):
    """Qt sends the change; these four widgets act on it.

    Each holds its source and would otherwise keep the old density for the
    rest of the session. What is asserted is that the redraw RAN through the
    widget's own route -- a fresh pixmap, not the one that was there -- since
    the density it lands at is the running process's, not a test's.
    """
    if build == "mask_canvas":
        widget = _mask_canvas(qtbot)
    elif build == "flow_pane":
        from spacr.qt.screens.make_masks import _FlowPane

        widget = _FlowPane()
        qtbot.addWidget(widget)
        widget.resize(400, 300)
        widget.show_rgb(np.full((240, 240, 3), 120, np.uint8))
    elif build == "crop_preview":
        from spacr.qt.widgets.umap_explorer import _ScaledPreview

        widget = _ScaledPreview()
        qtbot.addWidget(widget)
        widget.resize(160, 160)
        crop = QPixmap(500, 500)
        crop.fill(Qt.magenta)
        widget.setPixmap(crop)
    else:
        from spacr.qt.widgets.figure_grid_view import _FigureCell

        figure = QPixmap(1000, 500)
        figure.fill(Qt.green)
        cell = _FigureCell(0, figure, "t")
        qtbot.addWidget(cell)
        cell.fit_to(300)
        widget = cell._image

    before = widget.pixmap().cacheKey()
    assert widget.pixmap().width() > 0, redrawn

    # No change, no redraw.
    qapp.sendEvent(widget, QEvent(QEvent.Type.DevicePixelRatioChange))
    assert widget.pixmap().cacheKey() == before

    widget.devicePixelRatioF = lambda: 2.0
    qapp.sendEvent(widget, QEvent(QEvent.Type.DevicePixelRatioChange))
    qapp.processEvents()
    assert widget.pixmap().cacheKey() != before, (
        f"nothing redrew: {redrawn}, so it would stay at the old density")


def test_the_figures_strip_serves_a_dense_thumbnail(qapp, qtbot):
    """A QIcon keeps the density it was handed, and hands it back."""
    from spacr.qt.widgets.figure_queue import FigureQueue, THUMB_SIZE

    queue = FigureQueue()
    qtbot.addWidget(queue)
    assert queue._list.iconSize() == THUMB_SIZE, (
        "the strip renders to one size and the list draws another")

    source = QPixmap(1400, 900)
    source.fill(Qt.cyan)
    queue._list.devicePixelRatioF = lambda: 2.0
    icon = queue._thumb_icon(source)
    served = icon.pixmap(THUMB_SIZE, 2.0)
    assert (served.width(), served.height()) == (280, 180)
    assert logical_size(served) == QSize(140, 90)


# --------------------------------------------------------------------------
# The helper answers something for every shape of caller
# --------------------------------------------------------------------------

def test_a_target_whose_screen_will_not_answer_falls_through(qapp):
    """A widget mid-teardown has a ``screen()`` that raises. That is not 0."""

    class _Dying:
        def screen(self):
            raise RuntimeError("no window any more")

    assert device_ratio(_Dying()) == float(
        qapp.primaryScreen().devicePixelRatio())


class _NoApplication:
    """Stands in for ``QGuiApplication`` where asking is an error."""

    @staticmethod
    def instance():
        raise RuntimeError("no QGuiApplication")


class _ApplicationlessApplication:
    """Stands in for ``QGuiApplication`` before one has been created."""

    @staticmethod
    def instance():
        return None


@pytest.mark.parametrize("stand_in",
                         [_NoApplication, _ApplicationlessApplication])
def test_with_no_application_the_answer_is_one(monkeypatch, stand_in):
    """The last resort is the behaviour every call site had before."""
    monkeypatch.setattr(hidpi, "QGuiApplication", stand_in)
    assert device_ratio(None) == 1.0
    assert device_ratio(object()) == 1.0


def test_no_application_still_renders_at_the_old_size(monkeypatch, qapp):
    """A picture built before the app exists is the picture built today."""
    source = QPixmap(400, 400)
    source.fill(Qt.red)
    monkeypatch.setattr(hidpi, "QGuiApplication", _ApplicationlessApplication)
    out = scaled_for(source, None, 60)
    assert (out.width(), out.height()) == (60, 60)
    assert out.devicePixelRatio() == 1.0


def test_a_picture_that_cannot_report_its_own_size_is_measured_by_hand(qapp):
    """``deviceIndependentSize`` is Qt 6.5+; the arithmetic is the fallback."""

    class _OldPicture:
        def isNull(self):
            return False

        def deviceIndependentSize(self):
            raise AttributeError("not on this Qt")

        def devicePixelRatio(self):
            return 2.0

        def width(self):
            return 300

        def height(self):
            return 200

    assert logical_size(_OldPicture()) == QSize(150, 100)


def test_the_watcher_says_which_ratio_it_is_holding(qapp, qtbot):
    label = QLabel()
    qtbot.addWidget(label)
    watcher = follow_device_ratio(label, lambda: None)
    assert watcher.ratio() == device_ratio(label)

    label.devicePixelRatioF = lambda: 2.0
    qapp.sendEvent(label, QEvent(QEvent.Type.DevicePixelRatioChange))
    assert watcher.ratio() == 2.0


def test_something_that_cannot_be_watched_is_simply_not_watched(qapp):
    """No subscription is a soft picture on a second screen, not a crash."""

    class _NotAWidget:
        def devicePixelRatioF(self):
            return 1.0

    assert follow_device_ratio(_NotAWidget(), lambda: None) is None
