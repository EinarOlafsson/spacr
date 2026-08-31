"""Make Masks' edges, the app screen's dead guards, and the safe-mode launcher.

Round 6's remaining screen arcs split cleanly in two.

**Driven here, because a user reaches them.** A recrop box drawn before the
crop was named, so the mark goes on the field without a caption; a pan so
small it moves the view by less than one image pixel, which must NOT re-anchor
the drag; a settings panel folded away before the splitter has ever been laid
out, which has no width to remember; and a Cellpose ``eval`` whose signature
cannot be read at all, which has to be offered every setting rather than none.

**Proved dead, because nothing can reach them.** The action row's
``getattr(self, card_attr, None) is not None`` and its ``queue is not None and
hasattr(queue, "figure_clicked")``, both of which read collaborators the
screen has already built unconditionally; ``_bulk_apply_changes_form_shape``'s
``except ValueError`` around ``organelle_number``, which cannot fire because
the two guards above it leave only organelle-slot roles; and the four ``if
name in ordered`` tests in ``categories_for_app``'s ``classify_merged``
rebuild, whose five tuples enumerate the group table exactly.

The last two tests are the safe-mode launcher's own: a ``--no-setup`` the
caller already gave is not given twice, and ``python -m spacr.qt.safespacr``
really is a way in.

Each proof pins the guarantee that makes the arm dead, and every "nothing
happened" assertion is paired with the input that makes something happen.
"""
from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
pytest.importorskip("pytestqt")

from PySide6.QtCore import QEvent, QPointF, Qt                   # noqa: E402
from PySide6.QtGui import QColor, QImage, QMouseEvent            # noqa: E402

from spacr.qt.screens import settings_model as SM                # noqa: E402
from spacr.qt.screens.app_screen import AppScreen                # noqa: E402
from spacr.qt.screens.make_masks import (MakeMasksScreen,        # noqa: E402
                                         _MaskCanvas,
                                         cellpose_detect)
from spacr.qt.theme import active_palette                        # noqa: E402
from spacr.qt.widget_cleanup import retire_pyqtgraph_menus       # noqa: E402

pytestmark = pytest.mark.qt

CANVAS_W, CANVAS_H = 600, 400
IMG_N = 64
PIXMAP_N = 400


def _evt(kind, x, y, buttons=Qt.LeftButton, button=Qt.LeftButton,
         modifiers=Qt.NoModifier):
    pos = QPointF(float(x), float(y))
    return QMouseEvent(kind, pos, pos, button, buttons, modifiers)


def _pixels(qimg):
    image = qimg.convertToFormat(QImage.Format_RGB32)
    return np.frombuffer(image.constBits(), dtype=np.uint32).reshape(
        image.height(), image.bytesPerLine() // 4).copy()


@pytest.fixture
def canvas(qtbot, qt_theme_applied):
    """A canvas at a known size over a black field, so marks show up."""
    widget = _MaskCanvas()
    qtbot.addWidget(widget)
    widget.resize(CANVAS_W, CANVAS_H)
    widget.set_image_and_mask(np.zeros((IMG_N, IMG_N), np.uint16),
                              np.zeros((IMG_N, IMG_N), np.uint8))
    assert widget.pixmap().width() == PIXMAP_N, "geometry assumption broke"
    return widget


def _render(widget) -> QImage:
    image = QImage(widget.size(), QImage.Format_RGB32)
    image.fill(QColor("black"))
    widget.render(image)
    return image


# ---------------------------------------------------------------------------
# make_masks — the canvas
# ---------------------------------------------------------------------------

def test_a_recrop_box_with_no_name_is_still_marked_but_not_captioned(canvas):
    """A box is drawn from four numbers; the caption needs a fifth.

    ``_paint_recrop_boxes`` reads ``box[4]`` only when there is one, and
    writes the caption only when it is not empty. Both are real states: the
    screen appends ``(*box, name)`` after the crop is named, and a box put
    there before the name is a bare rectangle.

    The two renders differ only in the caption, so the fill is counted on
    both -- a paint that dropped the whole box would otherwise look exactly
    like one that dropped only the text.
    """
    box = (8, 8, 40, 40)
    accent = QColor(active_palette()["accent"])

    canvas.recrop_boxes = [box]
    bare = _pixels(_render(canvas))
    canvas.recrop_boxes = [(*box, "crop_1")]
    captioned = _pixels(_render(canvas))

    edge_bare = int((bare == np.uint32(accent.rgb())).sum())
    edge_captioned = int((captioned == np.uint32(accent.rgb())).sum())
    assert edge_bare > 0, "the unnamed box was not drawn at all"
    assert edge_bare == edge_captioned, "the rectangle itself must not change"

    # The only difference the fifth member makes is the caption, and it is
    # written inside the box, just below its top edge.
    rows, cols = np.nonzero(bare != captioned)
    assert rows.size > 0, "the caption was not written"
    top_left = canvas._image_to_canvas(8, 8)
    bottom_right = canvas._image_to_canvas(40, 40)
    assert rows.min() >= top_left.y() and rows.max() <= bottom_right.y()
    assert cols.min() >= top_left.x() and cols.max() <= bottom_right.x()


def test_a_drag_too_small_to_move_the_view_keeps_its_anchor(canvas):
    """Sub-pixel pans are discarded, not accumulated.

    The image is shown at 6.25 widget pixels per image pixel, so a three
    pixel drag is worth zero image pixels: ``(dx or dy)`` is false and the
    anchor must stay where the press put it. Re-anchoring on such a drag is
    what made a slow pan at high zoom stall completely -- every move would
    round to nothing from a fresh anchor.
    """
    canvas.mousePressEvent(_evt(QEvent.Type.MouseButtonPress, 300, 200,
                                modifiers=Qt.ShiftModifier))
    assert canvas._pan_from is not None
    anchor = canvas._pan_from

    assert canvas._image_delta(3, 2) == (0, 0)
    canvas.mouseMoveEvent(_evt(QEvent.Type.MouseMove, 303, 202))
    assert canvas._pan_from == anchor, "a sub-pixel drag moved the anchor"

    # A drag worth whole image pixels, on a zoomed view that can take it,
    # does re-anchor -- so the guard above is a switch and not a pan that
    # never works.
    canvas.zoom_at(32, 32, 4.0)
    assert canvas.is_zoomed()
    assert canvas._image_delta(80, 60) != (0, 0)
    canvas.mouseMoveEvent(_evt(QEvent.Type.MouseMove, 380, 260))
    assert canvas._pan_from != anchor
    assert (canvas._pan_from.x(), canvas._pan_from.y()) == (380, 260)


# ---------------------------------------------------------------------------
# make_masks — the screen and the segmenter
# ---------------------------------------------------------------------------

def test_a_settings_fold_never_remembers_a_zero_width(qtbot, qt_theme_applied):
    """The remembered panel width is only ever taken from a real layout.

    ``_on_toggle_settings`` remembers the width the user dragged so the next
    press puts the panel back there. A splitter that has not been laid out
    reports zero for every pane, and remembering that zero is what gave the
    second press a pane the user could not see -- so the width is only taken
    when there is a second pane and it has a size.
    """
    screen = MakeMasksScreen()
    qtbot.addWidget(screen)
    try:
        screen.resize(900, 600)
        laid_out = screen._body_splitter.sizes()
        assert len(laid_out) > 1 and laid_out[1] > 0

        screen._on_toggle_settings(False)
        remembered = screen._settings_width
        assert remembered == laid_out[1]
        assert screen._settings_scroll.isHidden() is True

        screen._on_toggle_settings(True)
        assert screen._settings_scroll.isHidden() is False

        # A SPLITTER THAT HAS NEVER BEEN LAID OUT reports zero for
        # everything -- the state this screen's own comment names. Offscreen
        # nothing is ever mapped, so Qt keeps handing back the size hints
        # and the state cannot be produced through the real splitter;
        # standing one in that answers as an unlaid-out splitter does is the
        # only way to drive the arm. The width already remembered has to
        # survive it.
        class _NeverLaidOut:
            def __init__(self):
                self.given = None

            def sizes(self):
                return [0, 0]

            def setSizes(self, sizes):
                self.given = list(sizes)

        screen._body_splitter = _NeverLaidOut()
        screen._on_toggle_settings(False)
        assert screen._settings_width == remembered, (
            "a zero-width pane was remembered as the panel's width")
        assert screen._settings_scroll.isHidden() is True
    finally:
        retire_pyqtgraph_menus(screen)
        screen.close()


def test_an_eval_whose_signature_cannot_be_read_is_offered_every_setting():
    """The filter only runs against a signature that LISTS what it takes.

    ``inspect.signature`` raises for a callable it cannot introspect -- a
    C-level ``eval``, or one carrying a ``__signature__`` that is not one --
    and the screen's answer is to pass every setting rather than to drop all
    of them, which is what filtering against an empty answer would do.
    """
    class _Unreadable:
        __signature__ = "cellpose did not say"

        def __init__(self):
            self.seen = None

        def __call__(self, batch, **kwargs):
            self.seen = dict(kwargs)
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[2:4, 2:4] = 1
            return ([labels], [[None, None, None]],
                    np.zeros(4, dtype=np.float32))

    class _Listed:
        def __init__(self):
            self.seen = None

        # THE WHOLE INSTALLED SIGNATURE, not a convenient subset: a double
        # that takes **kwargs, or omits a parameter, cannot fail when spaCR
        # passes an argument cellpose 4 removed. Enforced by
        # tests/test_cellpose_api_contract.py.
        def eval(self, batch, batch_size=8, resample=True, channels=None,
                 channel_axis=None, z_axis=None, normalize=True,
                 rescale=None, diameter=None, flow_threshold=0.4,
                 cellprob_threshold=0.0, do_3D=False, anisotropy=None,
                 flow3D_smooth=0, stitch_threshold=0.0, min_size=15,
                 max_size_fraction=0.4, niter=None, augment=False,
                 tile_overlap=0.1, bsize=None, compute_masks=True,
                 progress=None):
            self.seen = dict(batch_size=batch_size, normalize=normalize,
                             channel_axis=channel_axis, diameter=diameter)
            labels = np.zeros((8, 8), dtype=np.uint16)
            labels[2:4, 2:4] = 1
            return ([labels], [[None, None, None]],
                    np.zeros(4, dtype=np.float32))

    opaque = _Unreadable()
    model = type("Model", (), {})()
    model.eval = opaque
    with pytest.raises((TypeError, ValueError)):
        inspect.signature(model.eval)

    labels, _cellprob, _flow = cellpose_detect(
        np.zeros((8, 8), dtype=np.uint16), model,
        flow_threshold=0.7, cellprob_threshold=-1.5, min_size=11)

    assert int(labels.max()) == 1
    assert opaque.seen is not None
    # Nothing was dropped: the thresholds the panel chose all arrived.
    assert opaque.seen["flow_threshold"] == 0.7
    assert opaque.seen["cellprob_threshold"] == -1.5
    assert opaque.seen["min_size"] == 11

    # The same call against a signature that DOES list its parameters keeps
    # only what that signature names -- the behaviour the guard switches
    # between.
    listed = _Listed()
    cellpose_detect(np.zeros((8, 8), dtype=np.uint16), listed,
                    flow_threshold=0.7, cellprob_threshold=-1.5, min_size=11)
    assert set(listed.seen) == {"batch_size", "normalize", "channel_axis",
                                "diameter"}


# ---------------------------------------------------------------------------
# app_screen — guards nothing can trip
# ---------------------------------------------------------------------------

@pytest.fixture
def make_screen(qtbot):
    made = []

    def build(app_key):
        widget = AppScreen(app_key)
        qtbot.addWidget(widget)
        made.append(widget)
        return widget

    yield build
    for widget in made:
        retire_pyqtgraph_menus(widget)
        widget.close()


def test_every_app_with_a_live_switch_has_already_built_its_preview_card(
        make_screen):
    """Why ``getattr(self, card_attr, None) is not None`` cannot be false.

    The action row is built after the body, and the body builds each of the
    four preview cards unconditionally inside the branch for its own app
    key -- ``mask``, ``timelapse``, ``motility`` and ``measure`` each get
    theirs with no guard around the call. So the only apps that reach the
    test are the four that have just built the card it names.
    """
    cards = {"mask": "_live_preview_card",
             "timelapse": "_timelapse_preview_card",
             "motility": "_motility_preview_card",
             "measure": "_measure_preview_card"}
    for app_key, attr in cards.items():
        screen = make_screen(app_key)
        assert getattr(screen, attr, None) is not None, (
            f"{app_key} reached the action row with no {attr}")
        assert screen._preview_card_attr == attr
        assert screen._preview_switch is not None

    # An app that is NOT in the table gets no Live switch at all, which is
    # the other side of the outer guard and the reason the inner one looks
    # optional.
    other = make_screen("regression")
    assert getattr(other, "_preview_card_attr", None) is None
    assert getattr(other, "_preview_switch", None) is None


def test_the_umap_action_row_always_finds_a_figure_queue_to_listen_to(
        make_screen):
    """Why ``queue is not None and hasattr(queue, "figure_clicked")`` cannot
    be false.

    ``_figure_queue`` is assigned a ``FigureQueue`` with no guard, well
    before the action row is built, and ``FigureQueue`` declares
    ``figure_clicked`` as a class-level signal -- so every screen that
    reaches this line has both.
    """
    screen = make_screen("umap")
    assert screen._interactive_switch is not None, (
        "this test needs the arm that carries the guard")
    queue = screen._figure_queue
    assert queue is not None and hasattr(queue, "figure_clicked")
    assert hasattr(type(queue), "figure_clicked")

    # Any other screen has one too, so there is no app for which the guard
    # could be the thing that saves the connect.
    assert make_screen("regression")._figure_queue is not None


def test_a_key_that_reaches_the_slot_number_is_always_a_slot(make_screen):
    """Why ``except ValueError`` around ``organelle_number`` cannot fire.

    ``organelle_number`` raises for anything that is not ``organelle`` or
    ``organelle<letter>``. The role handed to it is ``object_of_setting``'s
    answer, which is either ``None``, one of ``CHANNELLED_OBJECTS`` or a role
    ``organelle_role_of`` produced -- and the two guards above the ``try``
    have already dropped ``None``, ``cell``, ``nucleus`` and ``pathogen``.
    Only slot roles are left, and every one of those has a number.
    """
    from spacr.organelle_types import (NUMBER_OF_ORGANELLES, organelle_number,
                                       organelle_role_of)
    from spacr.qt.screens.settings_model import (CHANNELLED_OBJECTS,
                                                 object_switch_keys,
                                                 object_of_setting)

    screen = make_screen("measure")
    checked = 0
    for role in ("organelle", "organelleb", "organellec", "organelled"):
        for key in object_switch_keys(role):
            answer = object_of_setting(key)
            assert answer not in (None, *CHANNELLED_OBJECTS)
            assert organelle_role_of(key) == answer
            assert isinstance(organelle_number(answer), int)
            checked += 1
    assert checked > 3

    # And the try really is entered: a slot beyond the requested count is
    # stepped over, while one inside it reports the change.
    current = {NUMBER_OF_ORGANELLES: 1, "organelleb_channel": None,
               "organelle_channel": None}
    assert screen._bulk_apply_changes_form_shape(
        {"organelleb_channel": 2}, current) is False
    assert screen._bulk_apply_changes_form_shape(
        {"organelle_channel": 2}, current) is True


# ---------------------------------------------------------------------------
# settings_model — the merged classifier rebuild
# ---------------------------------------------------------------------------

def test_the_merged_classifier_rebuild_finds_every_group_it_names():
    """Why the four ``if name in ordered`` tests cannot be false.

    ``classify_merged`` rebuilds its panel from five tuples of group names,
    and each of the four loops is guarded on the group being present. The
    group table those names are looked up in is the literal a hundred lines
    above plus this branch's own two additions, and the tuples enumerate it
    exactly -- so every lookup succeeds and every guard is true.

    Pinned on the rebuild's own output: each name arrives, in its declared
    order, with the family prefix where the tuple carries one.
    """
    ordered = SM.categories_for_app("classify_merged", SM.get_categories())
    names = list(ordered)

    expected = ["Classifier",
                "Plate Sources & Workflow", "Labels & Classes",
                SM._family_heading("Computer Vision", "Images & Cropping"),
                SM._family_heading("Computer Vision", "Model & Regularization"),
                SM._family_heading("Computer Vision", "Training & Loss"),
                SM._family_heading("Machine Learning", "Model & Features"),
                SM._family_heading("Machine Learning",
                                   "Plate & Batch Correction"),
                "Evaluation & Results"]
    assert names[:len(expected)] == expected, (
        "a group the rebuild names was not in the table it reads")

    # Every one of those headings carries settings, so a guard that had gone
    # false would have cost a whole tab rather than an empty one.
    for name in expected[1:]:
        assert ordered[name], f"{name} arrived empty"

    # The plain table -- what the guards read -- really does hold each name
    # under its unprefixed form.
    plain = SM.categories_for_app("classify", SM.get_categories())
    for name in ("Plate Sources & Workflow", "Labels & Classes",
                 "Evaluation & Results"):
        assert name in plain


# ---------------------------------------------------------------------------
# safespacr — the way in when a saved preference is what breaks the launch
# ---------------------------------------------------------------------------

@pytest.fixture
def launcher(monkeypatch):
    """``safespacr`` with its two collaborators stood in for.

    ``enable_safe_mode`` latches a module global for the life of the process
    and ``run`` opens a window; neither belongs in a test of the argument
    the launcher passes.

    ``SPACR_NO_GL`` is saved and restored by hand rather than through
    ``monkeypatch``: the launcher SETS it, and ``monkeypatch.delenv(...,
    raising=False)`` records nothing for a variable that was not there to
    begin with, so the flag would outlive the test and turn off OpenGL for
    everything after it.
    """
    from spacr.qt import preferences, safespacr
    import spacr.qt as qt_pkg

    seen = {}

    def _fake_run(argv):
        seen["argv"] = list(argv or [])
        return 3

    monkeypatch.setattr(qt_pkg, "run", _fake_run)
    monkeypatch.setattr(preferences, "_SAFE_MODE", False)

    before = {name: os.environ.get(name)
              for name in ("SPACR_NO_GL", "SPACR_TIMING")}
    os.environ["SPACR_TIMING"] = "1"
    os.environ.pop("SPACR_NO_GL", None)
    try:
        yield safespacr, seen
    finally:
        for name, value in before.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_safe_mode_does_not_repeat_a_no_setup_the_caller_already_gave(
        launcher):
    """The flag is added because safe mode needs it, not because it is missing.

    Reading preferences as defaults makes "has this profile been set up"
    read as "no", so the launcher forces ``--no-setup`` -- but a caller who
    already passed it must not get it twice, or the argument list the app
    parses stops matching the one the user wrote.
    """
    safespacr, seen = launcher

    assert safespacr.main(["--no-setup", "--srv"]) == 3
    assert seen["argv"] == ["--no-setup", "--srv"]

    assert safespacr.main(["--srv"]) == 3
    assert seen["argv"] == ["--no-setup", "--srv"]


def test_the_safe_mode_launcher_is_runnable_as_a_module(launcher,
                                                       monkeypatch):
    """``python -m spacr.qt.safespacr`` is the documented way in.

    The ``__main__`` guard is the whole of that entry point: it hands the
    application's exit code to ``sys.exit``, so a failed launch is a failed
    process rather than a silent zero.
    """
    import runpy
    import sys

    _safespacr, seen = launcher
    # A script run takes its arguments from sys.argv, which is the other
    # half of what the guard reaches.
    monkeypatch.setattr(sys, "argv", ["spacr-safe", "--srv"])

    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("spacr.qt.safespacr", run_name="__main__")

    assert exit_info.value.code == 3, "the app's exit code was not passed on"
    assert seen["argv"] == ["--no-setup", "--srv"]
    assert os.environ.get("SPACR_NO_GL") == "1"
    assert os.environ.get("SPACR_TIMING") is None
