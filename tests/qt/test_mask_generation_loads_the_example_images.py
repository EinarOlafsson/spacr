"""Mask Generation offers its example data the way Regression does.

Regression's button fills two TABLE slots from the packaged example screen.
Mask Generation needs IMAGES, which is a different set from a different
place -- `einarolafsson/toxo_mito` on Hugging Face, which spaCR already
downloads for its end-to-end demo. So it is a separate button of the same
shape rather than the same one widened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


@pytest.fixture
def mask(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


@pytest.fixture
def empty_example_cache(mask, tmp_path, monkeypatch):
    """Keep fake downloads independent of a real user-level cached plate."""
    cache = tmp_path / "example_images"
    monkeypatch.setattr(mask, "example_images_destination", lambda: cache)
    return cache


def _fake_download(images, settings, seen=None):
    class Result:
        dataset_path = images
        settings_path = settings

    def ask(parent, destination, done):
        if seen is not None:
            seen["destination"] = destination
        done(Result(), "")

    return ask


def test_mask_generation_has_the_button(mask):
    button = getattr(mask, "_example_images_button", None)
    assert button is not None, "Mask Generation offers no example data"
    assert "example" in button.text().lower()


def test_it_is_in_the_section_that_holds_src():
    """A button for filling `src` belongs where `src` is."""
    from spacr.qt.screens.app_screen import EXAMPLE_DATA_SECTIONS
    from spacr.qt.screens.settings_model import _APP_CATEGORY_SPECS

    section = EXAMPLE_DATA_SECTIONS["mask"]
    holds_src = [name for name, keys in _APP_CATEGORY_SPECS["mask"]
                 if "src" in keys]
    assert holds_src == [section], (section, holds_src)


def test_regression_keeps_its_own_and_does_not_gain_this_one(qtbot):
    from spacr.qt.screens.app_screen import AppScreen

    screen = AppScreen("regression")
    qtbot.addWidget(screen)
    assert getattr(screen, "_example_data_button", None) is not None
    assert getattr(screen, "_example_images_button", None) is None


def test_pressing_it_fills_src(mask, empty_example_cache, tmp_path):
    images = tmp_path / "plate1"
    images.mkdir()
    (images / "a.tif").write_bytes(b"")
    settings = tmp_path / "settings"
    settings.mkdir()

    placed = mask.load_the_example_images(
        ask=_fake_download(images, settings))
    assert placed["src"] == str(images)
    assert mask._settings_model._widgets["src"].text() == str(images)


def test_it_says_where_everything_went(mask, empty_example_cache, tmp_path):
    """Filling a field silently is indistinguishable from a button that did
    nothing -- and this one may have just moved 400 MB."""
    images = tmp_path / "plate1"
    images.mkdir()
    (images / "a.tif").write_bytes(b"")
    settings = tmp_path / "settings"
    settings.mkdir()

    said = []
    mask._console.append_stdout = lambda text: said.append(text)
    mask.load_the_example_images(ask=_fake_download(images, settings))
    joined = "".join(said)
    assert str(images) in joined
    assert str(settings) in joined


def test_a_failed_download_says_so_and_does_not_fill_src(
        mask, empty_example_cache):
    def ask(parent, destination, done):
        done(None, "the network is down")

    # UNCHANGED, not empty: mask's `src` opens with its own placeholder,
    # and the point is that a failed download does not overwrite whatever
    # the user already had there.
    control = mask._settings_model._widgets["src"]
    before = control.text()

    said = []
    mask._console.append_stdout = lambda text: said.append(text)
    placed = mask.load_the_example_images(ask=ask)
    assert placed == {}
    assert "the network is down" in "".join(said)
    assert control.text() == before


def test_a_plate_already_cached_is_not_downloaded_again(mask, tmp_path,
                                                       monkeypatch):
    """400 MB once. The second press must be free."""
    cache = tmp_path / "example_images"
    plate = cache / "plate1"
    plate.mkdir(parents=True)
    (plate / "a.tif").write_bytes(b"")
    monkeypatch.setattr(mask, "example_images_destination", lambda: cache)

    called = []

    def ask(parent, destination, done):          # must not run
        called.append(destination)

    placed = mask.load_the_example_images(ask=ask)
    assert called == [], "it downloaded a plate it already had"
    assert placed["src"] == str(plate)


def test_the_download_goes_beside_the_other_example_data(mask):
    """One cache folder, not two."""
    destination = mask.example_images_destination()
    assert ".cache" in str(destination) and "spacr" in str(destination)


# ---------------------------------------------------------------------------
# the paths a real screen cannot easily be driven into
# ---------------------------------------------------------------------------

class _StubScreen:
    """The three things ``load_the_example_images`` asks a screen for.

    A real ``AppScreen`` is used above for the ordinary path. These cases need
    the plate generator to FAIL, or the form to be missing controls, and
    forcing a real screen into those states would mostly be testing the
    forcing.
    """

    def __init__(self, widgets=None, console=None):
        self._settings_model = type("_M", (), {"_widgets": widgets or {}})()
        self._console = console
        self.applied = []

    def apply_settings_dict(self, settings):
        self.applied.append(dict(settings))
        return len([k for k in settings
                    if k in self._settings_model._widgets])


class _Console:
    def __init__(self, explode=False):
        self.text = []
        self.explode = explode

    def append_stdout(self, text):
        if self.explode:
            raise RuntimeError("the console widget has been deleted")
        self.text.append(text)


def test_a_plate_that_cannot_be_written_says_where_and_why(tmp_path,
                                                           monkeypatch):
    """The generator raises on a full disk or a read-only folder.

    Returning an empty mapping is what the caller checks, but the message is
    what the user acts on -- it has to name the folder, because the default is
    a cache path they have never seen and would otherwise go looking for the
    example plate in their project.
    """
    from spacr.qt.screens import mask as mask_module
    from spacr.qt import synthetic

    def refuse(dst):
        raise OSError("read-only file system")

    monkeypatch.setattr(synthetic, "generate_mask_demo", refuse)
    console = _Console()
    screen = _StubScreen(console=console)

    result = mask_module.load_the_example_images(screen, folder=str(tmp_path))

    assert result == {}
    said = "".join(console.text)
    assert str(tmp_path) in said
    assert "read-only file system" in said


def test_an_unreadable_destination_does_not_stop_the_plate_being_written(
        tmp_path, monkeypatch):
    """The "already there" count is a nicety; the plate is the point.

    ``os.listdir`` on the destination is only used to say how many files were
    written now versus already present. A folder it cannot read -- a mount
    that went away, a permissions change -- must not cost the user their
    example plate over a sentence in a summary.
    """
    from spacr.qt.screens import mask as mask_module
    from spacr.qt import synthetic

    monkeypatch.setattr(mask_module.os, "listdir",
                        lambda _p: (_ for _ in ()).throw(OSError("gone")))
    monkeypatch.setattr(mask_module.os.path, "isdir", lambda _p: True)

    layout = type("_L", (), {
        "image_files": [str(tmp_path / "a.tif"), str(tmp_path / "b.tif")],
        "src": str(tmp_path),
        "notes": {"channels": (0, 1), "n_fields": 2},
    })()
    monkeypatch.setattr(synthetic, "generate_mask_demo", lambda dst: layout)
    monkeypatch.setattr(synthetic, "demo_settings",
                        lambda app, src: {"src": src})

    console = _Console()
    screen = _StubScreen(widgets={"src": object()}, console=console)

    result = mask_module.load_the_example_images(screen, folder=str(tmp_path))

    assert result["images"] == layout.image_files
    # Nothing could be read beforehand, so everything counts as written now.
    assert len(result["written"]) == 2
    assert result["filled"] == ["src"]


def test_settings_with_no_control_on_the_form_are_named_not_dropped(
        tmp_path, monkeypatch):
    """A silently unapplied setting is the failure this sentence prevents.

    The example plate sets values the Mask form may not expose. Applying what
    it can and saying nothing about the rest leaves the user with a form that
    does not match the plate and no way to find out which parts.
    """
    from spacr.qt.screens import mask as mask_module
    from spacr.qt import synthetic

    layout = type("_L", (), {"image_files": [], "src": str(tmp_path),
                             "notes": {}})()
    monkeypatch.setattr(synthetic, "generate_mask_demo", lambda dst: layout)
    monkeypatch.setattr(synthetic, "demo_settings",
                        lambda app, src: {"src": src, "nucleus_channel": 1})

    console = _Console()
    screen = _StubScreen(widgets={"src": object()}, console=console)

    result = mask_module.load_the_example_images(screen, folder=str(tmp_path))

    assert result["unplaced"] == ["nucleus_channel"]
    said = "".join(console.text)
    assert "no control on this form" in said
    assert "nucleus_channel" in said


# ---------------------------------------------------------------------------
# _say
# ---------------------------------------------------------------------------

def test_a_screen_with_no_console_is_not_an_error():
    """The function is called from a button that may be on a bare screen."""
    from spacr.qt.screens.mask import _say

    _say(_StubScreen(console=None), "anything")


def test_a_console_without_the_method_is_left_alone():
    """Duck typing, checked rather than assumed.

    ``_console`` is whatever the screen put there, and an object that is not a
    console must not be called at.
    """
    from spacr.qt.screens.mask import _say

    _say(_StubScreen(console=object()), "anything")


def test_a_console_that_raises_does_not_lose_the_plate():
    """The plate is already written by the time anything is said about it.

    A deleted console widget raising from append_stdout must not turn a
    successful generation into a traceback.
    """
    from spacr.qt.screens.mask import _say

    _say(_StubScreen(console=_Console(explode=True)), "anything")


def test_a_working_console_really_is_written_to():
    """Otherwise the three tolerant cases above prove nothing."""
    from spacr.qt.screens.mask import _say

    console = _Console()
    _say(_StubScreen(console=console), "hello")

    assert console.text == ["hello"]
