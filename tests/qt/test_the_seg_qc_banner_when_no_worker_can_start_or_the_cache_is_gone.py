"""The segmentation-QC banner when no worker can start, and when a cached
read comes back after the digest it vouched for was dropped.

A refresh whose worker thread cannot be built leaves the banner idle -- not
stuck "reading" -- so the next refresh tries again. A read that reports "the
cache still holds" after the banner has let its digest go says nothing: the
banner stays hidden and announces an empty verdict rather than drawing
a verdict it no longer has.
"""
from PySide6.QtWidgets import QLineEdit, QWidget

from spacr.qt import bridge, prerun


class _Model:
    def __init__(self, widgets):
        self._widgets = dict(widgets)
        self._settings = {}

    def collect(self):
        return dict(self._settings)


class _Screen(QWidget):
    def __init__(self, src):
        super().__init__()
        self._thread = None
        field = QLineEdit()
        field.setText(src)
        self._settings_model = _Model({"src": field})


def test_a_refresh_with_no_worker_leaves_the_banner_ready_to_try_again(
        qtbot, tmp_path, monkeypatch):
    screen = _Screen(str(tmp_path))
    qtbot.addWidget(screen)
    banner = prerun.SegQCBanner(screen, threaded=True)
    attempts = []

    def _no_thread(*args, **kwargs):
        attempts.append(kwargs.get("app_key"))
        raise RuntimeError("no thread for you")

    monkeypatch.setattr(bridge, "make_thread", _no_thread)

    banner.refresh()
    assert banner._reading is False
    assert not banner.busy
    banner.refresh()

    assert len(attempts) == 2
    assert not banner.isVisible()


def test_a_cached_answer_for_a_digest_that_was_dropped_says_nothing(
        qtbot, tmp_path):
    screen = _Screen(str(tmp_path))
    qtbot.addWidget(screen)
    banner = prerun.SegQCBanner(screen, threaded=False)
    said = []
    banner.refreshed.connect(said.append)
    banner._digest = None
    banner._reading_gen = banner._refresh_gen

    banner._on_refreshed({"key": ("src", ((str(tmp_path), 1, 1),))})

    assert said == [""]
    assert not banner.isVisible()
