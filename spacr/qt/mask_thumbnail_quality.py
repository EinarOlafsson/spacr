"""Persistent display quality for Make Masks' image-organizing thumbnails.

Low retains the original 64-pixel sampling. Medium and High sample the
original image and mask at 256 and 1024 pixels respectively, never enlarging
the source. Logical cell size remains the separate Size preference.
"""
from PySide6.QtCore import QObject, Signal

QUALITY_SIZES = {"low": 64, "medium": 256, "high": 1024}
QUALITY_KEY = "make_masks/thumbnail_quality"


class _Changes(QObject):
    """Notify open thumbnail views when their shared display setting changes."""

    changed = Signal(str)


changes = _Changes()


def get_quality() -> str:
    """Return the persisted quality, with Low for old or invalid settings."""
    from .prefs import _s

    value = str(_s().value(QUALITY_KEY, "low")).lower()
    return value if value in QUALITY_SIZES else "low"


def set_quality(value: str) -> None:
    """Persist and broadcast a display-only quality choice.

    :param value: low, medium or high; an invalid value resets to low.
    """
    from .prefs import _s

    value = value if value in QUALITY_SIZES else "low"
    previous = get_quality()
    _s().setValue(QUALITY_KEY, value)
    if value != previous:
        changes.changed.emit(value)


def quality_combo(parent=None):
    """Build a live synchronized quality selector.

    :param parent: owning widget, if any.
    :returns: a combo whose item data uses stable untranslated quality keys.

    The slot is connected with a context QObject, so it disconnects
    automatically when that object is destroyed.
    """
    from PySide6.QtWidgets import QComboBox
    from .i18n import tr

    combo = QComboBox(parent)
    for key, label in (("low", "Low"), ("medium", "Medium"), ("high", "High")):
        combo.addItem(tr(label), key)
    combo.setCurrentIndex(combo.findData(get_quality()))
    combo.setToolTip(tr(
        "Thumbnail source resolution: Low 64 px, Medium 256 px, High 1024 px, "
        "limited by the original image. Cell size and saved images and masks "
        "are unchanged."))
    combo.currentIndexChanged.connect(lambda _index: set_quality(combo.currentData()))
    class _Sync(QObject):
        """Keeps one quality selector in step with changes made elsewhere."""
        def update(self, value):
            """Synchronize the selector after another view changed quality.

            :param value: the stable quality key.
            """
            combo.setCurrentIndex(combo.findData(value))
    sync = _Sync(combo)
    changes.changed.connect(sync.update)
    combo._quality_sync = sync
    return combo
