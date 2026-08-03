"""Shared source-selector controls for every live-preview panel.

Four modules ship a live preview — Mask, Measure, Timelapse and Motility —
and each of them opens exactly one source at a time. Before this module they
each had a single ``Choose …`` button and no way to move to the next field of
view or to look at a different channel without re-opening the file dialog.

This module supplies the two dropdowns those panels now share, plus the flat
"text only" look they wear:

* :class:`FlatComboBox` / :class:`FlatButton` — chrome-free controls that read
  like the **Live** toggle (:class:`~spacr.qt.widgets.ai_toggle_label.AiToggleLabel`):
  the theme's foreground colour, 600 weight, body font size, transparent
  background, no border, pointing-hand cursor. The accent colour on hover is
  the only affordance, exactly like the toggles they sit beside.
* :func:`populate_channel_combo` — fills a channel dropdown from a channel
  count, with an "All channels" entry first.
* :func:`sibling_sources` — lists the other fields of view that live beside
  the currently-loaded one.

The palette is resolved through :func:`spacr.qt.theme.active_palette` at build
time *and* again on every ``showEvent``, so a theme switch made in Preferences
lands the next time the panel is shown rather than baking the dark palette's
white text onto a light page. That is the failure
:mod:`spacr.qt.widgets.ai_toggle_label` documents.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QPushButton

from ..theme import FONT_SIZE, active_palette

#: Object name every flat preview control carries, so a stylesheet (or a test)
#: can find them without knowing which panel built them.
FLAT_CONTROL_NAME = "FlatPreviewControl"

#: Entry that means "do not single out a channel" in a channel dropdown.
ALL_CHANNELS = "All channels"


def _flat_qss(selector: str) -> str:
    """Return the chrome-free QSS block for ``selector``.

    Mirrors ``AiToggleLabel._refresh_style``: theme foreground, body size,
    600 weight, 4x10 padding, transparent background — and, for the combo,
    a drop-down button stripped of its own frame so only the theme's little
    triangle remains.
    """
    palette = active_palette()
    return (
        f"{selector}#{FLAT_CONTROL_NAME} {{"
        f"  color: {palette['fg']};"
        f"  font-size: {FONT_SIZE['body']}px;"
        f"  font-weight: 600;"
        f"  padding: 4px 10px;"
        f"  background: transparent;"
        f"  border: none;"
        f"  border-radius: 0px;"
        f"}}"
        f"{selector}#{FLAT_CONTROL_NAME}:hover {{"
        f"  color: {palette['button_accent']};"
        f"}}"
        f"{selector}#{FLAT_CONTROL_NAME}:focus {{"
        f"  border: none;"
        f"  outline: none;"
        f"}}"
        f"{selector}#{FLAT_CONTROL_NAME}:disabled {{"
        f"  color: {palette['fg_dim']};"
        f"  background: transparent;"
        f"  border: none;"
        f"}}"
        f"{selector}#{FLAT_CONTROL_NAME}::drop-down {{"
        f"  border: none;"
        f"  background: transparent;"
        f"}}"
    )


class _FlatStyleMixin:
    """Applies (and re-applies) the Live-toggle look to a widget."""

    _flat_selector = "QWidget"

    def _apply_flat_style(self) -> None:
        self.setStyleSheet(_flat_qss(self._flat_selector))

    def showEvent(self, event):      # noqa: N802 (Qt naming)
        # Preferences can change the theme while this panel is hidden; the
        # widget stylesheet keeps whatever palette it was born with until it
        # is rebuilt, so rebuild it every time the panel comes back.
        self._apply_flat_style()
        super().showEvent(event)


class FlatComboBox(_FlatStyleMixin, QComboBox):
    """Text-only dropdown styled like the **Live** toggle.

    :param parent: owning widget.
    :param tooltip: hover help; these controls carry no visible label, so the
        tooltip is the only place their meaning is written down.
    """

    _flat_selector = "QComboBox"

    def __init__(self, parent=None, tooltip: str = ""):
        super().__init__(parent)
        self.setObjectName(FLAT_CONTROL_NAME)
        self.setCursor(Qt.PointingHandCursor)
        # The entries are *data* (file names, channel indices), not prose.
        # Letting the language pass rewrite them would break every lookup
        # that reads ``currentText()`` back — the same trap that silently
        # reverted the live preview's outline colour to its default.
        self.setProperty("i18nSkipItems", True)
        if tooltip:
            self.setToolTip(tooltip)
        self._apply_flat_style()


class FlatButton(_FlatStyleMixin, QPushButton):
    """Text-only push button styled like the **Live** toggle."""

    _flat_selector = "QPushButton"

    def __init__(self, text: str = "", parent=None, tooltip: str = ""):
        super().__init__(text, parent)
        self.setObjectName(FLAT_CONTROL_NAME)
        self.setCursor(Qt.PointingHandCursor)
        self.setFlat(True)
        if tooltip:
            self.setToolTip(tooltip)
        self._apply_flat_style()


def channel_labels(n_channels: int, include_all: bool = True) -> List[str]:
    """Return the entries a channel dropdown shows for ``n_channels``."""
    labels = [ALL_CHANNELS] if include_all else []
    labels += [f"Ch {i}" for i in range(max(0, int(n_channels)))]
    return labels


def populate_channel_combo(combo: QComboBox, n_channels: int,
                           include_all: bool = True,
                           keep: Optional[str] = None) -> None:
    """Refill ``combo`` with ``n_channels`` entries, preserving the selection.

    :param combo: the dropdown to refill.
    :param n_channels: how many channels the loaded source holds.
    :param include_all: prepend the :data:`ALL_CHANNELS` entry.
    :param keep: entry to re-select; defaults to what is selected now.
    """
    wanted = combo.currentText() if keep is None else keep
    labels = channel_labels(n_channels, include_all=include_all)
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems(labels)
        index = combo.findText(wanted)
        combo.setCurrentIndex(index if index >= 0 else 0)
    finally:
        combo.blockSignals(blocked)


def selected_channel(combo: QComboBox) -> Optional[int]:
    """Return the channel index a channel dropdown selects, or ``None``.

    ``None`` means :data:`ALL_CHANNELS` (or an empty dropdown) — show the
    source exactly as it is stored.
    """
    text = combo.currentText().strip()
    if not text or text == ALL_CHANNELS:
        return None
    if text.lower().startswith("ch"):
        digits = text[2:].strip()
        if digits.isdigit():
            return int(digits)
    return None


def channel_view(image, channel: Optional[int]):
    """Return ``image`` reduced to ``channel``, or unchanged.

    Out-of-range indices and 2-D images fall through untouched — a stale
    selection must never raise while the user is loading a new field.
    """
    if image is None or channel is None:
        return image
    try:
        if getattr(image, "ndim", 0) != 3:
            return image
        if 0 <= int(channel) < image.shape[2]:
            return image[..., int(channel)]
    except (TypeError, ValueError, IndexError):
        return image
    return image


def sibling_sources(path, suffixes: Sequence[str],
                    directories: bool = False) -> List[Path]:
    """List every comparable source sitting beside ``path``.

    :param path: the currently-loaded file (or folder).
    :param suffixes: lower-case suffixes that count as a source.
    :param directories: when True, list sibling *folders* instead of files —
        the Timelapse preview's fields of view are folders of frames.
    :returns: sorted paths, always including ``path`` itself when it exists.
    """
    if not path:
        return []
    target = Path(os.fspath(path))
    parent = target.parent
    try:
        entries: Iterable[Path] = sorted(parent.iterdir())
    except (OSError, ValueError):
        return [target] if target.exists() else []
    out: List[Path] = []
    for entry in entries:
        if directories:
            if entry.is_dir():
                out.append(entry)
        elif entry.is_file() and entry.suffix.lower() in suffixes:
            out.append(entry)
    if target.exists() and target not in out:
        out.append(target)
        out.sort()
    return out


def populate_fov_combo(combo: QComboBox, sources: Sequence[Path],
                       current=None) -> None:
    """Refill an FOV dropdown with ``sources``, selecting ``current``.

    Each entry stores its full path as item data, so the caller never has to
    reconstruct a path from the (deliberately short) visible label.
    """
    current_text = str(current) if current is not None else ""
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        for source in sources:
            combo.addItem(source.name, str(source))
        index = combo.findData(current_text)
        if index >= 0:
            combo.setCurrentIndex(index)
    finally:
        combo.blockSignals(blocked)
