"""Shared source-selector controls for every live-preview panel.

Four modules ship a live preview — Mask, Measure, Timelapse and Motility —
and each of them opens exactly one source at a time. Before this module they
each had a single ``Choose …`` button and no way to move to the next field of
view or to look at a different channel without re-opening the file dialog.

This module supplies the two dropdowns those panels now share, plus the flat
"text only" look they wear:

* :class:`FlatComboBox` / :class:`FlatButton` / :class:`FlatSpinBox` —
  chrome-free controls that read like the **Live** toggle
  (:class:`~spacr.qt.widgets.ai_toggle_label.AiToggleLabel`):
  the theme's foreground colour, 600 weight, body font size, transparent
  background, no border, pointing-hand cursor. The accent colour on hover is
  the only affordance, exactly like the toggles they sit beside.
* :func:`populate_channel_combo` — fills a channel dropdown from a channel
  count, with an "All channels" entry first.
* :func:`sibling_sources` — lists the other fields of view that live beside
  the currently-loaded one.
* :func:`enumerate_image_sets` / :func:`sample_image_sets` — the *sampled*
  source list, described below.

The palette is resolved through :func:`spacr.qt.theme.active_palette` at build
time *and* again on every ``showEvent``, so a theme switch made in Preferences
lands the next time the panel is shown rather than baking the dark palette's
white text onto a light page. That is the failure
:mod:`spacr.qt.widgets.ai_toggle_label` documents.

Why the source list is a *sample*
---------------------------------
:func:`sibling_sources` lists **every** comparable file in the folder, and the
panels fed that straight into their field-of-view dropdown. Measured on a
384-well plate at 16 fields and 4 channels (24 576 files) and on one four times
larger (98 304 files):

==============================  ==========  =========  ==========  =========
measurement                     24k before  24k after  98k before  98k after
==============================  ==========  =========  ==========  =========
folder dropped → panel usable      279 ms     139 ms     1280 ms     579 ms
opening the sets dropdown           175 ms       2 ms      689 ms       2 ms
every change of field               270 ms       1 ms     1233 ms       3 ms
entries in the dropdown             24 576         20      98 304         20
resident memory it held            16.2 MB     3.1 MB     56.0 MB    39.4 MB
image files opened                       1          1           1          1
==============================  ==========  =========  ==========  =========

The third row is the one users feel: the panels rebuild their selectors on
every load, so re-listing and re-populating the whole plate was paid again on
each step through it. It was also the wrong list — four of those entries are
the same field of view, once per channel.

:func:`enumerate_image_sets` replaces it. It reads **file names only**, via
``os.scandir`` and the project's own acquisition regex
(``spacr.utils._get_regex``), and groups them into :class:`ImageSet` records
keyed by ``(plate, well, field)`` with the channel files hanging off each. No
image is opened, decoded or stacked to build that list.
:func:`sample_image_sets` then draws a bounded random sample — 20 sets by
default, adjustable from the :class:`FlatSpinBox` that sits immediately left of
the sets dropdown.

The sample is **random but reproducible**: the seed is a stable digest of
``"<folder name>|<total sets>|<cap>|<nonce>"`` (:func:`sample_seed`), so the
same plate at the same cap yields the same sets in this session, the next
session, and on another machine or mount point — a preview can be described and
returned to. It is deliberately *not* seeded from the clock or from ``hash()``,
which is salted per process. Re-rendering never re-draws; only an explicit act
does — changing the cap (which is in the seed) or calling
:func:`~ImageSetSampler.reshuffle` (which bumps the nonce).

The sample is drawn across the whole enumeration and then re-sorted into plate
order, so the dropdown still reads A01 → P24 while its *membership* spans the
plate rather than being the first N names alphabetically (which on a
plate-ordered folder means "all of row A").
"""
from __future__ import annotations

import ast
import contextlib
import hashlib
import importlib.util
import io
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field as _field
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QComboBox, QPushButton, QSpinBox

from ..theme import active_palette, font_px

LOG = logging.getLogger("spacr.qt.preview_controls")

#: Object name every flat preview control carries, so a stylesheet (or a test)
#: can find them without knowing which panel built them.
FLAT_CONTROL_NAME = "FlatPreviewControl"

#: Entry that means "do not single out a channel" in a channel dropdown.
ALL_CHANNELS = "All channels"

#: How many image sets a preview loads when the user has not said otherwise.
#: Twenty fields is enough to judge a segmentation setting and small enough
#: that the dropdown, and the memory behind it, cost nothing.
DEFAULT_MAX_SETS = 20

#: Hover help for the "how many image sets" box that sits immediately left of
#: every preview's sets dropdown. Shared, so all four panels say it once.
MAX_SETS_TOOLTIP = (
    "Maximum image sets the preview loads.\n\n"
    "A large experiment is never listed whole — its file names are grouped "
    "into image sets (one field of view, all its channels) and this many are "
    "drawn at random from across the plate. Nothing outside the sample is "
    "opened.\n\n"
    "The draw is reproducible: the same folder at the same maximum always "
    "gives the same sets. Changing this number draws a new sample; simply "
    "re-rendering never does.")

#: Acquisition-naming dialect handed to :func:`spacr.utils._get_regex`.
#: ``cellvoyager`` is the Yokogawa layout this project converts everything
#: into (:mod:`spacr.convert`), so it is what a preview folder normally holds.
DEFAULT_METADATA_TYPE = "cellvoyager"


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
        f"  font-size: {font_px('body')}px;"
        f"  font-weight: 600;"
        f"  padding: {max(2, round(font_px('body') * 4 / 13))}px"
        f" {max(4, round(font_px('body') * 10 / 13))}px;"
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
        # QSpinBox draws two framed arrow buttons of its own. Left alone they
        # are the only chrome in a row that is otherwise pure text, so strip
        # them the same way the combo's drop-down is stripped.
        f"{selector}#{FLAT_CONTROL_NAME}::up-button,"
        f"{selector}#{FLAT_CONTROL_NAME}::down-button {{"
        f"  border: none;"
        f"  background: transparent;"
        f"  width: 12px;"
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


class FlatSpinBox(_FlatStyleMixin, QSpinBox):
    """Text-only integer box styled like the **Live** toggle.

    Used for the "how many image sets may the preview load" cap that sits
    immediately left of the sets dropdown. The count of sets actually found is
    carried in the box's *suffix*, so the control states ``20 of 24576 sets``
    in one place and a sampled preview can never be mistaken for the whole
    plate.
    """

    _flat_selector = "QSpinBox"

    def __init__(self, parent=None, tooltip: str = "", value: int = 20):
        super().__init__(parent)
        self.setObjectName(FLAT_CONTROL_NAME)
        self.setCursor(Qt.PointingHandCursor)
        self.setButtonSymbols(QSpinBox.UpDownArrows)
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setMinimum(1)
        # Wide open until a folder is enumerated; configure_max_sets_box then
        # clamps it to the number of sets that actually exist.
        self.setMaximum(10_000_000)
        self.setValue(int(value))
        self.setAccelerated(True)
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
                       current=None, labels: Optional[Sequence[str]] = None
                       ) -> None:
    """Refill an FOV dropdown with ``sources``, selecting ``current``.

    Each entry stores its full path as item data, so the caller never has to
    reconstruct a path from the (deliberately short) visible label.

    :param labels: visible text per entry; defaults to each path's file name.
        Set-based enumeration passes ``A01 f003`` style labels so the entry
        names the *field of view* rather than one of its channel files.
    """
    current_text = str(current) if current is not None else ""
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        for index, source in enumerate(sources):
            if labels is not None and index < len(labels):
                text = labels[index]
            else:
                text = source.name
            combo.addItem(text, str(source))
        index = combo.findData(current_text)
        if index >= 0:
            combo.setCurrentIndex(index)
    finally:
        combo.blockSignals(blocked)


# ---------------------------------------------------------------------------
# Enumerating image sets without loading them
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageSet:
    """One field of view: every channel file that shares a (plate, well, field).

    Built from **file names alone**. Nothing here has been opened or decoded —
    :attr:`channels` maps a channel ID to a file name, and it is up to the
    panel to decide which single file it wants to read.
    """

    #: ``(plateID, wellID, fieldID)`` as the acquisition regex reports them.
    #: A folder whose names the regex does not understand gets one set per
    #: file, keyed ``("", "", <stem>)``.
    key: Tuple[str, str, str]
    #: Folder the files live in.
    directory: str
    #: ``{channel ID: file name}``, sorted by channel ID.
    channels: Dict[str, str] = _field(default_factory=dict)

    @property
    def label(self) -> str:
        """Short human label for the dropdown, e.g. ``A01 f003 (4ch)``."""
        _plate, well, fieldid = self.key
        stem = f"{well} f{fieldid}".strip() if well else str(fieldid)
        n = len(self.channels)
        return f"{stem} ({n}ch)" if n > 1 else stem

    def path(self, channel: Optional[str] = None) -> Path:
        """Full path to one channel's file — the lowest channel by default."""
        names = list(self.channels.values())
        if channel is not None and channel in self.channels:
            return Path(self.directory) / self.channels[channel]
        return Path(self.directory) / names[0]


@lru_cache(maxsize=1)
def _get_regex_callable():
    """Return :func:`spacr.utils._get_regex` without importing ``spacr.utils``.

    There must be exactly **one** definition of what a Yokogawa file name
    looks like, and it is ``spacr.utils._get_regex``. But importing
    ``spacr.utils`` costs a measured **3.2 s and ~900 MB of RSS** — it pulls in
    torch, cellpose and the rest of the scientific stack — and the Qt layer is
    built to never do that until a pipeline actually runs. Paying it so a
    dropdown can learn a filename pattern would replace the lag this module
    exists to remove.

    So: use the real function when the module happens to be loaded already,
    and otherwise compile *that one function* out of the source file. It has no
    module-level dependencies — only f-strings and ``print`` — so it executes
    standalone, and it is still the same single definition: edit ``_get_regex``
    and the previews follow it.

    :returns: the callable, or ``None`` if it could not be obtained.
    """
    module = sys.modules.get("spacr.utils")
    if module is not None:
        return getattr(module, "_get_regex", None)
    try:
        # find_spec locates the file without executing it.
        spec = importlib.util.find_spec("spacr.utils")
        source = Path(spec.origin).read_text(encoding="utf8")
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_get_regex":
                namespace: Dict[str, object] = {}
                exec(compile(ast.Module(body=[node], type_ignores=[]),
                             spec.origin, "exec"), namespace)
                return namespace["_get_regex"]
    except Exception:
        LOG.debug("Could not lift _get_regex from spacr.utils", exc_info=True)
    return None


@lru_cache(maxsize=32)
def _acquisition_regex(metadata_type: str = DEFAULT_METADATA_TYPE,
                       img_format: str = "tif",
                       custom_regex: Optional[str] = None):
    """Compile the project's own acquisition-filename regex.

    Reuses :func:`spacr.utils._get_regex` rather than growing a second parser
    — the preview must agree with the pipeline about what a file name means.
    That function prints its choice, which is right in a run log and noise in
    a GUI, so its stdout is swallowed here.

    :returns: a compiled pattern, or ``None`` when the dialect is unknown.
    """
    try:
        get_regex = _get_regex_callable()
        if get_regex is None:
            return None
        with contextlib.redirect_stdout(io.StringIO()):
            pattern = get_regex(metadata_type, img_format,
                                custom_regex=custom_regex)
        return re.compile(pattern, re.IGNORECASE)
    except Exception:
        # An unparsable custom regex must degrade to "one set per file",
        # never take the panel down.
        return None


def enumerate_image_sets(directory, suffixes: Sequence[str],
                         metadata_type: str = DEFAULT_METADATA_TYPE,
                         custom_regex: Optional[str] = None,
                         ) -> Tuple[List[ImageSet], List[str]]:
    """Group a folder's file *names* into image sets. Opens nothing.

    Reads the directory with :func:`os.scandir` — on Linux that answers
    "is this a file?" straight out of the dirent, so no ``stat`` is issued per
    entry — and matches each name against
    :func:`~spacr.utils._get_regex`. Names the regex understands are grouped by
    ``(plateID, wellID, fieldID)``; names it does not become one set each, so
    an ad-hoc folder of ``a.tif``/``b.tif`` still lists exactly as it always
    did.

    :param directory: folder to enumerate.
    :param suffixes: lower-case suffixes that count as a source.
    :param metadata_type: naming dialect, see :func:`spacr.utils._get_regex`.
    :param custom_regex: pattern body when ``metadata_type='custom'``.
    :returns: ``(sets sorted by key, channel IDs found across the folder)``.
    """
    try:
        directory = Path(os.fspath(directory))
    except TypeError:
        return [], []
    wanted = tuple(s.lower() for s in suffixes)
    # One pattern per file extension, looked up by the extension rather than
    # tried in turn: the acquisition regex back-tracks heavily, and on a
    # 98 304-file plate trying all five suffix variants per name cost 713 ms
    # against 368 ms for the single right one (os.scandir alone is 40 ms).
    patterns: Dict[str, "re.Pattern"] = {}
    for fmt in {s.lstrip(".").lower() for s in wanted} or {"tif"}:
        compiled = _acquisition_regex(metadata_type, fmt, custom_regex)
        if compiled is not None:
            patterns[fmt] = compiled

    grouped: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    channels: set = set()
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                name = entry.name
                lowered = name.lower()
                if not lowered.endswith(wanted):
                    continue
                try:
                    if not entry.is_file():
                        continue
                except OSError:
                    continue
                pattern = patterns.get(lowered.rpartition(".")[2])
                match = pattern.match(name) if pattern is not None else None
                if match:
                    groups = match.groupdict()
                    key = (str(groups.get("plateID") or ""),
                           str(groups.get("wellID") or ""),
                           str(groups.get("fieldID") or ""))
                    chan = str(groups.get("chanID") or "")
                    channels.add(chan)
                else:
                    # Not an acquisition name: one set per file, labelled with
                    # the file name exactly as the dropdown always showed it.
                    # Keyed on the *name*, not the stem, so ``a.tif`` and
                    # ``a.tiff`` stay two entries rather than colliding.
                    key = ("", "", name)
                    chan = ""
                grouped.setdefault(key, {}).setdefault(chan, name)
    except (OSError, ValueError):
        return [], []

    sets = [ImageSet(key=key, directory=str(directory),
                     channels=dict(sorted(chan_map.items())))
            for key, chan_map in sorted(grouped.items())]
    return sets, sorted(c for c in channels if c)


def sample_seed(directory, total: int, max_sets: int, nonce: int = 0) -> int:
    """The reproducible seed a sampled preview is drawn with.

    Digest of ``"<folder name>|<total sets>|<cap>|<nonce>"``. Stable across
    processes and machines — unlike :func:`hash`, which is salted per
    interpreter — so a user can name the plate and the cap and get the same
    sets back.

    Deliberately the folder's **name**, not its full path: the same plate read
    from a local copy and from the NAS it was acquired on must preview the same
    fields, or "the sample I looked at" is not a thing anyone can hand over.
    Two unrelated folders sharing a name draw the same *positions*, which
    selects different sets because their contents differ.
    """
    name = Path(os.fspath(directory)).name or str(directory)
    material = f"{name}|{int(total)}|{int(max_sets)}|{int(nonce)}"
    return int.from_bytes(
        hashlib.blake2b(material.encode("utf8"), digest_size=8).digest(),
        "big")


def sample_image_sets(sets: Sequence, max_sets: int, seed: int) -> List:
    """Draw at most ``max_sets`` entries from ``sets``, spread across it.

    The draw is random — so the sample represents the whole plate rather than
    the first N names, which on a plate-ordered folder is all of row A — but
    the winners are then restored to their original order, so the dropdown
    still reads front to back.

    ``max_sets`` of zero or less means "no cap".
    """
    items = list(sets)
    cap = int(max_sets)
    if cap <= 0 or len(items) <= cap:
        return items
    chosen = random.Random(seed).sample(range(len(items)), cap)
    chosen.sort()
    return [items[i] for i in chosen]


class ImageSetSampler:
    """Caches one folder's enumeration and hands out samples of it.

    The panels rebuild their selectors on every image they load. Enumerating
    the folder each time is what made stepping through a large plate cost
    292 ms a step, so the enumeration is done **once per folder** and every
    later call reuses it. Only :meth:`enumerate` touches the filesystem.

    Re-sampling is likewise deliberate: :meth:`sample` is a pure function of
    (folder, total, cap, nonce), so re-rendering after any settings change
    returns the identical sets. The sample changes only when the user changes
    the cap or calls :meth:`reshuffle`.
    """

    def __init__(self, max_sets: int = DEFAULT_MAX_SETS):
        self.max_sets = int(max_sets)
        self._directory: Optional[str] = None
        self._sets: List[ImageSet] = []
        self._channels: List[str] = []
        self._nonce = 0
        #: Set the user opened explicitly that the draw happened to miss.
        self._pinned: Optional[ImageSet] = None
        #: file name -> set, built on first lookup, dropped with the cache.
        self._by_name: Optional[Dict[str, ImageSet]] = None

    # -- enumeration (touches the filesystem) ------------------------------

    def enumerate(self, directory, suffixes: Sequence[str],
                  metadata_type: str = DEFAULT_METADATA_TYPE,
                  custom_regex: Optional[str] = None,
                  force: bool = False) -> List[ImageSet]:
        """Enumerate ``directory`` unless it is already the cached one."""
        key = str(directory)
        if not force and key == self._directory:
            return self._sets
        self._sets, self._channels = enumerate_image_sets(
            directory, suffixes, metadata_type, custom_regex)
        self._directory = key
        self._pinned = None
        self._by_name = None
        return self._sets

    def enumerate_paths(self, directory, lister, force: bool = False
                        ) -> List[ImageSet]:
        """Cache a caller-supplied listing of whole sources as one set each.

        For panels whose field of view is a *folder* of frames or a stacked
        array rather than a group of per-channel files. ``lister`` is only
        called when the folder is not the cached one, which is what keeps
        stepping through fields free.
        """
        key = str(directory)
        if not force and key == self._directory:
            return self._sets
        self._sets = sets_from_paths(lister())
        self._channels = []
        self._directory = key
        self._pinned = None
        self._by_name = None
        return self._sets

    def adopt(self, directory, sets: Sequence[ImageSet],
              channels: Sequence[str]) -> None:
        """Install an enumeration produced elsewhere — e.g. on a worker thread."""
        self._directory = str(directory)
        self._sets = list(sets)
        self._channels = list(channels)
        self._pinned = None
        self._by_name = None

    def invalidate(self) -> None:
        """Forget the cache, so the next :meth:`enumerate` really scans."""
        self._directory = None
        self._sets = []
        self._channels = []
        self._pinned = None
        self._by_name = None

    # -- sampling (pure) ---------------------------------------------------

    @property
    def total(self) -> int:
        """How many sets the folder holds, not how many are shown."""
        return len(self._sets)

    @property
    def directory(self) -> Optional[str]:
        return self._directory

    @property
    def channels(self) -> List[str]:
        """Channel IDs the enumeration found across the folder."""
        return list(self._channels)

    @property
    def sets(self) -> List[ImageSet]:
        """Every set the folder holds — the population, not the sample."""
        return list(self._sets)

    @property
    def seed(self) -> int:
        """The seed the current sample is drawn with."""
        return sample_seed(self._directory or "", self.total,
                           self.max_sets, self._nonce)

    def set_max(self, max_sets: int) -> bool:
        """Change the cap. Returns True when it actually changed."""
        value = int(max_sets)
        if value == self.max_sets:
            return False
        self.max_sets = value
        # A new cap is a new draw; the old pin has no claim on it.
        self._pinned = None
        return True

    def reshuffle(self) -> None:
        """Explicitly draw a different sample of the same folder."""
        self._nonce += 1
        self._pinned = None

    def pin(self, item: Optional[ImageSet]) -> None:
        """Keep ``item`` in the list even when the draw missed it.

        A user who drops one specific file on the panel must find it in the
        panel's own dropdown. The pin is **sticky**: it survives navigating
        away to a sampled field, so the file they opened stays reachable and
        the entry list does not shift under them while they browse. Redrawing
        the sample — the only thing that is allowed to change the list —
        clears it.
        """
        if item is not None and item in self._sets:
            self._pinned = item

    def sample(self, keep: Optional[ImageSet] = None) -> List[ImageSet]:
        """The sets to show: the draw, plus any pinned set.

        ``keep`` pins as a side effect, so callers can pass whatever is loaded
        without tracking the pin themselves.
        """
        self.pin(keep)
        picked = sample_image_sets(self._sets, self.max_sets, self.seed)
        pinned = self._pinned
        if pinned is not None and pinned not in picked and pinned in self._sets:
            picked = sorted(picked + [pinned], key=lambda s: s.key)
        return picked

    def set_for_path(self, path) -> Optional[ImageSet]:
        """The enumerated set a given file belongs to, if any.

        Indexed by file name on first use rather than scanned. This runs twice
        on every image load, and a linear scan of a 24 576-set plate put ~10 ms
        back onto each change of field — most of what the sampling had just
        taken off.
        """
        if path is None:
            return None
        if self._by_name is None:
            self._by_name = {name: item for item in self._sets
                             for name in item.channels.values()}
        return self._by_name.get(Path(os.fspath(path)).name)

    def describe(self, shown: int) -> str:
        """One sentence saying the preview is a sample, and of what.

        ``shown`` can exceed the cap by one when :meth:`sample` had to keep a
        loaded field that the draw missed; that extra entry is called out
        rather than quietly inflating the reported sample size.
        """
        if not self.total or shown >= self.total:
            return f"showing all {self.total} image sets"
        extra = ""
        if shown > self.max_sets > 0:
            shown = self.max_sets
            extra = ", plus the field you loaded"
        return (f"showing a random sample of {shown} of {self.total} "
                f"image sets{extra} (seed {self.seed:016x})")


def sets_from_paths(paths: Sequence[Path]) -> List[ImageSet]:
    """Wrap an already-listed set of sources as one :class:`ImageSet` each.

    The Timelapse and Motility previews' sources are whole *folders* of frames
    or stacked arrays — one source already is one field of view, so there is
    nothing to group. They still want the cap and the reproducible draw, so
    they feed their own listing through here and share the sampler.
    """
    out: List[ImageSet] = []
    for path in paths:
        p = Path(os.fspath(path))
        out.append(ImageSet(key=("", "", p.name), directory=str(p.parent),
                            channels={"": p.name}))
    return out


def apply_sample_to_combo(combo: QComboBox, box: Optional[QSpinBox],
                          sampler: "ImageSetSampler", current_path,
                          tooltip: str = "") -> str:
    """Point a sets dropdown at the sampler's current sample.

    Configures the cap box, draws the sample (keeping whatever is loaded), and
    refills the dropdown. Touches no file: the sampler must already have been
    enumerated.

    :returns: the sentence stating what fraction of the folder is on show.
    """
    if box is not None:
        sampler.set_max(configure_max_sets_box(box, sampler.total))
    current_set = sampler.set_for_path(current_path)
    shown = sampler.sample(keep=current_set)
    current = current_set.path() if current_set is not None else current_path
    populate_fov_combo(combo, [s.path() for s in shown], current=current,
                       labels=[s.label for s in shown])
    note = sampler.describe(len(shown))
    combo.setToolTip(f"{tooltip} — {note}.".lstrip(" —") if tooltip else note)
    return note


def configure_max_sets_box(box: QSpinBox, total: int) -> int:
    """Point a :class:`FlatSpinBox` at a freshly enumerated folder.

    The suffix carries the total, so the control reads ``20 of 24576 sets``.
    The maximum is clamped to the total so it can never read ``50 of 12`` —
    a cap above what exists is not a real cap.

    :returns: the cap the box now holds, which the clamp may have lowered.
        Callers must feed it back to the sampler, or a folder small enough to
        clamp would leave the box saying 12 while the dropdown showed 50.
    """
    total = max(0, int(total))
    blocked = box.blockSignals(True)
    try:
        box.setSuffix(f" of {total} sets" if total else " sets")
        box.setMaximum(max(1, total) if total else 10_000_000)
        box.setEnabled(total > 1)
    finally:
        box.blockSignals(blocked)
    return int(box.value())
