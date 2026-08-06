"""Watching a track break, rather than reading that one did.

``TrackStats`` can tell you that a field produced 41 tracks with a median
length of 6 frames in a 30-frame series. It cannot tell you *why*, and why
is the only thing that changes what you do next: two cells that touch for
three frames and come apart with their identities swapped needs a different
setting from one cell that leaves the field and comes back.

So this shows the frames. One field plays as a movie; clicking it opens a
filmstrip above, which is the same frames laid out at once and scrollable,
because a break is easiest to find by scrubbing and easiest to *understand*
by seeing the frames either side of it together.

Everything reuses the rendering already in
:mod:`spacr.qt.widgets.timelapse_preview` -- ``render_frame`` colours mask
outlines by track id and draws the trailing polyline, and ``track_colour``
is deterministic. That is what makes the colours mean something: an object
that keeps its track id keeps its colour in every frame and in every
thumbnail, so an identity swap reads as an object that changes colour
partway through the strip.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QSpinBox,
                               QVBoxLayout, QWidget)

LOG = logging.getLogger(__name__)

#: Height of one filmstrip thumbnail. Big enough that an object the size of
#: a nucleus is still a shape rather than a smudge, small enough that a
#: 30-frame series fits a couple of scrolls.
THUMB_H = 96

#: Frames per second the movie plays at when nothing says otherwise. Slow
#: on purpose: this is watched to catch a swap between two frames, not to
#: admire the motion.
DEFAULT_FPS = 4

#: How many fields may be stacked at once, whatever the user asks for.
#: Each one holds its own rendered frames, so this is a memory bound, not a
#: layout preference -- see :meth:`TimelapseMoviePanel.set_max_fields`.
MAX_FIELDS_CEILING = 8


def _to_pixmap(rgb: np.ndarray) -> QPixmap:
    """RGB uint8 array to a pixmap, without going through the disk."""
    from .live_preview import numpy_to_qpixmap

    return numpy_to_qpixmap(rgb, normalise=False)


class FilmStrip(QScrollArea):
    """One field's frames, side by side and scrollable.

    Horizontal only. A vertical scrollbar here would fight the panel's own,
    and there is never more than one row.
    """

    frame_picked = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("TimelapseFilmStrip")
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setFixedHeight(THUMB_H + 34)
        # Scaffolding: it positions thumbnails and must paint nothing, or it
        # is one more opaque rectangle over the page. See
        # `spacr.qt.theme.make_transparent`.
        self.viewport().setAutoFillBackground(False)

        self._body = QWidget()
        self._body.setAutoFillBackground(False)
        self._row = QHBoxLayout(self._body)
        self._row.setContentsMargins(4, 4, 4, 4)
        self._row.setSpacing(4)
        self._row.addStretch(1)
        self.setWidget(self._body)
        self._cells: List[QLabel] = []

    def set_frames(self, frames: Sequence[QPixmap]) -> None:
        """Replace the strip. Cheap to call: the pixmaps are already made."""
        while self._row.count() > 1:
            item = self._row.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cells = []

        for index, pixmap in enumerate(frames):
            cell = QLabel()
            cell.setObjectName("TimelapseFilmStripCell")
            cell.setPixmap(pixmap)
            cell.setToolTip(f"Frame {index}")
            cell.setCursor(Qt.PointingHandCursor)
            cell.mousePressEvent = (
                lambda _event, i=index: self.frame_picked.emit(i))
            self._row.insertWidget(self._row.count() - 1, cell)
            self._cells.append(cell)

    def highlight(self, index: int) -> None:
        """Ring the frame the movie is on, so the two views agree."""
        from ..theme import active_palette

        accent = active_palette().get("accent", "#4a9eff")
        for i, cell in enumerate(self._cells):
            cell.setStyleSheet(
                f"border: 2px solid {accent};" if i == index
                else "border: 2px solid transparent;")


class FovMovie(QWidget):
    """One field of view: a movie, and a filmstrip that opens above it.

    Renders lazily and caches by ``(frame, objects, tracks)``. Flipping a
    toggle on a 30-frame field would otherwise re-render every frame twice
    -- once for the movie and once for the strip -- on the GUI thread.
    """

    def __init__(self, title: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("TimelapseFovMovie")
        self.setAutoFillBackground(False)

        self._images: Optional[np.ndarray] = None
        self._labels: Optional[np.ndarray] = None
        self._tracks = None
        self._channel = 0
        self._show_objects = True
        self._show_tracks = True
        self._frame = 0
        self._cache: Dict[Tuple[int, bool, bool], np.ndarray] = {}

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4)

        # The strip lives ABOVE the movie and starts collapsed: the request
        # was that clicking the movie "expands upwards into a row of frames".
        self._strip = FilmStrip(self)
        self._strip.hide()
        self._strip.frame_picked.connect(self.show_frame)
        column.addWidget(self._strip)

        self._title = QLabel(title)
        self._title.setObjectName("Muted")
        column.addWidget(self._title)

        self._canvas = QLabel()
        self._canvas.setObjectName("TimelapseMovieCanvas")
        self._canvas.setAlignment(Qt.AlignCenter)
        self._canvas.setMinimumHeight(180)
        self._canvas.setCursor(Qt.PointingHandCursor)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._canvas.setToolTip(
            "Click to open the frames of this field side by side.")
        self._canvas.mousePressEvent = lambda _e: self.toggle_strip()
        column.addWidget(self._canvas, 1)

        controls = QHBoxLayout()
        controls.setSpacing(6)
        self._play = QPushButton("Play")
        self._play.setObjectName("GhostButton")
        self._play.setCursor(Qt.PointingHandCursor)
        self._play.clicked.connect(self.toggle_play)
        controls.addWidget(self._play)

        self._scrub = QSlider(Qt.Horizontal)
        self._scrub.setMinimum(0)
        self._scrub.setMaximum(0)
        self._scrub.valueChanged.connect(self.show_frame)
        controls.addWidget(self._scrub, 1)

        self._counter = QLabel("0 / 0")
        self._counter.setObjectName("Muted")
        controls.addWidget(self._counter)
        column.addLayout(controls)

        self._timer = QTimer(self)
        self._timer.setInterval(int(1000 / DEFAULT_FPS))
        self._timer.timeout.connect(self._advance)

    # -- content -------------------------------------------------------
    def set_sequence(self, images, labels=None, tracks=None,
                     channel: int = 0) -> None:
        """Bind one field's frames, its per-frame labels and its tracks.

        ``labels`` is expected to be ALREADY relabelled by track id (see
        ``relabel_by_track``). That is what makes a colour mean a track
        rather than a per-frame segmentation index, and it is the caller's
        job because the relabelling is what the tracker produced.
        """
        self._images = None if images is None else np.asarray(images)
        self._labels = None if labels is None else np.asarray(labels)
        self._tracks = tracks
        self._channel = int(channel)
        self._cache.clear()

        count = 0 if self._images is None else int(len(self._images))
        self._scrub.setMaximum(max(0, count - 1))
        self._frame = 0
        self._render_strip()
        self.show_frame(0)

    def set_overlays(self, *, objects: bool, tracks: bool) -> None:
        """Toggle the mask outlines and the track tails independently."""
        if (objects, tracks) == (self._show_objects, self._show_tracks):
            return
        self._show_objects = bool(objects)
        self._show_tracks = bool(tracks)
        self._render_strip()
        self.show_frame(self._frame)

    def frame_count(self) -> int:
        return 0 if self._images is None else int(len(self._images))

    # -- rendering -----------------------------------------------------
    def _rendered(self, index: int) -> Optional[np.ndarray]:
        """One composited frame, from the cache when it is already made."""
        if self._images is None or not len(self._images):
            return None
        index = max(0, min(int(index), len(self._images) - 1))
        key = (index, self._show_objects, self._show_tracks)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        from .timelapse_preview import render_frame

        labels = None
        if self._show_objects and self._labels is not None \
                and index < len(self._labels):
            labels = self._labels[index]
        try:
            rgb = render_frame(
                self._images[index],
                labels=labels,
                tracks=self._tracks if self._show_tracks else None,
                frame=index,
                channel=self._channel,
            )
        except Exception:
            LOG.debug("could not render frame %s", index, exc_info=True)
            return None
        self._cache[key] = rgb
        return rgb

    def _render_strip(self) -> None:
        if self._images is None or not len(self._images):
            self._strip.set_frames([])
            return
        thumbs = []
        for index in range(len(self._images)):
            rgb = self._rendered(index)
            if rgb is None:
                continue
            pixmap = _to_pixmap(rgb)
            thumbs.append(pixmap.scaledToHeight(
                THUMB_H, Qt.SmoothTransformation))
        self._strip.set_frames(thumbs)
        self._strip.highlight(self._frame)

    def show_frame(self, index: int) -> None:
        """Put ``index`` on the canvas and keep every control agreeing."""
        rgb = self._rendered(index)
        total = self.frame_count()
        self._frame = max(0, min(int(index), max(0, total - 1)))
        self._counter.setText(f"{self._frame + 1} / {total}" if total
                              else "0 / 0")
        if self._scrub.value() != self._frame:
            # Without the guard this re-enters through `valueChanged`.
            self._scrub.blockSignals(True)
            self._scrub.setValue(self._frame)
            self._scrub.blockSignals(False)
        self._strip.highlight(self._frame)
        if rgb is None:
            self._canvas.setText("No frames loaded.")
            return
        pixmap = _to_pixmap(rgb)
        target = self._canvas.size()
        if target.width() > 8 and target.height() > 8:
            pixmap = pixmap.scaled(target, Qt.KeepAspectRatio,
                                   Qt.SmoothTransformation)
        self._canvas.setPixmap(pixmap)

    # -- playback ------------------------------------------------------
    def toggle_play(self) -> None:
        if self._timer.isActive():
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        if self.frame_count() < 2:
            return
        self._timer.start()
        self._play.setText("Pause")

    def pause(self) -> None:
        self._timer.stop()
        self._play.setText("Play")

    def set_fps(self, fps: float) -> None:
        self._timer.setInterval(int(1000 / max(0.5, float(fps))))

    def _advance(self) -> None:
        total = self.frame_count()
        if total < 2:
            self.pause()
            return
        self.show_frame((self._frame + 1) % total)

    # -- the strip -----------------------------------------------------
    def toggle_strip(self) -> None:
        self.set_strip_open(not self.strip_is_open())

    def set_strip_open(self, open_: bool) -> None:
        self._strip.setVisible(bool(open_))
        if open_:
            self._strip.highlight(self._frame)

    def strip_is_open(self) -> bool:
        """Expanded or not -- a state, not a question about the screen.

        `isHidden()` and not `isVisible()`: a widget inside a parent that
        has not been shown yet reports `isVisible() == False` however it
        was configured, so the expanded state would read as collapsed for
        every movie built before its screen is on screen -- and the click
        that expanded it would appear to do nothing.
        """
        return not self._strip.isHidden()


class TimelapseMoviePanel(QWidget):
    """Several fields stacked, with one set of controls over all of them.

    Stacked rather than tabbed on purpose: comparing two fields is the
    point of showing more than one, and a tab hides the thing you are
    comparing against.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setObjectName("TimelapseMoviePanel")
        self.setAutoFillBackground(False)

        self._movies: List[FovMovie] = []
        self._max_fields = 2

        column = QVBoxLayout(self)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(6)

        row = QHBoxLayout()
        row.setSpacing(10)

        self._objects_check = QCheckBox("Objects")
        self._objects_check.setChecked(True)
        self._objects_check.setToolTip(
            "Outline each segmented object, coloured by the track it "
            "belongs to.")
        self._objects_check.toggled.connect(self._sync_overlays)
        row.addWidget(self._objects_check)

        self._tracks_check = QCheckBox("Tracks")
        self._tracks_check.setChecked(True)
        self._tracks_check.setToolTip(
            "Draw where each object came from over the preceding frames, "
            "in the same colour as its outline.")
        self._tracks_check.toggled.connect(self._sync_overlays)
        row.addWidget(self._tracks_check)

        row.addSpacing(12)
        row.addWidget(QLabel("Fields"))
        self._fields_spin = QSpinBox()
        self._fields_spin.setRange(1, MAX_FIELDS_CEILING)
        self._fields_spin.setValue(self._max_fields)
        self._fields_spin.setToolTip(
            "How many fields of view to stack. Each one keeps its own "
            "rendered frames in memory, so this is a memory setting as "
            "much as a layout one.")
        self._fields_spin.valueChanged.connect(self.set_max_fields)
        row.addWidget(self._fields_spin)

        self._play_all = QPushButton("Play all")
        self._play_all.setObjectName("GhostButton")
        self._play_all.setCursor(Qt.PointingHandCursor)
        self._play_all.clicked.connect(self._toggle_all)
        row.addWidget(self._play_all)

        row.addStretch(1)
        column.addLayout(row)

        self._stack = QVBoxLayout()
        self._stack.setSpacing(8)
        column.addLayout(self._stack, 1)

        self._empty = QLabel("Run a preview to see the fields here.")
        self._empty.setObjectName("Muted")
        self._empty.setAlignment(Qt.AlignCenter)
        column.addWidget(self._empty)

    # -- content -------------------------------------------------------
    def set_fields(self, fields: Sequence[dict]) -> None:
        """Show one movie per entry, up to the user's ceiling.

        :param fields: dicts of ``{"title", "images", "labels", "tracks",
            "channel"}``. Extra entries are dropped rather than queued --
            the ceiling is about memory, so holding the surplus would
            defeat it.
        """
        wanted = list(fields)[: self._max_fields]
        self._empty.setVisible(not wanted)

        while len(self._movies) > len(wanted):
            movie = self._movies.pop()
            movie.pause()
            self._stack.removeWidget(movie)
            movie.deleteLater()
        while len(self._movies) < len(wanted):
            movie = FovMovie(parent=self)
            self._stack.addWidget(movie)
            self._movies.append(movie)

        for movie, field in zip(self._movies, wanted):
            movie._title.setText(str(field.get("title", "")))
            movie.set_sequence(
                field.get("images"),
                labels=field.get("labels"),
                tracks=field.get("tracks"),
                channel=int(field.get("channel", 0)),
            )
        self._sync_overlays()

    def set_max_fields(self, count: int) -> None:
        """Cap how many fields are held at once.

        Applied by dropping the surplus immediately rather than at the next
        preview: a user lowering this has just been told the machine is
        short of memory, and the setting has to give it back now.
        """
        self._max_fields = max(1, min(int(count), MAX_FIELDS_CEILING))
        if self._fields_spin.value() != self._max_fields:
            self._fields_spin.blockSignals(True)
            self._fields_spin.setValue(self._max_fields)
            self._fields_spin.blockSignals(False)
        while len(self._movies) > self._max_fields:
            movie = self._movies.pop()
            movie.pause()
            self._stack.removeWidget(movie)
            movie.deleteLater()

    def max_fields(self) -> int:
        return self._max_fields

    def movies(self) -> List[FovMovie]:
        return list(self._movies)

    # -- controls ------------------------------------------------------
    def _sync_overlays(self, *_args) -> None:
        objects = self._objects_check.isChecked()
        tracks = self._tracks_check.isChecked()
        for movie in self._movies:
            movie.set_overlays(objects=objects, tracks=tracks)

    def _toggle_all(self) -> None:
        playing = any(m._timer.isActive() for m in self._movies)
        for movie in self._movies:
            movie.pause() if playing else movie.play()
        self._play_all.setText("Play all" if playing else "Pause all")

    def set_fps(self, fps: float) -> None:
        for movie in self._movies:
            movie.set_fps(fps)
