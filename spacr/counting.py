"""Manual counting on a points layer — click to add, click to remove, tally.

The oldest measurement in the building. Somebody opens a field, counts the
infected cells by eye, writes a number on a sticky note, and the number is the
result: no record of where the clicks were, no way to recount, no way for a
second scorer to agree or disagree with anything but the total.

This module is that job done on top of :class:`spacr.layers.PointsLayer`, so
the clicks are *data*. Every marker is a world coordinate on a layer that can
be hidden, recoloured, saved and reopened; the tally is derived from the
markers rather than typed; and the export is one row per click, so two scorers
can be compared point by point and a disputed count can be looked at rather
than argued about.

One layer per class
-------------------

A class is a whole :class:`~spacr.layers.PointsLayer`, not a property column on
a shared one. That is what buys per-class colour and per-class visibility from
the existing model — "hide the uninfected markers and count again" is a
checkbox in the layer list, not a feature — and it is why the class colours
here are the layer's own ``face_color``. The undo history that crosses classes
lives in the session, which is the one thing a per-layer view cannot own.

Coordinates are world coordinates
---------------------------------

A marker placed at 8× zoom and a marker placed at 1× are the same point, and a
count exported from a downsampled preview lines up with the full-resolution
mask it was counted on. :meth:`CountingSession.to_frame` writes the world
coordinates *and* the unit they are in, because a column of numbers headed
``x`` has been read as pixels when it was µm.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (Any, Dict, Iterable, List, Mapping, Optional, Sequence,
                    Tuple)

import numpy as np

from .layers import (FieldKey, LayerError, LayerStack, PointsLayer, Spacing,
                     to_rgba)

__all__ = [
    'CountClass',
    'CountingSession',
    'DEFAULT_CLASSES',
    'LAYER_PREFIX',
]

#: What a counting layer is called in the stack: ``count: infected``. The
#: prefix is how a session finds the layers it owns in a stack that also holds
#: the image, the mask and whatever else the user opened.
LAYER_PREFIX = 'count: '

#: The classes a session starts with when the caller does not say. Two, because
#: the commonest manual count in this codebase is "infected / uninfected" and a
#: single class is a tally counter rather than a scoring session.
DEFAULT_CLASSES: Tuple[Tuple[str, str], ...] = (
    ('infected', 'magenta'),
    ('uninfected', 'cyan'),
)


@dataclass(frozen=True)
class CountClass:
    """One thing being counted: a name, a colour and a shortcut.

    :param name: what it is called, in the tally and in the export. Unique
        within a session — two classes with one name is a count nobody can
        interpret.
    :param color: the marker colour, in any form :func:`spacr.layers.to_rgba`
        takes.
    :param shortcut: the key that selects it, blank by default —
        :meth:`CountingSession.add_class` is what fills in ``'1'``–``'9'`` by
        position. A counting session is a keyboard job.
    """

    name: str
    color: Any = 'yellow'
    shortcut: str = ''

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise LayerError('a counting class needs a non-blank name')
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'color', to_rgba(self.color))
        object.__setattr__(self, 'shortcut', str(self.shortcut))


class CountingSession:
    """Counting by hand over a :class:`~spacr.layers.LayerStack`.

    :param stack: the stack the markers are added to. Its layers supply the
        spacing, so a marker is placed in the same world as the image being
        counted.
    :param classes: the things being counted. Names, ``(name, colour)`` pairs
        or :class:`CountClass` instances; defaults to
        :data:`DEFAULT_CLASSES`.
    :param spacing: the spacing for the marker layers. Defaults to the first
        2-D layer's, which is what makes a count line up with the field.
    :param size: marker DIAMETER in world units — world, so a marker is the
        same physical size at every zoom.
    :param field: the :class:`~spacr.layers.FieldKey` being counted, if it is
        known. It is written into the export, which is what lets a count join
        the measurement tables instead of being a loose CSV.
    """

    def __init__(self, stack: LayerStack, *,
                 classes: Optional[Iterable[Any]] = None,
                 spacing: Optional[Spacing] = None, size: float = 12.0,
                 field: Optional[FieldKey] = None):
        if not isinstance(stack, LayerStack):
            raise LayerError(
                f'a counting session counts on a LayerStack, got {stack!r}')
        self._stack = stack
        self._spacing = spacing or self._default_spacing(stack)
        self._size = float(size)
        self._field = field
        self._classes: List[CountClass] = []
        self._layers: Dict[str, PointsLayer] = {}
        self._history: List[Tuple[str, str, np.ndarray]] = []
        self._active = ''
        for index, spec in enumerate(classes if classes is not None
                                     else DEFAULT_CLASSES):
            self.add_class(spec, shortcut_index=index)

    @staticmethod
    def _default_spacing(stack: LayerStack) -> Spacing:
        for layer in stack:
            if layer.ndim == 2:
                return layer.spacing
        return Spacing.isotropic(2, 1.0, units=stack.units)

    # -- classes ---------------------------------------------------------
    @property
    def classes(self) -> Tuple[CountClass, ...]:
        """Every class being counted, in the order they were added."""
        return tuple(self._classes)

    @property
    def class_names(self) -> Tuple[str, ...]:
        """Just the names, in order."""
        return tuple(c.name for c in self._classes)

    def add_class(self, spec: Any, *,
                  shortcut_index: Optional[int] = None) -> CountClass:
        """Add a class and its marker layer; returns the class.

        :param spec: a :class:`CountClass`, a ``(name, colour)`` pair, or a
            bare name (which is given the next default colour).
        :param shortcut_index: zero-based position deciding the ``'1'``–``'9'``
            key the class answers to; defaults to the number of classes already
            added. Ignored when ``spec`` already carries a shortcut, and no key
            is assigned past ``'9'``.
        :raises LayerError: on a duplicate name.
        """
        if isinstance(spec, CountClass):
            entry = spec
        elif isinstance(spec, str):
            entry = CountClass(spec, self._next_color())
        else:
            name, colour = tuple(spec)
            entry = CountClass(name, colour)
        if not entry.shortcut and shortcut_index is None:
            shortcut_index = len(self._classes)
        if not entry.shortcut and shortcut_index is not None \
                and shortcut_index < 9:
            entry = CountClass(entry.name, entry.color,
                               str(shortcut_index + 1))
        if entry.name in self._layers:
            raise LayerError(
                f'this session already counts {entry.name!r}. Two classes with '
                f'one name is a tally nobody can interpret.')
        layer = self._stack.add_points(
            name=f'{LAYER_PREFIX}{entry.name}', ndim=self._spacing.ndim,
            spacing=self._spacing, size=self._size, face_color=entry.color,
            border_color='black', border_width=self._size / 6.0)
        self._classes.append(entry)
        self._layers[entry.name] = layer
        if not self._active:
            self._active = entry.name
        return entry

    def _next_color(self) -> str:
        from .layers import DEFAULT_CHANNEL_COLORMAPS
        return DEFAULT_CHANNEL_COLORMAPS[
            len(self._classes) % len(DEFAULT_CHANNEL_COLORMAPS)]

    def layer(self, name: Optional[str] = None) -> PointsLayer:
        """The marker layer of a class (the active one by default)."""
        key = self._check_class(name)
        return self._layers[key]

    def class_for_shortcut(self, key: str) -> Optional[str]:
        """The class a keystroke selects, or ``None``."""
        for entry in self._classes:
            if entry.shortcut and entry.shortcut == str(key):
                return entry.name
        return None

    @property
    def active(self) -> str:
        """The class a new marker gets."""
        return self._active

    @active.setter
    def active(self, name: str) -> None:
        self._active = self._check_class(name)

    def _check_class(self, name: Optional[str]) -> str:
        key = self._active if name is None else str(name)
        if key not in self._layers:
            raise LayerError(
                f'this session does not count {key!r}; it counts '
                f'{list(self.class_names)}')
        return key

    # -- counting --------------------------------------------------------
    def add(self, world: Mapping[str, float],
            name: Optional[str] = None) -> int:
        """Place a marker at a world point; returns its index in its layer."""
        key = self._check_class(name)
        index = self._layers[key].add_world(world)
        self._history.append(('add', key, self._layers[key].data[index].copy()))
        return index

    def find(self, world: Mapping[str, float]
             ) -> Optional[Tuple[str, int]]:
        """The ``(class, index)`` of the marker under a world point, if any.

        Searched over every class, not just the active one, and the topmost
        class wins a tie. Clicking a marker means "that one", whatever it was
        scored as — a counter who has to re-select the class before they can
        take a marker back will leave the wrong marker there.
        """
        for entry in reversed(self._classes):
            found = self._layers[entry.name].nearest(world)
            if found is not None:
                return entry.name, found
        return None

    def remove_at(self, world: Mapping[str, float]) -> Optional[Tuple[str, int]]:
        """Take away the marker under a world point; returns what went."""
        found = self.find(world)
        if found is None:
            return None
        name, index = found
        layer = self._layers[name]
        coordinates = layer.data[index].copy()
        layer.remove(index)
        self._history.append(('remove', name, coordinates))
        return found

    def toggle(self, world: Mapping[str, float],
               name: Optional[str] = None) -> Tuple[str, str, int]:
        """One click: remove the marker there, or place one if there is none.

        :returns: ``(action, class, index)`` where ``action`` is ``'added'`` or
            ``'removed'``.
        """
        removed = self.remove_at(world)
        if removed is not None:
            return ('removed',) + removed
        key = self._check_class(name)
        return 'added', key, self.add(world, key)

    def undo(self) -> Optional[Tuple[str, str]]:
        """Reverse the last add or remove; returns ``(action, class)``.

        A counting session is thousands of clicks and some of them are wrong.
        Undo covers removals too, so a marker deleted by accident comes back
        where it was rather than where the cursor now is.
        """
        if not self._history:
            return None
        action, name, coordinates = self._history.pop()
        layer = self._layers[name]
        if action == 'add':
            index = self._index_of(layer, coordinates)
            if index is not None:
                layer.remove(index)
            return 'add', name
        layer.add(coordinates)
        return 'remove', name

    @staticmethod
    def _index_of(layer: PointsLayer,
                  coordinates: np.ndarray) -> Optional[int]:
        data = layer.data
        if len(data) == 0:
            return None
        matches = np.flatnonzero(np.all(np.isclose(data, coordinates), axis=1))
        return int(matches[-1]) if matches.size else None

    def clear(self, name: Optional[str] = None) -> int:
        """Remove every marker of a class, or of every class; returns how many."""
        names = self.class_names if name is None else (self._check_class(name),)
        removed = 0
        for key in names:
            layer = self._layers[key]
            removed += len(layer.data)
            layer.data = np.zeros((0, self._spacing.ndim), dtype=np.float64)
        self._history = [entry for entry in self._history
                         if entry[1] not in names]
        return removed

    # -- the tally -------------------------------------------------------
    def counts(self) -> Dict[str, int]:
        """``{class: how many markers}``, in class order."""
        return {entry.name: int(len(self._layers[entry.name].data))
                for entry in self._classes}

    @property
    def total(self) -> int:
        """How many markers there are altogether."""
        return sum(self.counts().values())

    def fraction(self, name: str) -> float:
        """A class's share of the total, or 0.0 when nothing is counted.

        The number a manual count is usually for — "42% infected" — computed
        rather than divided by hand, and 0.0 rather than a ZeroDivisionError on
        an empty session because a fresh panel asks for it before the first
        click.
        """
        counts = self.counts()
        total = sum(counts.values())
        return counts[self._check_class(name)] / total if total else 0.0

    def describe(self) -> str:
        """One line for a status bar, ``'nothing counted yet'`` when empty.

        Every class gets a percentage and the total is appended:
        ``infected 12 (40%) · uninfected 18 (60%) · 30 total``.
        """
        counts = self.counts()
        total = sum(counts.values())
        if not total:
            return 'nothing counted yet'
        parts = [f'{name} {n} ({n / total:.0%})' for name, n in counts.items()]
        return ' · '.join(parts) + f' · {total} total'

    # -- export ----------------------------------------------------------
    def to_frame(self):
        """One row per marker: class, world coordinates, units, field key.

        The world coordinates and the unit travel together on purpose: a
        column headed ``x`` has been read as pixels when it was µm, and the
        two counts differ by a factor nobody notices until the figure is
        drawn.
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        axes = self._spacing.axes
        for entry in self._classes:
            world = self._layers[entry.name].world
            for point in world:
                row: Dict[str, Any] = {'class': entry.name}
                if self._field is not None:
                    row.update(dict(self._field.values))
                row.update({axis: float(value)
                            for axis, value in zip(axes, point)})
                row['units'] = self._spacing.units
                rows.append(row)
        columns = ['class']
        if self._field is not None:
            columns += list(self._field.values)
        columns += list(axes) + ['units']
        return pd.DataFrame(rows, columns=columns)

    def summary(self):
        """One row per class: the field key, class, count, fraction and total."""
        import pandas as pd

        counts = self.counts()
        total = sum(counts.values())
        rows = []
        for name, n in counts.items():
            row: Dict[str, Any] = {}
            if self._field is not None:
                row.update(dict(self._field.values))
            row.update({'class': name, 'count': n,
                        'fraction': (n / total) if total else 0.0,
                        'total': total})
            rows.append(row)
        return pd.DataFrame(rows)

    def to_csv(self, path: str, *, summary: bool = False) -> str:
        """Write the export to ``path``; returns the absolute path.

        :param summary: write one row per class instead of one per marker.
        """
        import os

        target = os.path.abspath(str(path))
        parent = os.path.dirname(target)
        if parent:
            os.makedirs(parent, exist_ok=True)
        frame = self.summary() if summary else self.to_frame()
        frame.to_csv(target, index=False)
        return target

    def load_frame(self, frame) -> int:
        """Put a previously exported count back on the canvas; returns how many.

        Classes the session does not have are added as it goes, so reopening
        somebody else's count does not require declaring their classes first.
        Markers are placed by WORLD coordinate, which is what makes the reload
        land where the clicks were even on a differently-scaled view.

        :raises LayerError: if the frame was counted in different units — the
            coordinates would be silently wrong by whatever the two units
            differ by.
        """
        axes = self._spacing.axes
        missing = [name for name in ('class',) + tuple(axes)
                   if name not in frame.columns]
        if missing:
            raise LayerError(
                f'a counted frame needs {missing} to be placed; it has '
                f'{list(frame.columns)}')
        if 'units' in frame.columns and len(frame):
            units = {str(u) for u in frame['units']}
            if units != {self._spacing.units}:
                raise LayerError(
                    f'the count was made in {sorted(units)} and this session '
                    f'is in {self._spacing.units!r}. Placing one on the other '
                    f'would put every marker somewhere plausible and wrong.')
        placed = 0
        for _index, row in frame.iterrows():
            name = str(row['class'])
            if name not in self._layers:
                self.add_class(name)
            self.add({axis: float(row[axis]) for axis in axes}, name)
            placed += 1
        return placed

    def detach(self) -> None:
        """Take the marker layers out of the stack, leaving the tally readable.

        The counts are still available afterwards: the session keeps its
        layers, it just stops showing them. What a screen calls when it closes
        but the number is still wanted.
        """
        for layer in self._layers.values():
            if layer.stack is self._stack:
                self._stack.remove(layer)
