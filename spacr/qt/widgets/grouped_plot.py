"""One pyqtgraph plot that draws any grouped comparison, and can change kind.

    "I WANT ALL FIGURES TO BE IN pyqtgraph NOT matplotlib. i also want to be
     able to right click on the graph and change the graph typem to whatever
     is possibel with the underlying data. and i want to be able to save the
     figure intp afolder as stats, data, ong figure and pdf figure"

THE THREE ASKS ARE ONE ASK, and answering them separately is why they kept
coming back. A plot that HOLDS ITS DATA can be redrawn as any kind the data
supports, and can write that data and its statistics beside the picture. A
plot that holds only a rendering can do neither -- which is what a
matplotlib Figure in a queue is.

So the unit here is a SPEC -- the frame, which column is the group, which is
the value, and what kind to draw -- and the widget renders it. Changing the
kind is re-rendering the same spec, not redrawing from a recipe that has to
be kept in step with the picture.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fast_plots import FastPlot

LOG = logging.getLogger("spacr.qt.grouped_plot")

#: How a `graph_types` kind is drawn by `FastPlot.add_group_mark`.
#:
#: TWO VOCABULARIES, MAPPED ONCE. `graph_types` is the analysis vocabulary --
#: what fits what data -- and `MARK_TYPES` is the drawer's. Renaming either
#: to match the other would make one of them wrong in its own module.
MARKS: Dict[str, str] = {
    "bar": "bar",
    "bar_jitter": "jitter_bar",
    "jitter": "jitter",
    "box": "box",
    "violin": "violin",
    "line": "line",
    "scatter": "points",
}


@dataclass
class PlotSpec:
    """What a graph IS, before it is drawn.

    THE DATA TRAVELS WITH THE PICTURE. That is the whole design: retyping
    needs to know what the data can support, and the folder save needs the
    rows and the groups. A figure that carries only pixels can answer
    neither, and both features had to be bolted onto a side-channel recipe
    that could drift from what was drawn.
    """

    frame: Any
    value: str
    group: str = ""
    kind: str = ""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    #: What ONE observation is -- 'well', 'cell', 'guide'. Carried because
    #: the statistics file has to state it: a test across cells when the
    #: replicate is the well returns p < 1e-10 on noise.
    unit: str = "observation"
    #: Colour per group, when the user has chosen one.
    colours: Dict[str, str] = field(default_factory=dict)

    def shape(self) -> str:
        """The data shape, for deciding which kinds fit."""
        from ...graph_types import shape_of

        return shape_of(self.frame, self.group, self.value)

    def default_kind(self) -> str:
        """What this data is born as, when no kind is named."""
        from ...graph_types import default_for

        return default_for(self.shape())

    def groups(self) -> Dict[str, np.ndarray]:
        """``{label: values}``, in the frame's own order.

        THE FRAME'S ORDER, not sorted: a screen's conditions are usually
        laid out deliberately, and sorting them alphabetically puts the
        control in the middle.
        """
        frame = self.frame
        if frame is None or self.value not in getattr(frame, "columns", ()):
            return {}
        if not self.group or self.group not in frame.columns:
            values = pd.to_numeric(frame[self.value],
                                   errors="coerce").dropna()
            return {self.value: values.to_numpy(dtype=float)}
        out: Dict[str, np.ndarray] = {}
        for label in frame[self.group].astype(str):
            if label in out:
                continue
            part = frame.loc[frame[self.group].astype(str) == label,
                             self.value]
            out[label] = pd.to_numeric(part, errors="coerce").dropna(
            ).to_numpy(dtype=float)
        return out


class GroupedPlot(FastPlot):
    """A pyqtgraph plot of a :class:`PlotSpec`, redrawable as any kind.

    :ivar spec: what is drawn. Assign through :meth:`show_spec`.
    """

    def __init__(self, spec: Optional[PlotSpec] = None, parent=None,
                 **kwargs):
        super().__init__(parent=parent, **kwargs)
        self.spec: Optional[PlotSpec] = None
        if spec is not None:
            self.show_spec(spec)

    # ------------------------------------------------------------ drawing

    def show_spec(self, spec: PlotSpec) -> int:
        """Draw ``spec``. Returns the number of groups drawn."""
        self.spec = spec
        kind = str(spec.kind or spec.default_kind())
        self.spec = replace(spec, kind=kind)
        return self._draw()

    def show_as(self, kind: str) -> int:
        """Redraw as ``kind``. THE SAME SPEC, a different mark.

        :raises ValueError: for a kind this data cannot support. Drawing it
            anyway would answer a different question and say nothing --
            which is what a menu offering every kind on every figure does.
        """
        if self.spec is None:
            raise ValueError("this plot holds no data to redraw")
        from ...graph_types import fits, why_not

        shape = self.spec.shape()
        if not fits(shape, kind):
            raise ValueError(why_not(shape, kind))
        return self.show_spec(replace(self.spec, kind=kind))

    def _draw(self) -> int:
        from .fast_plots import colour_for

        spec = self.spec
        self._reset_scene()
        groups = spec.groups()
        if not groups:
            self.set_status("This table holds nothing to draw.")
            return 0

        mark = MARKS.get(str(spec.kind), "jitter_bar")
        # A SCATTER IS NOT A GROUPED MARK. It is two continuous axes, and
        # forcing it through `add_group_mark` would put every point at one
        # categorical position -- a jitter under another name, which is
        # exactly what `graph_types` refuses to offer.
        if str(spec.kind) in ("scatter", "line") and spec.group \
                and spec.group in getattr(spec.frame, "columns", ()) \
                and pd.api.types.is_numeric_dtype(spec.frame[spec.group]):
            return self._draw_xy(mark)

        labels = list(groups)
        for position, label in enumerate(labels):
            colour = spec.colours.get(label) or colour_for(position)
            self.add_group_mark(float(position), groups[label], mark,
                                colour=colour, seed=position)
        # THE n IS ON THE AXIS, not only in the caption. A three-point
        # group and a three-hundred-point group are the same bar, and the
        # label is the only place a reader meets the difference before they
        # have read the sentence underneath.
        self.plot.getAxis("bottom").setTicks(
            [[(position, f"{label}\n(n={len(groups[label]):,})")
              for position, label in enumerate(labels)]])
        self._label_axes(spec, categorical=True)
        self.set_status(self._caption(groups))
        return len(labels)

    def _draw_xy(self, mark: str) -> int:
        """Two continuous axes: a scatter, or a line through ordered x."""
        spec = self.spec
        frame = spec.frame.dropna(subset=[spec.group, spec.value])
        x = pd.to_numeric(frame[spec.group], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame[spec.value], errors="coerce").to_numpy(float)
        if mark == "line":
            import pyqtgraph as pg

            from .fast_plots import colour_for

            # SORTED BY x, because a line joins points in the order it is
            # given and an unsorted series draws a scribble. `graph_types`
            # only offers a line for an ordered x, and this is the other
            # half of that promise.
            order = np.argsort(x)
            self.plot.plot(x[order], y[order],
                           pen=pg.mkPen(colour_for(0), width=2))
        else:
            self.add_scatter(x, y)
        self._label_axes(spec, categorical=False)
        self.set_status(f"{len(x):,} point(s).")
        return 1

    def _label_axes(self, spec, *, categorical: bool) -> None:
        self.plot.setLabel("left", spec.y_label or spec.value)
        self.plot.setLabel(
            "bottom", spec.x_label or (spec.group if not categorical
                                       else spec.group or ""))
        if spec.title:
            self.plot.setTitle(spec.title)

    @staticmethod
    def _caption(groups) -> str:
        """n per group, always. A picture of groups whose sizes are not
        stated is a picture that hides three points among three hundred."""
        return "; ".join(f"{label} n={len(values):,}"
                         for label, values in groups.items())

    # -------------------------------------------------- what the menu asks

    def comparison_groups(self) -> Optional[dict]:
        """The groups, for the statistics in the saved folder.

        `FastPlot` returns None here because a bare plot has no groups; this
        one does, and that is what puts a real test in `statistics.csv`.
        """
        if self.spec is None:
            return None
        groups = self.spec.groups()
        return groups if len(groups) >= 2 else None

    def comparison_unit(self) -> str:
        return self.spec.unit if self.spec is not None else "observation"

    def export_settings(self) -> dict:
        out = super().export_settings()
        if self.spec is not None:
            out.update(kind=self.spec.kind, group=self.spec.group,
                       value=self.spec.value, unit=self.spec.unit,
                       shape=self.spec.shape())
        return out

    def frame(self):
        """The rows behind the picture, for `data.csv`."""
        if self.spec is not None and self.spec.frame is not None:
            return self.spec.frame
        return super().frame()
