"""Render data-backed grouped comparisons with interchangeable graph types.

A :class:`PlotSpec` retains the source frame, group and value columns,
observation unit, and selected graph type. Retaining these data allows the
widget to redraw compatible graph types and export the figure together with
its source rows and statistical comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fast_plots import FastPlot

LOG = logging.getLogger("spacr.qt.grouped_plot")

#: Map public graph-type names to marks accepted by
#: :meth:`~spacr.qt.widgets.fast_plots.FastPlot.add_group_mark`.
MARKS: Dict[str, str] = {
    "bar": "bar",
    "bar_jitter": "jitter_bar",
    "box_jitter": "jitter_box",
    "jitter": "jitter",
    "box": "box",
    "violin": "violin",
    "line": "line",
    "scatter": "points",
}


@dataclass
class PlotSpec:
    """Data and display metadata required to render a grouped plot.

    The retained frame supports graph-type compatibility checks, redrawing,
    data export, and statistical comparison without reconstructing the plot
    from rendered graphics.
    """

    frame: Any
    value: str
    group: str = ""
    kind: str = ""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    #: Experimental unit represented by one observation, such as ``well``,
    #: ``cell``, or ``guide``. Exported statistics record this unit explicitly.
    unit: str = "observation"
    #: Colour per group, when the user has chosen one.
    colours: Dict[str, str] = field(default_factory=dict)
    #: The group the figure is ABOUT, if any.
    #:
    #: THE HOUSE STYLE IS ONE ARGUMENT, NOT A RAINBOW: "everything is grey
    #: except what the sentence is about; a box per group in a different
    #: colour is a rainbow, not an argument". Naming a group here paints it
    #: in the highlight role and everything else in the data grey. Left
    #: empty, the groups take the categorical scale -- because a figure that
    #: has not been told what it is about has no subject to single out, and
    #: greying every group equally would leave a picture with no ink in it.
    highlight: str = ""
    #: A group that is by definition NOT the argument.
    #:
    #: The residual population -- "the rest", the unselected wells, the
    #: background -- is what the coloured groups are being compared
    #: AGAINST, so it takes the data grey whether or not a subject has been
    #: named. Naming it is the other half of the same rule: the ink goes on
    #: the claim, and this is the thing the claim is measured against.
    background: str = ""
    #: What a bar's whisker MEANS, from
    #: :data:`spacr.figures.spread.SPREAD_CHOICES`.
    #:
    #: SD describes the observations, SEM the confidence in their mean, and
    #: at n=3000 they differ by a factor of fifty-five -- so a reader who
    #: assumes the wrong one reads a real effect as noise or noise as a real
    #: effect. The choice is the user's and the caption names it.
    spread: str = "sem"

    def shape(self) -> str:
        """The data shape, for deciding which kinds fit."""
        from ...graph_types import shape_of

        return shape_of(self.frame, self.group, self.value)

    def default_kind(self) -> str:
        """What this data is born as, when no kind is named."""
        from ...graph_types import default_for

        return default_for(self.shape())

    def groups(self) -> Dict[str, np.ndarray]:
        """Return ``{label: values}`` while preserving frame order.

        Preserving first occurrence keeps deliberately ordered experimental
        conditions, including controls, in their configured display order.
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

    :param spec: what to draw. ``None`` builds an empty plot; assign later
        through :meth:`show_spec` rather than by setting :attr:`spec`.
    :param parent: parent widget.
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
        """Redraw the retained data using graph type ``kind``.

        :raises ValueError: for a kind this data cannot support. Drawing it
            would otherwise imply a relationship unsupported by the available
            data shape.
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
        subject = str(spec.highlight or "")
        background = str(spec.background or "")
        for position, label in enumerate(labels):
            colour = spec.colours.get(label) or self._ink_for(
                label, position, subject, background)
            self.add_group_mark(float(position), groups[label], mark,
                                colour=colour, seed=position,
                                spread=str(getattr(spec, "spread", "sem")))
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

    @staticmethod
    def _ink_for(label: str, position: int, subject: str = "",
                 background: str = ""):
        """The colour one group's mark is drawn in.

        The house style is one argument, not a rainbow. With a subject
        named, that group takes the highlight role and every other group is
        the data grey. A background group is grey either way -- it is what
        the others are being compared against, so it is never the claim.
        With neither named there is no claim to make, and the categorical
        scale is what tells the groups apart.
        """
        from .fast_plots import colour_for

        try:
            from ...figures.style import ROLES
        except Exception:                                    # noqa: BLE001
            return colour_for(position)
        if background and str(label) == background:
            return ROLES["data"]
        if not subject:
            return colour_for(position)
        return ROLES["highlight"] if str(label) == subject else ROLES["data"]

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

    def _caption(self, groups) -> str:
        """The sample count for every group, and what the whisker means.

        AN ERROR BAR WITH AN UNNAMED SPREAD IS NOT READABLE. A reader cannot
        tell a SEM from an SD without being told, and the two differ by
        sqrt(n) -- so wherever a whisker is drawn the sentence under the plot
        says which quantity it is.
        """
        counts = "; ".join(f"{label} n={len(values):,}"
                           for label, values in groups.items())
        spec = self.spec
        # THE MARK THAT WAS DRAWN, not the kind that was asked for: an
        # unrecognised kind falls back to the bar, and a caption that read
        # the kind would leave that bar's whisker unnamed.
        mark = MARKS.get(str(getattr(spec, "kind", "") or ""), "jitter_bar")
        if mark not in ("bar", "jitter_bar"):
            return counts
        from ...figures.spread import SPREAD_NONE, spread_label

        spread = str(getattr(spec, "spread", "") or SPREAD_NONE)
        if spread == SPREAD_NONE:
            return counts
        said = spread_label(spread)
        return f"{counts} — {said}" if counts else said

    # -------------------------------------------------- what the menu asks

    def comparison_groups(self) -> Optional[dict]:
        """Return grouped values for export-time statistical comparison.

        Returns ``None`` unless the retained specification contains at least
        two groups.
        """
        if self.spec is None:
            return None
        groups = self.spec.groups()
        return groups if len(groups) >= 2 else None

    def comparison_unit(self) -> str:
        """What one point on this plot IS -- a well, a field, an object.

        NAMED RATHER THAN ASSUMED, because it decides what a comparison
        means: the same data compared per well and per object gives
        different p-values, and only one of them answers the question asked.

        :returns: the unit's name, ``observation`` when unspecified.
        """
        return self.spec.unit if self.spec is not None else "observation"

    def export_settings(self) -> dict:
        """The parent's export settings, plus this plot's grouping.

        :returns: the settings dict.
        """
        out = super().export_settings()
        if self.spec is not None:
            out.update(kind=self.spec.kind, group=self.spec.group,
                       value=self.spec.value, unit=self.spec.unit,
                       shape=self.spec.shape())
        return out

    def frame(self):
        """Return the source rows exported to ``data.csv``."""
        if self.spec is not None and self.spec.frame is not None:
            return self.spec.frame
        return super().frame()
