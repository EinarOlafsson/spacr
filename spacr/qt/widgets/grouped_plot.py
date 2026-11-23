"""Render data-backed grouped comparisons with interchangeable graph types.

A :class:`PlotSpec` retains the source frame, group and value columns,
observation unit, and selected graph type. Retaining these data allows the
widget to redraw compatible graph types and export the figure together with
its source rows and statistical comparison.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .fast_plots import FastPlot, _CategoricalAxis
from ...graph_types import MARKS
from ..i18n import tr

LOG = logging.getLogger("spacr.qt.grouped_plot")

#: Map public graph-type names to marks accepted by
#: :meth:`~spacr.qt.widgets.fast_plots.FastPlot.add_group_mark`.



@dataclass
class PlotSpec:
    """Data and display metadata required to render a grouped plot.

    The retained frame supports graph-type compatibility checks, redrawing,
    data export, and statistical comparison without reconstructing the plot
    from rendered graphics.

    :param frame: the table holding the observations, normally a pandas
        DataFrame with ``value`` and optionally ``group`` columns; ``None``
        gives no groups.
    :param value: name of the column plotted on the value axis; its values are
        converted to numbers and non-numeric entries dropped.
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
        """The graph type used for this data when no kind is named."""
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
        """Create the grouped plot, optionally drawing a spec straight away.

        :param spec: the plot to draw; ``None`` leaves the plot empty.
        :param parent: parent widget, or ``None``.
        :param kwargs: passed through to the base plot.
        """
        super().__init__(parent=parent, **kwargs)
        self.spec: Optional[PlotSpec] = None
        if self.plots_available:
            self.plot.setAxisItems({"bottom": _CategoricalAxis(dpi=self.logicalDpiX())})
            self.plot.getAxis("bottom").setPen(self.plot.getAxis("left").pen())
            self.plot.getAxis("bottom").setTickPen(self.plot.getAxis("left").pen())
            self.apply_text_style()
        if spec is not None:
            self.show_spec(spec)


    def show_spec(self, spec: PlotSpec) -> int:
        """Draw ``spec``. Returns the number of groups drawn.

        :param spec: the data and display metadata to draw; an empty ``kind``
            is replaced by the default kind for the data's shape.
        """
        self.spec = spec
        kind = str(spec.kind or spec.default_kind())
        self.spec = replace(spec, kind=kind)
        return self._draw()

    def show_as(self, kind: str) -> int:
        """Redraw the retained data using graph type ``kind``.

        :param kind: graph-type name to draw; it must fit the retained data's
            shape (see :mod:`spacr.graph_types`).
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
        """Draw the spec and return how many groups were drawn.

        A scatter or line over a numeric grouping column is drawn as two
        continuous axes instead: forcing it through the grouped mark would put
        every point at one categorical position -- a jitter under another name,
        which the graph-type table refuses to offer.

        Finite observation counts select the shared automatic representation.
        Counts remain in the legend and status text rather than on axis ticks.

        :returns: the number of groups drawn; ``0`` when there was nothing.
        """
        from .fast_plots import colour_for

        spec = self.spec
        self._reset_scene()
        groups = spec.groups()
        if not groups:
            self.set_status("This table holds nothing to draw.")
            return 0

        mark = MARKS.get(str(spec.kind), "jitter_bar")
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
        if self.plots_available:
            self.plot.getAxis("bottom").set_categories(labels)
        self.plot.setXRange(-0.5, len(labels) - 0.5, padding=0.0)
        self._label_axes(spec, categorical=True)
        self.set_status(self._caption(groups))
        from ...figures.style import resolve_style

        controls = [index for index, label in enumerate(labels) if background and str(label) == background]
        self._spacr_style_spec = resolve_style("screen", "categorical", conditions=labels,
            sample_counts=[int(np.isfinite(values).sum()) for values in groups.values()],
            control=controls,
            identities={label: "highlight" if str(label) == subject else "data" for label in labels} if subject else None,
            comparison={"reference": background or None,
                                        "experimental_unit": spec.unit, "graph_type": spec.kind},
            legend={"caption": self._headline, "reference": background or None,
                    "quantities": {"x": spec.x_label or spec.group, "y": spec.y_label or spec.value},
                    "units": getattr(spec.frame, "attrs", {}).get("units") or {}})
        from ...figures.style import finalize_figure

        finalize_figure(self)
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
            return ROLES["control"]
        if not subject:
            return ROLES["primary"]
        return ROLES["highlight"] if str(label) == subject else ROLES["data"]

    def _draw_xy(self, mark: str) -> int:
        """Draw unchanged continuous coordinates with shared appearance and external semantics.

        The original numeric coercion, missing-row handling and line sorting
        remain unchanged. Only the data pen, declared units and line view
        origin use the shared presentation owner.

        :param mark: existing line or scatter rendering choice.
        :returns: one drawn continuous series.
        """
        spec = self.spec
        axis = self.plot.getAxis("bottom")
        if isinstance(axis, _CategoricalAxis):
            axis.clear_categories()
        frame = spec.frame.dropna(subset=[spec.group, spec.value])
        x = pd.to_numeric(frame[spec.group], errors="coerce").to_numpy(float)
        y = pd.to_numeric(frame[spec.value], errors="coerce").to_numpy(float)
        if mark == "line":
            import pyqtgraph as pg

            from .fast_plots import colour_for

            order = np.argsort(x)
            from ...figures.style import DATA_ALPHA, WEIGHTS

            artist = self.plot.plot(x[order], y[order],
                pen=pg.mkPen(colour_for(0, int(round(DATA_ALPHA*255))),
                             width=WEIGHTS["data"]*float(self.logicalDpiX())/72.))
        else:
            from ...figures.style import DATA_ALPHA
            from .fast_plots import colour_for

            artist = self.add_scatter(x, y,
                colours=[colour_for(0, int(round(DATA_ALPHA*255))) for _ in x])
        artist._spacr_condition_role = "data"
        self._label_axes(spec, categorical=False)
        self.set_status(tr("{count} point(s).").format(count=f"{len(x):,}"))
        from ...figures.style import resolve_style, finalize_figure, _line_origin_limits

        attrs = getattr(spec.frame, "attrs", {})
        reference = attrs.get("reference")
        comparison = {"reference": reference, "graph_type": spec.kind,
                      "experimental_unit": spec.unit}
        if attrs.get("cut_axis_reason"):
            comparison["cut_axis_reason"] = attrs["cut_axis_reason"]
        self._spacr_style_spec = resolve_style("screen", "continuous", comparison=comparison,
            legend={"caption": spec.title or self._headline, "reference": reference,
                    "quantities": {"x": spec.group, "y": spec.value},
                    "units": attrs.get("units") or {}})
        finite = y[np.isfinite(y)]
        if mark == "line" and len(finite) and not comparison.get("cut_axis_reason"):
            self.plot.getViewBox().updateAutoRange()
            limits = self.plot.getViewBox().viewRange()[1]
            low, high = _line_origin_limits(limits, (float(finite.min()), float(finite.max())))
            if low != high:
                self.plot.setYRange(low, high, padding=0.)
        finalize_figure(self)
        return 1

    def _label_axes(self, spec, *, categorical: bool) -> None:
        """Write the axis labels and the title from the spec.

        :param spec: the plot spec to read labels from.
        :param categorical: whether the horizontal axis is categorical, which
            decides what the fallback bottom label is.
        """
        self.plot.setLabel("left", spec.y_label or spec.value)
        self.plot.setLabel(
            "bottom", spec.x_label or (spec.group if not categorical
                                       else spec.group or ""))
        if spec.title:
            self.plot.setTitle(spec.title)
        self.apply_text_style()

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
        mark = MARKS.get(str(getattr(spec, "kind", "") or ""), "jitter_bar")
        if mark == "auto":
            from ...figures.spread import SPREAD_NONE, spread_label
            from ...figures.style import AUTO_N, DENSE_N

            caption = counts + " — " + tr("Means with every observation through twenty values; "
                "boxes with observations above twenty; violins without points from one thousand. "
                "Box whiskers show the fifth and ninety-fifth percentiles.")
            spread = str(getattr(spec, "spread", "") or SPREAD_NONE)
            if spread != SPREAD_NONE and any(1 < np.isfinite(values).sum() <= AUTO_N for values in groups.values()):
                caption += " " + tr("Bar whiskers: {spread}.").format(spread=spread_label(spread))
            if any(np.isfinite(values).sum() >= DENSE_N and np.ptp(values[np.isfinite(values)]) == 0
                   for values in groups.values()):
                caption += " " + tr("A zero-spread violin is a horizontal line at the shared value.")
            return caption
        if mark == "violin" and any(np.isfinite(values).any() and np.ptp(values[np.isfinite(values)]) == 0
                                     for values in groups.values()):
            return counts + " — " + tr("A zero-spread violin is a horizontal line at the shared value.")
        if mark not in ("bar", "jitter_bar"):
            return counts
        from ...figures.spread import SPREAD_NONE, spread_label

        spread = str(getattr(spec, "spread", "") or SPREAD_NONE)
        if spread == SPREAD_NONE:
            return counts
        said = spread_label(spread)
        return f"{counts} — {said}" if counts else said


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
        """What one point on this plot is -- a well, a field, an object.

        The unit is named rather than assumed, because it decides what a
        comparison means: the same data compared per well and per object
        gives different p-values, and which one is valid depends on the
        question asked.

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
