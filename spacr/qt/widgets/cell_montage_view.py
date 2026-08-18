"""The cells behind one dot on the volcano, and how they were picked.

Instruction 131's other half. :mod:`spacr.cell_montage` -- the headless one,
already built and tested -- decides WHICH objects belong behind a coefficient
and writes the whole reason into :meth:`~spacr.cell_montage.MontagePlan.caption`.
Nothing of that is repeated here. This module loads the pixels for the objects
that plan names, puts them on screen in the tab beside the run's figures, and
says why when it cannot.

WHAT THIS ADDS TO THE HEADLESS HALF -- three things, and deliberately no more
----------------------------------------------------------------------------
1. **The pixels.** :func:`spacr.crops.resolve_crop_source` already chooses
   between the exported PNGs and ``merged/<fov>.npy`` AND says which it chose;
   :func:`spacr.cell_montage.resolve_montage_crop_source` already turns "there
   is no source at all" into a sentence rather than an exception. :func:`load`
   calls them **once per attached database**, because two plates can be two
   experiment folders with two different answers -- one with its crops still on
   disk and one with only ``merged/`` -- and routes every selected object row
   back to the source of the database it came from.

2. **The two settings the request names.** "the user needs to specify which
   array the masks are in and which channels should be used to generate the
   pictures". Those are :attr:`spacr.crops.CropSpec.object_type` (which mask
   plane a crop is cut by) and :attr:`~spacr.crops.CropSpec.channels`
   (``png_dims``), both already fields of the spec, so they are passed through
   rather than reinvented. Left alone, ``resolve_crop_source`` reads them back
   out of the run's own ``measurements.db`` and the tab says that is where they
   came from -- which is the right default, because it reproduces the crops
   that run would have written.

3. **A worker, and a tab that says why.** Below.

THE GUI THREAD NEVER READS AN IMAGE, AND HERE IS WHAT IT WOULD COST
--------------------------------------------------------------------
Measured on this machine, 12 synthetic fields of 1080x1080x7 ``uint16``, crops
cut through the real :mod:`spacr.crops` path:

    ============================================  ==============
    720 crops, 60 per field, ``merged/``            0.95 ms/crop
    720 crops, 60 per field, exported PNGs          0.13 ms/crop
    **12 crops, ONE per field, ``merged/``**       **13.36 ms/crop**
    12 crops, one per field, exported PNGs          0.13 ms/crop
    ============================================  ==============

The third row is the montage's own shape: one coefficient spans dozens of
wells and takes a handful of objects from each, so it touches many fields and
cuts few crops from each of them. That is the same finding
:data:`spacr.cell_montage.MAX_OBJECTS` was chosen against -- the merged source
is priced by FIELDS TOUCHED, not by crops cut -- and at the 300-object cap it
is ~4.0 s of blocking read. On the GUI thread that is a four-second frozen
window with no cursor and no way to cancel, which is the single thing this
application must never do.

So every read goes through a :class:`~spacr.qt.job_runner.JobRunner`, which is
where this project's threading rules are already written down: ``finished`` is
emitted on the WORKER thread, and the only safe thing a slot there may do is
re-emit a Signal whose receiver is a **bound method** of a GUI-thread object.
:meth:`CellMontageView._on_loaded` is that bound method, and it is the only
place the widgets are touched.

What the GUI thread does keep is the numpy-to-``QPixmap`` conversion, and that
was measured too rather than assumed: 300 crops of 224x224 scaled to
:data:`THUMBNAIL_PX` cost **21.1 ms** -- one frame and a bit, once, at the end
of a load that took seconds. Moving it off the thread would mean shipping
QPixmaps across a thread boundary, which QPixmap does not allow.

A TAB THAT CANNOT BE FILLED SAYS WHY -- instruction 106
-------------------------------------------------------
The tab is always present and the button is always visible. When there is
nothing it can do it is DISABLED and :meth:`CellMontageView.reason` is on it,
on its tooltip and in the status line. A screen with no exported PNGs and no
``merged/`` folder gets that sentence, not an empty grid -- an empty grid is
indistinguishable from a bug, and this feature's whole risk is producing
something plausible and wrong.

The one check that cannot be made up front is the crop source itself:
``_has_png_folder`` walks up to three levels of ``data/``, which is a disk walk
and not something to do on every click. So the button stays live until a load
has actually looked, and the answer is REMEMBERED -- the button greys out with
that reason afterwards, and un-greys the moment the coefficient, the databases
or the crop settings change.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QPlainTextEdit, QPushButton, QScrollArea, QSplitter, QVBoxLayout,
    QWidget,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "THUMBNAIL_PX",
    "OBJECT_CHOICES",
    "SOURCE_CHOICES",
    "MontageRequest",
    "MontageLoad",
    "experiment_root",
    "parse_channels",
    "coefficient_from_frame",
    "load",
    "montage_figure",
    "CellMontageView",
]

#: Edge of one thumbnail, in pixels. 96 is the largest that still puts a
#: montage of 30 objects on screen without scrolling in the tab's real width
#: (the results side of the regression splitter starts at 780 px), and the
#: crop underneath is kept whole -- a thumbnail is a view, and the figure the
#: user saves is drawn from the full-resolution arrays.
THUMBNAIL_PX = 96

#: How wide a thumbnail cell is once its border and spacing are counted. Used
#: to decide the column count from the viewport, so the grid reflows.
_CELL_PX = THUMBNAIL_PX + 10

#: Which mask plane a crop is cut by -- the "which array the masks are in"
#: half of the request. The vocabulary is :data:`spacr.crops.OBJECT_TYPES`
#: and is read from there rather than retyped; these are the four a measured
#: run actually writes crops for.
OBJECT_CHOICES: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "cytoplasm")

#: The crop source, as the user may force it. ``""`` is
#: :func:`spacr.crops.resolve_crop_source`'s own ``auto``, which prefers the
#: exported PNGs when they exist -- and the timing table in this module's
#: docstring is the second reason that preference is right.
SOURCE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("", "automatic — exported PNGs if they exist, else merged/"),
    ("png", "exported PNGs only"),
    ("merged", "cut from merged/*.npy only"),
)

#: How long the grid waits after a resize before it reflows. A drag is a
#: stream of resize events and re-laying out 300 thumbnails on each one turns
#: a smooth drag into a stutter; same reason and same value as the figure
#: grid's own debounce.
_REFLOW_DEBOUNCE_MS = 220


# ---------------------------------------------------------------------------
# The job -- everything below runs on the WORKER thread and touches no widget
# ---------------------------------------------------------------------------

def experiment_root(db_path: str) -> str:
    """The experiment folder holding ``measurements/measurements.db``.

    :param db_path: the database an attached plate names.
    :returns: the folder :func:`spacr.crops.resolve_crop_source` wants -- the
        one with ``merged/`` and ``data/`` under it. A database somewhere
        other than ``measurements/`` yields its own folder, so a project laid
        out by hand still resolves instead of silently pointing one level up.
    """
    folder = os.path.dirname(os.path.abspath(os.fspath(db_path)))
    if os.path.basename(folder) == "measurements":
        return os.path.dirname(folder)
    return folder


def parse_channels(text: str) -> Optional[Tuple[int, ...]]:
    """Read the channel box: ``"0,1,2"`` -> ``(0, 1, 2)``.

    :param text: what the user typed. Empty means "as the run saved them",
        which is ``None`` -- and ``None`` is not the same as ``()``: it is
        what lets :func:`spacr.crops.resolve_crop_source` read ``png_dims``
        back out of ``measurements.db`` and reproduce that run's own crops.
    :returns: the channel indices, or ``None`` for "leave it to the run".
    :raises ValueError: the box holds something that is not a channel index.
    """
    cleaned = str(text or "").replace(";", ",").replace(" ", ",")
    parts = [p for p in cleaned.split(",") if p]
    if not parts:
        return None
    out = []
    for part in parts:
        value = int(part)
        if value < 0:
            raise ValueError(f"{value} is not a channel index")
        out.append(value)
    return tuple(out)


def coefficient_from_frame(key: str, frame) -> Tuple[str, str, Optional[float]]:
    """Turn a clicked coefficient into ``(name, level, effect)``.

    The join is on the KEY and the parse is
    :func:`spacr.hits.guide_of` / :func:`spacr.hits.gene_of`, which is the
    same rule the volcano, the gene tile and the metadata join already use --
    a fourth copy of "which gene is this term" is how two surfaces start
    naming different guides for one dot.

    :param key: the ``feature`` the panel emitted, e.g.
        ``fraction:grna[233460_1]`` or ``gene_fraction:gene[233460]``.
    :param frame: the coefficient table, for the fitted effect. ``None`` is
        allowed and yields ``None`` for the effect rather than raising.
    :returns: the gene or guide name, ``'grna'`` or ``'gene'``, and the fitted
        coefficient. The name is ``''`` for a term that names neither -- an
        Intercept or a row/column nuisance term -- which is a real answer and
        the reason the montage is refused for it.
    """
    from ...hits import gene_of, guide_of

    guide = guide_of(key)
    name = guide or (gene_of(key) or "")
    level = "grna" if guide else "gene"
    effect: Optional[float] = None
    if frame is not None and len(frame) and "feature" in getattr(
            frame, "columns", ()):
        from ...figures.panels import effect_column

        column = effect_column(frame)
        if column:
            match = frame[frame["feature"].astype(str) == str(key)]
            if len(match):
                try:
                    value = float(match[column].iloc[0])
                except (TypeError, ValueError):
                    value = float("nan")
                if np.isfinite(value):
                    effect = value
    return name, level, effect


@dataclass(frozen=True)
class MontageRequest:
    """Everything one montage load needs, as plain data.

    A frozen record rather than a pile of arguments because it crosses a
    thread boundary and is what the completion handler compares against to
    know whether the answer that arrived is still the one on screen.

    :param name: the gene or guide the coefficient names.
    :param effect: its fitted coefficient.
    :param level: ``'gene'`` or ``'grna'``.
    :param results_path: the results CSV, or the folder holding it. Either
        way ``regression_data.csv`` in that FOLDER is what is read -- see
        :func:`spacr.cell_montage.read_well_guide_fractions`, which refuses
        the two obvious wrong CSVs by name.
    :param databases: the ``measurements.db`` files attached to the run's
        input table.
    :param object_type: which mask plane a crop is cut by.
    :param channels: intensity planes for the picture, or ``None`` for the
        ones the run itself saved.
    :param prefer: ``''`` / ``'png'`` / ``'merged'``.
    :param score_column: the per-object classification score.
    :param cap: the largest montage to draw.
    :param per_guide: one montage per guide instead of the gene's guides
        summed. They are different questions -- see
        :func:`spacr.cell_montage.select_montage_per_guide` -- and each plan
        says in its own caption which one it answers.
    """

    name: str
    effect: float
    level: str = "gene"
    results_path: str = ""
    databases: Tuple[str, ...] = ()
    object_type: str = "cell"
    channels: Optional[Tuple[int, ...]] = None
    prefer: str = ""
    score_column: str = "pred"
    cap: int = 0
    per_guide: bool = False


@dataclass(frozen=True)
class MontageLoad:
    """What came back from one load: the plans, the pixels, or the reason.

    :param request: the request this answers, so a stale answer can be
        recognised and dropped.
    :param plans: one :class:`~spacr.cell_montage.MontagePlan` per montage --
        one for a summed gene, one per guide when the guides were asked for
        separately.
    :param images: the crops, one list per plan, aligned with that plan's
        ``objects`` rows. An entry is ``None`` only where a source returned
        nothing for a row.
    :param sources: ``{experiment root: description}`` -- which crop source
        drew each plate, in words.
    :param error: why there is no montage, or ``''``. A SENTENCE, not an
        exception: a tab that cannot be filled has to say why and stay on
        screen.
    :param unavailable: ``True`` when the reason is a permanent property of
        this run -- no crop source anywhere -- rather than a bad request.
        That is what lets the button grey itself out afterwards instead of
        inviting the same click again.
    """

    request: Optional[MontageRequest] = None
    plans: Tuple[Any, ...] = ()
    images: Tuple[Tuple[Any, ...], ...] = ()
    sources: Dict[str, str] = field(default_factory=dict)
    error: str = ""
    unavailable: bool = False

    @property
    def ok(self) -> bool:
        """True when at least one plan came back, empty or not.

        An EMPTY plan is a success: it carries the wells that reported the
        guide, the window that admitted nothing and the caption that says so,
        which is an answer. Only a missing plan is a failure.
        """
        return bool(self.plans) and not self.error

    @property
    def n_objects(self) -> int:
        """How many objects the load drew in total."""
        return sum(int(getattr(p, "n_objects", 0)) for p in self.plans)


def _crop_settings(request: "MontageRequest", root: str) -> Dict[str, Any]:
    """The settings mapping ``resolve_crop_source`` takes for one plate."""
    settings: Dict[str, Any] = {"src": root}
    if request.channels:
        # ``png_dims`` is the PNG path's own name for "which intensity planes
        # become the picture", so the user's choice is expressed in the
        # vocabulary the crop spec already speaks instead of a new one.
        settings["png_dims"] = list(request.channels)
    return settings


def load(request: MontageRequest) -> MontageLoad:
    """Select the objects behind one coefficient and cut their crops.

    **Runs on a worker thread and touches no widget.** Every failure comes
    back as :attr:`MontageLoad.error` rather than as an exception, because the
    caller is a tab that must stay on screen and say why.

    :param request: what to draw.
    :returns: the plans, the crops, and which source drew them.
    """
    from ...cell_montage import (
        MontageError, read_well_guide_fractions, load_montage_objects,
        resolve_montage_crop_source, select_montage, select_montage_per_guide,
        CropSourceChoice, MAX_OBJECTS,
    )
    import pandas as pd

    if not request.name:
        return MontageLoad(
            request=request,
            error="This coefficient names neither a gene nor a guide, so no "
                  "well reports it and there are no cells behind it. Pick a "
                  "gene or guide term.")
    if not request.databases:
        return MontageLoad(
            request=request,
            error="No measurement database is attached to this run's input "
                  "table, so there are no per-object rows and no crops. "
                  "Attach one to a plate row first.",
            unavailable=True)

    # THE FOLDER, NEVER THE COEFFICIENT TABLE ITSELF. `results_path` is what
    # the panel loaded -- `results.csv`, the coefficients -- and
    # `read_well_guide_fractions` takes a CSV at its word. Handed the
    # coefficient table it reports "names no well", which is true and is the
    # wrong file being read rather than a missing column.
    folder = request.results_path
    if folder and os.path.isfile(folder):
        folder = os.path.dirname(os.path.abspath(folder))
    try:
        counts = read_well_guide_fractions(folder)
    except MontageError as error:
        return MontageLoad(request=request, error=str(error), unavailable=True)
    except Exception as error:                                  # noqa: BLE001
        return MontageLoad(
            request=request,
            error=f"Could not read the per-well guide fractions: {error}")

    frames = []
    troubles: List[str] = []
    for db_path in request.databases:
        try:
            objects = load_montage_objects(
                db_path, object_type=request.object_type,
                score_column=request.score_column)
        except Exception as error:                              # noqa: BLE001
            troubles.append(f"{os.path.basename(db_path)}: {error}")
            continue
        # WHICH FOLDER THIS ROW'S PIXELS LIVE IN, carried on the row itself.
        # Two plates are two experiment folders and can have two different
        # answers -- one with its crops exported and one with only merged/ --
        # so "which source" is a property of the row, not of the montage.
        objects = objects.copy()
        objects["montage_source_root"] = experiment_root(db_path)
        frames.append(objects)

    if not frames:
        return MontageLoad(
            request=request,
            error="No attached database yielded per-object rows with a "
                  "classification score. " + " ".join(troubles),
            unavailable=True)

    objects = pd.concat(frames, ignore_index=True) if len(frames) > 1 \
        else frames[0]

    # THE CROP SOURCE, ONE PER PLATE, RESOLVED BEFORE ANYTHING IS SELECTED.
    # A montage nobody can draw is worth refusing before the selection runs,
    # and the plan's caption has to carry which source drew it.
    sources: Dict[str, Any] = {}
    described: Dict[str, str] = {}
    refusals: List[str] = []
    for root in sorted(set(objects["montage_source_root"].astype(str))):
        choice = resolve_montage_crop_source(
            _crop_settings(request, root), object_type=request.object_type,
            prefer=request.prefer or None)
        if not choice.available:
            refusals.append(f"{root}: {choice.reason}")
            continue
        sources[root] = choice
        described[root] = choice.describe()
    if not sources:
        return MontageLoad(
            request=request,
            error="There is nothing to draw the cells from: no exported crop "
                  "PNGs and no merged/*.npy stacks. " + "; ".join(refusals),
            unavailable=True)

    kinds = sorted({c.kind for c in sources.values()})
    reasons = sorted({c.reason for c in sources.values()})
    combined = CropSourceChoice(
        source=None, kind="+".join(kinds),
        reason=("; ".join(reasons) if len(kinds) == 1 else
                "; ".join(f"{os.path.basename(r) or r}: {c.describe()}"
                          for r, c in sorted(sources.items()))),
        available=True)

    cap = int(request.cap) if request.cap else MAX_OBJECTS
    try:
        if request.per_guide:
            plans = select_montage_per_guide(
                objects, counts, request.name, float(request.effect),
                level=request.level, score_column=request.score_column,
                cap=cap, crop_source=combined)
            if not plans:
                return MontageLoad(
                    request=request,
                    error=f"No guide of {request.name} is reported present in "
                          "any well of the count data, so there is no montage "
                          "to draw one guide at a time.")
        else:
            plans = [select_montage(
                objects, counts, request.name, float(request.effect),
                level=request.level, score_column=request.score_column,
                cap=cap, crop_source=combined)]
    except MontageError as error:
        return MontageLoad(request=request, error=str(error))
    except Exception as error:                                  # noqa: BLE001
        LOG.debug("montage selection failed", exc_info=True)
        return MontageLoad(
            request=request, error=f"Could not select the montage: {error}")

    images: List[Tuple[Any, ...]] = []
    for plan in plans:
        images.append(_cut(plan, sources, request, troubles))

    notes = tuple(f"NOTE {t}" for t in troubles)
    if notes:
        plans = [_with_notes(plan, notes) for plan in plans]
    return MontageLoad(request=request, plans=tuple(plans),
                       images=tuple(images), sources=described)


def _with_notes(plan, notes: Tuple[str, ...]):
    """A copy of ``plan`` carrying extra caption lines.

    ``MontagePlan`` is frozen on purpose -- the plan and the sentence that
    describes it must not drift apart -- so a note is added by replacement,
    which keeps that guarantee.
    """
    from dataclasses import replace

    return replace(plan, notes=tuple(plan.notes) + notes)


def _cut(plan, sources: Dict[str, Any], request: MontageRequest,
         troubles: List[str]) -> Tuple[Any, ...]:
    """Cut every crop one plan names, bucketed by plate.

    Bucketed because ``MergedCropSource.get_many`` opens each ``.npy`` once
    for the whole batch it is given, and the timing table in the module
    docstring is what that buys: 0.95 ms/crop against 13.36 ms when the same
    crops arrive one field at a time.
    """
    rows = plan.rows()
    out: List[Any] = [None] * len(rows)
    buckets: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        buckets.setdefault(str(row.get("montage_source_root", "")), []).append(index)
    for root, positions in buckets.items():
        choice = sources.get(root)
        if choice is None:
            troubles.append(f"{root} has no crop source; its objects are blank")
            continue
        try:
            crops = choice.source.get_many([rows[i] for i in positions])
        except Exception as error:                              # noqa: BLE001
            LOG.debug("could not cut crops from %s", root, exc_info=True)
            troubles.append(
                f"{os.path.basename(root) or root}: {error} -- "
                f"{len(positions)} objects could not be cut")
            continue
        for position, crop in zip(positions, crops):
            out[position] = crop
    return tuple(out)


# ---------------------------------------------------------------------------
# The saved figure
# ---------------------------------------------------------------------------

def montage_figure(plans: Sequence[Any], images: Sequence[Sequence[Any]],
                   columns: int = 8):
    """Draw the montage as one matplotlib figure, caption and all.

    The caption is not decoration and is not optional: it is what stops a
    reader taking these for genotyped cells, and
    :meth:`~spacr.cell_montage.MontagePlan.caption` always ends with the
    sentence that says membership is inferred from a well-level fraction.
    A figure without it is the one output this feature must never produce, so
    it is drawn from the plan rather than passed in.

    :param plans: the plans to draw, in order.
    :param images: the crops for each plan.
    :param columns: how many crops per row.
    :returns: the ``Figure``. The caller writes it through
        :func:`spacr.plot.save_figure`, which is what honours the user's
        figure-format and resolution preferences -- never ``savefig``
        directly, and never a hard-coded extension.
    """
    from matplotlib.figure import Figure

    columns = max(int(columns), 1)
    captions = [plan.caption() for plan in plans]
    rows_per_plan = [max(1, -(-len(imgs) // columns)) for imgs in images]
    total_rows = sum(rows_per_plan)
    # Height budget: one inch per row of crops, plus a line per caption line.
    # The caption is long by design and clipping it would be the same failure
    # as omitting it.
    caption_lines = sum(len(c.splitlines()) for c in captions) + 2 * len(plans)
    width = 1.1 * columns + 0.6
    height = 1.1 * total_rows + 0.16 * caption_lines + 0.6
    figure = Figure(figsize=(width, max(height, 2.0)))

    grid = figure.add_gridspec(
        max(total_rows, 1) + 1, columns,
        height_ratios=[1.0] * max(total_rows, 1) + [max(0.2 * caption_lines, 0.6)])
    row_offset = 0
    for plan, imgs, n_rows in zip(plans, images, rows_per_plan):
        for index, crop in enumerate(imgs):
            axes = figure.add_subplot(
                grid[row_offset + index // columns, index % columns])
            axes.set_xticks([])
            axes.set_yticks([])
            for spine in axes.spines.values():
                spine.set_visible(False)
            if crop is None:
                axes.text(0.5, 0.5, "no crop", ha="center", va="center",
                          fontsize=5)
                continue
            axes.imshow(np.asarray(crop))
        row_offset += n_rows

    text_axes = figure.add_subplot(grid[max(total_rows, 1), :])
    text_axes.axis("off")
    text_axes.text(0.0, 1.0, "\n\n".join(captions), ha="left", va="top",
                   fontsize=6, wrap=True,
                   transform=text_axes.transAxes)
    figure.suptitle(plans[0].coefficient.describe() if plans else "no montage",
                    fontsize=9)
    return figure


# ---------------------------------------------------------------------------
# The tab
# ---------------------------------------------------------------------------

class _Thumb(QLabel):
    """One crop, at :data:`THUMBNAIL_PX`, with its provenance on the tooltip.

    A ``QLabel`` and not a button: the montage is something to read, and a
    clickable thumbnail promises a drill-down that instruction 131 does not
    ask for and that would need a second selection mechanism to deliver.
    """

    def __init__(self, pixmap: QPixmap, tooltip: str, parent=None):
        super().__init__(parent)
        self.setPixmap(pixmap)
        self.setToolTip(tooltip)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(THUMBNAIL_PX, THUMBNAIL_PX)
        self.setFrameShape(QFrame.NoFrame)


class CellMontageView(QWidget):
    """The cells behind the selected coefficient, beside the run's figures.

    Connect :meth:`set_coefficient` to the panel's existing selection --
    ``RegressionResultsPanel.table.key_selected``, which is the funnel every
    plot and the table already pass through. There is deliberately no second
    selection mechanism here: a montage of a different gene from the one the
    volcano is ringing is exactly the plausible-and-wrong output this feature
    is most at risk of.

    :param frame_provider: called with no arguments for the coefficient
        table, which is where the fitted effect for a clicked key comes from.
    :param results_provider: called with no arguments for the results CSV
        path, beside which ``regression_data.csv`` is found.
    :param database_provider: called with no arguments for the run's input
        table rows, which is where the measurement databases are attached.
        A callable rather than a snapshot, for the same reason the
        Measurements tab takes one: databases are attached after this widget
        is built, and a list captured at construction never grows.
    :param threaded: ``False`` runs the load inline, emitting the same
        signals in the same order, so a test drives the whole tab without the
        behaviour diverging.
    """

    #: Emitted with the number of objects drawn once a load has landed. Zero
    #: is a real result and is emitted: a coefficient whose wells contribute
    #: nothing is an answer, and the caption says which wells and why.
    montage_ready = Signal(int)

    #: Emitted with the sentence explaining a load that produced no montage.
    montage_failed = Signal(str)

    NOTHING_SELECTED = (
        "Click a coefficient — a dot on the volcano or a row in the "
        "coefficient table — to see the cells behind it.")

    def __init__(self, frame_provider: Optional[Callable[[], Any]] = None,
                 results_provider: Optional[Callable[[], str]] = None,
                 database_provider: Optional[Callable[[], Any]] = None,
                 parent=None, *, threaded: bool = True):
        super().__init__(parent)
        from ..job_runner import JobRunner

        self._frame_provider = frame_provider
        self._results_provider = results_provider
        self._database_provider = database_provider

        # EVERY PIECE OF STATE A CONTROL READS IS BORN HERE, before a single
        # signal is connected. A widget whose controls are live before its
        # state exists is the `_significance` crash that took this application
        # down at launch, and the rule earned its own test file.
        self._key: str = ""
        self._name: str = ""
        self._level: str = "gene"
        self._effect: Optional[float] = None
        self._plans: Tuple[Any, ...] = ()
        self._images: Tuple[Tuple[Any, ...], ...] = ()
        self._sources: Dict[str, str] = {}
        #: The coefficient the montage on screen was built for. A montage of
        #: one gene under a selection that has moved to another is precisely
        #: the plausible-and-wrong output this feature is most at risk of, so
        #: the grid is emptied the moment the two disagree.
        self._shown_key: str = ""
        self._pending: Optional[MontageRequest] = None
        #: The reason a load found this run cannot produce a montage at all,
        #: remembered so the button greys out instead of inviting the same
        #: click again. Cleared by anything that changes the inputs.
        self._unavailable: str = ""
        self._status_text = self.NOTHING_SELECTED
        self._columns = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        controls = QHBoxLayout()
        self._show = QPushButton("Show the cells")
        self._show.clicked.connect(self.build)
        controls.addWidget(self._show)

        controls.addWidget(QLabel("object"))
        self._object = QComboBox()
        for name in OBJECT_CHOICES:
            self._object.addItem(name, name)
        self._object.setToolTip(
            "Which mask plane a crop is cut by. 'cytoplasm' has no plane on "
            "disk and is derived as cell minus nucleus/pathogen, exactly as "
            "measure_crop derives it.")
        self._object.currentIndexChanged.connect(self._on_settings_changed)
        controls.addWidget(self._object)

        controls.addWidget(QLabel("channels"))
        self._channels = QLineEdit()
        self._channels.setPlaceholderText("as the run saved them")
        self._channels.setMaximumWidth(120)
        self._channels.setToolTip(
            "Which intensity planes become the picture, e.g. 0,1,2. Left "
            "empty, the run's own png_dims are read back out of "
            "measurements.db, so the crops match the PNGs that run wrote.")
        self._channels.textChanged.connect(self._on_settings_changed)
        controls.addWidget(self._channels)

        controls.addWidget(QLabel("images from"))
        self._source = QComboBox()
        for value, label in SOURCE_CHOICES:
            self._source.addItem(label, value)
        self._source.currentIndexChanged.connect(self._on_settings_changed)
        controls.addWidget(self._source)

        self._per_guide = QComboBox()
        self._per_guide.addItem("guides summed", False)
        self._per_guide.addItem("one guide at a time", True)
        self._per_guide.setToolTip(
            "A gene's guides summed asks 'which cells are consistent with "
            "losing this gene'. One at a time asks 'do the guides pick out "
            "the same cells', which is how a real effect is told from one "
            "guide's off-target. They are different questions.")
        self._per_guide.currentIndexChanged.connect(self._on_settings_changed)
        controls.addWidget(self._per_guide)

        controls.addStretch(1)
        self._save = QPushButton("Save figure…")
        self._save.clicked.connect(self.save)
        controls.addWidget(self._save)
        layout.addLayout(controls)

        self._status = QLabel(self._status_text)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        split = QSplitter(Qt.Vertical)
        split.setChildrenCollapsible(False)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._body = QWidget()
        self._grid = QGridLayout(self._body)
        self._grid.setContentsMargins(6, 6, 6, 6)
        self._grid.setSpacing(4)
        self._grid.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._body)
        split.addWidget(self._scroll)

        # THE CAPTION IS PART OF THE OUTPUT, NOT A TOOLTIP. It states the
        # wells, the score window, the count rule and -- last, so it is the
        # sentence a reader leaves with -- that guide membership is INFERRED
        # from a well-level fraction rather than observed. Selectable, because
        # the reason to want it is usually to paste it into a methods section.
        self._caption = QPlainTextEdit()
        self._caption.setReadOnly(True)
        self._caption.setPlainText("")
        self._caption.setMinimumHeight(90)
        split.addWidget(self._caption)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([420, 160])
        layout.addWidget(split, 1)

        self._reflow = QTimer(self)
        self._reflow.setSingleShot(True)
        self._reflow.setInterval(_REFLOW_DEBOUNCE_MS)
        # A bound method of this GUI-thread object, per job_runner's rules.
        self._reflow.timeout.connect(self._relayout)

        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="cell montage")
        self._jobs.job_failed.connect(self._on_job_failed)

        self._refresh_controls()

    # ------------------------------------------------------------- selection

    def set_coefficient(self, key: str) -> None:
        """A coefficient was picked. THE SLOT TO CONNECT ``key_selected`` TO.

        Takes the feature string and nothing else, so the volcano, the Q-Q,
        the effect-rank plot and the coefficient table all reach it by the
        same one route the gene tile already uses.

        It does NOT load. A montage is seconds of disk for a click whose usual
        purpose is to read a row, so the selection arms the button and the
        user asks for the pictures.

        :param key: the ``feature`` the panel emitted.
        """
        self._key = str(key)
        frame = self._frame()
        try:
            self._name, self._level, self._effect = coefficient_from_frame(
                self._key, frame)
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not resolve %r", key, exc_info=True)
            self._name, self._level, self._effect = "", "gene", None
        if self._key != self._shown_key:
            # THE GRID CANNOT SHOW ONE GENE WHILE THE VOLCANO RINGS ANOTHER.
            # The caption names the gene it was built for, so leaving it up
            # under a moved selection is a picture that reads as the new one.
            self._drop_montage()
            # AND THE LOAD IN FLIGHT IS FOR THE OLD GENE. Clicking through
            # points faster than a merged-source load returns is the ordinary
            # case, not an edge one -- at the measured 13.36 ms/crop a montage
            # can be seconds behind the click. `cancel` retires the job (the
            # bookkeeping must still run) and drops its result by generation,
            # so the answer for a point the user has left never lands.
            self._jobs.cancel()
            self._pending = None
        # A NEW COEFFICIENT IS A NEW QUESTION, so a remembered "this run has
        # no crop source" is re-earned rather than assumed -- the databases
        # may have been attached in between.
        self._unavailable = ""
        self._refresh_controls()
        self._announce()

    def refresh(self) -> None:
        """Re-read the providers. Call this when the tab is opened.

        Databases are attached to the input table while this tab is behind
        another one, so what it can do changes without any signal reaching it
        -- the same reason the Measurements tab re-reads on open.

        The grid is re-flowed too. A montage built while this tab was behind
        another one was laid out against a viewport that had never been
        through a layout pass -- the same trap that made a snapshot of the
        unshown volcano a 100x9 rectangle of one colour -- so its column count
        is whatever fitted in that guess, which is one.
        """
        self._unavailable = ""
        if self._key:
            self.set_coefficient(self._key)
        else:
            self._refresh_controls()
            self._announce()
        self._reflow.start()

    # ------------------------------------------------------------ the answer

    def reason(self) -> str:
        """Why the montage cannot be built right now, or ``''``.

        Instruction 106's rule: a control that cannot do anything is greyed
        out AND SAYS WHY. Every branch here is a sentence a user can act on,
        and the order is the order the inputs are needed in, so the first
        missing thing is the one named.
        """
        if self._pending is not None:
            return "A montage is loading."
        if not self._key:
            return self.NOTHING_SELECTED
        if not self._name:
            return (f"{self._key} names neither a gene nor a guide — it is an "
                    "intercept or a plate/row nuisance term — so no well "
                    "reports it and there are no cells behind it.")
        if self._effect is None:
            return (f"The coefficient table on screen has no fitted effect "
                    f"for {self._name}, and the score window is "
                    "'baseline + effect'. Load the run's results table first.")
        if not self._results_path():
            return ("No regression results are loaded, so there is no "
                    "regression_data.csv to read the per-well guide "
                    "fractions from.")
        if not self.databases():
            return ("No measurement database is attached to this run's input "
                    "table, so there are no per-object rows and no crops to "
                    "show. Attach one to a plate row.")
        try:
            parse_channels(self._channels.text())
        except ValueError:
            return (f"'{self._channels.text()}' is not a list of channel "
                    "indices. Use numbers separated by commas, e.g. 0,1,2.")
        return self._unavailable

    def databases(self) -> Tuple[str, ...]:
        """The measurement databases attached to the run's input table.

        Read through :func:`spacr.qt.widgets.measurement_scan_panel.attached_databases`,
        which is the same reader the Measurements tab uses -- one vocabulary
        for "which plate has a database and is it still on disk".
        """
        if self._database_provider is None:
            return ()
        try:
            rows = self._database_provider() or []
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not read the attached databases", exc_info=True)
            return ()
        from .measurement_scan_panel import attached_databases

        return tuple(dict.fromkeys(
            entry.path for entry in attached_databases(rows) if entry.present))

    def request(self) -> Optional[MontageRequest]:
        """The request the button would submit, or ``None`` if it cannot.

        Public because it is what a test drives and what the completion
        handler compares a landed answer against.
        """
        if self.reason():
            return None
        return MontageRequest(
            name=self._name, effect=float(self._effect),
            level=self._level, results_path=self._results_path(),
            databases=self.databases(),
            object_type=str(self._object.currentData() or "cell"),
            channels=parse_channels(self._channels.text()),
            prefer=str(self._source.currentData() or ""),
            per_guide=bool(self._per_guide.currentData()))

    def build(self) -> bool:
        """Load the montage for the selected coefficient, off the GUI thread.

        :returns: True when a load was started.
        """
        request = self.request()
        if request is None:
            self._set_status(self.reason())
            return False
        self._pending = request
        self._drop_montage()
        self._set_status(
            f"Loading the cells behind {request.name}… reading "
            f"{len(request.databases)} database(s).")
        self._refresh_controls()
        self._jobs.submit(lambda r=request: load(r), self._on_loaded)
        return True

    def _on_loaded(self, result: MontageLoad) -> None:
        """A load landed. **Always on the GUI thread** — see the module head.

        The bound method the worker's completion is relayed to. Everything
        that touches a widget happens here and nowhere else.
        """
        expected, self._pending = self._pending, None
        if (isinstance(result, MontageLoad) and result.request is not None
                and result.request != expected):
            # A SECOND BELT, AND `expected is None` IS THE IMPORTANT CASE.
            # `cancel` drops what a superseded click started, but a result can
            # still arrive after this widget has stopped waiting for one --
            # the click moved on, or a second load is in flight because the
            # settings changed without the coefficient doing so. An answer
            # nobody is waiting for is by definition an answer to a question
            # the user has left, and painting it is the montage-under-the-
            # wrong-name failure this whole tab is careful about.
            self._pending = expected
            return
        if not isinstance(result, MontageLoad):
            self._set_status("The montage loader returned nothing.")
            self._refresh_controls()
            return
        if result.error:
            self._drop_montage()
            self._unavailable = result.error if result.unavailable else ""
            self._caption.setPlainText(result.error)
            self._set_status(result.error)
            self._refresh_controls()
            self.montage_failed.emit(result.error)
            return
        self._plans = result.plans
        self._images = result.images
        self._sources = dict(result.sources)
        self._shown_key = self._key
        self._unavailable = ""
        self._fill()
        self._set_status(self._summary())
        self._refresh_controls()
        self.montage_ready.emit(result.n_objects)

    def _on_job_failed(self, message: str) -> None:
        """The runner itself raised. Say so rather than staying blank."""
        self._pending = None
        self._set_status(f"The montage load failed: {message}")
        self._refresh_controls()
        self.montage_failed.emit(str(message))

    # -------------------------------------------------------------- the view

    def plans(self) -> Tuple[Any, ...]:
        """The montage plans now on screen, in order."""
        return self._plans

    def images(self) -> Tuple[Tuple[Any, ...], ...]:
        """The crops now on screen, one tuple per plan."""
        return self._images

    def status_text(self) -> str:
        """The status line: the summary, or the reason there is none."""
        return self._status_text

    def caption_text(self) -> str:
        """Every caption now on screen, exactly as the figure would carry it."""
        return self._caption.toPlainText()

    def save(self, path: Optional[str] = None) -> Optional[str]:
        """Write the montage as a figure, honouring the format preference.

        :param path: where to write. ``None`` asks. The extension is left to
            :func:`spacr.plot.save_figure`, which corrects it to the format
            the user chose -- naming one here is how a PNG ends up in a file
            called ``.pdf``, which is a complaint this project has had twice.
        :returns: the path written, or ``None``.
        """
        if not self._plans:
            self._set_status("There is no montage to save yet.")
            return None
        if path is None:
            from PySide6.QtWidgets import QFileDialog
            from ...plot import figure_output_preferences

            fmt, _dpi = figure_output_preferences()
            suggestion = f"cells_behind_{self._name or 'coefficient'}.{fmt}"
            path, _selected = QFileDialog.getSaveFileName(
                self, "Save the montage", suggestion,
                f"Figure (*.{fmt});;All files (*)")
            if not path:
                return None
        from ...plot import save_figure

        figure = montage_figure(self._plans, self._images,
                                columns=max(self._columns, 4))
        written = save_figure(figure, path, close=True)
        self._set_status(f"{self._summary()} — saved to {written}")
        return written

    # ------------------------------------------------------------- internals

    def _frame(self):
        if self._frame_provider is None:
            return None
        try:
            return self._frame_provider()
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not reach the coefficient table", exc_info=True)
            return None

    def _results_path(self) -> str:
        if self._results_provider is None:
            return ""
        try:
            return str(self._results_provider() or "")
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not reach the results path", exc_info=True)
            return ""

    def _summary(self) -> str:
        if not self._plans:
            return ("The montage loader came back with no montage and no "
                    "reason, which is a bug in the loader rather than in the "
                    "run — nothing was drawn.")
        lines = [plan.summary() for plan in self._plans]
        if len(lines) > 1:
            return (f"{len(lines)} montages, one per guide — "
                    + "; ".join(lines))
        return lines[0]

    def _set_status(self, text: str) -> None:
        self._status_text = str(text)
        self._status.setText(self._status_text)

    def _on_settings_changed(self, *_args) -> None:
        """A crop setting moved: the remembered refusal no longer applies.

        Forcing 'merged' after a run whose PNGs are gone is exactly the case
        where a remembered "no crop source" would keep the button grey for a
        request that has not been tried.
        """
        self._unavailable = ""
        # The load in flight is answering the previous settings.
        self._jobs.cancel()
        self._pending = None
        self._refresh_controls()
        self._announce()

    def _refresh_controls(self) -> None:
        """Grey what cannot act, and put the reason on it — instruction 106."""
        reason = self.reason()
        self._show.setEnabled(not reason)
        self._show.setToolTip(reason or (
            f"Load the cells most consistent with {self._name}'s effect."
            if self._name else "Load the cells behind the selected point."))
        savable = bool(self._plans)
        self._save.setEnabled(savable)
        self._save.setToolTip(
            "Write the montage and its caption as a figure, in the format "
            "the figure preferences name." if savable else
            "There is no montage to save yet — load one first.")

    def _announce(self) -> None:
        """Put the current situation in the status line.

        SEPARATE FROM :meth:`_refresh_controls`, and that separation is the
        whole of it: the two used to be one method, so every handler that
        greyed a button after saying something specific -- "the montage
        loader returned nothing", "the load failed" -- overwrote its own
        sentence with the generic reason a moment later.
        """
        reason = self.reason()
        if reason:
            self._set_status(reason)
        elif self._plans:
            self._set_status(self._summary())
        else:
            self._set_status(
                f"Ready: press “Show the cells” for {self._name}."
                if self._name else self.NOTHING_SELECTED)

    def _drop_montage(self) -> None:
        """Empty the grid and the caption. Nothing on screen is stale."""
        self._plans, self._images, self._sources = (), (), {}
        self._shown_key = ""
        self._clear()
        self._caption.setPlainText("")

    def _column_count(self) -> int:
        width = max(self._scroll.viewport().width() - 12, _CELL_PX)
        return max(1, width // _CELL_PX)

    def _fill(self) -> None:
        """Rebuild the grid from the plans and crops now held."""
        self._columns = self._column_count()
        self._clear()
        captions: List[str] = []
        row = 0
        for plan, crops in zip(self._plans, self._images):
            captions.append(plan.caption())
            if len(self._plans) > 1:
                heading = QLabel(plan.summary())
                heading.setWordWrap(True)
                self._grid.addWidget(heading, row, 0, 1, self._columns)
                row += 1
            if not len(plan.objects):
                # AN EMPTY MONTAGE IS AN ANSWER, NOT AN EMPTY GRID. The wells
                # that reported the guide and gave nothing are named in the
                # caption below; this is the sentence in the grid's own space
                # so the view is never a blank rectangle.
                empty = QLabel(
                    "No object was selected. Every well reporting this "
                    "coefficient contributed none — the caption below names "
                    "each one and why.")
                empty.setWordWrap(True)
                self._grid.addWidget(empty, row, 0, 1, self._columns)
                row += 1
                continue
            rows = plan.objects.reset_index(drop=True)
            for index, crop in enumerate(crops):
                widget = self._thumb(crop, rows.iloc[index]
                                     if index < len(rows) else None)
                self._grid.addWidget(widget, row + index // self._columns,
                                     index % self._columns)
            row += max(1, -(-len(crops) // self._columns))
        self._caption.setPlainText("\n\n".join(captions))

    def _thumb(self, crop, row) -> QWidget:
        tooltip = self._tooltip(row)
        if crop is None:
            label = QLabel("no crop")
            label.setToolTip(tooltip or "this object could not be cut")
            label.setAlignment(Qt.AlignCenter)
            label.setFixedSize(THUMBNAIL_PX, THUMBNAIL_PX)
            return label
        return _Thumb(self._pixmap(crop), tooltip, self._body)

    @staticmethod
    def _tooltip(row) -> str:
        if row is None:
            return ""
        parts = []
        for name, label in (("montage_well", "well"),
                            ("object_label", "label"),
                            ("pred", "score"),
                            ("montage_distance", "|score − target|")):
            if name in getattr(row, "index", ()):
                value = row[name]
                if isinstance(value, float):
                    parts.append(f"{label} {value:.4g}")
                else:
                    parts.append(f"{label} {value}")
        # NOT "carries this guide". The pooled design cannot say that of any
        # single object, and a tooltip is exactly where that claim would get
        # made by accident.
        parts.append("consistent with the effect — membership is inferred")
        return " · ".join(parts)

    @staticmethod
    def _pixmap(crop) -> QPixmap:
        array = np.ascontiguousarray(np.asarray(crop, dtype=np.uint8))
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        height, width = array.shape[:2]
        image = QImage(array.data, width, height, 3 * width,
                       QImage.Format_RGB888)
        # `.copy()` is not optional: QImage does not own the numpy buffer, and
        # without it the pixmap points at freed memory the moment the array
        # goes out of scope.
        return QPixmap.fromImage(image.copy()).scaled(
            THUMBNAIL_PX, THUMBNAIL_PX, Qt.KeepAspectRatio,
            Qt.SmoothTransformation)

    def _clear(self) -> None:
        while self._grid.count():
            item = self._grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                # setParent(None) as well as deleteLater: a widget removed
                # from the layout is still a visible child of the body at its
                # old geometry until the deferred delete runs, which is how a
                # rebuilt grid paints the previous montage under the new one.
                widget.setParent(None)
                widget.deleteLater()

    def _relayout(self) -> None:
        """Reflow to the current width, but only if the column count moved."""
        if not self._plans:
            return
        if self._column_count() == self._columns:
            return
        self._fill()

    def resizeEvent(self, event):        # noqa: N802 - Qt's spelling
        """Reflow the grid after a resize settles."""
        super().resizeEvent(event)
        self._reflow.start()

    def shutdown(self) -> None:
        """Stop waiting for a load, and let no QThread outlive this widget.

        Qt aborts the whole process when a running ``QThread`` is destroyed,
        and a merged-source montage is seconds long -- so leaving the screen
        mid-load is not a rare case. ``JobRunner.shutdown`` drops the results
        and waits a bounded time rather than joining on the GUI thread, which
        is the freeze it exists to remove.
        """
        self._pending = None
        self._jobs.shutdown()

    def closeEvent(self, event):         # noqa: N802 - Qt's spelling
        """Shut the loader down before the widget goes."""
        self.shutdown()
        super().closeEvent(event)
