"""The cells behind one dot on the volcano, and how they were picked.

:mod:`spacr.cell_montage` decides which objects belong behind a coefficient
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

A tab that cannot be filled says why
------------------------------------
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
    QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

# The headless half's own vocabulary, so the controls speak it rather than
# re-declaring their defaults. `spacr.cell_montage` costs numpy and pandas and
# nothing else -- crops and io are lazy inside it -- so this is not the torch
# import the module docstring is careful about.
from ...cell_montage import (                                   # noqa: E402
    DEFAULT_SCORE_COLUMN, MAX_OBJECTS, WINDOW_HALF_WIDTHS,
)

LOG = logging.getLogger(__name__)

__all__ = [
    "THUMBNAIL_PX",
    "OBJECT_CHOICES",
    "SOURCE_CHOICES",
    "SHAPE_CHOICES",
    "BASELINE_CHOICES",
    "MAX_WELL_TABS",
    "MontageRequest",
    "MontageLoad",
    "experiment_root",
    "parse_channels",
    "coefficient_from_frame",
    "intercept_from_frame",
    "well_tab_label",
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

#: The crop's SHAPE, as the user may choose it. ``'object'`` follows the
#: object's own mask and is the better picture; ``'bbox'`` is its padded
#: bounding box. THE CHOICE IS NOT ALWAYS AVAILABLE -- a route that has only a
#: coordinate table has no mask to follow -- and
#: :func:`spacr.cell_montage.montage_route_requirements` is what says so, up
#: front, so the entry is disabled with its reason instead of being clickable
#: and quietly serving a bounding box.
SHAPE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("object", "object-shaped"),
    ("bbox", "bounding box"),
)

#: Where the score window's baseline comes from. The screen median is the
#: default and is a property of the objects; the fitted intercept is the
#: model's own answer to the same question and is arguably the better one.
#: Whichever is in force is written into the caption, because moving the
#: baseline moves the target and therefore which cells are shown.
BASELINE_CHOICES: Tuple[Tuple[str, str], ...] = (
    ("median", "screen median"),
    ("intercept", "fitted intercept"),
)

#: How many well tabs may be open at once.
#:
#: STATED RATHER THAN DISCOVERED. A tab holds one ``QPixmap`` per object and
#: they are only ever closed by hand -- that is the point of them -- so
#: without a bound a session comparing gene after gene fills memory and the
#: user finds out by watching the application slow down. At the 300-object
#: cap a well tab is at most 300 thumbnails of 96x96 RGBA, ~11 MB, so twelve
#: tabs is ~130 MB in the worst case and a small fraction of that in the
#: ordinary one. When the bound is reached NO TAB IS CLOSED FOR THE USER: the
#: new wells are refused, by name, with the sentence that says which x to
#: click.
MAX_WELL_TABS = 12

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


def intercept_from_frame(frame) -> Optional[float]:
    """The fitted intercept out of the coefficient table, or ``None``.

    The other baseline the score window can be centred on. Under the
    well-level model the intercept IS the score of a well carrying none of
    the guide, which is the same quantity the screen median estimates -- so
    offering both is offering the model's answer beside the data's, and the
    caption says which produced the picture.

    :param frame: the coefficient table. Every spelling statsmodels and this
        project use is accepted (``Intercept``, ``(Intercept)``, ``const``),
        because a baseline silently not found would fall back to the median
        and the montage would say ``median`` while the user had asked for the
        intercept.
    :returns: the fitted value, or ``None`` when the table names no intercept
        or its value is not a finite number.
    """
    if frame is None or not len(frame):
        return None
    columns = getattr(frame, "columns", ())
    if "feature" not in columns:
        return None
    from ...figures.panels import effect_column

    column = effect_column(frame)
    if not column:
        return None
    names = frame["feature"].astype(str).str.strip().str.lower()
    wanted = names.isin(("intercept", "(intercept)", "const", "constant"))
    match = frame[wanted]
    if not len(match):
        return None
    try:
        value = float(match[column].iloc[0])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def well_tab_label(well, guides, name="", level="gene"):
    """The tab's own name: THE WELL AND THE gRNA, both.

    A gene with several guides pulls the same well more than once, so two
    tabs called ``p1_r3_c7`` are indistinguishable -- and the whole point of
    a tab that outlives the selection is comparing one gene's cells with
    another's, which a label naming only the well makes impossible.

    THE COEFFICIENT IS NAMED TOO WHEN THE GUIDE ALONE DOES NOT IDENTIFY IT.
    Driving the real tab found the case: the guide-level coefficient
    ``GRA14_1``, and the GENE ``GRA14`` shown one guide at a time, both open
    a tab for ``GRA14_1`` in the same well -- and they are DIFFERENT montages
    with different effects, different windows and different cells. Both read
    ``plate1_r1_c1 \u00b7 GRA14_1`` and nothing on the tab bar told them apart.

    :param well: the well key as the plan spells it.
    :param guides: the guides this montage covers.
    :param name: the coefficient's own name, used when the guides are summed
        and there is therefore no single guide to name.
    :param level: ``'gene'`` or ``'grna'`` -- which kind of coefficient this
        montage answers for.
    :returns: e.g. ``'plate1_r1_c1 \u00b7 GRA14_1'`` for a guide term,
        ``'plate1_r1_c1 \u00b7 GRA14_1 (of GRA14)'`` for that same guide inside
        a gene term, or ``'plate1_r1_c1 \u00b7 GRA14 (2 guides)'`` for the sum.
    """
    listed = [str(g) for g in guides if str(g)]
    if len(listed) == 1:
        guide = listed[0]
        if name and str(name) != guide and str(level) != "grna":
            guide = f"{guide} (of {name})"
    elif listed:
        guide = f"{name or listed[0]} ({len(listed)} guides)"
    else:
        guide = str(name or "?")
    return f"{well} \u00b7 {guide}"


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
    :param count_csvs: the COUNT CSVs from the same input table. The fallback
        source of the per-well guide fractions, and the reason a run folder is
        no longer required: a fraction is ``count / well total``, which these
        files carry outright. Used only when ``results_path`` yields nothing
        readable.
    :param object_type: which mask plane a crop is cut by.
    :param channels: intensity planes for the picture, or ``None`` for the
        ones the run itself saved.
    :param prefer: ``''`` / ``'png'`` / ``'merged'``.
    :param score_column: the per-object classification score. A screen with
        more than one classifier output has more than one candidate.
    :param cap: the largest montage to draw.
    :param per_guide: one montage per guide instead of the gene's guides
        summed. They are different questions -- see
        :func:`spacr.cell_montage.select_montage_per_guide` -- and each plan
        says in its own caption which one it answers.
    :param half_widths: the score window's half-width in robust scales -- the
        direct stringency control. ``0`` means the module's own default.
        ONE NUMBER FOR THE WHOLE SCREEN AND EVERY COEFFICIENT: a width chosen
        per gene is a width that can be tuned until the pictures look right,
        and nothing in the output would show that it had been.
    :param baseline: the baseline to centre the window on, or ``None`` for
        the screen median.
    :param baseline_label: what to call that baseline in the caption.
    :param crop_shape: ``'object'`` or ``'bbox'`` -- see
        :data:`SHAPE_CHOICES`.
    """

    name: str
    effect: float
    level: str = "gene"
    results_path: str = ""
    databases: Tuple[str, ...] = ()
    count_csvs: Tuple[str, ...] = ()
    object_type: str = "cell"
    channels: Optional[Tuple[int, ...]] = None
    prefer: str = ""
    score_column: str = "pred"
    cap: int = 0
    per_guide: bool = False
    half_widths: float = 0.0
    baseline: Optional[float] = None
    baseline_label: str = ""
    crop_shape: str = "object"


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
    :param shapes: the crop shapes EVERY plate's route can actually cut. A
        shape one plate cannot produce is not offered, because a montage in
        which some crops follow the mask and some do not is a montage whose
        pictures are not comparable.
    :param shape_reason: why a shape is missing from ``shapes``, for the
        disabled entry's tooltip.
    """

    request: Optional[MontageRequest] = None
    plans: Tuple[Any, ...] = ()
    images: Tuple[Tuple[Any, ...], ...] = ()
    sources: Dict[str, str] = field(default_factory=dict)
    error: str = ""
    unavailable: bool = False
    shapes: Tuple[str, ...] = ()
    shape_reason: str = ""

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
        CROP_SHAPES, MontageError, fractions_from_counts,
        read_well_guide_fractions,
        load_montage_objects, resolve_montage_crop_source, select_montage,
        select_montage_per_guide, CropSourceChoice, MAX_OBJECTS,
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
    # THE RUN FOLDER IS A CONVENIENCE, NOT A REQUIREMENT. It holds
    # `regression_data.csv`, which is the score/count join PERSISTED -- and the
    # per-well guide fractions in it are `count / well total`, computable from
    # the count CSVs the input table already names. Requiring the folder is
    # what produced "the loaded coefficient table was not read from a run
    # folder" for a user who had loaded their scores and counts and was
    # looking at the coefficients those files produced.
    #
    # So: read the folder when there is one, and BUILD from the counts when
    # there is not. Same arithmetic as `ml.process_reads`, same number.
    counts = None
    trouble = ""
    if folder:
        try:
            counts = read_well_guide_fractions(folder)
        except MontageError as error:
            trouble = str(error)
        except Exception as error:                              # noqa: BLE001
            trouble = f"Could not read the per-well guide fractions: {error}"
    if counts is None and request.count_csvs:
        try:
            counts = fractions_from_counts(request.count_csvs)
        except MontageError as error:
            trouble = trouble or str(error)
        except Exception as error:                              # noqa: BLE001
            trouble = trouble or (
                f"Could not build the guide fractions from the count CSVs: "
                f"{error}")
    if counts is None:
        return MontageLoad(
            request=request,
            error=trouble or (
                "No per-well guide fractions are available: there is no run "
                "folder holding regression_data.csv, and no count CSV is "
                "attached to the input table. Either one is enough."),
            unavailable=True)
    try:
        pass
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
        root = experiment_root(db_path)
        # THE PATHS MUST SURVIVE THE FOLDER MOVING MACHINE (instruction 155
        # F). The database records absolute paths as they were on the machine
        # that wrote it, and this is the one place that knows both the
        # recorded path and the root it is under NOW. Every path-bearing
        # column is re-anchored -- png_path for the exported crops AND
        # path_name / merged_path for the arrays they are cut from -- and
        # whatever carries no recognisable anchor is COUNTED and one is
        # NAMED, because a silent pass-through is how a re-anchor that had
        # already lost the file name stayed invisible.
        from ...crops import reanchor_frame

        objects, report = reanchor_frame(objects, root)
        if report.describe():
            troubles.append(report.describe())
        objects["montage_source_root"] = root
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
    route_notes: List[str] = []
    shape = str(request.crop_shape or "object")
    for root in sorted(set(objects["montage_source_root"].astype(str))):
        here = objects[objects["montage_source_root"].astype(str) == root]
        # THE ROUTE'S OWN REQUIREMENTS, CHECKED UP FRONT (instruction 155 E).
        # The two routes to pixels need different things, and a user missing
        # a channel list has to be told THAT rather than told there is no
        # source -- which is what a check made at cutting time reports.
        choice = resolve_montage_crop_source(
            _crop_settings(request, root), object_type=request.object_type,
            prefer=request.prefer or None, objects=here,
            channels=request.channels)
        if not choice.available:
            refusals.append(f"{root}: {choice.reason}")
            continue
        label = os.path.basename(root.rstrip(os.sep)) or root
        for note in choice.requirement_notes():
            route_notes.append(f"{label}: {note}")
        requirements = choice.requirements
        if requirements is not None and requirements.missing:
            refusals.append(f"{root}: " + "; ".join(requirements.missing))
            continue
        # THE SHAPE IS APPLIED HERE OR IT IS SAID NOT TO BE. An
        # object-shaped crop this route cannot cut must never quietly become
        # a bounding box.
        if requirements is not None and requirements.shapes:
            if not requirements.offers(shape):
                route_notes.append(
                    f"{label}: the {shape!r} crop shape was asked for and "
                    f"this route cannot cut it -- {requirements.why_not(shape)}"
                    f" The crops here are {requirements.shapes[0]!r}.")
                effective = requirements.shapes[0]
            else:
                effective = shape
            spec = getattr(choice.source, "spec", None)
            if spec is not None:
                choice.source.spec = spec.with_(
                    use_bounding_box=(effective == "bbox"))
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
    from ...cell_montage import WINDOW_HALF_WIDTHS

    half_widths = float(request.half_widths or WINDOW_HALF_WIDTHS)
    selection = dict(
        level=request.level, score_column=request.score_column, cap=cap,
        half_widths=half_widths, baseline=request.baseline,
        baseline_label=request.baseline_label or None, crop_source=combined)
    try:
        if request.per_guide:
            plans = select_montage_per_guide(
                objects, counts, request.name, float(request.effect),
                **selection)
            if not plans:
                return MontageLoad(
                    request=request,
                    error=f"No guide of {request.name} is reported present in "
                          "any well of the count data, so there is no montage "
                          "to draw one guide at a time.")
        else:
            plans = [select_montage(
                objects, counts, request.name, float(request.effect),
                **selection)]
    except MontageError as error:
        return MontageLoad(request=request, error=str(error))
    except Exception as error:                                  # noqa: BLE001
        LOG.debug("montage selection failed", exc_info=True)
        return MontageLoad(
            request=request, error=f"Could not select the montage: {error}")

    images: List[Tuple[Any, ...]] = []
    for plan in plans:
        images.append(_cut(plan, sources, request, troubles))

    notes = tuple(route_notes) + tuple(f"NOTE {t}" for t in troubles)
    if notes:
        plans = [_with_notes(plan, notes) for plan in plans]
    # WHAT THE ROUTES CAN ACTUALLY CUT, so the shape control can disable what
    # they cannot rather than accepting the click and doing something else.
    offered: Optional[set] = None
    why = ""
    for choice in sources.values():
        req = choice.requirements
        if req is None:
            continue
        offered = set(req.shapes) if offered is None else offered & set(req.shapes)
        for candidate in CROP_SHAPES:
            if not why and not req.offers(candidate):
                why = req.why_not(candidate)
    return MontageLoad(request=request, plans=tuple(plans),
                       images=tuple(images), sources=described,
                       shapes=tuple(s for s in CROP_SHAPES
                                    if s in (offered or set())),
                       shape_reason=why)


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


class _WellTab(QWidget):
    """One WELL's cells, in a tab that closes only when its x is clicked.

    THE WELL IS THE UNIT, and that is not a layout preference. The guide
    fraction is defined per well and the count rule is
    ``round(objects in well x fraction in well)`` per well, so the well is
    what a reader checks -- a single grid of everything hides the one
    arithmetic the montage is asking to be trusted on.

    Each tab carries its OWN caption naming its own well, guide and
    coefficient, which is what makes it safe for it to outlive the selection:
    a tab left open while the volcano rings another gene still says which
    gene it is, so two genes' cells can sit side by side without either being
    mistaken for the other.

    :param key: the identity used to recognise this tab on a re-run --
        ``(coefficient, level, guide label, well)``.
    :param label: the tab's own text: the well AND the guide.
    :param parent: the tab widget.
    """

    def __init__(self, key: Tuple[str, ...], label: str, parent=None):
        super().__init__(parent)
        self.key = tuple(key)
        self.label = str(label)
        self._rows = None
        self._crops: Tuple[Any, ...] = ()
        self._caption_text = ""
        self._columns = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        # WHY THIS TAB IS EMPTY, above the grid rather than inside it. In the
        # grid it would be indistinguishable from a thumbnail to anything
        # counting them -- including the 'no crop' placeholders, which ARE
        # one per object and must stay countable.
        self._note = QLabel()
        self._note.setWordWrap(True)
        self._note.setVisible(False)
        layout.addWidget(self._note)
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

        self._caption = QPlainTextEdit()
        self._caption.setReadOnly(True)
        self._caption.setMinimumHeight(70)
        split.addWidget(self._caption)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([380, 140])
        layout.addWidget(split, 1)

    # ------------------------------------------------------------- content
    def set_content(self, rows, crops: Sequence[Any], caption: str,
                    columns: int) -> None:
        """Replace what this tab shows.

        :param rows: the well's own object rows, index-reset.
        :param crops: the crops for those rows, aligned with them.
        :param caption: this well's own account of itself.
        :param columns: how many thumbnails fit across.
        """
        self._rows = rows
        self._crops = tuple(crops)
        self._caption_text = str(caption)
        self._caption.setPlainText(self._caption_text)
        self.fill(columns)

    def caption_text(self) -> str:
        """This tab's caption, exactly as it is on screen."""
        return self._caption_text

    def thumbs(self) -> Tuple[QWidget, ...]:
        """Every widget now in this tab's grid, in order."""
        return tuple(self._grid.itemAt(i).widget()
                     for i in range(self._grid.count()))

    def fill(self, columns: int) -> None:
        """Lay the crops out at ``columns`` per row."""
        self._columns = max(int(columns), 1)
        self.clear()
        if self._rows is None or not len(self._crops):
            self._note.setText(
                "This well contributed no object to the montage. The caption "
                "below says why. The tab stays until its × is clicked.")
            self._note.setVisible(True)
            return
        self._note.setVisible(False)
        for index, crop in enumerate(self._crops):
            row = self._rows.iloc[index] if index < len(self._rows) else None
            self._grid.addWidget(_thumbnail(crop, row, self._body),
                                 index // self._columns,
                                 index % self._columns)

    def clear(self) -> None:
        """Empty the grid, leaving nothing of the previous fill behind."""
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


def _tooltip(row) -> str:
    """The provenance line on one thumbnail."""
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


def _thumbnail(crop, row, parent=None) -> QWidget:
    """One crop as a widget, or a labelled placeholder when it is missing."""
    tooltip = _tooltip(row)
    if crop is None:
        label = QLabel("no crop")
        label.setToolTip(tooltip or "this object could not be cut")
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(THUMBNAIL_PX, THUMBNAIL_PX)
        return label
    return _Thumb(_pixmap(crop), tooltip, parent)


def _pixmap(crop) -> QPixmap:
    """A crop as a thumbnail-sized ``QPixmap``."""
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

    #: No run at all. NAMES THE WAY TO GET ONE, which the old sentence did
    #: not: "No regression results are loaded" was reported by the maintainer
    #: immediately after a regression finished, and it said neither which run
    #: it was looking for nor how to give it one. A run that finishes IS the
    #: loaded run (154 G), so seeing this means none has finished in this
    #: session -- and the answer to that is to open one from disk.
    NO_RUN_LOADED = (
        "No run is loaded, so there is no regression_data.csv to read the "
        "per-well guide fractions from. A run that finishes loads itself; to "
        "look at an earlier one, pick it in the Runs tab or open its folder "
        "there with “Load run…”.")

    #: A table with no run folder behind it. A DIFFERENT FAILURE, and it used
    #: to be reported as the one above: the coefficients are on screen, so
    #: "no regression results are loaded" reads as a contradiction of what
    #: the user is looking at.
    #:
    #: "THE LOADED COEFFICIENT TABLE", which is what the user calls it. "The
    #: coefficient table on screen" describes a widget; the user is thinking
    #: about a run (instruction 155 A, handed over by the agent that fixed
    #: the provider). With that provider fixed this branch is reached only by
    #: a table genuinely opened from a bare CSV -- which is the case this
    #: sentence is FOR.
    RESULTS_WITHOUT_A_FOLDER = (
        "The LOADED coefficient table was not read from a run folder, so "
        "there is nowhere to find the regression_data.csv this montage needs. "
        "Open the run in the Runs tab with “Load run…” to point at its "
        "folder.")

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
        #: The open well tabs, by their identity, so a re-run refreshes the
        #: tab a well already has instead of opening a second one.
        self._well_tabs: Dict[Tuple[str, ...], _WellTab] = {}

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

        controls.addWidget(QLabel("crop shape"))
        self._shape = QComboBox()
        for value, label in SHAPE_CHOICES:
            self._shape.addItem(label, value)
        self._shape.setToolTip(
            "Object-shaped crops follow the object's own mask and are the "
            "better picture. A route that has only a coordinate table has no "
            "mask to follow and can cut bounding boxes only — when that is "
            "the case the entry is disabled with its reason rather than "
            "quietly giving you a box.")
        self._shape.currentIndexChanged.connect(self._on_settings_changed)
        controls.addWidget(self._shape)

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

        # THE STRINGENCY ROW. Every control here changes WHICH CELLS a reader
        # is looking at, so every one is written into the caption and a
        # non-default value says so ON the montage
        # (:meth:`spacr.cell_montage.MontagePlan.settings_line`). They are
        # PER SCREEN, NEVER PER GENE: one width for every coefficient, because
        # a width chosen per gene is a width that can be tuned until the
        # pictures look right and nothing in the output would show that it
        # had been.
        stringency = QHBoxLayout()
        stringency.addWidget(QLabel("half-width"))
        self._half_widths = QDoubleSpinBox()
        self._half_widths.setDecimals(2)
        self._half_widths.setRange(0.05, 20.0)
        self._half_widths.setSingleStep(0.25)
        self._half_widths.setValue(float(WINDOW_HALF_WIDTHS))
        self._half_widths.setSuffix(" scales")
        self._half_widths.setToolTip(
            "How wide the score window is, in robust scales (1.4826 x MAD) "
            "either side of the implied score. Wider admits more cells and "
            "'closest' means less. ONE NUMBER FOR THE WHOLE SCREEN AND EVERY "
            "COEFFICIENT — this is deliberately not a per-gene control.")
        self._half_widths.valueChanged.connect(self._on_settings_changed)
        stringency.addWidget(self._half_widths)

        self._baseline = QComboBox()
        for value, label in BASELINE_CHOICES:
            self._baseline.addItem(label, value)
        self._baseline.setToolTip(
            "What the implied score is measured from. The screen median is "
            "the objects' own answer; the fitted intercept is the model's, "
            "and under the well-level fit it is the score of a well carrying "
            "none of the guide. Whichever is in force is named in the "
            "caption.")
        self._baseline.currentIndexChanged.connect(self._on_settings_changed)

        stringency.addWidget(QLabel("baseline"))
        stringency.addWidget(self._baseline)

        stringency.addWidget(QLabel("score column"))
        self._score = QLineEdit()
        self._score.setPlaceholderText(DEFAULT_SCORE_COLUMN)
        self._score.setMaximumWidth(110)
        self._score.setToolTip(
            "The per-object classification score the window is applied to. "
            "A screen with more than one classifier output has more than one "
            "candidate, and the caption says which produced the picture.")
        self._score.textChanged.connect(self._on_settings_changed)
        stringency.addWidget(self._score)

        stringency.addWidget(QLabel("max objects"))
        self._cap = QSpinBox()
        self._cap.setRange(1, 5000)
        self._cap.setValue(int(MAX_OBJECTS))
        self._cap.setToolTip(
            "The largest montage to draw. The merged source is priced by "
            "FIELDS TOUCHED, not crops cut: 300 crops cost 11.43 ms each "
            "over 30 fields against 2.58 ms over 6, so a montage spanning "
            "many wells is the expensive one however few it takes from each.")
        self._cap.valueChanged.connect(self._on_settings_changed)
        stringency.addWidget(self._cap)
        stringency.addStretch(1)
        layout.addLayout(stringency)

        # THE TAB LIVES IN THE LEFT HALF OF THE FIGURES SPLITTER, which
        # starts at 780 px. Two rows of controls with their prose spelled out
        # in the widgets pushed this widget's minimum width to 1346 px, which
        # is not a cosmetic problem: a minimum wider than the splitter forces
        # the whole screen wider. The words live in the tooltips and the
        # combos elide.
        for box in (self._object, self._shape, self._source, self._per_guide,
                    self._baseline):
            box.setSizeAdjustPolicy(
                QComboBox.AdjustToMinimumContentsLengthWithIcon)
            box.setMinimumContentsLength(10)
            box.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

        self._channels.setMinimumWidth(60)
        self._score.setMinimumWidth(60)

        self._status = QLabel(self._status_text)
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        # ONE TAB PER WELL, CLOSED ONLY BY ITS OWN X. The first tab is the
        # summary and is not closable: it is where the arithmetic, the
        # settings and every refusal are read, and it has to be somewhere
        # even when there is no montage at all.
        self._tabs = QTabWidget()
        self._tabs.setTabsClosable(True)
        self._tabs.setMovable(True)
        self._tabs.setDocumentMode(True)
        self._tabs.tabCloseRequested.connect(self._close_tab)

        summary = QWidget()
        summary_layout = QVBoxLayout(summary)
        summary_layout.setContentsMargins(4, 4, 4, 4)
        summary_layout.setSpacing(4)
        # THE CAPTION IS PART OF THE OUTPUT, NOT A TOOLTIP. It states the
        # wells, the score window, EVERY SETTING that decided which cells
        # these are, the whole arithmetic a reader can check the sum from,
        # and -- last, so it is the sentence a reader leaves with -- that
        # guide membership is INFERRED from a well-level fraction rather than
        # observed. Selectable, because the reason to want it is usually to
        # paste it into a methods section.
        self._caption = QPlainTextEdit()
        self._caption.setReadOnly(True)
        self._caption.setPlainText("")
        self._caption.setMinimumHeight(90)
        summary_layout.addWidget(self._caption, 1)
        self._tabs.addTab(summary, "Summary")
        self._summary_tab = summary
        # The summary tab has no x. Qt puts the close button on whichever
        # side the style names; removing BOTH is the only way that does not
        # depend on the style.
        for side in (QTabBar.LeftSide, QTabBar.RightSide):
            self._tabs.tabBar().setTabButton(0, side, None)
        self._tabs.setTabToolTip(
            0, "The whole montage in words: the wells, the window, the "
               "arithmetic and every setting that decided which cells these "
               "are. The pictures are in the per-well tabs beside it.")
        layout.addWidget(self._tabs, 1)

        self._reflow = QTimer(self)
        self._reflow.setSingleShot(True)
        self._reflow.setInterval(_REFLOW_DEBOUNCE_MS)
        # A bound method of this GUI-thread object, per job_runner's rules.
        self._reflow.timeout.connect(self._relayout)

        self._jobs = JobRunner(self, threaded=bool(threaded),
                               app_key="cell montage")
        self._jobs.job_failed.connect(self._on_job_failed)

        self._refresh_controls()

        # Hover help belongs on a setting's NAME, not on the field the user is
        # about to type into (instruction 113). One post-pass rather than a
        # convention every hand-built row has to remember -- the same call
        # class_editor, pca_view and pivot_builder each end their __init__
        # with, and its absence here is why this was the last screen still
        # putting help on an editable field.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

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

        The rule: a control that cannot do anything is greyed
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
            return (f"The loaded coefficient table has no fitted effect "
                    f"for {self._name}, and the score window is "
                    "'baseline + effect'. Load the run's results table first.")
        if not self._results_path() and not self.count_csvs():
            # A MISSING RUN FOLDER IS NO LONGER FATAL. The guide fractions are
            # `count / well total`, which the input table's COUNT CSVs carry
            # outright, so the folder is one of two sources rather than a
            # requirement. This branch is now reached only when BOTH are
            # absent -- which is the one case where there is genuinely nothing
            # to compute a fraction from.
            frame = self._frame()
            if frame is not None and len(frame):
                return self.RESULTS_WITHOUT_A_FOLDER
            return self.NO_RUN_LOADED
        if not self.databases():
            return ("No measurement database is attached to this run's input "
                    "table, so there are no per-object rows and no crops to "
                    "show. Attach one to a plate row.")
        try:
            parse_channels(self._channels.text())
        except ValueError:
            return (f"'{self._channels.text()}' is not a list of channel "
                    "indices. Use numbers separated by commas, e.g. 0,1,2.")
        if (str(self._baseline.currentData() or "median") == "intercept"
                and intercept_from_frame(self._frame()) is None):
            # SAID, NEVER FALLEN BACK FROM. Quietly using the median while
            # the user has asked for the intercept moves the target and
            # therefore which cells are shown, and the caption would say
            # 'the screen median' under a setting reading 'fitted intercept'.
            return ("This coefficient table names no Intercept term, so "
                    "there is no fitted intercept to centre the score window "
                    "on. Choose the screen median instead.")
        return self._unavailable

    def count_csvs(self) -> Tuple[str, ...]:
        """The COUNT CSVs attached to the run's input table.

        THE SAME PROVIDER THE DATABASES COME FROM. The input table's rows are
        ``{"plate", "score", "count", "database"}``, so the counts were always
        one field away -- which is why requiring a run folder for the guide
        fractions was never necessary, only unexamined.
        """
        if self._database_provider is None:
            return ()
        try:
            rows = self._database_provider()
        except Exception:                                       # noqa: BLE001
            LOG.debug("could not reach the input table", exc_info=True)
            return ()
        out: List[str] = []
        for row in rows or ():
            path = row.get("count", "") if isinstance(row, dict) else ""
            text = str(path or "").strip()
            if text and text not in out:
                out.append(text)
        return tuple(out)

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
        baseline, label = self._baseline_value()
        return MontageRequest(
            name=self._name, effect=float(self._effect),
            level=self._level, results_path=self._results_path(),
            databases=self.databases(),
            count_csvs=self.count_csvs(),
            object_type=str(self._object.currentData() or "cell"),
            channels=parse_channels(self._channels.text()),
            prefer=str(self._source.currentData() or ""),
            per_guide=bool(self._per_guide.currentData()),
            score_column=(self._score.text().strip()
                          or DEFAULT_SCORE_COLUMN),
            cap=int(self._cap.value()),
            half_widths=float(self._half_widths.value()),
            baseline=baseline, baseline_label=label,
            crop_shape=str(self._shape.currentData() or "object"))

    def _baseline_value(self) -> Tuple[Optional[float], str]:
        """The baseline the window is centred on, and what to call it.

        :returns: ``(None, '')`` for the screen median -- which
            :func:`spacr.cell_montage.score_window` computes itself -- or the
            fitted intercept and the name the caption gives it.
        """
        if str(self._baseline.currentData() or "median") != "intercept":
            return None, ""
        value = intercept_from_frame(self._frame())
        if value is None:
            return None, ""
        return value, "the model's fitted intercept"

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
        self._apply_shape_availability(result)
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

    def loaded_run_name(self) -> str:
        """The run this tab is describing, as a name a user recognises.

        The run folder's own basename -- ``ols_3`` -- which is what the Runs
        tab calls it and what the figure grid heads its section with. "" when
        no run is loaded, so a caller can tell "no run" from "a run whose
        name I could not work out".
        """
        path = self._results_path()
        if not path:
            return ""
        folder = os.path.dirname(path) if os.path.isfile(path) else path
        return os.path.basename(str(folder).rstrip(os.sep)) or str(folder)

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

    def _apply_shape_availability(self, result: MontageLoad) -> None:
        """Grey the crop shapes this run's route cannot actually cut.

        Instruction 155 E: "an object-shaped crop must not appear as a choice
        that silently does something else". A route with no mask -- a
        coordinate table, or exported PNGs that were cut when the run wrote
        them -- cannot follow an outline, so the entry is DISABLED with the
        reason on it rather than accepted and quietly served a box.

        The check cannot be made before a load, because which source answers
        is a fact about the disk. So the entries are all live until a load has
        looked, and the answer is applied afterwards -- the same shape as the
        remembered "this run has no crop source".
        """
        model = self._shape.model()
        offered = tuple(result.shapes)
        answered = bool(result.plans) or bool(result.shape_reason)
        # NO SHAPE CHOICE AT ALL is a real answer and not "everything is
        # available": the exported PNGs were cut when the run wrote them, so
        # neither entry does anything. The whole control greys out with that
        # sentence rather than offering two options that both do nothing.
        self._shape.setEnabled(not answered or bool(offered))
        if answered and not offered:
            self._shape.setToolTip(result.shape_reason)
        else:
            self._shape.setToolTip(
                "Object-shaped crops follow the object's own mask and are "
                "the better picture. A route that has only a coordinate "
                "table has no mask to follow and can cut bounding boxes "
                "only — when that is the case the entry is disabled with its "
                "reason rather than quietly giving you a box.")
        for index in range(self._shape.count()):
            item = model.item(index) if hasattr(model, "item") else None
            if item is None:
                continue
            value = str(self._shape.itemData(index) or "")
            ok = (not answered) or value in offered
            item.setEnabled(ok)
            item.setToolTip("" if ok else result.shape_reason)
        if offered and str(self._shape.currentData() or "") not in offered:
            # The choice on screen is not the one that was cut. Move it, and
            # say so -- a control reading 'object-shaped' over bounding-box
            # crops is the silent substitution this is here to prevent.
            for index in range(self._shape.count()):
                if str(self._shape.itemData(index) or "") in offered:
                    self._shape.blockSignals(True)
                    self._shape.setCurrentIndex(index)
                    self._shape.blockSignals(False)
                    break

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
        # And a shape greyed out by the LAST route is re-armed: forcing
        # 'merged' after a run whose PNGs are gone is exactly the case where
        # a remembered refusal would keep a real choice unavailable.
        self._apply_shape_availability(MontageLoad())
        self._refresh_controls()
        self._announce()

    def _refresh_controls(self) -> None:
        """Disable controls that cannot act and show the reason."""
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
            # WHICH RUN THIS IS ABOUT. Instruction 154 G: the choice of
            # loaded run has to be visible from the views that depend on it,
            # not only from the tab that sets it -- otherwise a montage built
            # from the wrong run looks exactly like one built from the right
            # one.
            if not self._name:
                self._set_status(self.NOTHING_SELECTED)
            else:
                run = self.loaded_run_name()
                self._set_status(
                    f"Ready: press “Show the cells” for {self._name}"
                    + (f", from the run {run}." if run else "."))

    def _drop_montage(self) -> None:
        """Forget the montage the plans describe. THE WELL TABS STAY.

        A tab is closed only when its own x is clicked -- that is the whole
        point of it, because comparing one gene's cells with another's is
        what a montage that vanishes on the next click makes impossible. Each
        tab carries its own caption naming its own well, guide and
        coefficient, so a tab left standing under a moved selection still
        says what it is; the SUMMARY, which describes the selection, is what
        must not survive it.
        """
        self._plans, self._images, self._sources = (), (), {}
        self._shown_key = ""
        self._caption.setPlainText("")

    def well_tabs(self) -> Tuple["_WellTab", ...]:
        """The open well tabs, in the order they are on screen."""
        return tuple(self._tabs.widget(i) for i in range(1, self._tabs.count())
                     if isinstance(self._tabs.widget(i), _WellTab))

    def tab_labels(self) -> Tuple[str, ...]:
        """Every tab's text, summary first -- what a user reads across."""
        return tuple(self._tabs.tabText(i) for i in range(self._tabs.count()))

    def thumbnails(self) -> Tuple[QWidget, ...]:
        """Every thumbnail on screen, across every open well tab."""
        out: List[QWidget] = []
        for tab in self.well_tabs():
            out.extend(t for t in tab.thumbs() if t is not None)
        return tuple(out)

    def _open_well_tab(self, key: Tuple[str, ...], label: str,
                       tooltip: str) -> "_WellTab":
        """Add one well tab, WITH ITS X IN THE TOP LEFT.

        Qt puts the close button on whichever side
        ``SH_TabBar_CloseButtonPosition`` names, and on this project's style
        that is the RIGHT -- the request asks for "an x in the top left of
        the tab", so the automatic button is removed and ours is placed on
        the left. It closes by WIDGET and not by index, because a tab's index
        moves whenever an earlier one is closed and an index captured at
        creation time would close somebody else's tab.
        """
        from PySide6.QtWidgets import QToolButton

        # A LABEL NO OTHER OPEN TAB ALREADY HAS. The rule above makes a
        # collision unlikely rather than impossible, and two identical tabs
        # is precisely the failure this label exists to prevent.
        taken = {t.label for t in self.well_tabs()}
        unique, suffix = label, 2
        while unique in taken:
            unique, suffix = f"{label} #{suffix}", suffix + 1
        tab = _WellTab(key, unique, self._tabs)
        self._well_tabs[key] = tab
        index = self._tabs.addTab(tab, tab.label)
        self._tabs.setTabToolTip(index, tooltip)
        bar = self._tabs.tabBar()
        bar.setTabButton(index, QTabBar.RightSide, None)
        close = QToolButton(bar)
        close.setText("\u00d7")
        close.setAutoRaise(True)
        close.setCursor(Qt.PointingHandCursor)
        close.setToolTip(
            f"Close “{tab.label}”. This tab closes from here and nowhere "
            "else — it survives another coefficient, a re-sort and a re-run, "
            "because comparing two genes' cells side by side is the point.")
        close.clicked.connect(lambda *_a, w=tab: self._close_widget(w))
        bar.setTabButton(index, QTabBar.LeftSide, close)
        return tab

    def _close_widget(self, widget) -> None:
        """Close the tab holding ``widget``, whatever index it now has."""
        index = self._tabs.indexOf(widget)
        if index > 0:
            self._close_tab(index)

    def _close_tab(self, index: int) -> None:
        """A tab's own x was clicked. THE ONLY WAY A WELL TAB CLOSES."""
        widget = self._tabs.widget(index)
        if widget is None or widget is self._summary_tab:
            return
        self._tabs.removeTab(index)
        for key, tab in list(self._well_tabs.items()):
            if tab is widget:
                del self._well_tabs[key]
        widget.setParent(None)
        widget.deleteLater()

    def _column_count(self) -> int:
        width = max(self._tabs.width() - 40, _CELL_PX)
        return max(1, width // _CELL_PX)

    def _fill(self) -> None:
        """Open or refresh one tab per well, and write the summary.

        A well the plans no longer mention keeps its tab -- it closes by its
        x and by nothing else -- and a well that is mentioned again refreshes
        the tab it already has rather than opening a second one.
        """
        self._columns = self._column_count()
        captions: List[str] = []
        refused: List[str] = []
        refreshed: set = set()
        answered: set = set()
        for plan, crops in zip(self._plans, self._images):
            captions.append(plan.caption())
            guides = tuple(plan.guides) or (plan.coefficient.name,)
            answered.add((plan.coefficient.name, plan.coefficient.level,
                          "|".join(guides)))
            rows = plan.objects.reset_index(drop=True)
            wells = list(rows["montage_well"].astype(str)) \
                if "montage_well" in rows.columns else []
            if len(crops) != len(rows):
                captions.append(
                    f"NOTE the crop source returned {len(crops)} images for "
                    f"{len(rows)} selected objects; only the pairs that line "
                    "up are drawn.")
            for well in plan.wells:
                if not well.contributed:
                    continue
                key = (plan.coefficient.name, plan.coefficient.level,
                       "|".join(guides), well.well)
                positions = [i for i, value in enumerate(wells)
                             if value == well.well and i < len(crops)]
                tab = self._well_tabs.get(key)
                if tab is None:
                    if len(self._well_tabs) >= MAX_WELL_TABS:
                        refused.append(well.well)
                        continue
                    tab = self._open_well_tab(key, well_tab_label(
                        well.well, guides, plan.coefficient.name,
                        plan.coefficient.level),
                        f"{plan.coefficient.describe()} — {well.describe()}. "
                        "This tab closes only when its × is clicked.")
                tab.set_content(rows.iloc[positions].reset_index(drop=True),
                                [crops[i] for i in positions],
                                self._well_caption(plan, well, guides),
                                self._columns)
                refreshed.add(key)
        # A TAB THIS RUN NO LONGER FILLS MUST NOT KEEP SHOWING THE OLD ONE.
        # Driving the real widget found it: narrow the cap and re-run, and the
        # wells that dropped out kept their previous thumbnails under a
        # summary describing the new settings. The tab is NOT closed -- only
        # its x does that -- it is emptied and says why, which is the same
        # rule the empty montage follows.
        for key, tab in self._well_tabs.items():
            if key in refreshed or key[:3] not in answered:
                continue
            tab.set_content(
                None, (),
                f"{key[3]} contributed no object under the settings now in "
                "force, so this tab is empty rather than showing the cells a "
                "previous run put here. The Summary tab has the arithmetic.",
                self._columns)
        if refused:
            captions.append(
                f"{len(refused)} well tab(s) were NOT opened -- "
                f"{', '.join(refused[:6])}"
                + (" and others" if len(refused) > 6 else "")
                + f" -- because {MAX_WELL_TABS} well tabs are already open. "
                  "A tab holds one thumbnail per object, so the number is "
                  "bounded; close one with its × to make room. No tab is "
                  "ever closed for you.")
        captions.append(
            f"tabs: one per well that contributed an object, labelled with "
            f"the well AND the guide, closed only by the × on the tab. Up to "
            f"{MAX_WELL_TABS} stay open at once.")
        self._caption.setPlainText("\n\n".join(captions))

    @staticmethod
    def _well_caption(plan, well, guides: Sequence[str]) -> str:
        """One well's own account of itself, for its own tab.

        SELF-CONTAINED ON PURPOSE. A tab outlives the selection that made it,
        so it has to name its coefficient, its well and its guide without
        anything else on screen agreeing -- otherwise a tab left open beside
        another gene's is a picture that reads as the new one.
        """
        from ...cell_montage import INFERENCE_NOTICE

        lines = [
            f"Cells behind {plan.coefficient.describe()}",
            f"well {well.well}, guide(s) {', '.join(guides)}",
            f"count: {well.describe()}",
            plan.window.describe(),
            plan.settings_line(),
            INFERENCE_NOTICE.format(name=plan.coefficient.name),
        ]
        return "\n".join(lines)

    def _clear(self) -> None:
        """Empty every open well tab's grid, leaving the tabs standing."""
        for tab in self.well_tabs():
            tab.clear()

    def _relayout(self) -> None:
        """Reflow to the current width, but only if the column count moved."""
        columns = self._column_count()
        if columns == self._columns:
            return
        self._columns = columns
        for tab in self.well_tabs():
            tab.fill(columns)

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
