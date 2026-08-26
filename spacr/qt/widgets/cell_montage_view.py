"""Show the measured cells represented by a regression coefficient.

:mod:`spacr.cell_montage` selects the objects and records the selection reason;
this module resolves and displays their image crops beside the run's figures.
Each attached database resolves its own exported-PNG or ``merged/<fov>.npy``
source so multi-plate montages can combine experiments with different storage
layouts. The crop mask plane and channels come from :class:`spacr.crops.CropSpec`
or, by default, the run's ``measurements.db`` metadata.

Image reads run through :class:`spacr.qt.job_runner.JobRunner` and only the
GUI-thread completion handler updates widgets. Converting at most 300 crops to
``QPixmap`` remains on the GUI thread because QPixmap is not transferable
across threads; the recorded 224-pixel thumbnail conversion takes about
21 milliseconds for 300 crops.

The tab stays visible when crops are unavailable and reports the reason in its
status, tooltip, and disabled action. Crop-source discovery is cached after a
load and invalidated when the coefficient, databases, or crop settings change.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout,
    QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSizePolicy,
    QSpinBox, QSplitter, QTabBar, QTabWidget, QVBoxLayout, QWidget,
)

# The headless half's own vocabulary, so the controls speak it rather than
# re-declaring their defaults. `spacr.cell_montage` costs numpy and pandas and
# nothing else -- crops and io are lazy inside it -- so this is not the torch
# import the module docstring is careful about.
from ...crops import (LOAD_IMAGES, LOAD_IMAGES_LABEL, STREAM_IMAGES,
                      STREAM_IMAGES_LABEL, picture_source_label)
from ...cell_montage import (                                   # noqa: E402
    DEFAULT_SCORE_COLUMN, MAX_OBJECTS, WINDOW_HALF_WIDTHS,
)
from ..hidpi import scaled_for                               # noqa: E402
from ..theme import close_mark_button, install_close_marks   # noqa: E402

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

#: Pixels between thumbnails in the montage grid.
GRID_SPACING = 10

#: The smallest a thumbnail is allowed to shrink to.
#:
#: A fixed column count means a narrow panel makes the pictures smaller
#: rather than showing fewer of them. Below this they stop being pictures of
#: a cell, and a scroll bar is the better answer.
MIN_THUMBNAIL_PX = 32

#: Fallback when the preference store cannot be read (a bare widget test).
DEFAULT_MONTAGE_COLUMNS = 6

#: Which mask plane a crop is cut by -- the "which array the masks are in"
#: half of the request. The vocabulary is :data:`spacr.crops.OBJECT_TYPES`
#: and is read from there rather than retyped; these are the four a measured
#: run actually writes crops for.
OBJECT_CHOICES: Tuple[str, ...] = ("cell", "nucleus", "pathogen", "cytoplasm")

#: The crop source, as the user may force it. ``""`` is
#: :func:`spacr.crops.resolve_crop_source`'s own ``auto``, which prefers the
#: exported PNGs when they exist -- and the timing table in this module's
#: docstring is the second reason that preference is right.
#:
#: The panel offers explicit ``load images`` and ``stream images`` choices.
#: Stored values remain ``'png'`` and ``'merged'`` for compatibility; an empty
#: legacy ``automatic`` value still resolves through the normal preference
#: order.
SOURCE_CHOICES: Tuple[Tuple[str, str], ...] = (
    (LOAD_IMAGES, f"{LOAD_IMAGES_LABEL} — the crops already in data/"),
    (STREAM_IMAGES, f"{STREAM_IMAGES_LABEL} — cut from merged/*.npy as it goes"),
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
#: user finds out by watching the application slow down. At the 2,000-object
#: cap a well tab is at most 2,000 thumbnails of 96x96 RGBA, ~74 MB, so
#: twelve tabs is ~880 MB in the worst case and a small fraction of that in
#: the ordinary one. When the bound is reached NO TAB IS CLOSED FOR THE USER: the
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


def _colour_to_source() -> dict:
    """``{'r': idx, 'g': idx, 'b': idx}`` -- which SOURCE channel each colour
    holds, from spaCR's own mapping rather than from position."""
    try:
        from ...crops import DEFAULT_PNG_CHANNEL_MAPPING

        return {k: int(v) for k, v in DEFAULT_PNG_CHANNEL_MAPPING.items()
                if v is not None}
    except Exception:                                        # noqa: BLE001
        return {"r": 2, "g": 1, "b": 0}


_COLOUR_TO_SOURCE = _colour_to_source()


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
        # THE ANNOTATION APP'S SPELLING IS ACCEPTED HERE. It asks which
        # COLOUR PLANES to show and writes them 'r,g,b' -- `_csv_to_list`
        # keeps whatever strings it is given and `filter_channels_pil` reads
        # the letters directly. This box asks a different question (which
        # SOURCE channels to cut) and answers it in indices, so typing the
        # annotator's answer here raised ValueError and the tab refused to
        # draw anything at all: "'rgb' is not a list of channel indeces ...
        # and this blocks the user from being able to spawn any images".
        #
        # THE LETTERS ARE TRANSLATED THROUGH THE MAPPING, never by position.
        # spaCR's default is {r: 2, g: 1, b: 0}, so 'r' is source channel TWO;
        # reading it as 0 would cut the planes in reverse and produce a crop
        # that looks entirely plausible and is wrong.
        letter = part.strip().lower()
        if letter in _COLOUR_TO_SOURCE:
            out.append(_COLOUR_TO_SOURCE[letter])
            continue
        # 'rgb' AND 'rg' TOO, not only 'r,g,b'. "the user should be able to
        # type in r,g,b or any combination" -- and a token made only of
        # colour letters has exactly one reading, so refusing it is pedantry
        # with a blocked montage on the other end. This is the form the
        # report actually used.
        if letter and all(c in _COLOUR_TO_SOURCE for c in letter):
            out.extend(_COLOUR_TO_SOURCE[c] for c in letter)
            continue
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
    #: Score CSVs used by the fit. When a database lacks the score column,
    #: values are read from these files in memory without modifying the
    #: database.
    score_csvs: Tuple[str, ...] = ()
    #: How the cells are drawn, in the annotator's own names. Empty means the
    #: annotator's defaults.
    picture: Optional[Dict[str, Any]] = None
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
    :param objects: EVERY object row the load read, not just the ones the
        plans selected. The Compare panel's wider contrasts (187 B) need the
        cells the montage did NOT pick -- "against every other well" has
        nothing on its other side without them -- and the join that makes
        every database measurement reachable (187 A) is keyed on these rows.
        A reference, not a copy: the frame is already in memory.
    :param counts: the per-well guide fractions this load resolved. Carried
        for the same reason: naming a control (184) is a question about the
        COUNT data, and the panel cannot answer it from object rows.
    """

    request: Optional[MontageRequest] = None
    plans: Tuple[Any, ...] = ()
    images: Tuple[Tuple[Any, ...], ...] = ()
    sources: Dict[str, str] = field(default_factory=dict)
    error: str = ""
    unavailable: bool = False
    shapes: Tuple[str, ...] = ()
    shape_reason: str = ""
    objects: Any = None
    counts: Any = None

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



def _thumb_px_of(picture) -> int:
    """How big to draw each cell, from the picture settings.

    `img_size` is the annotator's name for it, so a user who has set the
    crop size in one panel finds the same number here.
    """
    try:
        value = int((picture or {}).get("img_size") or 0)
    except (TypeError, ValueError):
        return 0
    # Bounded: a thumbnail larger than the panel is a scroll bar, and one
    # smaller than a few pixels is not a picture.
    return max(24, min(value, 512)) if value else 0


def fits_on_a_page(width: int, height: int, thumb_px: int,
                   spacing: int = 6) -> Tuple[int, int]:
    """Calculate the thumbnail grid capacity of a viewport.

    Parameters
    ----------
    width, height : int
        Available viewport dimensions in pixels.
    thumb_px : int
        Thumbnail width and height in pixels.
    spacing : int, default=6
        Space between adjacent thumbnails in pixels.

    Returns
    -------
    int
        Number of thumbnail columns.
    int
        Total thumbnails per page. Both values are at least one, including
        when the viewport is smaller than a thumbnail.
    """
    # THE LAST ROW NEEDS NO TRAILING SPACING, and charging it one loses a
    # whole row whenever the fit is close -- reported as "never more that
    # two rows ... where there could be 3 almost 4". n items span
    # n*thumb + (n-1)*spacing, so the capacity is (available + spacing)
    # divided by the step, not available // step.
    thumb = max(1, int(thumb_px))
    gap = max(0, int(spacing))
    step = thumb + gap
    columns = max(1, (int(width) + gap) // step)
    rows = max(1, (int(height) + gap) // step)
    return columns, columns * rows


def _per_page_of(picture) -> int:
    """RETIRED. Always 0, which means "work it out from the container".

    `cells_per_page` is gone from the settings and the defaults. This
    survives only to translate a settings CSV that still carries it: the
    value is ignored rather than honoured, because honouring a count that
    disagrees with the geometry is the bug the setting was removed for.
    """
    return 0


def _show_all_of(picture) -> bool:
    """Whether to show every cell in the well rather than only the candidates."""
    return bool((picture or {}).get("show_all_in_well"))


def _crop_settings(request: "MontageRequest", root: str) -> Dict[str, Any]:
    """The settings mapping ``resolve_crop_source`` takes for one plate."""
    settings: Dict[str, Any] = {"src": root}
    # WHAT THE USER ASKED FOR IN THE SETTINGS WINDOW (instruction 170 B),
    # translated into the crop layer's own vocabulary by
    # `picture_settings.to_crop_settings` -- which carries ONLY the settings
    # that change how a crop is CUT, and only the ones this mode uses. A
    # settings window whose values never reached the picture would be worse
    # than no settings window.
    if request.picture:
        from ...picture_settings import to_crop_settings

        settings.update(to_crop_settings(request.picture))
    if request.channels:
        # ``png_dims`` is the PNG path's own name for "which intensity planes
        # become the picture", so the user's choice is expressed in the
        # vocabulary the crop spec already speaks instead of a new one.
        settings["png_dims"] = list(request.channels)
    return settings


def no_score_refusal(score_csvs, troubles=()) -> str:
    """Explain where spaCR looked for a per-object classification score.

    Parameters
    ----------
    score_csvs
        Score files loaded with the regression result.
    troubles
        Additional diagnostics from the attached databases.

    Returns
    -------
    str
        A user-facing explanation that identifies both possible score
        sources and suggests the next useful action.
    """
    loaded = [str(path) for path in (score_csvs or ())]
    if loaded:
        shown = ", ".join(os.path.basename(path) for path in loaded[:3])
        more = f", +{len(loaded) - 3} more" if len(loaded) > 3 else ""
        noun = "score file" if len(loaded) == 1 else "score files"
        where = (f"The {len(loaded)} loaded {noun} ({shown}{more}) also "
                 "contain no matching per-object score.")
    else:
        where = ("No score file is loaded. Load the per-object score CSV used "
                 "for the regression; the montage can join it in memory "
                 "without modifying a database.")
    details = " ".join(str(item) for item in troubles).strip()
    message = ("No per-object classification score was found in the attached "
               "databases. " + where)
    return (message + (" " + details if details else "")).strip()


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
                score_column=request.score_column,
                scores=list(request.score_csvs) or None)
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
        return MontageLoad(request=request, unavailable=True,
                           error=no_score_refusal(request.score_csvs,
                                                  troubles))

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
    # THE WHOLE WELL, OR ONLY THE CHOSEN CELLS (instruction 172). Not the
    # default: the two answer different questions, and a reader who cannot
    # see the well cannot judge how many of it the fraction claims.
    selection["show_all"] = _show_all_of(request.picture)
    # WHICH CELLS BELONG TO THE COEFFICIENT (172, 173). 'attributed' and
    # 'assigned' both need every guide's fitted effect in the well, because a
    # posterior is a comparison -- so they read them out of the run the
    # montage is already showing.
    picking = str((request.picture or {}).get("cell_picking") or "rank")
    selection["picking"] = picking
    selection["threshold"] = float(
        (request.picture or {}).get("picking_threshold") or 0.55)
    if picking in ("attributed", "assigned", "multivariate"):
        from ...cell_montage import effects_from_results

        raw = effects_from_results(request.results_path)
        # The results name guides as the DESIGN did (`225160_1`) while the
        # counts name them as the library does (`TGGT1_225160_1`), so the two
        # are matched on the design spelling.
        #
        # THE PREFIX IS MEASURED, NOT HARD-CODED. This read
        # `str(g).split("TGGT1_")[-1]`, which is one organism's name written
        # into the matching rule: every guide of a Plasmodium or a human
        # library kept its prefix, matched nothing in `raw`, and the whole
        # `effects` map came back empty -- at which point the attribution
        # silently has no competition to compare against. See instruction
        # 184, which is this same assumption in the control fields.
        from ...control_names import common_prefix

        names = [str(g) for g in counts["grna"].unique()]
        prefix = common_prefix(names)
        head = f"{prefix}_" if prefix else ""

        def _design_spelling(name: str) -> str:
            text = str(name)
            return text[len(head):] if head and text.startswith(head) else text

        selection["effects"] = {
            str(g): raw[_design_spelling(g)]
            for g in names if _design_spelling(g) in raw} or None
    if picking == "multivariate":
        # THE GRID NOTHING EVER SET (186 A). `select_montage` has taken an
        # `effects_grid` since option C shipped and no caller supplied one,
        # so multivariate could never run: it found None every time, fell
        # back to the single-score attribution, and said so in the caption.
        # The fallback worked exactly as designed and hid the fact that what
        # it fell back FROM was unreachable.
        from ...cell_montage import effects_grid_from_results

        selection["effects_grid"] = effects_grid_from_results(
            request.results_path)
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
                       shape_reason=why,
                       objects=objects, counts=counts)


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
    """Display a clickable crop with the annotation app's tile styling.

    Shared tile chrome supplies rounded clipping and the resting, state, and
    hover rings. Activating the thumbnail opens the provenance details already
    summarized by its tooltip.
    """

    #: (tooltip). Emitted on a left click so the view can show the detail.
    clicked = Signal(str)

    def __init__(self, pixmap: QPixmap, tooltip: str, parent=None,
                 size: int = 0, highlight: str = ""):
        super().__init__(parent)
        self._pixmap = pixmap
        self.setToolTip(tooltip)
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(int(size or THUMBNAIL_PX), int(size or THUMBNAIL_PX))
        self.setFrameShape(QFrame.NoFrame)
        # Transparent, so the rounded tile sits on the grid without a grey
        # square peeking out at the corners.
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setCursor(Qt.PointingHandCursor)
        # THE ANNOTATION APP'S OWN BORDER, asked for by appearance:
        # "highlight the cells most likely to be whatever gene is picked ...
        # as if they were annotated in the annotations app". `label_to_hex`
        # is where that colour is decided, and it is theme-aware because
        # contrast is -- so this borrows it rather than picking a blue.
        self.highlight = str(highlight or "")
        self._hovered = False

    def pixmap(self):                       # noqa: D401 - Qt naming
        """The crop, as handed in. Painted by hand, so QLabel never holds it."""
        return self._pixmap

    # -- the look ------------------------------------------------------

    def _colours(self):
        """``(ring, hover)`` -- the one ring's colour, and what hover makes it.

        ONE RING HERE, TWO IN THE ANNOTATE GRID, and the difference is not an
        oversight. The annotator has to show the class AND the cursor at once
        because the class is what you are assigning; this tile's ring is only
        ever provenance, so the cursor can simply take it over. Asked for
        directly: "the wite rim is to thick and should replace the blue".
        """
        from ..screens.annotate import current_ring_color, resting_border_color

        return (self.highlight or resting_border_color()), current_ring_color()

    def paintEvent(self, event):            # noqa: N802 - Qt naming
        from .tile_chrome import paint_tile

        ring, hover = self._colours()
        painter = QPainter(self)
        try:
            # `current=False` and the colour swapped instead: one ring, which
            # hover RECOLOURS rather than surrounds. A picked cell therefore
            # keeps its blue everywhere the cursor is not -- which is the
            # whole of show-all, where the point is to compare the cells that
            # carry the inference against the ones that do not.
            paint_tile(painter, float(self.width()), float(self.height()),
                       self._pixmap,
                       border_colour=(hover if self._hovered else ring),
                       ring_colour="", current=False)
        finally:
            painter.end()

    # -- the cursor ----------------------------------------------------

    def enterEvent(self, event):            # noqa: N802 - Qt naming
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):            # noqa: N802 - Qt naming
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):       # noqa: N802 - Qt naming
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.toolTip())
        else:
            super().mousePressEvent(event)


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
        # NO SCROLLBAR (instruction 211). The visible area IS the page, and
        # a page that scrolls is not a page -- it is a grid with a smaller
        # window over it, which is what this replaces. If anything is below
        # the fold the page size is wrong, and hiding the bar makes that
        # visible instead of navigable.
        self._scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        split.addWidget(self._scroll)

        # HOW BIG THE CELLS ARE DRAWN AND HOW MANY FIT ON A PAGE. Both are
        # the user's, set from the picture settings; the defaults are what
        # this tab has always used, so a panel nobody touches is unchanged.
        self._thumb_px = THUMBNAIL_PX
        #: The size the user asked for, which is the CEILING the fitted
        #: size may grow to. Held separately because `_thumb_px` is
        #: recomputed from the viewport on every relayout, and a ceiling
        #: overwritten by a fitted value stops being a ceiling.
        self._requested_px = THUMBNAIL_PX
        self._per_page = 0
        self._page = 0
        #: How the crops are drawn -- the annotator's settings, or none.
        self._picture: dict = {}
        #: Detail windows this tab opened, kept so Python does not
        #: collect them the moment the click handler returns.
        self._details: list = []

        self._pager = QWidget()
        pager = QHBoxLayout(self._pager)
        pager.setContentsMargins(6, 0, 6, 0)
        self._prev = QPushButton("‹ previous")
        self._prev.clicked.connect(lambda: self.show_page(self._page - 1))
        pager.addWidget(self._prev)
        self._page_label = QLabel("")
        pager.addWidget(self._page_label, 1)
        self._next = QPushButton("next ›")
        self._next.clicked.connect(lambda: self.show_page(self._page + 1))
        pager.addWidget(self._next)
        self._pager.setVisible(False)
        layout.addWidget(self._pager)

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
                    columns: int, thumb_px: int = 0,
                    per_page: int = 0, picture=None) -> None:
        """Replace what this tab shows.

        :param rows: the well's own object rows, index-reset.
        :param crops: the crops for those rows, aligned with them.
        :param caption: this well's own account of itself.
        :param columns: how many thumbnails fit across.
        :param thumb_px: the size to draw each crop at; 0 keeps
            :data:`THUMBNAIL_PX`.
        :param per_page: how many crops one page holds; 0 means all of them.
        """
        self._rows = rows
        self._crops = tuple(crops)
        self._caption_text = str(caption)
        self._caption.setPlainText(self._caption_text)
        if thumb_px:
            self._thumb_px = int(thumb_px)
            self._requested_px = int(thumb_px)
        if per_page:
            self._per_page = int(per_page)
        if picture is not None:
            self._picture = dict(picture)
        self._page = 0
        self.fill(columns)

    # ------------------------------------------------------------- paging

    def geometry_page(self) -> tuple:
        """Return ``(columns, per_page)`` for the current scroll viewport.

        THE COLUMN COUNT IS DECIDED, the row count is measured. It used to be
        `viewport_width // cell_px`, so the number of cells on a row -- the
        thing a reader compares across wells -- changed every time the window
        did: "the cell tab shows 3 cells per well and then more if i change
        the size of the container". A montage whose shape depends on the
        window is not comparable with itself.

        So the columns come from the preference and the THUMBNAILS take up
        the slack: a wider panel draws the same cells bigger, up to the
        natural size, rather than fitting more of them. How many rows fit is
        still measured, because that is what paging means.
        """
        area = self._scroll.viewport()
        columns = self.column_count()
        self._thumb_px = self._thumbnail_px_for(area.width(), columns)
        _measured, per_page = fits_on_a_page(area.width(), area.height(),
                                             self._thumb_px)
        rows = max(1, per_page // max(1, _measured))
        return columns, columns * rows

    def column_count(self) -> int:
        """Cells per row: the user's preference, not the window's width."""
        try:
            from ..preferences import get_montage_columns

            return max(1, int(get_montage_columns()))
        except Exception:                                    # noqa: BLE001
            return max(1, int(DEFAULT_MONTAGE_COLUMNS))

    def _thumbnail_px_for(self, width: int, columns: int) -> int:
        """The thumbnail size that puts ``columns`` of them across ``width``.

        THE CROPS FILL THE ROW. The ceiling used to be the module constant,
        so a wide tab showing six columns drew six 96 px pictures with the
        rest of the row empty -- "the image crops are not filling the space
        where they should be in the cell tab". The ceiling is the size the
        user asked for instead, so raising it in the settings lets the
        pictures grow into the space that is there.

        There is still a ceiling, and the reason has not changed: past its
        natural size a crop is an interpolated blur. What changed is who
        decides where that is.
        """
        gap = GRID_SPACING
        usable = max(0, int(width)) - gap * max(0, columns - 1)
        ceiling = max(int(getattr(self, "_requested_px", THUMBNAIL_PX)),
                      MIN_THUMBNAIL_PX)
        if columns <= 0 or usable <= 0:
            return ceiling
        return max(MIN_THUMBNAIL_PX, min(ceiling, usable // columns))

    def per_page(self) -> int:
        """Return the number of crops that fit in the current viewport.

        Capacity is recalculated from the live viewport and thumbnail size so
        it follows window resizing.
        """
        return self.geometry_page()[1]

    def page_count(self) -> int:
        """Return the pages required for this well at the current capacity."""
        size = self.per_page()
        if not size or not self._crops:
            return 1
        return max(1, -(-len(self._crops) // size))

    def first_on_page(self) -> int:
        """Return the crop index anchoring the currently displayed page."""
        return self._page * max(1, self.per_page())

    def show_crop(self, index: int) -> int:
        """Turn to the page holding crop ``index``. Returns the page."""
        size = max(1, self.per_page())
        return self.show_page(int(index) // size)

    def resizeEvent(self, event):               # noqa: N802 - Qt naming
        """Relay out, and keep the reader where they were."""
        anchor = self.first_on_page()
        super().resizeEvent(event)
        # THE FIRST IMAGE ON THE PAGE STAYS PUT, not the page number.
        self.show_crop(anchor)

    def page(self) -> int:
        """The page now shown, counting from zero."""
        return self._page

    def show_page(self, index: int) -> int:
        """Show page ``index``, clamped. Returns the page actually shown."""
        self._page = max(0, min(int(index), self.page_count() - 1))
        self.fill(self._columns)
        return self._page

    def _page_slice(self):
        """The crops and rows for the page now shown."""
        size = self.per_page()
        if not size:
            return list(range(len(self._crops)))
        start = self._page * size
        return list(range(start, min(start + size, len(self._crops))))

    def caption_text(self) -> str:
        """This tab's caption, exactly as it is on screen."""
        return self._caption_text

    def crops(self) -> Tuple[Any, ...]:
        """Every crop this tab holds, across every page."""
        return tuple(self._crops)

    def thumbs(self) -> Tuple[QWidget, ...]:
        """Return the current page's thumbnail widgets in display order.

        Use :meth:`crops` to retrieve all crops held by the tab.
        """
        return tuple(self._grid.itemAt(i).widget()
                     for i in range(self._grid.count()))

    def fill(self, columns: int) -> None:
        """Lay the crops out, at the column count THIS TAB measures.

        `columns` is a HINT and is deliberately overridden: the caller
        derives it from a fixed cell size over the whole tab width, while
        the page size comes from the real thumbnail size over the scroll
        area's viewport. Two numbers for one thing disagree, and the symptom
        is cells running off the right edge with half the rows a page has
        room for.

        The argument stays because every caller passes it and it is still
        the right fallback before there is a viewport to measure.
        """
        measured, _count = self.geometry_page()
        self._columns = max(int(measured or columns), 1)
        self.clear()
        if self._rows is None or not len(self._crops):
            self._note.setText(
                "This well contributed no object to the montage. The caption "
                "below says why. The tab stays until its × is clicked.")
            self._note.setVisible(True)
            return
        self._note.setVisible(False)
        # ONE PAGE AT A TIME when a well has more cells than a page holds.
        # A well capped at 300 objects drawn as one grid is a scroll nobody
        # reads to the end of, and the alternative that was NOT taken is
        # silently truncating it -- a montage that shows some of a well and
        # says it showed the well is the failure this whole panel avoids
        # everywhere else.
        shown = self._page_slice()
        for position, index in enumerate(shown):
            crop = self._crops[index]
            row = self._rows.iloc[index] if index < len(self._rows) else None
            thumb = _thumbnail(crop, row, self._body, size=self._thumb_px,
                               picture=self._picture)
            # CLICK FOR THE PROVENANCE. The tile already carries it on its
            # tooltip -- which well, which object, which route cut it -- and
            # a tooltip is unreadable the moment you want to compare two of
            # them or copy a number out.
            if hasattr(thumb, "clicked"):
                thumb.clicked.connect(self._show_cell_detail)
            self._grid.addWidget(thumb,
                position // self._columns, position % self._columns)
        self._refresh_pager()

    def _show_cell_detail(self, text: str) -> None:
        """What this crop is, in a window that stays until it is closed."""
        from PySide6.QtWidgets import QDialog, QPlainTextEdit, QVBoxLayout

        if not str(text or "").strip():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Cell")
        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit(str(text), dialog)
        view.setReadOnly(True)
        # SELECTABLE, because the reason to open this is usually to copy a
        # path or an object id out of it.
        layout.addWidget(view)
        dialog.resize(560, 260)
        dialog.show()
        self._details.append(dialog)

    def _refresh_pager(self) -> None:
        """Say which page this is, and offer the others."""
        pages = self.page_count()
        if pages <= 1:
            self._pager.setVisible(False)
            return
        size = max(1, self.per_page())
        first = self._page * size + 1
        last = min(first + size - 1, len(self._crops))
        self._page_label.setText(
            f"cells {first}-{last} of {len(self._crops)}   "
            f"(page {self._page + 1} of {pages})")
        self._prev.setEnabled(self._page > 0)
        self._next.setEnabled(self._page < pages - 1)
        self._pager.setVisible(True)

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


def candidate_colour() -> str:
    """The border a likely cell wears, in the annotation app's own palette.

    Class 1 -- the first annotation colour -- because these ARE the panel's
    first class of thing: the cells consistent with the coefficient. Theme
    aware, because `label_to_hex` is: the same hue deepens against a light
    tile so it stays readable, which is issue #6.
    """
    try:
        from ..annotate_engine import label_to_hex
    except Exception:                                        # noqa: BLE001
        return "#3ea6ff"
    dark = True
    try:
        from ..preferences import resolve_effective_theme

        dark = str(resolve_effective_theme() or "").strip().lower() != "light"
    except Exception:                                        # noqa: BLE001
        pass
    return label_to_hex(1, dark=dark) or "#3ea6ff"


def _is_candidate(row) -> bool:
    """Whether this object is one of the cells the coefficient points at."""
    if row is None:
        return False
    try:
        if "montage_candidate" not in getattr(row, "index", ()):
            return False
        return bool(row["montage_candidate"])
    except Exception:                                        # noqa: BLE001
        return False


def _thumbnail(crop, row, parent=None, size: int = 0,
               picture=None) -> QWidget:
    """Build a crop widget or a labeled placeholder when data is missing.

    ``picture`` contains the annotation display settings applied by
    :func:`picture_settings.draw_crop`.
    """
    tooltip = _tooltip(row)
    px = int(size or THUMBNAIL_PX)
    if crop is not None and picture:
        from ...picture_settings import draw_crop

        crop = draw_crop(crop, picture)
    if crop is None:
        label = QLabel("no crop")
        label.setToolTip(tooltip or "this object could not be cut")
        label.setAlignment(Qt.AlignCenter)
        label.setFixedSize(px, px)
        return label
    highlight = candidate_colour() if _is_candidate(row) else ""
    return _Thumb(_pixmap(crop, px, parent), tooltip, parent, size=px,
                  highlight=highlight)


def _pixmap(crop, size: int = 0, target=None) -> QPixmap:
    """A crop as a thumbnail-sized ``QPixmap``.

    ``size`` is the LOGICAL side the tile occupies; ``target`` is the widget
    it will be drawn on, so the crop is rasterised at that screen's pixel
    density rather than at a fraction of it.
    """
    array = np.ascontiguousarray(np.asarray(crop, dtype=np.uint8))
    if array.ndim == 2:
        array = np.repeat(array[:, :, None], 3, axis=2)
    height, width = array.shape[:2]
    image = QImage(array.data, width, height, 3 * width,
                   QImage.Format_RGB888)
    # `.copy()` is not optional: QImage does not own the numpy buffer, and
    # without it the pixmap points at freed memory the moment the array
    # goes out of scope.
    px = int(size or THUMBNAIL_PX)
    return scaled_for(QPixmap.fromImage(image.copy()), target, px)


#: What the Annotate tab says on hover.
ANNOTATE_TAB_TOOLTIP = (
    "Ten ways of choosing WHICH of these cells get annotated, each saying "
    "what it is for and what it costs — the top-scoring cells against a "
    "matched random draw, uncertainty and diversity sampling, control wells "
    "as anchors, positive-unlabelled learning, self-training, two-view "
    "disagreement, score strata, neighbour propagation, and the plain random "
    "draw every one of them is measured against. Wells are never split "
    "across train and test, and the fit is reported with the score's own "
    "inputs removed as well as kept.")


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

    #: Message shown when no run folder is available. It names the missing
    #: input and tells the user how to load a current or earlier run.
    NO_RUN_LOADED = (
        "No run is loaded, so there is no regression_data.csv to read the "
        "per-well guide fractions from. A run that finishes loads itself; to "
        "look at an earlier one, pick it in the Runs tab or open its folder "
        "there with “Load run…”.")

    #: Message shown when coefficients came from a bare CSV rather than a run
    #: folder. It distinguishes a visible table from the missing per-run files
    #: required to build the montage.
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
        #: Whether this view's work goes to a worker thread. Remembered so a
        #: panel built later runs the way the view was asked to run -- an
        #: unthreaded view that grew a threaded tab would put a QThread into
        #: a test that was constructed to have none.
        self._threaded = bool(threaded)

        # EVERY PIECE OF STATE A CONTROL READS IS BORN HERE, before a single
        # signal is connected. A widget whose controls are live before its
        # state exists is the `_significance` crash that took this application
        # down at launch, and the rule earned its own test file.
        self._key: str = ""
        self._name: str = ""
        #: Every coefficient in the current selection. The grid shows one at
        #: a time while preserving the complete selection for linked views.
        self._keys: List[str] = []
        #: Coefficients still to load, when `build_every_selected` is
        #: walking the selection. Empty at rest.
        self._queue: List[str] = []
        self._level: str = "gene"
        self._effect: Optional[float] = None
        self._plans: Tuple[Any, ...] = ()
        self._images: Tuple[Tuple[Any, ...], ...] = ()
        #: The load signature the crops in `_images` answer, or None.
        self._loaded_signature = None
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

        self._object = QComboBox()
        for name in OBJECT_CHOICES:
            self._object.addItem(name, name)
        self._object.setToolTip(
            "Which mask plane a crop is cut by. 'cytoplasm' has no plane on "
            "disk and is derived as cell minus nucleus/pathogen, exactly as "
            "measure_crop derives it.")
        self._object.currentIndexChanged.connect(self._on_settings_changed)
        self._object.setVisible(False)

        # CHANNELS LIVES IN THE SETTINGS WINDOW, not on the toolbar. Asked
        # 2026-08-19: "in the cell tab there is no need to have object
        # channels outisde of the settings pannel, moove it to the settings
        # pannel with the other settings". It was already offered there under
        # the annotator's own name, so the toolbar copy was a second control
        # for one setting -- and two controls for one setting is two places
        # for it to be wrong.
        #
        # The widget stays, unparented and hidden, because `channels()` and
        # the run's saved state both read it and a removal would have been a
        # rename disguised as a deletion. It mirrors the settings window.
        self._channels = QLineEdit()
        self._channels.setPlaceholderText("as the run saved them")
        self._channels.setVisible(False)
        self._channels.setToolTip(
            "Which planes become the picture. Type the COLOUR LETTERS the "
            "annotation application uses — r, g, b, or any combination such "
            "as 'r,g,b', 'rg' or just 'b' — and each is resolved to the "
            "source channel this screen put in that colour. Source channel "
            "NUMBERS also work ('0,1,2') for anyone who knows them.\n\n"
            "Left empty, the run's own png_dims are read back out of "
            "measurements.db, so the crops match the PNGs that run wrote.")
        self._channels.textChanged.connect(self._on_settings_changed)

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
        self._shape.setVisible(False)

        self._source = QComboBox()
        for value, label in SOURCE_CHOICES:
            self._source.addItem(label, value)
        self._source.currentIndexChanged.connect(self._on_settings_changed)
        self._source.currentIndexChanged.connect(self._on_mode_changed)
        self._source.setVisible(False)

        # THE ANNOTATOR'S CONTROL OVER THE PICTURE (instruction 170 B).
        self._picture_settings: dict = {}
        #: What the last load resolved, for the settings window's choosers.
        self._last_source = None
        self._last_objects = None
        self._counts = None
        #: Comparison windows this tab opened, kept so Python does
        #: not collect them the moment the handler returns.
        self._comparisons: list = []
        self._picture_button = QPushButton("Picture settings…")
        self._picture_button.setToolTip(
            "How the cells are drawn: channels, size, normalisation, "
            "outlines. The same settings the annotation application offers, "
            "under the same names. What the chosen mode cannot use is greyed "
            "with the reason rather than hidden.")
        self._picture_button.clicked.connect(self.edit_picture_settings)
        controls.addWidget(self._picture_button)

        # COMPARE THE CELLS THIS TAB PICKED against the rest (177 F). It sits
        # beside the picture settings because it asks the same question of
        # the same selection: these cells, versus the ones the picker did not
        # choose.
        self._compare_button = QPushButton("Compare a measurement…")
        self._compare_button.setToolTip(
            "Compare any measurement between the cells this tab picked for "
            "each gene and the rest of the screen. Cell, well or plate "
            "level; five ways of drawing it; the test chosen from the "
            "normality and variance checks and reported with n; and one "
            "folder holding the figure, the data, the statistics and the "
            "settings.")
        self._compare_button.clicked.connect(self.compare_a_measurement)
        controls.addWidget(self._compare_button)

        # HOW THE CELLS GET ANNOTATED is a tab rather than a fourth button on
        # this row. The three buttons already here put this widget's minimum
        # width at 553 px against a splitter that floors at 520, and a fourth
        # took it to 713 -- a minimum wider than the panel it sits in forces
        # the whole regression screen wider, which is the failure the
        # stringency row was taken off this toolbar to fix.
        self._annotation_panel = None
        self._annotation_page = None
        self._annotation_placeholder = None
        self._annotation_tab = None

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
        # No argument: `clicked` carries a bool that `save` would read as its
        # path. The guard in `save` covers it too; this says the intent.
        self._save.clicked.connect(lambda: self.save())
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
        self._half_widths = QDoubleSpinBox()
        self._half_widths.setDecimals(2)
        self._half_widths.setRange(0.05, 20.0)
        self._half_widths.setSingleStep(0.25)
        self._half_widths.setValue(float(WINDOW_HALF_WIDTHS))
        self._half_widths.setSuffix(" scales")
        self._half_widths.setToolTip(
            "Score-window half-width in robust scales (1.4826 × MAD). Larger "
            "values admit more cells and make 'closest' less selective. ONE "
            "NUMBER FOR THE WHOLE SCREEN AND EVERY COEFFICIENT — this is "
            "deliberately not a per-gene control, so widening it to rescue "
            "one gene widens it for all of them.")
        self._half_widths.valueChanged.connect(self._on_settings_changed)
        self._half_widths.setVisible(False)

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

        self._baseline.setVisible(False)

        self._score = QLineEdit()
        self._score.setPlaceholderText(DEFAULT_SCORE_COLUMN)
        self._score.setMaximumWidth(110)
        self._score.setToolTip(
            "The per-object classification score the window is applied to. "
            "A screen with more than one classifier output has more than one "
            "candidate, and the caption says which produced the picture.")
        self._score.textChanged.connect(self._on_settings_changed)
        self._score.setVisible(False)

        self._cap = QSpinBox()
        # AS HIGH AS THE SETTINGS WINDOW CAN GO. This box is the value
        # `request()` reads, and the window writes its choice back into it
        # through `setValue`, which clamps silently -- so a lower ceiling
        # here would quietly turn a cap the user chose into a different one
        # with nothing on screen saying so.
        #
        # The floor stays at 1 rather than following the window down to 0: a
        # cap of 0 reaches `select_montage` as "no cap given" and draws the
        # default 2,000 under a caption that says 0, so it is the one value
        # in the window's range this control must not pass on.
        self._cap.setRange(1, 1_000_000)
        self._cap.setValue(int(MAX_OBJECTS))
        self._cap.setToolTip(
            "The largest montage to draw. The merged source is priced by "
            "FIELDS TOUCHED, not crops cut: 300 crops cost 11.43 ms each "
            "over 30 fields against 2.58 ms over 6, so a montage spanning "
            "many wells is the expensive one however few it takes from each.")
        self._cap.valueChanged.connect(self._on_settings_changed)
        self._cap.setVisible(False)
        # THE ROW IS GONE FROM THE TOOLBAR. Its four controls live in the
        # settings window now; the widgets stay because `request()` and the
        # saved state both read them, and the window writes back to them so
        # the two cannot drift.
        stringency.addStretch(1)

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

        # THE ANNOTATE TAB IS ALWAYS THERE AND ITS CONTENT IS NOT. The tab
        # is named from the start, so the strategies are a place a user can
        # find rather than a button that has to be discovered; the panel
        # inside it -- forty controls and a fitting runner -- is built the
        # first time somebody opens it. A montage that is never annotated
        # therefore costs a label and a layout.
        self._annotation_page = QWidget()
        page_layout = QVBoxLayout(self._annotation_page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        waiting = QLabel(ANNOTATE_TAB_TOOLTIP)
        waiting.setWordWrap(True)
        waiting.setMinimumWidth(160)
        waiting.setAlignment(Qt.AlignTop)
        page_layout.addWidget(waiting)
        self._annotation_placeholder = waiting
        self._annotation_tab = self._tabs.insertTab(
            1, self._annotation_page, "Annotate")
        self._tabs.setTabToolTip(1, ANNOTATE_TAB_TOOLTIP)
        self._tabs.currentChanged.connect(self._on_tab_changed)

        # THE GRAPH TAB (179 A), beside Summary and empty until a montage has
        # been generated: before that there are no groups to graph, and a tab
        # offering to would be a control that cannot work. It is created here
        # so it keeps its place in the tab ORDER -- added later it would
        # arrive after whichever well tabs a run opened.
        self._graph_tab = None
        self._graph_panel = None
        # The summary tab has no x. Qt puts the close button on whichever
        # side the style names; removing BOTH is the only way that does not
        # depend on the style.
        for side in (QTabBar.LeftSide, QTabBar.RightSide):
            for fixed in (0, self._tabs.indexOf(self._annotation_page)):
                self._tabs.tabBar().setTabButton(fixed, side, None)
        # THE APPLICATION'S CLOSE MARK, NOT THIS WIDGET'S. Asked for once;
        # the strip keeps it as tabs are opened and closed. See
        # `theme.install_close_marks`.
        install_close_marks(self._tabs)
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

    def set_coefficients(self, keys) -> None:
        """Store an ordered coefficient selection and show its latest member.

        A montage represents one coefficient because crop ranking depends on
        that coefficient's effect. The full selection remains available from
        :meth:`selected_coefficients`; :meth:`show_next_coefficient` cycles the
        displayed montage without discarding the other selected coefficients.

        :param keys: every selected ``feature``, in pick order.
        """
        keys = [str(k) for k in (keys or ()) if str(k)]
        self._keys = keys
        if not keys:
            return
        self.set_coefficient(keys[-1])

    def selected_coefficients(self) -> List[str]:
        """Every coefficient in the current selection, in pick order."""
        return list(getattr(self, "_keys", []) or
                    ([self._key] if self._key else []))

    def build_every_selected(self) -> int:
        """Queue one montage for each selected coefficient.

        Loads are chained in selection order because montage construction is
        asynchronous. Completed well tabs remain available, enabling comparison
        across coefficients while preserving each montage's guide-specific crop
        ranking.

        Returns
        -------
        int
            Number of selected coefficients queued for loading.
        """
        keys = self.selected_coefficients()
        if not keys:
            return 0
        self._queue = list(keys[1:])
        self.set_coefficient(keys[0])
        self.build()
        return len(keys)

    def _build_the_next_queued(self) -> bool:
        """Start the next queued coefficient. Returns whether one was taken.

        ONE COEFFICIENT THAT CANNOT LOAD MUST NOT STOP THE REST. `build`
        returns False when this key has no request -- no database, no crop
        source -- and stopping there would leave the rest of the selection
        queued forever with nothing on screen saying why. So the queue is
        walked until a load actually starts or it is empty, and the ones
        that could not load have already said so in their own status.
        """
        while getattr(self, "_queue", None):
            key = self._queue.pop(0)
            self.set_coefficient(key)
            if self.build():
                return True
        return False

    def show_next_coefficient(self) -> Optional[str]:
        """Move the grid to the next coefficient in the selection.

        :returns: the key now shown, or ``None`` if fewer than two are
            selected and there is nowhere to step to.
        """
        keys = self.selected_coefficients()
        if len(keys) < 2:
            return None
        try:
            position = keys.index(self._key)
        except ValueError:
            position = -1
        key = keys[(position + 1) % len(keys)]
        self.set_coefficient(key)
        return key

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
            return (f"'{self._channels.text()}' is not a channel list. Use "
                    "the colour letters the annotation application uses — "
                    "r, g, b or any combination of them — or source channel "
                    "numbers, separated by commas. For example 'r,g,b', "
                    "'r' or '0,1,2'.")
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

    def score_csvs(self) -> Tuple[str, ...]:
        """The SCORE CSVs attached to the run's input table.

        The same provider and the same rows as :meth:`count_csvs` -- the input
        table's rows are ``{"plate", "score", "count", "database"}``, so the
        scores were always one field away too.

        A database whose ``png_list`` has no ``pred`` column is not necessarily
        a screen without scores:
        these files carry one row per cell and the fit was run on exactly
        those numbers. `load_montage_objects` joins them in memory when the
        database has none, and writes nothing.
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
            path = row.get("score", "") if isinstance(row, dict) else ""
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
            score_csvs=self.score_csvs(),
            picture=self.picture_settings(),
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
        if not self._multivariate_is_ready(request):
            return False
        self._pending = request
        self._drop_montage()
        self._set_status(
            f"Loading the cells behind {request.name}… reading "
            f"{len(request.databases)} database(s).")
        self._refresh_controls()
        self._jobs.submit(lambda r=request: load(r), self._on_loaded)
        return True

    def clear_picking_override(self) -> None:
        """Clear a temporary picking fallback.

        The next montage request will ask again if the selected picking method
        remains unavailable.
        """
        self._picking_override = ""

    def multivariate_shortfall(self, request=None) -> str:
        """Return why multivariate picking is unavailable, or ``""``.

        Multivariate picking requires a gene-by-measurement effects grid.
        The message directs the user to create that grid or explicitly choose
        rank-based picking rather than silently changing the selected method.
        """
        picture = self.picture_settings() or {}
        if str(picture.get("cell_picking") or "rank") != "multivariate":
            return ""
        from ...cell_montage import effects_grid_from_results

        request = request or self._pending
        path = getattr(request, "results_path", "") or self._results_path()
        if effects_grid_from_results(path) is not None:
            return ""
        return ("Multivariate picking needs a gene × measurement sweep: it "
                "reads one effect per measurement per guide, and this run "
                "has none beside it. Run the sweep on the Measurements tab "
                "and press Show again, or pick by rank instead.")

    def _multivariate_is_ready(self, request) -> bool:
        """Ask before running, when multivariate cannot do what was asked.

        Returns False only when the user chose to go and sort it out. The
        third way out is not optional: a sweep is long, and a user who does
        not want to wait needs a path forward that is not Cancel -- which is
        what the silent fallback was trying to be, in the wrong place.
        """
        shortfall = self.multivariate_shortfall(request)
        if not shortfall:
            return True
        asker = getattr(self, "_ask_about_multivariate", None)
        answer = asker(shortfall) if callable(asker) else self._ask(shortfall)
        if answer == "rank":
            # Their choice, recorded where the caption will read it, so the
            # montage says "rank" because it IS rank.
            self._force_picking("rank")
            return True
        return False

    def _ask(self, shortfall: str) -> str:
        """The prompt. Split out so a test can answer it without a modal."""
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setWindowTitle("The sweep has not been run")
        box.setText(shortfall)
        rank = box.addButton("Pick by rank instead",
                             QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        return "rank" if box.clickedButton() is rank else "cancel"

    def _force_picking(self, picking: str) -> None:
        """Use ``picking`` for this montage, whatever the settings say."""
        self._picking_override = str(picking)
        self._set_status(f"Picking by {picking}: the sweep multivariate "
                         f"picking needs has not been run.")

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
        # WHAT THESE CROPS ANSWER. A later settings change compares against
        # it and redraws in place when only the DISPLAY settings moved --
        # "if they have been loaded it should take a verry shourt amoutn of
        # time to change nd reapply the settings".
        self._loaded_signature = self._load_signature()
        self._sources = dict(result.sources)
        self._shown_key = self._key
        self._unavailable = ""
        # THE INVENTORY, WHICH NOTHING WAS EVER FILLING. `remember_inventory`
        # has existed since the picture-settings window learned to offer this
        # screen's own mask planes and object columns, and it had no caller:
        # `_last_objects` was None every time, so those choosers silently fell
        # back to their generic lists. It is called here because this is the
        # one place that HAS the answer.
        self.remember_inventory(objects=result.objects)
        self._counts = result.counts
        self._apply_shape_availability(result)
        self._fill()
        self._set_status(self._summary())
        self._refresh_controls()
        self._ensure_graph_tab()
        # THE WELLS MOVED WITH THE COEFFICIENT. A strategy panel still
        # showing the previous gene's wells would take its positives from
        # wells the montage on screen has nothing to do with.
        if self._annotation_panel is not None:
            self._annotation_panel.refresh()
        self.montage_ready.emit(result.n_objects)
        # AND THE NEXT ONE, if `build_every_selected` queued any. Chained
        # here rather than started together, so the tabs arrive in the order
        # the user picked the guides rather than in the order the disk
        # happened to answer.
        self._build_the_next_queued()

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

    # ------------------------------------------------------- picture settings

    def picture_mode(self) -> str:
        """The mode the user chose, as a stored crop-source value."""
        return str(self._source.currentData() or LOAD_IMAGES)

    def picture_settings(self) -> dict:
        """How the cells are drawn. The annotator's keys, the annotator's
        defaults, and whatever the user has changed."""
        from ..widgets.picture_settings_dialog import picture_defaults

        out = dict(picture_defaults())
        # The widgets are the source of truth for the ones that used to be on
        # the toolbar, so a value set before this window ever opened is shown
        # in it rather than replaced by a default.
        out.update({k: v for k, v in self._read_widgets().items()
                    if v not in (None, "")})
        out.update(self._picture_settings)
        out["crop_source"] = self.picture_mode()
        # THE PICKER THE USER AGREED TO, when they were told multivariate
        # could not run and chose rank rather than Cancel. Applied here and
        # not by editing their settings, because it is a decision about THIS
        # montage: their saved choice of multivariate is still what they
        # want once the sweep exists, and silently rewriting it would take
        # that away for every future run too.
        override = getattr(self, "_picking_override", "")
        if override:
            out["cell_picking"] = override
        return out

    def edit_picture_settings(self) -> bool:
        """Open the settings window. Returns whether anything was changed."""
        from ..widgets.picture_settings_dialog import PictureSettingsDialog

        # BUILT FROM THIS SCREEN. The mask planes come from whatever crop
        # source the montage resolved, and the coordinate columns from the
        # object table it actually loaded -- so the choosers list what is
        # there rather than asking the user to remember it.
        source = getattr(self, "_last_source", None)
        objects = getattr(self, "_last_objects", None)
        dialog = PictureSettingsDialog(values=self._picture_settings,
                                       mode=self.picture_mode(), parent=self,
                                       source=source, objects=objects)
        if dialog.exec() != QDialog.Accepted:
            return False
        # EVERY key, not only the ones this mode uses: a user who set a
        # streaming setting, switched to load images and switched back must
        # find it where they left it.
        self._picture_settings = dialog.values()
        # A NEW CHOICE OF PICKER DESERVES A NEW QUESTION. "rank, just this
        # once" must not quietly become permanent.
        self.clear_picking_override()
        # ONE SETTING, ONE VALUE. `channels` is read from the hidden field by
        # `channels()` and from the picture settings by the renderer, so the
        # window writes it back rather than letting the two drift.
        self._write_back(self._picture_settings)
        self._on_settings_changed()
        return True

    def write_scores_into_the_databases(self, *, confirm=None) -> dict:
        """Merge loaded per-object scores into attached databases on request.

        The montage can use score files without modifying a database. This
        method writes only after explicit confirmation.

        Parameters
        ----------
        confirm
            Optional callable receiving ``(databases, score_files)`` and
            returning whether to proceed. When omitted, spaCR displays a
            confirmation dialog.
        Returns
        -------
        dict
            Mapping of each updated database path to its matched-row count.
            Returns an empty mapping when no inputs are available, the user
            declines, or no database can be updated.
        """
        import pandas as pd

        databases = [str(p) for p in self.databases()]
        score_files = [str(p) for p in self.score_csvs()]
        if not databases or not score_files:
            self._set_status(
                "Nothing to merge: attach at least one database and load at "
                "least one per-object score file.")
            return {}
        if confirm is None:
            confirm = self._ask_before_writing
        if not confirm(databases, score_files):
            return {}

        from ...predictions import merge_cv_predictions

        frame = pd.concat([pd.read_csv(path) for path in score_files],
                          ignore_index=True)
        written: dict = {}
        for database in databases:
            try:
                report = merge_cv_predictions(frame, database, verbose=False)
            except Exception as error:                      # noqa: BLE001
                LOG.debug("could not merge scores into %s", database,
                          exc_info=True)
                self._set_status(f"{os.path.basename(database)}: "
                                 f"{type(error).__name__}: {error}")
                continue
            written[database] = getattr(report, "matched", 0) if report else 0
        if written:
            total = sum(written.values())
            database_label = "database" if len(written) == 1 else "databases"
            row_label = "row" if total == 1 else "rows"
            self._set_status(
                f"Merged the run's scores into {len(written)} {database_label}; "
                f"{total} {row_label} matched. The montage already uses "
                "loaded scores in memory, so the displayed montage is "
                "unchanged.")
        return written

    def _ask_before_writing(self, databases, score_files) -> bool:
        """Ask whether to write the listed score files to the databases.

        Consent is the Write button and nothing else: a window closed by its
        title bar, or rejected programmatically, clicked no button at all and
        counts as a refusal.
        """
        from PySide6.QtWidgets import QMessageBox

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Write scores to databases?")
        database_label = "database" if len(databases) == 1 else "databases"
        box.setText(f"Write the run's per-object scores to {len(databases)} "
                    f"measurement {database_label}?")
        box.setInformativeText(
            "Score files:\n"
            + ", ".join(os.path.basename(p) for p in score_files[:4])
            + (f" (+{len(score_files) - 4} more)" if len(score_files) > 4
               else "")
            + "\n\nDatabases:\n"
            + ", ".join(os.path.basename(p) for p in databases[:4])
            + (f" (+{len(databases) - 4} more)" if len(databases) > 4 else "")
            + "\n\nThis writes classification scores to the databases. The "
              "montage can already use the loaded files without this step; "
              "continue only if another workflow needs the scores stored in "
              "the databases.")
        proceed = box.addButton("Write scores", QMessageBox.AcceptRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(cancel)
        box.exec()
        return box.clickedButton() is proceed

    def picked_groups(self) -> dict:
        """``{gene: the object index values this tab picked for it}``.

        THE PICKER'S OWN ANSWER, read off the plans rather than recomputed --
        `montage_candidate` is the column `select_montage` marks, so
        whichever mode is in force (rank, attributed, assigned,
        multivariate) this is what the montage actually drew.
        """
        out: dict = {}
        for plan in self.plans():
            rows = getattr(plan, "objects", None)
            if rows is None or not len(rows):
                continue
            name = str(getattr(plan.coefficient, "name", "") or "picked")
            if "montage_candidate" in rows.columns:
                chosen = rows.loc[rows["montage_candidate"].astype(bool)]
            else:
                chosen = rows
            if len(chosen):
                out.setdefault(name, []).extend(list(chosen.index))
        return out

    # -- instruction 180: what this panel contributes to a saved run --------

    def workspace_state(self) -> dict:
        """The montage the session had open, as data.

        THE SETTINGS AND THE CHOICE, NOT THE PIXELS. A montage is tens of
        megabytes of crops that the run's own images regenerate exactly; what
        cannot be regenerated is which coefficient was on screen, how the
        cells were picked, and how they were drawn. Those are what go in.

        The picked groups ride along as a RECORD, not as an input --
        `picked_groups()` is what the picker chose given these settings, and
        restoring the settings reproduces it. Written down because a reader
        of a saved run wants to know which cells the claim rested on without
        re-running anything.
        """
        return {
            "coefficient": str(self._key or ""),
            "level": str(self._level or ""),
            "results_path": self._results_path(),
            "widgets": self._read_widgets(),
            "picture_settings": dict(self._picture_settings),
            "picture_mode": self.picture_mode(),
            "picked_groups": {gene: list(values)
                              for gene, values in self.picked_groups().items()},
            "montage_shown": bool(self._plans),
        }

    def apply_workspace_state(self, state) -> bool:
        """Put the montage's settings back. Does NOT rebuild it.

        Returns whether anything was applied.

        DELIBERATELY NOT REBUILT. Loading the crops is the slow half -- the
        first montage of a run reads images off disk for seconds -- and a
        restore that started it would freeze a window the user had just
        opened to look around in. The settings are put back and the button is
        there; 155's "the montage says how it chose" is on screen either way.
        """
        if not isinstance(state, dict):
            return False
        applied = False
        picture = state.get("picture_settings")
        if isinstance(picture, dict):
            self._picture_settings = dict(picture)
            self._write_back(self._picture_settings)
            applied = True
        widgets = state.get("widgets")
        if isinstance(widgets, dict):
            self._write_back(widgets)
            applied = True
        mode = state.get("picture_mode")
        if mode:
            index = self._source.findData(mode)
            if index >= 0:
                self._source.setCurrentIndex(index)
                applied = True
        key = state.get("coefficient")
        if key:
            # LAST, and through the setter. It re-reads the frame and rebuilds
            # the level and the effect from it, so a coefficient applied
            # before the widgets would be described by the old settings.
            self.set_coefficient(str(key))
            applied = True
        return applied

    def compare_a_measurement(self, *_args):
        """Open the Compare tab for the cells selected by this montage.

        Returns the shared comparison panel, or ``None`` when no cells have
        been selected or the panel cannot be created. Reusing the tab keeps
        the comparison synchronized with the montage and avoids duplicate
        floating views.
        """
        rows = self._all_objects()
        groups = self.picked_groups()
        if rows is None or not len(rows) or not groups:
            self._set_status(
                "Show some cells first — the comparison groups them by what "
                "the picker chose, and nothing is picked yet.")
            return None
        self._ensure_graph_tab()
        if self._graph_panel is None:                    # pragma: no cover
            return None
        index = self._tabs.indexOf(self._graph_panel)
        if index >= 0:
            self._tabs.setCurrentIndex(index)
        return self._graph_panel

    def _all_objects(self):
        """Every object row behind the montage, picked or not."""
        frames = [getattr(plan, "objects", None) for plan in self.plans()]
        frames = [f for f in frames if f is not None and len(f)]
        if not frames:
            return None
        import pandas as pd

        return pd.concat(frames) if len(frames) > 1 else frames[0]

    def rows_to_compare(self):
        """Return object rows available to the measurement comparison panel.

        Returns
        -------
        pandas.DataFrame or None
            The full object inventory when it contains every montage-plan index;
            otherwise only the concatenated plan rows. ``None`` is returned
            when no plan contains object rows.

        Notes
        -----
        Control-well and other-well contrasts require objects that may not be
        displayed in the montage. The wider inventory is used only when its
        index preserves the plan rows' group identities.
        """
        picked = self._all_objects()
        everything = getattr(self, "_last_objects", None)
        if picked is None or everything is None or not len(everything):
            return picked
        try:
            covered = bool(picked.index.isin(everything.index).all())
        except Exception:                                    # noqa: BLE001
            return picked
        return everything if covered else picked

    def remember_inventory(self, source=None, objects=None) -> None:
        """Keep what the last load resolved, so the settings window can offer
        THIS screen's mask planes and object columns rather than free text."""
        if source is not None:
            self._last_source = source
        if objects is not None:
            self._last_objects = objects

    #: settings key -> the hidden widget that is still the source of truth.
    #: `request()` and the run's saved state read the WIDGETS, so the settings
    #: window writes back rather than letting the two drift -- one setting,
    #: one value, wherever it is edited.
    _MIRRORED = {
        "channels": "_channels",
        "object_type": "_object",
        "crop_source": "_source",
        "crop_shape": "_shape",
        "half_widths": "_half_widths",
        "baseline": "_baseline",
        "score_column": "_score",
        "cap": "_cap",
    }

    def _write_back(self, values) -> None:
        """Put what the settings window chose onto the widgets that read it."""
        from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QLineEdit,
                                       QSpinBox)

        for key, name in self._MIRRORED.items():
            if key not in (values or {}):
                continue
            widget = getattr(self, name, None)
            value = values[key]
            if widget is None or value is None:
                continue
            try:
                if isinstance(widget, QComboBox):
                    index = widget.findData(value)
                    if index < 0:
                        index = widget.findText(str(value))
                    if index >= 0:
                        widget.setCurrentIndex(index)
                elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                    widget.setValue(type(widget.value())(value))
                elif isinstance(widget, QLineEdit):
                    text = (", ".join(str(v) for v in value)
                            if isinstance(value, (list, tuple)) else str(value))
                    if text != widget.text():
                        widget.setText(text)
            except (TypeError, ValueError):
                continue

    def _read_widgets(self) -> dict:
        """What the hidden widgets currently hold, in settings terms."""
        from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QLineEdit,
                                       QSpinBox)

        out = {}
        for key, name in self._MIRRORED.items():
            widget = getattr(self, name, None)
            if widget is None:
                continue
            if isinstance(widget, QComboBox):
                data = widget.currentData()
                out[key] = widget.currentText() if data is None else data
            elif isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                out[key] = widget.value()
            elif isinstance(widget, QLineEdit):
                out[key] = widget.text()
        return out

    def _on_mode_changed(self, *_args) -> None:
        """Say which mode is in force, so a fallback is never silent."""
        label = picture_source_label(self.picture_mode())
        try:
            self._status.setText(f"Images: {label}.")
        except Exception:                                    # noqa: BLE001
            pass

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
        # `clicked` CARRIES A BOOL and this takes an optional first argument,
        # so Qt hands the checked state into `path`. `False is None` is False,
        # so the dialog never opened and `False` went on to the writer -- the
        # same fault that reached the user out of FastPlotWidget.export as
        # "QImage.save(bool)". See that method.
        if isinstance(path, bool):
            path = None
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
        """Disable crop shapes that the loaded source cannot produce.

        Sources without masks cannot create outline-following crops. Shape
        availability is therefore applied after source discovery, with the
        unavailable entries disabled and labeled with the reason.
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
        showing = bool(self._plans)
        self._jobs.cancel()
        self._pending = None
        # And a shape greyed out by the LAST route is re-armed: forcing
        # 'merged' after a run whose PNGs are gone is exactly the case where
        # a remembered refusal would keep a real choice unavailable.
        self._apply_shape_availability(MontageLoad())
        # A SETTING CHANGED WHILE CELLS WERE ON SCREEN MEANS REDRAW THEM.
        # Reported 2026-08-19: "after the cells are loaded it looks like i
        # cannot reapply the settings". This cancelled the load in flight and
        # stopped, so a montage already drawn kept the OLD settings and
        # nothing said why -- indistinguishable from a control that does
        # nothing.
        if showing:
            # REDRAW FROM THE CROPS ALREADY IN HAND WHERE THAT IS ENOUGH.
            # Reported 2026-08-19: "if they have been loaded it should take a
            # verry shourt amoutn of time to change nd reapply the settings
            # ... i think the current behaviour is that they are reloaded
            # every time something changes" -- which it was.
            #
            # `picture_settings` already separates the settings that decide
            # what is CUT from disk (channels, size, crop shape, object type,
            # source) from the ones that only decide how an obtained crop is
            # DRAWN (normalise, outline, edge, percentiles). Only the first
            # kind can need new pixels.
            if self._can_redraw_without_loading():
                self._redraw_from_cache()
            else:
                self.build()
        self._refresh_controls()
        self._announce()

    def _load_signature(self) -> tuple:
        """Everything that decides WHICH PIXELS are read off disk.

        The display settings are deliberately absent: two requests differing
        only in `normalize_channels` want the same crops and a different
        picture of them.
        """
        picture = self.picture_settings()
        cut = {k: picture.get(k) for k in
               ("crop_source", "image_type", "img_size", "channels",
                "crop_shape", "object_array", "coordinate_columns")}
        return (
            str(self._name), str(self._level), repr(sorted(cut.items())),
            str(self._object.currentData() or ""),
            self._channels.text().strip(),
            str(self._source.currentData() or ""),
            str(self._score.text().strip()),
            str(self._baseline.currentData() or ""),
            float(self._half_widths.value()), int(self._cap.value()),
            str(picture.get("cell_picking") or ""),
            str(picture.get("picking_threshold") or ""),
            bool(picture.get("show_all_in_well")),
        )

    def _can_redraw_without_loading(self) -> bool:
        """Whether the crops in hand still answer the current settings."""
        return (bool(self._plans) and bool(self._images)
                and self._loaded_signature == self._load_signature())

    def _ensure_graph_tab(self) -> None:
        """Put the Graph tab beside Summary once there is something to graph.

        Built on the first montage rather than at construction: the panel
        needs the object rows and the picker's groups, and a tab that said
        "nothing yet" would be a second way of saying what the Summary tab
        already says.
        """
        rows = self.rows_to_compare()
        groups = self.picked_groups()
        if rows is None or not len(rows) or not groups:
            return
        from .measurement_compare_dialog import MeasurementComparePanel

        if self._graph_panel is None:
            self._graph_panel = MeasurementComparePanel(
                rows, groups, parent=self._tabs,
                settings=self.picture_settings(),
                databases=self.databases(),
                counts=getattr(self, "_counts", None))
            # AFTER Summary, which is index 0, and before any well tab.
            # NAMED FOR THE BUTTON THAT OPENS IT. It was "Graph", and the
            # control the user presses says "Compare a measurement" -- one
            # thing under two names reads as two things.
            self._graph_tab = self._tabs.insertTab(1, self._graph_panel,
                                                   "Compare")
            self._hide_close_button(1)
        else:
            self._graph_panel.set_data(rows, groups,
                                       settings=self.picture_settings())

    def annotate_the_cells(self, *_args):
        """Open annotation strategies for the cells in the current montage.

        :returns: The annotation-strategy panel, constructed on first use.
        """
        # BUILT BEFORE THE TAB IS RAISED, so the hundred widgets go into a
        # page Qt is not in the middle of showing.
        panel = self._ensure_annotation_panel()
        index = self._tabs.indexOf(self._annotation_page)
        if index >= 0:
            self._tabs.setCurrentIndex(index)
        if panel is not None:
            panel.refresh()
        return panel

    def _on_tab_changed(self, index: int) -> None:
        """Fill the Annotate tab the first time somebody opens it.

        The build is POSTED rather than done here. This runs inside Qt's own
        tab change; filling the page being shown while that is still
        unwinding is reentrancy the builder is kept out of on purpose.
        """
        if self._tabs.widget(index) is not self._annotation_page:
            return
        if self._annotation_panel is not None:
            self._annotation_panel.refresh()
            return
        QTimer.singleShot(0, self._fill_the_annotation_tab)

    def _fill_the_annotation_tab(self) -> None:
        """Build the strategy panel and point it at what is on screen."""
        panel = self._ensure_annotation_panel()
        if panel is not None:
            panel.refresh()

    def _ensure_annotation_panel(self):
        """The strategy panel, built into the Annotate tab on first use.

        Built on opening rather than at construction because the panel is
        forty controls and a fitting runner, and a montage nobody annotates
        should not pay for them. The TAB is there either way, so the
        strategies are a named place rather than a hidden one.
        """
        if self._annotation_panel is not None:
            return self._annotation_panel
        from .annotation_strategy_panel import AnnotationStrategyPanel

        try:
            panel = AnnotationStrategyPanel(
                objects_provider=self.rows_to_compare,
                wells_provider=self._chosen_wells,
                score_provider=lambda: (self._score.text().strip()
                                        or DEFAULT_SCORE_COLUMN),
                folder_provider=self._annotation_folder,
                parent=self._annotation_page,
                threaded=self._threaded)
        except Exception:
            LOG.exception("Could not build the annotation strategies")
            return None
        layout = self._annotation_page.layout()
        if self._annotation_placeholder is not None:
            # HIDDEN, NOT DELETED. Destroying a widget out of the layout of
            # the page on screen is a teardown worth not asking Qt for, and a
            # hidden label costs one widget.
            self._annotation_placeholder.setVisible(False)
            self._annotation_placeholder = None
        layout.addWidget(panel, 1)
        # THE OUTCOME REACHES THE STATUS LINE, so a user who ran a strategy
        # and went back to the pictures is told it landed rather than having
        # to go and look.
        panel.finished.connect(self._on_annotation_finished)
        self._annotation_panel = panel
        return panel

    def _on_annotation_finished(self, key: str) -> None:
        """Say in the status line that a strategy has produced a result."""
        panel = self._annotation_panel
        result = panel.result() if panel is not None else None
        if result is None:
            return
        chosen = sum(n for role, n in result.role_counts().items()
                     if role != "holdout")
        self._set_status(
            f"Annotate: {result.title} chose {chosen:,} cell(s). The numbers "
            "it is allowed to claim are on the Annotate tab.")

    def _chosen_wells(self) -> Tuple[str, ...]:
        """The guide wells the montage on screen picked its cells from."""
        from .annotation_strategy_panel import wells_of_plans

        return wells_of_plans(self.plans())

    def _annotation_folder(self) -> str:
        """Where a saved annotation selection should go by default."""
        path = self._results_path()
        if not path:
            return ""
        return os.path.dirname(path) if os.path.isfile(path) else path

    def _hide_close_button(self, index: int) -> None:
        """A tab the user cannot close needs no x on either side."""
        from PySide6.QtWidgets import QTabBar

        bar = self._tabs.tabBar()
        for side in (QTabBar.LeftSide, QTabBar.RightSide):
            try:
                bar.setTabButton(index, side, None)
            except Exception:                                # noqa: BLE001
                continue

    def _redraw_from_cache(self) -> None:
        """Draw the crops already loaded, with the display settings as they
        are now. The whole point: no disk, no worker, no wait."""
        self._fill()
        self._set_status(self._summary())
        self.montage_ready.emit(sum(len(row) for row in self._images))

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

        # AND THE COMPARE BUTTON, which had neither (reported 2026-08-21:
        # "now i press compare a measurement and nothing hapens").
        #
        # It was ENABLED with nothing to compare, and its refusal went to
        # the status line -- which is exactly a button that appears to do
        # nothing. Show and Save have greyed with a reason since they were
        # written; this one was simply missed, and the same rule applies:
        # a control that cannot act says so before it is pressed rather
        # than after.
        # THE ANNOTATE TAB GREYS ITS OWN RUN BUTTON, with the reason, so it
        # follows the montage the same way the Compare tab does.
        if self._annotation_panel is not None:
            self._annotation_panel.refresh()

        comparable = bool(self._plans) and bool(self.picked_groups())
        self._compare_button.setEnabled(comparable)
        self._compare_button.setToolTip(
            "Compare any measurement between the cells this tab picked for "
            "each gene and the rest of the screen. Cell, well or plate "
            "level; five ways of drawing it; the test chosen from the "
            "normality and variance checks and reported with n; and one "
            "folder holding the figure, the data, the statistics and the "
            "settings." if comparable else
            "There is nothing to compare yet — press “Show the cells” "
            "first. The comparison groups the cells by what the picker "
            "chose, and nothing is picked until a montage is loaded.")

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
                # THE COUNT, WHENEVER THERE IS MORE THAN ONE. A selection you
                # cannot count is one you cannot trust (instruction 206), and
                # a grid showing one guide out of four with nothing saying so
                # reads as a grid of the whole selection.
                selection = self.selected_coefficients()
                more = (f" {len(selection)} coefficients are selected; this "
                        f"grid is one of them."
                        if len(selection) > 1 else "")
                self._set_status(
                    f"Ready: press “Show the cells” for {self._name}"
                    + (f", from the run {run}." if run else ".") + more)

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
        self._loaded_signature = None
        self._shown_key = ""
        self._caption.setPlainText("")

    def well_tabs(self) -> Tuple["_WellTab", ...]:
        """The open well tabs, in the order they are on screen."""
        return tuple(self._tabs.widget(i) for i in range(1, self._tabs.count())
                     if isinstance(self._tabs.widget(i), _WellTab))

    def tab_labels(self) -> Tuple[str, ...]:
        """Every tab's text, summary first -- what a user reads across."""
        return tuple(self._tabs.tabText(i) for i in range(self._tabs.count()))

    def crop_count(self) -> int:
        """Return the total number of crops held across all tab pages.

        This differs from ``len(thumbnails())``, which counts only widgets on
        the pages currently displayed.
        """
        return sum(len(tab.crops()) for tab in self.well_tabs())

    def thumbnails(self) -> Tuple[QWidget, ...]:
        """Return thumbnails visible on the currently displayed tab pages.

        Use :meth:`crop_count` to obtain the total number of objects held
        across all well tabs, including pages that are not displayed.
        """
        out: List[QWidget] = []
        for tab in self.well_tabs():
            out.extend(t for t in tab.thumbs() if t is not None)
        return tuple(out)

    def _open_well_tab(self, key: Tuple[str, ...], label: str,
                       tooltip: str) -> "_WellTab":
        """Add a well tab with a close control on its left edge.

        The close handler captures the widget rather than its mutable tab
        index, so closing an earlier tab cannot redirect a later button.
        """
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
        close = close_mark_button(
            bar,
            tooltip=(
                f"Close “{tab.label}”. This tab closes from here and "
                "nowhere else — it survives another coefficient, a re-sort "
                "and a re-run, because comparing two genes' cells side by "
                "side is the point."))
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
        """Cells per row. The preference, not the tab's width."""
        try:
            from ..preferences import get_montage_columns

            return max(1, int(get_montage_columns()))
        except Exception:                                    # noqa: BLE001
            return max(1, int(DEFAULT_MONTAGE_COLUMNS))

    def _fill(self) -> None:
        """Open or refresh one tab per well, and write the summary.

        A well the plans no longer mention keeps its tab -- it closes by its
        x and by nothing else -- and a well that is mentioned again refreshes
        the tab it already has rather than opening a second one.
        """
        self._columns = self._column_count()
        # THE VIEW'S OWN SETTINGS. `_fill` runs from `self._plans` and has no
        # request in scope -- reaching for one raised NameError on the real
        # screen the moment a montage was drawn, after the run had succeeded.
        picture = self.picture_settings()
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
                                self._columns,
                                thumb_px=_thumb_px_of(picture),
                                per_page=_per_page_of(picture),
                                picture=picture)
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
        if self._annotation_panel is not None:
            self._annotation_panel.shutdown()
        self._jobs.shutdown()

    def closeEvent(self, event):         # noqa: N802 - Qt's spelling
        """Shut the loader down before the widget goes."""
        self.shutdown()
        super().closeEvent(event)
