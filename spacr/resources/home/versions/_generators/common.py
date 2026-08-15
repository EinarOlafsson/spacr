"""Shared bootstrap + data for the Home-screen variant generators.

These modules render THIRTY candidate Home screens out of the real Qt
widgets (``spacr.qt.widgets.tile.HTile``, ``Card``, ``Section``, …) and
the real app registry (``spacr.qt.app.APPS``), then grab each one to a
PNG under ``spacr/resources/home/versions/``.

Nothing here is installed into the app. It is a review surface: the
user picks a layout, and because every layout is built from real
widgets, whatever they pick is known-buildable.

Determinism
-----------
* ``QT_QPA_PLATFORM=offscreen`` and a throwaway ``QSettings`` path, so
  the renders do not depend on the reviewer's saved theme / font scale.
* No live system state. Every "recent run", disk figure and GPU number
  is a fixed literal in :data:`MOCK` — a home screen that renders
  differently every run cannot be reviewed.
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import Dict, List, Sequence, Tuple

#: The realistic laptop case. Every variant is grabbed at exactly this.
CANVAS_W = 1440
CANVAS_H = 900


def here() -> str:
    """Absolute path of this ``_generators`` directory."""
    return os.path.dirname(os.path.abspath(__file__))


def versions_dir() -> str:
    """Absolute path of ``spacr/resources/home/versions``."""
    return os.path.normpath(os.path.join(here(), ".."))


def repo_root() -> str:
    """Absolute path of the spacr checkout root (five levels up)."""
    return os.path.normpath(os.path.join(here(), *([".."] * 5)))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap():
    """Create (or return) the offscreen QApplication, isolated + fonted.

    Redirects ``QSettings`` at a temp directory *before* anything reads
    a preference, so ``preferences.get_font_scale()`` (which
    ``HTile``/``scaled_px`` consult) always answers 1.0 regardless of
    what the reviewer has saved.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    root = repo_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    # QSettings.setDefaultFormat / setPath are PROCESS-GLOBAL. Redirecting
    # them when we did not create the QApplication reaches into a host that
    # is already running: under pytest-qt it repoints every other test's
    # preferences at a temp directory mid-session, which is how this file
    # took the whole tests/qt suite down with a segfault. Only isolate when
    # this really is our own standalone process.
    global _WE_OWN_THE_APP
    app = QApplication.instance()
    if app is None:
        # NativeFormat as well as Ini. `preferences._settings()` builds
        # `QSettings("spacr", "qt")`, which is a NativeFormat object and
        # ignores setDefaultFormat/setPath(IniFormat, ...) — redirecting only
        # Ini left every render reading the operator's own saved font scale
        # and theme, so "deterministic" renders differed per machine.
        sandbox = tempfile.mkdtemp(prefix="spacr-home-variants-")
        QSettings.setDefaultFormat(QSettings.IniFormat)
        for fmt in (QSettings.NativeFormat, QSettings.IniFormat):
            QSettings.setPath(fmt, QSettings.UserScope, sandbox)
        app = QApplication(sys.argv[:1])
        _WE_OWN_THE_APP = True
    _load_fonts()
    return app


def _load_fonts() -> None:
    """Register the bundled Open Sans faces so metrics match the app."""
    from PySide6.QtGui import QFontDatabase
    fonts = os.path.join(repo_root(), "spacr", "qt", "resources", "fonts")
    if not os.path.isdir(fonts):
        return
    for name in sorted(os.listdir(fonts)):
        if name.lower().endswith((".ttf", ".otf")):
            QFontDatabase.addApplicationFont(os.path.join(fonts, name))


def available_themes() -> Tuple[str, ...]:
    """Themes to render: dark + light, plus space when the palette exists.

    ``space`` is another agent's work-in-progress; this probes for it
    rather than depending on it.
    """
    out = ["dark", "light"]
    try:
        from spacr.qt.theme import palette_for
        pal = palette_for("space")
        if isinstance(pal, dict) and pal.get("bg") and pal is not palette_for("dark"):
            out.append("space")
    except Exception:
        pass
    return tuple(out)


# ---------------------------------------------------------------------------
# The real app registry
# ---------------------------------------------------------------------------

def _registry():
    from spacr.qt.app import APPS, _ICON_OVERRIDES, _FORCE_GLYPH
    return APPS, _ICON_OVERRIDES, _FORCE_GLYPH


def apps() -> List[Tuple[str, str, str, str]]:
    """The real ``(key, name, blurb, section)`` list, unmodified."""
    return list(_registry()[0])


def app_map() -> Dict[str, Tuple[str, str, str, str]]:
    """``key -> (key, name, blurb, section)``."""
    return {row[0]: row for row in apps()}


def name_of(key: str) -> str:
    """Display name of an app key."""
    return app_map()[key][1]


def blurb_of(key: str) -> str:
    """One-line description of an app key."""
    return app_map()[key][2]


def all_keys() -> List[str]:
    """Every app key, in registry order."""
    return [row[0] for row in apps()]


def core_keys() -> List[str]:
    """The Core-pipeline app keys, in registry order.

    Read from :data:`spacr.qt.app.SECTION_CORE` rather than compared
    against a typed section name. The section used to be called "Core
    pipeline"; it is now "Core", and the two variants that filtered on
    the old string silently produced an empty list — variant 18, whose
    entire content is the core nine, rendered nine missing tiles, and
    variant 24 lost every Ctrl+N badge.
    """
    from spacr.qt.app import SECTION_CORE
    return [row[0] for row in apps() if row[3] == SECTION_CORE]


def n_apps() -> int:
    """How many apps the registry holds *right now*.

    Every "N apps" the variants draw or write goes through here. The
    count used to be typed into two dozen strings as ``29``; the
    registry then grew Distributed Jobs, Classifier Evaluation and Run
    History and every one of those strings became a lie that no test
    could see, because a literal cannot disagree with itself.
    """
    return len(all_keys())


#: Icons are re-inked per theme by ``iconset`` (a PIL + numpy pass per
#: PNG), which is far too slow to repeat for each of the thirty
#: variants. Cached across contexts, keyed by ``(theme, key)``.
_ICON_CACHE: Dict[Tuple[str, str], object] = {}
_PIXMAP_CACHE: Dict[Tuple[str, str, int], object] = {}
_LOGO_CACHE: Dict[Tuple[str, int], object] = {}

#: True only when :func:`bootstrap` created the QApplication itself. When we
#: are a guest inside someone else's (pytest-qt), application-wide restyling
#: is off limits -- see :meth:`Ctx.apply_theme`.
_WE_OWN_THE_APP = False


class Ctx:
    """Per-theme rendering context: palette, stylesheet, icon cache."""

    def __init__(self, app, theme: str):
        from spacr.qt.theme import palette_for
        self.app = app
        self.theme = theme
        self.P = palette_for(theme)

    def qss(self) -> str:
        """This theme's stylesheet.

        background=None: the Space theme degrades to its gradient sky rather
        than depending on a cached generated image.
        """
        from spacr.qt.theme import stylesheet
        return stylesheet(self.theme, 1.0, background=None)

    def apply_theme(self, target=None) -> None:
        """Apply this theme, to ``target`` if given, else the application.

        QApplication.setStyleSheet re-polishes EVERY top-level widget. Inside
        pytest-qt the application is shared, so widgets belonging to other
        tests -- including ones mid-teardown whose C++ side is already gone --
        get re-polished, and the process SEGFAULTS. That is what took the
        tests/qt suite down here, and it is the same trap the theme work hit
        with QApplication.topLevelWidgets().

        A stylesheet set on a widget cascades to its children, so styling the
        root being rendered is equivalent for our purposes and touches nothing
        else. The application-wide path is used only when bootstrap() created
        the application, i.e. in the standalone generator.
        """
        from spacr.qt.theme import apply_qpalette
        if not _WE_OWN_THE_APP:
            # Guest inside someone else's QApplication: never touch it.
            if target is not None:
                target.setStyleSheet(self.qss())
            return
        apply_qpalette(self.app, self.theme)
        self.app.setStyleSheet(self.qss())

    def icon(self, key: str):
        """A themed :class:`QIcon` for an app key (same rules as the app)."""
        cache_key = (self.theme, key)
        if cache_key not in _ICON_CACHE:
            from spacr.qt import iconset
            _apps, overrides, force_glyph = _registry()
            if key in force_glyph:
                ic = iconset.icon(key, theme=self.theme)
            else:
                ic = iconset.app_icon(key, override=overrides.get(key),
                                      theme=self.theme)
            _ICON_CACHE[cache_key] = ic
        return _ICON_CACHE[cache_key]

    def pixmap(self, key: str, px: int):
        """The app icon rendered to a ``px`` square pixmap."""
        from PySide6.QtCore import QSize
        cache_key = (self.theme, key, px)
        if cache_key not in _PIXMAP_CACHE:
            _PIXMAP_CACHE[cache_key] = self.icon(key).pixmap(QSize(px, px))
        return _PIXMAP_CACHE[cache_key]

    def logo(self, px: int):
        """The spaCR wordmark logo, re-inked for this theme."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from spacr.qt.iconset import themed_pixmap
        cache_key = (self.theme, px)
        if cache_key in _LOGO_CACHE:
            return _LOGO_CACHE[cache_key]
        path = os.path.join(repo_root(), "spacr", "resources", "icons",
                            "logo_spacr.png")
        if not os.path.isfile(path):
            return None
        pix = themed_pixmap(path, self.theme) or QPixmap(path)
        if pix.isNull():
            return None
        scaled = pix.scaled(px, px, Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
        _LOGO_CACHE[cache_key] = scaled
        return scaled


# ---------------------------------------------------------------------------
# Mock content for the elements that do not exist yet
# ---------------------------------------------------------------------------
# Fixed literals, never live state — see the module docstring. Anything
# drawn from these is *proposed* UI, not something spaCR reports today.
MOCK = {
    "project":   "toxo_mito_screen",
    "plates":    "12 plates",
    "images":    "48 320 images",
    "objects":   "1.42 M objects",
    "version":   "1.3.6",
    "last_run":  ("measure", "plate_07", "finished 18 min ago"),
    "recent": [
        ("mask",     "plate_07",  "18 min ago",  True,  "22 m 04 s"),
        ("measure",  "plate_07",  "1 h ago",     True,  "41 m 11 s"),
        ("classify", "plate_06",  "yesterday",   False, "3 m 27 s"),
        ("annotate", "plate_06",  "yesterday",   True,  "—"),
    ],
    "system": [("GPU", 41, "RTX 4090"), ("VRAM", 62, "14.9 / 24 GB"),
               ("Disk", 68, "1.2 TB free"), ("RAM", 35, "22 / 64 GB")],
    "whats_new": [
        "Mask now runs on the Cellpose 4 (SAM) backend.",
        "Invasion Assay: two-colour outside/inside scoring.",
        "Model Zoo benches a model on three of your own fields.",
        "Report writes a shareable HTML/PDF with the QC verdict.",
    ],
    "queue": [("plate_08", "Mask → Measure", "queued"),
              ("plate_09", "Mask → Measure", "queued"),
              ("plate_10", "Measure", "queued")],
}

#: Six apps a returning user is assumed to have pinned.
PINNED = ["mask", "measure", "annotate", "classify", "plate_view", "report"]

#: Invented but plausible run counts, used by the frequency-ordered
#: variants. Labelled as such wherever they are drawn.
#:
#: Must name every key in the registry: :func:`by_frequency` sorts on
#: ``USE_COUNTS.get(k, 0)``, so an app missing here silently sinks to
#: the bottom of every frequency-ordered variant instead of failing.
USE_COUNTS = {
    "mask": 412, "measure": 388, "annotate": 250, "classify": 164,
    "plate_view": 131, "db_browser": 118, "ml_analyze": 96, "report": 88,
    "queue": 71, "batch": 64, "map_barcodes": 59, "regression": 52,
    "convert": 47, "umap": 41, "make_masks": 38, "run_history": 35,
    "timelapse": 33, "graph_builder": 31, "model_zoo": 29,
    "illumination": 28, "cellpose_masks": 27, "barcode_qc": 25,
    "align": 24, "layer_viewer": 23, "distributed_jobs": 22, "foreign": 21,
    "external_masks": 20, "agreement": 19, "activation": 17,
    "train_compare": 15, "model_compare": 13,
    "classifier_evaluation": 12, "motility": 11,
    "recruitment": 9, "analyze_plaques": 8, "invasion": 6,
    "replication": 6, "train_cellpose": 5,
}

#: What an app nobody has invented a count for is given. Below the
#: smallest real entry, so a newcomer sorts to the bottom of every
#: frequency-ordered variant, which is where a brand-new app belongs.
UNUSED_APP_COUNT = 4

for _key in all_keys():
    # Variant 14 reads ``USE_COUNTS[k]`` for the badge on every tile, so a
    # key missing here is not "sorts to the bottom", it is a KeyError that
    # takes all thirty variants down. That is a hand-edit a module which
    # registers itself from its own file cannot make, so the table fills
    # itself and the literals above stay a statement about the apps
    # somebody actually had an opinion on.
    USE_COUNTS.setdefault(_key, UNUSED_APP_COUNT)
del _key


# ---------------------------------------------------------------------------
# Categorisations — every one covers every real app key exactly once
# ---------------------------------------------------------------------------

def cats_current() -> "List[Tuple[str, List[str]]]":
    """Today's five sections, straight out of ``spacr.qt.app``."""
    from spacr.qt.app import SECTIONS
    grouped = {s: [] for s in SECTIONS}
    for key, _n, _d, section in apps():
        grouped.setdefault(section, []).append(key)
    return [(s, grouped[s]) for s in SECTIONS if grouped.get(s)]


def _with_late_registrations(
    cats: "Sequence[Tuple[str, Sequence[str]]]",
    fallback: str,
) -> "List[Tuple[str, List[str]]]":
    """``cats`` with every uncategorised registry key added to ``fallback``.

    Each table below is a hand-made judgement about where an app belongs
    and stays one: an app named in it lands where it was put. What this
    adds is that an app NOBODY has filed — one that registered itself
    from its own module after these literals were written — lands
    somewhere real instead of making :func:`check_coverage` raise and
    taking all thirty variants down with it.

    The fallback band is chosen per table as the one whose question a
    brand-new app is most likely to answer, and landing there is a
    prompt to file it properly, not an answer. Note that this does NOT
    relax the width rules: a band that overflows its grid still fails
    ``test_no_stage_band_exceeds_the_seven_column_grid_by_more_than_a_row``,
    which is the point — the layout decision has to be made by a person.

    :param cats: the literal categorisation, ``(title, keys)`` per band.
    :param fallback: the title of the band unfiled keys are appended to.
    :returns: a fresh list; the literal is not mutated.
    """
    placed = {key for _title, keys in cats for key in keys}
    missing = [key for key in all_keys() if key not in placed]
    result = [(title, list(keys)) for title, keys in cats]
    if missing:
        for title, keys in result:
            if title == fallback:
                keys.extend(missing)
                break
    return result


CATS_BROAD3 = _with_late_registrations([
    # Power / Design is the only app in the registry that runs BEFORE the
    # images exist. "Prepare" is the closest of these three to that, and
    # it is where a screener would look for it.
    ("Prepare", ["power", "convert", "align", "foreign", "external_masks",
                 "illumination", "make_masks", "train_cellpose",
                 "cellpose_masks", "model_zoo"]),
    ("Run", ["mask", "timelapse", "motility", "measure", "annotate",
             "classify_merged", "classify", "ml_analyze",
             "map_barcodes", "regression",
             "queue", "batch", "distributed_jobs", "analyze_plaques",
             "recruitment", "invasion", "replication"]),
    ("Review", ["plate_view", "agreement", "umap", "activation",
                "barcode_qc", "layer_viewer", "graph_builder",
                "anndata_export", "run_compare",
                "train_compare", "classifier_evaluation", "model_compare",
                "run_history", "db_browser", "data_manager", "report",
                "pipeline_graph", "hit_list", "profiler",
                "methods_export", "image_scatter", "explain_cv",
                "investigate_hit"]),
], fallback="Review")

#: Five stages of a run. Variants 02 and 23 draw these as one seven-wide
#: tile grid per band, so a band of more than seven takes a second row.
#:
#: Five bands of seven was thirty-five slots for a registry of
#: thirty-four, and the note here said the next app added would force a
#: real decision rather than a silent overflow. Four arrived at once —
#: Illumination, Barcode QC, Layer Viewer, Graph Builder — and thirty-
#: eight apps do not go into thirty-five slots. The decision taken:
#:
#: * not a sixth band. Variants 13, 15 and 16 lay these out as exactly
#:   five columns and solve the gap between them from that count.
#: * not a wider grid. At eight columns the tile is 166 px, and at that
#:   width thirty-four of the thirty-eight names elide however small the
#:   font is set — measured, not assumed.
#: * so: three bands hold eight and wrap onto a second row in those two
#:   variants, which is recorded in v02's own comment and in the
#:   argument it prints.
#:
#: The cap was eight, which kept that to ONE wrapped row per band, then
#: nine, ten, eleven, and it is now TWELVE. Each rise is the same arithmetic:
#: the cap is the smallest number that can hold the registry over five
#: bands, so forty-two apps forced nine, forty-nine forced ten and
#: fifty-one force eleven, and fifty-six force twelve. Twelve is still one
#: wrapped row (seven, then five) rather than a third, which is what the rule
#: was ever about: the
#: ceiling for "one wrapped row" is fourteen, and each cap is simply the
#: smallest number that fitted the registry of the day. Both alternatives
#: are still refused for the reasons below — a sixth band breaks the
#: five-column variants, a wider grid elides the names.
#:
#: Curate and Lineage arrived after the cap had already moved to ten and
#: Report was already holding ten, so the fallback made Report twelve and
#: the test went red. Nothing moved out of Report and the cap did not rise
#: again: NEITHER app belonged in Report. Fixing a mask by hand is
#: producing a mask (Segment, which had eight) and a containment tree is a
#: measurement (Measure, which had nine). Forty-nine over five bands is
#: 10/9/10/10/10, which is the floor exactly — that is what a fallback
#: overflow usually means, that the band the key really belongs in still
#: had room.
#:
#: Experiment Design and the QC Dashboard then took it to fifty-one, and
#: fifty-one over five bands is ELEVEN however it is shared out — so this
#: time the cap really did have to move, and it moved by the arithmetic
#: rather than by preference. Both were filed where they belong first:
#: Experiment Design beside Power in Acquire (the plate layout decided
#: before an image exists), the QC Dashboard in Report (five verdicts, none
#: recomputed). 11/9/10/10/11. Eleven is still one wrapped row — seven,
#: then four — and a third row does not start until fifteen.
#: ``test_no_stage_band_exceeds_the_seven_column_grid_by_more_than_a_row``
#: is what makes the next app a decision rather than a silently squashed
#: page, and it asserts the floor too, so a cap left loose after apps are
#: removed fails as loudly as one left too tight.
#:
#: The merged Classify module then took the registry to fifty-four and the
#: test went red at "Report has 12" again, for the same reason as last
#: time: it was filed nowhere, so the fallback piled it into Report. The
#: cap did NOT move — eleven is still the floor, fifty-four over five
#: bands — but fifty-four leaves exactly one spare slot in 5 × 11, so the
#: shape is now four bands of eleven and one of ten, and Segment's nine
#: was no longer a shape the arithmetic allowed. Classify went beside its
#: two originals in Analyse and Activation came out of that band to
#: Segment, which is the one move the counts left available and is argued
#: on its own terms below. 11/10/11/11/11.
#:
#: Explain CV Model and Investigate Hit then took the registry to fifty-six.
#: Both are explicitly filed rather than left in Report's fallback: Explain
#: CV sits beside Activation because both interpret a trained vision model,
#: while Investigate Hit follows Regression in Analyse. The resulting
#: 11/11/11/12/11 distribution has the required arithmetic floor of twelve.
CATS_STAGE5 = _with_late_registrations([
    # Illumination is a correction of the sensor, applied to the pixels
    # before anything is segmented or measured — it belongs with the
    # other things done to images on the way in, not with the results.
    #
    # Power / Design comes before even that: it is what you run to decide
    # how many wells to image at all. There is no band earlier than
    # Acquire and a sixth is refused above, so it leads this one.
    #
    # Data Manager moves back here from Report. It landed there by analogy
    # with Database Browser ("a question about a finished run"), but the two
    # are not the same question: the browser reads the results, while the
    # manager answers "what is this project costing me on disk and what of
    # it can go", which is housekeeping beside Plate Queue, Batch Runner and
    # Distributed Jobs. CATS_NARROW8 already files it under "Import &
    # batch", so this makes the two tables agree rather than holding a third
    # opinion.
    #
    # Experiment Design is beside Power for the reason Power leads: it is
    # the other half of this section's own note -- plate layout, controls
    # and replicates, all decided before an image exists.
    ("Acquire", ["power", "experiment_design", "convert", "align", "foreign",
                 "external_masks", "illumination", "queue", "batch",
                 "distributed_jobs", "data_manager"]),
    # Layer Viewer is here because looking at a label mask over its image
    # is how a segmentation is judged; it is the eye on this band's work.
    # Curate is the hand on it: Layer Viewer is where you see that a cell
    # was split in two, and this is where you join it back up. Fixing a
    # mask is producing a mask, so it is this band and not a later one —
    # ``EXPECTED_SECTIONS`` files it under Core beside Mask and Timelapse
    # on exactly that argument, and this is the second table agreeing.
    # Activation moves here from Analyse, and the reason is half
    # arithmetic and half the band's own argument. The arithmetic: the
    # merged Classify module took the registry to fifty-four, and
    # fifty-four over five bands under a cap of eleven is 11/11/11/11/10
    # however it is shared out — so filing Classify beside the two
    # originals it dispatches to meant one key had to leave Analyse, and
    # this was the only band under the cap. The argument: Activation is
    # the one app in Analyse that never touches the measurement table or
    # the screen — it takes a trained model and a folder of crops and
    # paints what the model looked at, which is the job Layer Viewer does
    # for a mask, in the band the models themselves are already in (Train
    # Cellpose, Model Compare, and Model Zoo, which benches "Cellpose +
    # classifier" models).
    ("Segment", ["mask", "timelapse", "cellpose_masks", "make_masks",
                 "train_cellpose", "model_zoo", "model_compare",
                 "layer_viewer", "curate", "activation", "explain_cv"]),
    # Annotator Agreement moves here from Report, beside Annotate. It is
    # not a report on the screen, it is the check on the labelling step:
    # kappa between two annotation columns says whether the labels this
    # band produced can be trusted, in exactly the way Layer Viewer says
    # it for masks. Both other tables already put it with Annotate —
    # CATS_NARROW8 under "Label", CATS_QUESTIONS under "I have objects.
    # What are they like?" — so this is the third table agreeing.
    # Image Scatter is here for the same reason Layer Viewer is in
    # Segment: it is the eye on this band's work. Two measurements against
    # each other with the object under the cursor beside them is how you
    # find out whether what was measured is what you meant to measure —
    # asked of the measurements, not of the screen they add up to.
    # Lineage is here for the third turn of the same argument, and it is
    # the strongest of the three: the cell → nucleus → pathogen tree is
    # not derived from the measurements, it IS one of them. Measure is
    # what writes the ``cell_id`` links, so the band that produced the
    # relationship is the band that reads it. The fallback would have put
    # it under Report, which is only where an uncategorised key lands —
    # not an argument that a containment tree is a deliverable.
    # Tabulate is here on the band's own argument, one more turn: a pivot
    # of the measurement table with the n behind each cell is a question
    # asked OF the measurements, which is what this band is. It had been
    # landing in Report, and Report is only where an uncategorised key
    # falls -- not an argument that a contingency table is a deliverable.
    ("Measure", ["measure", "annotate", "agreement", "motility",
                 "image_scatter", "lineage", "analyze_plaques",
                 "recruitment", "invasion", "replication", "tabulate"]),
    # Barcode QC sits beside Map Barcodes and Regression because the
    # number it derives — the abundance threshold — is what the
    # regression consumes as fraction_threshold. It is part of analysing
    # the screen, not of reporting it. Graph Builder is here for the
    # same reason: asking the measurements a question you did not plan
    # for is analysis, whatever you do with the answer afterwards. AnnData
    # Export is the same argument once more — the .h5ad exists to be
    # analysed in scanpy, and the export is the first step of that
    # analysis rather than something you hand to a collaborator. The
    # Prediction Profiler is the same argument a fourth time: sweeping one
    # input of the fitted model to see where the prediction goes is asking
    # the model a question, which is analysis — the reporting apps below
    # judge whether to believe a result, and this one produces results.
    # PCA joins on the same grounds as Image UMAP, which is already here:
    # both reduce the measurement table to a couple of components to see
    # what separates, and neither is something you hand to anybody. It was
    # falling through to Report for want of a line here.
    # Classify — the merged module — is here beside the two originals it
    # dispatches to, which is where every other table already files the
    # three together: CATS_BROAD3 under "Run", CATS_NARROW8 under
    # "Classify", CATS_QUESTIONS under "I have a screen. Which genes
    # matter?". It had been landing in Report, which took that band to
    # twelve against a cap of eleven — and Report is only where an
    # uncategorised key falls, never an argument that training a
    # classifier is a deliverable. Activation went the other way, to
    # Segment; the note there says why.
    ("Analyse", ["classify_merged", "classify", "ml_analyze", "map_barcodes",
                 "barcode_qc", "regression", "umap", "graph_builder",
                 "anndata_export", "profiler", "pca", "investigate_hit"]),
    # Report is "decide whether to believe it, then hand it on", which is
    # where the two model/provenance QC apps belong: Classifier Evaluation
    # judges the classifier the Analyse stage trained, Run History says what
    # settings produced the numbers. Database Browser moves here from
    # Acquire for the same reason — exporting measurements.db is something
    # you do with results, not to get images in. Run Compare joins them on
    # the same grounds and beside Run History in particular: "what did I
    # change between these two runs, and did the numbers move" is the
    # question Run History answers for one run and this one answers for
    # two. Pipeline Graph is the same question about one run's files
    # rather than about two runs' settings, and it is the sharpest form of
    # "decide whether to believe it" there is: it marks which outputs no
    # longer follow from their inputs. Hit List and Methods & Results are
    # the "hand it on" half made literal — the ranked table a collaborator
    # receives, and the two paragraphs of the paper.
    # The QC Dashboard is the most literal member this band has: five
    # verdicts on one screen, none of them recomputed, which is "decide
    # whether to believe it" with nothing else in it.
    ("Report",  ["plate_view", "train_compare", "classifier_evaluation",
                 "run_history", "run_compare", "db_browser", "report",
                 "pipeline_graph", "hit_list", "methods_export",
                 "qc_dashboard"]),
], fallback="Report")

CATS_NARROW8 = _with_late_registrations([
    # Segment stays exactly three, and Measure and Label exactly two:
    # variant 04's whole argument is that a narrow category can be named
    # honestly ("'Segment' is three apps and it is obvious which three")
    # at the cost of two categories too small for a heading. Layer Viewer
    # would be a fourth here on a technicality — it is where you LOOK at
    # a mask, not one of the three things that make one.
    ("Segment",          ["mask", "timelapse", "cellpose_masks"]),
    ("Train models",     ["make_masks", "train_cellpose", "model_zoo",
                          "model_compare"]),
    ("Measure",          ["measure", "motility", "image_scatter"]),
    ("Label",            ["annotate", "agreement"]),
    # Classify (the merged module) is one of the apps that trains a
    # per-object classifier, so it is here with the two originals it
    # dispatches to rather than in the fallback band, which had been
    # swallowing it.
    ("Classify",         ["classify_merged", "classify", "ml_analyze",
                          "activation", "train_compare",
                          "classifier_evaluation", "explain_cv"]),
    # The Prediction Profiler goes here rather than under "Classify":
    # what it sweeps is a screen's regression, which is this band's
    # subject, and variant 04's argument is that "Classify" is exactly
    # the six apps that train and judge a per-object classifier.
    ("Screens & reports", ["map_barcodes", "barcode_qc", "regression",
                           "umap", "graph_builder", "layer_viewer",
                           "anndata_export", "plate_view", "report",
                           "hit_list", "methods_export", "pipeline_graph",
                           "profiler", "investigate_hit"]),
    # Power / Design and Run Compare are both "things you do around a run
    # rather than to the images": one decides how big the run has to be,
    # the other reads two of them against each other. This is variant 04's
    # widest, most administrative category and it is where they belong.
    ("Import & batch",   ["power", "convert", "align", "foreign",
                          "external_masks",
                          "illumination", "queue", "batch",
                          "distributed_jobs", "run_history", "run_compare",
                          "db_browser", "data_manager"]),
    ("Toxoplasma",       ["analyze_plaques", "recruitment", "invasion",
                          "replication"]),
], fallback="Screens & reports")

CATS_QUESTIONS = _with_late_registrations([
    # Power / Design answers the question BEFORE the first one here — "do
    # I have enough images?" — and the honest place for it is the band
    # about getting images, since that is the decision it feeds.
    ("I have images. Where are my objects?",
     ["mask", "timelapse", "cellpose_masks", "make_masks", "train_cellpose",
      "model_zoo", "model_compare", "align", "convert", "illumination",
      "foreign", "external_masks", "power"]),
    ("I have objects. What are they like?",
     ["measure", "annotate", "motility", "analyze_plaques", "recruitment",
      "invasion", "replication", "agreement", "layer_viewer",
      "image_scatter"]),
    # Hit List answers this band's question in the most direct way there
    # is — it IS the list of genes that matter — and the Prediction
    # Profiler is how you interrogate the model that produced it.
    ("I have a screen. Which genes matter?",
     ["classify_merged", "classify", "ml_analyze", "map_barcodes",
      "regression", "umap", "activation", "graph_builder", "anndata_export",
      "hit_list", "profiler", "investigate_hit"]),
    # Pipeline Graph belongs here for the literal reason: it marks the
    # outputs that no longer follow from their inputs, which is the
    # question in the heading. Methods & Results is the other half — what
    # you write down once you have decided you do believe it.
    ("Should I believe any of this?",
     ["plate_view", "barcode_qc", "train_compare", "classifier_evaluation",
      "report", "run_history", "run_compare", "db_browser", "data_manager",
      "queue", "batch", "distributed_jobs", "pipeline_graph",
      "methods_export", "explain_cv"]),
], fallback="Should I believe any of this?")

CATS_INTENT4 = [
    ("Segment images", CATS_QUESTIONS[0][1]),
    ("Measure objects", CATS_QUESTIONS[1][1]),
    ("Analyse a screen", CATS_QUESTIONS[2][1]),
    ("Check & share", CATS_QUESTIONS[3][1]),
]


def by_frequency() -> List[str]:
    """All keys, most-used first (see :data:`USE_COUNTS`)."""
    return sorted(all_keys(), key=lambda k: (-USE_COUNTS.get(k, 0), k))


def alphabetical() -> List[str]:
    """All keys sorted by display name."""
    return sorted(all_keys(), key=lambda k: name_of(k).lower())


def pinned_first() -> List[str]:
    """Pinned keys first, then everything else by frequency."""
    rest = [k for k in by_frequency() if k not in PINNED]
    return list(PINNED) + rest


def check_coverage(cats: Sequence[Tuple[str, Sequence[str]]]) -> None:
    """Raise if a categorisation drops, duplicates or invents a key."""
    seen: List[str] = []
    for _title, keys in cats:
        seen.extend(keys)
    known = set(all_keys())
    dupes = {k for k in seen if seen.count(k) > 1}
    if dupes:
        raise AssertionError(f"duplicate keys in categorisation: {sorted(dupes)}")
    unknown = set(seen) - known
    if unknown:
        raise AssertionError(f"unknown keys: {sorted(unknown)}")
    missing = known - set(seen)
    if missing:
        raise AssertionError(f"keys not categorised: {sorted(missing)}")
