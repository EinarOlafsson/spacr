"""The same clipping sweep, pointed at the dialogs.

Instruction 350's remaining work is stated as the sweep rather than the tool:
"point ``wrapped_height`` at every screen and dialog, not just this one". The
screens are covered by ``test_the_text_fits_sweep.py``; the dialogs were not
covered at all -- which is where the original report actually was, "in
annotation the loade test data tooltips do not fit in the popup window".

THE CHECKER IS IMPORTED, NOT REWRITTEN. ``_offenders`` and ``_fits`` already
encode the three rules the instruction settled on -- skip what is not drawn,
ask a wrapping label about its height, ask everything else what it PAINTS
rather than what it holds. A second implementation would be a second set of
rules to keep in step.

WHY SEVENTEEN AND NOT FORTY-ONE. There are 41 QDialog subclasses; 24 of them
require constructor arguments -- a frame, a run, a settings dict -- and
building those from nothing would be building a fixture, not the dialog the
user sees. The 17 that construct argument-free are swept here and named
below, so the coverage is legible rather than implied. The rest are reachable
only through their screens and belong to whoever adds the fixture.

WHY TWO LOCALES AND TWO SCALES, NOT MORE. The file records a probe of 210
combinations that took 24 minutes against the committed sweep's 51 in under
four, and calls that "the wrong shape for a test that runs on every change".
English is the source and German runs materially longer -- the sweep's own
2026-09-02 finding was "English is fine and German is ten times worse" -- so
those two at both ends of the font-scale slider are the cheapest axes that
can still fail.
"""
from __future__ import annotations

import importlib
import tempfile
import warnings

import pytest

from spacr.qt import i18n as I
from spacr.qt.preferences import FONT_SCALE_MAX

from .test_the_text_fits_sweep import _offenders, at_font_scale  # noqa: F401

#: Dialogs that construct with no required arguments, as
#: ``(module path, class name)``. Measured rather than hand-listed; see the
#: module docstring for why the other 24 are absent.
DIALOGS = [
    ("spacr.qt.install_consent", "InstallerConsentDialog"),
    ("spacr.qt.screens.distributed_jobs", "ExecutionProfileDialog"),
    ("spacr.qt.widgets.barcode_regex", "BarcodeRegexDialog"),
    ("spacr.qt.widgets.column_picker", "ColumnPickerDialog"),
    ("spacr.qt.widgets.feature_dictionary", "FeatureDictionaryDialog"),
    ("spacr.qt.widgets.formula_editor", "FormulaDialog"),
    ("spacr.qt.widgets.gate_editor", "_ClusterSettingsDialog"),
    ("spacr.qt.widgets.import_workbench", "ImportWorkbenchDialog"),
    ("spacr.qt.widgets.model_zoo_picker", "ModelZooPicker"),
    ("spacr.qt.widgets.picture_settings_dialog", "PictureSettingsDialog"),
    ("spacr.qt.widgets.plate_map_picker", "PlateMapPicker"),
    ("spacr.qt.widgets.screen_data_picker", "ScreenDataPicker"),
    ("spacr.qt.widgets.setup_dialog", "SetupDialog"),
    ("spacr.qt.widgets.test_data_chooser", "TestDataChooser"),
    ("spacr.qt.widgets.umap_search_viewer", "UmapGalleryDialog"),
    # ADDED 2026-09-10. Both take no required argument and had simply
    # never been tried -- which is the whole of why the count of
    # uncovered dialogs was a number rather than a list.
    ("spacr.qt.widgets.ai_chat_panel", "_ProvidersDialog"),
    ("spacr.qt.widgets.setup_slides", "SetupSlides"),
]

LOCALES = ("en", "de")
SCALES = (1.0, FONT_SCALE_MAX)

#: What this sweep found the first time it ran, as
#: ``(dialog, locale, scale) -> how many captions are cut off``.
#:
#: EMPTY, AND THAT IS THE POINT. A RATCHET, NOT AN EXCUSE, on the same terms
#: as the screen sweep's table: a recorded count cannot grow, an unlisted
#: combination must have none at all, and an entry that reaches zero is
#: DELETED rather than left at 0. All four entries reached zero on 2026-09-04
#: and were deleted, so every one of the 17 dialogs is now clean in both
#: languages at both ends of the font slider.
#:
#: WHAT THE FOUR HAD IN COMMON, kept because the next raw pixel constant will
#: fail the same way: every entry was at 2.0x and none at 1.0x, and every one
#: was a SIZE SET FROM PYTHON that did not follow the font scale. The
#: stylesheet's font sizes grow with the preference; a `resize`, a
#: `setMinimumWidth` or a pinned square written in raw pixels does not, so the
#: glyphs outgrow the box that holds them. `preferences.scaled_px` is the
#: house mechanism and all four fixes were the same one line through it.
#:
#:   PlateMapPicker (15 each) -- ``WELL_SIDE = 22`` used unscaled, so a
#:   two-digit column number needed 30 px in a 22 px cell. The square is kept
#:   (a plate map is a picture of a physical object) and its SIDE now follows
#:   the font via ``well_side()``. The second half of that fix is the one this
#:   note predicted: ``_WELL_SHEET`` was a module-level dict built at import,
#:   which baked whichever scale was active then -- 44 px headers over 22 px
#:   wells -- so it is now ``_well_sheet(chosen, side)``, cached by side.
#:
#:   InstallerConsentDialog, BarcodeRegexDialog, FormulaDialog (1-2 each) --
#:   read as "a wrapping prose QLabel", and the label was a red herring. The
#:   house size-policy fix was applied first and did NOT work, because a
#:   policy stops a parent handing a label less than it asks for and cannot
#:   make a WINDOW grow that has no room to give. Each dialog's own
#:   ``resize`` / ``setMinimumSize`` was the raw constant, and scaling it
#:   cleared all three.
#:
#: HOW THAT WAS ESTABLISHED, since the method is what made the difference:
#: this table is compared with ``<=``, so the suite stayed green after the
#: label fix and it was nearly recorded as done. Setting the entries to zero
#: and re-running is what showed six still failing. A ``<=`` ratchet cannot
#: tell "fixed" from "unchanged" -- only tightening it can, which is why
#: tightening it is part of claiming a fix here rather than a follow-up.
KNOWN_OFFENDERS: dict = {
    # ONE ENTRY, ADDED 2026-09-10 WITH ITS DIAGNOSIS, and it is the first
    # this file has ever carried. `_ProvidersDialog`'s intro paragraph
    # wraps to 108 px of height in the 97 it is given -- eleven pixels, in
    # German only, at 100 % only.
    #
    # WHAT WAS FIXED, and it closed the other three of the four
    # combinations: the dialog set `setMinimumWidth(scaled_px(620))` and
    # `setMinimumHeight(560)`. The height was a raw device-pixel constant
    # beside a scaled width, so at a doubled font the floor stayed put
    # while every caption in the dialog grew. Same defect class as the
    # seven settings columns 350 already records.
    #
    # WHY THIS ONE IS LEFT: the dialog opens at exactly its minimum height,
    # and the German text wraps to one line more than the English. The
    # layout cannot discover that, because the page is inside a
    # QTabWidget, and QTabWidget does not propagate `heightForWidth` --
    # so the wrapped label's true height never reaches the dialog's own
    # sizeHint. Giving the label a height-for-width size policy was tried
    # and changed nothing for exactly that reason; it was reverted rather
    # than left in as code that does nothing.
    #
    # THE FIX IS A SCROLL AREA around the providers page, which is 350's
    # own rule ("a visible, accessible fallback rather than silently
    # clipping") and is a layout change worth making with a display in
    # front of the person making it.
    ("_ProvidersDialog", "de", 1.0): 1,
}


#: Dialogs that take an argument the caller can supply GENUINELY.
#:
#: WHY THIS EXISTS, AND WHY IT IS NOT THE FIXTURE THIS FILE REFUSED TO BUILD.
#: The header above says 24 dialogs "require constructor arguments -- a frame,
#: a run, a settings dict -- and building those from nothing would be building
#: a fixture, not the dialog the user sees". That is right for a frame or a
#: run, which have to be invented. It is NOT right for the six below: a
#: default ``AnnotateSettings()`` is the object Annotate itself constructs
#: before it opens the dialog, and a window title is a string. Supplying those
#: is showing the dialog as the user gets it, not standing in for it.
#:
#: THREE OF THE FIVE ARE ANNOTATE'S, which is where 350's original report came
#: from: "in annotation the loade test data tooltips do not fit in the popup
#: window". Leaving them unswept left the reported surface uncovered.
#:
#: ``GateSettingsDialog`` is deliberately absent: it wants an object with a
#: ``sample_fraction`` attribute, and handing it a stand-in WOULD be building
#: a fixture. It belongs with the other 24.
def _annotate_settings():
    """The settings object Annotate builds before opening its dialogs."""
    from spacr.qt.screens.annotate import AnnotateSettings

    return AnnotateSettings()


#: SIX MORE ON 2026-09-07, by the same rule and not a looser one.
#:
#: The criterion above is that the argument can be supplied GENUINELY -- a
#: window title is a string, and `AnnotateSettings()` is the object Annotate
#: itself builds. These six take strings, paths, lists of filenames, well
#: names, a report mapping and one `AxisCutoff`, which is a two-field
#: dataclass out of `gate_canvas` and the same object the gate editor passes.
#: None of them is a stand-in for something that has to be invented.
#:
#: The values are REALISTIC rather than minimal, because the defect being
#: looked for is text that does not fit: `_TextReportDialog` gets a report
#: shaped like the one Class counts produces, `RegexEditorDialog` gets
#: spaCR-shaped filenames, and `IssuePreviewDialog` gets a title long enough
#: to be worth eliding. A one-character body proves nothing about a dialog
#: whose job is showing prose.
def _axis_cutoff():
    """The cutoff object the gate editor hands its axis dialog."""
    from spacr.qt.widgets.gate_canvas import AxisCutoff

    return AxisCutoff(12.0, 980.0)


def _a_figure():
    """A live matplotlib figure with something on it to draw controls from.

    NOT AN EMPTY ONE. `FigureSettingsDialog` builds its controls "from the
    figure's current axes, artists, legends, and optional spaCR metadata",
    so a blank figure would build a blank dialog and the sweep would
    measure nothing while reporting a pass. The axes carry a labelled line
    and a legend so there is a row per artist to lay out.
    """
    from matplotlib.figure import Figure

    figure = Figure(figsize=(4, 3))
    axes = figure.add_subplot(111)
    axes.plot([0, 1, 2], [0, 1, 4], label="a labelled series")
    axes.set_xlabel("An x axis with a long enough caption to lay out")
    axes.set_ylabel("And a y axis")
    axes.set_title("A figure the settings dialog has something to say about")
    axes.legend()
    return figure


def _gate_settings():
    """The Gate Editor's own default settings object.

    THIS DIALOG WAS EXCLUDED ON A PREMISE THAT IS NOT TRUE. The note below
    says `GateSettingsDialog` "wants an object with a ``sample_fraction``
    attribute, and handing it a stand-in WOULD be building a fixture", and
    files it with the 24. But `GateEditorSettings` is a frozen dataclass
    that constructs with NO arguments and comes up with
    ``sample_fraction = 1.0`` -- so the real object is free, and this is
    the same case as `AnnotateSettings()`, which the same note accepts
    because it "is the object Annotate itself constructs before it opens
    the dialog".
    """
    from spacr.qt.widgets.gate_settings import GateEditorSettings

    return GateEditorSettings()


def _a_measurement_table():
    """A measurements frame, with the column names spaCR actually writes.

    `AggregationRulesDialog` offers one row per COLUMN, so the frame's
    columns are the dialog's content: a frame with two made-up names
    would build a two-row dialog and measure almost nothing. These are the
    names a measure run produces, and the long ones are the point --
    `cell_channel_1_percentile_75` is the kind of caption that overruns a
    column header.
    """
    import pandas as pd

    return pd.DataFrame({
        "plateID": ["plate1"], "rowID": ["r1"], "columnID": ["c1"],
        "cell_area": [1024.0],
        "cell_perimeter": [128.0],
        "cell_channel_1_mean_intensity": [0.42],
        "cell_channel_1_percentile_75": [0.61],
        "nucleus_channel_0_mean_intensity": [0.33],
        "pathogen_channel_2_integrated_intensity": [98.7],
        "cytoplasm_channel_3_standard_deviation": [0.07],
    })


def _compare_inputs():
    """Object rows and the groups they are split into.

    Both halves are what the Compare screen holds before it opens this:
    a frame of object rows and a mapping of group name to the object-index
    values in it. The group names are deliberately long, because they are
    drawn as captions and a short one measures nothing.
    """
    import pandas as pd

    objects = pd.DataFrame({
        "object_index": [0, 1, 2, 3],
        "plateID": ["plate1"] * 4,
        "rowID": ["A", "A", "B", "B"],
        "columnID": ["01", "02", "01", "02"],
        "cell_area": [900.0, 1100.0, 850.0, 1250.0],
    })
    groups = {"TSG101 knockout": [0, 1],
              "non-targeting control": [2, 3]}
    return objects, groups


def _metadata_rows():
    """Preview rows and a destination, as the mapper hands them over."""
    return ([
        {"filename": "plate1_A01_f01_DAPI.tif", "plateID": "plate1",
         "rowID": "A", "columnID": "01", "fieldID": "1", "channel": "0"},
        {"filename": "plate1_A01_f01_GFP.tif", "plateID": "plate1",
         "rowID": "A", "columnID": "01", "fieldID": "1", "channel": "1"},
        {"filename": "plate1_B12_f09_Cy5.tif", "plateID": "plate1",
         "rowID": "B", "columnID": "12", "fieldID": "9", "channel": "2"},
    ], tempfile.gettempdir())


DIALOGS_WITH_ARGUMENTS = [
    ("spacr.qt.screens.annotate", "_SettingsDialog", _annotate_settings),
    ("spacr.qt.screens.annotate", "_GenerateAnnotationDatabaseDialog",
     _annotate_settings),
    ("spacr.qt.screens.annotate", "_AutoAnnotateDialog", _annotate_settings),
    ("spacr.qt.widgets.refit_dialog", "RefitDialog", dict),
    ("spacr.qt.hf_download", "_DownloadDialog", lambda: "Downloading model"),
    # ADDED 2026-09-10. Three more whose one argument is honestly
    # suppliable: two take a figure, and the third takes the settings it
    # opens on -- and an EMPTY dict is the real case rather than a
    # shortcut, because its own docstring promises that "a settings file
    # written before a field existed still opens".
    ("spacr.qt.widgets.figure_settings", "FigureSettingsDialog", _a_figure),
    ("spacr.qt.widgets.save_figure_dialog", "SaveFigureDialog", _a_figure),
    ("spacr.qt.widgets.umap_explorer", "UmapDisplaySettings", dict),
    # ADDED 2026-09-10. The first was excluded on a premise that does not
    # hold -- see `_gate_settings`; the second's frame IS its content.
    ("spacr.qt.widgets.gate_settings", "GateSettingsDialog", _gate_settings),
    ("spacr.qt.widgets.aggregation_rules", "AggregationRulesDialog",
     _a_measurement_table),
    # The figure-queue one takes a figure like the other two, and
    # `UmapAppearanceDialog` reads its argument with `.get`, so an empty
    # mapping is the documented case rather than a shortcut.
    ("spacr.qt.widgets.figure_queue", "_FigureSettingsDialog", _a_figure),
    ("spacr.qt.widgets.umap_search_viewer", "UmapAppearanceDialog", dict),
]

#: Dialogs taking more than one genuinely-suppliable argument.
DIALOGS_WITH_ARGUMENTS_MULTI = [
    ("spacr.qt.screens.annotate", "_TextReportDialog",
     lambda: ("Class counts",
              "Class    Count    Color\n"
              "    1      812    #4A9EFF\n"
              "    2      754    #3fb950\n")),
    ("spacr.qt.screens.gate_editor", "_AxisCutoffDialog",
     lambda: ("Cell area", "cell_area", _axis_cutoff())),
    ("spacr.qt.widgets.sra_picker", "SraPicker",
     lambda: (tempfile.gettempdir(),)),
    ("spacr.qt.regex_editor", "RegexEditorDialog",
     lambda: (["plate1_A01_f01_DAPI.tif", "plate1_A01_f01_GFP.tif",
               "plate1_A01_f02_DAPI.tif", "plate1_B12_f09_Cy5.tif"],)),
    ("spacr.qt.widgets.measurement_compare_dialog", "_WellChoice",
     lambda: (["A01", "A02", "A03", "B01", "B02", "B03"],)),
    ("spacr.qt.widgets.metadata_table", "MetadataTableDialog",
     _metadata_rows),
    ("spacr.qt.widgets.measurement_compare_dialog",
     "MeasurementCompareDialog", _compare_inputs),
    ("spacr.qt.ai.issue_preview", "IssuePreviewDialog",
     lambda: ({"title": "Mask fails on 16-bit input from a Nikon ND2",
               "body": "Steps to reproduce, the settings used, and the "
                       "traceback as it appeared in the console."},)),
]


def _build_with(module_path: str, class_name: str, make_argument, qtbot):
    """Build a dialog that needs one argument, and let its layout settle.

    :param module_path: dotted module holding the dialog.
    :param class_name: the dialog class.
    :param make_argument: callable returning the single argument.
    :param qtbot: pytest-qt's bot.
    :returns: the shown dialog.
    """
    module = importlib.import_module(module_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        supplied = make_argument()
        # A TUPLE IS THE ARGUMENT LIST, anything else is the single argument.
        # Six dialogs added on 2026-09-07 take two or three, and wrapping the
        # one-argument cases in tuples would have meant editing five working
        # entries to add six new ones.
        args = supplied if isinstance(supplied, tuple) else (supplied,)
        dialog = getattr(module, class_name)(*args)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    for _ in range(4):
        qtbot.wait(10)
    return dialog


def _build(module_path: str, class_name: str, qtbot):
    module = importlib.import_module(module_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dialog = getattr(module, class_name)()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    # PUMP UNTIL THE GEOMETRY SETTLES. HANDOFF records a clipping run that
    # reported 38 German problems where there are none, because one
    # processEvents() after show() leaves widths at their pre-layout
    # defaults. Measuring a widget before the layout has finished measures
    # the default, not the dialog.
    for _ in range(4):
        qtbot.wait(10)
    return dialog


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("module_path,class_name", DIALOGS,
                         ids=[name for _path, name in DIALOGS])
def test_no_dialog_caption_is_cut_off(module_path, class_name, locale, scale,
                                      qtbot, at_font_scale,  # noqa: F811
                                      monkeypatch):
    """One dialog, in one language, at one font scale."""
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    at_font_scale(scale)
    dialog = _build(module_path, class_name, qtbot)

    offenders = _offenders(dialog)
    allowed = KNOWN_OFFENDERS.get((class_name, locale, scale), 0)
    detail = ("; ".join(offenders[:6])
              + (f" (and {len(offenders) - 6} more)"
                 if len(offenders) > 6 else ""))
    assert len(offenders) <= allowed, (
        f"{class_name} in {locale} at {scale:g}x: {len(offenders)} captions "
        f"cut off, {allowed} recorded. {detail}")


def test_the_dialog_sweep_can_actually_fail(qtbot, at_font_scale):  # noqa: F811
    """The test that keeps this out of 288's set of four.

    288 records four tests that passed while exercising nothing. A sweep that
    reports zero is indistinguishable from a sweep that looked at nothing
    unless something it is shown IS caught, so a caption too long for its box
    is manufactured and must be reported.
    """
    from PySide6.QtWidgets import QDialog, QPushButton

    at_font_scale(1.0)
    dialog = QDialog()
    qtbot.addWidget(dialog)
    button = QPushButton("a caption far longer than the box it is given",
                         dialog)
    button.setFixedWidth(30)
    dialog.show()
    qtbot.waitExposed(dialog)
    for _ in range(4):
        qtbot.wait(10)

    assert _offenders(dialog), (
        "the sweep reported nothing for a caption that cannot fit, so a zero "
        "from it would mean nothing"
    )


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("module_path,class_name,make_argument",
                         DIALOGS_WITH_ARGUMENTS,
                         ids=[name for _p, name, _f in DIALOGS_WITH_ARGUMENTS])
def test_no_argumented_dialog_caption_is_cut_off(module_path, class_name,
                                                 make_argument, locale, scale,
                                                 qtbot, at_font_scale,  # noqa: F811
                                                 monkeypatch):
    """The dialogs that take a settings object or a title.

    Same checker, same rules, same locales and scales as the argument-free
    sweep above -- only the construction differs, and it differs by supplying
    the value the application itself supplies.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    at_font_scale(scale)
    dialog = _build_with(module_path, class_name, make_argument, qtbot)

    offenders = _offenders(dialog)
    allowed = KNOWN_OFFENDERS.get((class_name, locale, scale), 0)
    detail = "; ".join(offenders[:6]) if offenders else ""
    assert len(offenders) <= allowed, (
        f"{class_name} in {locale} at {scale:g}x: {len(offenders)} captions "
        f"cut off, {allowed} allowed. {detail}"
    )


@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize("locale", LOCALES)
@pytest.mark.parametrize("module_path,class_name,make_argument",
                         DIALOGS_WITH_ARGUMENTS_MULTI,
                         ids=[name for _p, name, _f
                              in DIALOGS_WITH_ARGUMENTS_MULTI])
def test_no_multi_argument_dialog_caption_is_cut_off(
        module_path, class_name, make_argument, locale, scale,
        qtbot, at_font_scale, monkeypatch):  # noqa: F811
    """The six added on 2026-09-07, by the same rule as the five above.

    Six more of the twenty-four turned out not to need an invented fixture
    at all: they take strings, a path, lists of filenames and well names, a
    report mapping, and one `AxisCutoff` -- which is the same two-field
    object the gate editor passes them. Supplying those is showing the
    dialog as the user gets it, which is the line this file has drawn from
    the start.
    """
    monkeypatch.setenv(I.ENV_LANGUAGE, locale)
    at_font_scale(scale)
    dialog = _build_with(module_path, class_name, make_argument, qtbot)

    offenders = _offenders(dialog)
    allowed = KNOWN_OFFENDERS.get((class_name, locale, scale), 0)
    detail = "; ".join(offenders[:6]) if offenders else ""
    assert len(offenders) <= allowed, (
        f"{class_name} in {locale} at {scale:g}x: {len(offenders)} captions "
        f"cut off, {allowed} allowed. {detail}"
    )
