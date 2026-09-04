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
KNOWN_OFFENDERS: dict = {}


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
