"""Every settings window resizes, both ways.

Asked for as "i should be able to resize all settings windows horizontally
and verticaly", and the reason none of them did is not a flag anyone set.
There is one ``setFixedWidth`` in the whole application and it is not on a
settings window; no dialog sets a layout size constraint. THE FLOOR IS THE
CONTENT: a dialog's minimum size is its layout's total minimum, so the
window cannot be dragged in past the point where every field is fully
visible. Measured on ``PictureSettingsDialog`` -- size 645x318, minimum
645x318, a resize to 300x200 answered 645x318.

WHAT IS MEASURED HERE IS THE WINDOW, not a flag on it. Every assertion
below resizes a real dialog and reads its size back, counts a real
``QSizeGrip`` child, or reads a scroll bar's range -- because a test that
checks ``setSizeGripEnabled`` was called would have passed on the first
draft of this and the window would still have been stuck.

`QWidget.size(dialog)` RATHER THAN `dialog.size()`, throughout.
``UmapAppearanceDialog`` assigns ``self.size = QDoubleSpinBox(...)``, which
shadows the method on that one dialog; going through the class means the
sweep measures every dialog the same way instead of crashing on one.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


# --------------------------------------------------------------------------
# the application filter, on and off
# --------------------------------------------------------------------------

@pytest.fixture
def _no_glass():
    """Take the card and the rim off for the duration of one test.

    Glass widens every dialog's margins by ten pixels and makes it
    frameless, so a size measured with it installed is not the size the
    author laid out. It is installed by whichever test ran first, which is
    exactly the kind of dependence a measurement must not have -- so these
    tests decide for themselves, and the ones that care about the
    interaction install it on purpose.
    """
    from spacr.qt.widgets.glass import (install_glass_everywhere,
                                        uninstall_glass_everywhere)

    was_installed = uninstall_glass_everywhere()
    yield
    if was_installed:
        install_glass_everywhere()


@pytest.fixture
def bare(qtbot, _no_glass):
    """No application filter: dialogs exactly as their authors wrote them."""
    from PySide6.QtWidgets import QApplication

    from spacr.qt import dialogs

    app = QApplication.instance()
    saved = (dialogs._DETACHER, dialogs._DETACHED_APP)
    if saved[0] is not None:
        app.removeEventFilter(saved[0])
    dialogs._DETACHER = dialogs._DETACHED_APP = None
    yield app
    dialogs._DETACHER, dialogs._DETACHED_APP = saved
    if saved[0] is not None and saved[1] is app:
        app.installEventFilter(saved[0])


@pytest.fixture
def resizer(bare):
    """The application-wide filter, installed for one test."""
    from spacr.qt import dialogs

    app = bare
    dialogs.detach_all_dialogs(app)
    mine = dialogs._DETACHER
    yield app
    app.removeEventFilter(mine)
    dialogs._DETACHER = dialogs._DETACHED_APP = None


# --------------------------------------------------------------------------
# helpers -- every one of them reads a widget, not a setting
# --------------------------------------------------------------------------

def _open(qtbot, dialog):
    """Show ``dialog`` the way a user opens it, and let the panel settle."""
    from PySide6.QtWidgets import QApplication

    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitExposed(dialog)
    QApplication.processEvents()
    return dialog


def _size(widget):
    from PySide6.QtWidgets import QWidget

    return QWidget.size(widget)


def _resize(widget, width, height):
    from PySide6.QtWidgets import QApplication, QWidget

    QWidget.resize(widget, width, height)
    QApplication.processEvents()
    return QWidget.size(widget)


def _scroll_area(dialog):
    """The scroll area this module put in, or None."""
    from PySide6.QtWidgets import QWidget

    for child in QWidget.findChildren(dialog, QWidget):
        if type(child).__name__ == "_FormScroll":
            return child
    return None


def _grips(dialog):
    from PySide6.QtWidgets import QSizeGrip, QWidget

    return [g for g in QWidget.findChildren(dialog, QSizeGrip)
            if g.isVisible()]


def _picture_settings():
    from spacr.qt.widgets.picture_settings_dialog import PictureSettingsDialog

    return PictureSettingsDialog()


# --------------------------------------------------------------------------
# the defect, and the fix, on the dialog the item measured
# --------------------------------------------------------------------------

class TestTheFloorIsTheContent:

    def test_without_the_filter_it_will_not_shrink(self, bare, qtbot):
        """The defect, still reproducible with the filter taken off.

        This is what "cannot resize" was: dragging an edge inward does
        nothing, because the contents' minimum sizes add up to the whole
        window and Qt has no slack to give.
        """
        dialog = _open(qtbot, _picture_settings())
        natural = _size(dialog)

        assert _resize(dialog, 300, 200) == natural
        assert dialog.minimumSize() == natural

    def test_with_the_filter_it_goes_where_it_is_dragged(self, resizer,
                                                         qtbot):
        from PySide6.QtCore import QSize

        dialog = _open(qtbot, _picture_settings())
        natural = _size(dialog)

        assert _resize(dialog, 300, 200) == QSize(300, 200)
        assert natural.width() > 300 and natural.height() > 200

    def test_it_opens_at_the_size_it_always_did(self, resizer, qtbot,
                                               request):
        """Lowering the floor must not change what the user sees on open.

        The floor was load-bearing for the opening size: `adjustSize`
        bounds the size hint to two thirds of the screen and then clamps
        it back UP to the minimum, so taking the minimum away makes a wide
        dialog open narrow with a scroll bar already showing.
        """
        with_filter = _size(_open(qtbot, _picture_settings()))

        # the same dialog with the filter off, in the same session
        from PySide6.QtWidgets import QApplication

        from spacr.qt import dialogs
        app = QApplication.instance()
        app.removeEventFilter(dialogs._DETACHER)
        without = _size(_open(qtbot, _picture_settings()))
        app.installEventFilter(dialogs._DETACHER)

        assert with_filter == without

    def test_nothing_is_scrolled_away_at_that_size(self, resizer, qtbot):
        dialog = _open(qtbot, _picture_settings())
        scroll = _scroll_area(dialog)

        assert scroll is not None
        assert scroll.verticalScrollBar().maximum() == 0
        assert scroll.horizontalScrollBar().maximum() == 0

    def test_a_smaller_window_scrolls_rather_than_clipping(self, resizer,
                                                           qtbot):
        """The form keeps its own size; the window shows part of it."""
        dialog = _open(qtbot, _picture_settings())
        scroll = _scroll_area(dialog)
        form = scroll.widget()
        laid_out = form.size()

        _resize(dialog, 300, 200)

        assert scroll.verticalScrollBar().maximum() > 0
        assert scroll.horizontalScrollBar().maximum() > 0
        # not squashed into 300x200: the fields are still their own size,
        # which is what "scrolls rather than being clipped" means.
        assert form.width() >= laid_out.width() - 1
        assert form.height() >= laid_out.height() - 1

    def test_the_contents_are_wrapped_once_only(self, resizer, qtbot):
        """Shown, hidden and shown again is not a second scroll area."""
        from PySide6.QtWidgets import QWidget

        dialog = _open(qtbot, _picture_settings())
        dialog.hide()
        dialog.show()
        qtbot.waitExposed(dialog)

        wrappers = [c for c in QWidget.findChildren(dialog, QWidget)
                    if type(c).__name__ == "_FormScroll"]
        assert len(wrappers) == 1


class TestGrowingGivesTheRoomToTheForm:

    def test_the_fields_take_the_extra_width(self, resizer, qtbot):
        """A window that widens and leaves the form alone has not helped."""
        from PySide6.QtWidgets import QTabWidget, QWidget

        dialog = _open(qtbot, _picture_settings())
        natural = _size(dialog)
        pages = QWidget.findChildren(dialog, QTabWidget)[0]
        before = pages.width()

        _resize(dialog, natural.width() + 300, natural.height() + 200)

        assert pages.width() >= before + 290

    def test_the_holder_fills_the_viewport(self, resizer, qtbot):
        dialog = _open(qtbot, _picture_settings())
        scroll = _scroll_area(dialog)
        natural = _size(dialog)

        _resize(dialog, natural.width() + 300, natural.height() + 200)

        assert scroll.widget().width() == scroll.viewport().width()
        assert scroll.widget().height() == scroll.viewport().height()


class TestTheGrip:

    def test_a_settings_window_has_one_and_it_is_visible(self, resizer,
                                                         qtbot):
        dialog = _open(qtbot, _picture_settings())

        assert len(_grips(dialog)) == 1

    def test_it_sits_in_the_corner_the_hand_goes_to(self, resizer, qtbot):
        from PySide6.QtWidgets import QWidget

        dialog = _open(qtbot, _picture_settings())
        grip = _grips(dialog)[0]
        corner = QWidget.rect(dialog).bottomRight()

        assert abs(grip.geometry().right() - corner.x()) <= 2
        assert abs(grip.geometry().bottom() - corner.y()) <= 2

    def test_it_follows_the_corner_when_the_window_grows(self, resizer,
                                                         qtbot):
        from PySide6.QtWidgets import QWidget

        dialog = _open(qtbot, _picture_settings())
        grip = _grips(dialog)[0]
        natural = _size(dialog)

        _resize(dialog, natural.width() + 200, natural.height() + 150)
        corner = QWidget.rect(dialog).bottomRight()

        assert abs(grip.geometry().right() - corner.x()) <= 2
        assert abs(grip.geometry().bottom() - corner.y()) <= 2

    def test_a_message_is_left_entirely_alone(self, resizer, qtbot):
        """A sentence and two buttons gains nothing from either half of this.

        Built here rather than found, because the point is the RULE: no
        field, nothing scrollable, so nothing to scroll and no room to
        give.
        """
        from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel,
                                       QVBoxLayout)

        from spacr.qt import dialogs

        message = QDialog()
        column = QVBoxLayout(message)
        column.addWidget(QLabel("Delete these files?", message))
        column.addWidget(QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel, parent=message))
        _open(qtbot, message)

        assert dialogs.fields_in(message) == 0
        assert dialogs.more_than_a_message(message) is False
        assert _grips(message) == []
        assert _scroll_area(message) is None


class TestTheWindowStillMoves:
    """The wrap must not take the drag handle away.

    A glassed dialog is frameless, and `glass._DragByBackground` moves it
    from any press where `childAt` answers None -- the empty background
    between the controls. Filling the window with a scroll area makes that
    answer the scroll area everywhere, so the same rule is applied to the
    holder the fields now live on.
    """

    @staticmethod
    def _empty_spot(holder):
        """A point on the holder that is not on one of its children."""
        from PySide6.QtCore import QPoint

        for y in range(2, holder.height(), 7):
            for x in range(2, holder.width(), 7):
                point = QPoint(x, y)
                if holder.childAt(point) is None:
                    return point
        return None

    def test_dragging_the_empty_space_moves_the_window(self, resizer,
                                                       qtbot):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        dialog = _open(qtbot, _picture_settings())
        dialog.move(140, 110)
        qtbot.wait(10)
        holder = _scroll_area(dialog).widget()
        spot = self._empty_spot(holder)
        assert spot is not None, "no empty background left to drag from"
        start = dialog.pos()

        QTest.mousePress(holder, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, spot)
        QTest.mouseMove(holder, spot + QPoint(40, 25))
        QTest.mouseRelease(holder, Qt.MouseButton.LeftButton,
                           Qt.KeyboardModifier.NoModifier,
                           spot + QPoint(40, 25))
        qtbot.wait(10)

        assert dialog.pos() - start == QPoint(40, 25)

    @staticmethod
    def _occupied_spot(holder):
        """A point on the holder that IS on one of its children."""
        from PySide6.QtCore import QPoint

        for y in range(2, holder.height(), 7):
            for x in range(2, holder.width(), 7):
                point = QPoint(x, y)
                if holder.childAt(point) is not None:
                    return point
        return None

    def test_a_press_on_a_control_is_not_a_drag(self, resizer, qtbot):
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtTest import QTest

        dialog = _open(qtbot, _picture_settings())
        dialog.move(140, 110)
        qtbot.wait(10)
        holder = _scroll_area(dialog).widget()
        where = self._occupied_spot(holder)
        assert where is not None, "the form has nothing in it"
        start = dialog.pos()

        QTest.mousePress(holder, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier, where)
        QTest.mouseMove(holder, where + QPoint(40, 25))
        qtbot.wait(10)

        assert dialog.pos() == start


class TestShownAgain:

    def test_it_keeps_its_position(self, resizer, qtbot):
        """The detach filter's existing guarantee, still standing."""
        dialog = _open(qtbot, _picture_settings())
        dialog.move(180, 130)
        qtbot.wait(10)
        where = dialog.pos()

        dialog.hide()
        dialog.show()
        qtbot.waitExposed(dialog)

        assert dialog.pos() == where

    def test_it_keeps_the_size_the_user_left_it_at(self, resizer, qtbot):
        from PySide6.QtCore import QSize

        dialog = _open(qtbot, _picture_settings())
        _resize(dialog, 320, 240)

        dialog.hide()
        dialog.show()
        qtbot.waitExposed(dialog)

        assert _size(dialog) == QSize(320, 240)


class TestTheGlassSurvivesIt:
    """The card, the rim and the detached window type are unchanged."""

    @pytest.fixture
    def glassed(self, resizer):
        from spacr.qt.widgets.glass import (install_glass_everywhere,
                                            uninstall_glass_everywhere)

        install_glass_everywhere(resizer)
        yield resizer
        uninstall_glass_everywhere(resizer)

    def test_the_card_is_still_behind_the_form(self, glassed, qtbot):
        from spacr.qt.widgets.setup_card import SetupCard

        dialog = _open(qtbot, _picture_settings())
        cards = dialog.findChildren(SetupCard)

        assert len(cards) == 1

    def test_the_new_containers_paint_nothing(self, glassed, qtbot):
        """A scroll area added after glass tagged the tree is a black box.

        The palette's ``bg`` is #000000, so an untagged container between
        the card and the eye paints a black rectangle over it.
        """
        from spacr.qt.theme import TRANSPARENT_PROPERTY

        dialog = _open(qtbot, _picture_settings())
        scroll = _scroll_area(dialog)

        assert scroll.property(TRANSPARENT_PROPERTY)
        assert scroll.viewport().property(TRANSPARENT_PROPERTY)
        assert scroll.widget().property(TRANSPARENT_PROPERTY)

    def test_the_rim_still_has_its_band(self, glassed, qtbot):
        """Glass widens the dialog's margins so the rim is not painted over.

        Those pixels have to end up on the OUTER layout: left inside the
        scroll area they would scroll away with the form, and the band the
        rim runs along -- and that `glass._ResizeByEdge` grabs the window
        edge from -- would be gone.
        """
        from spacr.qt.widgets.glass import RIM_ROOM

        dialog = _open(qtbot, _picture_settings())
        left, top, right, bottom = dialog.layout().getContentsMargins()

        assert min(left, top, right, bottom) >= RIM_ROOM
        assert _scroll_area(dialog).geometry().left() >= RIM_ROOM

    def test_it_is_still_a_window_of_its_own(self, glassed, qtbot):
        from PySide6.QtCore import Qt

        dialog = _open(qtbot, _picture_settings())
        kind = dialog.windowFlags() & Qt.WindowType.WindowType_Mask

        assert kind == Qt.WindowType.Window


class TestTheRulesThemselves:
    """The predicates the filter asks, driven one at a time."""

    @staticmethod
    def _spacr_dialog():
        """A plain dialog with a form in it, which is all a settings window is.

        A bare ``QDialog`` on purpose: five of the real settings windows,
        Preferences among them, are a plain QDialog that a factory
        function fills. What the filter refuses is Qt's SPECIALISED
        dialogs, which is a question about type rather than about where a
        class was written -- see `TestQtsOwnDialogsAreLeftAlone`.
        """
        from PySide6.QtWidgets import QDialog

        return QDialog()

    @staticmethod
    def _fill(dialog, rows=8):
        from PySide6.QtWidgets import QLabel, QSpinBox, QFormLayout

        form = QFormLayout(dialog)
        for index in range(rows):
            box = QSpinBox(dialog)
            box.setMinimumWidth(160)
            form.addRow(QLabel(f"a setting with a long name {index}",
                               dialog), box)
        return dialog

    def test_a_form_built_after_the_polish_is_still_caught(self, resizer,
                                                           qtbot):
        """Show is the fallback, and it has to work on its own.

        A dialog that is polished before its form exists is invisible to
        the Polish half of the filter -- there is no layout to move. The
        contents are an ordinary layout change rather than a window
        recreation, so Show can pick it up.
        """
        from spacr.qt import dialogs

        dialog = self._spacr_dialog()
        qtbot.addWidget(dialog)
        dialog.ensurePolished()          # polished with nothing in it
        assert dialogs.wants_resizing(dialog) is False

        self._fill(dialog)
        _open(qtbot, dialog)

        assert dialog.property(dialogs.SCROLLS)
        assert _scroll_area(dialog) is not None
        natural = _size(dialog)
        smaller = _resize(dialog, 200, 150)
        assert smaller.width() < natural.width()
        assert smaller.height() < natural.height()

    def test_a_wizard_keeps_its_pages(self, resizer, qtbot):
        """A `QWizard` arranges its own children and is not ours to move."""
        from PySide6.QtWidgets import QWizard, QWizardPage

        from spacr.qt import dialogs

        wizard = QWizard()
        page = QWizardPage()
        page.setTitle("a page")
        wizard.addPage(page)
        _open(qtbot, wizard)

        assert dialogs.wants_resizing(wizard) is False
        assert _scroll_area(wizard) is None

    def test_a_dialog_can_say_no(self, resizer, qtbot):
        """``spacrNoScroll`` is the escape hatch ``spacrNoGlass`` is."""
        from spacr.qt import dialogs

        dialog = self._fill(self._spacr_dialog())
        dialog.setProperty(dialogs.NO_SCROLL, True)
        _open(qtbot, dialog)

        assert dialogs.wants_resizing(dialog) is False
        assert _scroll_area(dialog) is None
        assert _grips(dialog) == []

    def test_a_dialog_that_cannot_be_wrapped_is_still_a_dialog(
            self, resizer, qtbot, monkeypatch):
        """Decoration is never load-bearing, and half a transform is worse.

        The marker goes on before the work, so a dialog this failed on
        part-way is not attempted again on its next show.
        """
        from spacr.qt import dialogs

        def refuse(_dialog):
            raise RuntimeError("this layout will not move")

        monkeypatch.setattr(dialogs, "let_the_content_scroll", refuse)
        dialog = self._fill(self._spacr_dialog())

        _open(qtbot, dialog)

        assert dialog.property(dialogs.RESIZABLE)
        assert _scroll_area(dialog) is None
        assert dialog.isVisible()

    def test_an_explicit_minimum_is_taken_back(self, resizer, qtbot):
        """A minimum set by hand outranks the contents, and blocks all of this.

        ``BarcodeRegexDialog`` measured 760x430 with the filter off: the
        window would not go under its own `setMinimumSize`, whatever the
        contents did.
        """
        from spacr.qt.widgets.barcode_regex import BarcodeRegexDialog
        from PySide6.QtWidgets import QApplication, QWidget

        from spacr.qt import dialogs

        app = QApplication.instance()
        app.removeEventFilter(dialogs._DETACHER)
        without = _open(qtbot, BarcodeRegexDialog())
        floor = QWidget.minimumSize(without)
        app.installEventFilter(dialogs._DETACHER)

        dialog = _open(qtbot, BarcodeRegexDialog())

        assert floor.width() > 400 and floor.height() > 300
        assert QWidget.minimumSize(dialog).width() < floor.width()
        assert QWidget.minimumSize(dialog).height() < floor.height()
        assert _size(dialog) == _size(without)


class TestThePartsOnTheirOwn:
    """The three steps, each asked directly for the answer it gives alone."""

    def test_a_dialog_with_no_minimum_of_its_own_has_nothing_to_give_back(
            self, resizer, qtbot):
        from PySide6.QtWidgets import QDialog

        from spacr.qt import dialogs

        dialog = QDialog()
        qtbot.addWidget(dialog)

        assert dialogs.drop_the_explicit_floor(dialog) is False

    def test_a_dialog_that_kept_its_floor_is_not_resized_on_show(
            self, resizer, qtbot):
        """Only a window whose floor came off needs its size put back."""
        from PySide6.QtWidgets import QDialog

        from spacr.qt import dialogs

        dialog = QDialog()
        qtbot.addWidget(dialog)

        assert dialog.property(dialogs.OPENS_AT) is None
        assert dialogs.open_at_its_natural_size(dialog) is False

    def test_a_message_is_refused_by_name(self, resizer, qtbot):
        from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout

        from spacr.qt import dialogs

        message = QDialog()
        qtbot.addWidget(message)
        QVBoxLayout(message).addWidget(QLabel("nothing to edit here",
                                              message))

        assert dialogs.make_the_window_resizable(message) is False

    def test_an_empty_scroll_area_still_answers_for_its_size(self, resizer):
        """`_FormScroll` reads its widget's hint, and may not have one yet."""
        scroll = dialogs_scroll_class()()

        assert scroll.sizeHint().isValid()
        assert scroll.minimumSizeHint().width() == _smallest()

    def test_a_malformed_event_does_not_escape_the_drag_filter(
            self, resizer, qtbot):
        """This filter sees every event on the form; it may lose none.

        A `QEvent` of a mouse type carrying no button is not something Qt
        sends, which is the point: a filter that trusts the shape of what
        it is handed is one bad event away from taking the window with it.
        """
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QWidget

        from spacr.qt import dialogs

        holder = QWidget()
        qtbot.addWidget(holder)
        watcher = dialogs._drag_class()(holder)

        answer = watcher.eventFilter(
            holder, QEvent(QEvent.Type.MouseButtonPress))

        assert answer is False


def dialogs_scroll_class():
    from spacr.qt import dialogs

    return dialogs._form_scroll_class()


def _smallest():
    from spacr.qt import dialogs

    return dialogs.SMALLEST


class TestQtsOwnDialogsAreLeftAlone:

    def test_a_file_dialog_is_not_rebuilt(self, resizer, qtbot):
        """QFileDialog lays its own internals out and is not ours to move."""
        from PySide6.QtWidgets import QFileDialog

        from spacr.qt import dialogs

        chooser = QFileDialog()
        chooser.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        _open(qtbot, chooser)

        assert dialogs.wants_resizing(chooser) is False
        assert _scroll_area(chooser) is None
        # Not "it has no grip": QFileDialog turns its own on. The claim is
        # that nothing here touched it.
        assert not chooser.property(dialogs.RESIZABLE)
        assert not chooser.property(dialogs.SCROLLS)


# --------------------------------------------------------------------------
# the sweep: every settings window in the application
# --------------------------------------------------------------------------

def _every_settings_window():
    """One factory per dialog, named. Constructed inside the test."""
    import tempfile

    import pandas as pd

    def picture():
        return _picture_settings()

    def preferences():
        """A PLAIN QDialog that a factory fills -- the reason the rule is
        about type rather than about which module a class came from."""
        from spacr.qt.preferences import PreferencesDialog
        return PreferencesDialog()

    def settings_diff():
        from spacr.qt.settings_diff import SettingsDiffDialog
        return SettingsDiffDialog({"a": 1, "b": 2}, {"a": 1, "b": 3})

    def consent():
        from spacr.qt.install_consent import InstallerConsentDialog
        return InstallerConsentDialog()

    def refit():
        from spacr.qt.widgets.refit_dialog import RefitDialog
        return RefitDialog({})

    def setup():
        from spacr.qt.widgets.setup_dialog import SetupDialog
        return SetupDialog()

    def plate_map():
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker
        return PlateMapPicker()

    def barcode():
        from spacr.qt.widgets.barcode_regex import BarcodeRegexDialog
        return BarcodeRegexDialog()

    def umap_appearance():
        from spacr.qt.widgets.umap_search_viewer import UmapAppearanceDialog
        return UmapAppearanceDialog({})

    def umap_gallery():
        from spacr.qt.widgets.umap_search_viewer import UmapGalleryDialog
        return UmapGalleryDialog()

    def workbench():
        from spacr.qt.widgets.import_workbench import ImportWorkbenchDialog
        return ImportWorkbenchDialog(["a_A01_w1.tif", "a_A01_w2.tif"])

    def providers():
        from spacr.qt.widgets.ai_chat_panel import _ProvidersDialog
        return _ProvidersDialog()

    def aggregation():
        from spacr.qt.widgets.aggregation_rules import AggregationRulesDialog
        return AggregationRulesDialog(
            pd.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}))

    def columns():
        from spacr.qt.widgets.column_picker import ColumnPickerDialog
        return ColumnPickerDialog(db_path="")

    def wells():
        from spacr.qt.widgets.measurement_compare_dialog import _WellChoice
        return _WellChoice(["A01", "A02", "B03"])

    def features():
        from spacr.qt.widgets.feature_dictionary import FeatureDictionaryDialog
        return FeatureDictionaryDialog()

    def issue():
        from spacr.qt.ai.issue_preview import IssuePreviewDialog
        return IssuePreviewDialog({"title": "t", "body": "b"})

    def report():
        from spacr.qt.screens.annotate import _TextReportDialog
        return _TextReportDialog("t", "body")

    def annotate_settings():
        from spacr.qt.screens.annotate import AnnotateSettings, _SettingsDialog
        return _SettingsDialog(AnnotateSettings())

    def auto_annotate():
        from spacr.qt.screens.annotate import (AnnotateSettings,
                                               _AutoAnnotateDialog)
        return _AutoAnnotateDialog(AnnotateSettings())

    def regex():
        from spacr.qt.regex_editor import RegexEditorDialog
        return RegexEditorDialog(["p_A01_w1.tif"])

    def gates():
        from spacr.qt.widgets.gate_settings import (GateEditorSettings,
                                                    GateSettingsDialog)
        return GateSettingsDialog(GateEditorSettings())

    def clusters():
        from spacr.qt.widgets.gate_editor import _ClusterSettingsDialog
        return _ClusterSettingsDialog()

    def formula():
        from spacr.qt.widgets.formula_editor import FormulaDialog
        return FormulaDialog()

    def umap_display():
        from spacr.qt.widgets.umap_explorer import UmapDisplaySettings
        return UmapDisplaySettings({})

    def metadata_table():
        from spacr.qt.widgets.metadata_table import MetadataTableDialog
        return MetadataTableDialog([{"file": "a.tif", "plate": "p"}],
                                   tempfile.mkdtemp())

    def metadata_columns():
        from spacr.metadata_resolution import MetadataRequest
        from spacr.qt.widgets.metadata_mapper import MetadataColumnDialog
        return MetadataColumnDialog(MetadataRequest(
            missing=("plate", "well"), available=("a", "b"),
            examples={"a": ("1",), "b": ("2",)}, guesses={"plate": "a"}))

    def confirm_delete():
        from spacr.data_manager import PrunePlan
        from spacr.qt.screens.data_manager import ConfirmDeleteDialog
        return ConfirmDeleteDialog(PrunePlan(root="/tmp"))

    def execution_profile():
        from spacr.qt.screens.distributed_jobs import ExecutionProfileDialog
        return ExecutionProfileDialog()

    def advisor():
        from spacr.qt.widgets.settings_advisor_dialog import \
            SettingsAdvisorDialog
        from spacr.settings_advisor import Reading
        return SettingsAdvisorDialog(Reading(), {})

    def recipes():
        from spacr.qt.recipes import RecipeDialog
        return RecipeDialog(None)

    def _a_figure():
        """A matplotlib figure, built without touching a backend."""
        from matplotlib.figure import Figure

        figure = Figure()
        figure.add_subplot(111).plot([1, 2], [3, 4])
        return figure

    def save_figure():
        from spacr.qt.widgets.save_figure_dialog import SaveFigureDialog
        return SaveFigureDialog(_a_figure())

    def figure_settings():
        from spacr.qt.widgets.figure_settings import FigureSettingsDialog
        return FigureSettingsDialog(_a_figure())

    def queued_figure_settings():
        from spacr.qt.widgets.figure_queue import _FigureSettingsDialog
        return _FigureSettingsDialog(_a_figure())

    def axis_cutoff():
        from types import SimpleNamespace

        from spacr.qt.screens.gate_editor import _AxisCutoffDialog
        return _AxisCutoffDialog("t", "col",
                                 SimpleNamespace(low=None, high=None))

    return [
        ("PictureSettingsDialog", picture),
        ("PreferencesDialog", preferences),
        ("SettingsDiffDialog", settings_diff),
        ("InstallerConsentDialog", consent),
        ("RefitDialog", refit),
        ("SetupDialog", setup),
        ("PlateMapPicker", plate_map),
        ("BarcodeRegexDialog", barcode),
        ("UmapAppearanceDialog", umap_appearance),
        ("UmapGalleryDialog", umap_gallery),
        ("ImportWorkbenchDialog", workbench),
        ("_ProvidersDialog", providers),
        ("AggregationRulesDialog", aggregation),
        ("ColumnPickerDialog", columns),
        ("_WellChoice", wells),
        ("FeatureDictionaryDialog", features),
        ("IssuePreviewDialog", issue),
        ("_TextReportDialog", report),
        ("annotate._SettingsDialog", annotate_settings),
        ("_AutoAnnotateDialog", auto_annotate),
        ("RegexEditorDialog", regex),
        ("GateSettingsDialog", gates),
        ("_ClusterSettingsDialog", clusters),
        ("FormulaDialog", formula),
        ("UmapDisplaySettings", umap_display),
        ("MetadataTableDialog", metadata_table),
        ("MetadataColumnDialog", metadata_columns),
        ("ConfirmDeleteDialog", confirm_delete),
        ("ExecutionProfileDialog", execution_profile),
        ("SettingsAdvisorDialog", advisor),
        ("RecipeDialog", recipes),
        ("SaveFigureDialog", save_figure),
        ("FigureSettingsDialog", figure_settings),
        ("figure_queue._FigureSettingsDialog", queued_figure_settings),
        ("_AxisCutoffDialog", axis_cutoff),
    ]


SETTINGS_WINDOWS = _every_settings_window()


@pytest.mark.parametrize("name,build", SETTINGS_WINDOWS,
                         ids=[n for n, _ in SETTINGS_WINDOWS])
class TestEverySettingsWindow:
    """Thirty-five real dialogs, resized and read back one at a time.

    NOT A LOOP OVER `QDialog.__subclasses__`. A dialog that is never
    constructed proves nothing, and the ones that broke under this change
    broke in their constructors -- an explicit minimum size, a `resize()`
    of their own, a form already inside a scroll area.
    """

    def test_it_can_be_made_smaller_in_both_directions(self, resizer, qtbot,
                                                       name, build):
        dialog = _open(qtbot, build())
        natural = _size(dialog)

        smaller = _resize(dialog, natural.width() // 2,
                          natural.height() // 2)

        assert smaller.width() < natural.width(), name
        assert smaller.height() < natural.height(), name

    @staticmethod
    def _took_the_room(dialog, wider_by, taller_by, name):
        """Grow the window and say where the new pixels went.

        THE ROOM MUST NOT BECOME GREY, which is the only thing "it
        resized" has to mean to be worth anything. There are two honest
        answers and they are measured, not assumed: either a widget inside
        grew by what the window grew, or the form is still larger than the
        viewport in that direction -- in which case the room went to
        showing more of the form, which is what the user opened the corner
        for.
        """
        from PySide6.QtWidgets import QWidget

        scroll = _scroll_area(dialog)
        plumbing = set()
        if scroll is not None:
            plumbing = {id(scroll), id(scroll.viewport()), id(scroll.widget())}
        watched = [c for c in QWidget.findChildren(dialog, QWidget)
                   if c.isVisible() and c.width() > 40 and c.height() > 20
                   and id(c) not in plumbing]
        before = [(c, c.width(), c.height()) for c in watched]
        natural = _size(dialog)

        bigger = _resize(dialog, natural.width() + wider_by,
                         natural.height() + taller_by)
        assert bigger.width() == natural.width() + wider_by, name
        assert bigger.height() == natural.height() + taller_by, name

        form = scroll.widget() if scroll is not None else None
        view = scroll.viewport() if scroll is not None else None
        if wider_by:
            grew = [c for c, w, _h in before
                    if c.width() >= w + wider_by - 10]
            shows_more = form is not None and form.width() > view.width()
            assert grew or shows_more, f"{name}: the extra width went nowhere"
        if taller_by:
            grew = [c for c, _w, h in before
                    if c.height() >= h + taller_by - 10]
            shows_more = form is not None and form.height() > view.height()
            reaches_the_form = (form is not None
                                and form.height() == view.height())
            assert grew or shows_more or reaches_the_form, \
                f"{name}: the extra height went nowhere"

    def test_the_extra_width_goes_to_the_form(self, resizer, qtbot, name,
                                              build):
        self._took_the_room(_open(qtbot, build()), 200, 0, name)

    def test_the_extra_height_goes_to_the_form(self, resizer, qtbot, name,
                                               build):
        """The height, which a column of fields answers differently.

        A form with nothing stretchy in it -- nine of these are exactly
        that -- leaves the room below its last row, and did so before this
        item as well: measured on `ExecutionProfileDialog` without the
        filter, growing it by 150 pixels grew no widget in it at all. What
        this asserts is that the room REACHES the author's layout: the
        holder the form now sits on is exactly the viewport's height, so
        the layout is handed the same rectangle it used to be handed and
        distributes it the same way. `test_growing_it_shows_more_of_the_form`
        is where the height does visible work.
        """
        self._took_the_room(_open(qtbot, build()), 0, 150, name)

    def test_growing_it_shows_more_of_the_form(self, resizer, qtbot, name,
                                               build):
        """Room given back to a window that was made small.

        A column of fields has nothing that should stretch downwards --
        pulling a row of spin boxes to fill 150 extra pixels would be
        worse than leaving them alone -- so for those dialogs the room
        goes to the form by showing more of it. Measured from a window
        shrunk to half: growing it back puts the hidden rows on screen
        rather than adding a blank band under the same visible ones.

        A dialog holding a list, a table or a preview answers the same
        question the other way: that widget takes the height directly, so
        that is what is measured for those.
        """
        from PySide6.QtWidgets import QWidget

        dialog = _open(qtbot, build())
        natural = _size(dialog)
        scroll = _scroll_area(dialog)

        if scroll is None:
            watched = [c for c in QWidget.findChildren(dialog, QWidget)
                       if c.isVisible() and c.width() > 40
                       and c.height() > 20]
            before = [(c, c.height()) for c in watched]
            _resize(dialog, natural.width(), natural.height() + 150)
            assert [c for c, h in before if c.height() >= h + 140], \
                f"{name}: nothing inside took the extra height"
            return

        _resize(dialog, natural.width(), natural.height() // 2)
        hidden = scroll.verticalScrollBar().maximum()
        seen = scroll.viewport().height()

        _resize(dialog, natural.width(), natural.height())

        assert scroll.viewport().height() > seen, name
        if hidden:
            # Some of the form was scrolled away; growing brings it back.
            assert scroll.verticalScrollBar().maximum() < hidden, name
        else:
            # Nothing was hidden at half height, which means the form
            # itself gave way -- a dialog holding a list or a preview
            # shrinks it rather than scrolling. The room still went to the
            # form, which is what the viewport above says.
            assert scroll.widget().height() == scroll.viewport().height(), \
                name

    def test_it_has_a_size_grip(self, resizer, qtbot, name, build):
        dialog = _open(qtbot, build())

        assert len(_grips(dialog)) == 1, name

    def test_it_opens_at_the_size_it_always_did(self, resizer, qtbot, name,
                                               build):
        """The same dialog, opened with the filter off and then on."""
        from PySide6.QtWidgets import QApplication

        from spacr.qt import dialogs

        app = QApplication.instance()
        app.removeEventFilter(dialogs._DETACHER)
        without = _size(_open(qtbot, build()))
        app.installEventFilter(dialogs._DETACHER)

        assert _size(_open(qtbot, build())) == without, name

    def test_shown_again_it_is_where_it_was_left(self, resizer, qtbot,
                                                 name, build):
        dialog = _open(qtbot, build())
        dialog.move(150, 120)
        qtbot.wait(5)
        where = dialog.pos()

        dialog.hide()
        dialog.show()
        qtbot.waitExposed(dialog)

        assert dialog.pos() == where, name


class TestWhatWasLeftAlone:
    """Named here so the sweep's exclusions are a decision, not an omission."""

    def test_the_setup_wizard_lays_itself_out(self, resizer, qtbot):
        """`SetupSlides` puts nothing in its layout: the card holds the slides.

        There is no layout to move into a scroll area, and none is needed
        -- it is the one dialog in the application whose floor is already
        zero.
        """
        from PySide6.QtCore import QSize
        from PySide6.QtWidgets import QWidget

        from spacr.qt.widgets.setup_slides import SetupSlides
        from spacr.qt import dialogs

        slides = _open(qtbot, SetupSlides())

        assert slides.layout().count() == 0
        assert dialogs.wants_resizing(slides) is False
        assert QWidget.minimumSize(slides) == QSize(0, 0)
        natural = _size(slides)
        smaller = _resize(slides, natural.width() // 2,
                          natural.height() // 2)
        assert smaller.width() < natural.width()
        assert smaller.height() < natural.height()

    def test_a_dialog_with_slack_keeps_its_own_layout(self, resizer, qtbot):
        """`PlateMapPicker` holds its 384 wells in a scroll area already.

        Wrapping it would put a second set of scroll bars around the first,
        so it gets the grip and nothing else -- and it was already able to
        lose more than a third of each side.
        """
        from spacr.qt.widgets.plate_map_picker import PlateMapPicker

        picker = _open(qtbot, PlateMapPicker())

        assert _scroll_area(picker) is None
        assert len(_grips(picker)) == 1
        natural = _size(picker)
        smaller = _resize(picker, natural.width() // 2,
                          natural.height() // 2)
        assert smaller.width() <= natural.width() * 2 // 3
        assert smaller.height() <= natural.height() * 2 // 3
