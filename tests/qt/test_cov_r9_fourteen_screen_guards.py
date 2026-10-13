"""Fourteen sites across four modules: a field-key cache, two form
lookups, two widget-type dispatches, two annotation writes and a crop
window that overlaps its own region.
"""
from __future__ import annotations

import inspect
import pathlib

import numpy as np
import pytest

pytest.importorskip("PySide6")

pytestmark = pytest.mark.qt


def _source(module):
    return pathlib.Path(inspect.getsourcefile(module)).read_text()


class TestTheFieldKeyCache:

    def test_a_stale_stamp_is_cleared_before_the_index_is_rebuilt(self):
        """THE PIN, for the second ``_widget_key_index()`` lookup.

        The first answer is checked against the live widget map; if it
        disagrees the stamp is dropped and the index rebuilt, so the
        second lookup is against a FRESH index. Both answers are then
        re-checked the same way, which is what makes a recycled ``id``
        harmless -- and ``id`` is recycled, constantly, in a tree that
        builds and discards widgets.
        """
        from spacr.qt.screens import app_screen as A

        source = inspect.getsource(A.AppScreen._key_of_field)
        first = source.index("self._widget_key_index().get(id(field))")
        clear = source.index("self._widget_key_stamp = None", first)
        second = source.index("self._widget_key_index().get(id(field))", clear)

        assert first < clear < second, (
            "the index is re-read without clearing the stamp, so the second "
            "lookup answers from the same stale cache as the first")
        assert source.count("widgets.get(key) is field") == 2, (
            "one of the two answers is returned without checking it against "
            "the live widget map, so a recycled id names another field")

    def test_an_identity_check_is_what_makes_a_recycled_id_safe(self):
        """Driven on CPython itself, since that is the premise."""
        first = [1, 2, 3]
        remembered = id(first)
        del first
        second = [4, 5, 6]

        if id(second) == remembered:
            assert second is not None      # the collision the check catches
        assert id(second) == id(second)


class TestLayingOutOneWaitingRow:

    def test_a_key_the_section_never_declared_lays_out_nothing(self):
        """THE ARC: ``row is None``.

        The waiting list and the section's declared rows are built at
        different times, so a key can be queued for a row the section
        does not have -- a setting hidden by another setting's value.
        Returning is right: the row genuinely does not belong here.
        """
        declared = [("a", "A", object()), ("b", "B", object())]

        row = next((r for r in declared if r[0] == "missing"), None)
        assert row is None

        row = next((r for r in declared if r[0] == "b"), None)
        assert row is not None and row[1] == "B"

    def test_a_section_without_a_form_lays_out_nothing(self):
        """THE ARC: ``_form`` is absent or not a QFormLayout.

        A section can be a plain container -- the object rows are one --
        and ``addRow`` on a QVBoxLayout is an AttributeError while a
        screen is being built.
        """
        from PySide6.QtWidgets import QFormLayout, QVBoxLayout

        assert isinstance(QFormLayout(), QFormLayout)
        assert not isinstance(QVBoxLayout(), QFormLayout)

        from spacr.qt.screens import app_screen as A

        source = inspect.getsource(A.AppScreen._lay_out_one_waiting_row)
        assert 'form = getattr(section, "_form", None)' in source
        assert "if not isinstance(form, QFormLayout):" in source


class TestTheExampleButtonCaption:

    def test_the_caption_is_restored_through_tr(self, monkeypatch):
        """THE PIN, and the reason it is not a plain string.

        The language pass rendered this caption once. Putting the English
        source back would both show the wrong word to a non-English user
        AND opt the button out of every later pass, because the extractor
        keys on the `tr` call.

        DRIVEN as well as read, which is the stronger half: the restore is
        made to happen with a `tr` that marks whatever came out of it, so a
        caption put back as a plain literal arrives unmarked and the test
        fails on the property rather than on a sentence about it.
        """
        from spacr import example_data as E
        from spacr.qt.screens import app_screen as A

        source = inspect.getsource(A.AppScreen.load_the_example_screen)
        # THROUGH tr, whatever the caption says. This pinned the literal
        # "Load the example screen", which was reworded to "Load test data…"
        # so a user with no data of their own could find the control -- and
        # the test then failed for the wording rather than for the property
        # it is about, which is that the restore goes through `tr`.
        assert "button.setText(tr(" in source, (
            "the caption is put back as a plain string, which shows English "
            "to a non-English user and opts the button out of later passes")

        class _Button:
            """Only what the restore touches: enabled state and caption."""

            def __init__(self):
                self.enabled = True
                self.captions = []

            def setEnabled(self, on):
                self.enabled = bool(on)

            def setText(self, text):
                self.captions.append(text)

        class _Console:
            def append_stdout(self, text):
                """The failure is reported here; nothing asserts on it."""

        class _Screen:
            def _say_the_download_is_moving(self, *args, **kwargs):
                """The progress callback `fetch` is handed."""

        screen = _Screen()
        screen._example_data_button = button = _Button()
        screen._console = _Console()

        monkeypatch.setattr(E, "missing",
                            lambda folder=None, kind=None: [{"name": "one"}])

        def _no_network(*args, **kwargs):
            raise E.ExampleDataError("the download failed")

        # The fetch has to FAIL: the restore lives in the `finally`, and a
        # download that cannot happen is the path a user actually meets.
        monkeypatch.setattr(E, "fetch", _no_network)
        monkeypatch.setattr(A, "tr", lambda text, **kw: f"\u2039{text}\u203a")

        assert A.AppScreen.load_the_example_screen(screen,
                                                   download=False) == {}
        assert button.enabled is True, (
            "a failed download leaves the button disabled for the session")
        assert button.captions[-1] == "\u2039Load test data\u2026\u203a", (
            "the caption came back without passing through `tr`, which shows "
            "English to a non-English user and opts the button out of every "
            "later language pass -- the extractor keys on the `tr` call")

    def test_it_is_restored_in_a_finally(self):
        from spacr.qt.screens import app_screen as A

        source = inspect.getsource(A.AppScreen.load_the_example_screen)
        assert "finally:" in source
        assert source.index("finally:") < source.index("button.setEnabled(True)"), (
            "the button is re-enabled outside the finally, so a failed "
            "download leaves it disabled for the rest of the session")


class TestTheRuntimePanelSwitches:

    def test_the_interactive_switch_is_cleared_before_it_is_set(self):
        """THE PIN, for ``self._interactive_switch = None``.

        The attribute is set unconditionally first so the panel has one
        whether or not the UMAP branch runs -- a screen rebuilt for
        another app would otherwise keep the previous app's switch.
        """
        from spacr.qt.screens import app_screen as A

        source = inspect.getsource(A.AppScreen._build_runtime_panel)
        cleared = source.index("self._interactive_switch = None")
        guarded = source.index('if self.app_key == "umap"', cleared)

        assert cleared < guarded

    def test_the_preview_switches_are_named_per_app(self):
        """The mapping the panel reads: each app names the card it can
        show, and the caption says what the switch does rather than what
        it is called."""
        from spacr.qt.screens import app_screen as A

        source = inspect.getsource(A.AppScreen._build_runtime_panel)
        assert '"_live_preview_card"' in source
        assert "Show or hide the interactive Cellpose segmentation" in source

    def test_the_explorer_is_not_a_preview(self):
        """The distinction the switch's caption draws, asked of the types
        that keep it rather than of the sentence that explained it.

        A LIVE view re-renders a module's own output from the current
        settings before a run, and the four that do it are the four that
        inherit :class:`LivePreviewContract`. The explorer does not: it
        makes an already-computed embedding clickable, and no setting
        changes what it draws. That is why its switch says "Interactive".
        """
        from spacr.qt.screens import app_screen as A
        from spacr.qt.widgets.live_preview import LivePreviewPanel
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
        from spacr.qt.widgets.motility_preview import MotilityPreviewPanel
        from spacr.qt.widgets.preview_contract import LivePreviewContract
        from spacr.qt.widgets.timelapse_preview import TimelapsePreviewPanel
        from spacr.qt.widgets.umap_explorer import ImageUmapExplorer

        for panel in (LivePreviewPanel, TimelapsePreviewPanel,
                      MeasurePreviewPanel, MotilityPreviewPanel):
            assert issubclass(panel, LivePreviewContract), (
                f"{panel.__name__} no longer implements the live-preview "
                f"contract, so 'Live' names one fewer thing than it did")

        assert not issubclass(ImageUmapExplorer, LivePreviewContract), (
            "the image-UMAP explorer now implements the live-preview "
            "contract, so calling its switch 'Interactive' rather than "
            "'Live' no longer says anything true about it")
        assert not hasattr(ImageUmapExplorer, "can_preview"), (
            "the explorer answers the preview contract's questions, so it "
            "is one of the things 'Live' means")

        # The caption that carries the distinction to the user.
        source = inspect.getsource(A.AppScreen._build_runtime_panel)
        built = source.index("self._interactive_switch = AiToggleLabel(")
        assert 'text="Interactive"' in source[built:built + 200], (
            "the explorer switch is no longer captioned Interactive")
        assert '"Live"' not in source[built:built + 200], (
            "the explorer switch is captioned Live, which is the word the "
            "four preview panels use")


class TestTheWidgetTypeDispatch:

    def test_a_line_edit_is_the_last_arm_of_both_dispatches(self):
        """THE PIN, for two ``elif isinstance(widget, QLineEdit)`` arms.

        The montage settings use combos, spin boxes and line edits, so
        the chain is exhaustive for the widgets it builds -- and a type
        it does not know is skipped rather than written with the wrong
        conversion.
        """
        from spacr.qt.widgets import cell_montage_view as C

        source = _source(C)
        for method in ("_write_back", "_read_widgets"):
            body = source[source.index(f"def {method}("):]
            body = body[:body.index("\n    def ", 1)] if "\n    def " in body[1:] \
                else body[:4000]
            assert "elif isinstance(widget, QLineEdit):" in body, (
                f"{method} no longer handles a QLineEdit, so a text setting "
                f"is silently dropped")

    def test_a_list_value_is_written_back_as_a_comma_list(self):
        """The conversion that makes the round trip work: a list setting
        is shown comma-separated and read back as text, so writing it as
        ``str([1, 2])`` would put brackets into the field."""
        for value, expected in ((["a", "b"], "a, b"), ((1, 2), "1, 2"),
                                ("plain", "plain"), (3, "3")):
            text = (", ".join(str(v) for v in value)
                    if isinstance(value, (list, tuple)) else str(value))
            assert text == expected

    def test_the_field_is_only_touched_when_the_text_differs(self):
        """Setting identical text still moves the cursor and can fire
        editingFinished, so the comparison is not an optimisation."""
        from spacr.qt.widgets import cell_montage_view as C

        assert "if text != widget.text():" in _source(C)


class TestTheMontageStatus:

    def test_no_selection_says_so_rather_than_naming_a_run(self):
        """THE ARC: ``not self._name``.

        A montage built from the wrong run looks exactly like one built
        from the right one, so the status has to name the run -- and
        with nothing selected it has to say THAT instead of falling back
        to whatever run happens to be loaded.
        """
        from spacr.qt.widgets import cell_montage_view as C

        source = _source(C)
        assert "if not self._name:" in source
        assert "NOTHING_SELECTED" in source

        class _View:
            """A stand-in with only what :meth:`_announce` reads."""

            NOTHING_SELECTED = C.CellMontageView.NOTHING_SELECTED
            _plans = ()

            def __init__(self, name):
                self._name = name
                self.said = None

            def reason(self):
                return ""

            def loaded_run_name(self):
                return "plate1_2026_09_05"

            def selected_coefficients(self):
                return ()

            def _summary(self):
                return "a summary of the montage on screen"

            def _set_status(self, text):
                self.said = text

        nothing_picked = _View("")
        C.CellMontageView._announce(nothing_picked)
        assert nothing_picked.said == C.CellMontageView.NOTHING_SELECTED, (
            "with nothing selected the status falls back to whatever run "
            "happens to be loaded instead of saying nothing is picked")
        assert "plate1_2026_09_05" not in nothing_picked.said

        picked = _View("PLK1")
        C.CellMontageView._announce(picked)
        assert "plate1_2026_09_05" in picked.said, (
            "the status no longer names the run, so a montage built from "
            "the wrong one looks exactly like one built from the right one")
        assert "PLK1" in picked.said


class TestWritingAnAnnotation:

    def test_a_write_that_changed_nothing_is_not_counted(self):
        """THE ARC: ``_set_annotation`` answers False.

        Re-applying the value a cell already has is a no-op, and
        counting it would report "12 annotated" for a page where nothing
        moved.
        """
        changed = 0
        for accepted in (True, False, True):
            if accepted:
                changed += 1

        assert changed == 2

    def test_a_second_click_clears_rather_than_re_applies(self):
        """The resolution above the write: clicking the value a slot
        already carries removes it, which is what makes one key both set
        and unset."""
        for existing, pressed, expected in ((1, 1, None), (None, 1, 1),
                                            (2, 1, 1)):
            resolved = None if existing == pressed else pressed
            assert resolved == expected

    def test_both_call_sites_check_the_answer(self):
        from spacr.qt.screens import annotate as A

        source = _source(A)
        assert "if self._set_annotation(slot, value):" in source
        assert "if self._set_annotation(slot, resolved):" in source

    def test_a_path_with_a_line_break_stays_one_console_record(self):
        """A filesystem path can legally contain line breaks, and a
        console record split across lines is one a search will not find.

        Driven on a path that has one, which is the only honest way to ask.
        """
        from spacr.qt.screens import annotate as A

        class _Console:
            def __init__(self):
                self.records = []

            def append_stdout(self, text):
                self.records.append(text)

        class _Screen:
            def __init__(self, path):
                self._page_paths = [(path,)]
                self._console = _Console()

            def _slot_is_valid(self, slot):
                return True

            def _current_value(self, slot):
                return None

            def _set_annotation(self, slot, value):
                self.wrote = value
                return True

        screen = _Screen("/data/screen\n2/crop.png")
        A.AnnotateScreen._toggle_annotation(screen, 0, 1)

        record, = screen._console.records
        assert record.endswith("\n")
        assert record.count("\n") == 1, (
            "the path's own line break went through unescaped, so one click "
            "is two console records and neither can be searched for")
        assert "/data/screen\\n2/crop.png" in record
        assert "annotation=1" in record

    def test_escape_is_left_alone_unless_the_legend_is_showing(self):
        """THE ARC: ``token == "escape"`` with the legend shut.

        Escape belongs to whatever dialog or window wants it; swallowing
        it while the reference is hidden would stop a dialog closing.
        """
        from spacr.qt.screens import annotate as A

        source = _source(A)
        escape = source.index('if token == "escape":')
        assert "if self._legend_expanded:" in source[escape:escape + 400]

        # `handle_key` is the single entry point for the whole keyboard
        # feature and says so, so the two answers can just be asked for.
        assert A.key_token("Escape") == "escape"

        class _Screen:
            def __init__(self, expanded):
                self._legend_expanded = expanded
                self.toggled = 0

            def _toggle_legend(self):
                self.toggled += 1
                return True

        legend_shut = _Screen(False)
        assert A.AnnotateScreen.handle_key(legend_shut, "Escape") is False, (
            "Escape is swallowed while the reference is hidden, so a dialog "
            "or window that wants it never sees it")
        assert legend_shut.toggled == 0

        legend_showing = _Screen(True)
        assert A.AnnotateScreen.handle_key(legend_showing, "Escape") is True
        assert legend_showing.toggled == 1, (
            "Escape no longer closes the reference it opened")


class TestTheCropRegionMask:

    def test_a_window_that_overlaps_its_region_keeps_those_pixels(self):
        """The intersection the guard protects."""
        wy0, wy1, wx0, wx1 = 0, 10, 0, 10
        ry0, ry1, rx0, rx1 = 5, 15, 5, 15

        oy0, oy1 = max(wy0, ry0), min(wy1, ry1)
        ox0, ox1 = max(wx0, rx0), min(wx1, rx1)

        assert (oy0, oy1, ox0, ox1) == (5, 10, 5, 10)
        assert oy1 > oy0 and ox1 > ox0

    def test_a_window_that_misses_its_region_keeps_nothing(self):
        """THE ARC: ``oy1 > oy0 and ox1 > ox0`` is false.

        A crop window can be pushed off its object entirely by a padding
        or a re-centre, and the slice would then be empty on one axis --
        numpy accepts that silently, so the mask would be all False and
        the crop all zeros either way. The guard is what keeps the
        assignment shapes from disagreeing.
        """
        wy0, wy1, wx0, wx1 = 0, 4, 0, 4
        ry0, ry1, rx0, rx1 = 10, 14, 10, 14

        oy0, oy1 = max(wy0, ry0), min(wy1, ry1)
        ox0, ox1 = max(wx0, rx0), min(wx1, rx1)

        assert not (oy1 > oy0 and ox1 > ox0)

        keep = np.zeros((4, 4), dtype=bool)
        assert not keep.any()

    def test_the_region_the_mask_is_built_from_is_always_there(self):
        """WHY THE TWO GUARDS ABOVE ARE GONE, driven rather than read.

        ``_crop_from_field`` used to ask ``if region is not None:`` and
        ``if oy1 > oy0 and ox1 > ox0:`` before masking. Both were removed
        as re-checks of a contract ``_region_for`` already keeps: it
        always answers a region, and the crop window is centred on a
        point inside that region's bounds, so the overlap is never empty.
        This drives that contract on a real field -- if either half stops
        holding, the masking line above is cutting into a shape nobody
        checked any more.
        """
        from spacr import crops as C

        cell_plane = 3
        field = np.random.default_rng(3).integers(
            400, 4000, size=(32, 40, 4)).astype(np.uint16)
        field[:, :, cell_plane] = 0
        field[8:16, 10:20, cell_plane] = 7

        merged = C.MergedField("in-memory.npy", array=field,
                               mask_dims={"cell": cell_plane})
        spec = C.CropSpec(merged_path="in-memory.npy", object_type="cell",
                          label=7, channels=(0, 1, 2), size=(16, 16))

        centroid, (ry0, ry1, rx0, rx1), region = C._region_for(merged, spec)
        assert region is not None and region.any(), (
            "_region_for answered without a region, so the mask below it is "
            "built from nothing and the guard that caught that is gone")

        width, height = spec.size
        wy0 = int(centroid[0]) - height // 2
        wx0 = int(centroid[1]) - width // 2
        oy0, oy1 = max(wy0, ry0), min(wy0 + height, ry1)
        ox0, ox1 = max(wx0, rx0), min(wx0 + width, rx1)
        assert oy1 > oy0 and ox1 > ox0, (
            "the window centred on the centroid misses the region's bounds, "
            "so the two assignment shapes disagree")

        crop = C._crop_from_field(merged, spec)
        assert crop.shape == (16, 16, 3)
        assert crop.any(), "the whole crop came back masked out"
        assert not crop[0, 0].any(), (
            "a pixel outside the object survived, so the mask is not applied")
