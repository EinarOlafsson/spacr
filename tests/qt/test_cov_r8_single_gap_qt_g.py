"""Six more Qt-side single decisions: four driven, two pinned.

A colormap name the combo no longer holds, a credits file that is not a
record, a naming pass that has to invent neither well nor field, and a
comment on an existing issue that did not land -- driven. An icon that
re-inked to nothing and a label with no pixels left -- pinned, because
the code above each has already ruled it out.
"""
from __future__ import annotations

import json

import pytest

pytest.importorskip("PySide6")

import numpy as np

pytestmark = pytest.mark.qt


# ---------------------------------------------------------------------------
# iconset.py -- re-inking always produces an array
# ---------------------------------------------------------------------------

class TestReInkingAnIcon:

    def test_a_mask_is_painted_flat_in_the_theme_ink(self):
        from spacr.qt.iconset import reink

        rgba = np.zeros((8, 8, 4), dtype=float)
        rgba[2:6, 2:6, 3] = 255.0            # alpha is the shape

        inked = reink(rgba, "dark")

        assert inked is not None
        assert inked.dtype == np.uint8
        assert inked.shape == rgba.shape
        assert inked[2:6, 2:6, 3].all(), "the shape's alpha was lost"

    def test_an_entirely_transparent_icon_comes_back_unchanged(self):
        from spacr.qt.iconset import reink

        rgba = np.zeros((4, 4, 4), dtype=float)

        inked = reink(rgba, "light")
        assert inked is not None
        assert not inked[:, :, 3].any()

    def test_re_inking_never_returns_none(self):
        """THE PIN.

        ``_themed_array`` only writes its disk cache when ``reink``
        returned something, and ``reink`` has two returns, both arrays.
        Caching a None would put an empty PNG on disk that every later
        launch would read back as a blank icon -- which is why the guard
        is worth keeping even though nothing reaches it.
        """
        import inspect

        from spacr.qt import iconset

        source = inspect.getsource(iconset.reink)
        returns = [line.strip() for line in source.splitlines()
                   if line.strip().startswith("return")]
        assert returns, "reink has no return at all"
        assert all("None" not in line for line in returns), (
            f"reink can now return None: {returns}")

        for theme in ("dark", "light"):
            for alpha in (0.0, 255.0):
                rgba = np.zeros((4, 4, 4), dtype=float)
                rgba[:, :, 3] = alpha
                assert iconset.reink(rgba, theme) is not None


# ---------------------------------------------------------------------------
# layer_viewer.py -- a custom colormap name the combo no longer holds
# ---------------------------------------------------------------------------

class TestDroppingACustomColormapName:
    """A hex colour is added to the combo for the layer that uses it.

    ``ImageLayer`` accepts any colour spec, so a channel can be a
    ``#rrggbb`` ramp the built-in COLORMAPS list has no entry for. The
    viewer adds it, remembers that it did, and takes it back out when the
    selection moves on -- otherwise one layer's private colour would sit
    in the list offered for every other layer.
    """

    def _viewer(self, qtbot):
        from spacr.layers import LayerStack
        from spacr.qt import layer_viewer as lv

        stack = LayerStack()
        stack.add_image(np.full((16, 16), 500, dtype=np.uint16),
                        name="image", colormaps="#ff00aa")
        viewer = lv.LayerViewer(stack)
        qtbot.addWidget(viewer)
        viewer.resize(320, 320)
        return viewer, stack

    def test_the_custom_colour_is_offered_while_its_layer_is_selected(
            self, qtbot):
        viewer, _stack = self._viewer(qtbot)

        assert viewer.colormap_combo.findText("#ff00aa") >= 0
        assert viewer._custom_colormap_name == "#ff00aa"

    def test_it_is_taken_back_out_when_the_layer_stops_using_it(self,
                                                                qtbot):
        viewer, stack = self._viewer(qtbot)

        stack["image"].colormap = "magenta"
        viewer._sync_controls()

        assert viewer.colormap_combo.findText("#ff00aa") < 0, (
            "one layer's private colour was left in the shared list")
        assert viewer._custom_colormap_name in ("", "magenta")

    def test_a_remembered_name_the_combo_no_longer_holds_is_just_forgotten(
            self, qtbot):
        """THE UNCOVERED ARC.

        The combo is rebuilt on a theme change and by
        :meth:`_refresh_colormaps`, and the remembered name does not
        survive that. ``removeItem(-1)`` is not a no-op in Qt -- it is an
        out-of-range index -- so the lookup has to gate the removal, and
        the name is cleared either way.
        """
        viewer, stack = self._viewer(qtbot)

        viewer._custom_colormap_name = "a colour from a rebuilt combo"
        assert viewer.colormap_combo.findText(
            "a colour from a rebuilt combo") < 0
        before = viewer.colormap_combo.count()

        viewer._sync_controls()             # must not raise

        assert viewer.colormap_combo.count() >= before - 1
        assert viewer._custom_colormap_name != "a colour from a rebuilt combo"


# ---------------------------------------------------------------------------
# ortho_view.py -- every label the table reports still has pixels
# ---------------------------------------------------------------------------

class TestCentringOnALinkedSelection:

    def _view(self, qtbot):
        from spacr.layers import FieldKey, LayerStack, Spacing
        from spacr.qt import ortho_view as ov

        spacing = Spacing.from_map({"z": 2.0, "y": 0.65, "x": 0.65},
                                   units="um")
        stack = LayerStack()
        data = np.zeros((10, 64, 64), np.uint16)
        data[5, 30:34, 30:34] = 4000
        stack.add_image(data, name="volume", spacing=spacing,
                        contrast_limits=(0.0, 4000.0))
        mask = np.zeros((10, 64, 64), np.int32)
        mask[5, 30:34, 30:34] = 17
        field = FieldKey(values=dict(zip(FieldKey.columns(),
                                         ("plate1", "A", "1", "1"))))
        stack.add_labels(mask, name="mask", spacing=spacing, field=field)

        view = ov.OrthoView(stack)
        qtbot.addWidget(view)
        view.resize(520, 520)
        return view, stack

    def test_a_selection_from_another_view_moves_the_crosshair(self, qtbot):
        from spacr.qt.linked_selection import Selection

        view, stack = self._view(qtbot)

        view.on_linked_selection_changed(
            Selection(keys=["plate1_A_1_1_17"]))

        assert stack["mask"].selected_label == 17
        assert view.slice_index("z") == 5, (
            "the crosshair did not move onto the selected object")

    def test_a_key_no_layer_can_name_is_ignored(self, qtbot):
        from spacr.qt.linked_selection import Selection

        view, stack = self._view(qtbot)
        before = view.slice_index("z")

        view.on_linked_selection_changed(Selection(keys=["not_an_object"]))

        assert stack["mask"].selected_label == 0
        assert view.slice_index("z") == before

    def test_every_reported_label_still_has_pixels(self, qtbot):
        """THE PIN.

        ``if len(where):`` guards against a label the table reports that
        the array no longer holds -- ``where.mean(axis=0)`` over an empty
        array is nan, and moving the crosshair to nan leaves the view
        pointing at nowhere with no way back.

        It cannot fire, because ``labels()`` is computed from the very
        array the search then scans: ``np.unique`` of the layer's data.
        If labels() is ever backed by a stored table instead, this fails
        first.
        """
        _view, stack = self._view(qtbot)
        layer = stack["mask"]

        reported = list(layer.labels())
        assert reported == [17]
        for label in reported:
            where = np.argwhere(np.asarray(layer.data) == int(label))
            assert len(where), (
                f"label {label} is reported but has no pixels")

        import inspect

        from spacr.layers import LabelsLayer

        source = inspect.getsource(LabelsLayer.labels)
        assert "np.unique(self._data)" in source, (
            "labels() is no longer derived from the layer's own array, so "
            "it can report a label the search cannot find")


# ---------------------------------------------------------------------------
# space.py -- a credits file that is not a record
# ---------------------------------------------------------------------------

class TestReadingTheImageCredits:

    def _imagery(self, monkeypatch, tmp_path):
        from spacr.qt import space

        monkeypatch.setattr(space, "imagery_dir", lambda: tmp_path)
        return space

    def test_credits_naming_a_file_that_is_there_are_returned(
            self, monkeypatch, tmp_path):
        space = self._imagery(monkeypatch, tmp_path)

        (tmp_path / "backdrop.jpg").write_bytes(b"\xff\xd8\xff")
        (tmp_path / space.CREDITS_FILE).write_text(json.dumps(
            {"file": "backdrop.jpg", "title": "A nebula"}))

        credits = space.read_credits()
        assert credits is not None and credits["title"] == "A nebula"

    def test_credits_naming_a_file_that_is_gone_are_refused(
            self, monkeypatch, tmp_path):
        space = self._imagery(monkeypatch, tmp_path)

        (tmp_path / space.CREDITS_FILE).write_text(json.dumps(
            {"file": "deleted.jpg", "title": "A nebula"}))

        assert space.read_credits() is None, (
            "an attribution was returned for an image that is not there")

    def test_a_credits_file_that_is_not_a_record_is_refused(
            self, monkeypatch, tmp_path):
        """THE UNCOVERED ARC.

        A half-written or hand-edited file parses as JSON and is not a
        record -- a list, a bare string, an object with no ``file``.
        Returning it would put an attribution on screen for an image
        nobody can name.
        """
        space = self._imagery(monkeypatch, tmp_path)

        for content in ("[1, 2, 3]", '"just a string"', "{}",
                        '{"title": "no file key"}'):
            (tmp_path / space.CREDITS_FILE).write_text(content)
            assert space.read_credits() is None, (
                f"{content} was accepted as an attribution")


# ---------------------------------------------------------------------------
# folder_metadata.py -- neither well nor field to invent
# ---------------------------------------------------------------------------

class TestMintingTheMissingIdentifiers:

    def _names(self, n=4):
        return [f"/data/img_{i}.tif" for i in range(n)]

    def test_with_neither_known_the_field_counts_up(self):
        from spacr.qt.folder_metadata import assign_missing_fields

        mapped = assign_missing_fields(self._names(), plate="plate1",
                                       have_well=False, have_field=False,
                                       have_channel=False)

        assert [m.field for m in mapped] == [1, 2, 3, 4]
        assert {m.well for m in mapped} == {"A01"}

    def test_with_the_well_known_only_the_well_counts_up(self):
        from spacr.qt.folder_metadata import assign_missing_fields

        mapped = assign_missing_fields(self._names(), plate="plate1",
                                       have_well=False, have_field=True,
                                       have_channel=False)

        assert len({m.well for m in mapped}) == 4, (
            "every file landed in the same well")
        assert {m.field for m in mapped} == {1}

    def test_with_both_known_nothing_is_invented(self):
        """THE UNCOVERED ARC: neither counter advances.

        Both identifiers came from the folder structure, so this pass
        exists only to build the canonical names. Advancing a counter
        here would mint a second, contradictory set of ids over the ones
        the caller already resolved.
        """
        from spacr.qt.folder_metadata import assign_missing_fields

        mapped = assign_missing_fields(self._names(), plate="plate1",
                                       have_well=True, have_field=True,
                                       have_channel=True)

        assert {m.well for m in mapped} == {"A01"}
        assert {m.field for m in mapped} == {1}
        assert len({m.original_path for m in mapped}) == 4


# ---------------------------------------------------------------------------
# ai/issue_report.py -- a comment that did not land
# ---------------------------------------------------------------------------

class TestReportingAnIssueThatWasSeenBefore:
    """The dedupe path, through the module's own offline seam.

    ``_transport_refusal`` blocks real GitHub transport for the whole
    pytest session and has no environment escape hatch -- a test that
    reached a write path unmocked once filed a live issue on the public
    tracker. Substituting ``_HTTP_OPEN`` is the seam it deliberately
    admits, so that is what these use.
    """

    def _seam(self, monkeypatch, *, comment_ok):
        from spacr.qt.ai import github_auth, issue_report

        calls = {"created": 0, "commented": 0}

        def commented(*_a, **_k):
            calls["commented"] += 1
            return comment_ok, ""

        def created(*_a, **_k):
            calls["created"] += 1
            return True, "https://x/issues/99"

        monkeypatch.setattr(github_auth, "_HTTP_OPEN",
                            lambda *a, **k: None)
        monkeypatch.setattr(github_auth, "is_authenticated", lambda: True)
        monkeypatch.setattr(github_auth, "find_issue_by_fingerprint",
                            lambda repo, fp: (
                                True, {"number": 12,
                                       "html_url": "https://x/issues/12"}))
        monkeypatch.setattr(github_auth, "comment_on_issue", commented)
        monkeypatch.setattr(github_auth, "create_issue", created)
        return issue_report, calls

    def _report(self):
        return {"fingerprint": "abc123", "title": "It broke",
                "body": "the traceback"}

    def test_a_landed_comment_returns_the_existing_issue(self, monkeypatch):
        """A second occurrence is still information: it says the bug is
        reproducible and carries that run's environment."""
        issue_report, calls = self._seam(monkeypatch, comment_ok=True)

        url = issue_report.submit_report(self._report())

        assert url == "https://x/issues/12"
        assert calls == {"created": 0, "commented": 1}

    def test_a_comment_that_failed_falls_through_to_a_new_issue(self,
                                                                monkeypatch):
        """THE UNCOVERED ARC.

        The duplicate was found, but the comment did not land -- a
        locked issue, a revoked token, a transient 5xx. Returning the
        existing URL then would tell the user their report was filed
        when nothing was written anywhere.
        """
        issue_report, calls = self._seam(monkeypatch, comment_ok=False)

        url = issue_report.submit_report(self._report())

        assert url == "https://x/issues/99"
        assert calls == {"created": 1, "commented": 1}

    def test_without_the_seam_the_whole_flow_is_refused(self):
        """The fence itself: no environment escape hatch, by design."""
        from spacr.qt.ai import github_auth

        assert github_auth._HTTP_OPEN is github_auth._REAL_HTTP_OPEN
        assert github_auth._transport_refusal(), (
            "GitHub transport is no longer refused inside a test run")
