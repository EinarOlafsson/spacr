"""The four screen-owned previews must start from the module's settings.

Instruction 77, item (b). A preview is opened to make a decision -- "is this
diameter right", "will this crop cut the cell in half" -- which is the worst
possible place for it to disagree with the run it is predicting. Before this
file, all four screen-owned previews (Mask, Measure, Timelapse, Motility)
were built, shown and run at their own hardcoded defaults: ``AppScreen``
wired only the push direction (``set_propagate_callback``) and never the
pull.

Two separate defects, and the second is the one that would look fixed:

* nothing ever called ``apply_settings`` for these four, and

* for Mask, calling it would not have helped. The panel emits
  ``cell_diameter`` / ``cell_FT`` / ``cell_CP_prob`` and reads back
  ``diameter`` / ``flow_threshold`` / ``CP_prob``, so the two directions are
  not inverses. Mask declares no such keys -- ``cell_channel`` and
  ``nucleus_channel`` DO land, so a seed written without the rename map
  changes the preview visibly while silently dropping exactly the three
  settings the user opened it to check.

Every assertion below therefore goes through ``_build_request()`` -- what
actually reaches Cellpose -- rather than stopping at "apply_settings accepted
the dict".
"""
from __future__ import annotations

import numpy as np
import pytest

from spacr.qt.widgets import live_preview as LP


@pytest.fixture(autouse=True)
def _qapp(qapp):
    return qapp


def _mask_settings(**over):
    """A Mask settings dict in the module's OWN vocabulary."""
    out = {
        "cell_diameter": 42.5,
        "cell_FT": 0.9,
        "cell_CP_prob": -1.5,
        "nucleus_diameter": 17.0,
        "nucleus_FT": 0.15,
        "nucleus_CP_prob": 2.0,
        "cell_channel": 2,
        "nucleus_channel": 3,
        "pathogen_channel": 1,
        "organelle_channel": 0,
    }
    out.update(over)
    return out


class TestMaskPreviewSeeding:
    """Mask: the rename map, and that it reaches ``_build_request``."""

    def test_the_three_segmentation_settings_reach_the_cellpose_request(
            self, qtbot):
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings(_mask_settings())

        req = p._build_request()
        assert req.diameter == pytest.approx(42.5)
        assert req.flow_threshold == pytest.approx(0.9)
        assert req.cellprob == pytest.approx(-1.5)

    def test_the_channels_land_too(self, qtbot):
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings(_mask_settings())

        req = p._build_request()
        assert req.channels["cell"] == 2
        assert req.channels["nucleus"] == 3
        # Emitted by settings_for_propagation and never read back before.
        assert req.channels["pathogen"] == 1
        assert req.channels["organelle"] == 0

    def test_the_selected_compartment_decides_which_settings_are_read(
            self, qtbot):
        """Picking "nucleus" must seed from ``nucleus_*``, not ``cell_*``.

        The panel has ONE diameter/flow/prob triple and an object selector,
        so which compartment those three mean is decided by the selector.
        Seeding from ``cell_*`` while segmenting the nucleus is the same
        wrong-compartment error in the other direction.
        """
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p._object_box.setCurrentText("nucleus")
        p.apply_settings(_mask_settings())

        req = p._build_request()
        assert req.diameter == pytest.approx(17.0)
        assert req.flow_threshold == pytest.approx(0.15)
        assert req.cellprob == pytest.approx(2.0)

    def test_propagation_writes_back_to_the_selected_compartment(self, qtbot):
        """The tuned value must return to the setting it came from.

        With "nucleus" selected the panel segments the nucleus, so a
        diameter tuned here is a nucleus diameter. Writing it to
        ``cell_diameter`` leaves ``nucleus_diameter`` at its old value and
        the run then uses neither the tuned number nor the one on screen.
        """
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p._object_box.setCurrentText("nucleus")
        p._diameter.setValue(23.0)

        out = p.settings_for_propagation()
        assert out["nucleus_diameter"] == pytest.approx(23.0)
        assert "cell_diameter" not in out

    def test_cell_stays_the_default_vocabulary(self, qtbot):
        """The default selection is "cell", and must keep emitting ``cell_*``."""
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p._diameter.setValue(31.0)

        out = p.settings_for_propagation()
        assert out["cell_diameter"] == pytest.approx(31.0)

    def test_apply_then_propagate_is_a_fixed_point(self, qtbot):
        """The two directions must be inverses of one another.

        This is the invariant whose absence caused the bug: the panel spoke
        one vocabulary outward and a different one inward, and nothing
        compared them.
        """
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings(_mask_settings())

        out = p.settings_for_propagation()
        assert out["cell_diameter"] == pytest.approx(42.5)
        assert out["cell_FT"] == pytest.approx(0.9)
        assert out["cell_CP_prob"] == pytest.approx(-1.5)
        assert out["cell_channel"] == 2
        assert out["nucleus_channel"] == 3

    def test_a_module_that_speaks_the_panels_own_names_still_works(self, qtbot):
        """``cellpose_masks`` and ``analyze_plaques`` declare bare names.

        They reach the same panel through the preview registry, whose
        ``propagation`` map exists precisely because they call the settings
        ``diameter`` / ``flow_threshold`` / ``CP_prob``. Adding the
        compartment aliases must not cost them their own vocabulary.
        """
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings({"diameter": 12.0, "flow_threshold": 0.2,
                          "CP_prob": 1.0})

        req = p._build_request()
        assert req.diameter == pytest.approx(12.0)
        assert req.flow_threshold == pytest.approx(0.2)
        assert req.cellprob == pytest.approx(1.0)

    def test_the_native_name_wins_when_a_dict_carries_both(self, qtbot):
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings({"diameter": 12.0, "cell_diameter": 99.0})

        assert p._build_request().diameter == pytest.approx(12.0)

    def test_the_whole_dict_still_reaches_the_pre_and_post_routes(self, qtbot):
        """Seeding must not cost the filter settings their route.

        ``_build_request`` hands ``self._settings`` to both the
        preprocessing and the postprocessing side, so a rewrite that passed
        only the translated keys would silently drop every size and
        intensity filter.
        """
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings(_mask_settings(cell_min_size=250))

        req = p._build_request()
        assert req.preprocess_settings["cell_min_size"] == 250
        assert req.postprocess_settings["cell_min_size"] == 250

    def test_junk_values_do_not_take_the_panel_down(self, qtbot):
        p = LP.LivePreviewPanel()
        qtbot.addWidget(p)
        p.apply_settings({"cell_diameter": "thirty", "cell_FT": None})
        assert p._build_request() is not None


class TestMeasurePreviewSeeding:
    """Measure: the crop preview had no ``apply_settings`` at all."""

    def _panel(self, qtbot):
        from spacr.qt.widgets.measure_preview import MeasurePreviewPanel
        p = MeasurePreviewPanel(threaded=False)
        qtbot.addWidget(p)
        return p

    def test_the_crop_geometry_is_seeded(self, qtbot):
        """``png_size`` and the dilation decide what the crop LOOKS like.

        A preview shown at 224x224 while the run cuts 128x128 answers the
        wrong question, and the answer looks perfectly plausible.
        """
        p = self._panel(qtbot)
        p.apply_settings({
            "png_size": [96, 72],
            "dialate_pngs": True,
            "dialate_png_ratios": [1.75],
            "use_bounding_box": True,
        })

        out = p.settings_for_propagation()
        assert out["png_size"] == [96, 72]
        assert out["dialate_pngs"] is True
        assert out["dialate_png_ratios"] == [pytest.approx(1.75)]
        assert out["use_bounding_box"] is True

    def test_the_mask_dims_and_min_sizes_are_seeded(self, qtbot):
        p = self._panel(qtbot)
        p.apply_settings({
            "cell_mask_dim": 4,
            "nucleus_mask_dim": 5,
            "pathogen_mask_dim": 6,
            "organelle_mask_dim": 7,
            "cell_min_size": 210,
            "nucleus_min_size": 55,
        })

        out = p.settings_for_propagation()
        assert out["cell_mask_dim"] == 4
        assert out["nucleus_mask_dim"] == 5
        assert out["pathogen_mask_dim"] == 6
        assert out["organelle_mask_dim"] == 7
        assert out["cell_min_size"] == 210
        assert out["nucleus_min_size"] == 55

    def test_normalize_survives_both_of_its_shapes(self, qtbot):
        """``normalize`` is a bool OR a [lo, hi] percentile pair."""
        p = self._panel(qtbot)
        p.apply_settings({"normalize": [2.0, 98.5]})
        out = p.settings_for_propagation()
        assert out["normalize"] == [pytest.approx(2.0), pytest.approx(98.5)]

        p.apply_settings({"normalize": False})
        assert p.settings_for_propagation()["normalize"] is False

    def test_seeding_is_the_inverse_of_propagation(self, qtbot):
        p = self._panel(qtbot)
        first = p.settings_for_propagation()
        p.apply_settings(first)
        assert p.settings_for_propagation() == first

    def test_junk_values_do_not_take_the_panel_down(self, qtbot):
        p = self._panel(qtbot)
        p.apply_settings({"png_size": "large", "cell_mask_dim": None,
                          "dialate_png_ratios": []})
        assert p.settings_for_propagation() is not None

    def test_an_absent_mask_survives_the_round_trip_as_absent(self, qtbot):
        """``None`` and the spinbox's -1 must translate, both ways.

        Anything else turns "this plate has no organelle mask" into
        "the organelle mask is channel 0", which measures the wrong slice.
        """
        p = self._panel(qtbot)
        p.apply_settings({"organelle_mask_dim": None})
        assert p.settings_for_propagation()["organelle_mask_dim"] is None


class TestTheScreenSeedsOnFirstShow:
    """``AppScreen`` must pull, not only push."""

    def _screen(self, qtbot, app_key):
        from spacr.qt.screens.app_screen import AppScreen
        scr = AppScreen(app_key)
        qtbot.addWidget(scr)
        return scr

    def test_mask_seeds_the_panel_when_the_switch_is_turned_on(self, qtbot):
        scr = self._screen(qtbot, "mask")
        scr._settings_model.set_value_for_key("cell_diameter", 77.0)
        scr._settings_model.set_value_for_key("cell_FT", 0.75)

        scr._on_preview_switch(True)

        req = scr._live_preview._build_request()
        assert req.diameter == pytest.approx(77.0)
        assert req.flow_threshold == pytest.approx(0.75)

    def test_nothing_is_read_while_the_preview_stays_closed(self, qtbot):
        """A preview nobody opens must cost nothing.

        ``model.collect()`` is a pass over every widget on the screen, which
        is why priming is deferred to the first show rather than done at
        build time -- the same reasoning ``_PreviewHost.prime`` documents.
        """
        scr = self._screen(qtbot, "mask")
        calls = []
        real = scr._settings_model.collect
        scr._settings_model.collect = lambda *a, **k: (
            calls.append(1) or real(*a, **k))

        scr._on_preview_switch(False)
        assert calls == []

        scr._on_preview_switch(True)
        assert len(calls) == 1

    def test_the_seed_happens_once_not_on_every_reopen(self, qtbot):
        """Re-opening must not discard what the user tuned in the panel."""
        scr = self._screen(qtbot, "mask")
        scr._on_preview_switch(True)
        scr._live_preview._diameter.setValue(123.0)

        scr._on_preview_switch(False)
        scr._on_preview_switch(True)

        assert scr._live_preview._build_request().diameter == pytest.approx(123.0)

    @pytest.mark.parametrize("app_key", ["measure", "timelapse", "motility"])
    def test_every_screen_owned_preview_is_seeded(self, qtbot, app_key):
        """All four, not just the one that was noticed."""
        scr = self._screen(qtbot, app_key)
        panel_attr = {
            "measure": "_measure_preview",
            "timelapse": "_timelapse_preview",
            "motility": "_motility_preview",
        }[app_key]
        panel = getattr(scr, panel_attr)
        seen = []
        panel.apply_settings = lambda d: seen.append(dict(d))

        scr._on_preview_switch(True)

        assert len(seen) == 1, "the preview was never seeded from the form"
        assert seen[0], "the preview was seeded with an empty settings dict"

    def test_a_screen_without_a_settings_model_does_not_raise(self, qtbot):
        scr = self._screen(qtbot, "mask")
        scr._settings_model = None
        scr._on_preview_switch(True)          # must not raise
        assert not scr._live_preview_card.isHidden()

    def test_a_panel_that_refuses_the_seed_still_opens(self, qtbot):
        """A preview that cannot be seeded is still worth showing."""
        scr = self._screen(qtbot, "mask")

        def _explode(_d):
            raise RuntimeError("no")

        scr._live_preview.apply_settings = _explode
        scr._on_preview_switch(True)          # must not raise
        assert not scr._live_preview_card.isHidden()
