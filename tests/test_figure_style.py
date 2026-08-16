"""One general figure style, plus per-graph overrides.

Every plot inherited matplotlib's defaults, which is what "the graphs look
pretty ugly" means. Restyling by hand after a run is work the application
should do once -- and a per-figure restyle is lost the next time the analysis
is re-run, which during a revision is constantly.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(autouse=True)
def _matplotlib_is_left_as_it_was_found():
    """Put matplotlib's global drawing style back after every test here.

    ``figure_style.apply`` is deliberately global -- one style, every figure
    the run draws -- so the tests below that call it are changing the style
    for the whole process, not for themselves. Nothing put it back, and the
    settings it writes are the ones that decide where things land on the
    page: ``figure.autolayout`` on, ``figure.dpi`` at 150, a font family, a
    colour cycle.

    ``figure.autolayout`` was the one that bit. With it on, matplotlib
    re-lays-out the axes on every draw, so a later test that positioned its
    own axes found them moved -- and the failure was reported against that
    test, in another file, rather than against the one that changed the
    style. ``test_the_legend_fits_inside_the_figure`` in
    ``test_toxo_volcano_join_contract.py`` was the one that drew the short
    straw: it asserts a 27-entry legend overflows a 20x20 figure, and with
    autolayout on the axes had already been shrunk to make it fit.

    ``rc_context`` with no arguments is exactly this: snapshot on entry,
    restore on exit.
    """
    import matplotlib

    with matplotlib.rc_context():
        yield


class TestResolution:

    def test_a_graph_kind_only_states_what_it_differs_on(self):
        """So a change to the general font reaches every plot that has not
        overridden it."""
        from spacr.figure_style import resolve

        general = resolve()
        volcano = resolve("volcano")
        assert volcano["font_family"] == general["font_family"]
        assert volcano["marker_size"] != general["marker_size"]

    def test_a_per_graph_override_does_not_leak_to_another_graph(self):
        """The settings that make a volcano readable are not the ones that
        make a plate heatmap readable."""
        from spacr.figure_style import resolve

        overrides = {"volcano": {"marker_size": 99.0}}
        assert resolve("volcano", overrides=overrides)["marker_size"] == 99.0
        assert resolve("plate_heatmap", overrides=overrides)["marker_size"] \
            == resolve()["marker_size"]

    def test_the_user_beats_the_default_and_the_graph_beats_the_user(self):
        from spacr.figure_style import resolve

        style = resolve("volcano",
                        general={"marker_size": 50.0},
                        overrides={"volcano": {"marker_size": 5.0}})
        assert style["marker_size"] == 5.0

    def test_a_none_value_does_not_erase_a_default(self):
        """An unset control must not blank the setting underneath it."""
        from spacr.figure_style import resolve

        assert resolve(general={"font_family": None})["font_family"] \
            == resolve()["font_family"]

    def test_an_unknown_graph_kind_falls_back_to_the_general_style(self):
        from spacr.figure_style import resolve

        assert resolve("no_such_graph")["font_size"] == resolve()["font_size"]


class TestTheHeatmapKeepsItsShape:

    def test_a_plate_is_not_forced_square(self):
        """A plate is 24x16 wells; forcing it square stops the wells being
        square, which is the whole point of looking at one."""
        from spacr.figure_style import resolve

        assert resolve("plate_heatmap")["aspect"] == "equal"
        assert resolve("plate_heatmap")["per_row"] >= 1


class TestRcParams:

    def test_every_key_is_one_matplotlib_actually_has(self):
        """A style carries settings no rcParam expresses -- per_row,
        label_top_n -- and passing those to rcParams.update raises."""
        import matplotlib

        from spacr.figure_style import rc_params, resolve

        for kind in (None, "volcano", "plate_heatmap", "jitter_bar"):
            for key in rc_params(resolve(kind)):
                assert key in matplotlib.rcParams, f"{key} is not an rcParam"

    def test_applying_it_actually_changes_matplotlib(self):
        import matplotlib

        from spacr.figure_style import apply

        apply("volcano", general={"font_size": 17.0})
        assert matplotlib.rcParams["font.size"] == 17.0

    def test_spine_presets_are_translated(self):
        from spacr.figure_style import rc_params, resolve

        params = rc_params(resolve(general={"spines": "none"}))
        assert not any(params[f"axes.spines.{side}"]
                       for side in ("top", "right", "bottom", "left"))

    def test_styling_never_raises_into_a_run(self):
        """A bad preference must not sink an analysis."""
        from spacr.figure_style import apply

        apply("volcano", general={"font_size": "not a number"})


class TestPersistence:

    def test_nothing_is_stored_until_the_user_sets_something(self, qtbot):
        """Storing the defaults would freeze today's values into every user's
        settings and make improving them impossible."""
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.setOrganizationName("spacr-test")
        QCoreApplication.setApplicationName("figstyle-empty")
        from spacr.qt.preferences import (get_figure_style,
                                          get_figure_style_per_graph,
                                          set_figure_style,
                                          set_figure_style_per_graph)

        set_figure_style({})
        set_figure_style_per_graph({})
        assert get_figure_style() == {}
        assert get_figure_style_per_graph() == {}

    def test_a_round_trip_keeps_both_levels(self, qtbot):
        from PySide6.QtCore import QCoreApplication

        QCoreApplication.setOrganizationName("spacr-test")
        QCoreApplication.setApplicationName("figstyle-roundtrip")
        from spacr.qt.preferences import (apply_figure_style, set_figure_style,
                                          set_figure_style_per_graph)

        set_figure_style({"font_size": 16.0})
        set_figure_style_per_graph({"volcano": {"marker_size": 40.0}})

        volcano = apply_figure_style("volcano")
        heatmap = apply_figure_style("plate_heatmap")
        assert volcano["font_size"] == 16.0 and volcano["marker_size"] == 40.0
        assert heatmap["font_size"] == 16.0
        assert heatmap["marker_size"] != 40.0
