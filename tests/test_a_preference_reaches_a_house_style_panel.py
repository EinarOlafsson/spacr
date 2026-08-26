"""A user's figure preference reaches a house-style panel.

Instruction 127, finding 3, and the third line of its acceptance. spaCR has
TWO figure-style systems and they answer to different masters:

    spacr/figure_style.py    the USER's preference -- general and per-graph,
                             instruction 118, the Preferences dialog
    spacr/figures/style.py   the PUBLICATION house style, from the
                             apicomplexan-figures skill

Both are legitimate. The two being unaware of each other was not, and the bug
that overlap hid is the one measured here: every panel drawn through
`figures.style.figure_style` -- the whole regression QC suite, the toxo
figures, the house-style sheet -- ignored the Preferences dialog completely.

THE HOUSE STYLE IS THE BASE AND THE PREFERENCE IS THE OVERRIDE, in that order,
and only the settings the user actually moved are applied. `GENERAL_DEFAULTS`
is a complete style of its own -- 11 pt DejaVu Sans, gridlines on, a white
ground -- so applying it wholesale would replace the published look for every
user who has never opened Preferences. That case is tested first.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
pytest.importorskip("PySide6")

from spacr.figures import style as house  # noqa: E402


@pytest.fixture()
def prefs(monkeypatch):
    """Drive the two preference stores directly. They hold DELTAS, not the
    defaults, which is what makes the override layer computable at all."""
    from spacr.qt import preferences

    state = {"general": {}, "per_graph": {}}
    monkeypatch.setattr(preferences, "get_figure_style",
                        lambda: dict(state["general"]))
    monkeypatch.setattr(preferences, "get_figure_style_per_graph",
                        lambda: {k: dict(v) for k, v in state["per_graph"].items()})
    return state


def test_an_untouched_preference_leaves_the_published_look_alone(prefs):
    """A fresh install draws the house style, not `GENERAL_DEFAULTS`."""
    assert house.user_overrides() == {}
    params = house.rc("print")
    # The three the defaults would have trampled, and each is a house rule:
    # no gridlines ever, the measured type scale, a transparent ground.
    assert params["axes.grid"] is False
    assert params["font.size"] == house.TYPE_SCALE["tick"]
    assert params["figure.facecolor"] == house.TRANSPARENT


def test_the_general_preference_reaches_the_house_style(prefs):
    prefs["general"]["font_size"] = 22.0
    assert house.rc("print")["font.size"] == 22.0
    # And it changed ONLY what it names.
    assert house.rc("print")["axes.grid"] is False


def test_the_preference_reaches_a_drawn_panel(prefs):
    """Measured on the panel, not on the dict. `build_panel` is what the
    regression run and the figure grid actually call."""
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    from spacr.figures import build_panel

    rng = np.random.default_rng(2)
    frame = pd.DataFrame({
        "gene": [f"g{i}" for i in range(60)],
        "coefficient": rng.normal(0, 0.4, 60),
        "p_value": rng.uniform(1e-6, 1, 60),
    })

    figure, _panel = build_panel("volcano", frame, target="print")
    house_size = figure.axes[0].xaxis.get_ticklabels()[0].get_fontsize()

    prefs["general"]["tick_size"] = 24.0
    figure, _panel = build_panel("volcano", frame, target="print")
    chosen_size = figure.axes[0].xaxis.get_ticklabels()[0].get_fontsize()

    assert house_size == pytest.approx(house.TYPE_SCALE["tick"])
    assert chosen_size == 24.0, (
        "the Preferences dialog does not reach a house-style panel")


def test_a_per_graph_preference_reaches_only_its_own_graph(prefs):
    """"Changing the volcano's point size must not touch the heatmaps."" """
    prefs["per_graph"]["volcano"] = {"label_size": 30.0}

    assert house.rc("print", kind="volcano")["axes.labelsize"] == 30.0
    assert house.rc("print", kind="plate_heatmap")["axes.labelsize"] == \
        house.TYPE_SCALE["label"]
    assert house.rc("print")["axes.labelsize"] == house.TYPE_SCALE["label"]


def test_a_broken_preference_store_does_not_lose_the_figure(monkeypatch):
    """A preference is never worth losing a figure over."""
    from spacr.qt import preferences

    def angry():
        raise RuntimeError("the settings file is a directory")

    monkeypatch.setattr(preferences, "get_figure_style", angry)
    assert house.user_overrides() == {}
    assert house.rc("print")["font.size"] == house.TYPE_SCALE["tick"]


def test_no_palette_preference_leaves_the_house_colour_vocabulary_alone(prefs):
    """An untouched palette adds no colour cycle at all.

    The house style fixes the colour VOCABULARY -- grey for everything the
    sentence is not about, one highlight for what it is -- so a user who has
    never opened Preferences keeps it. Only an explicit choice overrides it.
    """
    assert house.user_overrides() == {}
    assert "axes.prop_cycle" not in house.rc("print")


def test_a_chosen_palette_does_reach_a_house_style_panel(prefs):
    """A palette the user picked is the cycle a house-style panel draws in.

    This was once the opposite assertion: the palette was excluded from the
    override layer on the grounds that a categorical cycle laid over the house
    style is the rainbow the style exists to forbid. That exclusion was
    reversed -- "changing the palette in Preferences changes the next run's
    figures without any per-figure work" -- and the reversal is what makes the
    exclusion visible as a CHOICE rather than as the accident it started as:
    `rc_params` simply did not emit the key, so nothing could apply it.

    The house vocabulary still holds by default; see the test above.
    """
    from spacr.figure_style import palette_colours

    prefs["general"]["palette"] = "deep"
    overrides = house.user_overrides()
    assert "axes.prop_cycle" in overrides
    cycle = house.rc("print")["axes.prop_cycle"]
    assert list(cycle.by_key()["color"]) == palette_colours("deep")
