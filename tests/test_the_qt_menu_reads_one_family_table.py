"""The Qt regression menu and the CLI menu read the same family table.

The panel used to build its own list out of the bare inventory --
``["auto", *REGRESSION_TYPES]`` -- while ``settings_spec`` asked
``regression_family_choices()``. Two routes over one inventory can disagree
about what a family is called and which of the three kinds it is in, and this
pair did: one showed nineteen unlabelled names, the other showed them grouped
and explained.

The stored value is the part that must not move. A settings CSV names a
family by the value, so the menu may show whatever explains the choice best
as long as ``quantile`` is still written to disk as ``quantile``.
"""
from __future__ import annotations

import os

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.qt

from PySide6.QtWidgets import QComboBox  # noqa: E402

from spacr.qt.screens import settings_model as sm  # noqa: E402
from spacr.regression_families import (  # noqa: E402
    GROUP_TITLES, family_group, regression_family_choices)
from spacr.settings_spec import _regression_type_choices  # noqa: E402


def _menu(qtbot) -> QComboBox:
    """The regression_type control the panel actually builds."""
    widgets = sm.SettingsWidgets("regression")
    widget = widgets._widget_for("entry", None, "mixed", "regression_type")
    assert isinstance(widget, QComboBox)
    qtbot.addWidget(widget)
    return widget


def _stored(combo: QComboBox) -> list:
    return [combo.itemData(i) for i in range(combo.count())]


def _shown(combo: QComboBox) -> list:
    return [combo.itemText(i) for i in range(combo.count())]


def test_the_panel_offers_exactly_the_families_the_shared_table_does(qtbot):
    """Same values, same order, plus 'auto' at the head.

    A family the table places and the panel omits is unreachable from the
    GUI; one the panel offers and the table does not place is a fit with no
    stated assumption.
    """
    combo = _menu(qtbot)

    assert _stored(combo) == ["auto", *[v for v, _ in
                                        regression_family_choices()]]


def test_both_routes_agree_value_for_value_and_label_for_label(qtbot):
    """The Qt panel and settings_spec render one table, not two.

    Checked as pairs rather than as two sets: the failure this replaces was
    the same nineteen values carrying different labels on the two routes.
    """
    combo = _menu(qtbot)
    spec = _regression_type_choices()

    qt_pairs = [(combo.itemData(i), combo.itemText(i))
                for i in range(1, combo.count())]

    assert qt_pairs == spec


def test_the_menu_stores_the_value_and_shows_the_label(qtbot):
    """Reading the control gives back the stored value, never the caption.

    ``_read_widget`` returns ``itemData``, so a settings CSV written from this
    panel holds ``quantile`` and not the sentence explaining it.
    """
    widgets = sm.SettingsWidgets("regression")
    combo = widgets._widget_for("entry", None, "mixed", "regression_type")
    qtbot.addWidget(combo)

    index = _stored(combo).index("quantile")
    combo.setCurrentIndex(index)

    assert widgets._read_widget(combo) == "quantile"
    assert combo.currentText() != "quantile"
    assert combo.currentText().startswith("quantile")


def test_a_settings_file_written_before_the_grouping_still_selects(qtbot):
    """A stored value preselects even though no item text equals it.

    The preselect loop matches ``itemData`` first; when it did not, a CSV
    saying ``rlm`` would have silently opened the panel on whatever sat at
    index 0.
    """
    widgets = sm.SettingsWidgets("regression")
    widgets._defaults["regression_type"] = "rlm"
    combo = widgets._widget_for("entry", None, "rlm", "regression_type")
    qtbot.addWidget(combo)

    assert widgets._read_widget(combo) == "rlm"


def test_auto_keeps_its_own_spelling_and_says_what_it_does(qtbot):
    """'auto' is not a family and must not be labelled as one.

    It is the readable spelling of the historical ``None``; the fit path
    normalises it back, so the stored value has to stay the bare word.
    """
    combo = _menu(qtbot)

    assert combo.itemData(0) == "auto"
    assert combo.itemText(0).startswith("auto")
    assert "check_distribution" in combo.itemText(0)
    assert all(title not in combo.itemText(0)
               for title in GROUP_TITLES.values())


def test_every_family_on_the_menu_states_its_kind(qtbot):
    """A name with no stated assumption is the menu this replaced."""
    combo = _menu(qtbot)

    for index in range(1, combo.count()):
        value = combo.itemData(index)
        label = combo.itemText(index)
        assert label.startswith(f"{value} "), label
        assert GROUP_TITLES[family_group(value)] in label, label


def test_nothing_merely_robust_is_called_nonparametric(qtbot):
    """`rlm`, `huber` and `quantile` fit a linear model.

    They are parametric in the coefficients and robust in the loss. Only
    ``rra`` reads nothing but the order of the wells, so it is the only entry
    described as nonparametric. A semiparametric method may explicitly say
    that it is *not fully* nonparametric without being misclassified.
    """
    combo = _menu(qtbot)

    for index in range(1, combo.count()):
        value = combo.itemData(index)
        label = combo.itemText(index).lower()
        if ("nonparametric" in label and
                "not fully nonparametric" not in label):
            assert value == "rra", label


def test_the_menu_does_not_reach_for_the_fitting_module():
    """Building the menu must not import spacr.ml.

    ``spacr.ml`` imports ``spacr.plot`` and therefore torch: 2.2 seconds and
    900 MB on the GUI thread to read a tuple of strings. The family table was
    split out of it for exactly this reason, so the menu asks
    ``spacr.regression_families``.
    """
    import inspect

    menu = inspect.getsource(sm._regression_type_menu)
    branch = inspect.getsource(sm.SettingsWidgets._widget_for)
    branch = branch.split('if key == "regression_type":', 1)[1]
    branch = branch.split("elif key ==", 1)[0]

    assert "from spacr.ml import" not in menu, menu
    assert "from spacr.regression_families import" in menu
    assert "_regression_type_menu()" in branch, branch


def test_every_caption_the_menu_shows_is_offered_to_the_translators(qtbot):
    """A caption on the dropdown is a caption in the catalog builder's sources.

    These captions are composed at runtime, so the literal-string extractor
    in ``tools/build_i18n_catalogs.py`` cannot see them at the ``addItem``
    call site. ``_REGRESSION_MENU_UI_SOURCES`` is how they are declared
    instead, and a caption the menu shows that the set omits is the one
    English line left in a Swedish panel -- or, if it is short enough to fall
    under the term matcher's ceiling, half-English instead.
    """
    combo = _menu(qtbot)

    shown = {combo.itemText(i) for i in range(combo.count())}

    assert shown == set(sm._REGRESSION_MENU_UI_SOURCES), (
        sorted(shown ^ set(sm._REGRESSION_MENU_UI_SOURCES)))


def test_the_menu_can_still_be_set_by_the_value_it_stores(qtbot):
    """``setCurrentText('ols')`` selects ols even though no caption says 'ols'.

    Qt matches ``setCurrentText`` against the CAPTION and, on a non-editable
    combo, does nothing at all when nothing matches -- no exception, no log.
    Every caller that says "choose this family" by name would have become a
    silent no-op the moment the caption stopped being the value, leaving the
    control on whatever it was showing while the caller believed it was set.
    """
    widgets = sm.SettingsWidgets("regression")
    combo = widgets._widget_for("entry", None, "mixed", "regression_type")
    qtbot.addWidget(combo)

    combo.setCurrentText("ols")

    assert widgets._read_widget(combo) == "ols"


def test_a_caption_still_wins_over_a_stored_value(qtbot):
    """Qt's own behaviour is unchanged; the stored value is only a fallback.

    A combo whose captions ARE its values -- which is most of them -- has to
    keep behaving exactly as QComboBox does, or this fallback would be a
    second way for two entries to compete for one name.
    """
    combo = sm._ValueCombo()
    qtbot.addWidget(combo)
    combo.addItem("first", userData="second")
    combo.addItem("second", userData="third")

    combo.setCurrentText("second")

    assert combo.currentIndex() == 1


def test_an_unknown_name_leaves_the_selection_where_it_was(qtbot):
    """No match is not a reason to move the control.

    Silently selecting index 0 would substitute a family the caller never
    asked for, which is worse than the no-op it replaced.
    """
    combo = sm._ValueCombo()
    qtbot.addWidget(combo)
    combo.addItem("alpha", userData="a")
    combo.addItem("beta", userData="b")
    combo.setCurrentIndex(1)

    combo.setCurrentText("gamma")

    assert combo.currentIndex() == 1
