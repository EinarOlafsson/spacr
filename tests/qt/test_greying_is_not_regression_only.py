"""A setting that cannot do anything is greyed out in EVERY module, not one.

Instruction 106, point 6. The rules in ``settings.setting_dependencies`` are
keyed by setting NAME and say nothing about which screen a setting appears
on -- ``batch_column`` is dead when ``batch_correction`` is ``'none'``
wherever the two are shown together. Three panels other than the regression
one show exactly that pair: Image UMAP, Classify (merged) and ML Analyze.

``SettingsBuilder._refresh_setting_dependencies`` nevertheless opened with
``if self.app_key != 'regression': return``, so on those three screens all
seven ``batch_*`` controls stayed live and editable under the default
``batch_correction='none'``. The table was module-agnostic; the wiring was
not.

The guard is not replaced with a wider allow-list, because an allow-list is
the same bug with a longer line in it: the next module to gain a gated
setting would silently not gate. A rule applies where its setting and its
sources are both on screen, and nowhere else -- which is what
``_rules_for_this_panel`` decides.
"""

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import SettingsWidgets


#: Every module screen that shows a setting the dependency table has a rule
#: for, other than the regression panel the mechanism was built against.
PANELS_WITH_GATED_SETTINGS = ("umap", "classify_merged", "ml_analyze")

#: The seven controls `batch_correction='none'` makes dead.
BATCH_KEYS = (
    "batch_column", "batch_control_column", "batch_control_values",
    "batch_covariate_column", "batch_combat_mean_only", "batch_min_samples",
    "batch_missing_control",
)


def _panel(qtbot, app_key):
    panel = SettingsWidgets(app_key)
    panel.build_sections()
    for widget in panel._widgets.values():
        qtbot.addWidget(widget)
    return panel


def _set_combo(panel, key, value):
    from PySide6.QtWidgets import QComboBox
    widget = panel._widgets.get(key)
    assert widget is not None, f"{key!r} is not on this panel"
    assert isinstance(widget, QComboBox), (
        f"{key!r} is a {type(widget).__name__}, not a choice")
    for index in range(widget.count()):
        if widget.itemData(index) == value or widget.itemText(index) == str(value):
            widget.setCurrentIndex(index)
            return
    raise AssertionError(f"{key!r} offers no {value!r}")


@pytest.mark.parametrize("app_key", PANELS_WITH_GATED_SETTINGS)
class TestGreyingReachesEveryPanelThatShowsAGatedSetting:

    def test_batch_correction_off_greys_its_seven_settings(
            self, qtbot, qt_theme_applied, app_key):
        """The defect the guard caused, in the user's terms.

        With correction off, choosing a batch column, a control column or a
        minimum sample count changes nothing about the run. Seven live
        controls said otherwise on three screens.
        """
        panel = _panel(qtbot, app_key)
        _set_combo(panel, "batch_correction", "none")
        panel._refresh_setting_dependencies()

        live = [key for key in BATCH_KEYS
                if key in panel._widgets and panel._widgets[key].isEnabled()]
        assert not live, (
            f"{app_key}: batch_correction is 'none' and these still invite an "
            f"edit that does nothing: {live}")

    def test_turning_correction_on_gives_them_back(
            self, qtbot, qt_theme_applied, app_key):
        """Disabling freezes a control; it must never strand one."""
        panel = _panel(qtbot, app_key)
        _set_combo(panel, "batch_correction", "none")
        panel._refresh_setting_dependencies()
        _set_combo(panel, "batch_correction", "zscore")
        panel._refresh_setting_dependencies()

        assert panel._widgets["batch_column"].isEnabled(), (
            f"{app_key}: batch_column is what zscore corrects on")
        assert panel._widgets["batch_min_samples"].isEnabled()

    def test_a_greyed_control_still_says_why(
            self, qtbot, qt_theme_applied, app_key):
        """A greyed control with no explanation is a dead end: the user
        cannot tell whether it is inapplicable or broken."""
        panel = _panel(qtbot, app_key)
        _set_combo(panel, "batch_correction", "none")
        panel._refresh_setting_dependencies()

        silent = [key for key in BATCH_KEYS
                  if key in panel._widgets
                  and not panel._widgets[key].isEnabled()
                  and key not in panel._widgets[key].toolTip()]
        assert not silent, (
            f"{app_key}: greyed with no reason naming the setting: {silent}")

    def test_a_setting_whose_sources_are_not_on_screen_is_left_alone(
            self, qtbot, qt_theme_applied, app_key):
        """The reason the guard existed, kept without the guard.

        A rule reads other settings. On a panel that shows the ruled setting
        but not the setting it depends on, the predicate would be evaluated
        against a DEFAULT the user cannot see or change -- and a control
        greyed by an invisible value is one nobody can ever re-enable. Such a
        rule must not fire at all.
        """
        panel = _panel(qtbot, app_key)
        rules = panel._rules_for_this_panel()
        for key, rule in rules.items():
            assert key in panel._widgets, (
                f"{app_key}: rule for {key!r}, which this panel does not show")
            assert any(source in panel._widgets
                       for source in rule["sources"]), (
                f"{app_key}: {key!r} would be gated on "
                f"{rule['sources']}, none of which is on this screen")


class TestTheRegressionPanelIsUnchangedByTheWidening:
    """The panel the mechanism was built against must not regress."""

    def test_the_estimator_rules_still_fire(self, qtbot, qt_theme_applied):
        from spacr.ml import REGRESSION_SETTINGS_USED

        panel = _panel(qtbot, "regression")
        # A PARAMETRIC INFERENCE FIRST, so the FAMILY rule is what is being
        # measured. `inference` defaults to 'nonparametric' (2026-08-19, at
        # the maintainer's direction), and that path fits no model at all --
        # so every estimator setting is greyed for a reason that has nothing
        # to do with which family was chosen, and the two rules cannot be
        # told apart.
        _set_combo(panel, "inference", "parametric")
        _set_combo(panel, "regression_type", "ols")
        panel._refresh_setting_dependencies()

        owned = {key for keys in REGRESSION_SETTINGS_USED.values()
                 for key in keys}
        expected = {key for key in REGRESSION_SETTINGS_USED["ols"]
                    if key in panel._widgets}
        enabled = {key for key in owned
                   if key in panel._widgets and panel._widgets[key].isEnabled()}
        assert enabled == expected

    def test_mask_greys_its_one_ruled_setting_and_nothing_else(
            self, qtbot, qt_theme_applied):
        """Mask has exactly one gated setting, and the sweep touches only it.

        THIS TEST USED TO ASSERT MASK HAD NONE, and that was true when it was
        written. `custom_regex` gained a rule -- it is read only when
        `metadata_type` is 'custom', or 'auto' with a regex supplied -- so the
        premise went stale rather than the behaviour going wrong.

        Rewritten rather than repointed at some other ruleless panel, because
        the property worth guarding is not "a panel with no rules greys
        nothing". It is that the sweep greys THE RULED SETTING AND NOTHING
        ELSE: a rule that reached past its own key would disable controls a
        user needs, and would look like the panel had broken.
        """
        panel = _panel(qtbot, "mask")

        assert set(panel._rules_for_this_panel()) == {"custom_regex"}

        panel._refresh_setting_dependencies()
        greyed = {key for key, widget in panel._widgets.items()
                  if not widget.isEnabled()}

        # The default convention is 'cellvoyager', which does not read the
        # regex, so the one ruled setting is off and every other is live.
        assert panel.collect().get("metadata_type") == "cellvoyager"
        assert greyed == {"custom_regex"}

    def test_masks_rule_lets_the_regex_back_on_both_conventions_that_read_it(
            self, qtbot, qt_theme_applied):
        """'auto' counts, and that is the half a tidy-up would get wrong.

        The obvious reading of the rule is "grey it unless metadata_type is
        custom". `metadata_type`'s own description says 'auto' renames using
        `custom_regex` WHEN SUPPLIED, so a user on 'auto' with a regex is
        relying on documented behaviour. Greying it there would remove that
        while looking like a cleanup.
        """
        panel = _panel(qtbot, "mask")

        for convention in ("custom", "auto"):
            panel.set_value_for_key("metadata_type", convention)
            panel._refresh_setting_dependencies()
            assert panel._widgets["custom_regex"].isEnabled(), (
                f"the regex is greyed under {convention!r}, which reads it")

        panel.set_value_for_key("metadata_type", "cq1")
        panel._refresh_setting_dependencies()
        assert not panel._widgets["custom_regex"].isEnabled()
