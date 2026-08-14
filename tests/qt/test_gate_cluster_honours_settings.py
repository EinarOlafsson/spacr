"""The Cluster button must use the clustering settings the user set.

Gate Settings has offered ``cluster_eps``, ``cluster_min_samples`` and
``cluster_scale`` for as long as the cluster dialog has existed, and the
dialog opened on its own hardcoded 0.30/10 regardless. So the three were
phantom settings: editable, saved, reloaded, and read by nothing.

That is worse than a missing feature. eps is the parameter that decides how
many populations DBSCAN finds, so a user who set it to 0.5 and got a
clustering computed at 0.30 got a DIFFERENT ANSWER than the one they asked
for, with no indication anywhere that their value had been dropped.

These tests construct the dialog directly rather than driving the button,
because `exec()` on a modal dialog never returns under the offscreen
platform.
"""

import pytest

pytest.importorskip("PySide6")

from spacr.qt.widgets.gate_settings import GateEditorSettings  # noqa: E402


@pytest.fixture
def dialog_cls():
    from spacr.qt.widgets.gate_editor import _ClusterSettingsDialog
    return _ClusterSettingsDialog


def test_the_dialog_opens_on_the_saved_values(qtbot, dialog_cls):
    """The regression: these three were set and then ignored."""
    settings = GateEditorSettings(cluster_eps=1.75, cluster_min_samples=42,
                                  cluster_scale=False)
    dialog = dialog_cls(settings=settings)
    qtbot.addWidget(dialog)

    assert dialog.eps() == pytest.approx(1.75)
    assert dialog.min_samples() == 42
    assert dialog.scale() is False


def test_without_settings_it_falls_back_to_the_dataclass_default(qtbot,
                                                                 dialog_cls):
    """One place decides each default, and it is not this dialog.

    The dialog is constructible before `apply_settings` has run, so `None`
    has to work -- but it must land on the SAME numbers Gate Settings shows,
    which is exactly what the hardcoded 0.30/10 did not do.
    """
    dialog = dialog_cls(settings=None)
    qtbot.addWidget(dialog)

    default = GateEditorSettings()
    assert dialog.eps() == pytest.approx(default.cluster_eps)
    assert dialog.min_samples() == default.cluster_min_samples
    assert dialog.scale() is default.cluster_scale


def test_a_settings_object_missing_a_field_does_not_raise(qtbot, dialog_cls):
    """A settings set saved before a field existed still opens the dialog.

    Reloading an old strategy should not be the thing that breaks
    clustering, so the read falls back per-field rather than per-object.
    """
    class Older:
        cluster_eps = 0.8  # and no min_samples or scale at all

    dialog = dialog_cls(settings=Older())
    qtbot.addWidget(dialog)

    default = GateEditorSettings()
    assert dialog.eps() == pytest.approx(0.8)
    assert dialog.min_samples() == default.cluster_min_samples


def test_the_panel_keeps_the_settings_the_cluster_button_needs(qtbot):
    """`apply_settings` used to hand everything to the canvas and keep none.

    That is why the button had nothing to read. The canvas takes the drawing
    settings; the clustering ones have to stay on the panel, because the
    button that uses them is here.
    """
    from spacr.qt.widgets.gate_editor import GateEditorPanel

    panel = GateEditorPanel()
    qtbot.addWidget(panel)
    assert panel._settings is None

    settings = GateEditorSettings(cluster_eps=2.5, cluster_min_samples=7)
    panel.apply_settings(settings)
    assert panel._settings is settings


def test_the_walk_controls_are_seeded_too(qtbot, dialog_cls):
    """`cluster_walk` and `cluster_walk_steps` had ZERO readers in spaCR.

    Not merely ignored like eps -- nothing anywhere read them, so the two
    were editable, saved, reloaded and inert. Instruction 48 needs Walk to
    work before the Gate Editor lesson can show it, so they are wired rather
    than deleted.
    """
    settings = GateEditorSettings(cluster_walk=True, cluster_walk_steps=31)
    dialog = dialog_cls(settings=settings)
    qtbot.addWidget(dialog)

    assert dialog.walk() is True
    assert dialog.walk_steps() == 31


def test_walk_steps_is_disabled_until_the_walk_is_on(qtbot, dialog_cls):
    """A step count that cannot act is a control that lies about doing so."""
    dialog = dialog_cls(settings=GateEditorSettings(cluster_walk=False))
    qtbot.addWidget(dialog)
    assert dialog._walk_steps.isEnabled() is False

    dialog._walk.setChecked(True)
    assert dialog._walk_steps.isEnabled() is True
