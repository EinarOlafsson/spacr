"""Rebuilding a settings panel must not silently drop the keys it was given.

`gui_core` had TWO if/elif chains on ``settings_type``: one in
`setup_settings_panel`, which built the Tk panel, and one in
`import_settings`, which rebuilt it from a CSV. They were maintained by hand
and had drifted.

`update_settings_from_csv` kept only keys the factory produced::

    for key, value in csv_settings.items():
        if key in new_settings:          # CSV keys the factory lacks are dropped

so when `import_settings` used a SMALLER factory than the panel, importing a
CSV silently deleted every key outside it. For 'classify' that was measured
at 80 widgets before the import and 46 after -- with `apply_model_to_dataset`,
`model_path`, `generate_training_dataset` and `score_threshold` among the 34
casualties, which are exactly the keys somebody saves a settings CSV to
preserve.

TWO TESTS ARE GONE WITH THAT INTERFACE. They parsed the two chains out of
`legacy_tk/gui_core.py` and required them to name the same factory; the file
is deleted and the shape cannot recur, because the Qt panel is built once
from `resolve_default_settings` and a CSV is applied ONTO it
(`AppScreen._on_import_settings` -> `apply_settings_dict`) rather than
rebuilt from a second factory. The panel-key arithmetic survives: it is what
made the deleted tests more than symbolic, and it is what says which of the
two `classify` factories the panel has to be built from.
"""

import pytest


def test_classify_really_does_lose_keys_under_the_old_factory():
    """The arithmetic that made the deleted pair more than symbolic."""
    import spacr.settings as S

    panel_keys = set(S.deep_spacr_defaults(settings={}))
    old_keys = set(S.set_default_train_test_model(settings={}))

    assert len(panel_keys) > len(old_keys)
    lost = panel_keys - old_keys
    assert len(lost) >= 30
    for key in ("apply_model_to_dataset", "model_path",
                "generate_training_dataset", "score_threshold"):
        assert key in lost, f"{key} was expected among the dropped keys"


def test_the_classify_panel_is_built_from_the_larger_factory():
    """The half of the old pair that still has a subject.

    `resolve_default_settings` is now the only factory the classify panel is
    built from, so it is the one that decides which keys an imported CSV can
    keep. Built from the smaller factory, the 37 keys counted above would be
    dropped from every CSV on import exactly as they were under Tk.

    The panel is a Qt widget module, so this half of the pair needs
    PySide6 to answer at all. A core-only install has no settings panel
    to guard, and must skip here rather than report a missing GUI extra
    as a settings key the panel drops.
    """
    pytest.importorskip("PySide6")
    from spacr.qt.screens.settings_model import resolve_default_settings
    import spacr.settings as S

    shown = set(resolve_default_settings("classify"))
    panel_keys = set(S.deep_spacr_defaults(settings={}))
    old_keys = set(S.set_default_train_test_model(settings={}))

    assert panel_keys <= shown, (
        f"the classify panel no longer offers "
        f"{sorted(panel_keys - shown)}; a CSV carrying them loses them")
    assert panel_keys - old_keys, "the two factories no longer differ"
