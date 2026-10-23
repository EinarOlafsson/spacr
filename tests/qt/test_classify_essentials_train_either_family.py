"""Classify's Essentials view is enough to pick and train a model.

Asked for on 2026-09-28: "for classify the essentialls category only includes
the classifier category. this is not enough. essentialls should include
enough to pick and train a computer vision mode (resnet or Maxvit, etc) or a
tabular based algorithm".

The Essentials view has to carry both families' training settings, and the
family switch hides the half that does not apply without losing its values.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from spacr.qt.screens.settings_model import essential_keys  # noqa: E402
from spacr.qt.settings_search import (  # noqa: E402
    ESSENTIALS,
    forget_disclosure,
    install,
)

SHARED = ("classifier_family", "src", "dataset_mode", "classes",
          "test_split", "learning_rate")
#: Shown or hidden by ``dataset_mode``, the class definition's own switch.
BY_THE_MODE = ("metadata_item_1_name", "metadata_item_1_value")
IMAGE = ("model_type", "epochs", "batch_size", "val_split",
         "train_channels", "image_size", "mixed_precision",
         "generate_training_dataset", "train")
TABULAR = ("model_type_ml", "n_estimators", "reg_alpha", "reg_lambda")


@pytest.fixture
def classify_screen(qtbot):
    forget_disclosure()
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("classify_merged")
    qtbot.addWidget(screen)
    yield screen
    forget_disclosure()


def test_essentials_name_both_families_training_settings():
    keys = set(essential_keys("classify_merged"))
    missing = [k for k in (*SHARED, *BY_THE_MODE, *IMAGE, *TABULAR)
               if k not in keys]
    assert not missing, missing


def test_essentials_leave_the_advanced_settings_in_their_categories():
    keys = set(essential_keys("classify_merged"))
    for advanced in ("focal_gamma", "tta_enabled", "batch_correction",
                     "leakage_hash_content", "prune_features"):
        assert advanced not in keys


def test_the_essentials_view_shows_them(classify_screen):
    bar = install(classify_screen)
    assert bar.level() == ESSENTIALS
    shown = set(bar.visible_keys())
    missing = [k for k in (*SHARED, *IMAGE) if k not in shown]
    assert not missing, missing
    assert not set(TABULAR) & shown
    assert len(shown) < len(bar.indexed_keys())


@pytest.mark.parametrize("family, live, greyed", [
    ("cv", IMAGE, TABULAR),
    ("ml", TABULAR, IMAGE),
])
def test_only_the_chosen_family_is_live(classify_screen, family, live,
                                        greyed):
    install(classify_screen)
    panel = classify_screen._settings_model
    control = panel._widgets.get("classifier_family")
    control.setCurrentIndex(control.findData(family))
    panel._on_classifier_family_changed()
    built = dict(panel._built_controls())
    for key in live:
        if key in built:
            assert built[key].isEnabled(), (family, key)
    for key in greyed:
        if key in built:
            assert not built[key].isEnabled(), (family, key)
    for key in SHARED[1:]:
        if key in built:
            assert built[key].isEnabled(), (family, key)
    assert set(live) & set(built) and set(greyed) & set(built)


@pytest.mark.parametrize("family", ["cv", "ml"])
def test_family_hides_every_opposite_row_and_category(classify_screen, qtbot, family):
    from spacr.classify import FAMILY_SETTINGS
    from spacr.qt.settings_search import ALL

    screen = classify_screen
    bar = install(screen)
    bar.set_level(ALL)
    screen.show()
    model = screen._settings_model
    selector = model._widgets["classifier_family"]
    selector.setCurrentIndex(selector.findData(family))
    qtbot.wait(0)
    other = "ml" if family == "cv" else "cv"
    shown = set(bar.visible_keys())
    assert not shown.intersection(FAMILY_SETTINGS[other])
    assert {"classifier_family", "learning_rate", "channel_of_interest", "tables", "src"} <= shown
    assert ("model_type" if family == "cv" else "model_type_ml") in shown
    assert "dataset" not in shown
    assert ("model_path" in shown) == (family == "cv")
    prefix = "Machine Learning" if family == "cv" else "Computer Vision"
    opposite = [section for section in screen.rendered_settings_sections()
                if str(section.property("settingsCategorySource") or "").startswith(prefix)]
    assert opposite
    assert all(section.isHidden() for section in opposite)


def test_switching_families_keeps_values_filters_and_navigation(classify_screen, qtbot):
    from spacr.classify import FAMILY_SETTINGS
    from spacr.qt.settings_search import ALL

    screen = classify_screen
    bar = install(screen)
    model = screen._settings_model
    selector = model._widgets["classifier_family"]
    assert model.set_value_for_key("epochs", 17)
    assert model.set_value_for_key("n_estimators", 37)
    assert model.set_hidden_value("dataset", "/saved/crops.tar")
    assert model.set_value_for_key("model_path", "/saved/model.pth")
    bar.set_level(ALL)
    screen.show()
    for family in ("ml", "cv", "ml", "cv"):
        selector.setCurrentIndex(selector.findData(family))
        qtbot.wait(0)
        other = "ml" if family == "cv" else "cv"
        inactive = "epochs" if family == "ml" else "n_estimators"
        assert model._widgets["classifier_family"] is selector
        assert model.collect()["epochs"] == 17
        assert model.collect()["n_estimators"] == 37
        assert model.collect()["dataset"] == "/saved/crops.tar"
        assert model.collect()["model_path"] == "/saved/model.pth"
        bar.set_query(inactive)
        assert inactive not in bar.visible_keys()
        assert not set(bar.visible_keys()).intersection(FAMILY_SETTINGS[other])
        assert not bar.reveal(inactive)
        assert bar.query() == inactive
        bar.set_query("")
        bar.set_modified_only(True)
        assert not set(bar.visible_keys()).intersection(FAMILY_SETTINGS[other])
        bar.set_modified_only(False)
        screen._open_every_waiting_heading()
        qtbot.wait(0)
        assert not set(bar.visible_keys()).intersection(FAMILY_SETTINGS[other])
        assert model.collect()["epochs"] == 17
        assert model.collect()["n_estimators"] == 37


def test_measurement_tables_remain_shared_and_preserved(classify_screen, qtbot):
    from spacr.classify import FAMILY_SETTINGS
    from spacr.qt.settings_search import ALL

    screen = classify_screen
    bar = install(screen)
    bar.set_level(ALL)
    model = screen._settings_model
    screen.show()
    selected = ["cell", "nucleus"]
    assert model.set_value_for_key("tables", selected)
    selector = model._widgets["classifier_family"]
    assert all("tables" not in keys for keys in FAMILY_SETTINGS.values())
    for family in ("ml", "cv", "ml"):
        selector.setCurrentIndex(selector.findData(family))
        qtbot.wait(0)
        assert "tables" in bar.visible_keys()
        assert model.collect()["tables"] == selected
        assert model._widgets["tables"].isEnabled()
