"""Recipes — a named settings bundle survives a round trip and says where
it came from.

The two things a recipe has to get right are the two a settings CSV got
wrong: it has to come back exactly as it went in, and it has to say which
spaCR wrote it so applying one from two releases ago is a decision rather
than a surprise.
"""
from __future__ import annotations

import json

import pytest

from spacr.qt.recipes import (
    FORMAT_VERSION,
    Recipe,
    apply_recipe,
    capture_recipe,
    compatibility_note,
    delete_recipe,
    list_recipes,
    load_recipe,
    recipes_dir,
    save_recipe,
    spacr_version,
    version_note,
)


@pytest.fixture(autouse=True)
def _isolated_store(tmp_path, monkeypatch):
    """Never write into the user's real ``~/.spacr/recipes``."""
    monkeypatch.setenv("SPACR_RECIPE_DIR", str(tmp_path / "recipes"))
    yield


@pytest.fixture
def mask_screen(qtbot):
    from spacr.qt.screens.app_screen import AppScreen
    screen = AppScreen("mask")
    qtbot.addWidget(screen)
    return screen


# ---------------------------------------------------------------------------
# 1. Round trip
# ---------------------------------------------------------------------------

def test_a_recipe_round_trips_through_a_file(mask_screen):
    """Captured, written, read back, applied — and the values are the ones
    that went in."""
    model = mask_screen._settings_model
    model.set_value_for_key("n_jobs", 7)
    model.set_value_for_key("cell_channel", 2)

    recipe = capture_recipe(mask_screen, "Toxo PVM, 40×")
    path = save_recipe(recipe)

    model.set_value_for_key("n_jobs", 1)
    model.set_value_for_key("cell_channel", 0)
    assert model.collect()["n_jobs"] == 1

    reloaded = load_recipe(path)
    assert reloaded.name == "Toxo PVM, 40×"
    assert reloaded.app_key == "mask"
    apply_recipe(reloaded, mask_screen)

    collected = model.collect()
    assert collected["n_jobs"] == 7
    assert collected["cell_channel"] == 2


def test_a_saved_recipe_is_listed_for_its_module(mask_screen):
    save_recipe(capture_recipe(mask_screen, "Plate A"))
    names = [r.name for r in list_recipes("mask")]
    assert "Plate A" in names
    assert [r.name for r in list_recipes("measure")] == []


def test_a_recipe_can_be_deleted(mask_screen):
    recipe = capture_recipe(mask_screen, "Throwaway")
    save_recipe(recipe)
    assert delete_recipe(recipe)
    assert [r.name for r in list_recipes("mask")] == []


def test_a_name_with_punctuation_still_has_a_usable_filename(mask_screen):
    """The display name lives inside the file, so the stem only has to be
    typeable — it is never what the user reads."""
    recipe = capture_recipe(mask_screen, "Toxo PVM, 40× / v2")
    path = save_recipe(recipe)
    assert path.endswith(".json")
    assert load_recipe(path).name == "Toxo PVM, 40× / v2"


def test_capture_records_every_setting_not_only_the_edits(mask_screen):
    """A bundle of only the edits reproduces something different the day a
    default changes; a recipe is meant to reproduce a result."""
    recipe = capture_recipe(mask_screen, "Everything")
    assert len(recipe.settings) == len(mask_screen._settings_model.collect())


# ---------------------------------------------------------------------------
# 2. The version stamp
# ---------------------------------------------------------------------------

def test_a_recipe_records_the_version_that_made_it(mask_screen):
    recipe = capture_recipe(mask_screen, "Stamped")
    save_recipe(recipe)
    assert recipe.spacr_version == spacr_version()
    assert recipe.created


def test_the_same_version_says_nothing(mask_screen):
    recipe = capture_recipe(mask_screen, "Same")
    save_recipe(recipe)
    assert version_note(recipe) == ""


def test_a_different_version_says_both_numbers(mask_screen):
    """A bare warning says something might be wrong and leaves the user no
    way to judge it. The sentence has to carry both versions."""
    recipe = capture_recipe(mask_screen, "Old")
    recipe.spacr_version = "1.3.0"
    note = version_note(recipe, current="1.3.6")
    assert note
    assert "1.3.0" in note and "1.3.6" in note


def test_settings_that_no_longer_exist_are_named(mask_screen):
    """The version gap says *that* something moved; this says *what*."""
    recipe = capture_recipe(mask_screen, "Stale")
    recipe.settings["a_setting_that_was_removed"] = 1
    note = compatibility_note(recipe, mask_screen._settings_model)
    assert "a_setting_that_was_removed" in note
    assert "ignored" in note


def test_a_recipe_with_no_stale_keys_reports_no_gap(mask_screen):
    recipe = capture_recipe(mask_screen, "Current")
    assert compatibility_note(recipe, mask_screen._settings_model) == ""


def test_many_stale_keys_are_summarised_rather_than_listed(mask_screen):
    recipe = capture_recipe(mask_screen, "Ancient")
    for i in range(12):
        recipe.settings[f"gone_{i}"] = i
    note = compatibility_note(recipe, mask_screen._settings_model)
    assert "12 setting" in note
    assert "and 8 more" in note


# ---------------------------------------------------------------------------
# 3. Refusals
# ---------------------------------------------------------------------------

def test_a_recipe_for_another_module_is_refused(mask_screen, qtbot):
    """The keys two modules share are the least meaningful ones — `src`,
    `verbose`, `n_jobs` — so a partial apply would write nothing that
    matters and report success."""
    from spacr.qt.screens.app_screen import AppScreen
    measure = AppScreen("measure")
    qtbot.addWidget(measure)
    recipe = capture_recipe(measure, "Measure setup")
    with pytest.raises(ValueError, match="measure"):
        apply_recipe(recipe, mask_screen)


def test_an_arbitrary_json_file_is_not_a_recipe(tmp_path):
    path = tmp_path / "package.json"
    path.write_text('{"name": "something", "version": "1.0.0"}')
    with pytest.raises(ValueError, match="not a spaCR settings recipe"):
        load_recipe(str(path))


def test_a_newer_format_is_refused_with_both_numbers(tmp_path):
    path = tmp_path / "future.json"
    path.write_text(json.dumps({
        "spacr_recipe": FORMAT_VERSION + 5,
        "name": "From the future",
        "app_key": "mask",
        "settings": {"n_jobs": 1},
    }))
    with pytest.raises(ValueError, match="newer than this spaCR"):
        load_recipe(str(path))


def test_one_corrupt_file_does_not_hide_the_rest(mask_screen):
    save_recipe(capture_recipe(mask_screen, "Good"))
    broken = recipes_dir("mask") + "/broken.json"
    with open(broken, "w", encoding="utf-8") as handle:
        handle.write("{not json")
    assert [r.name for r in list_recipes("mask")] == ["Good"]


# ---------------------------------------------------------------------------
# 4. The dialog
# ---------------------------------------------------------------------------

def test_the_dialog_lists_and_describes_a_recipe(mask_screen, qtbot):
    from spacr.qt.recipes import RecipeDialog
    save_recipe(capture_recipe(mask_screen, "Plate A"))

    dialog = RecipeDialog(mask_screen)
    qtbot.addWidget(dialog)
    assert [r.name for r in dialog.recipes()] == ["Plate A"]
    assert dialog.selected().name == "Plate A"
    assert "settings, saved" in dialog.detail_text()


def test_the_dialog_says_so_when_a_listed_recipe_is_from_another_version(
        mask_screen, qtbot):
    from spacr.qt.recipes import RecipeDialog
    recipe = capture_recipe(mask_screen, "Old one")
    recipe.spacr_version = "0.0.1"
    save_recipe(recipe)

    dialog = RecipeDialog(mask_screen)
    qtbot.addWidget(dialog)
    assert "0.0.1" in dialog.detail_text()


def test_an_empty_folder_tells_the_user_what_to_do(mask_screen, qtbot):
    from spacr.qt.recipes import RecipeDialog
    dialog = RecipeDialog(mask_screen)
    qtbot.addWidget(dialog)
    assert dialog.selected() is None
    assert "No recipes yet" in dialog.detail_text()
    assert not dialog._btn_apply.isEnabled()


def test_the_recipes_button_lands_in_the_settings_strip(mask_screen, qtbot):
    from spacr.qt.recipes import install
    from spacr.qt.settings_search import install as install_search
    bar = install_search(mask_screen)
    button = install(mask_screen)
    assert button is not None
    assert bar.isAncestorOf(button)
    assert install(mask_screen) is button


def test_the_button_is_skipped_when_there_is_no_strip(mask_screen):
    from spacr.qt.recipes import install
    assert install(mask_screen) is None


# ---------------------------------------------------------------------------
# 5. Sharing
# ---------------------------------------------------------------------------

def test_an_exported_file_is_readable_by_a_fresh_process(mask_screen,
                                                          tmp_path):
    """Sharing a recipe is sending a file, so the file has to stand alone."""
    recipe = capture_recipe(mask_screen, "Shared")
    shared = tmp_path / "shared.json"
    shared.write_text(json.dumps(recipe.to_json(), indent=2, default=str))

    reloaded = load_recipe(str(shared))
    assert reloaded.name == "Shared"
    assert reloaded.app_key == "mask"
    assert reloaded.settings == recipe.settings
    assert reloaded.spacr_version == spacr_version()


def test_the_file_carries_a_format_version_of_its_own(mask_screen):
    """The format version and the spaCR version change for unrelated
    reasons; conflating them is how a reader refuses a file it could
    read."""
    data = capture_recipe(mask_screen, "Versioned").to_json()
    assert data["spacr_recipe"] == FORMAT_VERSION
    assert data["spacr_version"] == spacr_version()
    assert data["spacr_recipe"] != data["spacr_version"]


def test_a_recipe_built_by_hand_still_applies(mask_screen):
    recipe = Recipe(name="Hand written", app_key="mask",
                    settings={"n_jobs": 5}, spacr_version="1.3.6")
    assert apply_recipe(recipe, mask_screen) >= 1
    assert mask_screen._settings_model.collect()["n_jobs"] == 5
