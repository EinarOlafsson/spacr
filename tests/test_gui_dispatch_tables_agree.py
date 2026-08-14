"""Importing a settings CSV must rebuild the panel the same way it was built.

`gui_core` has TWO if/elif chains on ``settings_type``: one in
`setup_settings_panel`, which builds the Tk panel, and one in
`import_settings`, which rebuilds it from a CSV. They are maintained by hand
and had drifted.

`update_settings_from_csv` keeps only keys the factory produced::

    for key, value in csv_settings.items():
        if key in new_settings:          # CSV keys the factory lacks are dropped

so when `import_settings` uses a SMALLER factory than the panel, importing a
CSV silently deletes every key outside it. For 'classify' that was measured
at 80 widgets before the import and 46 after -- with `apply_model_to_dataset`,
`model_path`, `generate_training_dataset` and `score_threshold` among the 34
casualties, which are exactly the keys somebody saves a settings CSV to
preserve.

The panel is the authority: it is what the user configured and what Run
reads. This test derives both chains from the source, so the pair cannot
drift again without saying so.
"""

import ast
import pathlib

import pytest

SOURCE = pathlib.Path(
    __import__("spacr").__file__).parent / "gui_core.py"
TREE = ast.parse(SOURCE.read_text(encoding="utf-8"))


def _dispatch(function_name):
    """``{settings_type: source of the assigned expression}`` for one chain."""
    found = {}
    for node in ast.walk(TREE):
        if not (isinstance(node, ast.FunctionDef)
                and node.name == function_name):
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.If):
                continue
            test = inner.test
            if not (isinstance(test, ast.Compare)
                    and isinstance(test.left, ast.Name)
                    and test.left.id == "settings_type"
                    and test.comparators
                    and isinstance(test.comparators[0], ast.Constant)):
                continue
            for stmt in inner.body:
                if isinstance(stmt, ast.Assign):
                    found[test.comparators[0].value] = ast.unparse(stmt.value)
    return found


IMPORTER = _dispatch("import_settings")
PANEL = _dispatch("setup_settings_panel")
SHARED = sorted(set(IMPORTER) & set(PANEL))


def test_the_scan_found_both_chains():
    """A scan that matched nothing would pass every test below."""
    assert len(PANEL) >= 8, f"only found {len(PANEL)} panel branches"
    assert len(IMPORTER) >= 8, f"only found {len(IMPORTER)} import branches"
    assert "classify" in SHARED, "the branch this test was written for is gone"


@pytest.mark.parametrize("settings_type", SHARED)
def test_the_importer_uses_the_panel_s_own_factory(settings_type):
    """Anything else drops the keys the smaller factory does not know."""
    assert IMPORTER[settings_type] == PANEL[settings_type], (
        f"importing a {settings_type!r} CSV rebuilds the panel from "
        f"{IMPORTER[settings_type]} while the panel was built from "
        f"{PANEL[settings_type]}; every key the first lacks is silently "
        f"dropped on import")


def test_classify_really_does_lose_keys_under_the_old_factory():
    """The arithmetic, so the parametrized test above is not just symbolic."""
    import spacr.settings as S

    panel_keys = set(S.deep_spacr_defaults(settings={}))
    old_keys = set(S.set_default_train_test_model(settings={}))

    assert len(panel_keys) > len(old_keys)
    lost = panel_keys - old_keys
    assert len(lost) >= 30
    for key in ("apply_model_to_dataset", "model_path",
                "generate_training_dataset", "score_threshold"):
        assert key in lost, f"{key} was expected among the dropped keys"
