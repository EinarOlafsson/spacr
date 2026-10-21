"""Contracts for readable, API-synchronized spaCR notebooks.

Notebook users do not have a graphical settings panel beside the analysis.
The generated Markdown therefore defines every exposed setting, while the
adjacent code cells organize editable values by scientific purpose and state
whether each value is required, conditionally required, or optional.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = REPO / "Notebooks"
TOOL = REPO / "tools" / "build_notebook_settings.py"


_TOOL_MODULE = None


def _tool():
    global _TOOL_MODULE
    if _TOOL_MODULE is not None:
        return _TOOL_MODULE
    spec = importlib.util.spec_from_file_location("_nb_settings", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_nb_settings"] = module
    spec.loader.exec_module(module)
    _TOOL_MODULE = module
    return _TOOL_MODULE


pytestmark = pytest.mark.skipif(
    not TOOL.exists() or not NOTEBOOKS.is_dir(),
    reason="run from a source checkout")

ALL = sorted(path for path in NOTEBOOKS.glob("*.ipynb")
             if path.name != "00_workflow_browser.ipynb") if NOTEBOOKS.is_dir() else []


def test_manifest_is_the_exact_maintained_notebook_inventory():
    tool = _tool()
    names = {path.name for path in ALL}

    assert len(tool.NOTEBOOK_SPECS) == 31
    assert set(tool.NOTEBOOK_SPECS) == names
    assert all(name == spec.filename
               for name, spec in tool.NOTEBOOK_SPECS.items())
    assert all(len(spec.callables) == len(spec.app_keys) >= 1
               for spec in tool.NOTEBOOK_SPECS.values())


def test_manifest_pins_the_four_consolidated_public_entry_points():
    tool = _tool()
    expected = {
        "04_classify_machine_learning.ipynb": "spacr.ml.generate_ml_scores",
        "09_apply_cellpose.ipynb": (
            "spacr.spacr_cellpose.identify_masks_finetune"),
        "24_interpret_vision_model.ipynb": "spacr.surrogate.run_explain_cv",
        "26_sequencing_stats.ipynb": "spacr.sequencing_qc.barcode_qc",
    }
    for name, dotted in expected.items():
        assert tool.NOTEBOOK_SPECS[name].callables == (dotted,)


def test_manifest_callables_resolve_and_desktop_routes_are_current():
    """Every lesson resolves and every route exists in the live Home registry."""
    import subprocess

    tool = _tool()
    unresolved = [
        dotted
        for spec in tool.NOTEBOOK_SPECS.values()
        for dotted in spec.callables
        if tool.resolve(dotted) is None
    ]
    assert not unresolved

    script = """
import json
from spacr.qt.app import APP_META
from spacr.qt import register_self_registering_modules
register_self_registering_modules()
print(json.dumps(sorted(APP_META)))
"""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SPACR_")}
    env.update({
        "PYTHONPATH": str(REPO),
        "QT_QPA_PLATFORM": "offscreen",
    })
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=str(REPO), env=env,
        capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    registered = set(json.loads(result.stdout.splitlines()[-1]))
    assert {spec.desktop_route for spec in tool.NOTEBOOK_SPECS.values()} <= (
        registered)


def test_stale_cell_metadata_cannot_change_the_manifest_callable():
    tool = _tool()
    notebook = json.loads(
        (NOTEBOOKS / "04_classify_machine_learning.ipynb").read_text(encoding="utf-8"))
    function_cell = next(
        cell for cell in notebook["cells"]
        if cell.get("metadata", {}).get("spacr", {}).get("generated")
        == "function-help")
    function_cell["metadata"]["spacr"]["functions"] = [
        "spacr.ml.removed_wrapper"]
    assert tool.declared_functions(notebook) == [
        "spacr.ml.generate_ml_scores"]


# ---------------------------------------------------------------------------
# The drift guard
# ---------------------------------------------------------------------------

def test_the_committed_notebooks_match_what_the_tool_generates():
    """Regenerate and compare. This is the whole instruction, enforced.

    If this fails, run::

        python tools/build_notebook_settings.py

    RUN AS A SUBPROCESS, and that is not incidental. spaCR's settings are
    assembled by IMPORT -- modules merge their keys and tooltips into the
    shared dicts from their own module bodies -- so both what a factory
    returns and which text a contested key carries depend on what the process
    has loaded. ``dst`` is registered by surrogate, hit_investigation AND
    sequencing_qc, and ``importlib.import_module`` is a no-op for a module
    already imported, so nothing can re-run a registration to make itself
    last.

    Generating INSIDE pytest therefore answered a different question from
    generating in the tool: the tool was idempotent -- a second run rewrote
    nothing -- while this test still called three notebooks stale. A guard
    that cannot be believed and cannot be silenced is worse than none.

    A subprocess asks the tool the same question a developer asks it.
    """
    import subprocess

    env = {key: value for key, value in os.environ.items()
           if not key.startswith("SPACR_")}
    env["PYTHONPATH"] = str(REPO)
    result = subprocess.run(
        [sys.executable, str(TOOL), "--check"],
        cwd=str(REPO), env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_shared_tooltips_do_not_depend_on_optional_module_imports():
    tool = _tool()
    expected = tool.CORE_TOOLTIPS["dst"]
    tool._load_registrations()
    assert tool.tooltip_for("dst", "factory", {}) == expected


def test_signature_docs_accept_scientific_python_parameter_sections():
    """Notebook comments follow the API writer's numpydoc convention."""
    tool = _tool()

    def example(first, second=True):
        """Example.

        Parameters
        ----------
        first : str
            Input table path.
        second : bool, optional
            Draw the result when true.

        Returns
        -------
        object
            The result.
        """

    assert tool.param_docs(example) == {
        "first": "Input table path.",
        "second": "Draw the result when true.",
    }


def test_every_notebook_has_a_settings_surface():
    """A notebook whose function nobody can enumerate is the defect itself."""
    tool = _tool()
    without = []
    for path in ALL:
        declared = tool.declared_functions(path)
        if not declared:
            without.append(f"{path.name}: names no function")
            continue
        if not any(tool.surface(tool.resolve(d), d)[1]
                   for d in declared if tool.resolve(d)):
            without.append(f"{path.name}: {declared} declare no settings")
    assert not without, without


# ---------------------------------------------------------------------------
# What the generated cell has to contain
# ---------------------------------------------------------------------------

def _generated(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        kind = cell.get("metadata", {}).get("spacr", {}).get("generated")
        if cell["cell_type"] == "code" and (
                kind == "settings" or source.startswith(
                    "# Generated by tools/build_notebook_settings.py")):
            return source
    return ""


def _generated_settings_cells(path):
    """Return all generated settings cells in execution order."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
        and cell.get("metadata", {}).get("spacr", {}).get("generated")
        in {"settings", "settings-advanced", "settings-organelle"}
    ]


def _generated_help(path):
    """The complete generated reference is deliberately outside initial cells."""
    return (NOTEBOOKS / "reference" / f"{path.stem}.md").read_text(encoding="utf-8")


def _generated_source(path, kind):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = [
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("metadata", {}).get("spacr", {}).get("generated") == kind
    ]
    assert len(matches) == 1, f"{path.name}: expected one {kind} cell"
    return matches[0]


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_the_generated_cell_is_valid_python(path):
    sources = _generated_settings_cells(path)
    if not sources:
        pytest.skip("no generated cell")
    for source in sources:
        tree = ast.parse(source)
        assert tree.body, "the generated cell has no statements"


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_manifest_owns_exact_imports_and_run_calls(path):
    tool = _tool()
    spec = tool.NOTEBOOK_SPECS[path.name]
    import_source = _generated_source(path, "function-import")
    run_source = _generated_source(path, "function-run")

    expected_imports = "\n".join(
        f"from {dotted.rpartition('.')[0]} import {dotted.rpartition('.')[2]}"
        for dotted in spec.callables)
    assert import_source == expected_imports

    expected_calls = []
    for dotted in spec.callables:
        shape = tool.surface(tool.resolve(dotted), dotted)[0]
        variable = ("settings" if len(spec.callables) == 1
                    else f"{dotted.rsplit('.', 1)[1]}_settings")
        spread = "**" if shape == "signature" else ""
        expected_calls.append(
            f"{dotted.rsplit('.', 1)[1]}({spread}{variable})")
    assert run_source == "\n".join(expected_calls)
    ast.parse(import_source)
    ast.parse(run_source)


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_all_outputs_and_execution_counts_are_cleared(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell.get("outputs") == []
            assert cell.get("execution_count") is None


def _explicit_settings_keys_by_variable(path):
    found = {}
    for source in _generated_settings_cells(path):
        for statement in ast.parse(source).body:
            if isinstance(statement, ast.Assign) and isinstance(
                    statement.value, ast.Dict):
                variable = statement.targets[0].id
                value = statement.value
            elif (isinstance(statement, ast.Expr)
                  and isinstance(statement.value, ast.Call)
                  and isinstance(statement.value.func, ast.Attribute)
                  and isinstance(statement.value.func.value, ast.Name)
                  and statement.value.func.attr == "update"
                  and statement.value.args
                  and isinstance(statement.value.args[0], ast.Dict)):
                variable = statement.value.func.value.id
                value = statement.value.args[0]
            else:
                continue
            found.setdefault(variable, []).extend(
                key.value for key in value.keys
                if isinstance(key, ast.Constant)
                and isinstance(key.value, str))
    return found


def _settings_keys_by_variable(path):
    namespace = {}
    for source in _generated_settings_cells(path):
        exec(source, namespace)
    entries = [(item, None, None, None) for item in _tool().declared_functions(path)]
    return {name: list(namespace[name]) for dotted, *_rest in entries
            for name in [_tool().setting_name(dotted, entries)]}


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_each_setting_is_declared_once_per_function(path):
    by_variable = _explicit_settings_keys_by_variable(path)
    assert by_variable
    for variable, keys in by_variable.items():
        duplicates = sorted(key for key, count in Counter(keys).items()
                            if count > 1)
        assert not duplicates, f"{path.name}:{variable}: {duplicates}"


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_organelle_separation_is_derived_from_the_actual_keys(path):
    tool = _tool()
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = {
        cell.get("metadata", {}).get("spacr", {}).get("generated"):
            "".join(cell["source"])
        for cell in notebook["cells"]
    }
    primary = cells["settings"]
    primary_names = set(re.findall(r"^\s*'([^']+)':", primary, re.M))
    assert not {key for key in primary_names if tool._is_organelle_key(key)}
    for names in _settings_keys_by_variable(path).values():
        for key in names:
            if tool._is_organelle_key(key):
                assert f"**`{key}`**" in _generated_help(path)



@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_every_key_carries_a_description(path):
    """Every code-cell key is explained in the adjacent Markdown cell."""
    sources = _generated_settings_cells(path)
    if not sources:
        pytest.skip("no generated cell")
    help_text = _generated_help(path)
    keys = []
    for source in sources:
        tree = ast.parse(source)
        for statement in tree.body:
            value = statement.value if isinstance(statement, ast.Assign) else None
            if (isinstance(statement, ast.Expr)
                    and isinstance(statement.value, ast.Call)
                    and statement.value.args):
                value = statement.value.args[0]
            if not isinstance(value, ast.Dict):
                continue
            keys.extend(key.value for key in value.keys
                        if isinstance(key, ast.Constant)
                        and isinstance(key.value, str))
    keys = [key for names in _settings_keys_by_variable(path).values() for key in names]
    undescribed = [key for key in keys if f"**`{key}`**" not in help_text]
    assert not undescribed, f"{path.name}: no description for {undescribed}"


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_explanations_settings_and_call_are_consecutive_cells(path):
    """Readers meet prose, categorized values, and execution in that order."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    help_index = next((i for i, cell in enumerate(cells)
                       if cell.get("metadata", {}).get("spacr", {}).get(
                           "generated") == "settings-help"), None)
    assert help_index is not None, f"{path.name}: no generated Markdown help"
    assert cells[help_index]["cell_type"] == "markdown"
    assert cells[help_index + 1]["cell_type"] == "code"
    assert cells[help_index + 1].get("metadata", {}).get("spacr", {}).get(
        "generated") == "settings"
    cursor = help_index + 1
    settings_cells = []
    while cursor < len(cells) and cells[cursor].get(
            "metadata", {}).get("spacr", {}).get("generated") in {
                "settings", "settings-advanced", "settings-organelle", "settings-editor"}:
        assert cells[cursor]["cell_type"] == "code"
        settings_cells.append(cells[cursor])
        cursor += 1
    assert settings_cells
    assert cells[cursor]["cell_type"] == "code", (
        f"{path.name}: function call does not follow its settings")
    assert cells[cursor].get("metadata", {}).get("spacr", {}).get(
        "generated") == "function-run"

    for settings_cell in settings_cells:
        if settings_cell["metadata"]["spacr"]["generated"] != "settings":
            continue
        source = "".join(settings_cell["source"])
        statuses = {
            "# Required settings",
            "# Conditionally required settings",
            "# Optional settings",
        }
        category_lines = [line for line in source.splitlines()
                          if line.startswith("    # ")
                          and line.strip() not in statuses]
        status_lines = [line for line in source.splitlines()
                        if line.strip() in statuses]
        assert category_lines, f"{path.name}: settings have no categories"
        assert status_lines, f"{path.name}: requirement status is absent"


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_each_reference_links_to_the_callable_api(path):
    tool = _tool()
    help_text = _generated_help(path)
    for dotted in tool.declared_functions(path):
        assert f"[`{dotted}`]({tool.api_url(dotted)})" in help_text


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_every_documented_setting_states_its_requirement_status(path):
    help_text = _generated_help(path)
    for line in help_text.splitlines():
        if line.startswith("- **`"):
            assert any(f"*({status})*" in line for status in (
                "required", "conditionally required", "optional")), (
                    f"{path.name}: missing requirement status: {line}")


def _documented_status(path, key):
    match = re.search(
        rf"^- \*\*`{re.escape(key)}`\*\* \*\(([^)]+)\)\*",
        _generated_help(path), re.M)
    assert match, f"{path.name}: {key} has no documented status"
    return match.group(1)


def test_cli_requirements_drive_the_structured_notebook_statuses():
    expected = {
        "04_classify_machine_learning.ipynb": {
            "src": "required",
            "positive_control_id": "conditionally required",
            "negative_control_id": "conditionally required",
            "annotation_column": "conditionally required",
        },
        "05_map_barcodes.ipynb": {
            "src": "required",
            "grna_csv": "required",
            "row_csv": "required",
            "column_csv": "required",
            "regex": "required",
        },
        "06_regression.ipynb": {
            "score_data": "required",
            "count_data": "required",
            "dependent_variable": "required",
        },
        "09_apply_cellpose.ipynb": {
            "src": "required",
            "model_name": "conditionally required",
            "custom_model": "conditionally required",
        },
        "11_activation_maps.ipynb": {
            "dataset": "required",
            "model_path": "required",
            "target_layer": "optional",
        },
        "24_interpret_vision_model.ipynb": {
            "db_path": "required",
            "predictions_file": "required",
        },
        "26_sequencing_stats.ipynb": {
            "count_data": "required",
            "target_grnas_per_well": "required",
        },
    }
    for name, statuses in expected.items():
        path = NOTEBOOKS / name
        assert {key: _documented_status(path, key) for key in statuses} == (
            statuses)


def _setting_categories(path):
    categories = {}
    current = ""
    for line in _generated_help(path).splitlines():
        if line.startswith("#### "):
            current = line[5:]
        match = re.match(r"^- \*\*`([^`]+)`", line)
        if match:
            categories[match.group(1)] = current
    return categories


def test_key_inputs_are_essential_and_advanced_groups_keep_curated_headings():
    expected = {
        "04_classify_machine_learning.ipynb": {
            "positive_control_id": "Essential inputs",
            "n_estimators": "Classifier & Validation",
        },
        "05_map_barcodes.ipynb": {
            "grna_csv": "Essential inputs",
            "regex": "Essential inputs",
        },
        "06_regression.ipynb": {
            "score_data": "Essential inputs",
            "count_data": "Essential inputs",
            "dependent_variable": "Essential inputs",
        },
        "09_apply_cellpose.ipynb": {
            "model_name": "Essential inputs",
            "CP_prob": "Detection Thresholds",
        },
        "11_activation_maps.ipynb": {
            "model_path": "Essential inputs",
            "target_layer": "Attribution Method",
        },
        "24_interpret_vision_model.ipynb": {
            "db_path": "Essential inputs",
            "surrogate_model": "Surrogate & validation",
        },
        "26_sequencing_stats.ipynb": {
            "count_data": "Essential inputs",
            "target_grnas_per_well": "Essential inputs",
        },
    }
    for name, headings in expected.items():
        categories = _setting_categories(NOTEBOOKS / name)
        assert {key: categories.get(key) for key in headings} == headings


def test_mask_workflows_are_separate_and_scientifically_described():
    expected = {
        "01_generate_masks.ipynb": [
            "spacr.core.preprocess_generate_masks",
        ],
        "01b_generate_timelapse_masks.ipynb": [
            "spacr.core.preprocess_generate_masks_timelapse",
        ],
        "14_motility_assay.ipynb": [
            "spacr.core.preprocess_generate_masks_timelapse",
            "spacr.timelapse.automated_motility_assay",
        ],
    }
    tool = _tool()
    for name, functions in expected.items():
        path = NOTEBOOKS / name
        assert tool.declared_functions(path) == functions

    mask = json.loads((NOTEBOOKS / "01_generate_masks.ipynb").read_text(encoding="utf-8"))
    overview = _generated_help(NOTEBOOKS / "01_generate_masks.ipynb")
    assert "Generate per-object masks" in overview
    assert all(object_name in overview for object_name in (
        "cells", "nuclei", "pathogens", "organelles"))
    assert "Turn raw" not in overview

    motility_cells = _generated_settings_cells(
        NOTEBOOKS / "14_motility_assay.ipynb")
    assert "'src': preprocess_generate_masks_timelapse_settings['src']" in (
        motility_cells[0])

    from spacr.settings import (
        categories,
        motility_advanced_settings,
        motility_settings,
        timelapse_settings,
    )
    timelapse_keys = (set(timelapse_settings)
                      | set(categories["4D Settings (Beta)"])
                      | {"timelapse"})
    motility_keys = (set(motility_settings)
                     | set(motility_advanced_settings)
                     | {"motility_analysis"})

    mask_keys = set(_settings_keys_by_variable(
        NOTEBOOKS / "01_generate_masks.ipynb")["settings"])
    timelapse_profile = set(_settings_keys_by_variable(
        NOTEBOOKS / "01b_generate_timelapse_masks.ipynb")["settings"])
    motility_profile = _settings_keys_by_variable(
        NOTEBOOKS / "14_motility_assay.ipynb")
    preprocessing = set(
        motility_profile["preprocess_generate_masks_timelapse_settings"])
    assay = set(motility_profile["automated_motility_assay_settings"])

    assert not mask_keys & (timelapse_keys | motility_keys)
    assert timelapse_profile & timelapse_keys
    assert not timelapse_profile & motility_keys
    assert not preprocessing & (timelapse_keys | motility_keys)
    assert {"src", "cell_channel"} <= preprocessing
    assert assay & motility_keys
    assert not assay & timelapse_keys


@pytest.mark.parametrize("name", [
    "01_generate_masks.ipynb",
    "01b_generate_timelapse_masks.ipynb",
    "14_motility_assay.ipynb",
])
def test_organelle_settings_have_a_separate_cell(name):
    cells = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))["cells"]
    general = next(cell for cell in cells
                   if cell.get("metadata", {}).get("spacr", {}).get(
                       "generated") == "settings")
    general_source = "".join(general["source"])
    help_text = _generated_help(NOTEBOOKS / name)
    assert "'organelle_channel':" not in general_source
    assert "'number_of_organelles':" not in general_source
    assert "**`organelle_channel`**" in help_text
    assert "**`number_of_organelles`**" in help_text
    assert "<summary>Advanced settings reference" in help_text



@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_notebook_overviews_use_scientific_section_labels(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    overview = _generated_help(path)
    assert "**Purpose.**" in overview
    assert "**Recommended use.**" in overview
    assert "**Primary outputs.**" in overview
    assert "**What it does.**" not in overview
    assert "**What you get.**" not in overview


def test_consolidated_notebooks_name_the_current_desktop_route():
    tool = _tool()
    for name, route in tool.NOTEBOOK_DESKTOP_ROUTES.items():
        notebook = json.loads((NOTEBOOKS / name).read_text(encoding="utf-8"))
        overview = _generated_help(NOTEBOOKS / name)
        assert f"**Desktop route.** {route}" in overview


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_user_facing_notebook_text_has_no_placeholder_or_informal_scaffolding(
        path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "markdown")
    banned = (
        "No description is available",
        "Turn raw",
        "What it does.",
        "What you get.",
        "## 3. Call it",
        "## 4. Run it",
        "game-changing",
        "seamlessly",
        "delve into",
        "one object's opinion",
        "chattier",
        "faster and blinder",
        "lucky split was flattering",
        "slam probabilities",
    )
    assert not [phrase for phrase in banned
                if phrase.casefold() in markdown.casefold()]


def test_volcano_overview_names_the_scientific_axes_and_actual_output():
    notebook = json.loads(
        (NOTEBOOKS / "30_volcano_plot.ipynb").read_text(encoding="utf-8"))
    overview = _generated_help(NOTEBOOKS / "30_volcano_plot.ipynb")
    assert "negative log10-transformed p-values" in overview
    assert "configured dimensions and file format" in overview


@pytest.mark.parametrize("path", ALL, ids=lambda p: p.stem)
def test_the_settings_dict_is_used_rather_than_decorative(path):
    """The dict has to BE the call's arguments, not sit beside them.

    Eight notebooks used to carry a commented-out call restating the whole
    signature by hand -- a second paste, and one a renamed parameter would
    have left silently wrong.
    """
    source = _generated(path)
    if not source:
        pytest.skip("no generated cell")
    notebook = json.loads(path.read_text(encoding="utf-8"))
    names = [n for n in ("settings",) if f"{n} = " in source]
    if not names:
        pytest.skip("multi-function notebook names its dicts per function")
    code = "\n".join("".join(c["source"]) for c in notebook["cells"]
                     if c["cell_type"] == "code" and not
                     "".join(c["source"]).startswith("# Generated"))
    assert "settings)" in code or "**settings" in code, (
        f"{path.name}: the generated dict is never passed to anything")


# ---------------------------------------------------------------------------
# The tool's own rules
# ---------------------------------------------------------------------------

def test_tooltips_are_read_now_and_never_pasted():
    """Tooltip prose comes from the public source rather than a stale copy."""
    source = TOOL.read_text(encoding="utf-8")
    assert "from spacr.settings import tooltips" in source or \
           "spacr.settings import tooltips" in source


def test_signature_parameters_fall_back_to_live_tooltips():
    tool = _tool()
    assert tool.tooltip_for("cmap", "signature", {}) == (
        tool.CORE_TOOLTIPS["cmap"])


def test_missing_setting_descriptions_fail_generation_instead_of_shipping_a_placeholder():
    tool = _tool()
    with pytest.raises(ValueError, match="has no API or tooltip description"):
        tool.render_reference([
            ("spacr.unknown.function", "signature",
             {"definitely_undocumented_key": None}, {}),
        ])


def test_the_registry_is_reused_rather_than_a_third_mapping():
    """Rule 1: the GUI and CLI already map module to defaults helper."""
    assert "from spacr.cli import MODULES" in TOOL.read_text(encoding="utf-8")


def test_a_factoryless_function_says_where_its_keys_came_from():
    """Four functions read settings.get() inline and declare nothing.

    A list recovered from a function body is a weaker claim than one read
    off a declaration; printing them identically would hide that.
    """
    tool = _tool()
    entries = [("spacr.submodules.count_phenotypes", "source",
                {"src": None}, {})]
    rendered = tool.render_reference(entries)
    assert "does not yet expose a settings factory" in rendered


def test_an_unrepresentable_default_is_flagged_not_silently_wrong():
    """A default that does not round-trip must not be written as if it did."""
    tool = _tool()

    class Odd:
        pass

    rendered = tool.render_cell([("spacr.x.y", "signature", {"k": Odd()}, {})])
    assert "set this yourself" in rendered
    ast.parse(rendered)


def test_bundled_resource_defaults_are_portable(monkeypatch):
    """Generated notebooks must not contain the checkout that built them."""
    import spacr

    tool = _tool()
    resource = REPO / "spacr" / "resources" / "data" / "barcodes_row.csv"
    rendered = tool._literal(str(resource))
    assert str(REPO) not in rendered
    assert "__path__" in rendered
    # Editable Python 3.9 installs can expose this state; the old
    # importlib.resources expression crashed while resolving ``spec.origin``.
    monkeypatch.setattr(spacr, "__spec__", None)
    assert eval(rendered) == str(resource.resolve())


@pytest.mark.parametrize("dotted", [
    "spacr.core.preprocess_generate_masks",
    "spacr.core.preprocess_generate_masks_timelapse",
])
def test_machine_sized_worker_defaults_are_rendered_as_expressions(dotted):
    """A notebook generated on CI must match one generated on a workstation."""
    tool = _tool()
    workstation = tool._literal(28, dotted=dotted, key="n_jobs")
    hosted_runner = tool._literal(1, dotted=dotted, key="n_jobs")

    assert workstation == hosted_runner
    assert "cpu_count" in workstation
    assert eval(workstation) >= 1


def test_check_mode_is_what_the_drift_guard_calls():
    """--check is the contract; the guard above runs it as a subprocess."""
    source = TOOL.read_text(encoding="utf-8")
    assert '"--check"' in source or "'--check'" in source


def test_collapsed_configuration_preserves_every_current_setting_and_default():
    tool = _tool()
    for path in ALL:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        actual_keys = _settings_keys_by_variable(path)
        namespace = {}
        for source in _generated_settings_cells(path):
            exec(source, namespace)
        for dotted in tool.declared_functions(path):
            _shape, expected = tool.notebook_surface(
                tool.resolve(dotted), dotted, notebook_name=path.name)
            name = tool.setting_name(dotted, [(item, None, None, None)
                                             for item in tool.declared_functions(path)])
            assert set(actual_keys[name]) == set(expected), (path.name, dotted)
            expected_values = {}
            for key, value in expected.items():
                literal = tool._literal(value, dotted=dotted, key=key)
                expected_values[key] = eval(literal)
            assert namespace[name] == expected_values, (path.name, dotted)
        for cell in notebook["cells"]:
            kind = cell.get("metadata", {}).get("spacr", {}).get("generated")
            if kind in {"settings-advanced", "settings-organelle"}:
                assert cell["metadata"]["jupyter"]["source_hidden"] is True
            if kind == "settings":
                assert not cell["metadata"].get("jupyter", {}).get("source_hidden")


def test_current_notebooks_validate_and_collapsible_reference_renders():
    nbformat = pytest.importorskip("nbformat")
    markdown = pytest.importorskip("nbconvert.filters.markdown")
    for path in [*ALL, NOTEBOOKS / "00_workflow_browser.ipynb"]:
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                ast.parse(cell.source)
        assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    rendered = markdown.markdown2html(_generated_help(ALL[0]))
    assert "<details>" in rendered and "<summary>" in rendered
    assert "<strong><code>src</code></strong>" in rendered


def test_browser_and_index_cover_the_actual_registry_and_give_direct_api_links():
    from spacr.cli import MODULES

    tool = _tool()
    browser = NOTEBOOKS / tool.BROWSER_FILENAME
    assert browser.is_file()
    assert {path.name for path in NOTEBOOKS.glob("*.ipynb")} == (
        set(tool.NOTEBOOK_SPECS) | {tool.BROWSER_FILENAME})
    index = (NOTEBOOKS / "README.md").read_text(encoding="utf-8")
    assert index == tool.render_notebook_index()
    for key, module in MODULES.items():
        assert f"| `{key}` |" in index
        assert tool.api_url(module.entry.replace(":", ".")) in index
    for filename in tool.NOTEBOOK_SPECS:
        assert f"]({filename})" in index


def test_browser_setup_and_editor_are_safe_and_preserve_rejected_settings(monkeypatch):
    monkeypatch.syspath_prepend(str(NOTEBOOKS))
    pytest.importorskip("ipywidgets")
    from spacr.cli import MODULES, resolve_settings

    browser = json.loads(_tool().render_workflow_browser())
    namespace = {}
    for cell in browser["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    assert namespace["RUN_WORKFLOW"] is False
    assert namespace["settings"] == resolve_settings(MODULES["measure"], None)
    assert namespace["advanced"].selected_index is None
    assert set(value for _label, value in namespace["module_picker"].options) == set(MODULES)
    before = dict(namespace["settings"])
    namespace["editor"].value = '{"unrecognized_setting": 1}'
    namespace["apply_button"].click()
    assert namespace["settings"] == before
    invalid = dict(before, batch_size="not a number")
    namespace["editor"].value = json.dumps(invalid, default=str)
    namespace["apply_button"].click()
    assert namespace["settings"] == before
    edits = dict(before, src="/a/plate", channels=[0, 1])
    namespace["editor"].value = json.dumps(edits, default=str)
    namespace["apply_button"].click()
    assert namespace["settings"]["src"] == "/a/plate"
    assert namespace["settings"]["channels"] == [0, 1]
    assert set(namespace["settings"]) == set(before)
    namespace["module_picker"].close()
    namespace["advanced"].close()


def test_browser_works_without_optional_widgets(monkeypatch):
    import builtins

    original = builtins.__import__

    def without_widgets(name, *args, **kwargs):
        if name == "ipywidgets":
            raise ImportError("optional notebook widgets absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_widgets)
    namespace = {}
    for cell in json.loads(_tool().render_workflow_browser())["cells"]:
        if cell["cell_type"] == "code":
            exec("".join(cell["source"]), namespace)
    assert namespace["module_picker"] is None
    assert namespace["settings"]
    assert namespace["RUN_WORKFLOW"] is False


def test_advanced_cell_generation_preserves_existing_notebook_tags():
    tool = _tool()
    cell = {"metadata": {"tags": ["parameters", "hide-output"]}}
    tool._mark_generated(cell, tool.ADVANCED_CELL_KIND)
    tool._mark_generated(cell, tool.ADVANCED_CELL_KIND)
    assert cell["metadata"]["tags"] == ["parameters", "hide-output", "advanced-settings"]
    assert cell["metadata"]["jupyter"]["source_hidden"] is True


def test_guided_advanced_editor_is_closed_edits_run_settings_and_resets_defaults():
    pytest.importorskip("ipywidgets")
    namespace = {}
    helper = NOTEBOOKS / "_spacr_notebook_widgets.py"
    assert helper.read_text(encoding="utf-8") == _tool()._ADVANCED_EDITOR
    exec(helper.read_text(encoding="utf-8"), namespace)
    values = {"src": "/plate", "channels": [0, 1], "batch_size": 8,
              "plot": False, "organelle_channel": None}
    defaults = dict(values, batch_size=16)
    widget = namespace["show_advanced_settings"](values, ["src", "channels"], defaults)
    assert widget.selected_index is None
    groups = widget.children[0]
    assert groups.selected_index is None
    found = []
    for panel in groups.children:
        shown = ast.literal_eval(panel.children[0].value)
        found.extend(shown)
        if "batch_size" in shown:
            chosen = panel
    assert set(found) == {"batch_size", "plot", "organelle_channel"}
    assert len(found) == len(set(found))
    editor, buttons, _message = chosen.children
    apply, reset = buttons.children
    prior = dict(values)
    editor.value = "not a dictionary"
    apply.click()
    assert values == prior
    reset.click()
    assert values["batch_size"] == 16
    edits = ast.literal_eval(editor.value)
    edits["batch_size"] = "not a number"
    editor.value = repr(edits)
    apply.click()
    assert values["batch_size"] == 16
    edits["batch_size"] = 32
    editor.value = repr(edits)
    apply.click()
    assert values["batch_size"] == 32
    assert values["src"] == "/plate" and values["channels"] == [0, 1]
    reset.click()
    assert values["batch_size"] == 16
    for panel in groups.children:
        for child in panel.children:
            child.close()
        panel.close()
    groups.close()
    widget.close()


def test_initial_markdown_has_no_parameter_wall_even_without_html_support():
    tool = _tool()
    for path in ALL:
        notebook = json.loads(path.read_text())
        markdown = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown")
        assert "<details>" not in markdown and "<summary>" not in markdown
        assert "**`src`**" not in markdown
        assert "**Purpose.**" not in markdown
        assert "Conditionally required settings" not in markdown
        assert len(markdown) < 1700, (path.name, len(markdown))
        assert f"reference/{path.stem}.md" in markdown
        reference = _generated_help(path)
        for dotted in tool.declared_functions(path):
            shape, keys = tool.notebook_surface(tool.resolve(dotted), dotted, notebook_name=path.name)
            for key in keys:
                assert f"**`{key}`**" in reference
        assert "show_help_menu" in _generated_source(path, "settings-editor")
    mask = NOTEBOOKS / "01_generate_masks.ipynb"
    assert set(_explicit_settings_keys_by_variable(mask)["settings"]) == {
        "src", "channels", "cell_channel", "nucleus_channel", "pathogen_channel"}
    code = "\n".join(_generated_settings_cells(mask))
    assert "'n_jobs':" not in code and "'cell_diameter':" not in code


@pytest.mark.parametrize("name", ["00_workflow_browser.ipynb", *[path.name for path in ALL]])
def test_colab_mirrors_current_workflows_with_native_forms_and_no_hidden_help_wall(name):
    import hashlib
    nbformat = pytest.importorskip("nbformat")
    tool = _tool()
    path = NOTEBOOKS / "colab" / name
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    assert notebook.metadata.spacr.colab_source == name
    assert notebook.metadata.colab.toc_visible is True
    assert len({cell.id for cell in notebook.cells}) == len(notebook.cells)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    assert len(markdown) < 1050 and "<details>" not in markdown
    assert all(f"## {section}" in markdown for section in ("Setup", "Data", "Inputs", "Advanced", "Run", "Outputs"))
    kinds = {}
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        ast.parse(cell.source)
        assert cell.metadata.cellView == "form"
        assert cell.source.startswith("#@title ")
        assert cell.execution_count is None and not cell.outputs
        kinds[cell.metadata.spacr.generated] = cell.source
    assert 'INSTALL_SPACR = False #@param' in kinds["colab-setup"]
    assert '[sys.executable, "-m", "pip", "install", PACKAGE, "ipywidgets"]' in kinds["colab-setup"]
    assert hashlib.sha256(tool._ADVANCED_EDITOR.encode()).hexdigest() in kinds["colab-helper"]
    assert 'DATA_SOURCE = "Existing runtime path" #@param' in kinds["colab-data"]
    assert 'PREVIEW_OUTPUTS = False #@param' in kinds["colab-preview"]
    assert 'DOWNLOAD_FILE = False #@param' in kinds["colab-download"]
    run = kinds["function-run"] if name != tool.BROWSER_FILENAME else kinds["browser-run"]
    assert 'RUN_WORKFLOW = False #@param' in run
    # These cells must run harmlessly even without Colab, inputs, or pipeline imports.
    exec(run, {})
    exec(kinds['colab-preview'], {})
    exec(kinds['colab-download'], {})
    if name == tool.BROWSER_FILENAME:
        return
    entries = tool.notebook_entries(tool.NOTEBOOK_SPECS[name])
    expected_fields = []
    namespace = {}
    exec(kinds["colab-defaults"], namespace)
    for index, (dotted, shape, keys, _docs) in enumerate(entries):
        source = kinds[f"colab-inputs-{index}"]
        essential = [key for key in keys if tool._essential_setting(dotted, shape, key, keys) and not tool._is_organelle_key(key)]
        form_lines = [line for line in source.splitlines() if "#@param" in line]
        assert len(form_lines) == len(essential)
        assert {line.partition(" = ")[0] for line in form_lines} == {f"stage_{index + 1}_{key}" for key in essential}
        expected_fields.extend(essential)
        exec(source, namespace)
        settings = namespace[tool.setting_name(dotted, entries)]
        assert set(settings) == set(keys)
        for key in essential:
            assert settings[key] == namespace[f"stage_{index + 1}_{key}"]
    assert "show_advanced_settings" in kinds["settings-editor"]
    assert "show_help_menu" in kinds["settings-editor"]


def test_legacy_is_replaced_by_two_current_sets_and_colab_navigation():
    tool = _tool()
    assert not (NOTEBOOKS / "legacy").exists()
    expected = set(tool.NOTEBOOK_SPECS) | {tool.BROWSER_FILENAME}
    assert {path.name for path in (NOTEBOOKS / "colab").glob("*.ipynb")} == expected
    index = (NOTEBOOKS / "README.md").read_text()
    for name in tool.NOTEBOOK_SPECS:
        assert f"blob/nightly/Notebooks/colab/{name}" in index
    assert "legacy/_spacr_notebook_widgets" not in TOOL.read_text()


def test_help_menu_is_closed_and_does_not_load_or_display_help_until_clicked(tmp_path, monkeypatch):
    pytest.importorskip("ipywidgets")
    namespace = {}
    exec(_tool()._ADVANCED_EDITOR, namespace)
    source = tmp_path / "help.md"
    source.write_text("# Overview\n\nLong workflow details\n\n## 2. API\n\nCall details\n\n## 3. Settings\n\nParameter details")
    import IPython.display
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    panel = namespace["show_help_menu"](str(source))
    assert panel.selected_index is None
    assert displayed == [panel]
    picker, buttons, output = panel.children[0].children
    assert picker.value == "" and not output.outputs
    picker.value = "api"
    buttons.children[0].click()
    assert len(displayed) == 2
    assert displayed[-1].data.strip() == "## 2. API\n\nCall details"
    assert "Long workflow details" not in displayed[-1].data
    panel.close()


def test_output_preview_is_bounded_and_reads_selected_csv_only_on_click(tmp_path, monkeypatch):
    pytest.importorskip("ipywidgets")
    import IPython.display
    import pandas as pd
    source = tmp_path / "measurements.csv"
    source.write_text("value\n" + "\n".join(str(value) for value in range(30)))
    namespace = {}
    exec(_tool()._ADVANCED_EDITOR, namespace)
    displayed = []
    monkeypatch.setattr(IPython.display, "display", displayed.append)
    picker, button, output = namespace["preview_outputs"](str(tmp_path), limit=1)
    assert len(picker.options) == 1
    assert len(displayed) == 1 and not output.outputs
    button.click()
    frames = [value for value in displayed if isinstance(value, pd.DataFrame)]
    assert len(frames) == 1 and frames[0]["value"].tolist() == list(range(10))
    assert displayed[-1].path == str(source)
    picker.close()
    button.close()
    output.close()
