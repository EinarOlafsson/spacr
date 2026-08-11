"""A setting nothing reads must be refused, not accepted.

``spacr-run mask --set remove_border_pathogens=True`` used to be accepted in
silence and do nothing. The key is declared: it is in ``expected_types``, it
has a tooltip, and the Pathogen category offers it in both GUIs -- so every
"is this a real setting?" test in the CLI passed it. Nothing reads it. On a
40-plate cluster job that is a GPU-week spent producing a plausible wrong
answer, and there is no error anywhere to find afterwards.

``spacr.settings.DEAD_SETTINGS`` names every such key and the spelling that
works instead; ``spacr.validate`` turns each into a pre-flight ERROR and
``spacr.cli.apply_overrides`` refuses a ``--set`` that names one.

The first test re-derives the registry from the source on every run, so it can
rot in neither direction: a key that gains a reader must leave it, and a key
that loses its last reader must join it.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re

import pytest

import spacr.settings as S
from spacr.settings import DEAD_SETTINGS


#: The dict literals in settings.py that only *declare* a setting. An
#: occurrence inside one of these is not a reader.
_DECLARATION_LITERALS = {
    "expected_types", "tooltips", "descriptions", "categories",
    "category_dependencies", "category_group_dependencies",
    "category_integer_dependencies", "category_value_dependencies",
    "DEAD_SETTINGS",
}

#: Modules that are ENTIRELY declaration -- a translated copy of the tooltip
#: table is the same kind of thing as the tooltip table itself, and carrying
#: a dead setting's help text does not give that setting a reader.
#:
#: Without this every dead key becomes "live" the moment its tooltip is
#: translated, and the registry test asserts the whole of DEAD_SETTINGS
#: should be emptied -- which would delete the very entries that keep old
#: settings CSVs loading.
_DECLARATION_MODULES = ("i18n_catalogs",)

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")

#: Phrases a dead setting's tooltip may use to admit that it is dead. The
#: point of the list is that it cannot be satisfied by accident.
_ADMITS_IT_IS_DEAD = re.compile(
    r"nothing (in spacr )?reads|no code (path )?in spacr reads|"
    r"never looks at|read by nothing|reads nowhere",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# source scan
# ---------------------------------------------------------------------------

def _package_dir():
    return pathlib.Path(S.__file__).parent


class _LiveTokens(ast.NodeVisitor):
    """Collect every name a module could read a setting through.

    Deliberately excluded, because none of them can be a read:

    * the declaration-only dict literals in settings.py (``expected_types``,
      ``tooltips``, ``categories``, ``DEAD_SETTINGS``, ...);
    * docstrings and comments -- prose *about* a dead key (this module's own
      commit message for it, the tooltip text, the comment in
      ``cli.apply_overrides`` explaining why ``pick_slice`` is refused) must
      not resurrect it.

    Included: identifiers, and the contents of every other string literal,
    because ``settings['key']`` is a string. That direction is the safe one --
    a key named in a runtime error message counts as live and simply stays out
    of the registry.
    """

    def __init__(self, skip_declarations):
        self.tokens = set()
        self._skip = skip_declarations
        self._docstrings = set()

    def _note_docstring(self, node):
        body = getattr(node, "body", None)
        if (body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            self._docstrings.add(id(body[0].value))

    def visit_Module(self, node):
        self._note_docstring(node)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        self._note_docstring(node)
        self.tokens.add(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._note_docstring(node)
        self.tokens.add(node.name)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Assign(self, node):
        if self._skip:
            names = {t.id for t in node.targets if isinstance(t, ast.Name)}
            if names & _DECLARATION_LITERALS:
                return
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if self._skip and isinstance(node.target, ast.Name) \
                and node.target.id in _DECLARATION_LITERALS:
            return
        self.generic_visit(node)

    def visit_Name(self, node):
        self.tokens.add(node.id)

    def visit_Attribute(self, node):
        self.tokens.add(node.attr)
        self.generic_visit(node)

    def visit_keyword(self, node):
        if node.arg:
            self.tokens.add(node.arg)
        self.generic_visit(node)

    def visit_arg(self, node):
        self.tokens.add(node.arg)
        self.generic_visit(node)

    def visit_alias(self, node):
        self.tokens.update(_IDENTIFIER.findall(node.name))
        if node.asname:
            self.tokens.add(node.asname)

    def visit_Constant(self, node):
        if isinstance(node.value, str) and id(node) not in self._docstrings:
            self.tokens.update(_IDENTIFIER.findall(node.value))


def _live_tokens():
    """Every name any spaCR module could read a setting through."""
    package = _package_dir()
    settings_path = os.path.abspath(S.__file__)
    tokens = set()
    files = 0
    for base, dirs, names in os.walk(package):
        dirs[:] = [d for d in dirs
                   if d not in {"__pycache__", "resources"}
                   and d not in _DECLARATION_MODULES]
        for name in names:
            if not name.endswith(".py"):
                continue
            path = os.path.join(base, name)
            source = pathlib.Path(path).read_text(encoding="utf-8")
            visitor = _LiveTokens(
                skip_declarations=os.path.abspath(path) == settings_path)
            visitor.visit(ast.parse(source))
            tokens |= visitor.tokens
            files += 1
    assert files > 30, f"only scanned {files} modules; the scan is not running"
    return tokens


def _declared_settings():
    """Every key spaCR advertises: typed, tooltipped, categorised or defaulted."""
    keys = set(S.expected_types) | set(S.tooltips)
    for group in S.categories.values():
        keys.update(k for k in group if isinstance(k, str))
    keys |= _every_default_key()
    return keys


def _every_default_key():
    """Union of every key any ``set_*`` / ``get_*`` defaults factory produces."""
    import contextlib
    import io

    keys = set()
    with contextlib.redirect_stdout(io.StringIO()):
        for name, fn in list(vars(S).items()):
            if not callable(fn):
                continue
            if not name.startswith(("set_", "get_", "default_", "deep_")):
                continue
            try:
                produced = fn({})
            except Exception:
                try:
                    produced = fn()
                except Exception:
                    continue
            if isinstance(produced, dict):
                keys.update(k for k in produced if isinstance(k, str))
    return keys


# ---------------------------------------------------------------------------
# 1. the registry is re-derived, not maintained by hand
# ---------------------------------------------------------------------------

def test_the_dead_registry_is_exactly_what_the_source_says():
    """A declared key whose name appears nowhere else in spacr/ is dead."""
    derived = _declared_settings() - _live_tokens()
    assert set(DEAD_SETTINGS) == derived, (
        "DEAD_SETTINGS disagrees with the source.\n"
        f"  gained a reader, remove from the registry: "
        f"{sorted(set(DEAD_SETTINGS) - derived)}\n"
        f"  lost its last reader, add to the registry: "
        f"{sorted(derived - set(DEAD_SETTINGS))}"
    )


def test_the_registry_is_not_empty_so_the_scan_is_really_running():
    assert "remove_border_pathogens" in DEAD_SETTINGS
    assert len(DEAD_SETTINGS) >= 11


def test_no_defaults_factory_produces_a_dead_setting():
    """Otherwise a stock settings dict would fail its own pre-flight."""
    overlap = sorted(set(DEAD_SETTINGS) & _every_default_key())
    assert not overlap, (
        f"{overlap} are both defaulted and registered dead, so every run of "
        "the pipeline that defaults them would be refused"
    )


@pytest.mark.parametrize("key", sorted(DEAD_SETTINGS))
def test_each_replacement_is_a_real_live_setting(key):
    replacement = DEAD_SETTINGS[key]
    if replacement is None:
        return
    assert replacement not in DEAD_SETTINGS, (
        f"{key} points at {replacement}, which is itself dead"
    )
    assert replacement in _declared_settings(), (
        f"{key} points at {replacement}, which spaCR does not declare"
    )


@pytest.mark.parametrize("key", sorted(DEAD_SETTINGS))
def test_each_dead_tooltip_admits_it_is_dead(key):
    """The tooltip is what the user reads before touching the knob.

    ``remove_border_pathogens`` used to read "Remove pathogen objects that
    touch the image border to avoid measuring partial pathogens." -- a flat
    description of behaviour that does not happen, while its three siblings
    all said they were dead. ``pick_slice`` ("keep a single z-slice instead of
    a maximum-intensity projection") did the same.
    """
    tooltip = S.tooltips.get(key)
    assert tooltip, f"{key} is registered dead but has no tooltip to warn in"
    assert _ADMITS_IT_IS_DEAD.search(tooltip), (
        f"the tooltip for {key} describes behaviour it does not have:\n"
        f"  {tooltip}"
    )


# ---------------------------------------------------------------------------
# 2. validate_settings refuses them
# ---------------------------------------------------------------------------

_SAMPLE_OF_TYPE = {bool: True, str: "x", int: 1, float: 1.0, list: ["c1"],
                   dict: {}, tuple: ()}


def _value_of_declared_type(key):
    """A value of the key's declared type, so the dead-key problem is alone.

    A bool in a list-typed key would add a type warning of its own and hide
    whether the dead check fired exactly once.
    """
    declared = S.expected_types.get(key, bool)
    first = declared[0] if isinstance(declared, tuple) else declared
    return _SAMPLE_OF_TYPE[first]

@pytest.mark.parametrize("key", sorted(DEAD_SETTINGS))
def test_validate_settings_reports_each_dead_key_as_an_error(key):
    from spacr.validate import validate_settings

    value = _value_of_declared_type(key)
    problems = [p for p in validate_settings({"src": "/nonexistent", key: value},
                                             "mask")
                if p.setting == key]
    assert len(problems) == 1, f"{key} produced {len(problems)}: {problems}"
    problem = problems[0]
    assert problem.is_error, (
        f"{key} is a warning, not an error; a warning does not stop "
        "`spacr-run` and the silent no-op survives"
    )
    assert "read by nothing" in problem.message
    replacement = DEAD_SETTINGS[key]
    if replacement:
        assert replacement in problem.fix
    else:
        assert "Delete" in problem.fix


def test_validate_settings_says_nothing_when_no_dead_key_is_present():
    from spacr.validate import validate_settings

    problems = validate_settings({"src": "/nonexistent",
                                  "pathogen_remove_border_objects": True},
                                 "mask")
    assert not [p for p in problems if "read by nothing" in p.message]


@pytest.mark.parametrize("module_key", sorted(__import__("spacr.cli", fromlist=["MODULES"]).MODULES))
def test_no_modules_stock_defaults_trip_the_dead_check(module_key):
    """Every ``spacr-run`` module must still be startable from its defaults."""
    from spacr.cli import MODULES, module_defaults
    from spacr.validate import validate_settings

    module = MODULES[module_key]
    problems = validate_settings(module_defaults(module), module.validate_key)
    dead = [p.setting for p in problems if "read by nothing" in p.message]
    assert not dead, f"{module_key} defaults carry dead settings: {dead}"


# ---------------------------------------------------------------------------
# 3. the CLI refuses them before anything is imported or started
# ---------------------------------------------------------------------------

def test_apply_overrides_refuses_a_dead_key_and_names_the_replacement():
    from spacr.cli import MODULES, SettingsError, apply_overrides

    with pytest.raises(SettingsError) as exc:
        apply_overrides({"src": "/tmp"}, ["remove_border_pathogens=True"],
                        MODULES["mask"])
    message = str(exc.value)
    assert "reads nowhere" in message
    assert "pathogen_remove_border_objects" in message
    assert "--describe mask" in message


def test_the_replacements_value_is_not_carried_over():
    """``--set z_projection=True`` would be a confident wrong answer.

    ``pick_slice`` is a bool; ``z_projection`` takes 'max' / 'mean' / 'sum' /
    'best_focus'. Suggesting the old value with the new key trades a silent
    no-op for a value the reader rejects.
    """
    from spacr.cli import MODULES, SettingsError, apply_overrides

    with pytest.raises(SettingsError) as exc:
        apply_overrides({"src": "/tmp"}, ["pick_slice=True"], MODULES["mask"])
    assert "z_projection" in str(exc.value)
    assert "z_projection=True" not in str(exc.value)


def test_apply_overrides_refuses_a_dead_key_with_no_replacement():
    from spacr.cli import SettingsError, apply_overrides

    with pytest.raises(SettingsError) as exc:
        apply_overrides({"src": "/tmp"}, ["skip_mode=skip"], None)
    message = str(exc.value)
    assert "reads nowhere" in message
    assert "Drop it" in message
    # module=None (the `validate` subcommand without --module) still renders.
    assert "<module>" not in message


def test_apply_overrides_without_a_module_still_names_the_replacement():
    from spacr.cli import SettingsError, apply_overrides

    with pytest.raises(SettingsError) as exc:
        apply_overrides({}, ["redunction_method=umap"], None)
    assert "reduction_method" in str(exc.value)
    assert "--describe <module>" in str(exc.value)


def test_a_live_setting_is_still_accepted():
    from spacr.cli import MODULES, apply_overrides

    settings = {"src": "/tmp", "pathogen_remove_border_objects": False}
    apply_overrides(settings, ["pathogen_remove_border_objects=True"],
                    MODULES["mask"])
    assert settings["pathogen_remove_border_objects"] is True


def test_spacr_run_exits_usage_rather_than_starting(tmp_path, capsys):
    """The whole point: the process must not reach the pipeline."""
    import pandas as pd

    from spacr.cli import EXIT_USAGE, main

    plate = tmp_path / "plate"
    (plate / "merged").mkdir(parents=True)
    settings_csv = tmp_path / "mask.csv"
    pd.DataFrame([("src", str(plate))],
                 columns=["Key", "Value"]).to_csv(settings_csv, index=False)

    code = main(["mask", "--settings", str(settings_csv),
                 "--set", "remove_border_pathogens=True"])
    assert code == EXIT_USAGE
    assert "reads nowhere" in capsys.readouterr().err


def test_a_settings_file_carrying_a_dead_key_fails_pre_flight(tmp_path, capsys):
    """``--set`` is not the only way in; an old settings CSV is the other."""
    import pandas as pd

    from spacr.cli import EXIT_USAGE, main

    plate = tmp_path / "plate"
    (plate / "merged").mkdir(parents=True)
    settings_csv = tmp_path / "mask.csv"
    pd.DataFrame([("src", str(plate)), ("remove_border_pathogens", "True")],
                 columns=["Key", "Value"]).to_csv(settings_csv, index=False)

    code = main(["validate", "--module", "mask", "--settings", str(settings_csv)])
    assert code == EXIT_USAGE
    assert "read by nothing" in capsys.readouterr().out
