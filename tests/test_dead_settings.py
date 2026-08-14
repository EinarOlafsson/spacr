"""A setting nothing reads must not EXIST, let alone be accepted.

``spacr-run mask --set remove_border_pathogens=True`` used to be accepted in
silence and do nothing. The key was declared -- it was in ``expected_types``,
it had a tooltip, and the Pathogen category offered it in both GUIs -- so
every "is this a real setting?" check passed it. Nothing read it. On a
40-plate cluster job that is a GPU-week spent producing a plausible wrong
answer, with no error anywhere to find afterwards.

THE POLICY CHANGED ON 2026-08-11, at the maintainer's instruction: "remove
dead settings entirely". There used to be a ``DEAD_SETTINGS`` registry that
kept such keys DECLARED -- so an old settings CSV still loaded far enough to
be told, by name, what to use instead -- plus a pre-flight validator and a
``--set`` refusal built on it. The registry, its 27 entries, the validator
and the CLI branch are all gone.

So the rule is absolute rather than documented: a setting spaCR declares is a
setting spaCR reads. There is no third state, and a key that loses its last
reader is DELETED rather than registered.

WHAT THAT COSTS, stated plainly because it is a real trade. Eighteen of the
twenty-seven entries were RENAMES with a working replacement --
``remove_border_cells`` -> ``cell_remove_border_objects``, ``pick_slice`` ->
``z_projection``. An old settings CSV naming one now fails with the generic
"names a setting that does not exist -- did you mean ...?" from
``_check_unknown_keys`` instead of a targeted migration hint. Both paths fail
LOUDLY; neither can produce a silent wrong answer, which is the property that
mattered.

This test re-derives its answer from the source on every run, so it cannot
rot in either direction.
"""
from __future__ import annotations

import ast
import os
import pathlib
import re

import pytest

import spacr.settings as S


#: The dict literals in settings.py that only *declare* a setting. An
#: occurrence inside one of these is not a reader.
_DECLARATION_LITERALS = {
    "expected_types", "tooltips", "descriptions", "categories",
    "category_dependencies", "category_group_dependencies",
    "category_integer_dependencies", "category_value_dependencies",
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
    # Secondary organelle settings are generated and read through a
    # role-scoped view. Their concrete spellings intentionally do not occur as
    # string literals; the closed runtime registry is the proof that the
    # dynamic reader covers them.
    tokens |= set(S.DYNAMIC_ORGANELLE_SETTINGS)
    return tokens - _NAMES_THAT_ARE_NOT_SETTING_READS


#: Keys whose SPELLING collides with a live string literal that has nothing
#: to do with the setting. The scan is deliberately name-based -- that is what
#: makes it impossible to fool by an indirect read -- but a two-letter name is
#: cheap enough that another meaning exists.
#:
#: 'nc' and 'pc' are the settings that named the negative and positive control
#: gRNA in get_map_barcodes_default_settings. Nothing reads either:
#: `settings['nc']` and `settings.get('nc')` appear nowhere in spacr/. What the
#: scan finds instead is the CLASS NAME 'nc' -- submodules.py:2387 and :3361
#: `settings.setdefault('pathogen_types', ['nc', 'pc'])`, io.py:5735
#: `classes = ['nc', 'pc']`, hits.py:561 and :848 filtering
#: `condition in ("nc", "pc", "control")`. Same two letters, different thing.
#:
#: Listed here rather than by loosening the scan, so the exemption is two
#: names with a reason and not a hole. Anything added here needs the same:
#: the grep that proves there is no settings read, and the line that shows
#: what the collision actually is.
_NAMES_THAT_ARE_NOT_SETTING_READS = {"nc", "pc"}


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
# the absolute rule
# ---------------------------------------------------------------------------

def test_no_declared_setting_is_unread():
    """Every key spaCR declares has a reader somewhere under spacr/.

    This replaced `test_the_dead_registry_is_exactly_what_the_source_says`,
    which asserted the registry EQUALLED the set of unread keys. With the
    registry gone the assertion is simply that the set is empty.

    When this fails the fix is to DELETE the key -- from `expected_types`,
    from `tooltips`/`descriptions`, from every `categories` list, and from
    whatever defaults factory produces it. Not to register it anywhere.
    """
    unread = sorted(_declared_settings() - _live_tokens())
    assert not unread, (
        "these settings are declared and read by nothing:\n  "
        + "\n  ".join(unread)
        + "\n\nDelete them. A setting spaCR declares is a setting spaCR "
          "reads; there is no third state."
    )


def test_the_scan_is_really_running():
    """A scan that silently matched nothing would pass the test above."""
    declared = _declared_settings()
    live = _live_tokens()
    assert len(declared) > 200, f"only {len(declared)} settings declared"
    assert len(live) > 2000, f"only {len(live)} live tokens found"
    # A key that certainly IS read must be seen as live, or the scan is broken.
    assert "cell_channel" in live
    assert "cell_channel" in declared


def test_the_retired_keys_are_gone_from_every_declaration_site():
    """The 27 that were retired on 2026-08-11, named so a revert is visible."""
    import spacr.settings as S

    retired = [
        "all_to_mip", "barecode_length_1", "barecode_length_2",
        "class_1_threshold", "custom_measurement", "gene_weights_csv",
        "metadata_types", "nc", "nc_loc", "nucleus_loc", "pc", "pc_loc",
        "pick_slice", "postprocess_cell_masks", "postprocess_nucleus_masks",
        "postprocess_organelle_masks", "postprocess_pathogen_masks",
        "redunction_method", "remove_border_cells", "remove_border_nuclei",
        "remove_border_organelles", "remove_border_pathogens",
        "signal_direction", "skip_mode", "use_sam_cell", "use_sam_nucleus",
        "use_sam_pathogen",
    ]
    back = []
    for key in retired:
        if key in S.expected_types or key in getattr(S, "tooltips", {}) \
                or key in S.descriptions:
            back.append(key)
        for group in S.categories.values():
            if key in group:
                back.append(f"{key} (in a category)")
    assert not back, f"retired settings are declared again: {sorted(set(back))}"


def test_the_registry_itself_is_gone():
    """`DEAD_SETTINGS` is not a thing spaCR has any more."""
    import spacr.settings as S
    import spacr.validate as V

    assert not hasattr(S, "DEAD_SETTINGS")
    assert not hasattr(V, "_check_dead_settings")


def test_no_defaults_factory_produces_an_unread_setting():
    """A stock settings dict cannot contain a key nothing reads."""
    unread = _declared_settings() - _live_tokens()
    produced = _every_default_key()
    assert not (produced & unread), (
        f"a defaults factory produces unread keys: {sorted(produced & unread)}")


# ---------------------------------------------------------------------------
# a retired key now fails as an UNKNOWN key -- loudly, just less specifically
# ---------------------------------------------------------------------------

def test_a_retired_key_is_refused_by_set():
    """It used to be refused as "declared but unread". Now it does not exist.

    The message is generic, but it still refuses -- which is the property
    that stops a GPU-week from being spent on a silent no-op.
    """
    from spacr.cli import MODULES, SettingsError, apply_overrides

    with pytest.raises(SettingsError, match="does not exist"):
        apply_overrides({"src": "/tmp"}, ["remove_border_pathogens=True"],
                        MODULES["mask"])


def test_a_live_setting_is_still_accepted():
    from spacr.cli import MODULES, apply_overrides

    out = apply_overrides({"src": "/tmp", "cell_channel": 0},
                          ["cell_channel=2"], MODULES["mask"])
    assert out["cell_channel"] == 2
