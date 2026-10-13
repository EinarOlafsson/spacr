"""Instruction 100's residual: three lookups that skipped the hyphen fold.

`validate._normalize_app` resolves an app key in two steps -- the exact
spelling first, then with "-" folded to "_" -- because PLUGIN ALIASES ARE
REGISTERED FOLDED. A caller who writes the hyphenated spelling reaches the
alias through `validate` and, until this was fixed, reached nothing at all
through `ports.module_ports`, `ports`'s source-key lookup, or
`chaining._canonical`, each of which wrote `APP_ALIASES.get(key, key)` by
hand.

`module_ports`'s own docstring promises "every alias
:data:`spacr.validate.APP_ALIASES` accepts", which made it the clearest of
the three: the contract was written down and not kept.

100 recorded this as out of scope for the pass that found it -- "same defect,
three call sites, in files outside this pass" -- and named all three.
"""
from __future__ import annotations

import pytest

from spacr import chaining, ports
from spacr.ports import ROOT_KEYS
from spacr.validate import (ALT_SRC_KEYS, APP_ALIASES,
                            canonical_app_key)


def _a_registered_alias_with_an_underscore() -> str:
    """A real alias from the table, so this cannot drift from the data."""
    for alias in sorted(APP_ALIASES):
        if "_" in alias and APP_ALIASES[alias] in ports.PORTS:
            return alias
    pytest.skip("no registered alias carries an underscore")


def test_the_fold_is_what_is_being_tested():
    """THE CONTROL. Without a hyphen there is nothing here to get wrong.

    Every assertion below depends on the hyphenated spelling differing from
    the registered one. If the table ever stops holding underscored aliases
    this file is vacuous, and it should say so rather than pass.
    """
    alias = _a_registered_alias_with_an_underscore()
    assert "-" not in alias, "the registered spelling already has a hyphen"
    assert alias.replace("_", "-") != alias


def test_canonical_app_key_folds_the_hyphen():
    """The behaviour the other three now borrow instead of reimplementing."""
    alias = _a_registered_alias_with_an_underscore()
    assert canonical_app_key(alias.replace("_", "-")) == APP_ALIASES[alias]


def test_module_ports_accepts_the_alias_its_docstring_promises():
    """The contract was written down in the docstring and not kept."""
    alias = _a_registered_alias_with_an_underscore()
    hyphenated = alias.replace("_", "-")

    declared = ports.module_ports(hyphenated)
    assert declared is ports.module_ports(APP_ALIASES[alias]), (
        f"{hyphenated!r} and {APP_ALIASES[alias]!r} are the same module and "
        f"must return the same declaration")


def test_project_root_picks_the_same_settings_key_for_either_spelling(
        monkeypatch):
    """`ports.project_root`, the second bare lookup in that file.

    THIS ONE IS LATENT AND THE TEST HAS TO MANUFACTURE ITS CONSEQUENCE.
    `project_root` uses the resolved key ONLY to choose which settings entry
    holds the source -- ROOT_KEYS, else ALT_SRC_KEYS, else "src". Measured
    2026-09-15: EVERY underscored alias in the table resolves to a module
    whose source key is "src", so today the fold changes nothing here and a
    straightforward test of the two spellings passes whether the site is
    fixed or not. Confirmed by mutation: reverting this site alone left the
    first version of this test green.

    So the test gives one module a non-"src" root key and then asks. That is
    the state a new module with its own source folder arrives in, and the
    day one does, an unfolded alias starts reading "src" and answering ""
    for a project that is sitting right there.

    AN EARLIER VERSION OF THIS TEST WAS WORSE THAN USELESS twice over: it
    guarded itself with `if hasattr(...) else True`, and the assertion it
    guarded could not fail. Both are named here because this suite exists
    against exactly that.
    """
    alias = _a_registered_alias_with_an_underscore()
    canonical = APP_ALIASES[alias]
    monkeypatch.setitem(ports.ROOT_KEYS, canonical, "its_own_folder")
    settings = {"its_own_folder": "/tmp/a_project", "src": "/tmp/the_wrong_one"}

    hyphenated = ports.project_root(settings, alias.replace("_", "-"))
    assert hyphenated == ports.project_root(settings, canonical), (
        f"project_root read a different settings key for "
        f"{alias.replace('_', '-')!r} than for {canonical!r}")
    assert hyphenated and hyphenated.endswith("a_project"), (
        f"project_root answered {hyphenated!r}; the hyphenated alias did not "
        f"resolve, so it read 'src' instead of this module's own root key")


def test_chaining_canonical_folds_it():
    """The third site, in another file again."""
    alias = _a_registered_alias_with_an_underscore()
    assert chaining._canonical(alias.replace("_", "-")) == APP_ALIASES[alias]


def test_an_unknown_key_is_still_handed_back_as_it_arrived():
    """THE HALF THAT MUST NOT REGRESS, and `_normalize_app` says why.

    "An UNKNOWN key passes through in the spelling it arrived in, not the
    folded one: the fold exists to reach a registered alias, and rewriting a
    key nothing matched would hand the caller back a name it never used."

    A fold applied unconditionally would satisfy every test above and quietly
    rename every unknown key in the application.
    """
    assert canonical_app_key("not-a-real-module") == "not-a-real-module"
    assert chaining._canonical("not-a-real-module") == "not-a-real-module"
    with pytest.raises(ports.UnknownModule):
        ports.module_ports("not-a-real-module")
