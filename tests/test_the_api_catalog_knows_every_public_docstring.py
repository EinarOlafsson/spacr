"""The API translation catalog is asked ONE question, in three seconds.

WHY THIS EXISTS, and it is a cost that has now landed twice on a session
that did not cause it. Every new public docstring in `spacr/` is 9 x N
blocks of catalog work and a rebuild measured in hours. The only thing
that reports the drift is `build_documentation_i18n.py --audit`, which
takes about fourteen minutes; the only workflow that runs the audit is
`docs`; and `docs` cannot conclude on nightly because of
cancel-in-progress. So a commit that adds a public function is silent,
and the bill arrives whenever somebody next runs the audit -- as 144
failure lines across nine locales, in whichever lane happened to run it.

THIS IS THE SAME RULE, MADE FAST AND MADE SPECIFIC. The audit already
carries it: `set(english_symbols) != expected` produces "en: API source
manifest keys are stale", which says that something drifted and not what.
This compares the same two sets, names the symbols in both directions,
and runs in about three seconds -- so the commit that adds a symbol is
the one that hears about it.

IT IS NOT A LICENCE TO SKIP THE REBUILD. A red line here means the
catalogs owe nine translations, not that the docstring should go. Batch
them: one rebuild for a body of work costs the same as one rebuild for
one function.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "build_documentation_i18n.py"


@pytest.fixture(scope="module")
def builder():
    """The documentation-i18n tool, loaded from its path.

    By path and not by import: `tools` is not a package, and the audit
    this shares its rules with loads it the same way.
    """
    if not TOOL.is_file():
        pytest.skip("the documentation-i18n builder is not in this checkout")
    # `tools` ON THE PATH FIRST. The builder imports its sibling
    # `build_i18n_catalogs` by plain module name, which resolves only when
    # the directory is importable -- the same thing the audit's own
    # invocation does by running from `tools/`.
    import sys

    if str(TOOL.parent) not in sys.path:
        sys.path.insert(0, str(TOOL.parent))
    spec = importlib.util.spec_from_file_location("spacr_doc_i18n", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def catalog(builder):
    path = Path(builder.API_DIR) / "en.json"
    if not path.is_file():
        pytest.skip("the English API catalog is not in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def test_the_english_catalog_names_exactly_the_public_docstrings(builder,
                                                                 catalog):
    """Both directions, and each one names what moved.

    ADDED symbols are a docstring the catalogs have never seen: nine
    locales owe a translation, and until they have one the API page shows
    English inside a translated page.

    REMOVED symbols are a docstring that went away and a catalog entry
    that did not. That one is cheap to fix and expensive to leave: the
    audit refuses the whole manifest for it, so a stale key blocks the
    next real translation from landing.
    """
    found = builder.public_docstrings()
    known = set(catalog.get("symbols", {}))

    missing = sorted(set(found) - known)
    stale = sorted(known - set(found))

    assert not missing, (
        f"{len(missing)} public docstring(s) have no catalog entry, so nine "
        "locales owe a translation. Run "
        "`python tools/build_documentation_i18n.py` and translate them, "
        "batching the whole body of work into one rebuild:\n  "
        + "\n  ".join(missing[:40]))
    assert not stale, (
        f"{len(stale)} catalog entries name a docstring that no longer "
        "exists, which makes the audit refuse the manifest:\n  "
        + "\n  ".join(stale[:40]))


def test_the_manifest_is_the_schema_the_audit_expects(catalog):
    """A schema bump is a rebuild, and a silent one reads as drift."""
    assert catalog.get("schema") == 2
    assert catalog.get("symbols"), "the catalog carries no symbols at all"


def test_the_check_is_the_audits_own_rule_rather_than_a_second_one(builder):
    """Guard against this test and the audit drifting apart.

    The audit compares its `expected` set -- built from
    `public_docstrings()` plus the alias table -- against the manifest's
    keys. If that seam is ever renamed or the aliases stop being folded
    in, this test would go on passing while checking something else.
    """
    assert callable(builder.public_docstrings)
    assert hasattr(builder, "API_DOC_ALIASES")
    assert hasattr(builder, "API_DIR")
