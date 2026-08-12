"""The test suite's own hygiene, enforced by AST over ``tests/``.

Three ways a test can look green while testing nothing, each of which this
suite has actually shipped:

1. **No assertion at all.** The test calls the function and checks nothing.
   ``tests/test_all_plotting_functions.py`` carried fourteen of these; every
   one of them passed against a blank figure.
2. **A broad ``except`` that turns a failure into a ``pytest.skip``.** A
   self-skip makes "this machine cannot run the test" and "the product is
   broken" indistinguishable. ``test_image_umap_end_to_end`` reported skipped
   for its entire life while ``generate_image_umap`` was never once called.
3. **A machine-specific absolute path.** A test whose precondition is
   ``/home/<someone>/datasets`` runs on exactly one computer and reports
   green everywhere else.

Each rule carries a RATCHET list: the violations that existed when the rule
was written, so the rule passes today and fails the moment a NEW one appears.
The lists may only shrink. They are keyed by (file, function) rather than by
file, so adding an assertion-free test to an already-listed module still
fails, and an entry naming a function that no longer exists is ignored rather
than fatal (renaming or deleting a test must never break this file).

Where one key can cover more than one violation -- a function holding two
broad excepts, a module holding two identically-named mocks -- the ratchet
records a COUNT, not just the key. A set under-reports: the broad-skip list
held 38 (file, function) pairs against 46 actual handlers, so a second
failure-swallowing handler could be added to any listed function and the
rule would still be green.

**Rules 1 and 2 are no longer ratchets: both lists are empty and both
ceilings are 0.** For rule 2 that took three changes to the enforcement, on
top of removing the 44 handlers, because a ratchet that only caps growth is
not a ban:

* the ceiling is asserted with ``==``, not ``<=``, so it can only be edited
  deliberately and the room freed by removing one violation cannot be spent on
  a new one somewhere else;
* an entry whose real count has DROPPED fails too ("lower the ratchet"),
  because an unspent allowance is a place the next violation can be added for
  free -- which is exactly how such a list rots into a permanent licence;
* the detector is proved against a table of evasions rather than trusted:
  ``except BaseException``, ``except (ValueError, Exception)``,
  ``except builtins.Exception``, ``raise unittest.SkipTest``, a ``pytest.skip``
  moved one or two helper frames away, a bare ``return`` instead of a skip, the
  handler pushed down into a nested ``def`` or hoisted to module scope, and the
  module renamed out from under its allowance. See ``BROAD_SKIP_EVASIONS``
  and ``BROAD_SKIP_ALLOWED`` for what is caught and what stays legal.

Fix, do not extend. Every entry below is a test that is not currently earning
its green.
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MISSING_CHANNEL_AXIS, check_cellpose_eval_call

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent


# ---------------------------------------------------------------------------
# AST helpers
# ---------------------------------------------------------------------------

#: Context managers that ARE an assertion: entering them asserts that the body
#: raises/warns.
_ASSERTING_CONTEXTS = frozenset({"raises", "warns", "deprecated_call"})


def _test_modules():
    """Every python module under tests/, path-relative to tests/."""
    return sorted(TESTS_DIR.rglob("*.py"))


def _rel(path):
    return str(Path(path).relative_to(TESTS_DIR))


def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _call_name(node):
    """The bare name of whatever a Call node calls (``a.b.c()`` -> ``'c'``)."""
    func = node.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")


def _functions(tree):
    return [n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _direct_assertions(fn):
    """``(has_direct_assertion, names_it_calls)`` for one function body."""
    called = set()
    found = False
    for node in ast.walk(fn):
        if node is fn:
            continue
        if isinstance(node, ast.Assert):
            found = True
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and _call_name(ctx) in _ASSERTING_CONTEXTS:
                    found = True
        elif isinstance(node, ast.Call):
            name = _call_name(node)
            called.add(name)
            # `assertEqual`, `assert_allclose`, `np.testing.assert_*`,
            # `pytest.fail`, `pytest.raises(...)` used bare.
            if name.startswith("assert") or name in _ASSERTING_CONTEXTS or name == "fail":
                found = True
    return found, called


def _asserts_something(fn, helpers, memo, stack=()):
    """True when ``fn`` asserts, directly or via a helper in the same module.

    Resolving one module's own helpers matters: a test whose whole body is
    ``_nonempty_file(path)`` or ``_assert_masks_match(a, b)`` is a real test,
    and a rule that could not see through the helper would push authors to
    inline everything.
    """
    direct, called = _direct_assertions(fn)
    if direct:
        return True
    for name in called:
        if name in stack:
            continue                      # recursive helper; already walked
        helper = helpers.get(name)
        if helper is None:
            continue
        if name in memo:
            if memo[name]:
                return True
            continue
        result = _asserts_something(helper, helpers, memo, stack + (name,))
        memo[name] = result
        if result:
            return True
    return False


#: Calls that end a test without running it.
_SKIP_CALLS = frozenset({"skip", "importorskip", "xfail"})

#: Exception classes whose ``raise`` IS a skip rather than a re-raise.
#: ``except Exception: raise unittest.SkipTest(...)`` is this whole
#: anti-pattern wearing a raise statement, and the first version of the rule --
#: "any ``raise`` in the handler means the failure survives" -- waved it
#: straight through.
_SKIP_EXCEPTIONS = frozenset({"SkipTest", "Skipped", "OutcomeException"})

#: Names that mean "catch everything".
_BROAD_NAMES = frozenset({"Exception", "BaseException"})


def _is_broad_type(caught):
    """True when this ``except`` clause catches essentially everything.

    Four spellings, identical in effect, of which the first version of this
    rule recognised two:

    * ``except:``                        -- ``caught is None``
    * ``except Exception``               -- a Name
    * ``except builtins.Exception``      -- an Attribute
    * ``except (ValueError, Exception)`` -- a Tuple with a broad member. The
      narrow name sitting beside it changes nothing about what is caught, but
      it makes the clause *look* specific to a reader and to a rule that only
      inspected ``ast.Name``.
    """
    if caught is None:
        return True
    if isinstance(caught, ast.Name):
        return caught.id in _BROAD_NAMES
    if isinstance(caught, ast.Attribute):
        return caught.attr in _BROAD_NAMES
    if isinstance(caught, ast.Tuple):
        return any(_is_broad_type(element) for element in caught.elts)
    return False


def _raise_is_a_skip(node):
    """True when this ``raise`` raises a skip instead of a failure.

    A bare ``raise`` re-raises whatever was caught, so the failure survives and
    the handler is honest. These do not:

    * ``raise unittest.SkipTest(...)`` / ``raise Skipped(...)`` -- named class;
    * ``raise pytest.skip.Exception(...)`` -- the SAME class, reached through
      the function instead of imported. It is spelled ``<something>.Exception``
      and slipped past the first version of this check, which only compared the
      final attribute against a list of class names.
    """
    exc = node.exc
    if exc is None:
        return False                      # bare `raise`: the failure survives
    if isinstance(exc, ast.Call):
        exc = exc.func
    if isinstance(exc, ast.Attribute):
        if exc.attr in _SKIP_EXCEPTIONS:
            return True
        # `pytest.skip.Exception`, `pytest.xfail.Exception`: the attribute is
        # `Exception` and the thing it hangs off is the skip function itself.
        if exc.attr == "Exception":
            owner = exc.value
            name = (owner.attr if isinstance(owner, ast.Attribute)
                    else getattr(owner, "id", ""))
            return name in _SKIP_CALLS
        return False
    if isinstance(exc, ast.Name):
        return exc.id in _SKIP_EXCEPTIONS
    return False


def _own_body(node):
    """Every node under ``node`` that is not inside a nested function.

    ``ast.walk`` would descend into a ``def`` written inside the body, which
    both mis-attributes that inner function's handlers to the outer scope and
    lets a ``return`` belonging to a nested closure read as an early exit from
    the test.
    """
    out = []

    def visit(parent):
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                  ast.Lambda)):
                continue
            out.append(child)
            visit(child)

    visit(node)
    return out


def _skipping_helpers(tree):
    """Module-local function names that reach a skip when called.

    ``except Exception: _bail(exc)``, where ``_bail`` calls ``pytest.skip``, is
    the banned handler with one extra stack frame in front of it. Resolved to a
    fixed point, so a chain of wrappers does not launder it either.
    """
    functions = {fn.name: fn for fn in _functions(tree)}
    skipping = set()
    while True:
        grew = False
        for name, fn in functions.items():
            if name in skipping:
                continue
            for node in ast.walk(fn):
                if isinstance(node, ast.Call) and (
                        _call_name(node) in _SKIP_CALLS
                        or _call_name(node) in skipping):
                    skipping.add(name)
                    grew = True
                    break
                if isinstance(node, ast.Raise) and _raise_is_a_skip(node):
                    skipping.add(name)
                    grew = True
                    break
        if not grew:
            return skipping


def _is_broad_skip(handler, skipping_helpers=frozenset(), in_a_test=False):
    """True when this ``except`` catches everything and ends the test green.

    Deliberately NOT covered: ``except Exception: pass`` / ``: continue``. It
    is a weaker sibling -- almost every instance in this suite is a fixture
    closing figures or a teardown that must not mask the real failure -- and
    folding it in here would flag ~15 honest cleanups while the rule this file
    is named for is about the skip. It wants its own rule, with its own list.
    """
    if not _is_broad_type(handler.type):
        return False
    body = _own_body(handler)
    raises = [n for n in body if isinstance(n, ast.Raise)]
    if any(not _raise_is_a_skip(r) for r in raises):
        return False                      # re-raises: the failure survives
    if raises:
        return True                       # ...and every raise was a skip
    for node in body:
        if isinstance(node, ast.Call):
            name = _call_name(node)
            if name in _SKIP_CALLS or name in skipping_helpers:
                return True
    # `except Exception: return` inside a test body is the same trade with the
    # evidence removed: the test stops and reports PASSED, so there is not even
    # a skipped line in the log for anyone to notice.
    return in_a_test and any(isinstance(n, ast.Return) for n in body)


def _owning_function(tree):
    """``id(handler) -> innermost enclosing function name`` for one module."""
    owner = {}
    for fn in _functions(tree):
        for node in _own_body(fn):
            if isinstance(node, ast.ExceptHandler):
                owner[id(node)] = fn.name
    return owner


# ---------------------------------------------------------------------------
# Rule 1 — every test asserts something
# ---------------------------------------------------------------------------

#: Tests that assert nothing today. Snapshot taken when this file was written;
#: this list may only shrink. Fix the test, do not add to it.
#:
#: **It is empty.** All 81 entries were given real assertions; the ceiling below
#: is 0, so the rule is now absolute rather than ratcheted and the next
#: assertion-free test to appear anywhere in ``tests/`` fails this file.
#:
#: What "a real assertion" meant, case by case, because the temptation is to
#: satisfy the rule rather than the test:
#:
#: * A "does not raise" test asserts the resulting STATE. A validator returns
#:   ``None`` for a legal pair *and refuses the neighbouring illegal one* --
#:   without the second half, a guard whose body is ``return`` passes.
#: * A test whose subject is "this renders" asserts PIXELS or widget geometry.
#:   ``test_cursor_overlay_draws_on_pixmap`` reads the QPixmap back and pins
#:   which pixels moved and where; the mask outline colour sat stuck on green
#:   for months behind a test that asserted the setting instead.
#: * A swallowed-exception test asserts that the broken collaborator was really
#:   reached and that a working one alongside still gets through. "Nothing
#:   raised" is also true of a call that never happened.
#:
#: One entry was deleted rather than fixed:
#: ``test_analysis_modules_t10.py::test_analyze_entrypoint_smoke`` skipped for
#: its entire life on a missing ``prcf`` column, so it had never executed a
#: line of the three analyze_* entry points it named -- all of which are driven
#: for real by the ``test_cov_submodules_*`` modules.
ASSERTION_FREE_RATCHET: dict[str, set[str]] = {}

#: Total entries above. Pinned so a fix that "resolves" a violation by adding
#: two more cannot pass. Zero: the debt is paid, and it may not be re-borrowed.
ASSERTION_FREE_CEILING = 0


def _assertion_free_tests():
    """Every ``test_*`` function in the suite that asserts nothing."""
    offenders = []
    for path in _test_modules():
        tree = _parse(path)
        helpers = {f.name: f for f in _functions(tree)}
        memo = {}
        for fn in _functions(tree):
            if not fn.name.startswith("test"):
                continue
            if not _asserts_something(fn, helpers, memo):
                offenders.append((_rel(path), fn.name, fn.lineno))
    return offenders


def test_no_new_assertion_free_tests():
    """A test that asserts nothing cannot fail, so it is not a test."""
    offenders = _assertion_free_tests()
    unlisted = [(f, n, ln) for f, n, ln in offenders
                if n not in ASSERTION_FREE_RATCHET.get(f, ())]
    assert not unlisted, (
        "these tests call code and assert nothing:\n" +
        "\n".join(f"  {f}:{ln} {n}()" for f, n, ln in unlisted) +
        "\n\nGive each one a real assertion. `assert x is not None` does not "
        "count -- assert the shape, the values, the file that was written. "
        "Adding them to ASSERTION_FREE_RATCHET is not a fix."
    )


def test_the_assertion_free_ratchet_only_shrinks():
    """The snapshot is a debt ceiling, not a budget to spend."""
    total = sum(len(v) for v in ASSERTION_FREE_RATCHET.values())
    assert total <= ASSERTION_FREE_CEILING, (
        f"ASSERTION_FREE_RATCHET grew to {total} entries (ceiling "
        f"{ASSERTION_FREE_CEILING}). Entries come off this list, never on.")
    # And nothing on it may be stale-by-file: a whole module disappearing is
    # worth noticing, unlike a single renamed test.
    missing = [f for f in ASSERTION_FREE_RATCHET if not (TESTS_DIR / f).is_file()]
    assert not missing, (
        f"ASSERTION_FREE_RATCHET names modules that no longer exist: {missing}")


# ---------------------------------------------------------------------------
# Rule 2 — a broad except may not hide a failure behind a skip
# ---------------------------------------------------------------------------

#: The scope name used for a handler that is not inside any function.
MODULE_SCOPE = "<module>"

#: Scopes containing ``except Exception: ... pytest.skip(...)`` with no
#: re-raise, and HOW MANY such handlers each holds, keyed
#: ``file -> scope -> count``.
#:
#: **It is empty.** All 44 came off, and the ceiling below is 0, so the shape
#: is now banned outright rather than ratcheted: the next one to appear
#: anywhere under ``tests/`` fails this file. Nothing goes back on this list --
#: :func:`test_the_broad_skip_ratchet_only_shrinks` pins the ceiling with
#: ``==``, so adding an entry fails whether or not you also edit the total.
#:
#: What the 44 turned out to be, and what replaced each:
#:
#: * **17 were import guards.** ``try: import torch / except Exception:
#:   pytest.skip(...)`` says "torch is missing" and means "torch raised". They
#:   became ``pytest.importorskip("torch")``, which catches ImportError only,
#:   so a package that IS installed and detonates on import now fails.
#: * **4 were at module scope** -- the ``try: import spacr.gui_elements /
#:   except Exception: pytest.skip(..., allow_module_level=True)`` shape, the
#:   highest-blast-radius form of this pattern, since one of them turns an
#:   import-time product bug into a whole FILE reporting skipped. All four were
#:   dead by the time they were read: ``tests/conftest.py`` stubs mouseinfo,
#:   pyautogui and screeninfo before any test module loads, so the display-less
#:   ImportError they were written for cannot happen. Deleted.
#: * **1 was a real environmental guard** with the wrong exception type: a
#:   ``subprocess.run(["git", ...])`` that skips where there is no checkout.
#:   Narrowed to ``(OSError, subprocess.SubprocessError)``.
#: * **The remaining 22 guarded PRODUCT BEHAVIOUR** -- ``try: measure_crop(...)
#:   / except Exception: pytest.skip("measure bailed on this dataset")``. Every
#:   one was deleted and the call left to fail. That is the point of the rule:
#:   "the pipeline crashed on its own output" and "you are not set up to run
#:   this" had been the same colour in the log.
#:
#: Two of those 22 were hiding a live bug, now pinned with
#: ``@pytest.mark.xfail(strict=True)`` in ``test_extended_coverage.py``: see
#: ``SECOND_TK_ROOT_ICON_BUG`` there.
#:
#: If you are here because this rule just failed you: the fix is a narrower
#: ``except`` (``pytest.importorskip`` for a package, ``OSError`` for a file or
#: a download), one of the markers pytest.ini already declares (``gpu``,
#: ``nas``, ``network``, ``gui``, ``heavy``, ``slow``, ``qt``), or letting the
#: failure fail. Not an entry here.
BROAD_SKIP_RATCHET: dict[str, dict[str, int]] = {}

#: Total HANDLERS above, not keys. Was 50, then 44, now zero. The ceiling is
#: asserted with ``==``: it is a debt that has been paid off, and re-borrowing
#: is the thing this file exists to prevent.
BROAD_SKIP_CEILING = 0


def _scan_broad_skips(tree):
    """``(scope, lineno)`` for every broad-except-to-skip handler in one tree.

    ``scope`` is the INNERMOST enclosing function's name, or
    :data:`MODULE_SCOPE` when the handler is not inside one. Both halves are
    load-bearing:

    * module scope, because an import guard at the top of a file skips the
      ENTIRE file, and the first version of this rule walked function bodies
      only and could not see a single one of them;
    * innermost, because attributing a handler to every enclosing ``def``
      double-counted nested helpers, and because a handler pushed down into a
      nested ``def _try_it()`` inside an already-listed test must land on a
      scope the ratchet does not name.
    """
    owner = _owning_function(tree)
    helpers = _skipping_helpers(tree)
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        scope = owner.get(id(node), MODULE_SCOPE)
        if _is_broad_skip(node, helpers, in_a_test=scope.startswith("test")):
            sites.append((scope, node.lineno))
    return sites


def _broad_skip_sites(modules=None):
    """``(file, scope, lineno)`` for every broad-except-to-skip handler.

    Walks EVERY ``*.py`` under ``tests/``, not just ``test_*.py``: moving the
    handler into ``tests/helpers.py`` and calling it from the test would
    otherwise retire it from this rule without retiring it from the suite.
    """
    sites = []
    for path in (_test_modules() if modules is None else modules):
        tree = _parse(path)
        for scope, lineno in _scan_broad_skips(tree):
            sites.append((_rel(path), scope, lineno))
    return sites


def _ratchet_verdict(sites, ratchet):
    """``(over_budget, slack)`` for a set of sites judged against a ratchet.

    ``over_budget`` -- a scope holding MORE handlers than it is allowed. That
    covers a brand-new handler, a handler moved into a file the list does not
    name (renaming the module retires its allowance), and a handler pushed into
    a differently-named function inside a listed file.

    ``slack`` -- a listed scope holding FEWER than its allowance. Slack is a
    failure too. An allowance nobody is using is a place the next handler can
    be added for free, which is precisely how a ratchet decays into a permanent
    licence; the entry has to come down at the moment the handler goes.
    """
    found = {}
    for f, scope, lineno in sites:
        found.setdefault((f, scope), []).append(lineno)
    over_budget = []
    for (f, scope), linenos in sorted(found.items()):
        allowed = ratchet.get(f, {}).get(scope, 0)
        if len(linenos) > allowed:
            over_budget.append((f, scope, sorted(linenos), allowed))
    slack = []
    for f, scopes in sorted(ratchet.items()):
        for scope, allowed in sorted(scopes.items()):
            actual = len(found.get((f, scope), ()))
            if actual < allowed:
                slack.append((f, scope, actual, allowed))
    return over_budget, slack


def test_no_new_failure_swallowing_skips():
    """``except Exception: pytest.skip(...)`` reports a bug as an excuse."""
    over_budget, _ = _ratchet_verdict(_broad_skip_sites(), BROAD_SKIP_RATCHET)
    assert not over_budget, (
        "these scopes turn any failure into a skip more often than the "
        "ratchet allows:\n" +
        "\n".join(f"  {f}: {scope} has {len(lns)} handler(s) at lines {lns}, "
                  f"ratchet allows {allowed}"
                  for f, scope, lns, allowed in over_budget) +
        "\n\nCatch the specific exception the environment can actually raise "
        "(pytest.importorskip for a missing package, OSError for a download), "
        "or re-raise. A skip must never be reachable from a bug in spaCR. "
        f"A scope of {MODULE_SCOPE!r} means the guard is at module level and "
        "skips the whole file. The ratchet is not somewhere to put a new one: "
        "it is exact, and adding a line to it fails the ceiling test below."
    )


def test_the_broad_skip_ratchet_carries_no_unspent_allowance():
    """A handler that was fixed must take its ratchet entry with it.

    Without this, the list rots: the allowance for a scope survives the
    handler it was written for, and the next broad-except dropped into that
    same function is free. "Only caps growth" is not a ban.
    """
    _, slack = _ratchet_verdict(_broad_skip_sites(), BROAD_SKIP_RATCHET)
    assert not slack, (
        "BROAD_SKIP_RATCHET allows more handlers than these scopes still "
        "hold:\n" +
        "\n".join(f"  {f}: {scope} now has {actual}, ratchet allows {allowed}"
                  for f, scope, actual, allowed in slack) +
        "\n\nLower the ratchet: set each entry to the count that is really "
        "there (drop the key entirely at zero) and subtract the difference "
        "from BROAD_SKIP_CEILING. Leaving the old number behind hands the "
        "next author a free handler in a scope nobody is watching."
    )


def test_the_broad_skip_ratchet_only_shrinks():
    """The ceiling is EXACT, so the total can only be edited downwards.

    ``<=`` was not a ban. It let the suite sit at its historical high-water
    mark forever and, worse, it meant a handler removed from one file bought
    room for a handler added to another without either test noticing. ``==``
    makes every change to the count a deliberate edit of this constant, and
    the only direction the constant may be edited is down.
    """
    total = sum(sum(scopes.values()) for scopes in BROAD_SKIP_RATCHET.values())
    assert total == BROAD_SKIP_CEILING, (
        f"BROAD_SKIP_RATCHET totals {total} handlers but BROAD_SKIP_CEILING "
        f"is {BROAD_SKIP_CEILING}.\n"
        f"  {total} > {BROAD_SKIP_CEILING}: you added a broad except -> skip. "
        f"Do not raise the ceiling; catch the specific exception, or use "
        f"pytest.importorskip / an existing marker (gpu, nas, network, gui, "
        f"heavy, slow, qt), or let the failure fail.\n"
        f"  {total} < {BROAD_SKIP_CEILING}: you removed one -- thank you -- "
        f"now lower BROAD_SKIP_CEILING to {total} so the room you freed "
        f"cannot be spent by somebody else.")
    missing = [f for f in BROAD_SKIP_RATCHET if not (TESTS_DIR / f).is_file()]
    assert not missing, (
        f"BROAD_SKIP_RATCHET names modules that no longer exist: {missing}")


# --- the detector's own coverage, proved rather than assumed ----------------

def _synthetic_sites(source):
    """``{scope: [lineno, ...]}`` for a snippet of test source."""
    sites = {}
    for scope, lineno in _scan_broad_skips(ast.parse(textwrap.dedent(source))):
        sites.setdefault(scope, []).append(lineno)
    return sites


#: Every way anyone has thought of to keep the shape and lose the rule, each
#: paired with the scope the walk must attribute it to. Written as source
#: rather than as live code so that proving the rule does not require shipping
#: a test that really does swallow its failures.
BROAD_SKIP_EVASIONS = {
    "bare except": ("""
        def test_x():
            try:
                spacr.thing()
            except:
                pytest.skip("bare")
    """, "test_x"),
    "except BaseException": ("""
        def test_x():
            try:
                spacr.thing()
            except BaseException as e:
                pytest.skip(str(e))
    """, "test_x"),
    "a narrow name beside the broad one": ("""
        def test_x():
            try:
                spacr.thing()
            except (ValueError, Exception) as e:
                pytest.skip(str(e))
    """, "test_x"),
    "the broad name reached through a module": ("""
        def test_x():
            try:
                spacr.thing()
            except builtins.Exception as e:
                pytest.skip(str(e))
    """, "test_x"),
    "raise SkipTest instead of calling skip": ("""
        def test_x():
            try:
                spacr.thing()
            except Exception as e:
                raise unittest.SkipTest(str(e))
    """, "test_x"),
    "raise the skip class reached through pytest.skip.Exception": ("""
        def test_x():
            try:
                spacr.thing()
            except BaseException as e:
                raise pytest.skip.Exception(str(e))
    """, "test_x"),
    "raise the skip class imported under its own name": ("""
        def test_x():
            try:
                spacr.thing()
            except Exception as e:
                raise Skipped(str(e))
    """, "test_x"),
    "the skip moved one frame away": ("""
        def _bail(exc):
            pytest.skip(f"nope: {exc}")

        def test_x():
            try:
                spacr.thing()
            except Exception as e:
                _bail(e)
    """, "test_x"),
    "...and two frames away": ("""
        def _really_bail(exc):
            pytest.skip(f"nope: {exc}")

        def _bail(exc):
            _really_bail(exc)

        def test_x():
            try:
                spacr.thing()
            except Exception as e:
                _bail(e)
    """, "test_x"),
    "return instead of skip, so not even a skipped line appears": ("""
        def test_x():
            try:
                result = spacr.thing()
            except Exception:
                return
            assert result
    """, "test_x"),
    "pushed down into a nested def inside a listed test": ("""
        def test_x():
            def _try_it():
                try:
                    return spacr.thing()
                except Exception as e:
                    pytest.skip(str(e))
            assert _try_it()
    """, "_try_it"),
    "hoisted to module level, where it skips the whole file": ("""
        try:
            import spacr.gui_elements as ge
        except Exception as e:
            pytest.skip(f"unavailable: {e}", allow_module_level=True)
    """, MODULE_SCOPE),
}

#: Handlers that are NOT this anti-pattern and must stay legal, or the rule
#: pushes authors towards a bare ``except`` with no guard at all.
BROAD_SKIP_ALLOWED = {
    "a narrow exception the environment really can raise": """
        def test_x():
            try:
                import optional_thing
            except ImportError as e:
                pytest.skip(str(e))
    """,
    "broad, but the failure survives": """
        def test_x():
            try:
                spacr.thing()
            except Exception:
                cleanup()
                raise
    """,
    "broad, but re-raised as a better message": """
        def test_x():
            try:
                spacr.thing()
            except Exception as e:
                raise AssertionError(f"spacr.thing() blew up: {e}") from e
    """,
    "a fixture swallowing its own teardown": """
        @pytest.fixture
        def _no_stray_figures():
            yield
            try:
                plt.close("all")
            except Exception:
                pass
    """,
}


@pytest.mark.parametrize("name", sorted(BROAD_SKIP_EVASIONS))
def test_the_broad_skip_rule_catches_every_known_evasion(name):
    """Each spelling that keeps the behaviour and dodges the old walk."""
    source, expected_scope = BROAD_SKIP_EVASIONS[name]
    sites = _synthetic_sites(source)
    assert expected_scope in sites, (
        f"the {name!r} spelling of except -> skip is invisible to "
        f"_scan_broad_skips; it found {sorted(sites)} and not "
        f"{expected_scope!r}")
    assert len(sites[expected_scope]) == 1


@pytest.mark.parametrize("name", sorted(BROAD_SKIP_ALLOWED))
def test_the_broad_skip_rule_leaves_honest_handlers_alone(name):
    """False positives would push authors towards no guard at all."""
    assert _synthetic_sites(BROAD_SKIP_ALLOWED[name]) == {}, (
        f"{name!r} is a legitimate handler and the rule flagged it")


def test_the_broad_skip_rule_reads_modules_that_are_not_named_test(tmp_path):
    """Moving the handler into a helper module must not retire it.

    Two halves, both needed. The classifier must not care what the file is
    called, and the collector must hand it files that are not called
    ``test_*``; either one alone leaves ``tests/helpers.py`` as a laundry.
    """
    # 1. The walk itself is name-blind: the same handler in a module named
    #    `helpers.py` is found exactly as it would be in a test file.
    helper = tmp_path / "helpers.py"
    helper.write_text(textwrap.dedent("""
        import pytest

        def load_the_thing():
            try:
                return spacr.thing()
            except Exception as e:
                pytest.skip(f"unavailable: {e}")
    """), encoding="utf-8")
    assert _scan_broad_skips(_parse(helper)) == [("load_the_thing", 7)]

    # 2. ...and the collector really does hand it such modules. tests/ holds
    #    plenty: conftest.py, resource_capabilities.py, synthetic_data.py.
    walked = {_rel(p) for p in _test_modules()}
    assert "conftest.py" in walked and "qt/conftest.py" in walked
    non_test = {f for f in walked if not Path(f).name.startswith("test_")}
    assert len(non_test) >= 3, (
        f"the module glob has narrowed to test_*.py; it found {sorted(walked)[:5]}")


def test_the_broad_skip_ratchet_does_not_survive_a_rename():
    """The allowance is keyed by file AND scope, and both keys are exact.

    Renaming the module, or moving the handler to a differently-named
    function inside it, must land on an allowance of zero -- and must also
    leave the vacated entry reporting slack, so the list cannot quietly keep
    room for a handler that has gone somewhere else.
    """
    ratchet = {"test_old_name.py": {"test_thing": 1}}
    same_place, slack = _ratchet_verdict(
        [("test_old_name.py", "test_thing", 10)], ratchet)
    assert not same_place and not slack, "the honest baseline is not clean"

    renamed_file, slack = _ratchet_verdict(
        [("test_new_name.py", "test_thing", 10)], ratchet)
    assert renamed_file == [("test_new_name.py", "test_thing", [10], 0)]
    assert slack == [("test_old_name.py", "test_thing", 0, 1)]

    renamed_scope, slack = _ratchet_verdict(
        [("test_old_name.py", "test_thing_again", 10)], ratchet)
    assert renamed_scope == [("test_old_name.py", "test_thing_again", [10], 0)]
    assert slack == [("test_old_name.py", "test_thing", 0, 1)]

    doubled, slack = _ratchet_verdict(
        [("test_old_name.py", "test_thing", 10),
         ("test_old_name.py", "test_thing", 14)], ratchet)
    assert doubled == [("test_old_name.py", "test_thing", [10, 14], 1)]
    assert not slack


# ---------------------------------------------------------------------------
# Rule 3 — no machine-specific absolute paths
# ---------------------------------------------------------------------------

#: A user home directory. Anything under one exists on exactly one machine, so
#: a test gated on it is green-by-default everywhere else. ``/tmp`` and paths
#: inside the repo are fine; those exist wherever the suite runs.
_USER_HOME_PATH = re.compile(r"^/(home|Users)/[^/\s]+/")

#: (module, reason) pairs allowed to mention a user-home path.
USER_HOME_PATH_RATCHET = {
    # A synthetic path whose point is the space in the directory name (URL
    # quoting); never touched on disk.
    "qt/test_space_theme.py",
}


def _string_constants(tree):
    """Every string literal in a module except the docstrings.

    Docstrings are prose -- a usage example may legitimately show
    ``/data/plate01`` -- while a literal in code is something the test uses.
    """
    docstring_nodes = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                docstring_nodes.add(id(node.body[0].value))
    return [n for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
            and id(n) not in docstring_nodes]


def test_no_user_home_paths_in_the_suite():
    """A path under someone's home directory is not a portable precondition."""
    offenders = []
    for path in _test_modules():
        rel = _rel(path)
        if rel in USER_HOME_PATH_RATCHET:
            continue
        for node in _string_constants(_parse(path)):
            for line in node.value.splitlines():
                if _USER_HOME_PATH.match(line.strip()):
                    offenders.append((rel, node.lineno, line.strip()[:70]))
                    break
    assert not offenders, (
        "these tests hard-code a path under a user's home directory:\n" +
        "\n".join(f"  {f}:{ln}  {t}" for f, ln, t in offenders) +
        "\n\nUse tmp_path, a fixture, or an environment variable. A test whose "
        "precondition only exists on one machine reports green everywhere else "
        "while running nothing."
    )


def _conftests():
    """Every conftest.py under tests/, nearest-the-root first.

    ``tests/conftest.py`` is not the only one: ``tests/qt/conftest.py`` is
    loaded for every Qt test in the suite and was never checked, so a stray
    absolute path there would have poisoned the whole Qt directory in silence.
    Any conftest added later is picked up automatically.
    """
    return sorted(TESTS_DIR.rglob("conftest.py"))


def test_conftest_hard_codes_no_absolute_path_at_all():
    """A conftest is loaded for EVERY test under it, so one stray path
    poisons the lot."""
    conftests = _conftests()
    assert {_rel(p) for p in conftests} >= {"conftest.py", "qt/conftest.py"}, (
        "the suite's known conftests are no longer being found; the glob in "
        "_conftests() has stopped matching")
    offenders = []
    for path in conftests:
        for node in _string_constants(_parse(path)):
            for line in node.value.splitlines():
                text = line.strip()
                if text.startswith("/") and len(text) > 1 \
                        and not text.startswith("/tmp"):
                    offenders.append((_rel(path), node.lineno, text[:70]))
    assert not offenders, (
        f"conftest(s) hard-code absolute path(s): {offenders}. Build "
        f"paths from tmp_path / tmp_path_factory or from the repo root.")


def test_the_e2e_dataset_paths_come_only_from_the_environment():
    """The real-dataset module must carry no built-in path at all.

    It used to *default* to a dataset under one developer's home directory and
    skip when that was absent, so on every other machine its four @slow stages
    reported green while running nothing. Env-var support alone was not enough
    to fix that -- the default is what made opting out invisible -- so the
    contract is now that the two variables are the only way in.
    """
    path = TESTS_DIR / "test_e2e_real_dataset.py"
    source = path.read_text(encoding="utf-8")
    assert "SPACR_E2E_DATA" in source
    assert "SPACR_E2E_SETTINGS" in source
    assert "os.environ.get" in source, (
        "the dataset path is not read from the environment, so the module can "
        "only ever run on the machine it was written on")
    # Docstrings are prose (the module's usage example shows a placeholder
    # path); a literal in code is a path the module would actually use.
    baked_in = [(n.lineno, n.value[:70])
                for n in _string_constants(_parse(path))
                if n.value.startswith("/") and len(n.value) > 1]
    assert not baked_in, (
        f"test_e2e_real_dataset.py bakes in absolute path(s): {baked_in}. A "
        "default path is what turns 'you have not opted in' into a silent "
        "pass; the environment variables must be the only source.")


# ---------------------------------------------------------------------------
# Rule 4 — the shared Cellpose mock contract actually rejects the bug
# ---------------------------------------------------------------------------
#
# tests/conftest.check_cellpose_eval_call is the guard the Cellpose mocks in
# this suite delegate to. A guard nobody tests is another way to be green for
# nothing, so it is exercised here against the exact call that survived
# fifteen tests and raised on every real run.


def test_the_cellpose_mock_contract_rejects_the_hardcoded_axis_3():
    """channel_axis=3 on a channels-last (H, W, C) image is the production bug.

    ``cellpose.transforms.convert_image`` indexes ``x.shape[channel_axis]``,
    and a 3-D array has no axis 3.
    """
    image = np.zeros((16, 16, 3), dtype=np.uint16)
    with pytest.raises(IndexError):
        check_cellpose_eval_call([image], 3)


def test_the_cellpose_mock_contract_rejects_an_axis_on_a_2d_image():
    """The other half of the same bug: a greyscale image takes no axis."""
    image = np.zeros((16, 16), dtype=np.uint16)
    with pytest.raises(ValueError, match="2D image"):
        check_cellpose_eval_call([image], -1)


def test_the_cellpose_mock_contract_accepts_what_spacr_actually_passes():
    """-1 on a channels-last stack, None on a 2-D image: both legal."""
    stack = np.zeros((16, 16, 2), dtype=np.uint16)
    converted = check_cellpose_eval_call([stack, stack], -1)
    assert len(converted) == 2
    # Cellpose pads to 3 channels; the spatial dims are untouched.
    assert converted[0].shape == (16, 16, 3)

    grey = np.zeros((16, 16), dtype=np.uint16)
    assert check_cellpose_eval_call(grey, None)[0].shape == (16, 16, 3)


def test_the_cellpose_mock_contract_notices_a_missing_channel_axis():
    """A mock that defaults the axis away is how channel_axis=3 got through."""
    stack = np.zeros((16, 16, 3), dtype=np.uint16)
    with pytest.raises(AssertionError, match="without channel_axis"):
        check_cellpose_eval_call([stack], MISSING_CHANNEL_AXIS)
    # ...and the sites that deliberately let Cellpose auto-detect opt out.
    check_cellpose_eval_call([stack], MISSING_CHANNEL_AXIS,
                             require_channel_axis=False)


#: CellposeModel doubles whose ``eval`` still absorbs ``channel_axis`` into
#: ``**kwargs``, and how many such ``eval`` methods each class holds. Seeded
#: from the modules outside the scope of the change that introduced this
#: contract; the list may only shrink. The fix is three lines per mock: name
#: the parameter, pass it to ``check_cellpose_eval_call``, and put the value
#: back into whatever the test records.
#:
#: The counts exist because a (file, class) key is not unique: a module can
#: define two classes with the same name in two different fixtures, and
#: ``test_cov_object_cellpose_masks.py`` defines ``_M`` twice.
#:
#: Six of these entries were invisible until the candidate filter stopped
#: reading class names. ``_M``, ``_RecordingCP`` and ``_FakeCP`` contain
#: neither "cellpose" nor "model", so the old ``if "cellpose" not in name and
#: "model" not in name: continue`` walked straight past them -- while the
#: docstring claimed the rule recognised a double by its method shape.
#: EMPTY, and it stays empty. Every one of the fourteen ``eval`` doubles that
#: used to live here now declares the installed ``CellposeModel.eval``
#: signature in full -- real parameter names, real defaults, no ``**kwargs`` --
#: so the argument list is enforced by Python's own binding rather than by this
#: rule. ``tests/test_cellpose_api_contract.py`` is the stronger successor: it
#: checks every double against ``inspect.signature`` of the installed cellpose
#: and fails if one drifts.
CELLPOSE_MOCK_RATCHET = {}

#: Total offending ``eval`` methods above, not keys.
CELLPOSE_MOCK_CEILING = 0


#: Sentinel for "this default is not a python literal" (a Name like
#: ``MISSING_CHANNEL_AXIS``), which is exactly what a compliant mock uses.
_NOT_A_LITERAL = object()


def _param_default(args, name):
    """``(has_default, default_node)`` for parameter ``name`` of ``args``."""
    positional = args.posonlyargs + args.args
    names = [a.arg for a in positional]
    if name in names:
        offset = len(names) - len(args.defaults)
        index = names.index(name)
        if index >= offset:
            return True, args.defaults[index - offset]
        return False, None
    for kwarg, default in zip(args.kwonlyargs, args.kw_defaults):
        if kwarg.arg == name:
            return default is not None, default
    return False, None


def _channel_axis_complaint(fn):
    """Why ``fn`` (an ``eval`` method) fails the channel_axis contract, or None.

    Three ways to fail, all of which leave the mock unable to tell a working
    call from the ``channel_axis=3`` that raised on every real run:

    1. the parameter is not named at all, so ``**kwargs`` eats it;
    2. it is named but DEFAULTED to a value a caller could legally pass
       (``None``, or an axis index). Then "the caller omitted it" and "the
       caller passed it" are the same state, which is exactly the hole
       ``MISSING_CHANNEL_AXIS`` exists to close;
    3. it is named and never read. ``def eval(self, x, channel_axis=None,
       **kwargs)`` that ignores the value satisfies a rule that only checks
       the signature -- proved against the previous version of this rule --
       and validates nothing.
    """
    args = fn.args
    named = {a.arg for a in args.posonlyargs + args.args + args.kwonlyargs}
    if "channel_axis" not in named:
        return "swallows channel_axis into **kwargs"
    has_default, default = _param_default(args, "channel_axis")
    if has_default:
        try:
            # literal_eval, not `isinstance(default, ast.Constant)`: `-1` is a
            # UnaryOp(USub, Constant(1)), and -1 is the single most important
            # legal axis in this codebase to reject as a default.
            literal = ast.literal_eval(default)
        except (ValueError, SyntaxError, TypeError):
            literal = _NOT_A_LITERAL
        if literal is None or isinstance(literal, (int, float)):
            return (f"defaults channel_axis to {ast.unparse(default)}, which a "
                    f"real caller could pass -- use a sentinel like "
                    f"MISSING_CHANNEL_AXIS")
    used = any(isinstance(n, ast.Name) and n.id == "channel_axis"
               and isinstance(n.ctx, ast.Load)
               for n in ast.walk(fn))
    if not used:
        return "names channel_axis and never reads it"
    return None


def _cellpose_mock_offenders():
    """``(file, class, lineno, complaint)`` per non-compliant ``eval``.

    A candidate is any method named ``eval`` that takes ``**kwargs`` -- the
    method SHAPE, with no reference to the class's name. That is what the
    rule's docstring has always promised and what it did not do.
    """
    offenders = []
    for path in _test_modules():
        tree = _parse(path)
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                if fn.name != "eval" or fn.args.kwarg is None:
                    continue      # no **kwargs: nothing is being swallowed
                complaint = _channel_axis_complaint(fn)
                if complaint:
                    offenders.append((_rel(path), cls.name, fn.lineno,
                                      complaint))
    return offenders


def test_the_cellpose_mocks_do_not_swallow_channel_axis():
    """Every CellposeModel stand-in names ``channel_axis`` on ``eval``,
    defaults it to a sentinel, and actually reads it.

    A mock spelled ``def eval(self, x, **kwargs)`` accepts every argument
    including the illegal ones, which is the mechanism -- not the symptom --
    behind the channel_axis=3 escape. Recognising the double by its method
    shape rather than its class name keeps this honest as new mocks appear.
    """
    found = {}
    for f, cls, lineno, complaint in _cellpose_mock_offenders():
        found.setdefault((f, cls), []).append((lineno, complaint))
    over_budget = []
    for key, hits in sorted(found.items()):
        allowed = CELLPOSE_MOCK_RATCHET.get(key, 0)
        if len(hits) > allowed:
            over_budget.append((key, sorted(hits), allowed))
    assert not over_budget, (
        "these CellposeModel doubles do not honour the channel_axis "
        "contract:\n" +
        "\n".join(f"  {f}: {c}.eval() at line(s) "
                  + ", ".join(f"{ln} ({why})" for ln, why in hits)
                  + f" -- ratchet allows {allowed}"
                  for (f, c), hits, allowed in over_budget) +
        "\n\nName it: `def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, "
        "**kwargs)` and hand the pair to check_cellpose_eval_call (see "
        "tests/conftest.py). A mock that accepts any axis cannot tell a "
        "working call from the one that crashed every real run."
    )


def test_the_cellpose_mock_ratchet_only_shrinks():
    total = sum(CELLPOSE_MOCK_RATCHET.values())
    assert total <= CELLPOSE_MOCK_CEILING, (
        f"CELLPOSE_MOCK_RATCHET grew to {total} eval methods (ceiling "
        f"{CELLPOSE_MOCK_CEILING}). Entries come off this list, never on.")
    missing = sorted({f for f, _ in CELLPOSE_MOCK_RATCHET
                      if not (TESTS_DIR / f).is_file()})
    assert not missing, (
        f"CELLPOSE_MOCK_RATCHET names modules that no longer exist: {missing}")


def test_the_cellpose_mock_rule_rejects_a_declared_but_ignored_axis():
    """The rule's own blind spots, pinned against synthesised mocks.

    Written as source rather than as live classes on purpose: adding a
    deliberately-broken CellposeModel double to the suite would be a mock
    other tests could pick up.
    """
    def complaint(src):
        cls = ast.parse(textwrap.dedent(src)).body[0]
        return _channel_axis_complaint(cls.body[0])

    # 1. Not named at all -- the original rule caught this one.
    assert "swallows" in complaint("""
        class _Double:
            def eval(self, x, **kwargs):
                return [], None
    """)
    # 2. Named, defaulted to a legal value, and ignored. This PASSED the
    #    previous rule, which only checked that the name appeared.
    assert "defaults channel_axis" in complaint("""
        class _Double:
            def eval(self, x, channel_axis=None, **kwargs):
                return [], None
    """)
    assert "defaults channel_axis" in complaint("""
        class _Double:
            def eval(self, x, channel_axis=-1, **kwargs):
                return [], None
    """)
    # 3. Named with a proper sentinel, but the value is never read.
    assert "never reads it" in complaint("""
        class _Double:
            def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
                return [], None
    """)
    # ...and the shape the suite's compliant mocks actually use passes.
    assert complaint("""
        class _Double:
            def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
                check_cellpose_eval_call(x, channel_axis)
                return [], None
    """) is None


def test_the_cellpose_mock_rule_does_not_read_class_names():
    """The detector matches on method SHAPE, never on the class's name.

    Six doubles were once invisible to this rule because they are not called
    ``*Model``: ``_M``, ``_RecordingCP`` and ``_FakeCP`` are CellposeModel
    stand-ins by behaviour and by nothing else, and a name filter cannot see
    them. All six have since been fixed, so the guarantee is pinned against
    synthesised source rather than against a broken mock the suite would
    otherwise have to keep shipping for the test's benefit.
    """
    def offenders(src):
        tree = ast.parse(textwrap.dedent(src))
        found = []
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            for fn in [n for n in cls.body
                       if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
                if fn.name != "eval" or fn.args.kwarg is None:
                    continue
                if _channel_axis_complaint(fn):
                    found.append(cls.name)
        return found

    # Three names with nothing cellpose-ish about them, all caught.
    assert offenders("""
        class _M:
            def eval(self, x, **kwargs):
                return [], None
        class _RecordingCP:
            def eval(self, x, **kwargs):
                return [], None
        class Wibble:
            def eval(self, x, **kwargs):
                return [], None
    """) == ["_M", "_RecordingCP", "Wibble"]

    # ...and a class NAMED like a cellpose model still passes when its eval is
    # compliant, so the rule is not reading the name in the other direction.
    assert offenders("""
        class CellposeModel:
            def eval(self, x, channel_axis=MISSING_CHANNEL_AXIS, **kwargs):
                check_cellpose_eval_call(x, channel_axis)
                return [], None
    """) == []


def test_the_cellpose_mock_ratchet_is_empty_and_stays_that_way():
    """The ``**kwargs`` doubles are gone; nothing may re-add one.

    ``CELLPOSE_MOCK_RATCHET`` is the record of doubles that swallowed
    ``channel_axis``. It reached zero when every double was rewritten to
    declare the installed ``CellposeModel.eval`` signature in full. An empty
    ratchet with a zero ceiling is what makes the next ``**kwargs`` double a
    failure instead of a new entry.
    """
    assert CELLPOSE_MOCK_RATCHET == {}
    assert CELLPOSE_MOCK_CEILING == 0
    assert not _cellpose_mock_offenders()


# ---------------------------------------------------------------------------
# 4. A DataFrame subclass that only overrides `_constructor`.
#
# This one only fails on the OLDEST pandas spaCR supports, which is what made
# it worth a rule. A subclass that overrides `_constructor` and nothing else
# sends pandas down `self._constructor(mgr)` on every internal
# reconstruction. pandas 2.2 DeprecationWarns there ("Passing a BlockManager
# to <Subclass> is deprecated"); pandas 2.3 stopped. setup.py declares
# `pandas>=2.2.1`, the "Minimum dependencies" CI job installs exactly that,
# and it turns warnings into errors -- so
# `test_cov_ml_shap_vision.py::test_calculate_similarity_reports_and_returns_
# on_assignment_failure` failed on that one job and passed everywhere else,
# looking like a spacr.ml defect and being a property of the fixture.
#
# `_constructor_from_mgr` exists on pandas 2.2 and 2.3 alike and preserves
# the subclass on both, so overriding it is not a workaround for an old
# pandas -- it is the correct way to subclass a DataFrame across the declared
# range.
# ---------------------------------------------------------------------------

def _dataframe_subclass_offenders(modules=None):
    """(file, class) for every tests/ DataFrame subclass missing the hook."""
    def _label(path):
        try:
            return _rel(path)
        except ValueError:      # a synthetic module outside tests/
            return str(path)

    offenders = []
    for path in (modules if modules is not None else _test_modules()):
        try:
            tree = _parse(path)
        except SyntaxError:                     # pragma: no cover - unparsable
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            bases = {
                base.attr if isinstance(base, ast.Attribute)
                else getattr(base, "id", "")
                for base in node.bases
            }
            if "DataFrame" not in bases:
                continue
            defined = {
                child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if "_constructor" in defined and "_constructor_from_mgr" not in defined:
                offenders.append((_label(path), node.name))
    return sorted(offenders)


def test_no_dataframe_subclass_overrides_constructor_alone():
    """Overriding `_constructor` without `_constructor_from_mgr` is a
    DeprecationWarning on the declared pandas floor and an error in the job
    that installs it."""
    offenders = _dataframe_subclass_offenders()
    assert offenders == [], (
        "these tests/ DataFrame subclasses override `_constructor` but not "
        "`_constructor_from_mgr`, which DeprecationWarns on pandas 2.2 (the "
        "floor setup.py declares) and fails the min-deps job:\n" +
        "\n".join(f"  {f}::{c}" for f, c in offenders)
    )


def test_the_dataframe_subclass_rule_discriminates(tmp_path):
    """The rule catches the real shape and leaves the fixed one alone.

    Without this, `offenders == []` above would also pass if the detector
    matched nothing at all.
    """
    bad = tmp_path / "test_bad.py"
    bad.write_text(textwrap.dedent("""
        import pandas as pd

        class Sub(pd.DataFrame):
            @property
            def _constructor(self):
                return Sub
    """), encoding="utf-8")
    good = tmp_path / "test_good.py"
    good.write_text(textwrap.dedent("""
        import pandas as pd

        class Sub(pd.DataFrame):
            @property
            def _constructor(self):
                return Sub

            def _constructor_from_mgr(self, mgr, axes):
                return Sub._from_mgr(mgr, axes=axes)
    """), encoding="utf-8")

    found = {cls for _, cls in _dataframe_subclass_offenders([bad])}
    assert found == {"Sub"}
    assert _dataframe_subclass_offenders([good]) == []
