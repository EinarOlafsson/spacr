"""A deliberately dispatched CI run must survive the next push.

Instruction 101. ``concurrency.cancel-in-progress`` is the right policy for a
branch that receives pushes faster than a suite can finish -- only the newest
commit is worth testing -- and it is the wrong policy for the one run somebody
asked for on purpose. On 2026-08-15 the last thirty ``tests`` runs on nightly
were twenty-nine cancellations and one still running: a full suite takes ~95
minutes and the gap between pushes to nightly is 5-25, so every run was killed
by the next commit and the integration branch had no test signal at all. Not a
degraded one -- none.

The repair splits the concurrency group by ``github.event_name`` and exempts
``workflow_dispatch`` from cancellation, so ``gh workflow run tests --ref
nightly`` starts a run the next twenty commits cannot kill, while a push still
cancels the previous PUSH run and the 5 August runner starvation stays fixed.

CI cannot be run from here, so this file asserts the property off the workflow
YAML instead: it evaluates the real ``group`` and ``cancel-in-progress``
expressions for each event and checks the two things that matter -- a push
cannot reach a dispatched run, and a push still cancels its own predecessor.
Asserting on the evaluated expressions rather than on the literal text means a
different but equally correct spelling passes, and a rewrite that quietly drops
the exemption fails.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO_ROOT / ".github" / "workflows"

#: A ref to substitute for ``github.ref`` while evaluating a group template.
#: Any value works -- the assertions compare groups to each other, never to a
#: literal -- but nightly is the branch the defect was measured on.
_REF = "refs/heads/nightly"

#: Workflows whose in-progress runs may be cancelled, and which therefore need
#: the dispatch exemption. Suites that never cancel anything are checked
#: separately by :func:`test_the_deploying_workflows_cancel_nothing_at_all`.
_CANCELLING = ("tests.yml", "compat-matrix.yml")


def _load(name):
    """Parse a workflow, returning ``(triggers, concurrency)``.

    PyYAML follows YAML 1.1, where the bare key ``on`` is the boolean
    ``True``. Reading triggers by that key rather than by the string ``"on"``
    is not a workaround for a broken file -- the file is correct GitHub
    Actions YAML -- so both spellings are accepted.
    """
    document = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    triggers = document.get("on", document.get(True))
    return triggers, document.get("concurrency")


def _substitute(template, event):
    """Resolve ``${{ ... }}`` placeholders in a concurrency ``group``."""
    resolved = str(template)
    resolved = resolved.replace("${{ github.ref }}", _REF)
    resolved = resolved.replace("${{ github.event_name }}", event)
    return resolved


def _cancels(expression, event):
    """Evaluate a ``cancel-in-progress`` value for ``event``.

    The value is either a plain YAML boolean or a GitHub expression. Only the
    operators GitHub's expression syntax actually offers here are supported,
    and anything outside that vocabulary raises rather than being guessed at
    -- a guard that silently mis-parses a rewritten expression would be worse
    than no guard.
    """
    if isinstance(expression, bool):
        return expression

    body = str(expression).strip()
    match = re.fullmatch(r"\$\{\{(.+)\}\}", body, flags=re.DOTALL)
    assert match, f"not a literal or a ${{{{ }}}} expression: {expression!r}"
    body = match.group(1).strip()

    body = body.replace("github.event_name", repr(event))
    body = body.replace("&&", " and ").replace("||", " or ")
    # ``!`` is negation, but ``!=`` is a comparison: only negate a ``!`` that
    # is not part of an operator.
    body = re.sub(r"(?<![!=<>])!(?!=)", " not ", body)
    body = re.sub(r"\btrue\b", "True", body)
    body = re.sub(r"\bfalse\b", "False", body)

    # Whitelist the characters a comparison of literals can contain, so an
    # expression using anything richer (a function call, a context lookup)
    # stops this helper rather than being evaluated as Python and believed.
    assert re.fullmatch(r"[\w\s'\"()=!<>.]+", body), (
        f"unsupported expression, teach this helper before using it: {body!r}"
    )
    return bool(eval(body, {"__builtins__": {}}, {}))  # noqa: S307


@pytest.mark.parametrize("name", _CANCELLING)
def test_a_push_cannot_cancel_a_dispatched_run(name):
    """The whole of instruction 101.

    A run is cancelled only by a newer run in the SAME concurrency group, so
    a dispatched run is safe exactly when a push lands in a different group.
    """
    _triggers, concurrency = _load(name)
    assert concurrency, f"{name} declares no concurrency group"

    push = _substitute(concurrency["group"], "push")
    dispatched = _substitute(concurrency["group"], "workflow_dispatch")

    assert push != dispatched, (
        f"{name}: a push and a dispatched run share the concurrency group "
        f"{push!r}, so the next commit cancels the run somebody asked for. "
        f"Split the group by github.event_name."
    )


@pytest.mark.parametrize("name", _CANCELLING)
def test_a_dispatched_run_is_never_cancelled_by_another(name):
    """Two dispatched runs must not kill each other either.

    Group isolation alone would still let a second ``gh workflow run`` cancel
    the first, which is the same lost verdict by a different route.
    """
    _triggers, concurrency = _load(name)

    assert _cancels(concurrency["cancel-in-progress"], "workflow_dispatch") is False, (
        f"{name}: a dispatched run is still cancellable, so the run a release "
        f"reads its verdict from can be destroyed before it finishes."
    )


@pytest.mark.parametrize("name", _CANCELLING)
def test_a_push_still_cancels_the_previous_push(name):
    """The 5 August protection, unchanged.

    Ten tutorial-rebuild pushes left ~30 jobs queued for hours and starved
    every runner. Exempting pushes -- the obvious fix, and the wrong one --
    would recreate that outage, so the exemption must be narrow.
    """
    _triggers, concurrency = _load(name)

    assert _cancels(concurrency["cancel-in-progress"], "push") is True, (
        f"{name}: a push no longer cancels the previous push run. That is the "
        f"5 August runner starvation, not a fix for instruction 101."
    )
    assert _substitute(concurrency["group"], "push") == _substitute(
        concurrency["group"], "push"
    )


@pytest.mark.parametrize("name", _CANCELLING)
def test_the_dispatch_escape_hatch_actually_exists(name):
    """An exemption for an event the workflow does not accept is decoration.

    ``schedule:`` is not a substitute: GitHub runs cron from the DEFAULT
    branch, so a scheduled run tests ``main`` and says nothing about nightly.
    ``workflow_dispatch`` can target any ref, which is why it is the hatch.
    """
    triggers, _concurrency = _load(name)

    assert "workflow_dispatch" in triggers, (
        f"{name}: nothing can be dispatched, so the cancellation exemption "
        f"for workflow_dispatch can never apply."
    )


@pytest.mark.parametrize("name", _CANCELLING)
def test_the_group_is_still_per_ref(name):
    """Splitting by event must not have merged the branches together."""
    _triggers, concurrency = _load(name)

    assert "github.ref" in str(concurrency["group"]), (
        f"{name}: the concurrency group no longer varies by ref, so a push to "
        f"one branch cancels the run of another."
    )


def test_the_deploying_workflows_cancel_nothing_at_all():
    """``docs`` and ``release`` queue instead of cancelling.

    They need no dispatch exemption because they have nothing to be exempt
    from -- but if either ever turns cancellation on, it inherits this
    instruction's defect and must be added to ``_CANCELLING``.
    """
    for name in ("docs.yml", "release.yml"):
        _triggers, concurrency = _load(name)
        assert _cancels(concurrency["cancel-in-progress"], "push") is False, (
            f"{name} now cancels in-progress runs; add it to _CANCELLING so "
            f"the dispatch exemption is enforced for it too."
        )


def test_every_workflow_that_cancels_is_covered_by_this_file():
    """The list above cannot silently fall behind a new workflow."""
    cancelling = set()
    for path in sorted(WORKFLOWS.glob("*.yml")):
        _triggers, concurrency = _load(path.name)
        if not concurrency:
            continue
        value = concurrency.get("cancel-in-progress", False)
        # A run that is cancellable under ANY event needs the exemption.
        if _cancels(value, "push") or _cancels(value, "workflow_dispatch"):
            cancelling.add(path.name)

    assert cancelling == set(_CANCELLING), (
        f"workflows that cancel in-progress runs are {sorted(cancelling)}, but "
        f"this file guards {sorted(_CANCELLING)}. A workflow that cancels and "
        f"is not listed can cancel its own dispatched run."
    )
