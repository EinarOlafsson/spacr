"""The rows ``expected_types`` cannot see are a SET, pinned by name.

Instruction 397. Seventy-seven rows of ``SETTING_API_TARGETS`` name a key that
is not in ``spacr.settings.expected_types`` on a fresh import, and the item
asked for the remainder to be "a named, committed list with a reason each, so
the next person measuring stranded rows can subtract a set rather than a
number".

A COUNT IN A NOTE IS NOT A PIN. Seventy-seven rows leaving and seventy-seven
other rows arriving reads identically from a total, which is the failure 364
found: a rename strands a generated row while every number stays where it was.
So the pin is ``tools.build_setting_consumer_map.ABSENT_FROM_EXPECTED_TYPES``,
the members are named, and the first test below reports ARRIVED and LEFT
separately -- a swap of one for one fails here and says which two.

WHAT THE MEASUREMENT DEPENDS ON, because on this item every wrong answer so
far came from an instrument rather than from the table:

  * WHEN. Fifty-one of the seventy-seven are in ``expected_types`` once the
    module that registers them has been imported -- ``register_defaults`` runs
    from a module body (HANDOFF.md trap 3c). ``import spacr.settings`` alone
    sees none of them, so the set is measured in a CHILD INTERPRETER: in a
    pytest process any earlier test that imported ``spacr.convert`` would make
    this one pass or fail on test ordering.
  * WHICH TABLE. Twenty-one are in the module-scope ``tooltips`` and in no
    type table. The note on 397 that dismissed this class -- "ALL 77 have a
    tooltip, so tooltip presence cannot discriminate" -- was reading the EN
    CATALOG's tooltips, which every row has by construction because the
    catalog is unioned into the generator's vocabulary.
  * WHAT COUNTS AS A SETTING. Three are app names from
    ``spacr.settings.descriptions``, carried in the catalog's
    ``SETTING_TOOLTIPS`` byte for byte beside the real per-setting prose.
  * WHICH CONTAINERS A READ-SEARCH RECOGNISES. The last test names them in
    its own failure message, per the rule 397 arrived at after calling two
    live settings dead: a count of unread settings without that sentence
    attached is a property of the grep.
"""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
from tests.child_env import child_env                       # noqa: E402

generator = importlib.import_module("build_setting_consumer_map")


#: Runs in a CHILD interpreter so the snapshot is taken before anything has
#: imported a module that registers defaults. It prints one ``@@``-prefixed
#: JSON line; anything a spaCR import writes to stdout is ignored.
#:
#: The Qt layer is left out of the walk deliberately -- it is the DISPLAY side
#: that this map excludes by design, and it needs a platform plugin. Including
#: it was measured on 2026-09-14 and moves nothing: the same twenty-six keys
#: remain absent either way.
_PROBE = r'''
import ast, importlib, json, pathlib, pkgutil, sys, warnings
warnings.filterwarnings("ignore")

root = pathlib.Path(sys.argv[1])
table = root / "spacr" / "qt" / "screens" / "setting_api_targets.py"
targets = {}
for node in ast.parse(table.read_text(encoding="utf-8")).body:
    if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "SETTING_API_TARGETS" for t in node.targets):
        targets = ast.literal_eval(node.value)

import spacr
import spacr.settings as S

absent = sorted(k for k in targets if k not in S.expected_types)
descriptions = set(S.descriptions)

failed = []
for mod in pkgutil.walk_packages(spacr.__path__, "spacr."):
    if ".qt" in mod.name:
        continue
    try:
        importlib.import_module(mod.name)
    except BaseException as exc:
        failed.append("%s: %r" % (mod.name, exc))

facts = {k: {"descriptions": k in descriptions,
             "expected_types": k in S.expected_types,
             "tooltips": k in S.tooltips} for k in absent}

try:
    from spacr.qt.screens.setting_api_targets import SETTING_API_TARGETS as live
    runtime = "same" if {k: tuple(v) for k, v in live.items()} == targets else "differs"
except BaseException as exc:
    runtime = "import failed: %r" % (exc,)

print("@@" + json.dumps({"rows": len(targets), "absent": absent,
                         "facts": facts, "runtime_table": runtime,
                         "import_failures": failed,
                         "targets": {k: list(v) for k, v in targets.items()}}))
'''


@pytest.fixture(scope="module")
def measured():
    """The live answer, from an interpreter that has imported nothing yet."""
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, str(ROOT)],
        capture_output=True, text=True, timeout=900,
        env=child_env(pythonpath=str(ROOT)),
    )
    assert proc.returncode == 0, (
        f"the probe interpreter failed:\n{proc.stdout[-2000:]}\n"
        f"{proc.stderr[-4000:]}")
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("@@")]
    assert lines, f"probe printed no result:\n{proc.stdout[-2000:]}"
    return json.loads(lines[-1][2:])


def test_the_pinned_set_is_exactly_the_rows_expected_types_cannot_see(measured):
    """The pin, by SET DIFFERENCE. A one-for-one swap must not read as zero."""
    pinned = set(generator.pinned_absences())
    live = set(measured["absent"])
    arrived = sorted(live - pinned)
    left = sorted(pinned - live)
    # WHICH INSTRUMENT, before blaming the table: fifty-one of the pinned
    # keys are only typed once the module that registers them imports, so a
    # module that failed to import in the probe makes one look ARRIVED for an
    # environmental reason. The probe swallows those failures to stay quiet on
    # optional dependencies; it must not swallow them silently HERE.
    failures = measured.get("import_failures") or []
    assert not (arrived or left), (
        "the rows absent from spacr.settings.expected_types have changed.\n"
        f"  ARRIVED ({len(arrived)}): {arrived}\n"
        f"  LEFT    ({len(left)}): {left}\n"
        "Neither is automatically a defect -- read the causes in "
        "tools/build_setting_consumer_map.py -- but each member has to be "
        "explained and added to, or removed from, "
        "ABSENT_FROM_EXPECTED_TYPES. Do not move a total."
        + (f"\nREAD THIS FIRST: {len(failures)} spacr module(s) failed to "
           f"import in the probe, and a key registered by one of them looks "
           f"ARRIVED here for that reason alone: {failures[:5]}"
           if failures else ""))


def test_every_pinned_row_still_has_the_cause_it_was_pinned_with(measured):
    """A row that changes CAUSE is invisible to a set check.

    ``layout`` moving out of ``late_registration`` because its module stopped
    registering it would leave the set identical and the explanation wrong.
    """
    pinned = generator.pinned_absences()
    wrong = {}
    for key, cause in sorted(pinned.items()):
        facts = measured["facts"].get(key)
        if facts is None:
            continue                      # absent from the set: the test above
        derived = generator.classify_absence(
            key,
            descriptions={key} if facts["descriptions"] else set(),
            expected_types={key} if facts["expected_types"] else set(),
            tooltips={key} if facts["tooltips"] else set(),
        )
        if derived != cause:
            wrong[key] = f"pinned {cause!r}, measured {derived!r}"
    assert not wrong, (
        "pinned rows whose reason for being absent has changed:\n  "
        + "\n  ".join(f"{k}: {v}" for k, v in wrong.items()))


def test_the_causes_are_a_closed_vocabulary_and_no_row_has_two(measured):
    """A cause nobody wrote down is a number again, under a longer name."""
    groups = generator.ABSENT_FROM_EXPECTED_TYPES
    assert set(groups) == set(generator.ABSENCE_CAUSES), (
        "ABSENT_FROM_EXPECTED_TYPES and ABSENCE_CAUSES name different causes: "
        f"{sorted(set(groups) ^ set(generator.ABSENCE_CAUSES))}")
    seen, twice = set(), []
    for cause, keys in groups.items():
        for key in keys:
            if key in seen:
                twice.append((key, cause))
            seen.add(key)
    assert not twice, f"pinned under more than one cause: {twice}"
    assert len(seen) == sum(len(v) for v in groups.values())


def test_the_unexplained_rows_each_carry_a_reason(measured):
    """Every ``catalog_only`` member carries prose of its own.

    It is the one group that is not a property of the instrument, so a group
    label does not explain it. This is what "a named, committed list with a
    reason each" asked for.
    """
    catalog_only = set(generator.ABSENT_FROM_EXPECTED_TYPES["catalog_only"])
    assert set(generator.CATALOG_ONLY_NOTES) == catalog_only, (
        "CATALOG_ONLY_NOTES and the catalog_only group disagree: "
        f"{sorted(set(generator.CATALOG_ONLY_NOTES) ^ catalog_only)}")
    thin = {k: v for k, v in generator.CATALOG_ONLY_NOTES.items()
            if len(v.split()) < 20}
    assert not thin, f"a reason that explains nothing: {sorted(thin)}"


def test_the_duplicate_row_is_still_degenerate_and_still_the_only_dotted_one(
        measured):
    """The anomaly 397 recorded rather than removed, pinned so a fix is seen.

    ``umap.reduction_method`` renders nothing -- empty symbol, so no anchor is
    emitted -- while the bare ``reduction_method`` carries the working row.
    Removing the dotted row is a change to a GENERATED table and belongs with
    the generator's next pass; when that happens this test and the set above
    both fail, which is the point.
    """
    targets = measured["targets"]
    dotted = sorted(k for k in targets if "." in k)
    assert dotted == ["umap.reduction_method"], (
        f"the dotted rows are no longer just the one: {dotted}")
    module, symbol, exact = targets["umap.reduction_method"]
    assert (symbol, exact) == ("", False), (
        f"the duplicate row now renders something: {(module, symbol, exact)}")
    bare_module, bare_symbol, bare_exact = targets["reduction_method"]
    assert bare_exact and bare_symbol, (
        "the bare reduction_method row is the one that works; it has stopped: "
        f"{(bare_module, bare_symbol, bare_exact)}")


#: Containers a read search recognises. ``generator.SETTINGS_NAMES`` is what
#: the map itself accepts; ``resolved`` is added because four pipeline entry
#: points open with ``resolved = default_settings(settings)`` and 397 called
#: two live settings dead for want of it.
_CONTAINERS = tuple(sorted(generator.SETTINGS_NAMES | {"resolved", "s", "out"}))


def _read_sites(key: str) -> list[str]:
    """Where ``key`` is read out of a settings mapping, container named."""
    names = "|".join(re.escape(n) for n in _CONTAINERS)
    quoted = f"['\"]{re.escape(key)}['\"]"
    pattern = re.compile(
        rf"\b({names})\s*(?:\[\s*{quoted}\s*\]"
        rf"|\.\s*(?:get|setdefault|pop)\s*\(\s*{quoted})")
    sites = []
    for path in sorted((ROOT / "spacr").rglob("*.py")):
        if "i18n_catalogs" in path.parts or path.name == "setting_api_targets.py":
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line):
                sites.append(f"{path.relative_to(ROOT)}:{number}")
    return sites


def test_the_catalog_only_rows_are_read_exactly_where_the_pin_says(measured):
    """One of the two is a live flag and one is read by nothing.

    Both directions are asserted, because a search that finds everything and a
    search that finds nothing both agree with a one-sided test.
    """
    catalog_only = generator.ABSENT_FROM_EXPECTED_TYPES["catalog_only"]
    unread = set(generator.CATALOG_ONLY_UNREAD)
    assert unread <= set(catalog_only), (
        f"CATALOG_ONLY_UNREAD names something not pinned: {sorted(unread)}")
    problems = []
    for key in catalog_only:
        sites = _read_sites(key)
        if key in unread and sites:
            problems.append(f"{key} is pinned as read by nothing, but "
                            f"{sites[:3]} read it")
        if key not in unread and not sites:
            problems.append(f"{key} is pinned as a live setting and no read "
                            f"of it was found")
    assert not problems, (
        "\n".join(problems)
        + "\n\nThe search recognised these containers only: "
        + ", ".join(_CONTAINERS)
        + " -- a setting read out of a mapping under any other local name "
          "looks unread to it.")


def test_the_parsed_table_is_the_one_the_gui_imports(measured):
    """The pin is parsed from the file; the panels import it. Same dict."""
    assert measured["runtime_table"] == "same", measured["runtime_table"]
