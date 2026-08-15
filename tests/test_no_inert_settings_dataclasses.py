"""A settings dataclass field nothing reads is a control that lies.

`tests/test_dead_settings.py` guards the pipeline settings in
`spacr/settings.py` and is thorough about it. It does not look at the
SETTINGS DATACLASSES the Qt screens use, and that gap is not theoretical:
seven of `GateEditorSettings`'s 35 fields had zero readers outside their own
module on 2026-08-11, and four of them produced wrong answers rather than no
answer.

    cluster_eps, cluster_min_samples, cluster_scale
        The cluster dialog opened on its own hardcoded 0.30/10 while these
        said 0.5/20. eps decides how many populations DBSCAN finds, so a
        user who set 0.5 got a clustering computed at 0.30 with nothing on
        screen to say their value had been dropped.

    cluster_walk, cluster_walk_steps
        Read by nothing anywhere -- editable, saved, reloaded, inert.

    voxel_bins, snap_to_axis, spin_speed
        The 3D workspace they configure does not exist yet.

This test re-derives the answer from the source every run, so it rots in
neither direction: a field that gains a reader must leave the allowlist, and
a field that loses its last reader must join it or be deleted.

WHAT IT CAN AND CANNOT SEE. "Reader" means the field name appears in some
other module under `spacr/`. That is deliberately generous -- it will count
a mention that merely passes the value along -- so a field this test calls
dead really is dead. It cannot see a field that is read but ignored, which
is a different defect and needs its own test.
"""
from __future__ import annotations

import ast
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[1]
PACKAGE = REPO / "spacr"


#: Fields that are declared on purpose before the feature that reads them.
#: Each entry needs the instruction that will wire it -- "we will get to it"
#: is how a phantom setting gets old.
#:
#: These three configure the 3D gating workspace instruction 52 builds.
#: `GateEditorPanel.apply_settings` says so in its own docstring: "the 3D
#: ones belong to a workspace that does not exist yet". They are kept rather
#: than deleted because 52 is what will read them, and `snap_to_axis` in
#: particular encodes a real decision -- a 3D gate is finally read square-on,
#: so the shape drawn is the shape stored.
#: Settings declared ahead of the code that reads them, with the instruction
#: that will read them. EMPTY, and that is the point: instruction 52 wired
#: voxel_bins, snap_to_axis and spin_speed, so all three came off this list.
#: A control that turns nothing is a promise the application does not keep,
#: and this dict is where that promise is tracked until it is met.
KNOWN_UNREAD = {}


def _is_dataclass(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return True
        if isinstance(dec, ast.Call) and getattr(dec.func, "id", "") == "dataclass":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
            return True
    return False


def _settings_fields():
    """Every ``(module, class, field)`` on a settings dataclass."""
    found = []
    for path in sorted(PACKAGE.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:                        # pragma: no cover
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef) or not _is_dataclass(node):
                continue
            if not node.name.endswith("Settings"):
                continue
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    found.append((path, node.name, stmt.target.id))
    return found


FIELDS = _settings_fields()


def test_the_scan_finds_the_dataclasses_it_is_supposed_to_guard():
    """A scan that silently matched nothing would pass every test below."""
    assert FIELDS, "no settings dataclasses found -- the scan is broken"
    classes = {cls for _, cls, _ in FIELDS}
    assert "GateEditorSettings" in classes, (
        "GateEditorSettings is the class this test was written for; if it was "
        "renamed, update the scan rather than losing the guard")


@pytest.mark.parametrize(
    ("path", "cls", "field"), FIELDS,
    ids=[f"{cls}.{field}" for _, cls, field in FIELDS])
def test_every_settings_field_is_read_somewhere(path, cls, field):
    """Declared, given a widget, saved, reloaded -- and used by nothing."""
    pattern = re.compile(rf"\b{re.escape(field)}\b")
    readers = [
        other.relative_to(REPO)
        for other in PACKAGE.rglob("*.py")
        if other != path
        and pattern.search(other.read_text(encoding="utf-8", errors="replace"))
    ]

    reason = KNOWN_UNREAD.get((cls, field))
    if reason is not None:
        assert not readers, (
            f"{cls}.{field} is now read by {[str(r) for r in readers]}, so it "
            f"is no longer parked -- remove it from KNOWN_UNREAD")
        pytest.skip(f"declared ahead of its reader: {reason}")

    assert readers, (
        f"{cls}.{field} is declared in {path.relative_to(REPO)} and read by "
        f"no other module. Either wire it up, delete it, or -- if the feature "
        f"that reads it is a named open instruction -- add it to KNOWN_UNREAD "
        f"with that instruction, so it is parked on purpose rather than "
        f"forgotten.")


def test_the_parked_list_does_not_name_a_field_that_no_longer_exists():
    """Otherwise the allowlist outlives the field and hides the next one."""
    live = {(cls, field) for _, cls, field in FIELDS}
    stale = sorted(key for key in KNOWN_UNREAD if key not in live)
    assert not stale, f"KNOWN_UNREAD names fields that are gone: {stale}"
