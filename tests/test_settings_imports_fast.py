"""Importing ``spacr.settings`` must not cost the user a force-quit dialog.

WHAT THIS IS ABOUT. ``spacr.settings`` is imported on the GUI thread the first
time a module screen is built: ``app.py`` builds the screen, which imports
``app_screen`` -> ``settings_model`` -> ``barcode_regex`` -> ``spacr.settings``.
Nothing about that chain is deferred, so every millisecond it costs is a
millisecond the window does not answer the compositor.

Measured 2026-09-05, before the fix: importing this one module executed
26,481,737 ``str.startswith`` calls and took 1.93 s warm and 6.7 s cold, which
made opening the Mask module 8.3 s. GNOME asks a window whether it is alive
after five seconds, so what the maintainer saw -- and reported four times --
was the "force quit" dialog. After the fix: 0.48 s warm, and the same open is
2.3 s.

The cause was two comprehensions that rebuilt a constant inside their own
inner loop: 38,557 declared settings x 701 organelle slot roles. The tests
below pin the SHAPE of the fix rather than a wall-clock number, because a
timing assertion on a shared CI box is a flake generator -- with one generous
end-to-end ceiling as a backstop for a regression of that magnitude.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def test_the_slot_prefixes_are_built_once_as_a_tuple():
    """The hoisted constant must exist, and be a tuple for C-level matching.

    ``str.startswith`` accepts a tuple and does the whole comparison in C.
    That is the entire reason this constant exists: the comprehension it
    serves used to build ``f'{role}_'`` inside an ``any()`` over every role,
    for every one of 38,557 keys.
    """
    from spacr import settings

    prefixes = settings._ORGANELLE_SLOT_PREFIXES
    assert isinstance(prefixes, tuple), (
        "must be a tuple -- a list or generator would defeat the C-level "
        "startswith that makes this fast"
    )
    assert prefixes, "no prefixes at all means the filter matches nothing"
    assert all(p.endswith("_") for p in prefixes)


def test_the_dynamic_settings_are_what_the_slow_predicate_answered():
    """The fast path must return exactly the old answer, not merely a fast one.

    This recomputes the ORIGINAL predicate verbatim and compares. If someone
    later 'optimises' the prefix match into something subtly different -- a
    partition on the first underscore, say, which breaks for a role whose name
    contains one -- this fails.
    """
    from spacr import settings

    roles = settings.ORGANELLE_SLOT_ROLES[1:]
    slow = frozenset(
        key for key in settings.expected_types
        if any(key.startswith(f"{role}_") for role in roles)
    )
    assert slow == settings.DYNAMIC_ORGANELLE_SETTINGS


def test_every_role_still_contributes_its_slot_keys():
    """Each role's generated slot keys must all survive into the categories.

    The predicate that was lifted out of the role loop -- ``startswith
    ('organelle_')`` -- does not mention the role, so hoisting it is provably
    equivalent. What this pins is the consequence: no role lost an entry.

    It asserts containment across ALL headings rather than counting one of
    them, because ``_regroup_advanced`` runs afterwards and redistributes
    entries between headings -- an earlier version of this test counted
    ``categories['Organelle']`` and was simply wrong about where they land.
    """
    from spacr import settings

    everywhere = {key for entries in settings.categories.values()
                  for key in entries}
    basic = [k for k in settings.organelle_basic_settings
             if k.startswith("organelle_")]
    assert basic, "nothing to check means the fixture stopped being meaningful"

    for role in settings.ORGANELLE_SLOT_ROLES[1:]:
        expected = {settings._organelle_slot_key(key, role) for key in basic}
        missing = expected - everywhere
        assert not missing, f"role {role!r} lost slot keys: {sorted(missing)[:5]}"


@pytest.mark.slow
def test_importing_settings_alone_stays_under_a_generous_ceiling():
    """A backstop for a regression of the 2026-09-05 magnitude, nothing finer.

    Three seconds is far above the 0.48 s this costs and far below the 6.7 s
    it cost before, so it catches the failure it was written for without
    failing on a loaded CI box. It runs in a subprocess because an import is
    only slow once per interpreter.
    """
    proc = subprocess.run(
        [sys.executable, "-c",
         "import time; t=time.perf_counter(); import spacr.settings; "
         "print(time.perf_counter()-t)"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    elapsed = float(proc.stdout.strip().splitlines()[-1])
    assert elapsed < 3.0, (
        f"importing spacr.settings took {elapsed:.2f}s. It is imported on the "
        "GUI thread when a module screen is first built; past five seconds "
        "the desktop offers to force-quit spaCR."
    )
