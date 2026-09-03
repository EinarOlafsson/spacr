"""Assigning regex groups to roles, and asking what version is installed.

``role_trouble`` reports problems BEFORE an import runs, and its docstring says
why: "no captured group is silently ignored". Two groups both marked as the
plate is not an error the importer would raise -- it would quietly use one and
drop the other, and the user would find out when half their plates were
missing.
"""
from __future__ import annotations

import builtins
from dataclasses import fields

import pytest


# ---------------------------------------------------------------------------
# import_plan.role_trouble
# ---------------------------------------------------------------------------

def test_a_planned_rename_documents_every_captured_identifier():
    """The optional timepoint is part of the import preview contract."""
    from spacr.import_plan import Renamed

    missing = [
        field.name for field in fields(Renamed)
        if f":ivar {field.name}:" not in (Renamed.__doc__ or "")
    ]
    assert not missing, f"undocumented Renamed fields: {missing}"

def test_a_complete_assignment_has_no_trouble():
    """The baseline: every required role taken exactly once."""
    from spacr.import_plan import REQUIRED, role_trouble

    roles = {f"g{i}": role for i, role in enumerate(REQUIRED)}

    assert role_trouble(roles) == ()


def test_two_groups_claiming_one_role_are_named_together():
    """The duplicate report, which names BOTH groups.

    Naming one would leave the user changing the wrong half. The importer
    would not raise on this -- it would use one group and drop the other --
    so reporting it before the import is the only place it can be caught.
    """
    from spacr.import_plan import REQUIRED, role_trouble

    role = REQUIRED[0]
    roles = {f"g{i}": r for i, r in enumerate(REQUIRED)}
    roles["extra"] = role

    trouble = role_trouble(roles)

    assert any(role in message for message in trouble)
    assert any("extra" in message and "g0" in message for message in trouble)


def test_a_missing_required_role_is_named():
    """The missing report, which lists exactly what is absent."""
    from spacr.import_plan import REQUIRED, role_trouble

    roles = {f"g{i}": r for i, r in enumerate(REQUIRED[1:], start=1)}

    trouble = role_trouble(roles)

    assert any(REQUIRED[0] in message for message in trouble)


def test_the_message_agrees_with_itself_about_how_many_are_missing():
    """"them" for several and "it" for one -- small, and read by a user.

    A sentence that says "spaCR cannot organise a file without them" for a
    single missing role reads as though something else is wrong too.
    """
    from spacr.import_plan import REQUIRED, role_trouble

    one_missing = role_trouble({f"g{i}": r
                                for i, r in enumerate(REQUIRED[1:], start=1)})
    assert any("without it" in m for m in one_missing)

    if len(REQUIRED) > 2:
        several = role_trouble({"g0": REQUIRED[0]})
        assert any("without them" in m for m in several)


def test_a_group_with_no_role_is_neither_a_duplicate_nor_a_claim():
    """The ``if role:`` guard: an unassigned group is the starting state.

    Every group begins unassigned, so counting a blank as a claim would make
    a half-filled dialog report duplicates that do not exist.
    """
    from spacr.import_plan import REQUIRED, role_trouble

    roles = {f"g{i}": r for i, r in enumerate(REQUIRED)}
    roles["unassigned_a"] = ""
    roles["unassigned_b"] = ""

    assert role_trouble(roles) == ()


def test_no_roles_at_all_reports_every_required_one():
    """The empty mapping, which is a dialog nobody has touched."""
    from spacr.import_plan import REQUIRED, role_trouble

    trouble = role_trouble({})

    assert trouble
    for role in REQUIRED:
        assert any(role in message for message in trouble)


# ---------------------------------------------------------------------------
# updater.installed_version
# ---------------------------------------------------------------------------

def test_a_package_that_is_installed_reports_its_version():
    """The ordinary answer, on a package that is certainly here."""
    from spacr.updater import installed_version

    assert installed_version("numpy")


def test_a_package_that_is_absent_reports_nothing():
    """PackageNotFoundError becomes None, which is the caller's "not installed"."""
    from spacr.updater import installed_version

    assert installed_version("a-package-that-is-not-installed-anywhere") is None


def test_an_environment_without_importlib_metadata_reports_nothing(monkeypatch):
    """The outer except, and the comment's reason.

    "A bundler that ships only what it saw imported can leave this out" -- a
    frozen desktop build is exactly that, and the updater must answer "I don't
    know" rather than crash the About dialog it is called from.
    """
    from spacr import updater

    real_import = builtins.__import__

    def refusing(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "importlib.metadata":
            raise ImportError("frozen build has no importlib.metadata")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", refusing)

    assert updater.installed_version("numpy") is None
