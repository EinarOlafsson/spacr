"""Item 288: ``picture_settings.drop_retired`` when the rename table cannot help.

A saved picture-settings blob is migrated as it is read: retired keys go,
renamed keys move onto the name this panel offers. Three cases where the
rename resolver cannot move a key, and the blob must come through intact
rather than lose values or raise:

* the settings module cannot be imported at all;
* the resolver raises on one key;
* a key renames to something this panel does not offer, or to several.
"""
from __future__ import annotations

import sys

import spacr.settings as settings_module
from spacr import picture_settings as ps


def test_without_the_resolver_the_blob_is_only_stripped_of_retired_keys(
        monkeypatch):
    retired = next(iter(ps.RETIRED))
    monkeypatch.setitem(sys.modules, "spacr.settings", None)
    out, notes = ps.drop_retired({retired: 1, "img_size": 200})
    assert out == {"img_size": 200}, "nothing could be renamed, so nothing is"
    assert notes == [f"{retired}: {ps.RETIRED[retired]}"]


def test_a_resolver_that_raises_leaves_the_key_where_it_is(monkeypatch):
    def broken(_key):
        raise RuntimeError("rename table unreadable")

    monkeypatch.setattr(settings_module, "surviving_setting_name", broken)
    out, notes = ps.drop_retired({"img_size": 200})
    assert out == {"img_size": 200}
    assert notes == []


def test_a_rename_to_a_key_this_panel_does_not_offer_is_not_made(
        monkeypatch):
    monkeypatch.setattr(settings_module, "surviving_setting_name",
                        lambda key: ("some_other_panels_key",)
                        if key == "old_a" else ("a", "b"))
    out, notes = ps.drop_retired({"old_a": 1, "old_b": 2})
    assert out == {"old_a": 1, "old_b": 2}
    assert notes == []
