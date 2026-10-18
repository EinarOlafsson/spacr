"""The newest installation can remove any older spaCR installation and keep itself.

Maintainer requirement 2026-10-18: from the newest install, remove older ones
individually or all at once, while the running one stays and reopens.
"""
import pytest

from spacr import install_cleanup as ic


def _preview(*rows):
    return {"operation_id": "op", "snapshot_id": "snap", "categories": list(ic._UNINSTALL_CATEGORIES),
            "cleanup": [], "installations": [
                {"key": key, "current": current, "selectable": selectable,
                 "record": {"root": "/" + key, "version": "1", "kind": "installer"}, "paths": ["/" + key]}
                for key, current, selectable in rows]}


@pytest.fixture
def preview():
    return _preview(("running", True, True), ("old-pkg", False, True),
                    ("conda", False, True), ("checkout", False, False))


def test_older_scope_is_every_removable_installation_except_the_running_one(preview):
    assert ic._uninstall_selection(preview, scope="older")["installations"] == ["old-pkg", "conda"]


def test_current_and_all_scopes_are_unchanged(preview):
    assert ic._uninstall_selection(preview, scope="current")["installations"] == ["running"]
    assert ic._uninstall_selection(preview, scope="all")["installations"] == ["running", "old-pkg", "conda"]


def test_ticked_installations_are_kept_in_preview_order(preview):
    chosen = ic._uninstall_selection(preview, scope="custom", installations=["conda", "old-pkg"])
    assert chosen["scope"] == "custom" and chosen["installations"] == ["old-pkg", "conda"]


@pytest.mark.parametrize("ticked", [["checkout"], ["unknown"], ["conda", "conda"], []])
def test_unremovable_unknown_duplicate_or_empty_ticks_are_refused(preview, ticked):
    with pytest.raises(ValueError):
        ic._uninstall_selection(preview, scope="custom", installations=ticked)


def test_a_named_scope_refuses_an_explicit_list(preview):
    with pytest.raises(ValueError):
        ic._uninstall_selection(preview, scope="older", installations=["conda"])


def test_older_with_nothing_older_is_refused():
    with pytest.raises(ValueError):
        ic._uninstall_selection(_preview(("running", True, True)), scope="older")


def test_validation_replays_the_ticks_and_rejects_a_tampered_list(preview):
    chosen = ic._uninstall_selection(preview, scope="custom", installations=["conda"])
    assert ic._uninstall_selection(preview, scope="custom", categories=chosen["categories"],
                                   installations=chosen["installations"]) == chosen
    for tampered in (dict(chosen, installations=["conda", "running"]),
                     dict(chosen, scope="older"), dict(chosen, installations=["checkout"])):
        with pytest.raises(ValueError):
            ic._validate_uninstall_selection(preview, tampered, ic._Machine(), after_shutdown=True)


def test_spacr_reopens_only_when_the_running_installation_is_kept(preview):
    older = ic._uninstall_selection(preview, scope="older")
    everything = ic._uninstall_selection(preview, scope="all")
    assert ic._uninstall_keeps_running(preview, older)
    assert not ic._uninstall_keeps_running(preview, everything)
