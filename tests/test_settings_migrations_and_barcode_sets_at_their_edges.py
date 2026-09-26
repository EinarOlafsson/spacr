"""Item 288: settings migrations, barcode sets and type checks at their edges.

* The bundled barcode references are fetched from a release tag -- the
  installed one or a named one -- and only when the packaged copy is gone;
  the default fetch goes through urllib and the bytes are verified before
  they are written.
* A barcode set can be given as a :class:`BarcodeSet`, as a mapping with its
  own count columns, or as a list mixing :class:`BarcodeEntry` objects and
  names; a shipped barcode with no table falls back to the shipped one, and
  an entry that is neither a name nor a mapping is refused.
* The migrations leave anything that is not a settings mapping alone, skip
  keys that are not strings, and a retired object bound that is not a
  bound is left in place with a warning, unless asked to be quiet; a bound
  of zero, or one already present in the filter list, adds nothing.
* A rename chain longer than the hop limit resolves to nothing.
* A regression data layout that is not one of the choices is refused.
* A setting that takes text or a switch keeps a switch as a switch.
"""
from __future__ import annotations

import logging

import pytest

import spacr.settings as S


def test_the_barcode_url_is_pinned_to_the_named_release():
    url = S._bundled_barcode_url("grna", version="1.2.3")
    assert "/v1.2.3/spacr/resources/data/" in url
    assert url.endswith(S._BUNDLED_BARCODE_FILES["grna"])


def test_a_missing_reference_is_fetched_through_urllib_and_verified(
        tmp_path, monkeypatch):
    import urllib.request

    shipped = S.bundled_barcode_path("row")
    payload = open(shipped, "rb").read()
    target = tmp_path / "data" / "row.csv"
    monkeypatch.setattr(S, "bundled_barcode_path", lambda kind: str(target))
    asked = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return payload

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda url, timeout: asked.append(url) or Response())
    assert S._ensure_bundled_barcode("row") == str(target)
    assert target.read_bytes() == payload
    assert len(asked) == 1 and asked[0].endswith(S._BUNDLED_BARCODE_FILES["row"])


def test_a_barcode_set_object_is_used_as_given():
    entries = (S.BarcodeEntry(name="grna", csv="/refs/g.csv"),)
    given = S.BarcodeSet(entries)
    assert S.barcode_set_from_settings({"barcode_set": given}) is given
    assert S.barcode_set_from_settings("not settings") is None
    assert given.sequence_columns == (entries[0].sequence_column,)


def test_a_mapping_carries_its_own_count_columns_and_mixed_entries():
    entry = S.BarcodeEntry(name="plate_code", csv="/refs/p.csv")
    result = S.barcode_set_from_settings({"barcode_set": {
        "count_columns": ["grna_name", "plate_codeID"],
        "entries": [entry, "grna"]}})
    assert result.count_columns == ("grna_name", "plate_codeID")
    assert result.names == ("plate_code", "grna")
    assert result.entries[1].csv == S.bundled_barcode_path("grna"), (
        "a shipped barcode with no table takes the shipped one")


def test_an_entry_that_is_neither_a_name_nor_a_mapping_is_refused():
    with pytest.raises(ValueError, match="received int"):
        S.barcode_set_from_settings({"barcode_set": ["grna", 7]})


def test_migrations_leave_what_is_not_a_settings_mapping_alone():
    for fold in (S._fold_renamed_settings, S._fold_toxoplasma,
                 S._fold_object_bounds):
        assert fold(["not", "a", "mapping"]) == ["not", "a", "mapping"]
    settings = {1: "numbered key", "min_cell_count": 50}
    S._fold_renamed_settings(settings)
    assert settings[1] == "numbered key"
    assert "min_cell_count" not in settings


def test_a_retired_bound_that_is_not_a_bound_stays_with_a_warning(caplog):
    key = next(k for k, (obj, bound) in S.RETIRED_OBJECT_BOUNDS.items()
               if bound == "min_area")
    settings = {key: "-4"}
    with caplog.at_level(logging.WARNING, logger=S.LOG.name):
        S._fold_object_bounds(settings)
    assert settings == {key: "-4"}
    assert "is not a bound of zero or more" in caplog.text

    caplog.clear()
    quiet = {key: "wide"}
    with caplog.at_level(logging.WARNING, logger=S.LOG.name):
        S._fold_object_bounds(quiet, quiet=True)
    assert quiet == {key: "wide"} and caplog.text == ""


def test_a_zero_bound_or_one_already_in_the_list_adds_no_row():
    obj, _bound = next(v for v in S.RETIRED_OBJECT_BOUNDS.values()
                       if v[1] == "min_area")
    min_key = next(k for k, v in S.RETIRED_OBJECT_BOUNDS.items()
                   if v == (obj, "min_area"))
    zero = {min_key: 0}
    S._fold_object_bounds(zero)
    assert min_key not in zero
    assert not zero.get("object_filters", {}).get(obj)

    present = {min_key: 40, "object_filters": {obj: [
        {"property": "area", "min": 10.0, "max": None}]}}
    S._fold_object_bounds(present)
    assert present["object_filters"][obj] == [
        {"property": "area", "min": 10.0, "max": None}], (
        "the row the file already had wins")


def test_a_rename_chain_past_the_hop_limit_resolves_to_nothing(monkeypatch):
    chain = {f"k{i}": f"k{i + 1}" for i in range(S._RENAME_HOP_LIMIT + 2)}
    monkeypatch.setattr(S, "RENAMED_SETTINGS", chain)
    names, hops = S._resolve_rename("k0")
    assert names == () and hops == S._RENAME_HOP_LIMIT


def test_a_regression_layout_outside_the_choices_is_refused():
    with pytest.raises(ValueError, match="model_data_layout='diagonal'"):
        S.get_perform_regression_default_settings(
            {"model_data_layout": "diagonal"})


def test_a_text_or_switch_setting_keeps_a_switch():
    class Var:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    settings, errors = S.check_settings(
        {"well_detection": ("label", None, Var(True), None)},
        {"well_detection": (str, bool)})
    assert errors == []
    assert settings["well_detection"] is True
