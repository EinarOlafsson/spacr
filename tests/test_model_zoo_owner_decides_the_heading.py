"""Models the maintainer trained and uploaded are listed under spaCR.

Maintainer request, 2026-10-11: every model uploaded from the einarolafsson
Hugging Face account belongs under "spaCR", not "spaCR community". The
heading follows the repository a row is fetched from, so a model published
there is spaCR's whatever its key; a third-party upload stays community even
when it borrows a spaCR key, and the Add button's inbox
(``einarolafsson/user-models``) stays community although that account owns it.
"""
from __future__ import annotations

import json

import pytest

from spacr import model_zoo as mz


@pytest.fixture(autouse=True)
def _no_cache():
    mz._SHARED_CATALOGUE_CACHE.update(fetched_at=0.0, entries=())
    yield
    mz._SHARED_CATALOGUE_CACHE.update(fetched_at=0.0, entries=())


def _plaque_v3():
    return next(record for record in mz.BUNDLED_REMOTE_MODELS
                if record["key"] == "toxoplasma_plaque_v3")


OWNED = "https://huggingface.co/einarolafsson/toxoplasma-new-cpsam/resolve/main/w"
OWNED_DATASET = ("https://huggingface.co/datasets/einarolafsson/models/"
                 "resolve/main/new/v1/model.pt")
STRANGER = "https://huggingface.co/someone/their-model/resolve/main/w"


def _mixed_owner_catalogue():
    """A shared catalogue with the maintainer's, a stranger's and spoofed rows."""
    plaque = _plaque_v3()
    return {"models": [
        {"key": "maintainer_new_v1", "name": "new_v1", "uri": OWNED,
         "sha256": "a" * 64},
        {"key": "maintainer_dataset_v1", "name": "model.pt",
         "uri": OWNED_DATASET, "sha256": "b" * 64},
        {"key": "maintainer_repo_id_v1", "name": "weights",
         "repo_id": "einarolafsson/live-cell-segmentation-cpsam",
         "repo_type": "model", "sha256": "c" * 64},
        dict(plaque, sha256="d" * 64),
        {"key": "stranger_v1", "name": "w", "uri": STRANGER,
         "sha256": "e" * 64},
        {"key": "toxoplasma_plaque_v3", "name": plaque["name"],
         "uri": STRANGER, "sha256": "f" * 64},
        {"key": "claims_owner_v1", "name": "w", "uri": STRANGER,
         "sha256": "1" * 64, "trained_by": "einarolafsson",
         "owner": "einarolafsson", "uploader": "einarolafsson",
         "repo_id": "einarolafsson/models", "source": "remote"},
        {"key": "spoofed_host_v1", "name": "w",
         "uri": "https://huggingface.co.example.org/einarolafsson/x/"
                "resolve/main/w", "sha256": "2" * 64},
        {"key": "inbox_v1", "name": "w",
         "uri": "https://huggingface.co/einarolafsson/user-models/resolve/"
                "main/staging/inbox/w.pt", "sha256": "3" * 64},
    ]}


@pytest.fixture
def mixed(tmp_path):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps(_mixed_owner_catalogue()))
    rows = mz.shared_catalogue(uri=path.as_uri(), block=True, force=True)
    return {row.key + ("@" + row.uri if row.key == "toxoplasma_plaque_v3"
                       else ""): row for row in rows}


def test_the_mixed_owner_catalogue_is_filed_by_owner(mixed):
    heading = {key: mz.source_of(row) for key, row in mixed.items()}
    plaque_owned = "toxoplasma_plaque_v3@" + _plaque_v3()["uri"]
    plaque_borrowed = "toxoplasma_plaque_v3@" + STRANGER
    assert heading == {
        "maintainer_new_v1": "spaCR",
        "maintainer_dataset_v1": "spaCR",
        "maintainer_repo_id_v1": "spaCR",
        plaque_owned: "spaCR",
        "stranger_v1": "spaCR community",
        plaque_borrowed: "spaCR community",
        "claims_owner_v1": "spaCR community",
        "spoofed_host_v1": "spaCR community",
        "inbox_v1": "spaCR community",
    }


def test_a_catalogue_row_cannot_declare_its_own_source(mixed):
    assert all(row.source == "shared" for row in mixed.values())


def test_a_borrowed_spacr_key_with_another_checksum_is_not_promoted():
    record = dict(_plaque_v3(), sha256="0" * 64, uri=STRANGER)
    for source in ("shared", "community"):
        row = mz._entry_from_mapping(record, source=source)
        assert not mz._is_spacr_published(row)
        assert not mz.is_spacr_owned(row)
        assert mz.source_of(row) == "spaCR community"


def test_the_bundled_checkpoint_mirrored_elsewhere_stays_spacr():
    row = mz._entry_from_mapping(dict(_plaque_v3(), uri=STRANGER),
                                 source="shared")
    assert mz.source_of(row) == "spaCR"


def test_every_published_spacr_model_is_spacr_from_any_transport():
    for record in mz.BUNDLED_REMOTE_MODELS:
        for source in ("remote", "shared", "community"):
            row = mz._entry_from_mapping(record, source=source)
            assert mz.is_spacr_owned(row), record["key"]
            assert mz.source_of(row) == "spaCR", (record["key"], source)


def test_a_row_with_no_location_is_fetched_from_spacrs_repo_and_is_spacr():
    row = mz._entry_from_mapping(
        {"key": "someone_v1", "name": "someone", "sha256": "1" * 64},
        source="shared")
    assert mz.hf_repo_of(row.uri) == mz.HF_MODELS_REPO
    assert mz.source_of(row) == "spaCR"


def test_a_community_inbox_upload_stays_community():
    row = mz.ModelEntry(
        key="community_test", name="hela_60x_v2.CP_model", source="community",
        uri=f"https://huggingface.co/{mz.COMMUNITY_REPO}/resolve/main/"
            "staging/test/hela_60x_v2.CP_model", sha256="9" * 64)
    assert not mz.is_spacr_owned(row)
    assert mz.source_of(row) == "spaCR community"


@pytest.mark.parametrize("uri, repo", [
    ("https://huggingface.co/einarolafsson/a-b/resolve/main/w",
     "einarolafsson/a-b"),
    ("https://huggingface.co/datasets/einarolafsson/models/resolve/main/x.pt",
     "einarolafsson/models"),
    ("https://HuggingFace.co/einarolafsson/a.b_c/resolve/main/w",
     "einarolafsson/a.b_c"),
    ("http://huggingface.co/einarolafsson/a/resolve/main/w", ""),
    ("https://huggingface.co:8443/einarolafsson/a/resolve/main/w", ""),
    ("https://user@huggingface.co/einarolafsson/a/resolve/main/w", ""),
    ("https://huggingface.co.example.org/einarolafsson/a/resolve/main/w", ""),
    ("https://example.org/huggingface.co/einarolafsson/a/resolve/main/w", ""),
    ("https://huggingface.co/someone/../einarolafsson/a/resolve/main/w", ""),
    ("https://huggingface.co/einarolafsson%2Fa/b/resolve/main/w", ""),
    ("https://huggingface.co/einarolafsson", ""),
    ("", ""),
    (None, ""),
])
def test_the_repo_is_read_strictly_from_the_url(uri, repo):
    assert mz.hf_repo_of(uri) == repo


def test_the_owner_comes_from_the_models_repo_constant():
    assert mz.SPACR_HF_OWNERS == ("einarolafsson",)
    assert mz.HF_MODELS_REPO.startswith("einarolafsson/")


def test_the_alternatives_are_the_records_that_say_so():
    marked = {r["key"] for r in mz.BUNDLED_REMOTE_MODELS
              if "alternative" in str(r.get("display_name", "")).lower()}
    assert mz.ALTERNATIVE_MODEL_KEYS == frozenset(marked)
    assert "toxoplasma_pv_v5" in mz.ALTERNATIVE_MODEL_KEYS
    assert "toxoplasma_pv_v4" not in mz.ALTERNATIVE_MODEL_KEYS


def test_the_published_alternatives_are_spacr():
    for key in mz.ALTERNATIVE_MODEL_KEYS:
        record = next(r for r in mz.BUNDLED_REMOTE_MODELS if r["key"] == key)
        row = mz._entry_from_mapping(record)
        assert mz.is_alternative(row)
        assert mz.source_of(row) == "spaCR"


def _family(stem):
    rows = [mz._entry_from_mapping(r) for r in mz.BUNDLED_REMOTE_MODELS
            if r["key"].rsplit("_v", 1)[0] == stem]
    rows.sort(key=lambda e: int(e.key.rsplit("_v", 1)[1]), reverse=True)
    return [(e.key.rsplit("_", 1)[1], e) for e in rows]


def test_the_pv_family_opens_on_v4_not_the_v5_alternative():
    pv = _family("toxoplasma_pv")
    assert pv[mz.default_version_index(pv)][1].key == "toxoplasma_pv_v4"


def test_the_plaque_family_opens_on_the_plaque_default():
    from spacr import submodules

    plaque = _family("toxoplasma_plaque")
    chosen = plaque[mz.default_version_index(plaque)][1]
    assert not mz.is_alternative(chosen)
    assert chosen.key == submodules.DEFAULT_PLAQUE_MODEL


def test_an_all_alternative_or_empty_family_opens_on_its_first_row():
    pv = _family("toxoplasma_pv")
    assert mz.default_version_index(pv[:1]) == 0
    assert mz.default_version_index([]) == 0
