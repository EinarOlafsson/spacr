"""API assembly uses real fresh processes and preserves explicit output paths."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import build_documentation_i18n as builder


@pytest.fixture
def assembly(tmp_path, monkeypatch):
    api = tmp_path / "api"
    reviews = tmp_path / "reviews"
    api.mkdir()
    reviews.mkdir()
    metadata = tmp_path / "docs"
    metadata.mkdir()
    (metadata / "media_revision.json").write_text((ROOT / "docs/media_revision.json").read_text())
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "API_DIR", api)
    monkeypatch.setattr(builder, "REVIEWED_API_DIR", reviews)
    monkeypatch.setattr(builder, "ASSEMBLED_STAMP", tmp_path / "assembled.json")
    fetched = []
    monkeypatch.setattr(builder, "_media_tool", lambda: SimpleNamespace(
        fetch=lambda group, root: fetched.append((group, root))))
    docs = {"spacr.example": "Return the task status."}
    translations = {"pt": "Retorna o estado da tarefa.",
                    "es": "Devuelve el estado de la tarea."}
    for language, target in translations.items():
        builder.write_language(docs, language, {"spacr.example": target})
    return api, reviews, docs, translations, fetched


def test_actual_spawned_workers_keep_temporary_paths_with_a_live_parent_thread(assembly):
    api, reviews, docs, translations, fetched = assembly
    started = threading.Event()
    release = threading.Event()

    def hold_parent_thread():
        started.set()
        release.wait()

    thread = threading.Thread(target=hold_parent_thread)
    thread.start()
    try:
        assert started.wait(5)
        assert builder.assemble_api_catalogs(translations, docs, jobs=2)
        assert thread.is_alive()
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert fetched == [(builder.API_MEMORY_GROUP, builder.ROOT)]
    for language, target in translations.items():
        payload = json.loads((api / (language + ".json")).read_text())
        record = payload["symbols"]["spacr.example"]
        assert payload["schema"] == 2 and payload["language"] == language
        assert record["text"] == target
        assert record["source_sha256"] == builder._source_hash(docs["spacr.example"])
        assert record["source_blocks_sha256"] == builder._source_block_hashes(
            docs["spacr.example"])
        assert record["translation_source_blocks_sha256"] == \
            builder._translation_source_block_hashes(docs["spacr.example"])
    assert json.loads((api / "en.json").read_text()) == builder._english_manifest(docs)
    assert json.loads(builder.ASSEMBLED_STAMP.read_text())["languages"] == ["es", "pt"]
    assert not builder.assemble_api_catalogs(translations, docs, jobs=2, if_stale=True)
    assert fetched == [(builder.API_MEMORY_GROUP, builder.ROOT)]
    assert builder.API_DIR == api and builder.REVIEWED_API_DIR == reviews


def test_spawned_worker_keeps_strict_review_rejection_at_the_explicit_path(assembly):
    api, reviews, docs, translations, fetched = assembly
    directory = reviews / "pt"
    directory.mkdir()
    source = docs["spacr.example"]
    (directory / "review.json").write_text(json.dumps({
        "schema": 1, "language": "pt", "records": [{
            "label": "spacr.example#0", "source": source,
            "source_sha256": "0" * 64,
            "context": builder._api_translation_source(source),
            "translation": translations["pt"],
        }]}))
    with pytest.raises(ValueError, match="stale reviewed API hash"):
        builder.assemble_api_catalogs(translations, docs, jobs=2)
    assert not (api / "en.json").exists()
    assert not builder.ASSEMBLED_STAMP.exists()
    assert fetched == [(builder.API_MEMORY_GROUP, builder.ROOT)]


@pytest.mark.parametrize("history_valid,review_present", [(True, True), (False, True), (True, False)])
def test_changed_document_assembly_requires_complete_review_and_proven_history(
    assembly, monkeypatch, history_valid, review_present,
):
    api, reviews, _docs, _translations, _fetched = assembly
    symbol = "spacr.example"
    old_source = "Return the task status.\n\nRead the saved configuration."
    source = "Return the task status.\n\nReview the selected image axes.\n\nRead the saved configuration."
    old_target = "Retorna o estado da tarefa.\n\nLeia a configuração salva."
    new_target = "Revise os eixos da imagem selecionada."
    builder.write_language({symbol: old_source}, "pt", {symbol: old_target})
    english = builder._english_manifest({symbol: old_source})
    if not history_valid:
        english["symbols"][symbol]["source_sha256"] = "0" * 64
    (api / "en.json").write_text(json.dumps(english))
    monkeypatch.setattr(builder, "_committed_english_api_symbols", lambda: {})
    monkeypatch.setattr(builder, "_archived_english_api_records", lambda: {})
    if review_present:
        directory = reviews / "pt"
        directory.mkdir()
        block = builder.translatable_blocks(source)[0][1]
        (directory / "revision.json").write_text(json.dumps({
            "schema": 1, "language": "pt", "records": [{
                "label": symbol + "#1", "source": block,
                "source_sha256": builder._source_hash(block),
                "context": builder._api_translation_source(block),
                "translation": new_target,
            }]}))
    before = json.loads((api / "pt.json").read_text())["symbols"][symbol]
    for _ in range(2):
        result = builder.assemble_api_catalog({symbol: source}, "pt")["symbols"][symbol]
        if history_valid and review_present:
            assert result["source_sha256"] == builder._source_hash(source)
            assert result["text"] == "Retorna o estado da tarefa.\n\n" + new_target + "\n\nLeia a configuração salva."
            assert result["translation_source_blocks_sha256"] == builder._translation_source_block_hashes(source)
        else:
            assert result == before


def test_reviewed_default_enabled_survives_fresh_and_cached_assembly(assembly):
    api, reviews, _docs, _translations, fetched = assembly
    symbol = "spacr.qt.preferences.get_easy_mode"
    old = "Whether the content-guided Easy mode Home is enabled (default: ``False``)."
    current = old.replace("``False``", "``True``")
    old_target = "Se o Home do modo Easy guiado pelo conteúdo está ativado (padrão: ``False``)."
    target = old_target.replace("``False``", "``True``")
    other = "spacr.example"
    docs = {symbol: current, other: "Return the task status."}
    builder.write_language({symbol: old, other: docs[other]}, "pt", {
        symbol: old_target, other: "Retorna o estado da tarefa."})
    (api / "en.json").write_text(json.dumps(builder._english_manifest({symbol: old, other: docs[other]})))
    directory = reviews / "pt"
    directory.mkdir()
    (directory / "default.json").write_text(json.dumps({
        "schema": 1, "language": "pt", "records": [{
            "label": symbol + "#0", "source": current,
            "source_sha256": builder._source_hash(current),
            "context": builder._api_translation_source(current), "translation": target,
        }]}))
    untouched = json.loads((api / "pt.json").read_text())["symbols"][other]
    assert builder.assemble_api_catalogs(["pt"], docs, jobs=1, if_stale=True)
    assert not builder.assemble_api_catalogs(["pt"], docs, jobs=1, if_stale=True)
    assert builder.assemble_api_catalogs(["pt"], docs, jobs=1)
    records = json.loads((api / "pt.json").read_text())["symbols"]
    assert records[symbol]["text"] == target
    assert records[symbol]["source_sha256"] == builder._source_hash(current)
    assert records[other] == untouched
    assert len(fetched) == 2
