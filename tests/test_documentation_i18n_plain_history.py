"""406: ordinary API builds must retain source-proven historical paragraphs.

Drive the real CLI orchestration, block selection and catalog writer against
temporary files. Decoding and the HEAD reader are recorders: these tests never
load a model, run git, or inspect a shared translation cache.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
OLD_BLOCK = "Return the task status."
NEW_BLOCK = "Return the processing status."
SHARED_BLOCK = "Keep the saved image."
OLD_SOURCE = f"{OLD_BLOCK}\n\n{SHARED_BLOCK}"
NEW_SOURCE = f"{NEW_BLOCK}\n\n{SHARED_BLOCK}"
PT_OLD = "Retorna o estado da tarefa."
PT_NEW = "Retorna o estado do processamento."
PT_SHARED = "Mantém a imagem salva."
PT_OTHER_SHARED = "Conserva a imagem salva."
PT_GENERATED_SHARED = "Preserva a imagem salva."


@pytest.fixture
def plain_build(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "tools"))
    import build_documentation_i18n as builder

    api_dir = tmp_path / "api"
    api_dir.mkdir()
    readme_dir = tmp_path / "readme"
    readme_dir.mkdir()
    readme_source = tmp_path / "README.rst"
    readme_source.write_text("Documentation.\n", encoding="utf-8")
    for language in ("pt", "de"):
        (readme_dir / f"README.{language}.rst").write_text(
            "Existing localized README.\n", encoding="utf-8",
        )
    reviewed_dir = tmp_path / "reviewed"
    reviewed_dir.mkdir()
    model_root = tmp_path / "models"
    state = SimpleNamespace(
        builder=builder, api_dir=api_dir, reviewed_dir=reviewed_dir,
        docs={}, head_records={}, head_reads=[], decode_calls=[], events=[],
        decoded_targets={}, fail_language=None, audit_calls=[], readme_calls=[],
    )
    monkeypatch.setattr(builder, "ROOT", tmp_path)
    monkeypatch.setattr(builder, "API_DIR", api_dir)
    monkeypatch.setattr(builder, "API_DOC_ALIASES", {})
    monkeypatch.setattr(builder, "README_DIR", readme_dir)
    monkeypatch.setattr(builder, "README_SOURCE", readme_source)
    monkeypatch.setattr(builder, "REVIEWED_API_DIR", reviewed_dir)
    monkeypatch.setattr(builder, "default_model_root", lambda: model_root)
    monkeypatch.setattr(builder, "public_docstrings", lambda: dict(state.docs))

    def english_manifest():
        return json.loads((api_dir / "en.json").read_text(encoding="utf-8"))

    def committed_history():
        state.head_reads.append(True)
        return state.head_records

    monkeypatch.setattr(builder, "_committed_english_api_symbols", committed_history)

    def decode(blocks, language, supplied_root, args, **kwargs):
        assert supplied_root == model_root
        assert args.device == "cpu"
        blocks = list(blocks)
        state.decode_calls.append({"language": language, "blocks": blocks,
                                   "kwargs": kwargs})
        state.events.append(("decode", language, english_manifest()))
        if language == state.fail_language:
            raise RuntimeError("deliberate later-locale decoder failure")
        targets = state.decoded_targets.get(language, {})
        return {block: targets.get(block, block) for block in blocks}

    monkeypatch.setattr(builder, "_translate_blocks", decode)

    def readme_translation(documents, language, supplied_root, args):
        # --force also refreshes README text. Keep that unrelated lane in the
        # temporary root and out of the API decoder assertions.
        assert supplied_root == model_root
        state.readme_calls.append(language)
        return dict(documents)

    monkeypatch.setattr(builder, "_translate_documents", readme_translation)
    real_write_language = builder.write_language

    def write_language(docs, language, translations):
        state.events.append(("write", language, english_manifest()))
        real_write_language(docs, language, translations)

    monkeypatch.setattr(builder, "write_language", write_language)

    def audit(docs, languages):
        # The full documentation/README inventory audit is outside this
        # focused fixture; publication and its source metadata remain real.
        assert english_manifest() == builder._english_manifest(docs)
        state.audit_calls.append(tuple(languages))
        return 0

    monkeypatch.setattr(builder, "audit", audit)

    def seed(old_docs, translations):
        manifest = builder._english_manifest(old_docs)
        builder._write_json(api_dir / "en.json", manifest)
        for language, localized in translations.items():
            for key, text in localized.items():
                source_blocks, _ = builder.translatable_blocks(old_docs[key])
                target_blocks, _ = builder.translatable_blocks(text)
                assert len(source_blocks) == len(target_blocks)
                for source, target in zip(source_blocks, target_blocks):
                    assert builder._api_block_valid(source, target, language)
                    assert builder._api_block_valid(
                        builder._api_translation_source(source), target, language,
                    )
                    assert builder._contextualize(target, language, source) == target
            real_write_language(old_docs, language, localized)
        return manifest

    def run(languages=("pt",), *, force=False):
        arguments = [
            "build_documentation_i18n.py", "--languages", *languages,
            "--model-root", str(model_root), "--device", "cpu",
        ]
        if force:
            arguments.append("--force")
        monkeypatch.setattr(sys, "argv", arguments)
        return builder.main()

    def texts(language):
        payload = json.loads((api_dir / f"{language}.json").read_text(encoding="utf-8"))
        for key, source in state.docs.items():
            record = payload["symbols"][key]
            assert record["source_sha256"] == builder._source_hash(source)
            assert record["source_blocks_sha256"] == builder._source_block_hashes(source)
            assert record["translation_source_blocks_sha256"] == (
                builder._translation_source_block_hashes(source))
        return {key: record["text"] for key, record in payload["symbols"].items()}

    state.seed = seed
    state.run = run
    state.texts = texts
    state.english_manifest = english_manifest
    state.context = builder._api_translation_source
    return state


def _seed_one_changed_symbol(build, *, languages=("pt",)):
    key = "spacr.example"
    localized = {
        "pt": f"{PT_OLD}\n\n{PT_SHARED}",
        "de": "Gibt den Aufgabenstatus zurück.\n\nBehält das gespeicherte Bild.",
    }
    old_manifest = build.seed(
        {key: OLD_SOURCE},
        {language: {key: localized[language]} for language in languages},
    )
    build.docs = {key: NEW_SOURCE}
    build.decoded_targets = {
        "pt": {build.context(NEW_BLOCK): PT_NEW,
               build.context(SHARED_BLOCK): PT_GENERATED_SHARED},
        "de": {build.context(NEW_BLOCK): "Gibt den Verarbeitungsstatus zurück.",
               build.context(SHARED_BLOCK): "Bewahrt das gespeicherte Bild."},
    }
    return key, old_manifest


@pytest.mark.parametrize("history_location", ["working_manifest", "head_fallback"])
def test_plain_api_build_preserves_each_symbols_untouched_paragraph(
    plain_build, history_location,
):
    build = plain_build
    keys = ("spacr.first", "spacr.second")
    # The same English paragraph has different valid committed translations.
    # Historical reuse must stay per symbol, not collapse to a global source
    # dictionary and overwrite one symbol's target with the other's.
    old_manifest = build.seed(
        {key: OLD_SOURCE for key in keys},
        {"pt": {keys[0]: f"{PT_OLD}\n\n{PT_SHARED}",
                keys[1]: f"{PT_OLD}\n\n{PT_OTHER_SHARED}"}},
    )
    build.docs = {key: NEW_SOURCE for key in keys}
    build.decoded_targets["pt"] = {
        build.context(NEW_BLOCK): PT_NEW,
        build.context(SHARED_BLOCK): PT_GENERATED_SHARED,
    }
    if history_location == "head_fallback":
        # --sources-only has already advanced the working manifest; only the
        # existing hash-bound committed-history reader can prove the old text.
        build.head_records = old_manifest["symbols"]
        build.builder._write_json(
            build.api_dir / "en.json", build.builder._english_manifest(build.docs),
        )

    assert build.run() == 0

    assert [(call["language"], call["blocks"]) for call in build.decode_calls] == [
        ("pt", [build.context(NEW_BLOCK)])]
    assert all(not call["kwargs"].get("force", False) for call in build.decode_calls)
    assert build.texts("pt") == {
        keys[0]: f"{PT_NEW}\n\n{PT_SHARED}",
        keys[1]: f"{PT_NEW}\n\n{PT_OTHER_SHARED}",
    }
    assert bool(build.head_reads) is (history_location == "head_fallback")
    assert build.audit_calls == [("pt",)]


@pytest.mark.parametrize("later_failure", [False, True])
def test_plain_api_build_keeps_old_manifest_until_all_locales_finish(
    plain_build, later_failure,
):
    build = plain_build
    _, old_manifest = _seed_one_changed_symbol(build, languages=("pt", "de"))
    previous_bytes = (build.api_dir / "en.json").read_bytes()
    if later_failure:
        build.fail_language = "de"
        with pytest.raises(RuntimeError, match="later-locale decoder failure"):
            build.run(("pt", "de"))
        assert (build.api_dir / "en.json").read_bytes() == previous_bytes
        assert build.audit_calls == []
    else:
        assert build.run(("pt", "de")) == 0
        assert build.english_manifest() == build.builder._english_manifest(build.docs)
        assert build.audit_calls == [("pt", "de")]
    assert any(event[:2] == ("write", "pt") for event in build.events)
    assert any(event[:2] == ("decode", "de") for event in build.events)
    for _operation, _language, observed_manifest in build.events:
        assert observed_manifest == old_manifest


def test_plain_api_build_force_bypasses_historical_paragraph_reuse(plain_build):
    build = plain_build
    key, _ = _seed_one_changed_symbol(build)

    assert build.run(force=True) == 0

    assert len(build.decode_calls) == 1
    call = build.decode_calls[0]
    assert set(call["blocks"]) == {build.context(NEW_BLOCK), build.context(SHARED_BLOCK)}
    # Preserve the existing decoder/cache contract: CLI force bypasses the
    # catalog; this history fix must not introduce repair's forced decoding.
    assert not call["kwargs"].get("force", False)
    assert build.texts("pt") == {key: f"{PT_NEW}\n\n{PT_GENERATED_SHARED}"}
    assert build.head_reads == []
    assert build.readme_calls == ["pt"]


@pytest.mark.parametrize("force", [False, True])
def test_plain_api_build_reviewed_target_outranks_history_and_decoder(
    plain_build, force,
):
    build = plain_build
    key, _ = _seed_one_changed_symbol(build)
    reviewed_target = "Guarda a imagem salva."
    language_dir = build.reviewed_dir / "pt"
    language_dir.mkdir()
    evidence_path = language_dir / "review.json"
    evidence_path.write_text(json.dumps({
        "schema": 1, "language": "pt", "records": [{
            "label": f"{key}#1", "source": SHARED_BLOCK,
            "source_sha256": build.builder._source_hash(SHARED_BLOCK),
            "context": build.context(SHARED_BLOCK),
            "translation": reviewed_target,
        }],
    }), encoding="utf-8")
    before = evidence_path.read_bytes()
    assert build.builder.reviewed_api_block_translations(build.docs, "pt") == {
        SHARED_BLOCK: reviewed_target}

    assert build.run(force=force) == 0

    assert [call["blocks"] for call in build.decode_calls] == [
        [build.context(NEW_BLOCK)]]
    assert build.texts("pt") == {key: f"{PT_NEW}\n\n{reviewed_target}"}
    assert evidence_path.read_bytes() == before


@pytest.mark.parametrize("fault", ["no_matching_manifest", "stale_context_hash"])
def test_plain_api_build_does_not_reuse_unproven_history(plain_build, fault):
    build = plain_build
    key, _ = _seed_one_changed_symbol(build)
    if fault == "no_matching_manifest":
        build.builder._write_json(
            build.api_dir / "en.json", build.builder._english_manifest(build.docs),
        )
        assert build.head_records == {}
    else:
        path = build.api_dir / "pt.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["symbols"][key]["translation_source_blocks_sha256"][1] = "stale-context"
        build.builder._write_json(path, payload)

    assert build.run() == 0

    assert len(build.decode_calls) == 1
    assert set(build.decode_calls[0]["blocks"]) == {
        build.context(NEW_BLOCK), build.context(SHARED_BLOCK)}
    assert build.texts("pt") == {key: f"{PT_NEW}\n\n{PT_GENERATED_SHARED}"}
