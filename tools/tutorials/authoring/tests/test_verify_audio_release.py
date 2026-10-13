from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import verify_audio_release as verifier  # noqa: E402
from narration_audio import (  # noqa: E402
    cadence_profile_identity,
    render_fingerprint,
)


def valid_timing() -> dict:
    return {
        "schema": 1,
        "language": "en",
        "voice": "am_santa",
        "sample_rate": 24_000,
        "total_duration": 2.0,
        "scenes": [
            {
                "scene": 1,
                "speech_start": 0.0,
                "speech_end": 0.5,
                "scene_end": 1.0,
                "duration": 1.0,
                "text": "First scene.",
                "speech_text": "First scene.",
                "sentences": [
                    {
                        "sentence": 1,
                        "speech_start": 0.0,
                        "speech_end": 0.5,
                        "audible_start": 0.1,
                        "audible_end": 0.4,
                        "duration": 0.5,
                        "gap_from_previous": None,
                        "text": "First scene.",
                        "speech_text": "First scene.",
                        "phonemes": "fɜːst siːn",
                    }
                ],
            },
            {
                "scene": 2,
                "speech_start": 1.0,
                "speech_end": 1.5,
                "scene_end": 2.0,
                "duration": 1.0,
                "text": "Second scene.",
                "speech_text": "Second scene.",
                "sentences": [
                    {
                        "sentence": 1,
                        "speech_start": 1.0,
                        "speech_end": 1.5,
                        "audible_start": 1.1,
                        "audible_end": 1.4,
                        "duration": 0.5,
                        "gap_from_previous": None,
                        "text": "Second scene.",
                        "speech_text": "Second scene.",
                        "phonemes": "sɛkənd siːn",
                    }
                ],
            },
        ],
    }


def track_path(root: Path) -> Path:
    return root / "lesson" / "audio" / "en" / "am_santa.m4a"


def track_spec(root: Path) -> verifier.TrackSpec:
    lesson = {
        "id": "lesson",
        "scenes": [
            {"narration": "First scene.", "hold_after": 0.5},
            {"narration": "Second scene.", "hold_after": 0.5},
        ],
    }
    plans = [
        {
            "display_text": f"{name} scene.",
            "sentences": [{"text": f"{name} scene.", "speech_text": f"{name} scene."}],
            "authored_hold": 0.5,
            "hold": 0.5,
            "speed_multiplier": 1.0,
            "effective_speed": 1.0,
        }
        for name in ("First", "Second")
    ]
    return verifier.TrackSpec(
        path=track_path(root),
        lesson=lesson,
        language="en",
        lang_code="a",
        voice="am_santa",
        dialect="us",
        speed=1.0,
        scene_plans=plans,
    )


def add_spec_fields(timing: dict) -> dict:
    timing["dialect"] = "us"
    timing["speed"] = 1.0
    for scene in timing["scenes"]:
        scene.update(
            {
                "phonemes": "fəʊniːmz",
                "authored_hold_after": 0.5,
                "hold_after": 0.5,
                "base_speed": 1.0,
                "speed_multiplier": 1.0,
                "effective_speed": 1.0,
            }
        )
    return timing


def test_timing_contract_accepts_contiguous_scenes_with_end_holds(tmp_path):
    total, errors = verifier.validate_timing(track_path(tmp_path), valid_timing())
    assert total == 2.0
    assert errors == []


def test_timing_contract_rejects_gaps_overlaps_and_missing_holds(tmp_path):
    path = track_path(tmp_path)

    gap = deepcopy(valid_timing())
    gap["scenes"][1]["speech_start"] = 1.1
    _, errors = verifier.validate_timing(path, gap)
    assert any("gap in scene coverage" in error for error in errors)

    overlap = deepcopy(valid_timing())
    overlap["scenes"][1]["speech_start"] = 0.9
    _, errors = verifier.validate_timing(path, overlap)
    assert any("overlap in scene coverage" in error for error in errors)

    no_hold = deepcopy(valid_timing())
    no_hold["scenes"][1]["speech_end"] = 2.0
    _, errors = verifier.validate_timing(path, no_hold)
    assert any("no silent hold" in error for error in errors)


def test_timing_contract_rejects_inconsistent_scene_and_track_ends(tmp_path):
    timing = valid_timing()
    timing["scenes"][1]["duration"] = 0.8
    timing["total_duration"] = 2.2
    _, errors = verifier.validate_timing(
        track_path(tmp_path), timing, require_sentence_metadata=True
    )
    assert any(
        "does not cover speech_start through scene_end" in error for error in errors
    )
    assert any("final scene ends" in error for error in errors)


def test_timing_contract_requires_exact_catalog_scene_count(tmp_path):
    timing = add_spec_fields(valid_timing())
    timing["scenes"].pop()
    _, errors = verifier.validate_timing(
        track_path(tmp_path), timing, track_spec(tmp_path)
    )
    assert any("current catalog requires exactly 2" in error for error in errors)


def test_timing_contract_rejects_non_monotonic_sentence_bounds(tmp_path):
    timing = valid_timing()
    timing["scenes"][0]["sentences"] = [
        {
            "sentence": 1,
            "speech_start": 0.0,
            "speech_end": 0.3,
            "audible_start": 0.05,
            "audible_end": 0.25,
            "duration": 0.3,
            "gap_from_previous": None,
            "text": "First",
            "speech_text": "First",
            "phonemes": "fɜːst",
        },
        {
            "sentence": 2,
            "speech_start": 0.2,
            "speech_end": 0.5,
            "audible_start": 0.2,
            "audible_end": 0.45,
            "duration": 0.3,
            "gap_from_previous": -0.05,
            "text": "scene.",
            "speech_text": "scene.",
            "phonemes": "siːn",
        },
    ]
    _, errors = verifier.validate_timing(
        track_path(tmp_path), timing, require_sentence_metadata=True
    )
    assert any("overlaps the previous sentence" in error for error in errors)
    assert any("non-monotonic audible timing" in error for error in errors)


def test_strict_decode_rejects_error_output_even_with_zero_exit(monkeypatch, tmp_path):
    path = track_path(tmp_path)

    def fake_run(command, **kwargs):
        assert "-xerror" in command
        assert command[command.index("-err_detect") + 1] == "explode"
        assert command[command.index("-threads") + 1] == "1"
        assert command[-3:] == ["s16le", "pipe:1"] or command[-3:] == [
            "-f",
            "s16le",
            "pipe:1",
        ]
        return SimpleNamespace(
            returncode=0,
            stdout=b"\0" * (24_000 * 2),
            stderr=b"Error processing packet in decoder",
        )

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    samples, duration, pcm, errors = verifier._decode_track(path)
    assert samples == 24_000
    assert duration == 1.0
    assert len(pcm) == 48_000
    assert any("decoder reported errors" in error for error in errors)


def test_track_check_compares_counted_samples_to_sidecar(monkeypatch, tmp_path):
    path = track_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"placeholder")
    timing = valid_timing()
    timing["media_bytes"] = path.stat().st_size
    timing["media_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(timing), encoding="utf-8")
    monkeypatch.setattr(verifier, "_probe_track", lambda unused: (24_000, 1, []))
    monkeypatch.setattr(
        verifier,
        "_decode_track",
        lambda unused, sample_rate, channels: (45_600, 1.9, b"\0" * (45_600 * 2), []),
    )
    monkeypatch.setattr(verifier, "_scene_activity_errors", lambda *args, **kwargs: [])
    monkeypatch.setattr(verifier, "_dead_air_errors", lambda unused: [])

    errors = verifier.check_track(path)
    assert any("decoded_samples=45600" in error for error in errors)
    assert any("exceeds 75ms tolerance" in error for error in errors)


def test_media_identity_rejects_replaced_decodable_file(tmp_path):
    path = track_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_bytes(b"replacement bytes")
    timing = {
        "media_bytes": len(b"original"),
        "media_sha256": hashlib.sha256(b"original").hexdigest(),
    }
    errors = verifier._media_identity_errors(path, timing)
    assert any("media_bytes=" in error for error in errors)
    assert any("media_sha256 does not match" in error for error in errors)


def test_scene_activity_rejects_silence_only_track_and_accepts_speech(tmp_path):
    path = track_path(tmp_path)
    timing = valid_timing()
    silent_pcm = b"\0\0" * (2 * verifier.SAMPLE_RATE)
    errors = verifier._scene_activity_errors(
        path,
        timing,
        silent_pcm,
        sample_rate=verifier.SAMPLE_RATE,
        channels=1,
    )
    assert len([error for error in errors if "no credible" in error]) == 2

    voiced_pcm = b"\0\x10" * (2 * verifier.SAMPLE_RATE)
    assert (
        verifier._scene_activity_errors(
            path,
            timing,
            voiced_pcm,
            sample_rate=verifier.SAMPLE_RATE,
            channels=1,
        )
        == []
    )


def test_render_identity_rejects_stale_inputs_and_fingerprint(monkeypatch, tmp_path):
    spec = track_spec(tmp_path)
    expected_inputs = {"renderer": "current"}
    monkeypatch.setattr(
        verifier,
        "_expected_render_metadata",
        lambda unused_spec, unused_timing: ("current-fingerprint", expected_inputs, []),
    )
    timing = {
        "render_inputs": {"renderer": "old"},
        "render_fingerprint": "old-fingerprint",
    }
    errors = verifier._render_identity_errors(spec, timing)
    assert any("render_inputs are stale" in error for error in errors)
    assert any("render_fingerprint is stale" in error for error in errors)


def test_expected_render_identity_uses_recorded_device_and_loaded_voice_hash(
    monkeypatch, tmp_path
):
    spec = track_spec(tmp_path)
    seen = {}

    def fake_runtime(voice, device):
        seen["runtime"] = (voice, device)
        return {
            "voice_pack": {
                "tensor_sha256": "tensor-hash",
                "loaded_tensor_sha256": "tensor-hash",
            }
        }

    def fake_fingerprint(*args):
        seen["identity"] = args[-1]
        return "fingerprint", {"inputs": "current"}

    monkeypatch.setattr(verifier, "_current_runtime_identity", fake_runtime)
    monkeypatch.setattr(verifier, "track_fingerprint", fake_fingerprint)
    timing = {"render_inputs": {"synthesis_runtime": {"inference": {"device": "cuda"}}}}
    fingerprint, inputs, errors = verifier._expected_render_metadata(spec, timing)
    assert errors == []
    assert fingerprint == "fingerprint"
    assert inputs == {"inputs": "current"}
    assert seen["runtime"] == ("am_santa", "cuda")
    assert seen["identity"]["voice_pack"]["loaded_tensor_sha256"] == ("tensor-hash")


def test_verifier_rejects_stale_cadence_enabled_flag_or_source_hash(
    monkeypatch, tmp_path
):
    spec = track_spec(tmp_path)
    runtime_identity = {"inference": {"device": "cpu"}}
    monkeypatch.setattr(
        verifier,
        "_current_runtime_identity",
        lambda voice, device: runtime_identity,
    )
    seed_timing = {
        "render_inputs": {"synthesis_runtime": {"inference": {"device": "cpu"}}}
    }
    expected_fingerprint, expected_inputs, errors = (
        verifier._expected_render_metadata(spec, seed_timing)
    )

    assert errors == []
    assert expected_inputs is not None
    assert expected_fingerprint == render_fingerprint(expected_inputs)
    assert expected_inputs["cadence_profile"] == cadence_profile_identity()
    assert expected_inputs["cadence_profile"]["enabled"] is False

    for field, changed_value in (
        ("enabled", True),
        ("sha256", "0" * 64),
    ):
        stale_inputs = deepcopy(expected_inputs)
        stale_inputs["cadence_profile"][field] = changed_value
        stale_timing = {
            "render_inputs": stale_inputs,
            "render_fingerprint": render_fingerprint(stale_inputs),
        }
        stale_errors = verifier._render_identity_errors(spec, stale_timing)
        assert any("render_inputs are stale" in error for error in stale_errors)
        assert any("render_fingerprint is stale" in error for error in stale_errors)


def test_freshness_scope_allows_legacy_non_english_but_rejects_legacy_english(
    monkeypatch, tmp_path
):
    decoded = []

    def fake_decode(path, sample_rate, channels):
        decoded.append(path)
        pcm = b"\0\x10" * (2 * verifier.SAMPLE_RATE)
        return 2 * verifier.SAMPLE_RATE, 2.0, pcm, []

    monkeypatch.setattr(
        verifier, "_probe_track", lambda unused: (verifier.SAMPLE_RATE, 1, [])
    )
    monkeypatch.setattr(verifier, "_decode_track", fake_decode)
    monkeypatch.setattr(verifier, "_dead_air_errors", lambda unused: [])

    legacy = valid_timing()
    for scene in legacy["scenes"]:
        scene.pop("sentences")

    english_spec = track_spec(tmp_path)
    english_spec.path.parent.mkdir(parents=True)
    english_spec.path.write_bytes(b"legacy English AAC placeholder")
    english_spec.path.with_suffix(".json").write_text(
        json.dumps(legacy), encoding="utf-8"
    )
    english_errors = verifier.check_track(english_spec)
    assert any(
        "sentences must be a non-empty list" in error for error in english_errors
    )
    assert any("media_bytes must be an integer" in error for error in english_errors)
    assert any(
        "does not record the synthesis device" in error for error in english_errors
    )

    spanish_path = tmp_path / "lesson" / "audio" / "es" / "em_santa.m4a"
    spanish_spec = replace(
        english_spec,
        path=spanish_path,
        language="es",
        lang_code="e",
        voice="em_santa",
        freshness_required=False,
    )
    spanish_timing = deepcopy(legacy)
    spanish_timing["language"] = "es"
    spanish_timing["voice"] = "em_santa"
    spanish_path.parent.mkdir(parents=True)
    spanish_path.write_bytes(b"legacy Spanish AAC placeholder")
    spanish_path.with_suffix(".json").write_text(
        json.dumps(spanish_timing), encoding="utf-8"
    )
    assert verifier.check_track(spanish_spec) == []
    assert decoded == [english_spec.path, spanish_path]


def test_supported_matrix_excludes_files_not_in_renderer_languages(tmp_path):
    catalog = tmp_path / "lessons_en.json"
    catalog.write_text(
        json.dumps({"lessons": [{"id": "01_example"}]}), encoding="utf-8"
    )
    production = tmp_path / "production"
    retired = production / "01_example/audio/en/retired_voice.m4a"
    retired.parent.mkdir(parents=True)
    retired.write_bytes(b"not part of the matrix")
    matrix = {"en": ("a", ["supported_voice"])}

    paths = verifier.supported_track_paths(
        production=production, catalog_path=catalog, languages=matrix
    )
    assert paths == [production / "01_example/audio/en/supported_voice.m4a"]
    assert retired not in paths


def test_supported_specs_use_localized_catalog_and_current_planning(tmp_path):
    catalog = tmp_path / "lessons_en.json"
    catalog.write_text(
        json.dumps(
            {
                "lessons": [
                    {
                        "id": "01_example",
                        "scenes": [{"narration": "A concise tutorial sentence."}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = verifier.supported_track_specs(
        production=tmp_path / "production",
        catalog_path=catalog,
        languages={"en": ("a", ["am_santa"])},
    )
    assert len(specs) == 1
    assert specs[0].lesson["id"] == "01_example"
    assert specs[0].scene_plans[0]["display_text"] == ("A concise tutorial sentence.")


def test_production_supported_matrix_is_69_lessons_by_50_voices():
    voice_count = sum(len(voices) for _, voices in verifier.LANGUAGES.values())
    paths = verifier.supported_track_paths()
    assert voice_count == 50
    assert len(paths) == 73 * 50 == 3650
    assert not any(path.stem == "af_nicole" for path in paths)
    assert not any(path.stem == "af_alloy" for path in paths)
    assert not any(path.stem == "af_kore" for path in paths)
    assert not any(path.stem == "af_nova" for path in paths)


def test_main_fails_if_frozen_renderer_source_changes_during_verification(
    monkeypatch, tmp_path, capsys
):
    media = tmp_path / "lesson/audio/en/am_santa.m4a"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"placeholder")
    media.with_suffix(".json").write_text("{}", encoding="utf-8")
    spec = SimpleNamespace(path=media)
    guard_calls = 0

    def changing_source_guard():
        nonlocal guard_calls
        guard_calls += 1
        if guard_calls == 2:
            raise RuntimeError("pronunciation source changed on disk")

    monkeypatch.setattr(
        verifier, "assert_frozen_sources_current", changing_source_guard
    )
    monkeypatch.setattr(verifier, "supported_track_specs", lambda **kwargs: [spec])
    monkeypatch.setattr(verifier, "check_track", lambda unused: [])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_audio_release.py",
            "--workers",
            "1",
            "--freshness-languages",
            "en",
        ],
    )

    assert verifier.main() == 1
    output = capsys.readouterr().out
    assert guard_calls == 2
    assert "frozen renderer source changed during verification" in output
    assert "pronunciation source changed on disk" in output
