#!/usr/bin/env python3
"""Translate the canonical tutorial catalog locally with NLLB-200.

The selected model is CC-BY-NC and is used only for this non-commercial
tutorial project.  Every localized catalog retains the English lesson ids,
scene order, visual instructions, and technical brand spelling while adding
language-appropriate speech text for Kokoro.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from pronunciation import normalize_display_terms, spoken_form


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "project" / "translation_models" / "nllb-200-distilled-600M"
CATALOG = ROOT / "catalog" / "lessons_en.json"

LANGUAGES = {
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "hi": "hin_Deva",
    "it": "ita_Latn",
    "pt-BR": "por_Latn",
    "ja": "jpn_Jpan",
    "zh-CN": "zho_Hans",
}

def translatable_fields(catalog: dict) -> list[tuple[dict, str]]:
    fields: list[tuple[dict, str]] = []
    for series in catalog["series"]:
        fields.append((series, "title"))
    for lesson in catalog["lessons"]:
        for key in ("title", "section", "description", "prerequisite"):
            fields.append((lesson, key))
        for index in range(len(lesson["objectives"])):
            fields.append((lesson["objectives"], index))
        for scene in lesson["scenes"]:
            fields.append((scene, "narration"))
    return fields


def normalize_brand(text: str) -> str:
    return normalize_display_terms(text)


def speech_text(text: str, language: str) -> str:
    return spoken_form(text, language)


def translate_language(source: dict, language: str, batch_size: int,
                       threads: int, device: str) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(threads)
    target_code = LANGUAGES[language]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL, local_files_only=True)
    model.to(device).eval()
    localized = copy.deepcopy(source)
    fields = translatable_fields(localized)
    strings = [str(container[key]) for container, key in fields]
    translated: list[str] = []
    forced_id = tokenizer.convert_tokens_to_ids(target_code)

    for start in range(0, len(strings), batch_size):
        batch = strings[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=320)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_id,
                max_new_tokens=320,
                num_beams=1,
            )
        translated.extend(tokenizer.batch_decode(
            generated, skip_special_tokens=True))
        print(
            f"{language}: {min(start + len(batch), len(strings))}/"
            f"{len(strings)}",
            flush=True,
        )

    if len(translated) != len(fields):
        raise RuntimeError("translation count does not match field count")
    for (container, key), value in zip(fields, translated):
        container[key] = normalize_brand(value.strip())

    localized["language"] = language
    for lesson in localized["lessons"]:
        for scene in lesson["scenes"]:
            scene["speech_text"] = speech_text(scene["narration"], language)
    return localized


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", choices=LANGUAGES,
        default=list(LANGUAGES))
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--append-missing", action="store_true",
        help=("translate only lessons absent from an existing localized "
              "catalog, preserving every reviewed existing lesson"))
    args = parser.parse_args()

    source = json.loads(CATALOG.read_text())
    for language in args.languages:
        target = ROOT / "catalog" / f"lessons_{language}.json"
        if target.exists() and args.append_missing:
            localized = json.loads(target.read_text())
            source_ids = [lesson["id"] for lesson in source["lessons"]]
            current_ids = [lesson["id"] for lesson in localized["lessons"]]
            if current_ids != source_ids[:len(current_ids)]:
                raise RuntimeError(
                    f"{language} lesson ids are not an exact English prefix")
            missing = source["lessons"][len(current_ids):]
            if not missing:
                print(f"current {target}")
                continue
            partial = {
                "schema": source["schema"],
                "title": source["title"],
                "series": [],
                "lessons": missing,
            }
            translated = translate_language(
                partial, language, args.batch_size, args.threads, args.device)
            localized["lessons"].extend(translated["lessons"])
            localized["language"] = language
            target.write_text(
                json.dumps(localized, indent=2, ensure_ascii=False) + "\n")
            print(f"appended {len(missing)} lessons to {target}", flush=True)
            continue
        if target.exists() and not args.force:
            print(f"skip existing {target}")
            continue
        localized = translate_language(source, language, args.batch_size,
                                       args.threads, args.device)
        target.write_text(
            json.dumps(localized, indent=2, ensure_ascii=False) + "\n")
        print(target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
