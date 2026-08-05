#!/usr/bin/env python3
"""Generate caption-only tutorial catalogs with a local NLLB model.

The compact output intentionally contains only lesson ids and aligned scene
text.  Narration voices and lesson-page localization remain independent.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path


# Transformers probes optional backends during import.  This generator is
# PyTorch-only; disabling TensorFlow also avoids a known startup segfault on
# the tutorial workstation.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog" / "lessons_en.json"
OUTPUT = ROOT / "docs" / "source" / "_extra" / "tutorials" / "catalog"
LOCAL_MODEL = Path(
    "/media/carruthers/mnt3/claude/tutorials/project/translation_models/"
    "nllb-200-distilled-600M"
)

LANGUAGES = {
    "de": "deu_Latn",
    "sv": "swe_Latn",
    "is": "isl_Latn",
    "nb": "nob_Latn",
    "ko": "kor_Hang",
    "da": "dan_Latn",
}


def normalize_terms(text: str) -> str:
    """Keep product and model names consistent across machine translation."""
    substitutions = (
        (r"(?i)spa\s*[- ]?c\s*r", "spaCR"),
        (r"(?i)spacr", "spaCR"),
        (r"(?i)cell\s*pose", "Cellpose"),
        (r"(?i)xg\s*boost", "XGBoost"),
        (r"(?i)py\s*pi", "PyPI"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    return text.strip()


def caption_fields(catalog: dict) -> tuple[list[dict], list[str]]:
    scenes: list[dict] = []
    strings: list[str] = []
    for lesson in catalog["lessons"]:
        for scene in lesson["scenes"]:
            scenes.append(scene)
            strings.append(scene["narration"])
    return scenes, strings


def translate(source: dict, language: str, model_path: Path,
              batch_size: int, threads: int) -> dict:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True)
    model.eval()
    _, strings = caption_fields(source)
    translated: list[str] = []
    forced_id = tokenizer.convert_tokens_to_ids(LANGUAGES[language])

    for start in range(0, len(strings), batch_size):
        batch = strings[start:start + batch_size]
        encoded = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=320)
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

    lessons = []
    cursor = 0
    for lesson in source["lessons"]:
        count = len(lesson["scenes"])
        lessons.append({
            "id": lesson["id"],
            "scenes": [
                {"narration": normalize_terms(value)}
                for value in translated[cursor:cursor + count]
            ],
        })
        cursor += count
    if cursor != len(translated):
        raise RuntimeError("translated scene count does not match source")
    return {
        "schema": 1,
        "language": language,
        "source_language": "en",
        "lessons": lessons,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES,
                        required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(os.environ.get("SPACR_NLLB_MODEL", LOCAL_MODEL)),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = json.loads(SOURCE.read_text())
    for language in args.languages:
        target = OUTPUT / f"captions_{language}.json"
        if target.exists() and not args.force:
            print(f"skip existing {target}")
            continue
        localized = translate(
            source, language, args.model, args.batch_size, args.threads)
        target.write_text(
            json.dumps(localized, indent=2, ensure_ascii=False) + "\n")
        print(target, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
