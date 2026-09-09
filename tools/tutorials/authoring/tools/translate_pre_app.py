#!/usr/bin/env python3
"""Translate tutorial lessons into every caption language.

The default refreshes the four pre-application lessons. ``--all-lessons`` is
used when a newly added caption language needs complete course coverage.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
from pathlib import Path

from pronunciation import spoken_form


os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "catalog"
REVIEWED_LOCALIZATION = ROOT / "localization" / "reviewed"
MODEL = ROOT / "project" / "translation_models" / "nllb-200-distilled-600M"
LESSON_IDS = (
    "01_pypi_github",
    "02_conda_install",
    "03_pip_install",
    "04_platform_installers",
)

LANGUAGES = {
    "de": "deu_Latn",
    "sv": "swe_Latn",
    "is": "isl_Latn",
    "ja": "jpn_Jpan",
    "nb": "nob_Latn",
    "ko": "kor_Hang",
    "da": "dan_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "it": "ita_Latn",
    "pt-BR": "por_Latn",
    "zh-CN": "zho_Hans",
    "hi": "hin_Deva",
}

NARRATED = {"ja", "es", "fr", "it", "pt-BR", "zh-CN", "hi"}

CATALOG_TITLES = {
    "es": "Biblioteca completa de tutoriales de spaCR",
    "fr": "Bibliothèque complète des tutoriels spaCR",
    "hi": "spaCR की संपूर्ण ट्यूटोरियल लाइब्रेरी",
    "it": "Raccolta completa dei tutorial spaCR",
    "ja": "spaCR チュートリアル完全版",
    "pt-BR": "Biblioteca completa de tutoriais do spaCR",
    "zh-CN": "spaCR 完整教程库",
}

SERIES_TITLES = {
    "es": (
        "Primeros pasos y análisis principales",
        "Análisis con resolución temporal y modelos de segmentación",
        "Control de calidad de las anotaciones y ensayos biológicos",
        "Operaciones, informes y utilidades de datos",
        "Módulos adicionales registrados",
    ),
    "fr": (
        "Prise en main et analyses principales",
        "Analyses résolues dans le temps et modèles de segmentation",
        "Contrôle qualité des annotations et essais biologiques",
        "Opérations, rapports et utilitaires de données",
        "Modules supplémentaires enregistrés",
    ),
    "hi": (
        "आरंभ और मुख्य विश्लेषण",
        "समय-आधारित विश्लेषण और सेगमेंटेशन मॉडल",
        "एनोटेशन गुणवत्ता नियंत्रण और जैविक परीक्षण",
        "संचालन, रिपोर्टिंग और डेटा उपयोगिताएँ",
        "अतिरिक्त पंजीकृत मॉड्यूल",
    ),
    "it": (
        "Primi passi e analisi principali",
        "Analisi temporali e modelli di segmentazione",
        "Controllo qualità delle annotazioni e saggi biologici",
        "Operazioni, report e utilità per i dati",
        "Moduli aggiuntivi registrati",
    ),
    "ja": (
        "入門と主要解析",
        "時系列解析とセグメンテーションモデル",
        "アノテーションの品質管理と生物学的アッセイ",
        "運用、レポート、データユーティリティ",
        "追加登録モジュール",
    ),
    "pt-BR": (
        "Primeiros passos e análises principais",
        "Análises temporais e modelos de segmentação",
        "Controle de qualidade das anotações e ensaios biológicos",
        "Operações, relatórios e utilitários de dados",
        "Módulos adicionais registrados",
    ),
    "zh-CN": (
        "入门和核心分析",
        "时间分辨分析和分割模型",
        "注释质量控制和生物学分析",
        "运行、报告和数据工具",
        "其他已注册模块",
    ),
}

SECTION_LABELS = {
    "es": {
        "spaCR": "spaCR", "Core": "Módulos principales",
        "Segmentation models": "Modelos de segmentación",
        "Results and quality control": "Resultados y control de calidad",
        "Toxoplasma assays": "Ensayos de Toxoplasma",
        "Data and batch runs": "Datos y ejecuciones por lotes",
        "Data": "Datos", "Explore": "Explorar", "Design": "Diseño",
    },
    "fr": {
        "spaCR": "spaCR", "Core": "Modules principaux",
        "Segmentation models": "Modèles de segmentation",
        "Results and quality control": "Résultats et contrôle qualité",
        "Toxoplasma assays": "Essais Toxoplasma",
        "Data and batch runs": "Données et traitements par lots",
        "Data": "Données", "Explore": "Exploration",
        "Design": "Conception",
    },
    "hi": {
        "spaCR": "spaCR", "Core": "मुख्य मॉड्यूल",
        "Segmentation models": "सेगमेंटेशन मॉडल",
        "Results and quality control": "परिणाम और गुणवत्ता नियंत्रण",
        "Toxoplasma assays": "टोक्सोप्लाज़्मा परीक्षण",
        "Data and batch runs": "डेटा और बैच रन", "Data": "डेटा",
        "Explore": "अन्वेषण", "Design": "डिज़ाइन",
    },
    "it": {
        "spaCR": "spaCR", "Core": "Moduli principali",
        "Segmentation models": "Modelli di segmentazione",
        "Results and quality control": "Risultati e controllo qualità",
        "Toxoplasma assays": "Saggi su Toxoplasma",
        "Data and batch runs": "Dati ed esecuzioni batch", "Data": "Dati",
        "Explore": "Esplora", "Design": "Progettazione",
    },
    "ja": {
        "spaCR": "spaCR", "Core": "主要モジュール",
        "Segmentation models": "セグメンテーションモデル",
        "Results and quality control": "結果と品質管理",
        "Toxoplasma assays": "トキソプラズマアッセイ",
        "Data and batch runs": "データとバッチ実行", "Data": "データ",
        "Explore": "探索", "Design": "設計",
    },
    "pt-BR": {
        "spaCR": "spaCR", "Core": "Módulos principais",
        "Segmentation models": "Modelos de segmentação",
        "Results and quality control": "Resultados e controle de qualidade",
        "Toxoplasma assays": "Ensaios de Toxoplasma",
        "Data and batch runs": "Dados e execuções em lote", "Data": "Dados",
        "Explore": "Explorar", "Design": "Planejamento",
    },
    "zh-CN": {
        "spaCR": "spaCR", "Core": "核心模块",
        "Segmentation models": "分割模型",
        "Results and quality control": "结果与质量控制",
        "Toxoplasma assays": "弓形虫实验",
        "Data and batch runs": "数据和批处理运行", "Data": "数据",
        "Explore": "探索", "Design": "实验设计",
    },
}

# These tokens are proper names or literal commands.  NLLB is allowed to
# translate the prose around them but not to invent localized spellings.
PROTECTED = {
    "Explain CV Model": "EXPLAINCVMODELTOKEN",
    "Investigate Hit": "INVESTIGATEHITTOKEN",
    "Volcano Explorer": "VOLCANOEXPLORERTOKEN",
    "Parameter Sweep": "PARAMETERSWEEPTOKEN",
    "SHAP": "SHAPMETHODTOKEN",
    "CSV": "CSVFILETOKEN",
    "PDF": "PDFFILETOKEN",
    "PNG": "PNGFILETOKEN",
    "conda install conda-forge::spacr": "CONDAINSTALLCOMMANDTOKEN",
    "python -m pip install --upgrade spacr": "PIPUPGRADECOMMANDTOKEN",
    "pip install spacr": "PIPINSTALLCOMMANDTOKEN",
    "spacr-doctor": "SPACRDOCTORTOKEN",
    "--torch-backend auto": "TORCHBACKENDTOKEN",
    "python -m pip": "PYTHONPIPTOKEN",
    "GitHub Releases": "GITHUBRELEASESTOKEN",
    "SHA-256": "SHACHECKSUMTOKEN",
    "install.log": "INSTALLLOGTOKEN",
    "spaCR": "SPACRBRANDTOKEN",
    "PyPI": "PYPIPACKAGETOKEN",
    "conda-forge": "CONDAFORGETOKEN",
    "GitHub": "GITHUBTOKEN",
    "PyTorch": "PYTORCHTOKEN",
    "Cellpose": "CELLPOSETOKEN",
    "CQ1": "CQONETOKEN",
    "LoG": "LOGDETECTORTOKEN",
    "HeLa": "HELACELLTOKEN",
    "RAMP4": "RAMPFOURTOKEN",
    "GFP": "GFPTOKEN",
    "siRNA": "SIRNATOKEN",
    "nightly": "NIGHTLYBRANCHTOKEN",
}

MANUAL_BRANCH = {
    "de": "Auf GitHub bezeichnet main die freigegebene Quelllinie; nightly enthält neuere Änderungen, die noch getestet werden. Verwenden Sie nightly nur, wenn Sie ausdrücklich eine noch nicht veröffentlichte Änderung benötigen.",
    "sv": "På GitHub är main den publicerade källkodslinjen, medan nightly innehåller nyare ändringar som fortfarande testas. Använd endast nightly när du uttryckligen behöver en ännu opublicerad ändring.",
    "is": "Á GitHub er main útgefna kóðalínan, en nightly inniheldur nýrri breytingar sem eru enn í prófun. Notaðu nightly aðeins þegar þú þarft sérstaklega á óútgefinni breytingu að halda.",
    "nb": "På GitHub er main den publiserte kildekodelinjen, mens nightly inneholder nyere endringer som fortsatt testes. Bruk nightly bare når du uttrykkelig trenger en endring som enn ikke er utgitt.",
    "ko": "GitHub에서 main은 릴리스된 소스 코드 계열이고, nightly에는 아직 테스트 중인 최신 변경 사항이 포함됩니다. 아직 릴리스되지 않은 변경 사항이 꼭 필요한 경우에만 nightly를 사용하세요.",
    "da": "På GitHub er main den udgivne kildekodelinje, mens nightly indeholder nyere ændringer, der stadig testes. Brug kun nightly, når du specifikt har brug for en endnu ikke udgivet ændring.",
    "es": "En GitHub, main es la línea de código fuente publicada y nightly contiene cambios más recientes que todavía están en pruebas. Use nightly solo cuando necesite específicamente un cambio aún no publicado.",
    "fr": "Sur GitHub, main correspond à la branche source publiée, tandis que nightly contient des modifications plus récentes encore en cours de test. Utilisez nightly uniquement si vous avez précisément besoin d’une modification non encore publiée.",
    "hi": "GitHub पर main जारी स्रोत कोड की शाखा है, जबकि nightly में नए बदलाव होते हैं जिनका अभी परीक्षण चल रहा है। nightly का उपयोग केवल तभी करें जब आपको किसी अप्रकाशित बदलाव की विशेष रूप से आवश्यकता हो।",
    "it": "Su GitHub, main è la linea di codice sorgente pubblicata, mentre nightly contiene modifiche più recenti ancora in fase di test. Usa nightly solo quando ti serve specificamente una modifica non ancora pubblicata.",
    "pt-BR": "No GitHub, main é a linha de código-fonte publicada, enquanto nightly contém alterações mais recentes que ainda estão em teste. Use nightly somente quando precisar especificamente de uma alteração ainda não publicada.",
    "ja": "GitHub では main が公開済みのソース系列で、nightly にはテスト中の新しい変更が含まれます。未公開の変更が明確に必要な場合にだけ nightly を使用してください。",
    "zh-CN": "在 GitHub 上，main 是已发布的源代码分支，nightly 包含仍在测试中的较新更改。仅当您明确需要尚未发布的更改时，才使用 nightly。",
}

MANUAL_DOCTOR = {
    "de": "Führen Sie vor dem ersten Start spacr-doctor aus. Das Diagnoseprogramm prüft die aktive Installation, Qt, PyTorch, optionale Komponenten und die Hardware-Unterstützung und schlägt für jede fehlgeschlagene Prüfung eine Lösung vor.",
    "sv": "Kör spacr-doctor före den första starten. Diagnosverktyget kontrollerar den aktiva installationen, Qt, PyTorch, valfria komponenter och maskinvarustöd och föreslår en lösning för varje kontroll som misslyckas.",
    "is": "Keyrðu spacr-doctor áður en forritið er ræst í fyrsta sinn. Greiningartólið athugar virku uppsetninguna, Qt, PyTorch, valfrjálsa íhluti og vélbúnaðarstuðning og leggur til lausn fyrir hverja athugun sem mistekst.",
    "ja": "初回起動の前に spacr-doctor を実行します。診断ツールは、現在のインストール、Qt、PyTorch、オプション機能、ハードウェア対応を確認し、失敗した項目ごとに修正方法を提示します。",
    "nb": "Kjør spacr-doctor før første oppstart. Diagnoseverktøyet kontrollerer den aktive installasjonen, Qt, PyTorch, valgfrie komponenter og maskinvarestøtte og foreslår en løsning for hver kontroll som mislykkes.",
    "ko": "처음 실행하기 전에 spacr-doctor를 실행하세요. 진단 도구는 현재 설치, Qt, PyTorch, 선택적 구성 요소 및 하드웨어 지원을 확인하고 실패한 항목마다 해결 방법을 제안합니다.",
    "da": "Kør spacr-doctor før den første start. Diagnoseværktøjet kontrollerer den aktive installation, Qt, PyTorch, valgfrie komponenter og hardwareunderstøttelse og foreslår en løsning for hver kontrol, der mislykkes.",
    "es": "Ejecute spacr-doctor antes del primer inicio. La herramienta de diagnóstico comprueba la instalación activa, Qt, PyTorch, los componentes opcionales y la compatibilidad del hardware, y propone una solución para cada comprobación fallida.",
    "fr": "Exécutez spacr-doctor avant le premier démarrage. L'outil de diagnostic vérifie l'installation active, Qt, PyTorch, les composants facultatifs et la prise en charge du matériel, puis propose une solution pour chaque contrôle en échec.",
    "it": "Esegui spacr-doctor prima del primo avvio. Lo strumento diagnostico controlla l'installazione attiva, Qt, PyTorch, i componenti opzionali e il supporto hardware e propone una soluzione per ogni controllo non superato.",
    "pt-BR": "Execute spacr-doctor antes da primeira inicialização. A ferramenta de diagnóstico verifica a instalação ativa, o Qt, o PyTorch, os componentes opcionais e o suporte de hardware e sugere uma solução para cada verificação com falha.",
    "zh-CN": "首次启动前请运行 spacr-doctor。诊断工具会检查当前安装、Qt、PyTorch、可选组件和硬件支持，并为每个失败的检查提供修复建议。",
    "hi": "पहली बार शुरू करने से पहले spacr-doctor चलाएँ। यह निदान उपकरण सक्रिय इंस्टॉलेशन, Qt, PyTorch, वैकल्पिक घटकों और हार्डवेयर समर्थन की जाँच करता है तथा हर विफल जाँच के लिए समाधान सुझाता है।",
}


def protect(text: str) -> str:
    for term, token in PROTECTED.items():
        if term == "nightly":
            text = re.sub(rf"\b{re.escape(term)}\b", token, text)
        else:
            text = text.replace(term, token)
    return text


def restore(text: str) -> str:
    # Normal translation output preserves tokens.  The regex fallback catches
    # tokenizers that insert whitespace between characters in all-caps names.
    for term, token in PROTECTED.items():
        text = text.replace(token, term)
        spaced = r"\s*".join(re.escape(char) for char in token)
        text = re.sub(spaced, term, text)
    # NLLB occasionally shortens the compound GitHub Releases placeholder.
    text = re.sub(r"GITHUBRELEA[A-Z]*TOKEN", "GitHub Releases", text)
    # German generation may normalize the English word FOUR inside the
    # placeholder even though it is emitted as a single protected token.
    text = re.sub(r"RAMPFO(?:UR|R)TOKEN", "RAMP4", text)
    # NLLB occasionally rewrites the middle of the spaCR placeholder as a
    # natural-language conjunction (for example SPACRANDTOKEN in German).
    # The stable SPACR prefix and TOKEN suffix still identify it uniquely.
    text = re.sub(r"SPACR[A-Z]*TOKEN", "spaCR", text)
    # The sentence model occasionally duplicates one consonant inside a long
    # all-caps module placeholder (for example PARAMETTERSWEEPTOKEN in
    # Italian).  These anchored forms still identify one exact reviewed UI
    # label and cannot collide with ordinary prose.
    text = re.sub(r"EXPLAINCV[A-Z]*TOKEN", "Explain CV Model", text)
    text = re.sub(r"INVESTIGATEHIT[A-Z]*TOKEN", "Investigate Hit", text)
    text = re.sub(r"VOLCANOEXPLORER[A-Z]*TOKEN", "Volcano Explorer", text)
    text = re.sub(r"PARAMET+ERSWEEP[A-Z]*TOKEN", "Parameter Sweep", text)
    text = re.sub(r"SHAP[A-Z]*TOKEN", "SHAP", text)
    text = re.sub(r"CSV[A-Z]*TOKEN", "CSV", text)
    text = re.sub(r"PDF[A-Z]*TOKEN", "PDF", text)
    text = re.sub(r"PNG[A-Z]*TOKEN", "PNG", text)
    if any(token in text for token in PROTECTED.values()) or re.search(
            r"[A-Z]{5,}TOKEN", text):
        raise ValueError(f"unrestored translation token: {text}")
    # Brand names are protected explicitly. Avoid a broad normalization pass
    # here because it would rewrite the literal lowercase ``spacr`` inside
    # shell commands shown in captions.
    return text.strip()


def lesson_strings(lesson: dict) -> list[str]:
    values = [lesson["title"], lesson["description"], lesson["prerequisite"]]
    values.extend(lesson["objectives"])
    values.extend(scene["narration"] for scene in lesson["scenes"])
    return values


def apply_strings(source: dict, values: list[str], language: str) -> dict:
    localized = copy.deepcopy(source)
    cursor = 0
    localized["title"] = values[cursor]; cursor += 1
    localized["description"] = values[cursor]; cursor += 1
    localized["prerequisite"] = values[cursor]; cursor += 1
    localized["objectives"] = values[cursor:cursor + len(source["objectives"])]
    cursor += len(source["objectives"])
    for scene in localized["scenes"]:
        scene["narration"] = values[cursor]
        scene["speech_text"] = spoken_form(scene["narration"], language)
        cursor += 1
    if cursor != len(values):
        raise RuntimeError("localized field count does not match source")
    return localized


def reviewed_records() -> list[tuple[Path, str, dict]]:
    """Return legacy full reviews and schema-2 sparse lesson reviews."""
    records: list[tuple[Path, str, dict]] = []
    for path in sorted(REVIEWED_LOCALIZATION.glob("*.json")):
        reviewed = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(reviewed.get("lessons"), dict):
            for lesson_id, payload in reviewed["lessons"].items():
                languages = payload.get("languages", {})
                if not isinstance(languages, dict):
                    raise ValueError(
                        f"{path}: {lesson_id} languages must be a mapping"
                    )
                records.append((path, str(lesson_id), languages))
            continue
        metadata = reviewed.get("_meta", {})
        lesson_id = reviewed.get(
            "lesson_id", metadata.get("lesson_id", metadata.get("lesson"))
        )
        languages = reviewed.get("languages", {
            key: value for key, value in reviewed.items()
            if not key.startswith("_") and isinstance(value, list)
        })
        if lesson_id:
            records.append((path, str(lesson_id), languages))
    return records


def apply_manual_overrides(lessons: list[dict], language: str) -> None:
    """Replace safety-critical machine translations with reviewed wording."""
    by_id = {lesson["id"]: lesson for lesson in lessons}
    for path, lesson_id, languages in reviewed_records():
        if lesson_id not in by_id or language not in languages:
            continue
        language_review = languages[language]
        if isinstance(language_review, dict):
            narrations = language_review.get("narrations", {})
            for field in ("description", "prerequisite"):
                if field in language_review:
                    by_id[lesson_id][field] = language_review[field]
            if "objectives" in language_review:
                objectives = language_review["objectives"]
                if isinstance(objectives, dict):
                    for raw_index, value in objectives.items():
                        index = int(raw_index) - 1
                        if not 0 <= index < len(by_id[lesson_id]["objectives"]):
                            raise ValueError(
                                f"{path}: {lesson_id}/{language} objective "
                                f"index {raw_index} is out of range"
                            )
                        by_id[lesson_id]["objectives"][index] = value
                else:
                    if len(objectives) != len(by_id[lesson_id]["objectives"]):
                        raise ValueError(
                            f"{path}: {lesson_id}/{language} has "
                            f"{len(objectives)} reviewed objectives; expected "
                            f"{len(by_id[lesson_id]['objectives'])}"
                        )
                    by_id[lesson_id]["objectives"] = objectives
        else:
            narrations = language_review
        scenes = by_id[lesson_id]["scenes"]
        if isinstance(narrations, dict):
            reviewed_scenes = []
            for raw_index, narration in narrations.items():
                index = int(raw_index) - 1
                if not 0 <= index < len(scenes):
                    raise ValueError(
                        f"{path}: {lesson_id}/{language} scene index "
                        f"{raw_index} is out of range"
                    )
                reviewed_scenes.append((scenes[index], narration))
        elif len(narrations) != len(scenes):
            raise ValueError(
                f"{path}: {language} has {len(narrations)} reviewed scenes; "
                f"expected {len(scenes)}"
            )
        else:
            reviewed_scenes = list(zip(scenes, narrations))
        for scene, narration in reviewed_scenes:
            scene["narration"] = narration
            scene["speech_text"] = spoken_form(narration, language)
    if "01_pypi_github" in by_id:
        branch = by_id["01_pypi_github"]["scenes"][3]
        branch["narration"] = MANUAL_BRANCH[language]
        branch["speech_text"] = spoken_form(branch["narration"], language)
    if "02_conda_install" in by_id:
        doctor = by_id["02_conda_install"]["scenes"][5]
        doctor["narration"] = MANUAL_DOCTOR[language]
        doctor["speech_text"] = spoken_form(doctor["narration"], language)


def reviewed_lesson_ids(language: str) -> set[str]:
    """Return lessons with a maintained full or sparse review for a locale."""
    return {
        lesson_id for _path, lesson_id, languages in reviewed_records()
        if language in languages
    }


def reviewed_refresh_bases(sources: list[dict], language: str) -> list[dict]:
    """Preserve localized prose while refreshing generated lesson structure.

    A reviewed file may intentionally contain only narrations. Starting a
    ``--reviewed-only`` refresh from the English source would therefore replace
    localized descriptions, prerequisites, and objectives with English. Use
    the current localized lesson as the prose base, while copying routing and
    scene metadata from the current English catalog before applying the
    maintained review.
    """
    path = CATALOG / f"lessons_{language}.json"
    if language in NARRATED:
        existing_catalog = json.loads(path.read_text(encoding="utf-8"))
        existing_by_id = {
            lesson["id"]: lesson for lesson in existing_catalog["lessons"]
        }
    else:
        # Caption-only catalogs intentionally store only scene narration.
        # Reuse those reviewed captions as the prose base while refreshing the
        # scene structure from English. This preserves every existing scene
        # when a lesson gains a new one; the maintained review for that lesson
        # then replaces the complete current narration set below.
        caption_path = CATALOG / f"captions_{language}.json"
        existing_catalog = json.loads(caption_path.read_text(encoding="utf-8"))
        existing_by_id = {
            lesson["id"]: lesson for lesson in existing_catalog["lessons"]
        }
    refreshed: list[dict] = []
    for source in sources:
        localized = copy.deepcopy(existing_by_id.get(source["id"], source))
        # Routing metadata follows the English authority.  ``title`` and
        # ``section`` are user-facing text, so preserve their localized values
        # during a reviewed-only structural refresh.
        for key in ("id", "number", "slug", "series", "app_key"):
            localized[key] = copy.deepcopy(source[key])
        if "host_app_key" in source:
            localized["host_app_key"] = source["host_app_key"]
        else:
            localized.pop("host_app_key", None)

        existing_scenes = localized.get("scenes", [])
        localized["scenes"] = copy.deepcopy(source["scenes"])
        for scene, previous in zip(localized["scenes"], existing_scenes):
            for key in ("narration", "speech_text"):
                if key in previous:
                    scene[key] = previous[key]
        refreshed.append(localized)
    return refreshed


def translate_batch(tokenizer, model, strings: list[str], target_code: str,
                    batch_size: int, device: str,
                    num_beams: int = 3) -> list[str]:
    import torch

    # NLLB occasionally drops the second sentence in a technical caption.
    # Translate sentence-sized units and reassemble them one-for-one so an
    # omitted instruction cannot silently disappear from the course.
    groups = [re.split(r"(?<=[.!?])\s+", value) for value in strings]
    fragments = [fragment for group in groups for fragment in group]
    translated_fragments: list[str] = []
    forced_id = tokenizer.convert_tokens_to_ids(target_code)
    for start in range(0, len(fragments), batch_size):
        batch = [protect(value) for value in fragments[start:start + batch_size]]
        encoded = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=True, max_length=384)
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                forced_bos_token_id=forced_id,
                max_new_tokens=384,
                num_beams=num_beams,
            )
        translated_fragments.extend(restore(value) for value in tokenizer.batch_decode(
            generated, skip_special_tokens=True))
    translated: list[str] = []
    cursor = 0
    for group in groups:
        count = len(group)
        translated.append(" ".join(
            translated_fragments[cursor:cursor + count]))
        cursor += count
    if cursor != len(translated_fragments):
        raise RuntimeError("translated sentence count does not match source")
    return translated


def update_narrated_catalog(language: str, localized_lessons: list[dict]) -> Path:
    path = CATALOG / f"lessons_{language}.json"
    catalog = json.loads(path.read_text()) if path.exists() else {
        "schema": 1, "language": language, "series": [], "lessons": []}
    by_id = {lesson["id"]: lesson for lesson in catalog["lessons"]}
    for lesson in localized_lessons:
        by_id[lesson["id"]] = lesson
    source_order = json.loads((CATALOG / "lessons_en.json").read_text())["lessons"]
    catalog["lessons"] = [by_id[item["id"]] for item in source_order
                          if item["id"] in by_id]
    source_by_id = {lesson["id"]: lesson for lesson in source_order}
    section_labels = SECTION_LABELS[language]
    for lesson in catalog["lessons"]:
        english_section = source_by_id[lesson["id"]]["section"]
        lesson["section"] = section_labels[english_section]
    catalog["title"] = CATALOG_TITLES[language]
    catalog["series"] = [
        {"number": number, "title": title}
        for number, title in enumerate(SERIES_TITLES[language], start=1)
    ]
    catalog["language"] = language
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    return path


def update_caption_catalog(language: str, localized_lessons: list[dict]) -> Path:
    path = CATALOG / f"captions_{language}.json"
    if path.exists():
        catalog = json.loads(path.read_text())
    else:
        catalog = {"schema": 1, "language": language,
                   "source_language": "en", "lessons": []}
    by_id = {lesson["id"]: lesson for lesson in catalog["lessons"]}
    for lesson in localized_lessons:
        by_id[lesson["id"]] = {
            "id": lesson["id"],
            "scenes": [{"narration": scene["narration"]}
                       for scene in lesson["scenes"]],
        }
    source_order = json.loads((CATALOG / "lessons_en.json").read_text())["lessons"]
    catalog["lessons"] = [by_id[item["id"]] for item in source_order
                          if item["id"] in by_id]
    path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", choices=LANGUAGES,
                        default=list(LANGUAGES))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=3,
                        help="Translation beam width; use 1 for a faster draft")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--all-lessons", action="store_true",
                        help="Translate the complete catalog instead of lessons 1-4")
    parser.add_argument(
        "--lessons", nargs="+",
        help="Translate only these lesson ids, for example 07_mask")
    parser.add_argument(
        "--reviewed-only", action="store_true",
        help=("rebuild requested lessons only from maintained human-reviewed "
              "localizations, without loading a translation model"))
    args = parser.parse_args()
    english = json.loads((CATALOG / "lessons_en.json").read_text())
    if args.lessons:
        requested = set(args.lessons)
        sources = [lesson for lesson in english["lessons"]
                   if lesson["id"] in requested]
        missing = requested - {lesson["id"] for lesson in sources}
        if missing:
            raise ValueError(f"unknown lesson ids: {sorted(missing)}")
    else:
        sources = (english["lessons"] if args.all_lessons else
                   [lesson for lesson in english["lessons"]
                    if lesson["id"] in LESSON_IDS])

    if args.reviewed_only:
        requested_ids = {lesson["id"] for lesson in sources}
        for language in args.languages:
            missing = requested_ids.difference(reviewed_lesson_ids(language))
            if missing:
                raise ValueError(
                    f"{language} lacks reviewed localization for "
                    f"{sorted(missing)}")
            localized = reviewed_refresh_bases(sources, language)
            apply_manual_overrides(localized, language)
            target = (update_narrated_catalog(language, localized)
                      if language in NARRATED else
                      update_caption_catalog(language, localized))
            print(f"{language}: {target}", flush=True)
        return 0

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA translation requested but is not available")
    strings = [value for lesson in sources for value in lesson_strings(lesson)]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, src_lang="eng_Latn", local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, local_files_only=True)
    model = model.to(args.device).eval()

    for language in args.languages:
        translated = translate_batch(tokenizer, model, strings,
                                     LANGUAGES[language], args.batch_size,
                                     args.device, args.num_beams)
        localized = []
        cursor = 0
        for source in sources:
            count = len(lesson_strings(source))
            localized.append(apply_strings(
                source, translated[cursor:cursor + count], language))
            cursor += count
        apply_manual_overrides(localized, language)
        target = (update_narrated_catalog(language, localized)
                  if language in NARRATED else
                  update_caption_catalog(language, localized))
        print(f"{language}: {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
