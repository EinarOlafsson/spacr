"""Reviewed locale templates for generated README workflow markup."""
from __future__ import annotations

import re

WORKFLOW_MODULE_ALT_TEMPLATES = {
    "de": "API für {module} öffnen",
    "es": "Abrir la API de {module}",
    "fr": "Ouvrir l’API de {module}",
    "hi": "{module} API खोलें",
    "is": "Opna API-skjölin fyrir {module}",
    "ko": "{module} API 열기",
    "pt": "Abrir a API de {module}",
    "sv": "Öppna API-dokumentationen för {module}",
    "zh_CN": "打开 {module} API",
}

WORKFLOW_SECTION_LABELS = {
    "de": {
        "Data": "Daten",
        "Segmentation models": "Segmentierungsmodelle",
        "Results & QC": "Ergebnisse & Qualitätskontrolle",
        "Explore": "Erkunden",
        "Assays": "Assays",
        "Design": "Versuchsplanung",
    },
    "es": {
        "Data": "Datos",
        "Segmentation models": "Modelos de segmentación",
        "Results & QC": "Resultados y control de calidad",
        "Explore": "Explorar",
        "Assays": "Ensayos",
        "Design": "Diseño",
    },
    "fr": {
        "Data": "Données",
        "Segmentation models": "Modèles de segmentation",
        "Results & QC": "Résultats et contrôle qualité",
        "Explore": "Explorer",
        "Assays": "Essais",
        "Design": "Conception",
    },
    "hi": {
        "Data": "डेटा",
        "Segmentation models": "सेगमेंटेशन मॉडल",
        "Results & QC": "परिणाम और गुणवत्ता नियंत्रण",
        "Explore": "अन्वेषण",
        "Assays": "परख",
        "Design": "डिज़ाइन",
    },
    "is": {
        "Data": "Gögn",
        "Segmentation models": "Líkön fyrir hlutun",
        "Results & QC": "Niðurstöður og gæðaeftirlit",
        "Explore": "Kanna",
        "Assays": "Prófanir",
        "Design": "Hönnun",
    },
    "ko": {
        "Data": "데이터",
        "Segmentation models": "세그멘테이션 모델",
        "Results & QC": "결과 및 품질 관리",
        "Explore": "탐색",
        "Assays": "분석",
        "Design": "설계",
    },
    "pt": {
        "Data": "Dados",
        "Segmentation models": "Modelos de segmentação",
        "Results & QC": "Resultados e controle de qualidade",
        "Explore": "Explorar",
        "Assays": "Ensaios",
        "Design": "Planejamento",
    },
    "sv": {
        "Data": "Data",
        "Segmentation models": "Segmenteringsmodeller",
        "Results & QC": "Resultat och kvalitetskontroll",
        "Explore": "Utforska",
        "Assays": "Analyser",
        "Design": "Design",
    },
    "zh_CN": {
        "Data": "数据",
        "Segmentation models": "分割模型",
        "Results & QC": "结果与质控",
        "Explore": "探索",
        "Assays": "实验分析",
        "Design": "设计",
    },
}


def localize_workflow_markup(text: str, language: str) -> str:
    """Localize generated workflow headings/actions, retaining module names."""
    template = WORKFLOW_MODULE_ALT_TEMPLATES[language]

    def replace_alt(match: re.Match[str]) -> str:
        return (
            f"{match.group('indent')}:alt: "
            f"{template.format(module=match.group('module'))}"
        )

    localized = re.sub(
        r"(?m)^(?P<indent>\s*):alt: Open the (?P<module>.+) API$",
        replace_alt,
        str(text),
    )
    for source, target in WORKFLOW_SECTION_LABELS[language].items():
        localized = re.sub(
            rf"(?m)^\*\*{re.escape(source)}\*\*$",
            lambda _match, value=target: f"**{value}**",
            localized,
        )
    return localized
