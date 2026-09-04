"""Reviewed locale templates for generated README workflow markup."""
from __future__ import annotations

import re
import unicodedata

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
        "Core": 'Kern',
        "Tools": 'Werkzeuge',
        "Data": "Daten",
        "Segmentation models": "Segmentierungsmodelle",
        "Results & QC": "Ergebnisse & Qualitätskontrolle",
        "Explore": "Erkunden",
        "Assays": "Assays",
        "Design": "Versuchsplanung",
    },
    "es": {
        "Core": 'Principal',
        "Tools": 'Herramientas',
        "Data": "Datos",
        "Segmentation models": "Modelos de segmentación",
        "Results & QC": "Resultados y control de calidad",
        "Explore": "Explorar",
        "Assays": "Ensayos",
        "Design": "Diseño",
    },
    "fr": {
        "Core": 'Cœur',
        "Tools": 'Outils',
        "Data": "Données",
        "Segmentation models": "Modèles de segmentation",
        "Results & QC": "Résultats et contrôle qualité",
        "Explore": "Explorer",
        "Assays": "Essais",
        "Design": "Conception",
    },
    "hi": {
        "Core": 'मुख्य',
        "Tools": 'उपकरण',
        "Data": "डेटा",
        "Segmentation models": "सेगमेंटेशन मॉडल",
        "Results & QC": "परिणाम और गुणवत्ता नियंत्रण",
        "Explore": "अन्वेषण",
        "Assays": "एसे",
        "Design": "डिज़ाइन",
    },
    "is": {
        "Core": 'Kjarni',
        "Tools": 'Verkfæri',
        "Data": "Gögn",
        "Segmentation models": "Líkön fyrir hlutun",
        "Results & QC": "Niðurstöður og gæðaeftirlit",
        "Explore": "Kanna",
        "Assays": "Prófanir",
        "Design": "Hönnun",
    },
    "ko": {
        "Core": '핵심',
        "Tools": '도구',
        "Data": "데이터",
        "Segmentation models": "세그멘테이션 모델",
        "Results & QC": "결과 및 품질 관리",
        "Explore": "탐색",
        "Assays": "어세이",
        "Design": "설계",
    },
    "pt": {
        "Core": 'Principal',
        "Tools": 'Ferramentas',
        "Data": "Dados",
        "Segmentation models": "Modelos de segmentação",
        "Results & QC": "Resultados e controle de qualidade",
        "Explore": "Explorar",
        "Assays": "Ensaios",
        "Design": "Planejamento",
    },
    "sv": {
        "Core": 'Kärna',
        "Tools": 'Verktyg',
        "Data": "Data",
        "Segmentation models": "Segmenteringsmodeller",
        "Results & QC": "Resultat och kvalitetskontroll",
        "Explore": "Utforska",
        "Assays": "Analyser",
        "Design": "Design",
    },
    "zh_CN": {
        "Core": '核心',
        "Tools": '工具',
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
        # THE SAME LABEL ALSO APPEARS AS A SECTION HEADING, and for a while
        # only the bold form was rewritten. The workflow block writes the
        # four bands as underlined headings --
        #
        #     Core
        #     ^^^^
        #
        # -- so a bold-only pattern matched none of them, and every band
        # heading was dropped from all nine translated READMEs: the canonical
        # README carries four and each localized one carried zero. No gate saw
        # it, because the gate counts ``**`` pairs and a heading that vanishes
        # takes its markup with it.
        localized = re.sub(
            rf"(?m)^{re.escape(source)}\n(?P<rule>[=~^\-'\"`#*+])(?P=rule){{2,}}$",
            lambda match, value=target: (
                f"{value}\n"
                f"{match.group('rule') * _underline_width(value)}"
            ),
            localized,
        )
    return localized


def _underline_width(value: str) -> int:
    """Return the column width an rST underline needs for ``value``.

    NOT ``len``. An underline shorter than its title is a docutils error, and
    a CJK glyph occupies two terminal columns while counting as one character
    -- so ``len`` under-measures every Chinese, Japanese and Korean heading
    and over-measures nothing. Combining marks are the opposite case and take
    no width of their own, which matters for the Hindi bands.
    """
    return sum(
        0 if unicodedata.combining(character)
        else 2 if unicodedata.east_asian_width(character) in {"F", "W"}
        else 1
        for character in value
    )
