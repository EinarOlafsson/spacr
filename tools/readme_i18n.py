"""Reviewed locale templates for generated README workflow markup."""
from __future__ import annotations

from html import escape, unescape
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


WORKFLOW_ORGANISM_NOTE = "Organism-specific image analysis and quantitative assay readouts."
WORKFLOW_ORGANISM_NOTES = {'sv': 'Organismspecifik bildanalys och kvantitativa resultat från biologiska '
       'analyser.',
 'de': 'Organismusspezifische Bildanalyse und quantitative Assay-Ergebnisse.',
 'es': 'Análisis de imágenes específico del organismo y resultados cuantitativos de '
       'ensayos.',
 'zh_CN': '针对特定生物体的图像分析和定量实验测定结果。',
 'pt': 'Análise de imagens específica do organismo e resultados quantitativos de '
       'ensaios.',
 'hi': 'जीव-विशिष्ट छवि विश्लेषण और परिमाणात्मक परीक्षण परिणाम।',
 'ko': '생물체별 이미지 분석과 정량적 분석 결과.',
 'is': 'Myndgreining fyrir tilteknar lífverur og megindlegar niðurstöður líffræðilegra '
       'prófana.',
 'fr': 'Analyse d’images spécifique à l’organisme et résultats quantitatifs des '
       'essais.'}

HARDWARE_LEGEND_SOURCE = "🟢 supported (stable) \u2003 🟣 implemented (beta) \u2003 🔴 CPU support only"
HARDWARE_LEGEND_TARGETS = {'sv': 'Stödda (stabila)  och genomförda (beta) - CPU stöd endast',
 'de': 'Nur unterstützte (stabile) Unterstützung implementierte (beta) Unterstützung '
       'CPU',
 'es': 'soportado (estable)  implementado (beta) CPU soporte solamente',
 'zh_CN': '支持(稳定) 实施(beta) 🔴 CPU 仅支持',
 'pt': 'suportado (estável)  implementado (beta) ? CPU apenas suporte',
 'hi': 'समर्थित (स्थिर)  लागू (बेटा) 🔴 CPU समर्थन केवल',
 'ko': '지원 (안정)  구현 (베타) 🔴 CPU 지원만',
 'is': 'stuðlað (stabil)  framkvæmd (beta) 🔴 CPU stuðning aðeins',
 'fr': 'Soutien (stable) Soutien (bêta) CPU seulement'}

for _language, _label in {'sv': 'Organism', 'de': 'Organismus', 'es': 'Organismo', 'zh_CN': '生物体', 'pt': 'Organismo', 'hi': 'जीव', 'ko': '생물체', 'is': 'Lífvera', 'fr': 'Organisme'}.items():
    WORKFLOW_SECTION_LABELS[_language]["Organism"] = _label


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
    localized = re.sub(
        r'\balt="Open the (?P<module>[^"]+) API"',
        lambda match: 'alt="' + escape(
            template.format(module=unescape(match.group("module"))),
            quote=True,
        ) + '"',
        localized,
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
    localized = re.sub(
        rf"(?m)^{re.escape(WORKFLOW_ORGANISM_NOTE)}$",
        lambda _match: WORKFLOW_ORGANISM_NOTES[language],
        localized,
    )
    return localized


def localize_hardware_markup(text: str, language: str) -> str:
    """Retain the accepted native hardware legend during normal regeneration."""
    return re.sub(
        rf"(?m)^{re.escape(HARDWARE_LEGEND_SOURCE)}$",
        lambda _match: HARDWARE_LEGEND_TARGETS[language],
        str(text),
    )


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


_HEADING = re.compile(
    r"(?m)^(?P<title>\S[^\n]*)\n(?P<rule>[=~^\-'\"`#*+])(?P=rule){2,}[ \t]*$"
)
_INTERNAL_REFERENCE = re.compile(r"`(?P<name>[^`<>\n]+?)`_")


def localize_internal_references(source: str, localized: str) -> str:
    """Point rST section references at the translated heading they name.

    ```Citing spaCR`_`` is an *implicit* reference: docutils resolves it
    against the section title spelt the same way. Translation rewrites the
    title and leaves the reference alone -- inline markup is protected from
    the model -- so all nine READMEs referred to a heading that no longer
    existed and docutils raised ``Unknown target name: "citing spacr"``.
    GitHub renders an unresolved reference as its own text, so the "see
    below" in the licence paragraph quietly stopped being a link in every
    language except English, and the page still looked finished.

    Headings are matched by POSITION, not by text: the localized document
    is the same document, so the *n*-th heading is the translation of the
    *n*-th English one. Matching by text cannot work here -- the whole
    point is that the text changed. If the two disagree on how many
    headings they have the pass returns the input untouched, because a
    mis-aligned rename would point the reference at the wrong section,
    which is worse than leaving it broken where a gate can see it.
    """
    source_titles = [
        match.group("title").strip() for match in _HEADING.finditer(source)
    ]
    localized_titles = [
        match.group("title").strip() for match in _HEADING.finditer(localized)
    ]
    if len(source_titles) != len(localized_titles):
        return localized
    renames = {
        english: translated
        for english, translated in zip(source_titles, localized_titles)
        if english != translated
    }
    if not renames:
        return localized

    def replace(match: re.Match[str]) -> str:
        name = match.group("name").strip()
        return f"`{renames.get(name, match.group('name'))}`_"

    return _INTERNAL_REFERENCE.sub(replace, localized)
