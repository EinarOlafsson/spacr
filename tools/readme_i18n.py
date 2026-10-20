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


#: Labels of the README's generated model-zoo block, by language. English is
#: the key; a language or label missing here is written in English.
MODEL_ZOO_LABELS: dict[str, dict[str, str]] = {
    "de": {
        'Mask generator': 'Maskengenerator',
        'Default models · held-out scores': 'Standardmodelle · Hold-out-Scores',
        'Older models': 'Ältere Modelle',
        'Alternatives': 'Alternativen',
        'Model': 'Modell',
        'candidate': 'Kandidat',
    },
    "es": {
        'Mask generator': 'Generador de máscaras',
        'Default models · held-out scores': 'Modelos predeterminados · puntuaciones en datos reservados',
        'Older models': 'Modelos anteriores',
        'Alternatives': 'Alternativas',
        'Model': 'Modelo',
        'candidate': 'candidato',
    },
    "fr": {
        'Mask generator': 'Générateur de masques',
        'Default models · held-out scores': 'Modèles par défaut · scores hors apprentissage',
        'Older models': 'Modèles précédents',
        'Alternatives': 'Alternatives',
        'Model': 'Modèle',
        'candidate': 'candidat',
    },
    "pt": {
        'Mask generator': 'Gerador de máscaras',
        'Default models · held-out scores': 'Modelos padrão · pontuações em dados reservados',
        'Older models': 'Modelos anteriores',
        'Alternatives': 'Alternativas',
        'Model': 'Modelo',
        'candidate': 'candidato',
    },
    "sv": {
        'Mask generator': 'Maskgenerator',
        'Default models · held-out scores': 'Standardmodeller · poäng på undanhållna data',
        'Older models': 'Äldre modeller',
        'Alternatives': 'Alternativ',
        'Model': 'Modell',
        'candidate': 'kandidat',
    },
    "is": {
        'Mask generator': 'Grímugerð',
        'Default models · held-out scores': 'Sjálfgefin líkön · skor á ósnortnum gögnum',
        'Older models': 'Eldri líkön',
        'Alternatives': 'Valkostir',
        'Model': 'Líkan',
        'candidate': 'frambjóðandi',
    },
    "ko": {
        'Mask generator': '마스크 생성기',
        'Default models · held-out scores': '기본 모델 · 홀드아웃 점수',
        'Older models': '이전 모델',
        'Alternatives': '대안',
        'Model': '모델',
        'candidate': '후보',
    },
    "hi": {
        'Mask generator': 'मास्क जनरेटर',
        'Default models · held-out scores': 'डिफ़ॉल्ट मॉडल · होल्ड-आउट स्कोर',
        'Older models': 'पुराने मॉडल',
        'Alternatives': 'विकल्प',
        'Model': 'मॉडल',
        'candidate': 'उम्मीदवार',
    },
    "zh_CN": {
        'Mask generator': '掩膜生成器',
        'Default models · held-out scores': '默认模型 · 留出集得分',
        'Older models': '旧模型',
        'Alternatives': '替代模型',
        'Model': '模型',
        'candidate': '候选',
    },
}


def localize_model_zoo_label(label: str, language: str) -> str:
    """The README model-zoo ``label`` in ``language``, English when absent."""
    return MODEL_ZOO_LABELS.get(language, {}).get(label, label)


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


#: The hardware legend's three entries, one label per colour, in each
#: language. AI-reviewed (Claude, 2026-10-12, second pass), no native-speaker
#: signoff.
#:
#: The legend is drawn as an image now, a coloured dot before each label, so
#: it needs the three labels apart. The accepted one-line translations in
#: ``HARDWARE_LEGEND_TARGETS`` had lost the dots and run the three entries
#: together ("Soutien (stable) Soutien (bêta) CPU seulement"), which an image
#: cannot split back into dot and label; they stay as the reviewed record.
HARDWARE_LEGEND_LABELS = {
    "sv": ("stöds (stabilt)", "implementerat (beta)", "endast CPU-stöd"),
    "de": ("unterstützt (stabil)", "implementiert (Beta)",
           "nur CPU-Unterstützung"),
    "es": ("compatible (estable)", "implementado (beta)",
           "solo compatible con CPU"),
    "zh_CN": ("支持（稳定）", "已实现（beta）", "仅支持 CPU"),
    "pt": ("suportado (estável)", "implementado (beta)",
           "apenas suporte a CPU"),
    "hi": ("समर्थित (स्थिर)", "लागू (बीटा)", "केवल CPU समर्थन"),
    "ko": ("지원 (안정)", "구현됨 (베타)", "CPU만 지원"),
    "is": ("með stuðningi (stöðugt)", "útfært (beta)",
           "aðeins CPU-stuðningur"),
    "fr": ("pris en charge (stable)", "implémenté (bêta)",
           "uniquement sur CPU"),
}


def localize_hardware_markup(text: str, language: str) -> str:
    """The hardware table's legend in ``language``."""
    stable, beta, cpu = HARDWARE_LEGEND_LABELS[language]
    legend = (f"\U0001F7E2 {stable} \u2003 \U0001F7E3 {beta} "
              f"\u2003 \U0001F534 {cpu}")
    localized = re.sub(
        rf"(?m)^{re.escape(HARDWARE_LEGEND_SOURCE)}$",
        lambda _match: legend,
        str(text),
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
