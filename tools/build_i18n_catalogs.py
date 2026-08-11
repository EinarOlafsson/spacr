#!/usr/bin/env python3
"""Build and audit spaCR's external runtime localization catalogs.

The application keeps a compact hand-reviewed chrome catalog in
``spacr.qt.i18n``.  This tool extracts the much larger surfaces directly from
their canonical English sources:

* every setting label and scientific tooltip body;
* every written settings-category explanation; and
* static text owned by Qt widgets, actions, dialogs and notices.

Translations are generated with permissively licensed Helsinki OPUS models
or M2M100, according to the target language.  Those checkpoints use
Apache-2.0, CC-BY-4.0 or MIT terms, unlike the
research-only NLLB checkpoint used by the separate non-commercial tutorial
project.  Identifiers, paths, URLs, format fields, units and scientific brand
names are protected before generation.  The output is one ordinary Python
module per language under ``spacr/qt/i18n_catalogs`` plus standalone installer
JSON under ``packaging/i18n``; no translated prose is inserted into
application functions.

Run ``--sources-only`` first to refresh the English manifest.  ``--audit``
performs no generation and exits non-zero on missing/stale keys, placeholder
damage, leaked protection tokens or suspicious untranslated prose.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import json
from pathlib import Path
import pprint
import re
import sys
from typing import Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
CATALOG_DIR = ROOT / "spacr" / "qt" / "i18n_catalogs"

MODEL_SPECS = {
    "sv": ("Helsinki-NLP/opus-mt-en-sv", "en-sv", "Apache-2.0", ""),
    "de": ("Helsinki-NLP/opus-mt-en-de", "en-de", "CC-BY-4.0", ""),
    "es": ("Helsinki-NLP/opus-mt-en-es", "en-es", "Apache-2.0", ""),
    "zh_CN": ("facebook/m2m100_418M", "../m2m100_418M", "MIT", ""),
    "pt": (
        "Helsinki-NLP/opus-mt-tc-big-en-pt", "en-pt", "CC-BY-4.0",
        ">>por<< ",
    ),
    # M2M100 gives materially more coherent technical prose than the rejected
    # Chinese, Hindi, Korean and Icelandic OPUS outputs.  Those checkpoints
    # produced repetition, corrupted mixed script, or severe false friends in
    # their own simple examples.  M2M100 is the stable MIT-licensed replacement.
    "hi": ("facebook/m2m100_418M", "../m2m100_418M", "MIT", ""),
    "ko": ("facebook/m2m100_418M", "../m2m100_418M", "MIT", ""),
    "is": ("facebook/m2m100_418M", "../m2m100_418M", "MIT", ""),
    "fr": ("Helsinki-NLP/opus-mt-en-fr", "en-fr", "Apache-2.0", ""),
}

NATIVE_LANGUAGE_NAMES = {
    "sv": "Svenska", "de": "Deutsch", "es": "Español",
    "zh_CN": "简体中文", "pt": "Português", "hi": "हिन्दी",
    "ko": "한국어", "is": "Íslenska", "fr": "Français",
}

# Calls whose literal arguments are presentation text.  Dynamic values and
# table/model data are deliberately absent: localization must not mutate them.
_TEXT_METHODS = {
    "setText", "setTitle", "setToolTip", "setStatusTip",
    "setPlaceholderText", "setAccessibleName", "setAccessibleDescription",
    "setInformativeText", "setDetailedText", "append_notice",
}
_TEXT_CONSTRUCTORS = {
    "QLabel", "QPushButton", "QToolButton", "QCheckBox", "QRadioButton",
    "QGroupBox", "QAction",
}
_DIALOG_METHODS = {"information", "warning", "critical", "question"}
_FILE_DIALOG_METHODS = {
    "getOpenFileName", "getOpenFileNames", "getSaveFileName",
    "getExistingDirectory",
}
_INPUT_DIALOG_METHODS = {"getText", "getInt", "getDouble", "getItem"}

_IDENTITY_TEXT = {
    "API", "CPU", "CUDA", "CV", "DNA", "EC50", "FOV", "GPU", "JSON",
    "ML", "NaN", "PCA", "PDF", "PNG", "QC", "RGB", "RNA", "ROI",
    "SAM", "SHAP", "SQL", "TIFF", "UMAP", "ViT", "X", "XGBoost", "Y",
    "Z", "log10", "spaCR", "t", "x", "y", "µM", "µm/pixel",
    "|Tutorials|",
}

_PROTECTED_TERMS = tuple(sorted({
    "spaCR", "Cellpose", "PyTorch", "TensorBoard", "NumPy", "pandas",
    "SciPy", "scikit-image", "scikit-learn", "XGBoost", "LightGBM",
    "CatBoost", "Grad-CAM", "Graphviz", "Napari", "AnnData", "Scanpy",
    "OMERO", "OME-Zarr", "TIFF", "OME-TIFF", "SQLite", "HDF5", "CSV",
    "JSON", "Parquet", "PyPI", "GitHub", "conda-forge", "Qt", "PySide6",
    "CUDA", "CPU", "GPU", "UMAP", "PCA", "t-SNE", "CNN", "ViT", "SAM",
    "SHAP", "API", "RGB", "PDF", "PNG", "FOV", "ROI", "QC", "EC50",
    "CRISPR", "gRNA", "siRNA", "DNA", "RNA", "DAPI", "GFP", "LoG",
    "NVIDIA", "Python", "Windows", "Linux", "macOS", "OpenGL", "XCB",
    "PATH", "SPEC", "SSH", "Slurm", "HPC",
}, key=len, reverse=True))

_PROTECT_PATTERNS = (
    re.compile(r"^:[A-Za-z][\w-]*(?:\s+[^:]+)?:", re.MULTILINE),
    re.compile(r"^\s*(?:[*-]|#\.)\s+", re.MULTILINE),
    re.compile(r"\|[A-Za-z][^|\n]*\|"),
    re.compile(r"\*\*"),
    re.compile(r"</?[A-Za-z][^>]*>"),
    re.compile(r"\{[^{}]+\}"),
    re.compile(r":(?:class|func|mod|meth|attr|data|doc):`[^`]+`"),
    re.compile(r"``[^`]+``|`[^`]+`_?"),
    re.compile(r"https?://\S+"),
    re.compile(r"(?<!\w)(?:--?[A-Za-z][\w-]*)(?!\w)"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    # Literal option values are part of the settings API, even when they are
    # embedded in otherwise translatable prose.
    re.compile(r"'[A-Za-z][A-Za-z0-9_.:/-]*'|\"[A-Za-z][A-Za-z0-9_.:/-]*\""),
    re.compile(r">=|<=|==|!="),
    re.compile(r"%(?:\d+\$)?[sd]"),
)

_PROTECT_RE = re.compile(
    "|".join(
        [f"(?:{pattern.pattern})" for pattern in _PROTECT_PATTERNS]
        + [
            r"(?<![\w-])(?:"
            + "|".join(re.escape(term) for term in _PROTECTED_TERMS)
            + r")(?![\w-])"
        ]
    ),
    re.MULTILINE,
)

_TOKEN_RE = re.compile(r"ZXQ(\d{4})QXZ|<\s*[xX]\s*\d+\s*>")

# These are deliberately small and human-readable.  They correct observed
# context errors rather than trying to become a second translation engine.
CONTEXT_REPLACEMENTS: Mapping[str, tuple[tuple[str, str], ...]] = {
    "sv": (
        ("Cellpose flow-fält", "Cellpose-flödesfält"),
        ("Cellpose flowfält", "Cellpose-flödesfält"),
        ("löpande ansökan", "körande programmet"),
    ),
    "de": (
        ("Cellpose-Flow-Feld", "Cellpose-Flussfeld"),
        ("Flow-Feld", "Flussfeld"),
    ),
    "es": (
        ("más pequeños de este umbral", "más pequeños que este umbral"),
    ),
    "zh_CN": (
        ("Cellpose 流量字段", "Cellpose 流场"),
        ("流量字段", "流场"),
        ("这个门", "此阈值"),
        ("输入文件包含", "输入文件夹包含"),
    ),
    "pt": (),
    "hi": (),
    "ko": (),
    "is": (
        ("Cellpose flow-svæðið", "Cellpose-flæðisviðið"),
        ("flow-svæðið", "flæðisviðið"),
        ("innskránni möppuna", "inntaksmöppuna"),
    ),
    "fr": (
        ("champ de débit Cellpose", "champ de flux Cellpose"),
        ("champ de débit", "champ de flux"),
        ("L'criblage", "Le criblage"),
        ("l'criblage", "le criblage"),
        ("L’criblage", "Le criblage"),
        ("l’criblage", "le criblage"),
    ),
}

# Source-conditioned repairs for common scientific false friends.  These are
# deliberately narrower than CONTEXT_REPLACEMENTS: for example, Chinese 门 is
# ordinary in navigation prose but means the wrong thing for a cytometry gate.
SOURCE_CONTEXT_REPLACEMENTS: Mapping[
    str, tuple[tuple[str, str, str], ...]
] = {
    "sv": (
        (r"\bgates?\b", "stängnings", "gate-"),
        (r"\bwells?\b", "bra", "brunn"),
        (r"\bruns?\b|\brunning\b", "loppet", "körningen"),
        (r"\bruns?\b|\brunning\b", "lopp", "körning"),
        (r"\bcrops?\b", "grödor", "bildutsnitt"),
        (r"\bcrops?\b", "gröda", "bildutsnitt"),
        (r"\bworkers?\b", "arbetare", "worker-processer"),
        (r"\bscreens?\b", "skärmar", "screeningar"),
        (r"\bscreens?\b", "skärm", "screening"),
        (r"\bheadless(?:ly)?\b", "utan huvud", "utan grafiskt gränssnitt"),
        (r"\bpipelines?\b", "rörledningar", "arbetsflöden"),
    ),
    "de": (
        (r"\bgates?\b", "Tor", "Gate"),
        (r"\bflow field\b", "Durchflussfeld", "Flussfeld"),
        (r"\bwells?\b", "Brunnen", "Wells"),
        (r"\bguides?\b", "Leitfäden", "Guide-RNAs"),
        (r"\bguides?\b", "Leitfaden", "Guide-RNA"),
        (r"\bruns?\b|\brunning\b", "Rennen", "Ausführung"),
        (r"\bcrops?\b", "Kulturen", "Bildausschnitte"),
        (r"\bcrops?\b", "Kultur", "Bildausschnitt"),
        (r"\bworkers?\b", "Arbeiter", "Worker-Prozesse"),
        (r"\bbatches?\b", "Stapel", "Batches"),
        (r"\bscreens?\b", "Bildschirme", "Screenings"),
        (r"\bscreens?\b", "Bildschirm", "Screening"),
        (r"\bheadless(?:ly)?\b", "kopflos", "ohne grafische Oberfläche"),
        (r"\bpipelines?\b", "Rohrleitungen", "Pipelines"),
        (r"\bguides?\b", "Leitidentitäten", "Guide-RNA-Identitäten"),
    ),
    "es": (
        (r"\bgates?\b", "puerta", "compuerta"),
        (r"\bmasks?\b", "mascarilla", "máscara"),
        (r"\bwells?\b", "bien", "pocillo"),
        (r"\bhits?\b", "golpes", "aciertos"),
        (r"\bruns?\b|\brunning\b", "carreras", "ejecuciones"),
        (r"\bruns?\b|\brunning\b", "carrera", "ejecución"),
        (r"\bruns?\b|\brunning\b", "recorridos", "ejecuciones"),
        (r"\bruns?\b|\brunning\b", "recorrido", "ejecución"),
        (r"\bcrops?\b", "cultivos", "recortes"),
        (r"\bcrops?\b", "cultivo", "recorte"),
        (r"\bcrops?\b", "culturas", "recortes"),
        (r"\bcrops?\b", "cultura", "recorte"),
        (r"\bworkers?\b", "trabajadores", "procesos de trabajo"),
        (r"\bbatches?\b", "lotes", "batches"),
        (r"\bscreens?\b", "pantallas", "cribados"),
        (r"\bscreens?\b", "pantalla", "cribado"),
        (r"\bheadless(?:ly)?\b", "sin cabeza", "sin interfaz gráfica"),
        (r"\bpipelines?\b", "tuberías", "flujos de trabajo"),
        (r"\bpipelines?\b", "tubos", "flujos de trabajo"),
    ),
    "zh_CN": (
        (r"\bgates?\b", "箱门", "箱式门控"),
        (r"\bgates?\b", "门", "门控"),
        (r"\bflow field\b", "流量", "流"),
        (r"\bmasks?\b", "口罩", "掩膜"),
        (r"\bcells?\b", "电池", "细胞"),
        (r"\bwells?\b", "井", "孔"),
        (r"\bhits?\b", "点击", "命中"),
        (r"\bguides?\b", "指南", "向导 RNA"),
        (r"\bmasks?\b", "面具", "掩膜"),
        (r"\bplates?\b", "板块", "孔板"),
        (r"\bruns?\b|\brunning\b", "赛跑", "运行"),
        (r"\bruns?\b|\brunning\b", "一跑", "一次运行"),
        (r"\bcrops?\b", "作物", "图像裁剪"),
        (r"\bworkers?\b", "工人", "工作进程"),
        (r"\bpass/fail\b", "通行证/不及格", "通过/失败"),
        (r"\bscreens?\b", "屏幕", "筛选"),
        (r"\bheadless(?:ly)?\b", "无头", "无图形界面"),
        (r"\bpipelines?\b", "管道", "流程"),
    ),
    "pt": (
        (r"\bgates?\b", "portão", "gate"),
        (r"\bmasks?\b", "máscara facial", "máscara"),
        (r"\bscan\b", "Verificar", "Escanear"),
        (r"\bwells?\b", "bem", "poço"),
        (r"\bhits?\b", "golpes", "acertos"),
        (r"\bruns?\b|\brunning\b", "corridas", "execuções"),
        (r"\bruns?\b|\brunning\b", "corrida", "execução"),
        (r"\bcrops?\b", "culturas", "recortes"),
        (r"\bcrops?\b", "cultura", "recorte"),
        (r"\bworkers?\b", "trabalhadores", "processos worker"),
        (r"\bbatches?\b", "lotes", "batches"),
        (r"\bscreens?\b", "telas", "triagens"),
        (r"\bscreens?\b", "tela", "triagem"),
        (r"\bheadless(?:ly)?\b", "sem cabeça", "sem interface gráfica"),
        (r"\bpipelines?\b", "tubos", "pipelines"),
    ),
    "hi": (
        (r"\bgates?\b", "द्वार", "गेट"),
        (r"\bmasks?\b", "चेहरे का मास्क", "मास्क"),
        (r"\bcells?\b", "बैटरी", "कोशिका"),
        (r"\bwells?\b", "कुआँ", "वेल"),
        (r"\bhits?\b", "मार", "हिट"),
        (r"\bruns?\b|\brunning\b", "दौड़", "रन"),
        (r"\bcrops?\b", "फसल", "क्रॉप"),
        (r"\bworkers?\b", "श्रमिक", "वर्कर प्रोसेस"),
        (r"\bscreens?\b", "स्क्रीन", "स्क्रीनिंग"),
    ),
    "ko": (
        (r"\bgates?\b", "문", "게이트"),
        (r"\bmasks?\b", "얼굴 마스크", "마스크"),
        (r"\bcells?\b", "배터리", "세포"),
        (r"\bannotation\b", "주석", "어노테이션"),
        (r"\bwells?\b", "우물", "웰"),
        (r"\bhits?\b", "타격", "히트"),
        (r"\bruns?\b|\brunning\b", "달리기", "실행"),
        (r"\bruns?\b|\brunning\b", "달리지", "실행되지"),
        (r"\bcrops?\b", "작물", "크롭"),
        (r"\bworkers?\b", "작업자", "워커 프로세스"),
        (r"\bscreens?\b", "화면", "스크리닝"),
    ),
    "is": (
        (r"\bhits?\b", "högg", "niðurstöður"),
        (r"\bruns?\b|\brunning\b", "hlaupið", "keyrslan"),
        (r"\bruns?\b|\brunning\b", "hlauparinn", "keyrslan"),
        (r"\bruns?\b|\brunning\b", "hlaup", "keyrsla"),
        (r"\bcrops?\b", "ræktun", "myndskurðir"),
        (r"\bworkers?\b", "starfsfólk", "vinnsluþræðir"),
        (r"\bscreens?\b", "skjáir", "skimanir"),
        (r"\bscreens?\b", "skjár", "skimun"),
        (r"\bheadless(?:ly)?\b", "höfuðlaust", "án grafísks viðmóts"),
        (r"\bpipelines?\b", "leiðslur", "vinnsluferli"),
    ),
    "fr": (
        (r"\bgates?\b", "fermeture", "gate"),
        (r"\bgates?\b", "porte", "gate"),
        (r"\bflow field\b", "débit", "flux"),
        (r"\bmasks?\b", "masque facial", "masque"),
        (r"\bclusters?\b", "groupe", "cluster"),
        (r"\bwells?\b", "bien", "puits"),
        (r"\bhits?\b", "touches", "hits"),
        (r"\bhits?\b", "touche", "hit"),
        (r"\bhits?\b", "points marqués", "hits"),
        (r"\bhits?\b", "point marqué", "hit"),
        (r"\bruns?\b|\brunning\b", "courses", "exécutions"),
        (r"\bruns?\b|\brunning\b", "course", "exécution"),
        (r"\bruns?\b|\brunning\b", "parcours", "exécution"),
        (r"\bcrops?\b", "cultures", "vignettes"),
        (r"\bcrops?\b", "culture", "vignette"),
        (r"\bworkers?\b", "travailleurs", "processus workers"),
        (r"\bbatches?\b", "lots", "batchs"),
        (r"\bscreens?\b", "écrans", "criblages"),
        (r"\bscreens?\b", "écran", "criblage"),
        (r"\bheadless(?:ly)?\b", "sans tête", "sans interface graphique"),
    ),
}

MANUAL_UI: dict[str, dict[str, str]] = {
    "Spatial phenotype analysis of CRISPR screens.": {
        "sv": "Spatial fenotypanalys av CRISPR-screeningar.",
        "de": "Räumliche Phänotypanalyse von CRISPR-Screens.",
        "es": "Análisis espacial de fenotipos en cribados CRISPR.",
        "zh_CN": "CRISPR 筛选的空间表型分析。",
        "pt": "Análise espacial de fenótipos em triagens CRISPR.",
        "hi": "CRISPR स्क्रीन का स्थानिक फीनोटाइप विश्लेषण।",
        "ko": "CRISPR 스크린의 공간적 표현형 분석.",
        "is": "Rýmisbundin svipgerðargreining CRISPR-skimana.",
        "fr": "Analyse spatiale des phénotypes des criblages CRISPR.",
    },
    "Regex": {code: "Regex" for code in MODEL_SPECS},
    "Ft": {
        "sv": "Flödeströskel (FT)", "de": "Flussschwellenwert (FT)",
        "es": "Umbral de flujo (FT)", "zh_CN": "流场阈值（FT）",
        "pt": "Limiar de fluxo (FT)", "hi": "फ्लो थ्रेशोल्ड (FT)",
        "ko": "흐름 임계값(FT)", "is": "Flæðisþröskuldur (FT)",
        "fr": "Seuil de flux (FT)",
    },
    "Cp prob": {
        "sv": "Cellsannolikhet (CP)", "de": "Zellwahrscheinlichkeit (CP)",
        "es": "Probabilidad celular (CP)", "zh_CN": "细胞概率（CP）",
        "pt": "Probabilidade celular (CP)", "hi": "कोशिका प्रायिकता (CP)",
        "ko": "세포 확률(CP)", "is": "Frumulíkur (CP)",
        "fr": "Probabilité cellulaire (CP)",
    },
    "Cp probability": {
        "sv": "Cellsannolikhet (CP)", "de": "Zellwahrscheinlichkeit (CP)",
        "es": "Probabilidad celular (CP)", "zh_CN": "细胞概率（CP）",
        "pt": "Probabilidade celular (CP)", "hi": "कोशिका प्रायिकता (CP)",
        "ko": "세포 확률(CP)", "is": "Frumulíkur (CP)",
        "fr": "Probabilité cellulaire (CP)",
    },
    "Verbose": {
        "sv": "Utförlig logg", "de": "Ausführliches Protokoll",
        "es": "Registro detallado", "zh_CN": "详细日志",
        "pt": "Registro detalhado", "hi": "विस्तृत लॉग",
        "ko": "상세 로그", "is": "Ítarleg keyrsluskrá",
        "fr": "Journal détaillé",
    },
    "Dependent variable": {
        "sv": "Beroende variabel", "de": "Abhängige Variable",
        "es": "Variable dependiente", "zh_CN": "因变量",
        "pt": "Variável dependente", "hi": "आश्रित चर",
        "ko": "종속 변수", "is": "Háð breyta",
        "fr": "Variable dépendante",
    },
    "Power reads per well": {
        "sv": "Power: läsningar per brunn", "de": "Power: Reads pro Well",
        "es": "Potencia: lecturas por pocillo", "zh_CN": "功效分析：每孔读数",
        "pt": "Poder: leituras por poço", "hi": "पावर: प्रति वेल रीड्स",
        "ko": "검정력: 웰당 리드 수", "is": "Styrkur: lestrar á brunn",
        "fr": "Puissance : lectures par puits",
    },
    "Use checkpoint": {
        "sv": "Använd gradient-checkpointing",
        "de": "Gradient-Checkpointing verwenden",
        "es": "Usar checkpoint de gradiente", "zh_CN": "使用梯度检查点",
        "pt": "Usar checkpoint de gradiente",
        "hi": "ग्रेडिएंट चेकपॉइंटिंग का उपयोग करें",
        "ko": "그래디언트 체크포인팅 사용",
        "is": "Nota gradient-checkpointing",
        "fr": "Utiliser le gradient checkpointing",
    },
    "Pc loc": {
        "sv": "Kolumn för positiv kontroll",
        "de": "Spalte für positive Kontrolle",
        "es": "Columna de control positivo", "zh_CN": "阳性对照列",
        "pt": "Coluna de controle positivo", "hi": "सकारात्मक नियंत्रण कॉलम",
        "ko": "양성 대조군 열", "is": "Dálkur fyrir jákvætt viðmið",
        "fr": "Colonne du contrôle positif",
    },
    "Nc loc": {
        "sv": "Kolumn för negativ kontroll",
        "de": "Spalte für negative Kontrolle",
        "es": "Columna de control negativo", "zh_CN": "阴性对照列",
        "pt": "Coluna de controle negativo", "hi": "नकारात्मक नियंत्रण कॉलम",
        "ko": "음성 대조군 열", "is": "Dálkur fyrir neikvætt viðmið",
        "fr": "Colonne du contrôle négatif",
    },
    "Rows / fetch": {
        "sv": "Rader per hämtning", "de": "Zeilen pro Abruf",
        "es": "Filas por carga", "zh_CN": "每次获取的行数",
        "pt": "Linhas por busca", "hi": "प्रति फ़ेच पंक्तियाँ",
        "ko": "가져오기당 행 수", "is": "Raðir í hverri sókn",
        "fr": "Lignes par chargement",
    },
    "Which quantile, when quantile is ticked": {
        "sv": "Vilken kvantil som används när Kvantil är markerad",
        "de": "Verwendetes Quantil, wenn Quantil aktiviert ist",
        "es": "Cuantil usado cuando Cuantil está activado",
        "zh_CN": "勾选分位数时使用的分位数",
        "pt": "Quantil usado quando Quantil está marcado",
        "hi": "क्वांटाइल चुने जाने पर उपयोग किया जाने वाला क्वांटाइल",
        "ko": "분위수를 선택했을 때 사용할 분위수",
        "is": "Hvaða fjórðungur er notaður þegar Fjórðungur er valinn",
        "fr": "Quantile utilisé lorsque Quantile est coché",
    },
    "B qc": {code: "B QC" for code in MODEL_SPECS},
    "Seg qc": {
        "sv": "Segmenterings-QC", "de": "Segmentierungs-QC",
        "es": "QC de segmentación", "zh_CN": "分割质控",
        "pt": "QC de segmentação", "hi": "सेगमेंटेशन QC",
        "ko": "분할 QC", "is": "Gæðamat hlutunar",
        "fr": "QC de segmentation",
    },
    "Ig baseline": {
        "sv": "IG-baslinje", "de": "IG-Basislinie",
        "es": "Línea base de IG", "zh_CN": "IG 基线",
        "pt": "Linha de base de IG", "hi": "IG बेसलाइन",
        "ko": "IG 기준선", "is": "IG-grunnlína",
        "fr": "Référence IG",
    },
    "Cells per well": {
        "sv": "Celler per brunn", "de": "Zellen pro Well",
        "es": "Células por pocillo", "zh_CN": "每孔细胞数",
        "pt": "Células por poço", "hi": "प्रति वेल कोशिकाएँ",
        "ko": "웰당 세포 수", "is": "Frumur á brunn",
        "fr": "Cellules par puits",
    },
    "Remove selected": {
        "sv": "Ta bort markerade",
        "de": "Auswahl entfernen",
        "es": "Eliminar la selección",
        "zh_CN": "移除所选项",
        "pt": "Remover selecionados",
        "hi": "चयनित हटाएँ",
        "ko": "선택 항목 제거",
        "is": "Fjarlægja val",
        "fr": "Supprimer la sélection",
    },
    "Queue": {
        "sv": "Kö", "de": "Warteschlange", "es": "Cola",
        "zh_CN": "队列", "pt": "Fila", "hi": "कतार",
        "ko": "대기열", "is": "Biðröð", "fr": "File d’attente",
    },
    "Viewer": {
        "sv": "Visare", "de": "Betrachter", "es": "Visor",
        "zh_CN": "查看器", "pt": "Visualizador", "hi": "व्यूअर",
        "ko": "뷰어", "is": "Skoðari", "fr": "Visionneuse",
    },
    "Flow threshold": {
        "sv": "Flödeströskel", "de": "Flussschwellenwert",
        "es": "Umbral de flujo", "zh_CN": "流场阈值",
        "pt": "Limiar de fluxo", "hi": "फ्लो थ्रेशोल्ड",
        "ko": "흐름 임계값", "is": "Flæðisþröskuldur",
        "fr": "Seuil de flux",
    },
    "Minimum area": {
        "sv": "Minsta area", "de": "Mindestfläche",
        "es": "Área mínima", "zh_CN": "最小面积",
        "pt": "Área mínima", "hi": "न्यूनतम क्षेत्रफल",
        "ko": "최소 면적", "is": "Lágmarksflatarmál",
        "fr": "Surface minimale",
    },
    "Gate Editor": {
        "sv": "Gate-redigerare", "de": "Gate-Editor",
        "es": "Editor de compuertas", "zh_CN": "门控编辑器",
        "pt": "Editor de gates", "hi": "गेट एडिटर",
        "ko": "게이트 편집기", "is": "Gate-ritill",
        "fr": "Éditeur de gates",
    },
    "Save gates": {
        "sv": "Spara gates", "de": "Gates speichern",
        "es": "Guardar compuertas", "zh_CN": "保存门控",
        "pt": "Salvar gates", "hi": "गेट सहेजें",
        "ko": "게이트 저장", "is": "Vista gates",
        "fr": "Enregistrer les gates",
    },
    "Load gates": {
        "sv": "Läs in gates", "de": "Gates laden",
        "es": "Cargar compuertas", "zh_CN": "加载门控",
        "pt": "Carregar gates", "hi": "गेट लोड करें",
        "ko": "게이트 불러오기", "is": "Hlaða gates",
        "fr": "Charger les gates",
    },
    "Gate Editor…": {
        "sv": "Gate-redigerare…", "de": "Gate-Editor…",
        "es": "Editor de compuertas…", "zh_CN": "门控编辑器…",
        "pt": "Editor de gates…", "hi": "गेट एडिटर…",
        "ko": "게이트 편집기…", "is": "Gate-ritill…",
        "fr": "Éditeur de gates…",
    },
    "Gate editor settings": {
        "sv": "Inställningar för Gate-redigeraren",
        "de": "Gate-Editor-Einstellungen",
        "es": "Ajustes del editor de compuertas",
        "zh_CN": "门控编辑器设置",
        "pt": "Configurações do editor de gates",
        "hi": "गेट एडिटर सेटिंग्स", "ko": "게이트 편집기 설정",
        "is": "Stillingar Gate-ritils",
        "fr": "Paramètres de l’éditeur de gates",
    },
    "Box gate": {
        "sv": "Box-gate", "de": "Box-Gate", "es": "Compuerta de caja",
        "zh_CN": "箱式门控", "pt": "Box gate", "hi": "बॉक्स गेट",
        "ko": "박스 게이트", "is": "Box-gate",
        "fr": "Gate 3D rectangulaire",
    },
    "pca": {code: "PCA" for code in MODEL_SPECS},
    "hexbin": {code: "Hexbin" for code in MODEL_SPECS},
    "iou": {code: "IoU" for code in MODEL_SPECS},
    "Nc": {code: "NC" for code in MODEL_SPECS},
    "Pc": {code: "PC" for code in MODEL_SPECS},
    "Volcano": {
        "sv": "Vulkandiagram", "de": "Vulkandiagramm",
        "es": "Gráfico volcán", "zh_CN": "火山图",
        "pt": "Gráfico vulcão", "hi": "वोल्केनो प्लॉट",
        "ko": "볼케이노 플롯", "is": "Eldfjallarit",
        "fr": "Graphique volcan",
    },
    "Coef.": {
        "sv": "Koeff.", "de": "Koeff.", "es": "Coef.",
        "zh_CN": "系数", "pt": "Coef.", "hi": "गुणांक",
        "ko": "계수", "is": "Stuðull", "fr": "Coeff.",
    },
    "Y lims": {
        "sv": "Y-gränser", "de": "Y-Grenzen", "es": "Límites de Y",
        "zh_CN": "Y 轴范围", "pt": "Limites de Y", "hi": "Y सीमाएँ",
        "ko": "Y축 범위", "is": "Y-mörk", "fr": "Limites de Y",
    },
    "Step 1 / 5": {
        "sv": "Steg 1 / 5", "de": "Schritt 1 / 5",
        "es": "Paso 1 / 5", "zh_CN": "第 1 / 5 步",
        "pt": "Etapa 1 / 5", "hi": "चरण 1 / 5",
        "ko": "1 / 5단계", "is": "Skref 1 / 5", "fr": "Étape 1 / 5",
    },
    "Cytoplasm": {
        "sv": "Cytoplasma", "de": "Zytoplasma", "es": "Citoplasma",
        "zh_CN": "细胞质", "pt": "Citoplasma", "hi": "कोशिकाद्रव्य",
        "ko": "세포질", "is": "Umfrymi", "fr": "Cytoplasme",
    },
    "Organelle unet threshold": {
        "sv": "U-Net-tröskel för organeller",
        "de": "U-Net-Schwellenwert für Organellen",
        "es": "Umbral U-Net de orgánulos", "zh_CN": "细胞器 U-Net 阈值",
        "pt": "Limiar U-Net de organelas", "hi": "कोशिकांग U-Net थ्रेशोल्ड",
        "ko": "소기관 U-Net 임계값", "is": "U-Net-þröskuldur frumulíffæra",
        "fr": "Seuil U-Net des organites",
    },
}

# Cellpose exposes the same two abbreviated thresholds for four object types.
# Keep the established CP/FT names intact and localize the object name; asking
# a general translation model to infer these abbreviations produced labels
# such as "chairman" and "organic" in otherwise plausible catalogs.
_OBJECT_LABELS = {
    "Cell": {
        "sv": "Cell", "de": "Zelle", "es": "Célula", "zh_CN": "细胞",
        "pt": "Célula", "hi": "कोशिका", "ko": "세포", "is": "Fruma",
        "fr": "Cellule",
    },
    "Nucleus": {
        "sv": "Cellkärna", "de": "Zellkern", "es": "Núcleo",
        "zh_CN": "细胞核", "pt": "Núcleo", "hi": "नाभिक", "ko": "핵",
        "is": "Kjarni", "fr": "Noyau",
    },
    "Organelle": {
        "sv": "Organell", "de": "Organelle", "es": "Orgánulo",
        "zh_CN": "细胞器", "pt": "Organela", "hi": "कोशिकांग",
        "ko": "소기관", "is": "Frumulíffæri", "fr": "Organite",
    },
    "Pathogen": {
        "sv": "Patogen", "de": "Pathogen", "es": "Patógeno",
        "zh_CN": "病原体", "pt": "Patógeno", "hi": "रोगजनक",
        "ko": "병원체", "is": "Sýkill", "fr": "Pathogène",
    },
}
for _object_source, _localized_names in _OBJECT_LABELS.items():
    MANUAL_UI[f"{_object_source} cp prob"] = {
        language: f"{name} — CP" for language, name in _localized_names.items()
    }
    MANUAL_UI[f"{_object_source} ft"] = {
        language: f"{name} — FT" for language, name in _localized_names.items()
    }


def _call_name(node: ast.Call) -> str:
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    if isinstance(node.func, ast.Name):
        return node.func.id
    return ""


def _literal(node: ast.AST) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(
    node.value, str) else None


def _literal_strings(
    node: ast.AST,
    constants: Mapping[str, ast.AST],
) -> Iterable[str]:
    """Yield static string members from a literal or module constant."""
    value = _literal(node)
    if value is not None:
        yield value
        return
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for item in node.elts:
            yield from _literal_strings(item, constants)
        return
    if isinstance(node, ast.Name) and node.id in constants:
        yield from _literal_strings(constants[node.id], constants)
        return
    if (isinstance(node, ast.Call) and _call_name(node) in {"list", "tuple"}
            and len(node.args) == 1):
        yield from _literal_strings(node.args[0], constants)


def _candidate_arguments(node: ast.Call, name: str) -> Iterable[ast.AST]:
    if name == "addTab" and len(node.args) >= 2:
        yield node.args[1]
        return
    if name == "addItem":
        # QComboBox.addItem(text, data) or addItem(icon, text, data).
        for arg in node.args[:2]:
            if _literal(arg) is not None:
                yield arg
                return
        return
    if name == "QAction":
        # QAction(text, parent) or QAction(icon, text, parent).
        for arg in node.args[:2]:
            if _literal(arg) is not None:
                yield arg
                return
        return
    if name in _DIALOG_METHODS:
        # QMessageBox.<kind>(parent, title, message, ...).
        yield from node.args[1:3]
        return
    if name in _FILE_DIALOG_METHODS:
        # parent, caption, directory, filter. Paths and filter syntax are not
        # prose; only the window caption is safe to localize automatically.
        if len(node.args) >= 2:
            yield node.args[1]
        return
    if name in _INPUT_DIALOG_METHODS:
        # parent, title, label, value/options...
        yield from node.args[1:3]
        return
    if name == "QProgressDialog":
        # label text and cancel-button text precede the numeric range.
        yield from node.args[:2]
        return
    if node.args:
        yield node.args[0]


def _looks_translatable(text: str) -> bool:
    source = text.strip()
    if not source or source in _IDENTITY_TEXT:
        return False
    if "\n" in source and len(source) > 1200:
        return False
    if source.startswith(("/", "\\", "#", "rgb(", "rgba(")):
        return False
    if "://" in source or re.search(r"[\\/]\w+[\\/]", source):
        return False
    if re.fullmatch(r"[\W\d_]+", source):
        return False
    if re.fullmatch(r"[A-Z0-9_.+-]{1,8}", source):
        return False
    # Stylesheets, regexes and serialized records are not presentation prose.
    if any(marker in source for marker in (
        "QWidget {", "font-size:", "background-color:", "(?P<", "SELECT ",
    )):
        return False
    return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", source))


def extract_static_ui_sources() -> tuple[str, ...]:
    """Return literal spaCR-owned Qt presentation strings from the AST."""
    found: set[str] = set()
    for path in sorted((ROOT / "spacr" / "qt").rglob("*.py")):
        if "i18n_catalogs" in path.parts:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        constants: dict[str, ast.AST] = {}
        for statement in tree.body:
            if (isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)):
                constants[statement.targets[0].id] = statement.value
            elif (isinstance(statement, ast.AnnAssign)
                  and isinstance(statement.target, ast.Name)
                  and statement.value is not None):
                constants[statement.target.id] = statement.value
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if not (
                name in _TEXT_METHODS
                or name in _TEXT_CONSTRUCTORS
                or name in _DIALOG_METHODS
                or name in _FILE_DIALOG_METHODS
                or name in _INPUT_DIALOG_METHODS
                or name == "QProgressDialog"
                or name in {"addTab", "addItem", "addItems",
                            "setHorizontalHeaderLabels", "setHeaderLabels"}
            ):
                continue
            if name in {"addItems", "setHorizontalHeaderLabels",
                        "setHeaderLabels"} and node.args:
                for value in _literal_strings(node.args[0], constants):
                    if _looks_translatable(value):
                        found.add(value.strip())
                continue
            for argument in _candidate_arguments(node, name):
                value = _literal(argument)
                if value is not None and _looks_translatable(value):
                    found.add(value.strip())

    # The compact catalog already owns these and has stronger human review.
    from spacr.qt.i18n import _ROWS
    return tuple(sorted(found - set(_ROWS)))


def canonical_sources() -> dict[str, object]:
    """Read every canonical English source from the application."""
    from spacr.qt.screens.settings_model import (
        CATEGORY_TOOLTIPS,
        CATEGORY_TOOLTIPS_BY_APP,
        _humanize,
        _strip_type_prefix,
        get_tooltips,
        resolve_default_settings,
        SettingsWidgets,
    )

    raw_tooltips = get_tooltips()
    tooltips = {
        str(key): " ".join(_strip_type_prefix(text).split())
        for key, text in raw_tooltips.items()
        if str(text).strip()
    }
    labels = {key: _humanize(key) for key in tooltips}
    categories = set(CATEGORY_TOOLTIPS.values())
    categories.update(
        text for entries in CATEGORY_TOOLTIPS_BY_APP.values()
        for text in entries.values()
    )
    installer = json.loads(
        (ROOT / "packaging" / "i18n" / "en.json").read_text(
            encoding="utf-8"
        )
    )
    from spacr.qt.app import APPS, _SECTION_NOTE_LIBRARY
    from spacr.qt.screens.app_screen import (
        APP_INTROS,
        APP_TITLES,
        DEFAULT_INSTRUCTION,
    )
    module_summaries = {
        str(key): str(description)
        for key, _name, description, _section in APPS
    }
    label_model = SettingsWidgets.__new__(SettingsWidgets)
    for app_key, _name, _description, _section in APPS:
        label_model.app_key = app_key
        try:
            setting_keys = resolve_default_settings(app_key)
        except Exception:
            continue
        for key in setting_keys:
            actual = label_model._label_for(str(key))
            generic = _humanize(str(key))
            # Labels are visible UI even when a setting has no authored help
            # paragraph.  Inventory every setting, not merely tooltip keys.
            labels.setdefault(str(key), generic)
            if actual != generic:
                labels[f"{app_key}.{key}"] = actual
    ui_sources = set(extract_static_ui_sources())
    ui_sources.update(str(value) for value in APP_INTROS.values())
    ui_sources.update(str(value) for value in APP_TITLES.values())
    ui_sources.update(str(value) for value in _SECTION_NOTE_LIBRARY.values())
    # Reviewed domain terms are part of the supported translation contract
    # even when their current UI occurrence is assembled dynamically rather
    # than visible to the literal-string AST extractor.
    ui_sources.update(MANUAL_UI)
    ui_sources.add(DEFAULT_INSTRUCTION)
    return {
        "setting_labels": dict(sorted(labels.items())),
        "setting_tooltips": dict(sorted(tooltips.items())),
        "categories": tuple(sorted(categories)),
        "ui": tuple(sorted(ui_sources)),
        "installer": dict(sorted(installer.items())),
        "module_summaries": dict(sorted(module_summaries.items())),
    }


def _render_assignment(name: str, value: object) -> str:
    if isinstance(value, frozenset):
        rendered = f"frozenset({tuple(sorted(value))!r})"
    elif isinstance(value, dict):
        # One entry per line is deterministic and reviewable without pprint's
        # repeated continuation indentation multiplying large tooltip files.
        rows = ["{"]
        rows.extend(f"    {key!r}: {item!r}," for key, item in value.items())
        rows.append("}")
        rendered = "\n".join(rows)
    else:
        rendered = pprint.pformat(value, width=100, sort_dicts=True)
    return f"{name} = {rendered}\n"


def write_english(sources: Mapping[str, object]) -> Path:
    path = CATALOG_DIR / "en.py"
    text = (
        '"""Canonical English sources for generated localization catalogs.\n\n'
        "Generated by tools/build_i18n_catalogs.py; do not hand-edit.\n"
        '"""\n\n'
        + _render_assignment("SETTING_LABELS", sources["setting_labels"])
        + "\n"
        + _render_assignment("SETTING_TOOLTIPS", sources["setting_tooltips"])
        + "\n"
        + _render_assignment("CATEGORY_SOURCES", frozenset(sources["categories"]))
        + "\n"
        + _render_assignment("UI_SOURCES", frozenset(sources["ui"]))
        + "\n"
        + _render_assignment("MODULE_SUMMARIES", sources["module_summaries"])
    )
    path.write_text(text, encoding="utf-8")
    return path


def _protect(
    text: str,
    marker_style: str = "xml",
) -> tuple[str, dict[str, str]]:
    values: list[str] = []

    def token(value: str) -> str:
        # Marian preserves short XML-like x-tags far more reliably than long
        # invented words (notably in en→zh, which can drop letters from
        # ZXQ0000QXZ).  The restore pass also accepts stripped angle brackets.
        index = len(values)
        marker = f"<x{index}>" if marker_style == "xml" else f"{index}X{index}"
        values.append(value)
        return marker

    # Apply one left-to-right substitution. Sequential substitutions can
    # accidentally protect a marker created by an earlier pattern (for
    # example a dictionary containing ``<feature>``), producing nested tokens
    # that cannot be restored safely.
    protected = _PROTECT_RE.sub(lambda match: token(match.group(0)), str(text))
    markers = (
        (f"<x{i}>" if marker_style == "xml" else f"{i}X{i}")
        for i in range(len(values))
    )
    return protected, dict(zip(markers, values))


def _restore(text: str, protected: Mapping[str, str]) -> str:
    restored = str(text)
    for marker, value in protected.items():
        digits = re.search(r"\d+", marker).group(0)
        if marker.startswith("<"):
            fuzzy = (
                rf"(?:<\s*[xX]\s*{digits}\s*>|"
                rf"\b[xX]\s*{digits}\b)"
            )
        else:
            fuzzy = rf"\b{digits}\s*[xX]\s*{digits}\b"
        restored, count = re.subn(fuzzy, lambda _match, v=value: v, restored)
        if count != 1:
            raise ValueError(
                f"translation did not preserve {marker} exactly once: {text!r}"
            )
    if _TOKEN_RE.search(restored) or re.search(r"Z\s*X\s*Q\s*\d", restored):
        raise ValueError(f"unrestored protection token: {restored!r}")
    return restored.strip()


def _translation_chunks(text: str) -> list[str]:
    """Split long prose without separating its API-bearing punctuation.

    OPUS models are markedly more reliable at retaining several independent
    protection markers in a sentence than dozens of markers in a whole help
    paragraph.  Whitespace is normalized when chunks are joined; runtime help
    text already has that same normalization.
    """
    chunks = [
        part.strip()
        for part in re.findall(r".+?(?:[.!?;](?:\s+|$)|$)", text, re.DOTALL)
        if part.strip()
    ]
    if not chunks:
        return [text]
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) <= 420:
            result.append(chunk)
            continue
        comma_chunks = [
            part.strip()
            for part in re.findall(r".+?(?:,(?:\s+|$)|$)", chunk, re.DOTALL)
            if part.strip()
        ]
        result.extend(comma_chunks or [chunk])
    return result


def _contextualize(value: str, language: str, source: str = "") -> str:
    corrected = str(value)
    # Some models echo a closing parenthesis after a protected URL that
    # already carried its source ``).`` punctuation.
    corrected = re.sub(
        r"(https?://[^\s)]+)\)\.\)", r"\1).", corrected
    )
    # A few Marian tokenizers emit ``<x0>>`` for a protected marker.  The
    # restore correctly consumes ``<x0>`` but the second angle bracket would
    # otherwise leak into README prose after product names and RST substitutions.
    corrected = re.sub(r"(\|[A-Za-z][^|\n]*\|)>", r"\1", corrected)
    for term in _PROTECTED_TERMS:
        if f"{term}>" not in str(source):
            corrected = corrected.replace(f"{term}>", term)
    for wrong, right in CONTEXT_REPLACEMENTS.get(language, ()):
        corrected = corrected.replace(wrong, right)
    for source_pattern, wrong, right in SOURCE_CONTEXT_REPLACEMENTS.get(
        language, ()
    ):
        if re.search(source_pattern, str(source), flags=re.IGNORECASE):
            if right.startswith(wrong) and len(right) > len(wrong):
                corrected = re.sub(
                    re.escape(wrong)
                    + rf"(?!{re.escape(right[len(wrong):])})",
                    right,
                    corrected,
                )
            elif wrong[:1].isalnum() and wrong[-1:].isalnum():
                corrected = re.sub(
                    rf"\b{re.escape(wrong)}\b",
                    right,
                    corrected,
                    flags=re.IGNORECASE,
                )
            else:
                corrected = corrected.replace(wrong, right)
    return corrected


def _syntax_preserved(source: str, value: str) -> bool:
    if not str(value).strip():
        return False

    # Python-format fields are runtime API, not prose.  A translated string
    # with an unmatched brace is especially dangerous because it looks fine
    # in the catalog and then raises only when the tooltip is displayed.
    from string import Formatter

    def format_fields(text: str) -> set[str] | None:
        try:
            return {
                name for _literal, name, _spec, _conversion
                in Formatter().parse(str(text)) if name is not None
            }
        except ValueError:
            return None

    if format_fields(source) != format_fields(value):
        return False

    patterns = (
        r"</?[A-Za-z][^>]*>",
        r"\{[^{}]+\}",
        r"%(?:\d+\$)?[sd]",
        r"\|[A-Za-z][^|\n]*\|",
        r"\*\*",
        r":(?:class|func|mod|meth|attr|data|doc):`[^`]+`",
        r"``[^`]+``|`[^`]+`_?",
    )
    structural = all(
        re.findall(pattern, str(source)) == re.findall(pattern, str(value))
        for pattern in patterns
    )
    protected_terms = all(
        source.count(term) == value.count(term)
        for term in _PROTECTED_TERMS
    )
    return structural and protected_terms


def _looks_degenerate(source: str, value: str, language: str) -> bool:
    """Detect obvious model loops without rejecting normal repeated prose."""
    rendered = str(value).strip()
    if not rendered:
        return True
    if any(marker in rendered.casefold() for marker in (
        "city name (optional",
        "probably does not need a translation",
        "unit description in lists",
    )):
        return True
    # A short label expanding into hundreds of characters is a generation
    # loop, not a linguistically plausible translation.
    if len(source) < 100 and len(rendered) > max(48, len(source) * 6):
        return True
    if len(source) >= 100 and len(rendered) > len(source) * 3:
        return True
    latin_loop = re.search(
        r"\b([A-Za-zÀ-ÖØ-öø-ÿ]{3,})\b"
        r"(?:[\s,;:/—-]+\1\b){3,}",
        rendered,
        flags=re.IGNORECASE,
    )
    cjk_loop = re.search(r"([\u3400-\u9fff]{1,6})\1{3,}", rendered)
    source_latin = re.search(
        r"\b([A-Za-zÀ-ÖØ-öø-ÿ]{3,})\b"
        r"(?:[\s,;:/—-]+\1\b){3,}",
        str(source),
        flags=re.IGNORECASE,
    )
    source_cjk = re.search(r"([\u3400-\u9fff]{1,6})\1{3,}", str(source))
    return bool((latin_loop and not source_latin) or (cjk_loop and not source_cjk))


def _seed_cache_from_catalog(language: str, cache: dict[str, str]) -> None:
    """Reuse a previously generated module when adding new source surfaces."""
    try:
        from spacr.qt.i18n_catalogs import en as english
        target = __import__(
            f"spacr.qt.i18n_catalogs.{language}", fromlist=["*"]
        )
    except (ImportError, ModuleNotFoundError):
        return
    if getattr(target, "MODEL", None) != MODEL_SPECS[language][0]:
        # Never seed a replacement model from output produced by a rejected
        # checkpoint; fluent-looking stale text is worse than retranslating.
        return

    for name, canonical_name in (
        ("SETTING_LABELS", "SETTING_LABELS"),
        ("SETTING_TOOLTIPS", "SETTING_TOOLTIPS"),
    ):
        canonical = getattr(english, canonical_name, {})
        translated = getattr(target, name, {})
        for key, source in canonical.items():
            value = translated.get(key)
            if (
                isinstance(value, str)
                and value.strip()
                and _syntax_preserved(source, value)
                and not _looks_degenerate(source, value, language)
                and (value != source or not _looks_translatable(source))
            ):
                cache.setdefault(str(source), value)
    for name in ("CATEGORY_HELP", "UI"):
        for source, value in getattr(target, name, {}).items():
            if (
                isinstance(value, str)
                and value.strip()
                and _syntax_preserved(source, value)
                and not _looks_degenerate(source, value, language)
                and (value != source or not _looks_translatable(source))
            ):
                cache.setdefault(str(source), value)
    canonical_modules = getattr(english, "MODULE_SUMMARIES", {})
    translated_modules = getattr(target, "MODULE_SUMMARIES", {})
    for key, source in canonical_modules.items():
        value = translated_modules.get(key)
        if (
            isinstance(value, str)
            and value.strip()
            and _syntax_preserved(source, value)
            and not _looks_degenerate(source, value, language)
            and (value != source or not _looks_translatable(source))
        ):
            cache.setdefault(str(source), value)


def _translate_batches(
    strings: list[str],
    language: str,
    model_root: Path,
    *,
    device: str,
    batch_size: int,
    beams: int,
    threads: int,
) -> dict[str, str]:
    """Translate unique strings with one local OPUS model."""
    model_id, folder, _license, prefix = MODEL_SPECS[language]
    is_m2m = language in {"zh_CN", "hi", "ko", "is"}
    model_path = model_root / folder
    if not model_path.exists():
        raise FileNotFoundError(
            f"missing {model_path}; download {model_id} before generation"
        )
    cache_dir = model_root / ".spacr_translation_cache"
    cache_path = cache_dir / f"{language}.json"
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    _seed_cache_from_catalog(language, cache)

    translated: dict[str, str] = {}
    generated_sources: list[str] = []
    generated_inputs: list[str] = []
    protection: list[dict[str, str]] = []
    retry_sources: list[str] = []
    cached_repair_sources: list[str] = []

    from spacr.qt.i18n import CATALOGS
    compact = CATALOGS[language]
    for source in strings:
        if source in MANUAL_UI and language in MANUAL_UI[source]:
            translated[source] = MANUAL_UI[source][language]
        elif source in compact:
            translated[source] = compact[source]
        elif source in _IDENTITY_TEXT:
            translated[source] = source
        elif (
            source in cache
            and str(cache[source]).strip()
            and _syntax_preserved(source, str(cache[source]))
            and not _looks_degenerate(source, str(cache[source]), language)
            and (str(cache[source]) != source or not _looks_translatable(source))
        ):
            translated[source] = _contextualize(
                str(cache[source]), language, source
            )
        elif source in cache:
            # A model occasionally damages an RST/code marker while translating
            # otherwise useful prose.  Preserve that checkpoint as repair
            # input.  Release generation falls back to canonical English for
            # these entries; re-decoding the paragraph with a second invented
            # marker is both slower and less reliable.
            translated[source] = _contextualize(
                str(cache[source]).strip() or source, language, source
            )
            cached_repair_sources.append(source)
        else:
            protected, mapping = _protect(source)
            generated_sources.append(source)
            generated_inputs.append(prefix + protected)
            protection.append(mapping)

    if generated_inputs:
        packed = sorted(
            zip(generated_sources, generated_inputs, protection),
            key=lambda item: (len(item[1]), item[0]),
        )
        generated_sources = [item[0] for item in packed]
        generated_inputs = [item[1] for item in packed]
        protection = [item[2] for item in packed]

    if not generated_inputs:
        for source in cached_repair_sources:
            translated[source] = source
        return translated

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(max(1, threads))
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # Both OPUS and M2M can otherwise continue a high-probability word or CJK
    # character until ``max_new_tokens`` on terse technical labels.  These
    # decoding constraints do not alter terminology; they only make a repeated
    # 3-gram impossible and gently discourage immediate token loops.
    generation_kwargs: dict[str, object] = {
        "early_stopping": False,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.12,
    }

    def output_budget(encoded: Mapping[str, object]) -> int:
        """Allow generous translation expansion without 480-token label loops."""
        input_width = int(encoded["input_ids"].shape[1])
        ceiling = 224 if language == "zh_CN" else (256 if is_m2m else 320)
        return min(ceiling, max(48, input_width * 2 + 32))
    m2m_target = {
        "zh_CN": "zh", "hi": "hi", "ko": "ko", "is": "is",
    }.get(language)
    if m2m_target:
        tokenizer.src_lang = "en"
        generation_kwargs["forced_bos_token_id"] = tokenizer.get_lang_id(
            m2m_target
        )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, local_files_only=True,
    )
    if device == "cuda":
        model = model.half().to("cuda")
    model.eval()

    for start in range(0, len(generated_inputs), batch_size):
        batch = generated_inputs[start:start + batch_size]
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=480,
        )
        if device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=output_budget(encoded),
                num_beams=beams,
                **generation_kwargs,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        for offset, value in enumerate(decoded):
            index = start + offset
            source = generated_sources[index]
            try:
                value = _restore(value, protection[index])
            except ValueError:
                value = source
                retry_sources.append(source)
            value = _contextualize(value, language, source)
            translated[source] = value.strip() or source
            cache[source] = translated[source]
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(cache, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        print(
            f"{language}: {min(start + len(batch), len(generated_inputs))}/"
            f"{len(generated_inputs)}",
            flush=True,
        )

    # A structurally damaged translation falls back to canonical English.
    # Secondary marker/fragment translations lose too much sentence context
    # for scientific API prose; the code below remains available for targeted
    # experiments but is deliberately disabled for release generation.
    allow_secondary_repairs = False

    # Marian occasionally strips angle brackets or a letter from a protected
    # token, especially in Chinese. Retry only those strings with a second,
    # independently tested numeric-X marker before accepting English fallback.
    # Keeping this inside the loaded-model lifetime makes the retry cheap.
    if retry_sources and not is_m2m and allow_secondary_repairs:
        retry_sources = list(dict.fromkeys(retry_sources))
        retry_inputs: list[str] = []
        retry_maps: list[dict[str, str]] = []
        for source in retry_sources:
            protected, mapping = _protect(source, marker_style="numeric")
            retry_inputs.append(prefix + protected)
            retry_maps.append(mapping)
        for start in range(0, len(retry_inputs), batch_size):
            batch = retry_inputs[start:start + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=480,
            )
            if device == "cuda":
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=output_budget(encoded), num_beams=beams,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            for offset, value in enumerate(decoded):
                index = start + offset
                source = retry_sources[index]
                try:
                    value = _restore(value, retry_maps[index])
                except ValueError:
                    value = source
                value = _contextualize(value, language, source)
                translated[source] = value.strip() or source
                cache[source] = translated[source]
            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(cache, ensure_ascii=False, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(cache_path)
        print(
            f"{language}: retried protected strings={len(retry_sources)}",
            flush=True,
        )

    # A final sentence-sized pass handles dense help paragraphs where even the
    # numeric retry asks one model sequence to carry too many protected values.
    # Only accept the recomposed translation when every chunk restores and the
    # complete paragraph retains its structural/API tokens.
    chunk_sources = [] if is_m2m or not allow_secondary_repairs else [
        source for source in generated_sources
        if translated.get(source, source) == source
        or not _syntax_preserved(source, translated.get(source, source))
        or _looks_degenerate(source, translated.get(source, source), language)
    ]
    if chunk_sources:
        chunk_inputs: list[str] = []
        chunk_maps: list[dict[str, str]] = []
        chunk_owners: list[tuple[str, int]] = []
        chunks_by_source: dict[str, list[str]] = {
            source: _translation_chunks(source) for source in chunk_sources
        }
        for source, chunks in chunks_by_source.items():
            for index, chunk in enumerate(chunks):
                protected, mapping = _protect(chunk, marker_style="numeric")
                chunk_inputs.append(prefix + protected)
                chunk_maps.append(mapping)
                chunk_owners.append((source, index))
        restored_chunks: dict[str, list[str | None]] = {
            source: [None] * len(chunks)
            for source, chunks in chunks_by_source.items()
        }
        for start in range(0, len(chunk_inputs), batch_size):
            batch = chunk_inputs[start:start + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=480,
            )
            if device == "cuda":
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=output_budget(encoded), num_beams=beams,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            for offset, value in enumerate(decoded):
                owner, chunk_index = chunk_owners[start + offset]
                try:
                    value = _restore(value, chunk_maps[start + offset])
                except ValueError:
                    value = None
                restored_chunks[owner][chunk_index] = value
        accepted = 0
        for source, values in restored_chunks.items():
            if any(value is None for value in values):
                continue
            candidate = " ".join(str(value).strip() for value in values)
            candidate = _contextualize(candidate, language, source)
            if (candidate != source and _syntax_preserved(source, candidate)
                    and not _looks_degenerate(source, candidate, language)):
                translated[source] = candidate
                cache[source] = candidate
                accepted += 1
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(cache, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        print(
            f"{language}: sentence retry accepted={accepted}/"
            f"{len(chunk_sources)}",
            flush=True,
        )

    # Some target tokenizers (especially Devanagari and Hangul models) can
    # still mutate an otherwise simple marker.  As a deterministic last
    # resort, translate only the prose spans *between* protected API values
    # and then splice the untouched values back in.  No synthetic marker ever
    # reaches the model in this pass, so HTML/identifiers cannot leak or drift.
    fragment_sources = [] if not allow_secondary_repairs else [
        source for source in dict.fromkeys(
            [*generated_sources, *cached_repair_sources]
        )
        if not is_m2m
        if translated.get(source, source) == source
        or not _syntax_preserved(source, translated.get(source, source))
        or _looks_degenerate(source, translated.get(source, source), language)
    ]
    if fragment_sources:
        pieces_by_source: dict[str, list[str]] = {}
        fragment_inputs: list[str] = []
        fragment_owners: list[tuple[str, int]] = []
        for source in fragment_sources:
            protected, mapping = _protect(source)
            pieces = re.split(r"(<x\d+>)", protected)
            for index, piece in enumerate(pieces):
                if piece in mapping:
                    pieces[index] = mapping[piece]
                elif _looks_translatable(piece):
                    fragment_inputs.append(prefix + piece.strip())
                    fragment_owners.append((source, index))
            pieces_by_source[source] = pieces
        for start in range(0, len(fragment_inputs), batch_size):
            batch = fragment_inputs[start:start + batch_size]
            encoded = tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=480,
            )
            if device == "cuda":
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=output_budget(encoded), num_beams=beams,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            for offset, value in enumerate(decoded):
                owner, piece_index = fragment_owners[start + offset]
                original = pieces_by_source[owner][piece_index]
                leading = " " if original[:1].isspace() else ""
                trailing = " " if original[-1:].isspace() else ""
                pieces_by_source[owner][piece_index] = (
                    leading + value.strip() + trailing
                )
        accepted = 0
        for source, pieces in pieces_by_source.items():
            candidate = _contextualize("".join(pieces).strip(), language, source)
            if (candidate != source and _syntax_preserved(source, candidate)
                    and not _looks_degenerate(source, candidate, language)):
                translated[source] = candidate
                cache[source] = candidate
                accepted += 1
        cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(cache, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        print(
            f"{language}: fragment retry accepted={accepted}/"
            f"{len(fragment_sources)}",
            flush=True,
        )

    for source, value in tuple(translated.items()):
        if (not _syntax_preserved(source, value)
                or _looks_degenerate(source, value, language)):
            translated[source] = source
            cache[source] = source

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return translated


def _unique_translation_sources(sources: Mapping[str, object]) -> list[str]:
    values: set[str] = set(sources["setting_labels"].values())
    values.update(sources["setting_tooltips"].values())
    values.update(sources["categories"])
    values.update(sources["ui"])
    values.update(sources["module_summaries"].values())
    return sorted(values)


def write_language(
    language: str,
    sources: Mapping[str, object],
    translations: Mapping[str, str],
) -> Path:
    model_id, _folder, license_name, _prefix = MODEL_SPECS[language]
    setting_labels = {
        key: translations[value]
        for key, value in sources["setting_labels"].items()
    }
    setting_tooltips = {
        key: translations[value]
        for key, value in sources["setting_tooltips"].items()
    }
    categories = {value: translations[value] for value in sources["categories"]}
    ui = {value: translations[value] for value in sources["ui"]}
    from spacr.qt.i18n_module_summaries import (
        MODULE_SUMMARIES as reviewed_module_summaries,
    )
    reviewed = reviewed_module_summaries.get(language, {})
    module_summaries = {
        key: reviewed.get(key) or translations[value]
        for key, value in sources["module_summaries"].items()
    }
    path = CATALOG_DIR / f"{language}.py"
    text = (
        f'"""spaCR localization catalog for {language}.\n\n'
        f"Drafted with {model_id} ({license_name}) and corrected by spaCR's "
        "technical-context review. Generated by tools/build_i18n_catalogs.py.\n"
        '"""\n\n'
        + f'MODEL = {model_id!r}\nLICENSE = {license_name!r}\n\n'
        + _render_assignment("SETTING_LABELS", setting_labels)
        + "\n"
        + _render_assignment("SETTING_TOOLTIPS", setting_tooltips)
        + "\n"
        + _render_assignment("CATEGORY_HELP", categories)
        + "\n"
        + _render_assignment("UI", ui)
        + "\n"
        + _render_assignment("MODULE_SUMMARIES", module_summaries)
    )
    path.write_text(text, encoding="utf-8")
    return path


def audit(sources: Mapping[str, object], languages: Iterable[str]) -> int:
    """Validate key coverage, source freshness and basic translation safety."""
    from string import Formatter
    from types import SimpleNamespace

    failures: list[str] = []

    def fields(text: str) -> set[str]:
        try:
            return {
                name for _literal, name, _spec, _conversion
                in Formatter().parse(text) if name is not None
            }
        except ValueError:
            return set()

    def html_tags(text: str) -> list[str]:
        return re.findall(r"</?[A-Za-z][^>]*>", str(text))

    expected_labels = set(sources["setting_labels"])
    expected_tips = set(sources["setting_tooltips"])
    expected_categories = set(sources["categories"])
    expected_ui = set(sources["ui"])
    expected_modules = set(sources["module_summaries"])
    script_pattern = {
        "zh_CN": re.compile(r"[\u3400-\u9fff]"),
        "hi": re.compile(r"[\u0900-\u097f]"),
        "ko": re.compile(r"[\uac00-\ud7af]"),
    }
    for language in languages:
        catalog_path = CATALOG_DIR / f"{language}.py"
        try:
            namespace: dict[str, object] = {}
            exec(
                compile(
                    catalog_path.read_text(encoding="utf-8"),
                    str(catalog_path),
                    "exec",
                ),
                namespace,
            )
            module = SimpleNamespace(**namespace)
        except FileNotFoundError:
            failures.append(f"{language}: catalog module is missing")
            continue
        tables = {
            "SETTING_LABELS": expected_labels,
            "SETTING_TOOLTIPS": expected_tips,
            "CATEGORY_HELP": expected_categories,
            "UI": expected_ui,
            "MODULE_SUMMARIES": expected_modules,
        }
        source_tables = {
            "SETTING_LABELS": sources["setting_labels"],
            "SETTING_TOOLTIPS": sources["setting_tooltips"],
            "CATEGORY_HELP": {},
            "UI": {},
            "MODULE_SUMMARIES": sources["module_summaries"],
        }
        for name, expected in tables.items():
            table = getattr(module, name, {})
            missing = expected - set(table)
            extra = set(table) - expected
            blank = [key for key, value in table.items() if not str(value).strip()]
            if missing:
                failures.append(f"{language}/{name}: {len(missing)} missing")
            if extra:
                failures.append(f"{language}/{name}: {len(extra)} stale")
            if blank:
                failures.append(f"{language}/{name}: {len(blank)} blank")
            degenerate = [
                key for key, value in table.items()
                if _looks_degenerate(
                    str(source_tables[name].get(key, key)),
                    str(value), language,
                )
            ]
            if degenerate:
                failures.append(
                    f"{language}/{name}: {len(degenerate)} degenerate "
                    f"translations ({', '.join(map(str, degenerate[:5]))})"
                )
        for key, source in sources["setting_tooltips"].items():
            value = module.SETTING_TOOLTIPS.get(key, "")
            if fields(value) != fields(source):
                failures.append(f"{language}/tooltip/{key}: format fields changed")
            if html_tags(value) != html_tags(source):
                failures.append(f"{language}/tooltip/{key}: HTML tags changed")
            if _TOKEN_RE.search(value):
                failures.append(f"{language}/tooltip/{key}: leaked token")
        for key, source in sources["setting_labels"].items():
            reviewed = MANUAL_UI.get(str(source), {}).get(language)
            if reviewed is not None and module.SETTING_LABELS.get(key) != reviewed:
                failures.append(
                    f"{language}/label/{key}: reviewed translation changed"
                )
        for source in sources["ui"]:
            reviewed = MANUAL_UI.get(str(source), {}).get(language)
            if reviewed is not None and module.UI.get(source) != reviewed:
                failures.append(
                    f"{language}/ui/{source!r}: reviewed translation changed"
                )
        unchanged_tips = sum(
            module.SETTING_TOOLTIPS.get(key) == source
            for key, source in sources["setting_tooltips"].items()
        )
        unchanged_ui = sum(
            module.UI.get(source) == source for source in sources["ui"]
        )
        if unchanged_tips > max(10, len(expected_tips) // 20):
            failures.append(
                f"{language}: {unchanged_tips} tooltip bodies remain English"
            )
        if unchanged_ui > max(25, len(expected_ui) // 6):
            failures.append(
                f"{language}: {unchanged_ui} static UI strings remain English"
            )
        if language in script_pattern:
            missing_script = sum(
                len(source) >= 40
                and bool(re.search(r"[A-Za-z]{4}", source))
                and not script_pattern[language].search(
                    str(module.SETTING_TOOLTIPS.get(key, ""))
                )
                for key, source in sources["setting_tooltips"].items()
            )
            if missing_script > max(10, len(expected_tips) // 20):
                failures.append(
                    f"{language}: {missing_script} prose tooltips lack target script"
                )
        for source in sources["ui"]:
            value = module.UI.get(source, "")
            if fields(value) != fields(source):
                failures.append(f"{language}/ui/{source!r}: format fields changed")
            if html_tags(value) != html_tags(source):
                failures.append(f"{language}/ui/{source!r}: HTML tags changed")
        installer_path = ROOT / "packaging" / "i18n" / f"{language}.json"
        try:
            installer = json.loads(installer_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            failures.append(f"{language}/installer: catalog is missing or invalid")
        else:
            if set(installer) != set(sources["installer"]):
                failures.append(f"{language}/installer: keys differ from English")
            if installer.get("language_name") != NATIVE_LANGUAGE_NAMES[language]:
                failures.append(
                    f"{language}/installer: native language name is wrong"
                )
            for key, source in sources["installer"].items():
                value = str(installer.get(key, ""))
                if re.findall(r"%(?:\d+\$)?[sd]", value) != re.findall(
                    r"%(?:\d+\$)?[sd]", source
                ):
                    failures.append(
                        f"{language}/installer/{key}: placeholders changed"
                    )
    if failures:
        print("\n".join(failures[:200]), file=sys.stderr)
        if len(failures) > 200:
            print(f"... and {len(failures) - 200} more", file=sys.stderr)
        return 1
    print(
        "verified external runtime catalogs: "
        f"languages={len(tuple(languages))} "
        f"settings={len(expected_tips)} categories={len(expected_categories)} "
        f"ui={len(expected_ui)} modules={len(expected_modules)}"
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--languages", nargs="+", choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument(
        "--model-root", type=Path,
        default=Path(
            "/mnt/firecuda2/Claude/toxoplasma_projects/tutorials/project/"
            "translation_models/opus"
        ),
    )
    parser.add_argument("--sources-only", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    sources = canonical_sources()
    path = write_english(sources)
    print(
        f"wrote {path}: settings={len(sources['setting_tooltips'])} "
        f"categories={len(sources['categories'])} ui={len(sources['ui'])}"
    )
    if args.sources_only:
        return 0
    if args.audit:
        return audit(sources, args.languages)

    values = _unique_translation_sources(sources)
    for language in args.languages:
        translations = _translate_batches(
            values,
            language,
            args.model_root,
            device=args.device,
            batch_size=args.batch_size,
            beams=args.beams,
            threads=args.threads,
        )
        print(f"wrote {write_language(language, sources, translations)}")
    return audit(sources, args.languages)


if __name__ == "__main__":
    raise SystemExit(main())
