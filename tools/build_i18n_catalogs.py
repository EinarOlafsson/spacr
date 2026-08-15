#!/usr/bin/env python3
"""Build and audit spaCR's external runtime localization catalogs.

The application keeps a compact hand-reviewed chrome catalog in
``spacr.qt.i18n``.  This tool extracts the much larger surfaces directly from
their canonical English sources:

* every setting label and scientific tooltip body;
* every written settings-category explanation; and
* static text owned by Qt widgets, actions, dialogs and notices.

Translations are generated with permissively licensed Helsinki OPUS models
or M2M100, according to the target language. Strictly rejected hard tails may
be retried with the Apache-2.0 MADLAD-400 checkpoint. Those checkpoints use
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
from collections import Counter, defaultdict
from contextlib import contextmanager
import ctypes
import ctypes.util
import fcntl
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import pprint
import re
import stat
import sys
import tempfile
import time
from typing import Callable, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
# Executing ``python tools/build_i18n_catalogs.py`` otherwise puts only the
# ``tools`` directory at sys.path[0]. An unrelated editable spaCR checkout can
# then win import resolution and silently generate/audit catalogs for the
# wrong repository. Pin this script's own repository before importing any
# runtime catalog or Qt source.
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))
CATALOG_DIR = ROOT / "spacr" / "qt" / "i18n_catalogs"
REVIEWED_RUNTIME_DIR = ROOT / "docs" / "i18n" / "reviewed" / "runtime"

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

# OPUS and M2M remain the stable primary routes and preserve existing cache
# identity. MADLAD is a local-only, permissively licensed secondary route for
# entries that still fail every primary whole-sentence, clause, and fragment
# attempt. Its output receives exactly the same structural and semantic gates;
# its presence can never turn a failed candidate into an accepted one.
SECONDARY_MODEL = "google/madlad400-7b-mt"
SECONDARY_MODEL_FOLDER = "../madlad400-7b-mt"
SECONDARY_LICENSE = "Apache-2.0"
SECONDARY_LANGUAGE_TAGS = {
    "sv": "sv", "de": "de", "es": "es", "zh_CN": "zh",
    "pt": "pt", "hi": "hi", "ko": "ko", "is": "is", "fr": "fr",
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
    "torchcam", "torchvision", "torch", "captum", "napari", "scanpy",
    "btrack", "pylibCZIrw", "czifile", "ComBat", "Hugging Face",
    # Source docstrings use the import/package spelling for these names as
    # often as the display spelling.  Protect both explicitly: allowing a
    # translation model to title-case ``cellpose`` or ``numpy`` changes a
    # Python-facing literal even though the resulting brand name looks
    # superficially reasonable.
    "cellpose", "numpy", "scipy", "skimage", "umap", "python", "spacr",
    "PIL", "cv2", "TensorFlow", "Tk", "QThread", "ConsolePanel",
    "DirectConnection", "GUI", "UI", "DEBUG",
    "RSS", "Yokogawa", "MAD", "IQR", "Tukey", "SUM", "MIN", "MAX",
    "SUCCESS", "FAILED",
    "SKIPPED", "QUEUED", "RUNNING", "TODO", "TO-DO",
    "True", "False", "None", "HomePage",
    "statsmodels", "sklearn", "matplotlib", "DBSCAN", "DataFrame",
    "QApplication", "MainWindow", "QSystemTrayIcon", "uint8", "NaN",
    "Z-prime",
    "NVIDIA", "Python", "Windows", "Linux", "macOS", "OpenGL", "XCB",
    "PATH", "SPEC", "SSH", "Slurm", "HPC", "WHERE",
}, key=len, reverse=True))

_SHORT_QUOTED_LITERAL_RE = re.compile(
    # Quoted option values and UI labels are literals, but a few reviewed
    # quotations are explanatory English prose and must be translated. Keep
    # those out of the literal contract explicitly rather than weakening the
    # general quoted-value protection.
    r'(?<!\w)"(?!(?:the user chose this|we put it there|it is lazy|'
    r'not scored|exclude this debris|measure this colony|Edit mode|'
    r'the run that worked)")'
    r'[A-Za-z][A-Za-z0-9_.:/…-]*'
    r'(?: [A-Za-z0-9_.:/…-]+){0,3}"|'
    r"(?<!\w)'(?!(?:the user chose this|we put it there|it is lazy|"
    r"not scored|exclude this debris|measure this colony|Edit mode)')"
    r"[A-Za-z][A-Za-z0-9_.:/…-]*"
    r"(?: [A-Za-z0-9_.:/…-]+){0,3}'(?!\w)"
)
_SINGLE_QUOTED_LITERAL_RE = re.compile(
    r"(?<!\w)'[A-Za-z][A-Za-z0-9_.:/-]*'(?!\w)|"
    r'(?<!\w)"[A-Za-z][A-Za-z0-9_.:/-]*"'
)
_TRAILING_SPACE_LITERAL_RE = re.compile(
    r"(?<!\w)'[A-Za-z][A-Za-z0-9_.:/ -]*\s+'(?!\w)|"
    r'(?<!\w)"[A-Za-z][A-Za-z0-9_.:/ -]*\s+"'
)
_QUOTE_PROTECT_PATTERNS = frozenset({
    _SHORT_QUOTED_LITERAL_RE,
    _SINGLE_QUOTED_LITERAL_RE,
    _TRAILING_SPACE_LITERAL_RE,
})

# Any explicit RST role is an API/literal contract.  Restricting this to a
# hand-written list missed standard roles such as ``:exc:``, ``:math:`` and
# namespaced roles such as ``:py:meth:``.  Preserve both the role prefix and
# payload byte-for-byte while translating the surrounding prose.
_RST_ROLE_PATTERN = r":(?:[A-Za-z][\w-]*:)?[A-Za-z][\w-]*:`[^`]+`"

_SCIENTIFIC_NOTATION_RE = re.compile(
    # Visible scientific units and variables are data, not prose.  Marian can
    # otherwise drop the non-ASCII glyph while leaving a fluent sentence
    # (``5 µm`` becoming merely ``5``), which silently changes its meaning.
    r"(?<!\w)µ(?:m|M|s)(?:²|³)?(?:/(?:px|pixel|s|min))?(?!\w)|"
    r"(?<!\w)p[ₒₑ](?!\w)|"
    r"[κπδΔ]|§\s*\d+|©"
)

_ORCID_RE = re.compile(
    # ORCID identifiers are scientific provenance, not prose. A translation
    # model must never be allowed to rewrite a digit while translating the
    # surrounding attribution. The final check digit may be ``X``.
    r"(?<![\w-])\d{4}-\d{4}-\d{4}-\d{3}[\dX](?![\w-])",
    re.IGNORECASE,
)

_PROTECT_PATTERNS = (
    re.compile(
        r"^:(?!(?:class|func|mod|meth|attr|data|doc):)"
        r"[A-Za-z][\w-]*(?:\s+[^:]+)?:",
        re.MULTILINE,
    ),
    re.compile(r"^\s*(?:[*-]|#\.)\s+", re.MULTILINE),
    re.compile(r"\|[A-Za-z][^|\n]*\|"),
    re.compile(r"\*\*"),
    re.compile(r"\*"),
    # RST literal-block introducer. API extraction detaches this chrome before
    # generation, while runtime prose can still contain it directly.
    re.compile(r"(?<!:):{2}(?!:)"),
    re.compile(r"</?[A-Za-z][^>]*>"),
    re.compile(r"\{[^{}]+\}"),
    # Inline RST field chrome can follow prose on the same physical line.
    # Preserve the field name and argument while translating its body;
    # otherwise exception disambiguation rewrites ``:raises SpecError:`` into
    # a non-RST ``:throws SpecError:`` field.
    re.compile(r"(?<!\w):raises?\s+[^:\n]+:"),
    re.compile(_RST_ROLE_PATTERN),
    re.compile(r"``[^`]+``|`[^`]+`_?"),
    # Dotted Python names and filenames are identifiers, not prose. Protect
    # them even when an old docstring omitted inline-code markup; otherwise a
    # model can turn ``measurements.db`` into a natural-language phrase or
    # translate one component of ``spacr.settings.descriptions``.
    # Swedish ``t.ex.`` ("for example") is prose, not a dotted Python name.
    # Excluding this exact abbreviation prevents valid Swedish translations
    # from inventing an apparent identifier while every real dotted token
    # remains protected byte-for-byte.
    re.compile(
        r"(?<!\w)(?!(?:t|T)\.ex(?:\.|\b))"
        r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w+)+(?!\w)"
    ),
    re.compile(r"(?<![\w#])#[0-9A-Fa-f]{3,8}(?![0-9A-Fa-f])"),
    re.compile(r"https?://\S+"),
    # Inline installation commands are executable text even when a docstring
    # mentions them inside prose rather than as an indented literal block.
    re.compile(
        r"(?<!\w)(?:python\s+-m\s+)?pip\s+install\s+"
        r"(?:\"[^\"\n]+\"|'[^'\n]+'|[^\s,.;:()]+)"
    ),
    re.compile(r"(?<!\w)(?:--[A-Za-z][\w-]*|-[A-Za-z](?!\w))"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    # These two unquoted identifiers occur inside otherwise-human table cells.
    # Keep them exact without hiding every CamelCase English cue from the
    # semantic source classifier (``DataLoader`` and ``spaCR`` are meaningful
    # context there).
    re.compile(r"(?<!\w)(?:deleteLater|preview_)(?!\w)"),
    # Short quoted values name literal UI labels, log snippets, or option
    # values the reader must be able to find verbatim.  Preserve up to four
    # simple words (for example "Verbose logging" or 'using 12 cpu cores').
    # Longer quoted sentences remain prose and are translated normally.  The
    # word boundary on single quotes keeps apostrophes in contractions and
    # possessives from opening a false literal.
    _SHORT_QUOTED_LITERAL_RE,
    # Literal option values are part of the settings API, even when they are
    # embedded in otherwise translatable prose.  These narrower expressions
    # are retained for compatibility with empty/minimal literal forms.
    _SINGLE_QUOTED_LITERAL_RE,
    # String literals whose trailing space is semantically relevant, for
    # example ``("Train ", "Val ")`` prefixes.
    _TRAILING_SPACE_LITERAL_RE,
    # Mapping/status arrows and comparisons are executable/technical notation,
    # not punctuation a translation model may drop.  Keeping even bare ``>``
    # and ``<`` in this contract also turns an invented angle bracket into a
    # global syntax failure instead of inviting a lossy cleanup pass.
    re.compile(
        r"(?<![-=])(?:->|=>|>=|<=|==|!=|>|<)(?![=>])|"
        r"[→←↔⇒↑↓×≤≥±≈−·²³ⓘ▸◀▶]"
    ),
    _SCIENTIFIC_NOTATION_RE,
    _ORCID_RE,
    re.compile(r"%(?:\d+\$)?[sd]"),
)

_PRODUCT_PROTECT_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(re.escape(term) for term in _PROTECTED_TERMS)
    + r")(?:s)?(?!\w)"
)

_PROTECT_RE = re.compile(
    "|".join(
        [f"(?:{pattern.pattern})" for pattern in _PROTECT_PATTERNS]
        + [f"(?:{_PRODUCT_PROTECT_RE.pattern})"]
    ),
    re.MULTILINE,
)

# The final fragment retry sends prose spans between hard API literals to the
# model.  Emphasis delimiters are hard RST syntax too: allowing Marian to emit
# them caused otherwise good Portuguese translations to lose one of the four
# stars in ``**phrase**``.  Keep the delimiters byte-for-byte while translating
# the words between them.  Isolated emphasized terms can lose sentence context
# in this last-resort pass, so the source-conditioned residue table below also
# translates the small reviewed set of English words that models commonly echo
# there (for example ``*not*``).  Code, CLI flags, products, quoted literals,
# links and format fields remain byte-for-byte protected.
_FRAGMENT_PROTECT_PATTERNS = (
    # Keep nested emphasis around a hard inline API literal atomic.  The prose
    # inside ordinary emphasis still travels with its sentence, but splitting
    # ``*`` away from an adjacent code span would otherwise give the model two
    # punctuation-only fragments and lose the delimiters deterministically.
    re.compile(
        r"(?<!\*)\*(?:``[^`]+``|`[^`]+`_?|"
        r":[A-Za-z]+:`[^`]+`)\*(?!\*)"
    ),
    re.compile(
        r"\*\*(?:``[^`]+``|`[^`]+`_?|:[A-Za-z]+:`[^`]+`)\*\*"
    ),
    # A short quoted UI/log value is one literal island.  These complete
    # patterns must precede the quote-edge fallbacks below: Python chooses the
    # first alternative at the same offset, and protecting just the two quote
    # marks would expose the literal's interior to the fragment model.
    _SHORT_QUOTED_LITERAL_RE,
    _SINGLE_QUOTED_LITERAL_RE,
    _TRAILING_SPACE_LITERAL_RE,
    # Keep prose quotation marks as reconstruction chrome.  The ordinary
    # protection pass deliberately lets quoted sentences travel with their
    # context, but the last-resort fragment pass sees only a small span.  M2M
    # can otherwise turn an unmatched source edge such as ``\"Preview`` into
    # a target angle bracket, invalidating an otherwise usable translation.
    re.compile(r'(?<!\w)["“‘](?=[A-Za-zÀ-ÖØ-öø-ÿ])'),
    re.compile(r'(?<=[A-Za-zÀ-ÖØ-öø-ÿ])["”’](?!\w)'),
    *(
        pattern for pattern in _PROTECT_PATTERNS
        if pattern not in _QUOTE_PROTECT_PATTERNS
    ),
)
_FRAGMENT_PROTECT_RE = re.compile(
    "|".join(
        [f"(?:{pattern.pattern})" for pattern in _FRAGMENT_PROTECT_PATTERNS]
        + [f"(?:{_PRODUCT_PROTECT_RE.pattern})"]
    ),
    re.MULTILINE,
)

# Contextual terminology fixes operate on prose only.  Product names remain
# visible so a source-conditioned phrase can repair grammar around them (for
# example ``GUI thread``), but code spans, quoted option values, CLI flags,
# format fields, links and snake_case identifiers must never be rewritten.
_CONTEXT_HARD_PROTECT_PATTERNS = tuple(
    pattern for pattern in _PROTECT_PATTERNS
    if pattern.pattern not in {r"\*\*", r"\*"}
)
_CONTEXT_HARD_PROTECT_RE = re.compile(
    "|".join(
        f"(?:{pattern.pattern})"
        for pattern in _CONTEXT_HARD_PROTECT_PATTERNS
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
        ("termination devuelve a el hilo de GUI", "la finalización vuelve al hilo de GUI"),
        ("fuera de el hilo de GUI", "fuera del hilo de GUI"),
        ("a el hilo de GUI", "al hilo de GUI"),
        ("GUI objeto", "objeto de GUI"),
        ("base de bases de datos", "base de datos"),
        ("pandas cae las claves", "pandas descarta las claves"),
        ("otros valores aumentan", "otros valores producen"),
        (
            "un atípico produce la desviación estándar",
            "un valor atípico aumenta la desviación estándar",
        ),
        (
            "un valor atípico produce la desviación estándar",
            "un valor atípico aumenta la desviación estándar",
        ),
        (
            "se ejecuta hacia arriba",
            "se conserva en las ejecuciones posteriores",
        ),
        ("Menos cuadrados", "Mínimos cuadrados"),
        ("per-til", "por tesela"),
        ("una sondas fallidas", "una sonda fallida"),
        ("carpeta caída", "carpeta soltada"),
        ("operación de caída", "operación de arrastrar y soltar"),
        ("base de datos informa", "base de datos, se informa"),
        ("una Booleana", "un valor booleano"),
        ("una bool", "un valor booleano"),
        ("Nunca uses", "No use"),
        ("Primero", "primero"),
        ("DeferidoDelete", "DeferredDelete"),
        (". método (str):", ". Método (str):"),
        (". umbral (", ". Umbral ("),
        ("; Esto se desprende", "; esto se desprende"),
        ("congela el GUI", "congela la GUI"),
        ("pertenece al GUI", "pertenece a la GUI"),
        ("el ranking de características", "la clasificación de características"),
        ("El ranking de características", "La clasificación de características"),
        ("Prefijo ``g_``", "Anteponer ``g_``"),
        (
            "su desviación estándar indefinida se mantiene por debajo de "
            "'zscore', coincidiendo con 'iqr'",
            "la desviación estándar puede no estar definida bajo 'zscore', pero "
            "el grupo también se conserva con 'iqr'",
        ),
        (
            "**Requisito de receptor QObject.**Conéctese ``QThread.finished`` a "
            "un método en un QObject que pertenece a la GUI Un cierre no es seguro. "
            "PySide6 asigna la QThread como receptor de un cierre. "
            ":func:`spacr.qt.bridge.make_thread` primero conecta "
            "``thread.finished -> thread.deleteLater``. Ranuras ejecutar en orden "
            "de conexión. DeferredDelete es por lo tanto publicado antes de la "
            "devolución de llamada de cierre. Qt desecha el callback en cola "
            "después de destruir su receptor. El trabajo entonces permanece "
            "activo y ``active_jobs()`` Nunca llega a cero.",
            "**Requisito de receptor QObject.** Conecte ``QThread.finished`` a un "
            "método de un QObject perteneciente al hilo de la GUI. No use un "
            "cierre. PySide6 asigna el propio QThread como receptor del cierre. "
            ":func:`spacr.qt.bridge.make_thread` conecta primero "
            "``thread.finished -> thread.deleteLater``. Las ranuras se ejecutan "
            "en orden de conexión, de modo que DeferredDelete se publica antes "
            "de la llamada al cierre. Qt descarta entonces la llamada en cola "
            "tras destruir su receptor. El trabajo queda activo y "
            "``active_jobs()`` nunca llega a cero.",
        ),
        (
            "**GUI-Relé de hilo.** :class:`_ConsoleRelay` es un QObject que "
            "pertenece al hilo de GUI. Es su ``line`` señal llama a un método en "
            "ese relé. Qt colas una señal emitida de otro hilo. Por lo tanto, el "
            "panel accede a los widgets sólo desde su propio hilo.",
            "**Relé del hilo de la GUI.** :class:`_ConsoleRelay` es un QObject "
            "que pertenece al hilo de la GUI. Su señal ``line`` llama a uno de "
            "los métodos del propio relé. Qt pone en cola las señales emitidas "
            "desde otro hilo. Por tanto, el panel accede a los widgets únicamente "
            "desde el hilo que los posee.",
        ),
    ),
    "zh_CN": (
        ("Cellpose 流量字段", "Cellpose 流场"),
        ("流量字段", "流场"),
        ("这个门", "此阈值"),
        ("输入文件包含", "输入文件夹包含"),
        ("图像图像裁剪", "图像裁剪"),
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
        ("la exécution", "l’exécution"),
        ("L'criblage", "Le criblage"),
        ("l'criblage", "le criblage"),
        ("L’criblage", "Le criblage"),
        ("l’criblage", "le criblage"),
    ),
}

# A few reviewed corrections span protected inline-code tokens.  The ordinary
# contextual pass masks those tokens before changing prose, so these narrowly
# scoped, syntax-preserving replacements run only after literals are restored.
# Each entry has been checked with ``_syntax_preserved``; this is not a general
# post-processor and must not contain broad lexical substitutions.
POST_CONTEXT_REPLACEMENTS: Mapping[
    str, tuple[tuple[str, str], ...]
] = {
    "es": (
        ("Prefijo ``g_``", "Anteponer ``g_``"),
        (
            "su desviación estándar indefinida se mantiene por debajo de "
            "'zscore', coincidiendo con 'iqr'",
            "la desviación estándar puede no estar definida bajo 'zscore', pero "
            "el grupo también se conserva con 'iqr'",
        ),
        (
            "**Requisito de receptor QObject.**Conéctese ``QThread.finished`` a "
            "un método en un QObject que pertenece a la GUI Un cierre no es seguro. "
            "PySide6 asigna la QThread como receptor de un cierre. "
            ":func:`spacr.qt.bridge.make_thread` primero conecta "
            "``thread.finished -> thread.deleteLater``. Ranuras ejecutar en orden "
            "de conexión. DeferredDelete es por lo tanto publicado antes de la "
            "devolución de llamada de cierre. Qt desecha el callback en cola "
            "después de destruir su receptor. El trabajo entonces permanece "
            "activo y ``active_jobs()`` Nunca llega a cero.",
            "**Requisito de receptor QObject.** Conecte ``QThread.finished`` a un "
            "método de un QObject perteneciente al hilo de la GUI. No use un "
            "cierre. PySide6 asigna el propio QThread como receptor del cierre. "
            ":func:`spacr.qt.bridge.make_thread` conecta primero "
            "``thread.finished -> thread.deleteLater``. Las ranuras se ejecutan "
            "en orden de conexión, de modo que DeferredDelete se publica antes "
            "de la llamada al cierre. Qt descarta entonces la llamada en cola "
            "tras destruir su receptor. El trabajo queda activo y "
            "``active_jobs()`` nunca llega a cero.",
        ),
        (
            "**GUI-Relé de hilo.** :class:`_ConsoleRelay` es un QObject que "
            "pertenece al hilo de GUI. Es su ``line`` señal llama a un método en "
            "ese relé. Qt colas una señal emitida de otro hilo. Por lo tanto, el "
            "panel accede a los widgets sólo desde su propio hilo.",
            "**Relé del hilo de la GUI.** :class:`_ConsoleRelay` es un QObject "
            "que pertenece al hilo de la GUI. Su señal ``line`` llama a uno de "
            "los métodos del propio relé. Qt pone en cola las señales emitidas "
            "desde otro hilo. Por tanto, el panel accede a los widgets únicamente "
            "desde el hilo que los posee.",
        ),
    ),
    "pt": (
        (
            ":func:`sibling_sources` returns **cada** comparable file in the "
            "folder. The panels previously sent that complete list directly "
            "to their field-of-view selector. We measured this behavior on a "
            "384-well plate with 16 fields and 4 channels (24 576 files), and "
            "on another plate four times larger (98 304 files):",
            ":func:`sibling_sources` retorna **cada** arquivo comparável da "
            "pasta. Antes, os painéis enviavam essa lista completa diretamente "
            "ao seletor de campo de visão. Medimos esse comportamento em uma "
            "placa de 384 poços com 16 campos e 4 canais (24 576 arquivos) e em "
            "outra placa quatro vezes maior (98 304 arquivos):",
        ),
        (
            "o título da janela. Usa por padrão the mask's nome do arquivo.",
            "o título da janela. Usa por padrão o nome do arquivo de máscara.",
        ),
        (
            "onde escrever os arquivos NPZ intermediários (opcional). Usa por "
            "padrão a scratch subfolder under the stack folder.",
            "onde escrever os arquivos NPZ intermediários opcionais. Usa por "
            "padrão uma subpasta temporária sob a pasta da pilha.",
        ),
        (
            "Compat cadeia de caracteres for the old KeysDialog agora descreve "
            "o estado install/login da CLI.",
            "Cadeia de caracteres de compatibilidade para o KeysDialog antigo; "
            "agora descreve o estado de instalação e login da CLI.",
        ),
        (
            ":func:`sibling_sources` listas **cada** arquivo comparável na pasta, "
            "e os painéis alimentados que em linha reta em seu campo de visão "
            "dropdown.Medido em uma placa de 384 poços em 16 campos e 4 canais "
            "(24 576 arquivos) e em um quatro vezes maior (98 304 arquivos):",
            ":func:`sibling_sources` lista **cada** arquivo comparável na pasta, "
            "e os painéis usavam essa lista diretamente no seletor de campo de "
            "visão. Isso foi medido em uma placa de 384 poços com 16 campos e 4 "
            "canais (24 576 arquivos) e em outra quatro vezes maior (98 304 "
            "arquivos):",
        ),
        (
            "A diferença entre duas configurações é ditada por consequência.",
            "Classifica por consequência as diferenças entre dois "
            "dicionários de configurações.",
        ),
        (
            "Configurações de simulação ditadas como a primeira linha.",
            "Dicionário de configurações da simulação registrado na "
            "primeira linha.",
        ),
        (
            "marcador de texto *>*>  ``'list of paths'``",
            "marcador de *cadeia de caracteres*  ``'list of paths'``",
        ),
        (
            "a escolha inicial  *>*> é síncrona",
            "a escolha *padrão* é síncrona",
        ),
    ),
}

# A computation run and a tracked trajectory are both commonly rendered as
# "run" in English.  Keep automatic repairs away from any sentence that is
# explicitly about tracks, tracking, or tracked objects.
_COMPUTE_RUN_SOURCE = (
    r"(?is)\A(?=.*\b(?:runs?|running)\b)"
    r"(?=.*\b(?:pipelines?|jobs?|workers?|modules?|process(?:es|ing)?|"
    r"execut(?:e|ed|es|ing|ion)|CLI|commands?|folders?|outputs?|settings?|"
    r"cancel(?:led|lation)?|fail(?:ed|ure)?|status|ledger|artifacts?|models?|"
    r"training|inference|analysis|batches?|replay|completed?|resume|previous|"
    r"current)\b)"
    r"(?!.*\b(?:tracks?|tracked|tracking|trajector(?:y|ies)|streak|"
    r"run[- ]length|run of characters?)\b)"
)

_COMPUTE_THREAD_SOURCE = (
    r"(?is)\A(?=.*\b(?:threads?|threading|threaded)\b)"
    r"(?=.*\b(?:QThread|GUI|workers?|main|background|thread[- ]safe|"
    r"threading|threaded|affinity|queues?|executors?|concurr\w*|connections?|"
    r"jobs?|signals?|slots?|process(?:es)?|OpenCV|parallel|locks?|SQLite|"
    r"callbacks?|widgets?|Qt)\b)"
    r"(?!.*\b(?:needle|sewing|cloth|forum|conversation|discussion thread)\b)"
)

_PIPELINE_SOURCE = r"(?i)\bpipelines?\b"
_DATA_GATE_SOURCE = (
    r"(?is)\A(?=.*\b(?:gates?|gating)\b)"
    r"(?!.*\b(?:logic|electronic|airport|fence|entrance|door)\s+gates?\b)"
)
_PLANE_SOURCE = (
    r"(?is)\A(?=.*\bplanes?\b)"
    r"(?!.*\b(?:aircraft|airplanes?|aeroplanes?|aviation|flight|airport|"
    r"pilots?|runways?|take[- ]?off|land(?:ed|ing|s)?|Cartesian|geometric|"
    r"geometry|Euclidean|coordinate|woodwork\w*|carpenter(?:'s)?|hand tool)\b)"
)
_IMAGE_TILE_SOURCE = (
    r"(?is)\A(?=.*\btiles?\b)"
    r"(?=.*\b(?:images?|pixels?|stitch(?:ed|ing)?|align(?:ed|ment|ing)?|"
    r"mosaics?|overlaps?|registration|channels?|arrays?|crops?|PNG|TIFF|OME|"
    r"ND2|coordinates?|offsets?|seams?|composit(?:e|ing))\b)"
)
_SCIENTIFIC_PLATE_SOURCE = (
    r"(?is)\A(?=.*\bplates?\b)"
    r"(?=.*\b(?:wells?|microplates?|assays?|96|384|1536|imaging|sequencing|"
    r"microscope|layout|barcodes?|gRNAs?|experiment|spot|phenotypes?|samples?|"
    r"rows?|columns?|fields?)\b)"
)
_IMAGE_CROP_SOURCE = (
    r"(?is)\A(?=.*\bcrops?\b)"
    r"(?!.*\b(?:agriculture|agricultural|farm(?:ing)?|harvest(?:ed|ing)?|"
    r"farmers?|grow(?:s|ing|n)?|soil|crop rotation)\b)"
)
_WINDOW_RAISE_SOURCE = (
    r"(?is)\A(?=.*(?:\braise or focus\b|\braise\b.{0,50}\b(?:window|screen)\b))"
)

_SCIENTIFIC_HIT_SOURCE = (
    r"(?is)\A(?=.*\bhits?\b)(?=.*\b(?:screen(?:s|ing)?|genes?|gRNAs?|"
    r"library|libraries|rank(?:s|ed|ing)?|FDR|effect|candidate|call(?:s|ed|ing)?|"
    r"threshold|phenotypes?|follow-up|positive)\b)"
)

# ``key`` means a keyboard key only in explicit input-event prose. Everywhere
# else in spaCR's API it names a mapping/database identifier, for which
# Portuguese ``tecla`` is the wrong sense.
_MAPPING_KEY_SOURCE = (
    r"(?is)\A(?=.*\bkeys?\b)"
    r"(?=.*\b(?:mapping|dictionary|dict|json|settings?|configuration|config|"
    r"databases?|tables?|rows?|records?|schemas?|fields?|columns?|identifiers?|identity|"
    r"lookup|cache|metadata|payload|entries|values?|namespace|parameters?|"
    r"kwargs)\b)"
    r"(?!.*(?:\b(?:keyboard|keypress|key press|key event|shortcut|hotkey|"
    r"Backspace|Escape|arrow keys?|modifier keys?|keystroke|Qt key)\b|"
    r"\bpress(?:ed|ing)?(?:\s+\w+){0,3}\s+keys?\b|"
    r"\bkeys?(?:\s+\w+){0,3}\s+press(?:ed|ing)?\b))"
)

_DICTIONARY_SOURCE = (
    r"(?is)\A(?=.*\b(?:dicts?|dictionaries|dictionary)\b)"
    r"(?!.*\bdictat(?:e|es|ed|ing|ion|ions)\b)"
)

# Additional software/imaging senses observed in the M2M Chinese/Korean API
# pass.  These predicates are narrow and source-conditioned; ordinary land,
# broadcast, human-actor, food and navigation senses remain valid.
_SOFTWARE_QUEUE_SOURCE = (
    r"(?is)\A(?=.*\bqueues?\b)"
    r"(?!.*(?:\b(?:rear|back|tails?|tail[- ]end)\s+of\s+the\s+queues?\b|"
    r"\b(?:physical|waiting)\s+(?:line|queue)\b|"
    r"\b(?:people|persons?|customers?|passengers?|visitors?|shoppers?|"
    r"travellers?|travelers?)\b.{0,40}\b(?:wait\w*|stand\w*|form\w*|"
    r"line\s+up|queu(?:ed|ing))\b|"
    r"\b(?:wait\w*|stand\w*|form\w*|line\s+up)\b.{0,40}\bqueues?\b|"
    r"\bqueues?\b.{0,40}\b(?:held|contained)?\s*(?:waiting|standing)\s+"
    r"(?:people|persons?|customers?|passengers?|visitors?|shoppers?|"
    r"travellers?|travelers?)\b|"
    r"\bqueues?\b.{0,40}\b(?:outside|airport|bank|store)\b))"
)
_IMAGING_FIELD_SOURCE = (
    r"(?is)\A(?=.*\b(?:fields?|FOVs?)\b)"
    r"(?=.*\b(?:images?|wells?|plates?|microscop\w*|acquisition|masks?|"
    r"crops?|objects?|planes?|FOVs?|channels?|tiles?)\b)"
    r"(?!.*\b(?:farm(?:er|ers|ing)?|agricultur\w*|grow(?:s|ing|n)?|soil|"
    r"meadow|land|domain|region|area|scope)\b)"
)
_IMAGING_CHANNEL_SOURCE = (
    r"(?is)\A(?=.*\bchannels?\b)"
    r"(?=.*\b(?:images?|planes?|microscop\w*|fluorescen\w*|RGB|masks?|"
    r"arrays?|TIFF|PNG|Cellpose|intensit\w*|stains?|lasers?|DAPI)\b)"
    r"(?!.*\bbroadcast\w*\b)"
)
_SOFTWARE_CLASSIFIER_SOURCE = (
    r"(?is)\A(?=.*\bclassifiers?\b)(?!.*\b(?:human|person|manual)\b)"
)
_HUMAN_READABLE_SOURCE = (
    r"(?i)\bhuman(?:[- ]readable|\s+(?:description|summary|text|form|label|"
    r"reference))\b"
)
_SECRET_KEY_SOURCE = (
    r"(?is)\b(?:API[- ]?keys?|secret keys?|encryption keys?|credentials?|"
    r"passwords?|tokens?|vault)\b"
)

_ZH_WEB_ARTIFACT_RE = re.compile(
    r"(?:此分類|本分類)?\s*上一篇|(?:此分類|本分類)?\s*下一篇|"
    r"首頁\s*[>〉]\s*外文書|收藏此帖子|(?:点|點)击此处|"
    r"(?:相关|相關)文章|原文\s*[:：]|英文名",
)
_ZH_WEB_SOURCE_ESCAPE_RE = re.compile(
    r"(?i)\b(?:previous|next)\s+(?:article|post)|home\s*[>〉]|"
    r"foreign books|bookmark (?:this )?post|click here|related articles|"
    r"original text\s*[:：]|English name",
)


def _simplify_chinese_prose(value: str) -> str:
    """Normalize generated zh_CN prose with Apache-2.0 OpenCC ``t2s``.

    Generation and audit both fail loudly when the host dependency is absent.
    Protected API/RST spans bypass OpenCC byte-for-byte.
    """
    library_name = ctypes.util.find_library("opencc")
    config_path = Path("/usr/share/opencc/t2s.json")
    if not library_name or not config_path.is_file():
        raise RuntimeError(
            "zh_CN generation requires OpenCC 1.1+ and "
            "/usr/share/opencc/t2s.json"
        )
    library = ctypes.CDLL(library_name)
    library.opencc_open.argtypes = [ctypes.c_char_p]
    library.opencc_open.restype = ctypes.c_void_p
    library.opencc_convert_utf8.argtypes = [
        ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
    ]
    library.opencc_convert_utf8.restype = ctypes.c_void_p
    library.opencc_convert_utf8_free.argtypes = [ctypes.c_void_p]
    library.opencc_close.argtypes = [ctypes.c_void_p]
    handle = library.opencc_open(str(config_path).encode("utf-8"))
    if not handle or handle == ctypes.c_void_p(-1).value:
        raise RuntimeError("OpenCC could not load t2s.json")

    def convert(fragment: str) -> str:
        if not fragment:
            return fragment
        encoded = fragment.encode("utf-8")
        pointer = library.opencc_convert_utf8(
            handle, encoded, len(encoded),
        )
        if not pointer or pointer == ctypes.c_void_p(-1).value:
            raise RuntimeError("OpenCC failed to normalize zh_CN prose")
        try:
            return ctypes.string_at(pointer).decode("utf-8")
        finally:
            library.opencc_convert_utf8_free(pointer)

    try:
        source = str(value)
        pieces: list[str] = []
        cursor = 0
        for match in _CONTEXT_HARD_PROTECT_RE.finditer(source):
            pieces.append(convert(source[cursor:match.start()]))
            pieces.append(match.group(0))
            cursor = match.end()
        pieces.append(convert(source[cursor:]))
        return "".join(pieces)
    finally:
        library.opencc_close(handle)


def _has_traditional_chinese_prose(value: str) -> bool:
    """Return whether OpenCC would simplify any unprotected zh_CN prose.

    The answer is a release contract, so a missing or broken OpenCC runtime is
    an error rather than a best-effort character-list heuristic.  Generation
    and audit therefore use exactly the same t2s conversion and fail closed.
    """
    return _simplify_chinese_prose(value) != str(value)

# Distinguish raising an exception from increasing a value. The exclusions
# cover the few genuine increase senses in the API; the positive cues cover
# explicit error classes and the ordinary exception-control phrasing.
_EXCEPTION_RAISE_SOURCE = (
    r"(?is)\A(?=.*(?:"
    r"\b(?:raise|raises|raised|raising|re-raises?)\b.{0,120}"
    r"(?:[A-Za-z_]\w*(?:Error|Exception|Cancelled)|error|exception|invalid|"
    r"unsupported|unrecognised|unrecognized|unknown|missing|mismatch|"
    r"failure|strict mode)|"
    r"(?:[A-Za-z_]\w*(?:Error|Exception|Cancelled)|error|exception|invalid|"
    r"unsupported|unrecognised|unrecognized|unknown|missing|mismatch|"
    r"failure|strict mode).{0,120}\b(?:raise|raises|raised|raising|re-raises?)\b|"
    r"\b(?:never|does not|do not|did not|cannot|can't)\s+"
    r"(?:ever\s+)?(?:raise|raises|raised)\b))"
)

# Source-conditioned repairs for common scientific false friends.  These are
# deliberately narrower than CONTEXT_REPLACEMENTS: for example, Chinese 门 is
# ordinary in navigation prose but means the wrong thing for a cytometry gate.
SOURCE_CONTEXT_REPLACEMENTS: Mapping[
    str, tuple[tuple[str, str, str], ...]
] = {
    "sv": (
        (r"\bgates?\b", "stängnings", "gate-"),
        (_COMPUTE_RUN_SOURCE, "loppet", "körningen"),
        (_COMPUTE_RUN_SOURCE, "lopp", "körning"),
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
        (_COMPUTE_RUN_SOURCE, "Rennen", "Ausführung"),
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
        (r"\bhits?\b", "golpes", "aciertos"),
        (_COMPUTE_RUN_SOURCE, "carreras", "ejecuciones"),
        (_COMPUTE_RUN_SOURCE, "carrera", "ejecución"),
        (_COMPUTE_RUN_SOURCE, "recorridos", "ejecuciones"),
        (_COMPUTE_RUN_SOURCE, "recorrido", "ejecución"),
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
        (r"\bpipelines?\b", "tubería", "flujo de trabajo"),
        (r"\bpipelines?\b", "tubos", "flujos de trabajo"),
        (r"\bpipelines?\b", "gasoductos", "flujos de trabajo"),
        (r"\bpipelines?\b", "gasoducto", "flujo de trabajo"),
    ),
    "zh_CN": (
        (r"\bgates?\b", "箱门", "箱式门控"),
        (r"\bgates?\b", "门", "门控"),
        (r"\bflow field\b", "流量", "流"),
        (r"\bmasks?\b", "口罩", "掩膜"),
        (r"\bcells?\b", "电池", "细胞"),
        (r"\bwells?\b", "水井", "孔"),
        (r"\bwells?\b", "井", "孔"),
        (r"\bhits?\b", "点击", "命中"),
        (r"\bguides?\b", "指南", "引导 RNA"),
        (r"\bguides?\b", "向导 RNA", "引导 RNA"),
        (r"\bmasks?\b", "面具", "掩膜"),
        (r"\bplates?\b", "板块", "微孔板"),
        (_COMPUTE_RUN_SOURCE, "赛跑", "运行"),
        (_COMPUTE_RUN_SOURCE, "一跑", "一次运行"),
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
        (r"\bhits?\b", "golpes", "acertos"),
        (_COMPUTE_RUN_SOURCE, "corridas", "execuções"),
        (_COMPUTE_RUN_SOURCE, "corrida", "execução"),
        (r"\bcrops?\b", "culturas", "recortes"),
        (r"\bcrops?\b", "cultura", "recorte"),
        (r"\bworkers?\b", "trabalhadores", "processos worker"),
        (r"\bbatches?\b", "lotes", "batches"),
        (r"\bscreens?\b", "telas", "triagens"),
        (r"\bscreens?\b", "tela", "triagem"),
        (r"\bheadless(?:ly)?\b", "sem cabeça", "sem interface gráfica"),
        (r"\bpipelines?\b", "tubos", "fluxos de trabalho"),
    ),
    "hi": (
        (r"\bgates?\b", "द्वार", "गेट"),
        (r"\bmasks?\b", "चेहरे का मास्क", "मास्क"),
        (r"\bcells?\b", "बैटरी", "कोशिका"),
        (r"\bwells?\b", "कुआँ", "वेल"),
        (r"\bhits?\b", "मार", "हिट"),
        (_COMPUTE_RUN_SOURCE, "दौड़", "रन"),
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
        (_COMPUTE_RUN_SOURCE, "달리기", "실행"),
        (_COMPUTE_RUN_SOURCE, "달리지", "실행되지"),
        (r"\bcrops?\b", "농작물", "크롭"),
        (r"\bcrops?\b", "작물", "크롭"),
        (r"\bworkers?\b", "작업자", "워커 프로세스"),
        (r"\bscreens?\b", "화면", "스크리닝"),
    ),
    "is": (
        (r"\bhits?\b", "högg", "niðurstöður"),
        (_COMPUTE_RUN_SOURCE, "hlaupið", "keyrslan"),
        (_COMPUTE_RUN_SOURCE, "hlauparinn", "keyrslan"),
        (_COMPUTE_RUN_SOURCE, "hlaup", "keyrsla"),
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
        (r"\bhits?\b", "touches", "hits"),
        (r"\bhits?\b", "touche", "hit"),
        (r"\bhits?\b", "points marqués", "hits"),
        (r"\bhits?\b", "point marqué", "hit"),
        (_COMPUTE_RUN_SOURCE, "courses", "exécutions"),
        (_COMPUTE_RUN_SOURCE, "course", "exécution"),
        (_COMPUTE_RUN_SOURCE, "parcours", "exécution"),
        (r"\bcrops?\b", "cultures", "vignettes"),
        (r"\bcrops?\b", "culture", "vignette"),
        (r"\bworkers?\b", "travailleurs", "processus workers"),
        (r"\bbatches?\b", "lots", "batchs"),
        (r"\bscreens?\b", "écrans", "criblages"),
        (r"\bscreens?\b", "écran", "criblage"),
        (r"\bheadless(?:ly)?\b", "sans tête", "sans interface graphique"),
    ),
}

# Regex repairs cover inflected variants that literal substitutions cannot
# catch.  Every rule is conditioned on the English source sense, so ordinary
# words such as French ``piste`` (a tracking trajectory) or German ``Kultur``
# are changed only when the source says run/crop/gate/etc. in spaCR's domain.
_UI_SCREEN_SOURCE = (
    r"(?is)\A(?=.*\bscreens?\b)(?=.*(?:"
    r"\b(?:Qt|GUI|AppScreen|QWidget|widget|popover|toggle|canvas|backdrop)\b|"
    r"\b(?:home|module|application|app|settings|report|main|generic|blank|"
    r"empty|current|own)\s*[- ]?screens?\b|"
    r"\bscreens?\s+(?:key|content|layout|title|header|control|widget|factory|"
    r"state|selection|insights|dashboard|needs?|wants?|calls?|shows?|reads?|"
    r"asks?|writes?|exists?|exposes?|reuses?|restores?|finds?|starts?|opens?|appears?|"
    r"edits?|receives?)\b|"
    r"\bscreens?\s+of\s+(?:its|their|one's)\s+own\b|"
    r"\b(?:draw|drawing|open|opening|switch|return|leave|leaving|remain|"
    r"remaining|pre-fill|fill|restore|restoring|show|shown|display|displayed|"
    r"place|placed|paint|painted)\b.{0,50}\bscreens?\b|"
    r"\blower\s+DPI\s+for\s+the\s+screens?\b|"
    r"\bscreens?\b.{0,50}\b(?:drop|folder|file picker|Run button|settings "
    r"form|settings panel|interface|window|shown)\b|"
    r"\bdrop\b.{0,80}\bscreens?\b|"
    # Named application surfaces and common on-screen interactions.  These
    # were previously rendered as Portuguese ``triagem`` (scientific screen)
    # even though the source unambiguously means a GUI screen.
    r"\b(?:Annotate|Classify|Gate Editor|Image UMAP|Power|Report|Mask|"
    r"Measure|Batch|Convert|Project Browser|DB Browser|Database Browser|"
    r"Plate View|Home|Pipeline Graph|QC)\s+screens?\b|"
    r"\bscreens?\s+(?:itself|filter|dropdown|dialog|button|form|panel|picker|"
    r"view|GUI|route|module|navigation|sidebar|tab)\b|"
    r"\b(?:on[- ]screen|screen width|screen height|full screen|startup screen|"
    r"screen nobody|screen crashes?|screen opens?|screen closes?|screen is "
    r"seeded|screen was seeded|screen currently|screen exposes?|screen "
    r"displays?|screen renders?|screen restores?|screen writes?|screen reads?|"
    r"screen chooses?|screen receives?|screen returns?)\b))"
    r"(?!.*\b(?:real|simulated|pooled|CRISPR|widefield)\b.{0,40}\bscreens?\b)"
    r"(?!.*\bscreens?\s+(?:features?|hits?|wells?|data|scores?|"
    r"phenotypes?|regression|classification)\b)"
)

_SCIENTIFIC_SCREEN_SOURCE = (
    r"(?is)\A(?=.*\bscreens?\b)"
    r"(?=.*\b(?:CRISPR|pooled|genome[- ]wide|gRNAs?|genes?|genotypes?|hits?|"
    r"phenotypes?|libraries?|widefield|assay|regression|classification|"
    r"screening)\b)"
    r"(?!.*\b(?:Qt|GUI|AppScreen|QWidget|widget|window|canvas|popover|sidebar|"
    r"button|dialog|dropdown|layout|title|header|home|module|application|"
    r"settings)\b)"
)

# Plain ``screen`` is intrinsically ambiguous. These reviewed API blocks are
# GUI surfaces that lack enough surrounding vocabulary for the general sense
# detector above; the five scientific-screen blocks are deliberately absent.
_GUI_SCREEN_SOURCE_SHA256 = frozenset({
    "6fd6bff9288e5324729f83b8238aa9152544f7f91845c2df5934b01c38b020be",
    "bc903bbc38ad17e7f3fa76466efe634320fcca1d103661e435e06a1e5ed59298",
    "f550be59d613467473065be728d1a8ef3d5c260450bf2ed35841289382f9aaed",
    "9d1a3fbf509f9e7769db235d928dd11e07f8c40bab8c1d9b8e9dbd11f269de67",
    "0dd486e48740f287e23e78edebe8a861e52fdae62ab309aaac6cf89eb4e99971",
    "0f3ace26c508e3ce172585d8e2c9aa923a2e6a0d27174ae0fdad9a25f95cd5ab",
    "fc5fcedff1daa0125022ac4954822d062bbfc6c0e6f1d75396da55fc83bf8b5c",
    "02847f1d829396161e8b9d1a7465f53db5cd95fc6b562fd8ad739660a76d1f6f",
    "578b870f2c8294483c7d7162d4ca00cf313706f6e7b20310e496459017500714",
    "c8f6f913607746584c705f0f9ecf3747a7cfd66a37ae4c8f66b3faa847b4081e",
    "c022034b4454738c44010a0bc1f4c15e6d440703b032923a06646226f3284670",
    "eb45da9d1e29931d0f58b11b24dcc56f359e30da48d83c077134a81c310689c9",
    "4ae0d1e4c26f72676891351b076f6ad6f834cd288d6dda84ae9930da1fc3538b",
    "3c5eee8e034abf6cfa0d3a700fe0c3d9bd8f71f20863eb94e31d02626e373f85",
})


def _gui_screen_source(source: str) -> bool:
    return bool(
        hashlib.sha256(str(source).encode("utf-8")).hexdigest()
        in _GUI_SCREEN_SOURCE_SHA256
        or re.search(_UI_SCREEN_SOURCE, str(source), flags=re.IGNORECASE)
    )


def _english_well_sense_counts(source: str) -> tuple[int, int]:
    """Return ``(all well tokens, scientific/container well tokens)``."""
    text = str(source)
    matches = list(re.finditer(r"(?i)(?<![A-Za-z])wells?(?![A-Za-z])", text))
    noun_count = 0
    for match in matches:
        token = match.group(0).casefold()
        before = text[max(0, match.start() - 60):match.start()]
        after = text[match.end():match.end() + 60]
        is_noun = token == "wells" or bool(
            re.search(
                r"(?i)(?:\b(?:a|an|the|this|that|these|those|each|every|"
                r"single|control|sample|empty|positive|negative|treated|"
                r"untreated|per|one|two|three|96|384|1536)[ -]*)$",
                before,
            )
            or re.match(
                r"(?i)(?:[-_ ](?:id|row|column|level|mean|median|count|index|"
                r"name|position|coverage|effect|score|data|table|layout|plate))",
                after,
            )
            or re.search(r"(?i)\b(?:plate|microplate|assay)\b", before)
            or re.search(r"(?i)\b(?:plate|microplate|assay)\b", after)
        )
        noun_count += int(is_noun)
    return len(matches), noun_count


_SEMANTIC_BAD_TARGETS: Mapping[str, Mapping[str, str]] = {
    "ui-screen": {
        "fr": r"\b(?:criblages?|dépistages?)\b",
        "hi": r"स्क्रीनिंग",
        "ko": r"스크리닝",
        "is": r"\b(?:skimun\w*|skiman\w*)\b",
        "zh_CN": r"(?:筛选|筛查)",
        "pt": r"\btriagens?\b",
    },
    "scientific-screen": {
        "fr": r"\b(?:écrans?|interfaces?)\b",
        "hi": r"स्क्रीन(?!िंग)",
        "ko": r"화면",
        "is": r"\bskjá\w*\b",
        "zh_CN": r"(?:屏幕|界面)",
        "pt": r"\btelas?\b",
    },
    "pipeline": {
        "fr": r"\b(?:canalisations?|tuyaux?|oléoducs?|gazoducs?)\b",
        "hi": r"(?:नलिकाएँ?|नलियाँ?|तेल पाइपलाइन|गैस पाइपलाइन)",
        "ko": r"(?:배관|관로|송유관|가스관)",
        "is": r"\b(?:leiðsl\w*|píp\w*)\b",
        "zh_CN": r"(?:管道|油管|气管)",
        "pt": r"\b(?:gasodutos?|oleodutos?|tubula(?:ção|ções))\b",
    },
    "gate": {
        "fr": r"\b(?:portes?|fermetures?)\b",
        "hi": r"(?:द्वार|दरवाज़[ाे]|दरवाज[ाे])",
        "ko": r"(?<![가-힣])문(?:들)?(?![가-힣])",
        "is": r"\b(?:hlið\w*|hurð\w*|dyr)\b",
        "zh_CN": r"(?<!部)(?<!入)(?<!专)(?<!类)(?<!出)门(?!控|槛|户|口|类|窗|外)",
        "pt": r"\b(?:portas?|portões?)\b",
    },
    "power": {
        "fr": r"\bpouvoirs?\b",
        "hi": r"(?:सत्ता|अधिकार|हुकूमत)",
        "ko": r"(?:권력|권한|전력)",
        "is": r"\b(?:vald|völd|yfirráð|orka|afl)\b",
        "zh_CN": r"(?:权力|权势|权限|电力|功率)",
        "pt": r"\bpoder(?:es)?\b",
    },
    "dictionary": {
        "fr": r"\b(?:dictées?|dictations?)\b",
        "hi": r"(?:श्रुतलेख|डिक्टेशन)",
        "ko": r"(?:받아쓰기|구술)",
        "is": r"\b(?:einræði|upplestur|fyrirmæli)\b",
        "zh_CN": r"(?:听写|口述)",
        "pt": r"\bditad\w*\b",
    },
    "mapping-key": {
        "fr": r"\btouches?\b",
        "hi": r"(?:कीबोर्ड\s+(?:की|कुंजी)|कुंजीपटल\s+कुंजी)",
        "ko": r"(?:키보드\s*키|자판\s*키)",
        "is": r"\b(?:lyklaborðslykill|lyklaborðslyklar)\b",
        "zh_CN": r"(?:键盘(?:键|按键))",
        "pt": r"\bteclas?\b",
        "es": r"\bteclas?\b",
    },
    "plane": {
        "fr": r"\b(?:avions?|aéronefs?)\b",
        "hi": r"(?:विमान|हवाई जहा[ज़ज])",
        "ko": r"(?:비행기|항공기)",
        "is": r"\b(?:flugvél\w*|loftfar|loftför)\b",
        "zh_CN": r"(?:飞机|航空器|飞行器)",
        "pt": r"\bavi(?:ão|ões)\b",
    },
    "tile": {
        "fr": r"\bcarreaux?\b",
        "hi": r"(?:(?:छत|फ़र्श|फर्श)\s+(?:की\s+)?टाइल)",
        "ko": r"(?:(?:지붕|바닥)\s*타일)",
        "is": r"\b(?:þakflís\w*|gólfflís\w*)\b",
        "zh_CN": r"(?:瓷砖|地砖|屋顶瓦|地板砖)",
        "pt": r"\b(?:azulejos?|telhas?)\b",
    },
    "plate": {
        "fr": (
            r"(?:\b(?:une|la|cette|chaque|par|de la|d['’]une)\s+assiettes?\b|"
            r"\b(?:un|le|du|au|ce|chaque)\s+plats?\b|\bplats\b)"
        ),
        "hi": r"(?:थाली|तश्तरी)",
        "ko": r"(?:접시|그릇)",
        "is": r"\b(?:diskur|diskar|diskinn|skálar?)\b",
        "zh_CN": r"(?:盘子|碟子|餐盘)",
        "pt": r"\bpratos?\b",
    },
    "crop": {
        "fr": r"\b(?:cultures?|récoltes?|moissons?)\b",
        "hi": r"(?:फसल|कटाई|खेती)",
        "ko": r"(?:농작물|작물|수확물?|수확)",
        "is": r"\b(?:ræktun|uppskera|uppskeru|uppskerur|uppskerunnar)\b",
        "zh_CN": r"(?:作物|收获|种植)",
        "pt": r"\b(?:culturas?|colheitas?)\b",
    },
    "run": {
        "fr": r"\b(?:courses?|pistes?|parcours)\b",
        "hi": r"(?:दौड़\S*|भागना)",
        "ko": r"(?:달리기|경주)",
        "is": r"\b(?:hlaup|hlaupið|hlauparinn|kapphlaup)\b",
        "zh_CN": r"(?:赛跑|賽跑|跑步|竞赛|競賽|走路|步行|跑程|一跑)",
        "pt": r"\bcorridas?\b",
    },
    "thread": {
        "fr": r"\bfils?\s+(?:de discussion|de conversation)\b",
        "hi": r"(?:धागा|धागे|तार|तारों|चर्चा सूत्र)",
        "ko": (
            r"(?<![가-힣])(?:실|트레일|스트립)"
            r"(?:에서|으로|을|를|은|는|이|가|과|와|에)?(?![가-힣])|"
            r"(?:토론|대화)\s*(?:스레드|쓰레드)"
        ),
        "is": r"(?:skrifarhátíð|GUI-tré|vinnuflu|rökvöldum|vinnumálinu|\bstríð\b)",
        "zh_CN": r"(?:字线|呼叫线|工人线|工作线(?!程)|GUI线|线程讨论|讨论串|对话串|聊天线程)",
        "pt": r"\b(?:throws?|thoughts?)\b|\b[Oo]\s+threads\s+da GUI\b",
    },
    "exception-increase": {
        "fr": r"\b(?:augment\w*|accro(?:î|i)\w*|hauss\w*)\b",
        "hi": r"(?:बढ़\S*)",
        "ko": r"(?:증가(?:시키|하)?\S*|높이\S*|올리\S*)",
        "is": r"\b(?:hækk\w*|aukn\w*)\b",
        "zh_CN": r"(?:增加|提高|升高)",
        "pt": r"\b(?:aument\w*|elev\w*|re-elev\w*)\b",
        "es": r"\b(?:elev\w*|aument\w*)\b",
    },
    "window-raise": {
        "fr": r"\b(?:soulev\w*|élev\w*)\b",
        "hi": r"(?:(?:इसे|उसे|स्क्रीन|विंडो).{0,25}उठा|उठा.{0,25}(?:फोकस|ध्यान))",
        "ko": r"(?:(?:화면|창|그것).{0,20}올리|올리.{0,20}(?:포커스|초점))",
        "is": r"\b(?:hækk\w*|aukn\w*)\b",
        "zh_CN": r"(?:增加|提高|升高|抬起)",
        "pt": r"\b(?:aument\w*|elev\w*)\b",
    },
}

_SCIENTIFIC_WELL_BAD_TARGET = {
    "fr": r"\bbien\b",
    "hi": r"(?:कुआँ|कुएँ|कुओं)",
    "ko": r"우물",
    "zh_CN": r"(?<!矿)(?<!油)(?<!水)井(?!然|口)",
    "pt": r"\bbem\b",
    "es": r"\bbien\b",
    "sv": r"\bbra\b",
}


def _context_prose(text: str) -> str:
    """Hide hard API literals while retaining offsets for proximity tests."""
    return _CONTEXT_HARD_PROTECT_RE.sub(
        lambda match: " " * len(match.group(0)), str(text)
    )


def _target_has(family: str, language: str, target: str) -> bool:
    pattern = _SEMANTIC_BAD_TARGETS.get(family, {}).get(language)
    return bool(pattern and re.search(pattern, target, re.IGNORECASE))


def _statistical_power_source(source: str) -> bool:
    text = str(source)
    return bool(
        re.search(
            r"(?i)\b(?:power analysis|statistical power|power\s+"
            r"(?:estimate|curve|figure|model|design)|"
            r"(?:reported|estimated|lower|higher|raise[sd]?|increase[sd]?|"
            r"decrease[sd]?|overstate[sd]?|understate[sd]?)\s+power)\b",
            text,
        )
        or re.search(
            r"(?is)(?:\bpower\b.{0,80}\b(?:sample size|replicates?|detection|"
            r"AUROC)\b|\b(?:sample size|replicates?|detection|AUROC)\b"
            r".{0,80}\bpower\b)",
            text,
        )
    )


def _raise_sense_counts(source: str) -> tuple[int, int, int]:
    """Return exception, quantitative-increase and window-raise counts."""
    text = str(source)
    exception = quantitative = window = 0
    cue = re.compile(
        r"(?i)(?:[A-Za-z_]\w*(?:Error|Exception|Cancelled)|error|exception|"
        r"invalid|unsupported|unrecognised|unrecognized|unknown|missing|"
        r"mismatch|failure|strict mode)"
    )
    quantitative_after = re.compile(
        r"^\s+(?:the\s+)?(?:reported\s+)?(?:power|threshold|value|count|"
        r"rate|score|number|limit|amount|share|mean|floor|ceiling|contrast|"
        r"tolerance|failure\s+tolerance)\b|"
        r"^.{0,70}\b(?:by|to)\s+[-+]?\d",
        re.IGNORECASE,
    )
    for match in re.finditer(
        r"(?i)\b(?:raise|raises|raised|raising|re-raises?)\b", text
    ):
        before = text[max(0, match.start() - 120):match.start()]
        after = text[match.end():match.end() + 120]
        context = before + match.group(0) + after
        if re.search(r"(?i)\braise or focus\b|\braise\b.{0,50}\b(?:window|screen)\b", context):
            window += 1
        elif quantitative_after.search(after) or re.search(
            r"(?i)(?:power|threshold|value|count|rate|score|number|limit|"
            r"amount|share|mean|floor|ceiling|contrast|tolerance)\b.{0,45}$",
            before,
        ):
            quantitative += 1
        elif (
            cue.search(context)
            or re.search(
                r"(?i)\b(?:never|does not|do not|did not|cannot|can't)\s+"
                r"(?:ever\s+)?(?:raise|raises|raised)\b|"
                r"\brais(?:e|es|ed|ing)\s+(?:if|when|on)\b|"
                r"\b(?:without|instead of|rather than)\s+raising\b",
                context,
            )
        ):
            exception += 1
    # A translated quantitative word may legitimately come from source prose
    # other than ``raise``. Count those source occurrences as allowances too.
    quantitative += len(re.findall(
        r"(?i)\b(?:increase\w*|higher|growth|augmentation|amplif\w*)\b",
        text,
    ))
    return exception, quantitative, window


def _mapping_key_bad_target(language: str, target: str) -> bool:
    if _target_has("mapping-key", language, target):
        return True
    proximity = {
        "hi": (r"बटन", r"(?:मैपिंग|कॉन्फ़िग|कॉन्फिग|शब्दकोश|सेटिंग)"),
        "ko": (r"버튼", r"(?:매핑|딕셔너리|사전|설정|구성)"),
        "is": (r"takk\w*", r"(?:vörpun|orðabók|stilling)"),
        "zh_CN": (r"按钮", r"(?:映射|字典|设置|配置|记录|JSON)"),
    }.get(language)
    if not proximity:
        return False
    key_word, mapping_cue = proximity
    return bool(re.search(
        rf"(?is)(?:{key_word}.{{0,60}}{mapping_cue}|"
        rf"{mapping_cue}.{{0,60}}{key_word})",
        target,
    ))


def _semantic_false_friends(
    source: str, value: str, language: str,
) -> tuple[str, ...]:
    """Return reviewed source/target sense errors that must never ship.

    These predicates are deliberately source-conditioned: the target word is
    rejected only when the canonical English block uses the scientific or
    software sense in question.  This makes the checks suitable both for
    generation and for an independent catalog audit without outlawing an
    otherwise ordinary target-language word globally.
    """
    source_raw = str(source)
    target_raw = str(value)
    source_text = _context_prose(source_raw)
    target_text = _context_prose(target_raw)
    failures: list[str] = []

    if (
        language == "zh_CN"
        and _ZH_WEB_ARTIFACT_RE.search(target_text)
        and not _ZH_WEB_SOURCE_ESCAPE_RE.search(source_text)
    ):
        failures.append("web-corpus-contamination")
    if re.search(_SOFTWARE_QUEUE_SOURCE, source_text, re.I):
        queue_bad = {
            "zh_CN": r"(?:尾巴|尾部)",
            "ko": r"(?:꼬리|꼬리가|꼬리를|꼬리는|꼬리로|꼬리에)",
        }.get(language)
        if queue_bad and re.search(queue_bad, target_text, re.I):
            failures.append("software-queue-as-tail")
    if re.search(_IMAGING_FIELD_SOURCE, source_text, re.I):
        field_bad = {
            "zh_CN": r"(?:田野|田地|领域|領域)",
            "ko": r"(?:밭|들판|논밭)",
        }.get(language)
        if field_bad and re.search(field_bad, target_text, re.I):
            failures.append("imaging-field-as-land-or-domain")
    if (
        language == "zh_CN"
        and re.search(_IMAGING_CHANNEL_SOURCE, source_text, re.I)
        and re.search(r"(?:频道|頻道)", target_text)
    ):
        failures.append("imaging-channel-as-broadcast-channel")
    if re.search(_SOFTWARE_CLASSIFIER_SOURCE, source_text, re.I):
        classifier_bad = {
            "zh_CN": r"(?:分类师|分類師|分类机|分類機)",
            "ko": r"(?:분류자|분류사)",
        }.get(language)
        if classifier_bad and re.search(classifier_bad, target_text, re.I):
            failures.append("software-classifier-as-person-or-machine")
    if (
        language == "zh_CN"
        and re.search(_HUMAN_READABLE_SOURCE, source_text, re.I)
        and re.search(r"人文", target_text)
    ):
        failures.append("human-readable-as-humanities")
    if (
        language == "zh_CN"
        and re.search(_MAPPING_KEY_SOURCE, source_text, re.I)
        and not re.search(_SECRET_KEY_SOURCE, source_text, re.I)
        and re.search(r"(?:密钥|密鑰)", target_text)
    ):
        failures.append("mapping-key-as-secret-key")

    if target_raw.count(">") > source_raw.count(">"):
        failures.append("surplus-angle-bracket")
    bad_well_word = _SCIENTIFIC_WELL_BAD_TARGET.get(language)
    if bad_well_word:
        total_well, noun_well = _english_well_sense_counts(source_text)
        allowed_adverbs = total_well - noun_well
        if (
            noun_well
            and len(re.findall(bad_well_word, target_text, re.I))
                > allowed_adverbs
        ):
            failures.append("scientific-well-as-adverb")

    has_gui_screen = _gui_screen_source(source_text)
    has_scientific_screen_cues = bool(
        re.search(r"\bscreens?\b", source_text, re.I)
        and re.search(
            r"\b(?:CRISPR|pooled|genome[- ]wide|gRNAs?|genes?|genotypes?|"
            r"hits?|phenotypes?|libraries?|widefield|assay|regression|"
            r"classification|screening)\b",
            source_text,
            re.I,
        )
    )
    # A paragraph that genuinely uses both senses needs human review; neither
    # one-word inverse gate can safely decide which target occurrence is wrong.
    if not (has_gui_screen and has_scientific_screen_cues):
        if has_gui_screen and _target_has("ui-screen", language, target_text):
            failures.append("gui-screen-as-scientific-screen")
        elif (
            re.search(_SCIENTIFIC_SCREEN_SOURCE, source_text, re.I)
            and _target_has("scientific-screen", language, target_text)
        ):
            failures.append("scientific-screen-as-ui-screen")

    source_target_families = (
        (_PIPELINE_SOURCE, "pipeline", "pipeline-as-pipe"),
        (_DATA_GATE_SOURCE, "gate", "data-gate-as-door"),
        (_DICTIONARY_SOURCE, "dictionary", "dictionary-as-dictation"),
        (_PLANE_SOURCE, "plane", "image-plane-as-aircraft"),
        (_IMAGE_TILE_SOURCE, "tile", "image-tile-as-roof-or-floor-tile"),
        (_SCIENTIFIC_PLATE_SOURCE, "plate", "scientific-plate-as-dish"),
        (_IMAGE_CROP_SOURCE, "crop", "image-crop-as-agriculture"),
        (_COMPUTE_RUN_SOURCE, "run", "compute-run-as-race"),
        (_COMPUTE_THREAD_SOURCE, "thread", "thread-corruption"),
    )
    for source_pattern, family, label in source_target_families:
        if (
            re.search(source_pattern, source_text, re.I)
            and _target_has(family, language, target_text)
        ):
            failures.append(label)

    # A terse API block such as ``One plate to process.`` may not contain a
    # second laboratory cue, even though spaCR's QueueItem/Batch context makes
    # the microplate sense unambiguous.  The food sense remains explicitly
    # excluded and accepted as a negative control.
    if (
        language in {"zh_CN", "ko"}
        and re.search(r"(?i)\bplates?\b", source_text)
        and not re.search(
            r"(?i)\b(?:food|meal|dish|dining|kitchen|serve|restaurant|ceramic)\b",
            source_text,
        )
        and _target_has("plate", language, target_text)
    ):
        failures.append("scientific-plate-as-dish")

    if (
        _statistical_power_source(source_text)
        and _target_has("power", language, target_text)
    ):
        failures.append("statistical-power-as-authority-or-electricity")
    if (
        re.search(_MAPPING_KEY_SOURCE, source_text, re.I)
        and _mapping_key_bad_target(language, target_text)
    ):
        failures.append("mapping-key-as-keyboard-key")

    exception_raises, quantitative_increases, window_raises = (
        _raise_sense_counts(source_text)
    )
    target_increases = len(re.findall(
        _SEMANTIC_BAD_TARGETS.get("exception-increase", {}).get(
            language, r"(?!x)x"
        ),
        target_text,
        re.I,
    ))
    if (
        exception_raises
        and target_increases > quantitative_increases
    ):
        failures.append("exception-raise-as-increase")
    if (
        window_raises
        and _target_has("window-raise", language, target_text)
    ):
        failures.append("raise-window-as-increase")
    if language != "pt":
        return tuple(dict.fromkeys(failures))
    if (
        re.search(r"\bcrops?\b", source_text, re.I)
        and re.search(
            r"\b(?:a|as|uma|essa|esta)\s+recortes?\b", target_text, re.I,
        )
    ):
        failures.append("crop-gender")
    return tuple(dict.fromkeys(failures))


def _translation_candidate_valid(
    source: str,
    value: str,
    language: str,
    *,
    force: bool = False,
) -> bool:
    """Apply every release gate to one runtime translation candidate."""
    return not _translation_rejection_reasons(
        source, value, language, force=force,
    )


def _translation_rejection_reasons(
    source: str,
    value: str,
    language: str,
    *,
    force: bool = False,
    raw_semantic_failure: bool = False,
    candidate_validator: Callable[[str, str, str], bool] | None = None,
) -> frozenset[str]:
    """Classify why a candidate cannot pass the release contract.

    The categories are also the retry state machine.  Only marker restoration
    and protected-syntax failures are mechanical enough for the final
    context-free fragment pass.  Semantic, target-script, exact-copy,
    degeneration, EOS and caller-contract failures must retain sentence or
    clause context and fail closed if those contextual retries do not rescue
    them.
    """
    source_text = str(source)
    raw_value = str(value)
    failures: set[str] = set()
    if raw_semantic_failure or _semantic_false_friends(
        source_text, raw_value, language,
    ):
        failures.add("semantic")

    candidate = _contextualize(raw_value, language, source_text)
    if language == "zh_CN":
        candidate = _simplify_chinese_prose(candidate)
    if not candidate.strip():
        failures.add("degenerate")
    if not _syntax_preserved_or_reviewed(source_text, candidate, language):
        failures.add("protected_syntax")
    if _looks_degenerate(source_text, candidate, language):
        failures.add("degenerate")
    if not _has_expected_script(
        source_text, candidate, language, force=force,
    ):
        failures.add("target_script")
    if _semantic_false_friends(source_text, candidate, language):
        failures.add("semantic")
    if (
        language == "zh_CN"
        and _has_traditional_chinese_prose(candidate)
    ):
        failures.add("target_script")
    # Context repair must never disguise an English fallback as a translation.
    # Several source-conditioned substitutions also match canonical English
    # (for example ``Default`` -> ``Padrão``); checking only the repaired value
    # admitted a mostly-English source after changing that one word.
    if (
        (raw_value == source_text or candidate == source_text)
        and (force or _looks_translatable(source_text))
    ):
        failures.add("exact")

    # A caller validator is an additional contract, not a replacement for the
    # shared gates.  Report it only for an otherwise viable candidate so a
    # structural or semantic failure cannot be mislabeled as caller-only.
    if (
        not failures
        and candidate_validator is not None
        and not candidate_validator(source_text, candidate, language)
    ):
        failures.add("caller_gate")
    return frozenset(failures)


SOURCE_CONTEXT_REGEX_REPLACEMENTS: Mapping[
    str, tuple[tuple[str, str, str], ...]
] = {
    "sv": (
        (r"\bcrops?\b", r"\bGrödläge\b", "Bildutsnittsläge"),
        (r"\bcrops?\b", r"\bGrödkälla\b", "Bildutsnittskälla"),
        (r"\bcrops?\b", r"beställgrödor", "bildutsnitt på begäran"),
        (r"\bcrops?\b", r"utbildningsgrödor", "träningsbildutsnitt"),
        (r"\bcrops?\b", r"utbildningsgröda", "träningsbildutsnitt"),
        (r"\bcrops?\b", r"grödgaller", "bildutsnittsgaller"),
        (r"\bcrops?\b", r"bildgrödor", "bildutsnitt"),
        (r"\bcrops?\b", r"grödans", "bildutsnittets"),
        (r"\bcrops?\b", r"grödor", "bildutsnitt"),
        (r"\bcrops?\b", r"gröda", "bildutsnitt"),
        (r"\bcrops?\b", r"skördens", "bildutsnittets"),
        (r"\bcrops?\b", r"skördarna", "bildutsnitten"),
        (r"\bcrops?\b", r"skördar", "bildutsnitt"),
        (r"\bcrops?\b", r"skörden", "bildutsnittet"),
        (r"\bcrops?\b", r"skörd", "bildutsnitt"),
        (r"\bcrops?\b", r"\bGrödans\b", "Bildutsnittets"),
        (r"\bcrops?\b", r"\bgrödorna\b", "bildutsnitten"),
        (r"\bcrops?\b", r"\bgrödor\b", "bildutsnitt"),
        (r"\bcrops?\b", r"\bgrödan\b", "bildutsnittet"),
        (r"\bcrops?\b", r"\bgröda\b", "bildutsnitt"),
        (_COMPUTE_RUN_SOURCE, r"\bloppet\b", "körningen"),
        (_COMPUTE_RUN_SOURCE, r"\blopp\b", "körning"),
        (_UI_SCREEN_SOURCE, r"\bscreeningar\b", "skärmar"),
        (_UI_SCREEN_SOURCE, r"\bscreening\b", "skärm"),
    ),
    "de": (
        (r"\bcrops?\b", r"\bPNG-Befruchtung\b", "PNG-Bildausschnitt"),
        (r"\bcrops?\b", r"\bObjekt-Befruchtungsart\b", "Objekt-Bildausschnittart"),
        (r"\bcrops?\b", r"\bBeispielobjektkulturen\b", "Beispiel-Bildausschnitte"),
        (r"\bcrops?\b", r"\bObjektkulturen\b", "Objekt-Bildausschnitte"),
        (r"\bcrops?\b", r"\bZellkulturen\b", "Zell-Bildausschnitte"),
        (r"\bcrops?\b", r"\bTrainingskulturen\b", "Trainingsbildausschnitte"),
        (r"\bcrops?\b", r"\bTrainingskultur\b", "Trainingsbildausschnitt"),
        (r"\bcrops?\b", r"\bBeispielkulturen\b", "Beispiel-Bildausschnitte"),
        (r"\bcrops?\b", r"\bRohkulturen\b", "rohen Bildausschnitten"),
        (r"\bcrops?\b", r"\bjeder Quadratkultur\b", "jedes quadratischen Bildausschnitts"),
        (r"\bcrops?\b", r"\bBildkulturen\b", "Bildausschnitte"),
        (r"\bcrops?\b", r"\bFreilandkulturen\b", "Bildausschnitte öffnen"),
        (r"\bcrops?\b", r"\bWiederauffrischungskulturen\b", "Bildausschnitte aktualisieren"),
        (r"\bcrops?\b", r"\bKulturquelle\b", "Bildausschnittquelle"),
        (r"\bcrops?\b", r"\bErntepfade\b", "Pfade der Bildausschnitte"),
        (r"\bcrops?\b", r"\bErntesatz\b", "Bildausschnittsatz"),
        (r"\bcrops?\b", r"\bErntegröße\b", "Bildausschnittgröße"),
        (r"\bcrops?\b", r"\bErntegitter\b", "Bildausschnittgitter"),
        (r"\bcrops?\b", r"\bErntemodus\b", "Bildausschnittmodus"),
        (r"\bcrops?\b", r"\bErnteeinstellungen\b", "Bildausschnitteinstellungen"),
        (r"\bcrops?\b", r"\bKulturen\b", "Bildausschnitte"),
        (r"\bcrops?\b", r"\bKultur\b", "Bildausschnitt"),
        (r"\bcrops?\b", r"\bErnten\b", "Bildausschnitte"),
        (r"\bcrops?\b", r"\bErnte\b", "Bildausschnitt"),
        (_COMPUTE_RUN_SOURCE, r"\bRennen\b", "Ausführung"),
        (_SCIENTIFIC_HIT_SOURCE, r"\bLeistungsschlagrate\b", "Power-Trefferanteil"),
        (_SCIENTIFIC_HIT_SOURCE, r"\bSchlagrate\b", "Trefferanteil"),
        (_UI_SCREEN_SOURCE, r"\bScreenings\b", "Bildschirme"),
        (_UI_SCREEN_SOURCE, r"\bScreening\b", "Bildschirm"),
        (r"\bgates?\b", r"\bTore\b", "Gates"),
        (r"\bgates?\b", r"\bTor\b", "Gate"),
        (r"\bgates?\b", r"\bFormtor\b", "Form-Gate"),
        (r"\bwells?\b", r"\bBrunnen\b", "Wells"),
    ),
    "es": (
        (r"\bnot\b", r"(?i)\bnot\b", "no"),
        (r"\bdoes\b", r"(?i)\bdoes\b", "sí"),
        (r"\bpre-write\b", r"(?i)\bpre-write\b", "previa a la escritura"),
        (r"\btemp-then\b", r"(?i)\btemp-then\b", "temporal y luego"),
        (r"\bwere worth\b", r"(?i)\bwere\b(?=\s*\*?\s*vale la pena)", "sí"),
        (r"\bsummary\b", r"(?i)\bsummary\b", "resumen"),
        (r"\bboth\b", r"(?i)\bboth\b", "ambos"),
        (r"\band\b", r"(?i)\band\b", "y"),
        (r"\bevery\b", r"(?i)\bevery\b", "cada"),
        (r"\bresult\b", r"(?i)\bresult\b", "resultado"),
        (r"\bbusy\b", r"(?i)\bbusy\b", "ocupada"),
        (r"\bonly\b", r"(?i)\bonly\b", "solo"),
        (r"\bbound methods?\b", r"(?i)\bbound method\b", "método enlazado"),
        (r"\bsame sections?\b", r"(?i)\bsame\b", "mismas"),
        (r"\bunset\b", r"(?i)\bunset\b", "sin establecer"),
        (r"\bcaller\b", r"(?i)\bcaller\b", "código llamador"),
        (r"\bstate\b", r"(?i)\bstate\b", "estado"),
        (r"\bevent\b", r"(?i)\bevent\b", "evento"),
        (r"\babsent\b", r"(?i)\babsent\b", "ausente"),
        (r"\bbecause\b", r"(?i)\bbecause\b", "porque"),
        (r"\bdefaults?\b", r"(?i)\bdefaults\b", "usa de forma predeterminada"),
        (r"\bproducing\b", r"(?i)\bproducing\b", "productor"),
        (r"\bunits?\b", r"(?i)\bunits\b", "Unidades"),
        (r"\bno-op\b", r"(?i)\bno-op\b", "sin efecto"),
        (
            r"\binvaded\s*/\s*inside\b",
            r"(?i)\binvaded\s*/\s*inside\b",
            "invadida / dentro",
        ),
        (r"\bwrite stack\b", r"(?i)\bwrite stack\b", "Escribir pila"),
        (r"\brun order\b", r"(?i)\brun order\b", "orden de ejecución"),
        (
            r"\bclear RAM\b.*\bcheck disk space\b",
            r"(?i)\bclear RAM\b", "liberar RAM",
        ),
        (
            r"\bclear RAM\b.*\bcheck disk space\b",
            r"(?i)\bclear VRAM\b", "liberar VRAM",
        ),
        (
            r"\bclear RAM\b.*\bcheck disk space\b",
            r"(?i)\bclear CPU\b", "liberar CPU",
        ),
        (
            r"\bclear RAM\b.*\bcheck disk space\b",
            r"(?i)\bcheck disk space\b", "comprobar espacio en disco",
        ),
        (
            r"\bEssentials\b.*\bAll\b",
            r"(?i)\ban Essentials / All switch\b",
            "un selector Esenciales / Todo",
        ),
        (
            r"\bPrepare\s*/\s*Run\s*/\s*Review\b",
            r"Prepare\s*/\s*Run\s*/\s*Review",
            "Preparar / Ejecutar / Revisar",
        ),
        (
            r"``False``.*\braises\b",
            r"(?i)se eleva\s+(``False``)",
            r"\1 genera una excepción",
        ),
        (
            r"\bfit that raises\b",
            r"(?i)ajuste que aumenta",
            "ajuste que genera una excepción",
        ),
        (
            r"\bthis fit crashed\b",
            r"(?i)\bthis fit crashed\b",
            "este ajuste falló",
        ),
        (
            r"\bother spelling\b",
            r"(?i)\bthe other spelling\b",
            "la otra ortografía",
        ),
        (
            r"\bKeyset paging\b",
            r"(?i)paginación de conjuntos de teclas",
            "paginación por clave",
        ),
        (
            r"\bcrops?\b",
            r"qué objeto cultiva las cargas del anotador",
            "qué recortes de objetos se cargan en el anotador",
        ),
        (r"\bcrops?\b", r"\bcosechas\b", "recortes"),
        (r"\bcrops?\b", r"\bcosecha\b", "recorte"),
        (r"\bcrops?\b", r"\bcultivos\b", "recortes"),
        (r"\bcrops?\b", r"\bcultivo\b", "recorte"),
        (r"\bcrops?\b", r"\bculturas\b", "recortes"),
        (r"\bcrops?\b", r"\bcultura\b", "recorte"),
        (_COMPUTE_RUN_SOURCE, r"\bcarreras\b", "ejecuciones"),
        (_COMPUTE_RUN_SOURCE, r"\bcarrera\b", "ejecución"),
        (_COMPUTE_RUN_SOURCE, r"\brecorridos\b", "ejecuciones"),
        (_COMPUTE_RUN_SOURCE, r"\brecorrido\b", "ejecución"),
        (_SCIENTIFIC_HIT_SOURCE, r"\btasa de golpeo\b", "tasa de aciertos"),
        (_SCIENTIFIC_HIT_SOURCE, r"\bgolpes\b", "aciertos"),
        (_SCIENTIFIC_HIT_SOURCE, r"\bgolpe\b", "acierto"),
        (_UI_SCREEN_SOURCE, r"\bcribados\b", "pantallas"),
        (_UI_SCREEN_SOURCE, r"\bcribado\b", "pantalla"),
        (r"\bgates?\b", r"\bpuertas\b", "compuertas"),
        (r"\bgates?\b", r"\bpuerta\b", "compuerta"),
        (
            r"\bresum(?:e|es|ed|ing)?\b",
            r"\bun curr[ií]culum vitae\b",
            "una reanudación",
        ),
        (
            r"\bresum(?:e|es|ed|ing)?\b",
            r"\bcurr[ií]culum vitae\b",
            "reanudación",
        ),
        (r"\bidempotent\b", r"\bidempotent\b", "idempotente"),
        (r"\bthreads?\b", r"\bregla de discusión\b", "regla de ejecución en hilos"),
        (r"\bthreads?\b", r"\bregla de roscado\b", "regla de ejecución en hilos"),
        (r"\bthreads?\b", r"\bhilo de discusión\b", "hilo"),
        (r"\bthreads?\b", r"\bGUI hilo\b", "hilo de GUI"),
        (r"\bthreads?\b", r"\bGUI-\s*Affine\b", "con afinidad con GUI"),
        (r"\bthreads?\b", r"\bGUI-\s*widget de hilo\b", "widget del hilo de GUI"),
        (r"\bbound methods?\b", r"\bMÉTODO DE BUENA\b", "MÉTODO ENLAZADO"),
        (r"\bbound methods?\b", r"\bmétodo encuadernado\b", "método enlazado"),
        (r"\bbound methods?\b", r"\bmétodo consolidado\b", "método enlazado"),
        (r"\bbound methods?\b", r"\bencuadernado\b", "enlazado"),
        (
            r"\bbound methods?\b",
            r"\bmétodo \*encuadernado\*",
            "método *enlazado*",
        ),
        (r"\bwidget children\b", r"\bniños widget\b", "widgets secundarios"),
        (
            r"\btick list\b",
            r"\blista de garrapatas\b",
            "lista de valores seleccionables",
        ),
        (r"\btables?\b", r"\bEl tabla\b", "La tabla"),
        (r"\borganelles?\b", r"\borganele\b", "orgánulo"),
        (r"\bthreads?\b", r"\bla hilo de GUI\b", "el hilo de GUI"),
        (r"\bthreads?\b", r"\bGUI-[Hh]ilo\b", "hilo de GUI"),
        (r"\bthreads?\b", r"\bGUI hilo\b", "hilo de GUI"),
        (r"\bthreads?\b", r"\bGUI Hilo\b", "hilo de GUI"),
        (r"\bpickers?\b", r"\brecolector\b", "selector"),
        (r"\btiles?\b", r"\bfichas\b", "teselas"),
        (r"\btiles?\b", r"\bficha\b", "tesela"),
        (r"\bspot plate\b", r"\bplaca de mancha\b", "placa de puntos"),
        (r"\bnon-hits?\b", r"\bno-hits\b", "genes sin efecto"),
        (r"\bfit(?:s|ting)?\b", r"\bmontaje\b", "ajuste"),
        (r"\btooltips?\b", r"\btooltip\b", "descripción emergente"),
        (r"\bpopups?\b", r"\bpopup\b", "ventana emergente"),
        (
            r"\bchild(?:ren)?\b",
            r"\bpadre de cada niño\b",
            "padre de cada nodo hijo",
        ),
        (
            r"\bchild(?:ren)?\b",
            r"\bcampo de ese niño\b",
            "campo de ese nodo hijo",
        ),
        (
            r"\bchunks?|chunked\b",
            r"\bAlmacenamiento cortado\b",
            "Almacenamiento por bloques",
        ),
        (r"\bchunks?|chunked\b", r"\btrozos\b", "bloques"),
        (r"\bchunks?|chunked\b", r"\btrozo\b", "bloque"),
        (
            r"\bregularized horseshoe\b.*\bprior\b|\bprior\b.*\bhorseshoe\b",
            r"\bherradura regularizada anterior\b",
            "distribución a priori de herradura regularizada",
        ),
        (r"\bfloat\b", r"\bflota\b", "número de punto flotante"),
        (r"\btables?\b", r"\bese tabla\b", "esa tabla"),
        (
            r"\bsame\b.*\bmeasurements?\b|\bmeasurements?\b.*\bsame\b",
            r"\*same\*",
            "*mismas*",
        ),
        (r"\bsame\b.*\bsets?\b|\bsets?\b.*\bsame\b", r"\*same\*", "*mismo*"),
        (
            r"\bsame\b.*\bgenerator\b|\bgenerator\b.*\bsame\b",
            r"\*same\*",
            "*mismo*",
        ),
        (r"\bsame\b.*\bcells?\b|\bcells?\b.*\bsame\b", r"\*same\*", "*mismas*"),
        (r"\bsame\b.*\bobjects?\b|\bobjects?\b.*\bsame\b", r"\*same\*", "*mismo*"),
        (
            r"\bsame\b.*\blabel images?\b|\blabel images?\b.*\bsame\b",
            r"\*same\*",
            "*mismas*",
        ),
        (r"\bsame\b.*\bcolumns?\b|\bcolumns?\b.*\bsame\b", r"\*same\*", "*misma*"),
        (r"\bsame column\b", r"\bla columna SAME\b", "la misma columna"),
        (r"\bsame two classes\b", r"\bel SAME dos clases\b", "las mismas dos clases"),
        (r"\bsame setting\b", r"\bla configuración de SAME\b", "la misma configuración"),
        (
            r"\bsettings hash\b",
            r"\bsettings hash\b",
            "hash de configuración",
        ),
        (
            r"\bsettings dict\b",
            r"\bsettings dict\b",
            "diccionario de configuración",
        ),
        (
            r"\bsettings dict\b",
            r"\bconfiguración dict\b",
            "diccionario de configuración",
        ),
        (r"\bsettings\b", r"\bSettings-diff\b", "Comparación de configuraciones"),
        (r"\bsettings\b", r"\bLive Settings\b", "Configuración en vivo"),
        (r"\btables?\b", r"\bcuadros\b", "tablas"),
        (r"\btables?\b", r"\bcuadro\b", "tabla"),
        (r"\bstrings?\b", r"\bcuerdas\b", "cadenas"),
        (r"\bstrings?\b", r"\bcuerda\b", "cadena"),
        (r"\bplanes?\b", r"\baviones\b", "planos"),
        (r"\bplanes?\b", r"\bavión\b", "plano"),
        (r"\bdisagreements?\b", r"\bdisconformidades\b", "desacuerdos"),
        (r"\bdisagreements?\b", r"\bdisconformidad\b", "desacuerdo"),
        (r"\bdatabase\b", r"\bbase de información\b", "base de datos"),
        (r"\bdensity\b", r"\bDensity\b", "Densidad"),
        (r"\btoggles?\b", r"\btoggles\b", "conmutadores"),
        (r"\btoggles?\b", r"\btoggle\b", "conmutador"),
        (r"\btilde expansion\b", r"\btilde expansion\b", "expansión de tilde"),
        (r"\bpre-flight\b", r"\bpre-vuelo\b", "validación previa"),
    ),
    "zh_CN": (
        (r"\bcrops?\b", r"收获", "裁剪"),
        (r"\bcrops?\b", r"种植", "图像裁剪"),
        (r"\bcrops?\b", r"作物", "图像裁剪"),
        (_COMPUTE_RUN_SOURCE, r"赛跑", "运行"),
        (_UI_SCREEN_SOURCE, r"筛选", "界面"),
        (r"\bgates?\b", r"门(?!控)", "门控"),
        (r"\bwells?\b", r"井", "孔"),
    ),
    "pt": (
        (
            r"(?is)\b(?:rather than|instead of|without)\s+raising\b",
            r"\b(?:em|ao) vez de aumentar\b",
            "em vez de gerar um erro",
        ),
        (_DICTIONARY_SOURCE, r"\b[Dd]itad(?:o|os|a|as)\b", "dicionário"),
        (r"(?is)\bwell matches\b", r"\bcombina bem com\b", "o poço corresponde a"),
        (r"\bcrops?\b", r"\bcolheitas\b", "recortes"),
        (r"\bcrops?\b", r"\bcolheita\b", "recorte"),
        (r"\bcrops?\b", r"\bculturas\b", "recortes"),
        (r"\bcrops?\b", r"\bcultura\b", "recorte"),
        (_COMPUTE_RUN_SOURCE, r"\bcorridas\b", "execuções"),
        (_COMPUTE_RUN_SOURCE, r"\bcorrida\b", "execução"),
        (r"\bpower\b", r"\b[Pp]oder(?:es)?\b", "potência"),
        (_UI_SCREEN_SOURCE, r"\btriagens\b", "telas"),
        (_UI_SCREEN_SOURCE, r"\btriagem\b", "tela"),
        (r"\b(?:gates?|gating|gated)\b", r"\b[Pp]ortões\b", "gates"),
        (r"\b(?:gates?|gating|gated)\b", r"\b[Pp]ortão\b", "gate"),
        (r"\b(?:gates?|gating|gated)\b", r"\b[Pp]ortas\b", "gates"),
        (r"\b(?:gates?|gating|gated)\b", r"\b[Pp]orta\b", "gate"),
        (r"\bpipelines?\b", r"\b[Gg]asodutos\b", "fluxos de trabalho"),
        (r"\bpipelines?\b", r"\b[Gg]asoduto\b", "fluxo de trabalho"),
        (r"\bpipelines?\b", r"\b[Oo]leodutos\b", "fluxos de trabalho"),
        (r"\bpipelines?\b", r"\b[Oo]leoduto\b", "fluxo de trabalho"),
        (r"\bpipelines?\b", r"\b[Tt]ubulações\b", "fluxos de trabalho"),
        (r"\bpipelines?\b", r"\b[Tt]ubulação\b", "fluxo de trabalho"),
        (r"\bplanes?\b", r"\b[Aa]viões\b", "planos"),
        (r"\bplanes?\b", r"\b[Aa]vião\b", "plano"),
        (r"\bplates?\b", r"\b[Pp]ratos\b", "placas"),
        (r"\bplates?\b", r"\b[Pp]rato\b", "placa"),
        (r"\btiles?\b", r"\b[Aa]zulejos\b", "teselas"),
        (r"\btiles?\b", r"\b[Aa]zulejo\b", "tesela"),
        (r"\btiles?\b", r"\b[Tt]elhas\b", "teselas"),
        (r"\btiles?\b", r"\b[Tt]elha\b", "tesela"),
        (r"\bcrops?\b", r"\b[Aa]s recortes\b", "os recortes"),
        (r"\bcrops?\b", r"\b[Aa] recorte\b", "o recorte"),
        (r"\bcrops?\b", r"\b[Uu]ma recorte\b", "um recorte"),
        (r"\bcrops?\b", r"\b[Ee]ssa recorte\b", "esse recorte"),
        (r"\bcrops?\b", r"\b[Ee]sta recorte\b", "este recorte"),
        (
            r"\braise or focus\b",
            r"\baumentar ou foc(?:ar|á)-?la\b",
            "trazê-la para a frente ou dar-lhe foco",
        ),
        # OPUS-Portuguese commonly keeps these programming words in English.
        # They are ordinary explanatory prose here, not identifiers: hard
        # literals have already been hidden by ``_contextualize``.  Keeping
        # the replacements source-conditioned prevents a coincidental target
        # word from being rewritten when the English API block did not use it.
        (r"\bstrings?\b", r"\b[Ss]trings\b", "cadeias de caracteres"),
        (r"\bstrings?\b", r"\b[Ss]tring\b", "cadeia de caracteres"),
        (r"\bloops?\b", r"\bevent loop\b", "ciclo de eventos"),
        (r"\bloops?\b", r"\b[Ll]oops\b", "ciclos"),
        (r"\bloops?\b", r"\b[Ll]oop\b", "ciclo"),
        (r"\bdefaults?\b", r"\bDefaults to\b", "Usa por padrão"),
        (r"\bdefaults?\b", r"\bdefaults to\b", "usa por padrão"),
        (r"\bdefaults?\b", r"\b[Dd]efaults\b", "valores padrão"),
        (r"\bdefaults?\b", r"\b[Dd]efault\b", "padrão"),
        (r"\bidempotent\b", r"\b[Ii]dempotent\b", "idempotente"),
        (r"\bfilename\b", r"\bfilename\b", "nome do arquivo"),
        (r"\bomitted\b", r"\b[Oo]mitted\b", "omitido"),
        (r"\bwithout\b", r"\bwithout\b", "sem"),
        (r"\bneither\b", r"\bneither\b", "nem"),
        (r"\beither\b", r"\bEITHER\s+", ""),
        (r"\beither\b", r"\beither\b", "qualquer uma das opções"),
        (r"\brather\b", r"\brather than\b", "em vez de"),
        (r"\brather\b", r"\brather\b", "em vez disso"),
        (r"\bthrough\b", r"\bthrough\b", "por meio de"),
        (r"\bcaller\b", r"\bcaller's\b", "do chamador"),
        (r"\bcaller\b", r"\bcaller\b", "chamador"),
        (r"\bbound\b", r"\bbound\b", "vinculado"),
        (r"\btoggle\b", r"\btoggle\b", "alternância"),
        (r"\bunset\b", r"\bunset\b", "não definido"),
        (r"\bmatch\b", r"\bmatch\b", "correspondência"),
        (r"\bcalls\b", r"\bcalls\b", "chama"),
        (r"\bwhich\b", r"\bwhich\b", "que"),
        (r"\bdoes\b", r"\bdoes\b", "faz"),
        (r"\bwhere\b", r"\bwhere\b", "onde"),
        (r"\bsee\b", r"\bsee\b", "consulte"),
        (r"\bleft\b", r"\bLeft\b", "Deixado"),
        (r"\bleft\b", r"\bleft\b", "deixado"),
        (r"\bevery\b", r"\b[Ee]very\b", "cada"),
        (r"\bonly\b", r"\bonly\b", "somente"),
        (r"\bnot\b", r"\b[Nn]ot\b", "não"),
        (r"\bsame\b", r"\bSAME\b", "MESMO"),
        (r"\bsame\b", r"\bsame\b", "mesmo"),
        # Restore the standard Portuguese technical loanword after an earlier
        # broad cleanup expanded it to ungrammatical forms such as ``O linhas
        # de execução`` and even model artifacts such as ``thought worker``.
        # The API residue gate explicitly allows this reviewed PT loanword;
        # Python identifiers and inline code remain hidden from these rules.
        (r"\bthreads?\b", r"\blinhas de execução\b", "threads"),
        (r"\bthreads?\b", r"\blinha de execução\b", "thread"),
        (r"\bthreads?\b", r"\bthoughts\b", "threads"),
        (r"\bthreads?\b", r"\bthought\b", "thread"),
        (r"\bthreads?\b", r"\bthrows\b", "threads"),
        (r"\bthreads?\b", r"\bthline\b", "thread"),
        (r"\bthreads?\b", r"\bthread-safe\b", "seguro para threads"),
        (r"\bthreads?\b", r"\bthread-safety\b", "segurança entre threads"),
        (
            r"\bGUI[- ]thread\b",
            r"\b(?:na|da|do|no|o|a) threads da GUI\b",
            "na thread da GUI",
        ),
        (r"\bGUI[- ]thread\b", r"\bthreads? GUI\b", "thread da GUI"),
        (r"\bworker thread\b", r"\bthreads? worker\b", "thread de trabalho"),
        (r"\bworker thread\b", r"\bthread WORKER\b", "thread de trabalho"),
        (r"\bworker thread\b", r"\bthrows? worker\b", "thread de trabalho"),
        (r"\bthreads?\b", r"\bdesse throw\b", "da thread que a chamou"),
        (r"\bGUI[- ]thread\b", r"\bO threads da GUI\b", "A thread da GUI"),
        # ``well`` is both a plate position and an adverb.  An older blanket
        # ``bem -> poço`` rule damaged the latter sense.  Repair only reviewed
        # adverbial Portuguese phrases, conditioned on an English block that
        # actually contains ``well``.
        (r"\bwell\b", r"\bquão poço\b", "quão bem"),
        (r"\bwell\b", r"\bpoço como\b", "bem como"),
        (r"\bwell\b", r"\bpoço[- ]definid([oa]s?)\b", r"bem definid\1"),
        (r"\bwell\b", r"\bpoço[- ]separad([oa]s?)\b", r"bem separad\1"),
        (r"\bwell\b", r"\bpoço[- ]comportad([oa]s?)\b", r"bem comportad\1"),
        (r"\bwell\b", r"\bpoço (abaixo|acima|além|dentro|fora)\b", r"bem \1"),
        (r"\bwell\b", r"\bfunciona(?:m)? tão poço\b", "funciona tão bem"),
        (r"\bwell\b", r"\btão poço\b", "tão bem"),
        (r"\bwell\b", r"\bpoço menos\b", "bem menos"),
        (
            r"\bwell\b",
            r"\bpoço (conhecid|calibrad|condicionad|especificad|suportad|"
            r"misturad|correspondid|alinhad|representad|documentad|testad|"
            r"adequad|estabelecid|compreendid|controlad|resolvid|fundad)"
            r"([oa]s?)\b",
            r"bem \1\2",
        ),
        # Translate English compounds before their individual words. OPUS
        # often leaves these software/statistics expressions inside an
        # otherwise Portuguese sentence. All rules are source-conditioned,
        # and code/quoted literals have already been masked, so an identifier
        # such as ``run_id`` or an option value such as ``'off'`` is untouched.
        (r"\bone-vs-rest\b", r"\bone-vs-rest\b", "um contra os demais"),
        (r"\bone-hot\b", r"\bone-hot\b", "indicador binário"),
        (r"\ball-zero\b", r"\ball-zero\b", "somente zeros"),
        (
            r"\bout-of-distribution\b",
            r"\bout-of-distribution\b",
            "fora da distribuição",
        ),
        (
            r"\bcontinue-on-error\b",
            r"\bcontinue-on-err(?:or|o)\b",
            "continuar em caso de erro",
        ),
        (r"\bheld-out\b", r"\b(?:held|holded)-out\b", "reservado"),
        (r"\bhold-out\b", r"\bhold-out\b", "conjunto reservado"),
        (r"\bin-place\b", r"\bin-place\b", "no mesmo local"),
        (r"\bopt-in\b", r"\bopt-in\b", "opcional"),
        (r"\bdrop-in\b", r"\bdrop-in\b", "substituição direta"),
        (r"\bin-app\b", r"\bin-app\b", "no aplicativo"),
        (r"\boff-thread\b", r"\boff-thread\b", "fora da thread"),
        (r"\bon/off\b", r"\bon/off\b", "ativado/desativado"),
        (r"\brun-to-run\b", r"\brun-to-run\b", "entre execuções"),
        (r"\bper-run\b", r"\bper-run\b", "por execução"),
        (r"\brun[- ]ids?\b", r"\brun[- ]ids?\b", "id da execução"),
        (r"\bt-first\b", r"\bt-first\b", "com t primeiro"),
        (r"\bz-first\b", r"\bz-first\b", "com z primeiro"),
        (r"\bwidth-first\b", r"\bwidth-first\b", "com largura primeiro"),
        (r"\b\d+-by-\d+\b", r"\b(\d+)-by-(\d+)\b", r"\1 por \2"),
        (r"\bstand-in\b", r"\bstand-in\b", "substituto"),
        (r"\blay-out\b", r"\bre-lays-out\b", "reorganiza"),
        (r"\blay-out\b", r"\blay-out\b", "organizar"),
        (r"\bwrite-back\b", r"\bwrite-back\b", "gravação de retorno"),
        (r"\bfit-on-load\b", r"\bfit-on-load\b", "ajuste ao carregar"),
        (
            r"\bbuffered-and-upscaled\b",
            r"\bbuffered-and-upscaled\b",
            "armazenado em buffer e ampliado",
        ),
        (r"\bslide-in\b", r"\bslide-in\b", "deslizante"),
        (r"\bcall-to-action\b", r"\bcall-to-action\b", "chamada para ação"),
        (r"\bin-memory\b", r"\bin-memory\b", "na memória"),
        (
            r"\bupdate-in-place\b",
            r"\bupdate-in-place\b",
            "atualização no mesmo local",
        ),
        (r"\bon the right\b", r"\bon the right\b", "à direita"),
        (r"\bzero out\b", r"\bzero out\b", "zerar"),
        (r"\bAND-ed\b", r"\bAND-ed\b", "combinadas por E"),
    ),
    "hi": (
        (r"\bcrops?\b", r"कटाई", "इमेज क्रॉप"),
        (r"\bcrops?\b", r"फसल", "इमेज क्रॉप"),
        (_COMPUTE_RUN_SOURCE, r"दौड़", "रन"),
        (_UI_SCREEN_SOURCE, r"स्क्रीनिंग", "स्क्रीन"),
        (r"\bgates?\b", r"दरवाजे", "गेट"),
        (r"\bgates?\b", r"दरवाजा", "गेट"),
        (r"\bgates?\b", r"द्वार", "गेट"),
        (r"\bwells?\b", r"कुआँ", "वेल"),
    ),
    "ko": (
        (r"\bcrops?\b", r"수확물?", "크롭"),
        (r"\bcrops?\b", r"농?작물", "크롭"),
        (_COMPUTE_RUN_SOURCE, r"달리기", "실행"),
        (_UI_SCREEN_SOURCE, r"스크리닝", "화면"),
        (r"\bgates?\b", r"문(?!\s*자)", "게이트"),
        (r"\bwells?\b", r"우물", "웰"),
    ),
    "is": (
        (r"\bcrops?\b", r"\buppsker(?:a|u|ur|unnar)?\b", "myndúrklippa"),
        (r"\bcrops?\b", r"\bræktun\b", "myndúrklippur"),
        (_COMPUTE_RUN_SOURCE, r"\bhlaupið\b", "keyrslan"),
        (_COMPUTE_RUN_SOURCE, r"\bhlaup\b", "keyrsla"),
        (_UI_SCREEN_SOURCE, r"\bskimanir\b", "skjáir"),
        (_UI_SCREEN_SOURCE, r"\bskimun\b", "skjár"),
    ),
    "fr": (
        (r"\bcrops?\b", r"\brécoltes\b", "vignettes"),
        (r"\bcrops?\b", r"\brécolte\b", "vignette"),
        (r"\bcrops?\b", r"\bcultures\b", "vignettes"),
        (r"\bcrops?\b", r"\bculture\b", "vignette"),
        (r"\bcrops?\b", r"\bmoissons\b", "vignettes"),
        (r"\bcrops?\b", r"\bmoisson\b", "vignette"),
        (_COMPUTE_RUN_SOURCE, r"\bpistes\b", "exécutions"),
        (_COMPUTE_RUN_SOURCE, r"\bpiste\b", "exécution"),
        (_COMPUTE_RUN_SOURCE, r"\bcourses\b", "exécutions"),
        (_COMPUTE_RUN_SOURCE, r"\bcourse\b", "exécution"),
        (_COMPUTE_RUN_SOURCE, r"\bparcours\b", "exécution"),
        (_UI_SCREEN_SOURCE, r"\b(à|sur) le criblage\b", r"\1 l’écran"),
        (_UI_SCREEN_SOURCE, r"\bde le criblage\b", "de l’écran"),
        (_UI_SCREEN_SOURCE, r"\ble criblage\b", "l’écran"),
        (_UI_SCREEN_SOURCE, r"\bcet criblage\b", "cet écran"),
        (_UI_SCREEN_SOURCE, r"\bcriblages\b", "écrans"),
        (_UI_SCREEN_SOURCE, r"\bcriblage\b", "écran"),
        (r"\bgates?\b", r"\bportes\b", "gates"),
        (r"\bgates?\b", r"\bporte\b", "gate"),
        (r"\bclusters?\b", r"\bgroupes\b", "clusters"),
        (r"\bclusters?\b", r"\bgroupe\b", "cluster"),
    ),
}

MANUAL_TRANSLATIONS: dict[str, dict[str, str]] = {
    (
        "Wells occupied by each entry of cell_types, one inner list per cell "
        "type in the same order, e.g. [['c2','c3'],['c4']]. Every identifier "
        "must start with 'c' (column) or 'r' (row); anything else is SILENTLY "
        "skipped and those wells get no host_cells label. An unlabelled well "
        "is not lost -- 'condition' joins whichever labels do exist -- so a "
        "typo here quietly changes what is being compared rather than "
        "raising. Default None."
    ): {
        "es": "Posiciones de la microplaca ocupadas por cada entrada de "
              "cell_types, una lista interna por tipo de célula en el mismo "
              "orden, p. ej. [['c2','c3'],['c4']]. Cada identificador debe "
              "comenzar por 'c' (columna) o 'r' (fila); cualquier otra cosa "
              "se omite SILENCIOSAMENTE y esas posiciones no reciben una "
              "etiqueta host_cells. Una posición sin etiqueta no se pierde "
              "-- 'condition' combina las etiquetas que existan --, por lo "
              "que un error tipográfico cambia silenciosamente qué se "
              "compara en vez de generar una excepción. Valor predeterminado: "
              "None.",
    },
    (
        "gRNA names treated as controls when the mixed-condition fraction is "
        "computed; every fraction is measured relative to these. A wrong or "
        "incomplete list shifts every fraction on the plate in the same "
        "direction rather than raising, so check it against the library. "
        "Default None."
    ): {
        "es": "Nombres de gRNA tratados como controles al calcular la "
              "fracción de la condición mixta; cada fracción se mide con "
              "respecto a ellos. Una lista incorrecta o incompleta desplaza "
              "todas las fracciones de la microplaca en la misma dirección "
              "en vez de generar una excepción, así que compárela con la "
              "biblioteca. Valor predeterminado: None.",
    },
    (
        "Train/test independence: 'cell', 'field', 'well' (default), or "
        "'plate'. Cell can place sibling crops from one well on both sides; "
        "field narrows but does not close that leak; well matches the usual "
        "experimental assignment unit; plate holds out a complete batch. "
        "Whole groups make the requested fraction approximate, so runs report "
        "held-out groups and cells. Legacy 'none'/'off' alias 'cell'. Crop "
        "identities come from spaCR's plate_well_field_object.png names; "
        "unverifiable grouped designs are refused rather than silently "
        "randomized."
    ): {
        "es": "Independencia entre entrenamiento y prueba: 'cell', 'field', "
              "'well' (valor predeterminado) o 'plate'. 'cell' puede colocar "
              "recortes hermanos de una misma posición de microplaca en "
              "ambos conjuntos; 'field' reduce esa fuga, pero no la elimina; "
              "'well' coincide con la unidad habitual de asignación "
              "experimental; 'plate' reserva un lote completo. Los grupos "
              "enteros hacen que la fracción solicitada sea aproximada, por "
              "lo que las ejecuciones informan de los grupos y las células "
              "reservados. Los valores heredados 'none'/'off' son alias de "
              "'cell'. Las identidades de los recortes proceden de los "
              "nombres plate_well_field_object.png de spaCR; los diseños "
              "agrupados que no se puedan verificar se rechazan en vez de "
              "aleatorizarse silenciosamente.",
    },
    (
        "Three-letter code choosing the RGB colours for cell, nucleus and "
        "pathogen outlines, in that order: 'rgb', 'bgr', 'gbr' or 'rbg'. The "
        "default 'gbr' draws cells green, nuclei blue and pathogens red. An "
        "unrecognised string SILENTLY falls back to 'rbg', so a typo changes "
        "your figure colours rather than raising. Change it when an outline "
        "clashes with a channel. Default 'gbr'."
    ): {
        "es": "Código de tres letras que elige los colores RGB de los "
              "contornos de célula, núcleo y patógeno, en ese orden: 'rgb', "
              "'bgr', 'gbr' o 'rbg'. El valor predeterminado 'gbr' dibuja "
              "las células en verde, los núcleos en azul y los patógenos en "
              "rojo. Una cadena no reconocida vuelve SILENCIOSAMENTE a "
              "'rbg', por lo que un error tipográfico cambia los colores de "
              "la figura en vez de generar una excepción. Cámbielo cuando "
              "un contorno se confunda con un canal. Valor predeterminado: "
              "'gbr'.",
    },
}

MANUAL_UI: dict[str, dict[str, str]] = {
    "Go to the screen this run belongs to.": {
        "fr": "Accédez à l’écran auquel appartient cette exécution.",
    },
    "One threshold per measurement": {
        "sv": "Ett tröskelvärde per mätvärde",
        "de": "Ein Schwellenwert pro Messgröße",
        "es": "Un umbral por medición",
        "zh_CN": "每项测量一个阈值",
        "pt": "Um limiar por medição",
        "hi": "प्रत्येक माप के लिए एक थ्रेशोल्ड",
        "ko": "측정값마다 하나의 임계값",
        "is": "Eitt þröskuldsgildi fyrir hverja mælingu",
        "fr": "Un seuil par mesure",
    },
    (
        "The simulator parameters this screen does not ask for, held at the "
        "values fitted to the real T. gondii screen. Change them by "
        "constructing a DesignSpec and calling set_spec()."
    ): {
        "sv": "Simulatorparametrarna som inte visas på den här skärmen behåller "
              "de värden som anpassats till den verkliga T. gondii-screeningen. "
              "Ändra dem genom att skapa en DesignSpec och anropa set_spec().",
        "de": "Simulatorparameter, die auf diesem Bildschirm nicht abgefragt "
              "werden, behalten die an den realen T.-gondii-Screen angepassten "
              "Werte. Ändern Sie sie, indem Sie eine DesignSpec erstellen und "
              "set_spec() aufrufen.",
        "es": "Los parámetros del simulador que no se solicitan en esta pantalla "
              "conservan los valores ajustados al cribado real de T. gondii. "
              "Para cambiarlos, cree un DesignSpec y llame a set_spec().",
        "zh_CN": "此界面未显示的模拟器参数将沿用根据真实弓形虫筛选实验拟合的值。"
                 "如需更改，请构造 DesignSpec 并调用 set_spec()。",
        "pt": "Os parâmetros do simulador não solicitados nesta tela mantêm os "
              "valores ajustados à triagem real de T. gondii. Para alterá-los, "
              "crie um DesignSpec e chame set_spec().",
        "hi": "इस स्क्रीन पर न दिखाए गए सिम्युलेटर पैरामीटर वास्तविक T. gondii "
              "स्क्रीनिंग के लिए फिट किए गए मानों पर बने रहते हैं। इन्हें बदलने के लिए "
              "DesignSpec बनाएँ और set_spec() को कॉल करें।",
        "ko": "이 화면에서 설정하지 않는 시뮬레이터 매개변수에는 실제 T. gondii "
              "스크리닝에 맞춰 적합한 값이 유지됩니다. 변경하려면 DesignSpec을 "
              "생성하고 set_spec()을 호출하세요.",
        "is": "Færibreytur hermisins sem þessi skjár biður ekki um halda gildunum "
              "sem voru aðlöguð að raunverulegri T. gondii-skimun. Breyttu þeim "
              "með því að búa til DesignSpec og kalla á set_spec().",
        "fr": "Les paramètres du simulateur qui ne figurent pas sur cet écran "
              "conservent les valeurs ajustées au criblage réel de T. gondii. "
              "Pour les modifier, construisez un DesignSpec et appelez "
              "set_spec().",
    },
    "Spatial phenotype analysis of CRISPR screens.": {
        "sv": "Spatial fenotypanalys av CRISPR-screeningar.",
        "de": "Räumliche Phänotypanalyse von CRISPR-Screens.",
        "es": "Análisis espacial de fenotipos en cribados CRISPR.",
        "zh_CN": "CRISPR 筛选的空间表型分析。",
        "pt": "Análise espacial de fenótipos em triagens CRISPR.",
        "hi": "CRISPR स्क्रीनिंग का स्थानिक फीनोटाइप विश्लेषण।",
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

_REVIEWED_RUNTIME_LOADING: set[str] = set()

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

# The four organelle slots expose their number in the generated setting label.
# Keep the slot while retaining the reviewed object name and CP/FT abbreviation;
# generic model output turned ``Organelle`` into unrelated loanwords in Korean.
for _slot in range(1, 5):
    for _source_suffix, _target_suffix in (("Cp prob", "CP"), ("Ft", "FT")):
        MANUAL_TRANSLATIONS[f"Organelle {_slot} — {_source_suffix}"] = {
            language: f"{name} {_slot} — {_target_suffix}"
            for language, name in _OBJECT_LABELS["Organelle"].items()
        }

MANUAL_TRANSLATIONS["Power hit rate"] = {
    "sv": "Effektens träfffrekvens",
    "de": "Trefferquote der Teststärke",
    "es": "Tasa de detección de potencia",
    "zh_CN": "功效检出率",
    "pt": "Taxa de detecção de potência",
    "hi": "सांख्यिकीय शक्ति की पहचान दर",
    "ko": "검정력 검출률",
    "is": "Greiningarhlutfall tölfræðiafls",
    "fr": "Taux de détection de puissance",
}


def _reviewed_translation(source: str, language: str) -> str | None:
    """Return exact reviewed prose without adding it to the static UI set."""
    static = (
        MANUAL_TRANSLATIONS.get(str(source), {}).get(language)
        or MANUAL_UI.get(str(source), {}).get(language)
    )
    if static is not None or language in _REVIEWED_RUNTIME_LOADING:
        return static
    return reviewed_runtime_translations(language).get(str(source))


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
    from spacr.qt.app import APPS, _SECTION_NOTE_LIBRARY
    from spacr.qt.screens.app_screen import (
        APP_INTROS,
        APP_TITLES,
        DEFAULT_INSTRUCTION,
    )

    # Several self-contained modules contribute defaults, tooltips, and a
    # module description only when their registered defaults module is first
    # imported.  Resolve every application before snapshotting the shared
    # tooltip tables.  Otherwise the generated source inventory depends on
    # import order: a clean generator omitted Barcode QC while a test process
    # that had already imported ``spacr.sequencing_qc`` gained one extra key.
    resolved_settings: dict[str, dict[str, object]] = {}
    for app_key, _name, _description, _section in APPS:
        try:
            resolved_settings[app_key] = resolve_default_settings(app_key)
        except Exception:
            continue

    raw_tooltips = get_tooltips()
    # ``spacr.runctx`` registers CLI/settings-file help when that module is
    # imported, but its own contract deliberately does not inject those keys
    # into any application's defaults. They therefore have no settings-panel
    # row to translate. Excluding them here keeps the runtime catalog source
    # inventory independent of whether an unrelated test imported runctx
    # first. ``random_seed`` is intentionally retained: it is a real setting
    # owned by many applications.
    for non_panel_key in ("on_error", "on_error_attempts", "on_error_backoff"):
        raw_tooltips.pop(non_panel_key, None)
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
    module_summaries = {
        str(key): str(description)
        for key, _name, description, _section in APPS
    }
    label_model = SettingsWidgets.__new__(SettingsWidgets)
    for app_key, _name, _description, _section in APPS:
        label_model.app_key = app_key
        setting_keys = resolved_settings.get(app_key, {})
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


@lru_cache(maxsize=None)
def reviewed_runtime_translations(language: str) -> dict[str, str]:
    """Return exact, source-bound runtime translations from review evidence.

    Review files are inputs to the ordinary candidate gates, not catalogs and
    not an audit allowlist.  Each record is bound to one current source table,
    key, source hash, and target language.  Source drift or a target that no
    longer passes the current syntax, semantic, script, and exact-copy gates
    is therefore a hard error.
    """
    directory = REVIEWED_RUNTIME_DIR / language
    if not directory.is_dir():
        return {}
    sources = canonical_sources()
    reviewed: dict[str, str] = {}
    expected_fields = {
        "table", "key", "source_sha256", "source", "translation",
    }
    _REVIEWED_RUNTIME_LOADING.add(language)
    try:
        for path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    f"invalid reviewed runtime evidence {path}"
                ) from exc
            if payload.get("schema") != 1 or payload.get("language") != language:
                raise ValueError(
                    f"invalid reviewed runtime evidence header: {path}"
                )
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValueError(
                    f"invalid reviewed runtime record list: {path}"
                )
            for record in records:
                if not isinstance(record, Mapping) or set(record) != expected_fields:
                    raise ValueError(f"invalid reviewed runtime record: {path}")
                table_name = str(record["table"])
                key = str(record["key"])
                table = sources.get(table_name)
                if isinstance(table, Mapping):
                    current_source = table.get(key)
                elif isinstance(table, (tuple, list, set, frozenset)):
                    current_source = key if key in table else None
                else:
                    current_source = None
                source = str(record["source"])
                target = str(record["translation"])
                if current_source != source:
                    raise ValueError(
                        f"stale reviewed runtime source {table_name}/{key}: {path}"
                    )
                if record["source_sha256"] != hashlib.sha256(
                    source.encode("utf-8")
                ).hexdigest():
                    raise ValueError(
                        f"stale reviewed runtime hash {table_name}/{key}: {path}"
                    )
                if _contextualize(target, language, source) != target:
                    raise ValueError(
                        f"non-idempotent reviewed runtime target "
                        f"{table_name}/{key}: {path}"
                    )
                if _translation_rejection_reasons(
                    source, target, language, force=_looks_translatable(source),
                ):
                    raise ValueError(
                        f"rejected reviewed runtime target "
                        f"{table_name}/{key}: {path}"
                    )
                previous = reviewed.setdefault(source, target)
                if previous != target:
                    raise ValueError(
                        f"conflicting reviewed runtime targets for {source!r}"
                    )
    finally:
        _REVIEWED_RUNTIME_LOADING.discard(language)
    return reviewed


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


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination_mode = stat.S_IMODE(path.stat().st_mode)
    except FileNotFoundError:
        destination_mode = 0o664
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        temporary.chmod(destination_mode)
        temporary.replace(path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


@contextmanager
def _exclusive_cache_lock(path: Path):
    """Hold a stable advisory lock while merging one locale cache.

    The lock file is deliberately persistent. Unlinking it after unlock lets
    a third process lock a new inode while an earlier waiter still owns the
    old one. ``flock`` also avoids the O_EXCL/empty-PID race where a contender
    can observe the owner file between creation and its first write.
    """
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    acquired = False
    try:
        for _attempt in range(1200):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
                break
            except BlockingIOError:
                time.sleep(0.05)
        if not acquired:
            raise TimeoutError(
                f"timed out waiting for translation cache {path}"
            )
        os.ftruncate(descriptor, 0)
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        try:
            if acquired:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            # Closing the descriptor also releases flock, including when an
            # injected/platform unlock error occurs above.
            os.close(descriptor)


def _merge_write_translation_cache(
    path: Path, cache: Mapping[str, str], baseline: dict[str, str],
) -> None:
    """Atomically merge only this process's updates into a shared cache."""
    updates = {
        str(key): str(value)
        for key, value in cache.items()
        if baseline.get(str(key)) != str(value)
    }
    deletions = {
        key: value for key, value in baseline.items() if key not in cache
    }
    if not updates and not deletions:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with _exclusive_cache_lock(path):
        try:
            current = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(current, dict):
                current = {}
        except (FileNotFoundError, json.JSONDecodeError):
            current = {}
        for key, old_value in deletions.items():
            # Never erase a checkpoint another lane updated after our read.
            if current.get(key) == old_value:
                current.pop(key, None)
        current.update(updates)
        _atomic_write_text(
            path,
            json.dumps(current, ensure_ascii=False, sort_keys=True),
        )
    for key in deletions:
        baseline.pop(key, None)
    baseline.update(updates)


def _source_hash(text: object) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def _source_hashes(sources: Mapping[str, object]) -> dict[tuple[str, str], str]:
    """Return one current-English hash for every runtime catalog entry."""
    hashes: dict[tuple[str, str], str] = {}
    for table_name, source_name in (
        ("SETTING_LABELS", "setting_labels"),
        ("SETTING_TOOLTIPS", "setting_tooltips"),
        ("MODULE_SUMMARIES", "module_summaries"),
    ):
        for key, source in sources[source_name].items():
            hashes[(table_name, str(key))] = _source_hash(source)
    for table_name, source_name in (
        ("CATEGORY_HELP", "categories"),
        ("UI", "ui"),
    ):
        for source in sources[source_name]:
            hashes[(table_name, str(source))] = _source_hash(source)
    return dict(sorted(hashes.items()))


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
        + "\n"
        + _render_assignment("SOURCE_HASHES", _source_hashes(sources))
    )
    _atomic_write_text(path, text)
    return path


def _protect(
    text: str,
    marker_style: str = "xml",
    pattern: re.Pattern[str] = _PROTECT_RE,
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
    protected = pattern.sub(lambda match: token(match.group(0)), str(text))
    markers = (
        (f"<x{i}>" if marker_style == "xml" else f"{i}X{i}")
        for i in range(len(values))
    )
    return protected, dict(zip(markers, values))


def _ascii_adjacent_kind(char: str) -> str | None:
    """Classify only ambiguous ASCII marker neighbours."""
    if re.fullmatch(r"[0-9]", char):
        return "digit"
    if re.fullmatch(r"[A-Za-z_]", char):
        return "word"
    return None


def _marker_source_contract(
    protected_text: str | None, marker: str,
) -> tuple[str | None, str | None]:
    """Return the exact marker's source-side ASCII adjacency contract."""
    if protected_text is None:
        return None, None
    occurrences = list(re.finditer(re.escape(marker), protected_text))
    if len(occurrences) != 1:
        raise ValueError(
            f"protected input did not contain {marker} exactly once"
        )
    match = occurrences[0]
    left = protected_text[match.start() - 1] if match.start() else ""
    right = (
        protected_text[match.end()] if match.end() < len(protected_text)
        else ""
    )
    return _ascii_adjacent_kind(left), _ascii_adjacent_kind(right)


def _restore(
    text: str,
    protected: Mapping[str, str],
    *,
    protected_text: str | None = None,
) -> str:
    """Restore one unique occurrence of every protected marker.

    ``protected_text`` is the exact model input produced by :func:`_protect`.
    It permits a shortened marker to remain joined to a target word or number
    only when that same marker had the same kind of source adjacency (on both
    numeric sides where applicable). This recovers deterministic Marian
    tokenization damage without globally treating word-like ``x7>`` text as a
    placeholder.
    """
    restored = str(text)
    expected_xml_ids = {
        re.search(r"\d+", marker).group(0)
        for marker in protected
        if marker.startswith("<")
    }
    explicit_xml = re.compile(
        r"<\s*[xX]\s*(\d+)\s*>|[xX]\s*(\d+)\s*>"
    )
    for match in explicit_xml.finditer(restored):
        marker_id = match.group(1) or match.group(2)
        if marker_id not in expected_xml_ids:
            raise ValueError(
                f"translation invented protection token x{marker_id}>: "
                f"{text!r}"
            )
    if re.search(r"Z\s*X\s*Q\s*\d", restored):
        raise ValueError(f"unrestored protection token: {restored!r}")

    matches: list[tuple[int, int, str, str]] = []
    for marker, value in protected.items():
        digits = re.search(r"\d+", marker).group(0)
        source_left, source_right = _marker_source_contract(
            protected_text, marker,
        )
        if marker.startswith("<"):
            # Marian commonly preserves the marker number and closing angle
            # bracket while dropping only the opening ``<``. It can also
            # attach that shortened marker directly to a preceding source
            # number (``0<x6>`` -> ``0x6>``). Accept only this narrow form:
            # the expected number, exactly once, with ``>`` still present.
            # The ASCII-letter guard prevents ``matrix6>`` from becoming a
            # false marker match.
            allowed_left = r"(?<![A-Za-z0-9_])"
            if source_left == "word":
                allowed_left = (
                    r"(?:(?<=[A-Za-z_])|(?<![A-Za-z0-9_]))"
                )
            elif source_left == "digit":
                allowed_left = (
                    r"(?:(?<=[0-9])|(?<![A-Za-z0-9_]))"
                )
            shortened = rf"{allowed_left}[xX]\s*{digits}\s*>"
            fuzzy = (
                rf"(?:<\s*[xX]\s*{digits}\s*>|{shortened})"
            )
        else:
            # Do not use Unicode word boundaries here. Translation models
            # routinely remove the space beside a marker: Marian emits
            # ``0X0A tradução`` and Korean attaches particles such as
            # ``0X0을``. In both cases the marker is present exactly once,
            # but ``\b`` sees the neighbouring Latin/Hangul letter as another
            # word character and falsely rejects the entire translation.
            # Digit-only guards still keep marker 1 out of 10X01 while
            # allowing ordinary target-language text to touch either edge.
            left_guard = "" if source_left == "digit" else r"(?<![0-9])"
            right_guard = "" if source_right == "digit" else r"(?![0-9])"
            fuzzy = (
                rf"{left_guard}{digits}\s*[xX]\s*{digits}{right_guard}"
            )
        marker_matches = list(re.finditer(fuzzy, restored))
        if len(marker_matches) != 1:
            raise ValueError(
                f"translation did not preserve {marker} exactly once: {text!r}"
            )
        match = marker_matches[0]
        matches.append((match.start(), match.end(), marker, value))

    # Target grammar may legitimately reorder literals (especially in SOV
    # languages). Sort the unique output spans rather than imposing English
    # marker order, reject only an impossible overlap, and replace from right
    # to left so the preflight offsets remain stable.
    matches.sort(key=lambda item: (item[0], item[1]))
    if any(left[1] > right[0] for left, right in zip(matches, matches[1:])):
        raise ValueError(f"translation overlapped protection tokens: {text!r}")

    # ``N X N`` is also ordinary dimension notation. Preserve any such raw
    # tokens not claimed as generated markers exactly as the protected model
    # input did; hallucinated numeric markers must not slip through merely
    # because their index was not expected.
    numeric_shape = re.compile(r"(?<!\d)(\d+)\s*[xX]\s*(\d+)(?!\d)")

    def unclaimed_numeric_shapes(
        value: str, excluded: Iterable[tuple[int, int]],
    ) -> Counter[str]:
        spans = tuple(excluded)
        return Counter(
            f"{match.group(1)}x{match.group(2)}"
            for match in numeric_shape.finditer(value)
            if not any(
                match.start() < end and start < match.end()
                for start, end in spans
            )
        )

    target_numeric = unclaimed_numeric_shapes(
        restored, ((start, end) for start, end, _marker, _value in matches),
    )
    source_marker_spans: list[tuple[int, int]] = []
    source_contract_text = protected_text or ""
    if protected_text is not None:
        for marker in protected:
            occurrence = re.search(re.escape(marker), protected_text)
            if occurrence is not None:
                source_marker_spans.append(occurrence.span())
    source_numeric = unclaimed_numeric_shapes(
        source_contract_text, source_marker_spans,
    )
    if target_numeric != source_numeric:
        raise ValueError(
            "translation changed unprotected numeric marker/dimension "
            f"tokens: {text!r}"
        )

    for start, end, _marker, value in reversed(matches):
        restored = restored[:start] + value + restored[end:]
    return restored.strip()


def _translation_chunks(text: str, limit: int = 320) -> list[str]:
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
        if len(chunk) <= limit:
            result.append(chunk)
            continue
        comma_chunks = [
            part.strip()
            for part in re.findall(r".+?(?:,(?:\s+|$)|$)", chunk, re.DOTALL)
            if part.strip()
        ]
        for piece in comma_chunks or [chunk]:
            while len(piece) > limit:
                protected_ranges = [
                    (match.start(), match.end())
                    for match in _PROTECT_RE.finditer(piece)
                ]
                candidates = [
                    match.start() for match in re.finditer(r"\s+", piece)
                    if match.start() <= limit
                    and not any(
                        start < match.start() < end
                        for start, end in protected_ranges
                    )
                ]
                split_at = candidates[-1] if candidates else -1
                if split_at < limit // 2:
                    # An unusually long protected literal must stay whole; the
                    # tokenizer preflight will reject it rather than corrupting
                    # it. Ordinary prose can split at the hard boundary.
                    covering = next((
                        end for start, end in protected_ranges
                        if start < limit < end
                    ), None)
                    split_at = covering or limit
                result.append(piece[:split_at].strip())
                piece = piece[split_at:].strip()
            if piece:
                result.append(piece)
    return result


_CONTEXT_CLAUSE_BOUNDARY_RE = re.compile(
    # Keep the exact punctuation/whitespace as reconstruction chrome.  The
    # connective belongs to the following translated clause so the model sees
    # its grammatical role.  Colons require following whitespace, avoiding
    # URLs and compact type/shape notation.
    r":\s+|"
    r"\s+(?:—|–|--)\s+|"
    r",\s+(?=(?:and|but|so|because|although|while|whereas|which|when|if|"
    r"unless|rather\s+than)\b)|"
    r"(?<![,;:])\s+(?=(?:but|because|although|whereas)\b)",
    re.IGNORECASE,
)


def _context_clause_plan(text: str) -> list[tuple[str, bool]]:
    """Return exact source spans for a protected-aware clause retry.

    ``True`` spans are prose sent independently to the same translation model;
    ``False`` spans are byte-exact punctuation/whitespace chrome.  Empty means
    the source has fewer than two useful clauses and should retain the result
    of the sentence retry instead of losing context for no benefit.
    """
    source = str(text)
    protected_ranges = [
        (match.start(), match.end())
        for match in _CONTEXT_HARD_PROTECT_RE.finditer(source)
    ]

    def protected(start: int, end: int) -> bool:
        return any(start < protected_end and end > protected_start
                   for protected_start, protected_end in protected_ranges)

    boundaries = [
        match for match in _CONTEXT_CLAUSE_BOUNDARY_RE.finditer(source)
        if not protected(match.start(), match.end())
    ]
    if not boundaries:
        return []

    plan: list[tuple[str, bool]] = []
    cursor = 0
    for boundary in boundaries:
        if boundary.start() > cursor:
            plan.append((source[cursor:boundary.start()], True))
        plan.append((boundary.group(0), False))
        cursor = boundary.end()
    if cursor < len(source):
        plan.append((source[cursor:], True))

    def has_prose(piece: str) -> bool:
        unprotected = _CONTEXT_HARD_PROTECT_RE.sub(" ", piece)
        return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", unprotected))

    plan = [
        (piece, translate and has_prose(piece))
        for piece, translate in plan
    ]
    if sum(translate for _piece, translate in plan) < 2:
        return []
    return plan


def _contextualize(value: str, language: str, source: str = "") -> str:
    reviewed = _reviewed_translation(str(source), language)
    if reviewed is not None:
        return reviewed
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
    if "**>" not in str(source):
        corrected = corrected.replace("**>", "**")
    # Restoration can leave an angle bracket on either side of a protected
    # product marker.  Neither form is meaningful prose; remove it only when
    # the exact adjacency was absent from the canonical source.
    for term in _PROTECTED_TERMS:
        if f">{term}" not in str(source):
            corrected = corrected.replace(f">{term}", term)
    # Translation tokenizers sometimes insert whitespace immediately inside
    # RST emphasis delimiters.  Strip only the captured interior edges, never
    # the surrounding prose whitespace.
    inline_pattern = (
        _RST_ROLE_PATTERN + r"|"
        r"``[^`]+``|`[^`]+`_?"
    )

    def emphasis_view(text: str) -> str:
        return re.sub(
            inline_pattern,
            lambda match: "X" * len(match.group(0)),
            str(text),
        )

    source_view = emphasis_view(source)
    target_view = emphasis_view(corrected)
    source_markers = list(re.finditer(r"\*{1,2}", source_view))
    target_markers = list(re.finditer(r"\*{1,2}", target_view))
    if (
        [match.group(0) for match in source_markers]
        == [match.group(0) for match in target_markers]
    ):
        source_marker_index = {
            match.start(): index for index, match in enumerate(source_markers)
        }
        pairs: list[tuple[int, int]] = []
        for pattern, width in (
            (re.compile(r"\*\*(?!\s)([^*\n]*?\S)\*\*"), 2),
            (re.compile(r"(?<!\*)\*(?![\s*])([^*\n]*?\S)\*(?!\*)"), 1),
        ):
            for match in pattern.finditer(source_view):
                opening = source_marker_index.get(match.start())
                closing = source_marker_index.get(match.end() - width)
                if opening is not None and closing is not None:
                    pairs.append((opening, closing))
        # Work right-to-left so trimming one emphasized body cannot shift the
        # marker offsets used for an earlier body.
        for opening, closing in sorted(pairs, reverse=True):
            body_start = target_markers[opening].end()
            body_end = target_markers[closing].start()
            corrected = (
                corrected[:body_start]
                + corrected[body_start:body_end].strip()
                + corrected[body_end:]
            )
    for literal in re.findall(
        _RST_ROLE_PATTERN + r"|"
        r"``[^`]+``|`[^`]+`_?",
        str(source),
    ):
        if f"{literal}>" not in str(source):
            corrected = corrected.replace(f"{literal}>", literal)
        if (
            language != "zh_CN"
            and not re.search(re.escape(literal) + r"(?=[A-Za-zÀ-ÖØ-öø-ÿ])", str(source))
        ):
            corrected = re.sub(
                re.escape(literal) + r"(?=[A-Za-zÀ-ÖØ-öø-ÿ])",
                lambda _match, value=literal: value + " ",
                corrected,
            )
    for term in _PROTECTED_TERMS:
        if f"{term}>" not in str(source):
            corrected = corrected.replace(f"{term}>", term)

    # Hide hard literals while applying lexical false-friend corrections.
    # A prior implementation rewrote ``toggle`` inside
    # ``bool_key -> [categories to toggle]`` and only the later syntax gate
    # prevented that corrupted example from reaching a catalog.
    context_literals: dict[str, str] = {}

    def hide_context_literal(match: re.Match[str]) -> str:
        token = f"\ue000{len(context_literals)}\ue001"
        context_literals[token] = match.group(0)
        return token

    corrected = _CONTEXT_HARD_PROTECT_RE.sub(
        hide_context_literal, corrected,
    )
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
            elif re.search(r"[\u3400-\u9fff]", wrong):
                # ``\b`` treats adjacent Han characters as word characters,
                # so it cannot match a term embedded in a compound such as
                # ``细胞面具``. Han-script terminology has no whitespace word
                # boundary; replace the reviewed literal directly.
                corrected = corrected.replace(wrong, right)
            elif wrong[:1].isalnum() and wrong[-1:].isalnum():
                corrected = re.sub(
                    rf"\b{re.escape(wrong)}\b",
                    right,
                    corrected,
                    flags=re.IGNORECASE,
                )
            else:
                corrected = corrected.replace(wrong, right)
    for source_pattern, wrong_pattern, right in (
        SOURCE_CONTEXT_REGEX_REPLACEMENTS.get(language, ())
    ):
        if re.search(source_pattern, str(source), flags=re.IGNORECASE):
            corrected = re.sub(wrong_pattern, right, corrected)
    # A source-conditioned replacement can expose a second global cleanup
    # (for example Chinese ``图像作物`` first becomes ``图像图像裁剪``).
    # Reapplying this small, idempotent table keeps compound terms natural.
    for wrong, right in CONTEXT_REPLACEMENTS.get(language, ()):
        corrected = corrected.replace(wrong, right)
    for token, literal in context_literals.items():
        corrected = corrected.replace(token, literal)
    for wrong, right in POST_CONTEXT_REPLACEMENTS.get(language, ()):
        corrected = corrected.replace(wrong, right)
    if language == "pt" and _gui_screen_source(source):
        corrected = re.sub(r"\bTriagens\b", "Telas", corrected)
        corrected = re.sub(r"\btriagens\b", "telas", corrected)
        corrected = re.sub(r"\bTriagem\b", "Tela", corrected)
        corrected = re.sub(r"\btriagem\b", "tela", corrected)
    # Never delete a generic ``>`` here. It may be the meaningful half of
    # ``->`` or a numeric comparison. The narrow adjacency repairs above can
    # prove their bracket is a tokenizer artifact; every remaining surplus is
    # rejected by the global syntax/semantic gate and regenerated.
    return corrected


def _syntax_preserved(
    source: str,
    value: str,
    *,
    check_emphasis: bool = True,
    allow_reviewed_product_normalization: bool = False,
) -> bool:
    if not str(value).strip():
        return False

    # Python-format fields are runtime API, not prose.  A translated string
    # with an unmatched brace is especially dangerous because it looks fine
    # in the catalog and then raises only when the tooltip is displayed.
    from string import Formatter

    def format_fields(text: str) -> set[str] | None:
        try:
            return {
                re.sub(r"\s+", " ", name) for _literal, name, _spec, _conversion
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
        _RST_ROLE_PATTERN,
        r"``[^`]+``|`[^`]+`_?",
        r"(?<![-=])(?:->|=>|>=|<=|==|!=|>|<)(?![=>])",
    )
    structural = all(
        [re.sub(r"\s+", " ", match) for match in re.findall(
            pattern, str(source)
        )]
        == [re.sub(r"\s+", " ", match) for match in re.findall(
            pattern, str(value)
        )]
        for pattern in patterns
    )
    def without_inline_code(text: str) -> str:
        return re.sub(
            _RST_ROLE_PATTERN + r"|"
            r"``[^`]+``|`[^`]+`_?",
            # Non-whitespace filler preserves whether an inline literal sits
            # directly against an emphasis delimiter (``**``foo``**``).
            lambda match: "X" * len(match.group(0)),
            str(text),
        )

    emphasis_source = without_inline_code(source)
    emphasis_value = without_inline_code(value)
    source_strong = re.findall(
        r"\*\*(?!\s)([^*\n]*?\S)\*\*", emphasis_source
    )
    value_strong = re.findall(
        r"\*\*(?!\s)([^*\n]*?\S)\*\*", emphasis_value
    )
    source_emphasis = re.findall(
        r"(?<!\*)\*(?![\s*])([^*\n]*?\S)\*(?!\*)", emphasis_source
    )
    value_emphasis = re.findall(
        r"(?<!\*)\*(?![\s*])([^*\n]*?\S)\*(?!\*)", emphasis_value
    )
    emphasis_valid = (
        len(source_strong) == len(value_strong)
        and len(source_emphasis) == len(value_emphasis)
    )
    def matches(
        pattern: re.Pattern[str], text: str,
    ) -> Counter[str]:
        return Counter(
            re.sub(r"\s+", " ", match.group(0))
            for match in pattern.finditer(str(text))
        )

    # Validate each source-side contract independently.  A combined regex is
    # appropriate for marker substitution, where matches cannot overlap, but
    # it is not sufficient for validation: a quoted label can itself contain
    # GitHub, a CLI flag, or a snake_case identifier.  Per-pattern counters
    # ensure that every nested literal is retained byte-for-byte.
    missing_or_changed = False
    unexpected_nonquotes = False
    for pattern in _PROTECT_PATTERNS:
        protected = matches(pattern, source)
        rendered = matches(pattern, value)
        if any(rendered[literal] < count
               for literal, count in protected.items()):
            missing_or_changed = True
        # A translated long quotation can become a short quotation in the
        # target language.  Only source-side short quotes are contracts; new
        # target-language quotes are prose.  Every code/CLI/RST pattern keeps
        # the stricter no-new-literals rule.
        if (pattern not in _QUOTE_PROTECT_PATTERNS
                and rendered - protected):
            unexpected_nonquotes = True

    def product_matches(text: str) -> Counter[str]:
        products = matches(_PRODUCT_PROTECT_RE, text)
        # English acronym plurals use a bare trailing ``s`` while reviewed
        # target prose normally inflects the surrounding noun (for example
        # ``CSVs`` -> ``CSV-filer``).  Preserve the acronym itself exactly and
        # normalize only this explicit reviewed set; product names in general
        # retain the strict byte-for-byte contract.
        for plural, singular in (("CSVs", "CSV"), ("PNGs", "PNG"), ("UMAPs", "UMAP")):
            count = products.pop(plural, 0)
            if count:
                products[singular] += count
        return products

    protected_products = product_matches(source)
    rendered_products = product_matches(value)
    product_mismatch = (
        any(rendered_products[literal] < count
            for literal, count in protected_products.items())
        or bool(rendered_products - protected_products)
    )
    return (
        structural
        and (emphasis_valid or not check_emphasis)
        and not missing_or_changed
        and not unexpected_nonquotes
        and (not product_mismatch or allow_reviewed_product_normalization)
    )


def _syntax_preserved_or_reviewed(
    source: str, value: str, language: str,
) -> bool:
    """Accept exact reviewed UI wording without weakening generated prose.

    A handful of short reviewed labels intentionally canonicalize lowercase
    acronyms (``pca`` -> ``PCA`` and ``B qc`` -> ``B QC``).  The generic
    structural gate correctly rejects that change for model output.  For the
    exact source/language/value triple in :data:`MANUAL_UI`, retain every
    source-side field, RST role, code literal and delimiter while allowing
    only the reviewed product-token normalization.
    """
    if _syntax_preserved(source, value):
        return True
    reviewed = _reviewed_translation(str(source), language)
    return bool(
        reviewed is not None
        and str(value) == reviewed
        and _syntax_preserved(
            source,
            value,
            allow_reviewed_product_normalization=True,
        )
    )


def _looks_degenerate(source: str, value: str, language: str) -> bool:
    """Detect obvious model loops without rejecting normal repeated prose."""
    rendered = str(value).strip()
    if not rendered:
        return True
    if any(marker in rendered.casefold() for marker in (
        "city name (optional",
        "probably does not need a translation",
        "unit description in lists",
        "omited",
        "oh my god",
        "dios mío",
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


def _has_expected_script(
    source: str,
    value: str,
    language: str,
    *,
    force: bool = False,
) -> bool:
    # Exact reviewed labels may be technical acronyms whose correct localized
    # spelling is still Latin script (``pca`` -> ``PCA``, ``iou`` -> ``IoU``).
    # The source-bound review contract is stronger than a generic script
    # heuristic; generated or cached candidates still require target script.
    reviewed = _reviewed_translation(str(source), language)
    if reviewed is not None and str(value) == reviewed:
        return True
    pattern = {
        "zh_CN": r"[\u3400-\u9fff]",
        "hi": r"[\u0900-\u097f]",
        "ko": r"[\uac00-\ud7af]",
    }.get(language)
    if pattern is None or (not force and not _looks_translatable(source)):
        return True
    return bool(re.search(pattern, str(value)))


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
    target_hashes = getattr(target, "SOURCE_HASHES", {})

    def hash_is_current(table_name: str, key: object, source: object) -> bool:
        return target_hashes.get(
            (table_name, str(key))
        ) == _source_hash(source)

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
                and hash_is_current(name, key, source)
                and _translation_candidate_valid(source, value, language)
            ):
                cache.setdefault(
                    str(source), _contextualize(value, language, source)
                )
    for name in ("CATEGORY_HELP", "UI"):
        for source, value in getattr(target, name, {}).items():
            if (
                isinstance(value, str)
                and hash_is_current(name, source, source)
                and _translation_candidate_valid(source, value, language)
            ):
                cache.setdefault(
                    str(source), _contextualize(value, language, source)
                )
    canonical_modules = getattr(english, "MODULE_SUMMARIES", {})
    translated_modules = getattr(target, "MODULE_SUMMARIES", {})
    for key, source in canonical_modules.items():
        value = translated_modules.get(key)
        if (
            isinstance(value, str)
            and hash_is_current("MODULE_SUMMARIES", key, source)
            and _translation_candidate_valid(source, value, language)
        ):
            cache.setdefault(
                str(source), _contextualize(value, language, source)
            )


def _current_invalid_sources(
    sources: Iterable[str],
    translated: Mapping[str, str],
    candidate_valid: Callable[[str, str], bool],
) -> list[str]:
    """Recompute failures after a retry without trusting historical state."""
    return [
        source for source in sources
        if not candidate_valid(source, translated.get(source, source))
    ]


def _fragment_retry_sources(
    sources: Iterable[str],
    translated: Mapping[str, str],
    candidate_valid: Callable[[str, str], bool],
    latest_failures: Mapping[str, Iterable[str]],
) -> list[str]:
    """Return current failures eligible for context-free fragment repair.

    Sentence retry can rescue a source that failed primary marker restoration.
    Selecting the later fragment pass from the historical failure list would
    translate that source again without sentence context and could overwrite
    the valid rescue. Recompute from current state, then admit only a latest
    marker-restoration or protected-syntax failure. Every linguistic, script,
    EOS and caller failure stays out.
    """
    mechanical = frozenset({"marker_restore", "protected_syntax"})
    return [
        source for source in _current_invalid_sources(
            sources, translated, candidate_valid,
        )
        if bool(frozenset(latest_failures.get(source, ())))
        and frozenset(latest_failures.get(source, ())) <= mechanical
    ]


def _join_completed_fragments(
    pieces: Iterable[str], failure_reasons: Iterable[str],
) -> str | None:
    """Join fragment output only when every model sequence completed."""
    if frozenset(failure_reasons):
        return None
    joined = ""
    for piece in pieces:
        if (joined and piece and joined[-1].isalnum()
                and piece[0].isalnum()):
            joined += " "
        joined += piece
    return joined.strip()


def _ranked_generation_kwargs(beams: int) -> dict[str, int]:
    """Return one ranked sequence per beam without launching another search."""
    beam_count = int(beams)
    if beam_count < 1:
        raise ValueError("translation beam count must be at least one")
    return {
        "num_beams": beam_count,
        "num_return_sequences": beam_count,
    }


def _group_ranked_outputs(
    output: object,
    decoded: list[str],
    batch_count: int,
    rank_count: int,
) -> list[list[tuple[object, str]]]:
    """Group Hugging Face's input-major beam output by source input."""
    expected = int(batch_count) * int(rank_count)
    if len(decoded) != expected or len(output) != expected:  # type: ignore[arg-type]
        raise ValueError(
            "ranked generation returned an unexpected sequence count: "
            f"expected={expected} output={len(output)} "  # type: ignore[arg-type]
            f"decoded={len(decoded)}"
        )
    return [
        [
            (output[input_index * rank_count + rank],  # type: ignore[index]
             decoded[input_index * rank_count + rank])
            for rank in range(rank_count)
        ]
        for input_index in range(batch_count)
    ]


def _first_valid_ranked_candidate(
    candidates: Iterable[tuple[object, str]],
    *,
    completed: Callable[[object], bool],
    restore: Callable[[str], str],
    evaluate: Callable[[str], tuple[str, Iterable[str]]],
) -> tuple[str | None, frozenset[str]]:
    """Select the first complete, restorable candidate passing every gate."""
    rejections: set[str] = set()
    for sequence, decoded in candidates:
        if not completed(sequence):
            rejections.add("eos")
            continue
        try:
            raw_value = restore(decoded)
        except ValueError:
            rejections.add("marker_restore")
            continue
        candidate, failures = evaluate(raw_value)
        rank_failures = frozenset(failures)
        if not rank_failures:
            return candidate, frozenset()
        rejections.update(rank_failures)
    return None, frozenset(rejections or {"caller_gate"})


def _rank_aligned_joins(
    ranked_pieces: Iterable[list[str | None]],
    joiner: Callable[[list[str]], str],
) -> list[str | None]:
    """Join only pieces with the same beam rank; never mix local winners."""
    pieces = list(ranked_pieces)
    if not pieces:
        return []
    rank_count = len(pieces[0])
    if any(len(piece) != rank_count for piece in pieces):
        raise ValueError("ranked translation pieces have inconsistent widths")
    joined: list[str | None] = []
    for rank in range(rank_count):
        values = [piece[rank] for piece in pieces]
        joined.append(
            None if any(value is None for value in values)
            else joiner([str(value) for value in values])
        )
    return joined


def _translate_batches(
    strings: list[str],
    language: str,
    model_root: Path,
    *,
    device: str,
    batch_size: int,
    beams: int,
    threads: int,
    force_sources: Iterable[str] = (),
    repair_protected: bool = False,
    cache_namespace: str = "",
    candidate_validator: Callable[[str, str, str], bool] | None = None,
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
    cache_baseline = {
        str(key): str(value) for key, value in cache.items()
    }
    _seed_cache_from_catalog(language, cache)
    forced = frozenset(map(str, force_sources))

    def cache_key(source: str) -> str:
        return f"{cache_namespace}\0{source}" if cache_namespace else source

    def checkpoint_cache() -> None:
        _merge_write_translation_cache(cache_path, cache, cache_baseline)

    def candidate_valid(source: str, candidate: str) -> bool:
        """Apply shared release gates plus an optional caller contract."""
        return bool(
            _translation_candidate_valid(
                source, candidate, language, force=source in forced,
            )
            and (
                candidate_validator is None
                or candidate_validator(source, candidate, language)
            )
        )

    def candidate_failures(
        source: str,
        candidate: str,
        *,
        raw_semantic_failure: bool = False,
    ) -> frozenset[str]:
        """Return the latest hard-gate state for one model attempt."""
        return _translation_rejection_reasons(
            source,
            candidate,
            language,
            force=source in forced,
            raw_semantic_failure=raw_semantic_failure,
            candidate_validator=candidate_validator,
        )

    def normalize_candidate(source: str, candidate: str) -> str:
        normalized = _contextualize(candidate, language, source)
        if language == "zh_CN" and normalized != source:
            normalized = _simplify_chinese_prose(normalized)
        return normalized

    def checkpoint_candidate(source: str, candidate: str) -> None:
        """Persist only a candidate that passes every current hard gate."""
        if language == "zh_CN":
            candidate = _simplify_chinese_prose(candidate)
        if candidate_valid(source, candidate):
            cache[cache_key(source)] = candidate
        else:
            cache.pop(cache_key(source), None)

    translated: dict[str, str] = {}
    generated_sources: list[str] = []
    generated_inputs: list[str] = []
    generated_protected_inputs: list[str] = []
    protection: list[dict[str, str]] = []
    latest_failures: dict[str, frozenset[str]] = {}

    from spacr.qt.i18n import CATALOGS
    compact = CATALOGS[language]
    for source in strings:
        source_cache_key = cache_key(source)
        reviewed_value = _reviewed_translation(source, language)
        compact_value = compact.get(source) if source not in forced else None
        if (
            reviewed_value is not None
            and candidate_valid(source, reviewed_value)
        ):
            translated[source] = _contextualize(
                _simplify_chinese_prose(reviewed_value)
                if language == "zh_CN" else reviewed_value,
                language, source,
            )
        elif (
            compact_value is not None
            and candidate_valid(source, compact_value)
        ):
            translated[source] = _contextualize(
                _simplify_chinese_prose(compact_value)
                if language == "zh_CN" else compact_value,
                language, source,
            )
        elif source in _IDENTITY_TEXT:
            translated[source] = source
        elif (
            source not in forced
            and source_cache_key in cache
            and candidate_valid(source, str(cache[source_cache_key]))
        ):
            translated[source] = _contextualize(
                _simplify_chinese_prose(str(cache[source_cache_key]))
                if language == "zh_CN"
                else str(cache[source_cache_key]),
                language, source
            )
        elif source_cache_key in cache and source in forced:
            # A forced API repair has rejected this checkpoint under a stricter
            # canonical/context contract. Remove it and retry the complete
            # sentence/block; never route a semantic/script failure straight
            # to context-free fragments.
            cache.pop(source_cache_key, None)
            protected, mapping = _protect(source)
            generated_sources.append(source)
            generated_inputs.append(prefix + protected)
            generated_protected_inputs.append(protected)
            protection.append(mapping)
        elif source_cache_key in cache and source not in forced:
            # A rejected checkpoint is not translation input. Remove it and
            # retry the complete sentence/block before any syntax-only
            # fragment fallback, just as for a forced API repair.
            cache.pop(source_cache_key, None)
            protected, mapping = _protect(source)
            generated_sources.append(source)
            generated_inputs.append(prefix + protected)
            generated_protected_inputs.append(protected)
            protection.append(mapping)
        else:
            protected, mapping = _protect(source)
            generated_sources.append(source)
            generated_inputs.append(prefix + protected)
            generated_protected_inputs.append(protected)
            protection.append(mapping)

    if generated_inputs:
        packed = sorted(
            zip(
                generated_sources,
                generated_inputs,
                generated_protected_inputs,
                protection,
            ),
            key=lambda item: (len(item[1]), item[0]),
        )
        generated_sources = [item[0] for item in packed]
        generated_inputs = [item[1] for item in packed]
        generated_protected_inputs = [item[2] for item in packed]
        protection = [item[3] for item in packed]

    if not generated_inputs:
        return translated

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    torch.set_num_threads(max(1, threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits setting the inter-op pool only before parallel work
        # starts. A reused process has already fixed the same one-thread pool.
        pass
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
    ranked_generation = _ranked_generation_kwargs(beams)
    rank_count = ranked_generation["num_return_sequences"]

    def output_budget(encoded: Mapping[str, object]) -> int:
        """Allow expansion; EOS checks reject and re-split exhausted output."""
        input_width = int(encoded["input_ids"].shape[1])
        return min(480, max(64, input_width * 3 + 48))
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

    model_input_limit = min(
        480,
        int(getattr(tokenizer, "model_max_length", 480) or 480),
    )

    def split_oversized_source(source: str) -> list[str]:
        """Recursively split prose until every protected piece fits."""
        pending = _translation_chunks(source, limit=320)
        safe: list[str] = []
        while pending:
            piece = pending.pop(0)
            protected, _mapping = _protect(piece, marker_style="numeric")
            width = len(tokenizer(prefix + protected, add_special_tokens=True)[
                "input_ids"
            ])
            if width <= model_input_limit:
                safe.append(piece)
                continue
            if len(piece) < 2:
                raise ValueError(
                    "protected translation literal exceeds model token limit"
                )
            midpoint = len(piece) // 2
            candidates = [
                match.start() for match in re.finditer(r"\s+", piece)
            ]
            split_at = min(candidates, key=lambda value: abs(value - midpoint)) \
                if candidates else midpoint
            left, right = piece[:split_at].strip(), piece[split_at:].strip()
            if not left or not right:
                raise ValueError(
                    "translation input cannot be split below model token limit"
                )
            pending[:0] = [left, right]
        return safe

    # Primary input is never silently truncated. Route any unexpectedly long
    # source into the complete sentence/chunk retry before model invocation.
    oversized_sources: set[str] = set()
    kept_sources: list[str] = []
    kept_inputs: list[str] = []
    kept_protected_inputs: list[str] = []
    kept_protection: list[dict[str, str]] = []
    for source, model_input, protected_input, mapping in zip(
        generated_sources,
        generated_inputs,
        generated_protected_inputs,
        protection,
    ):
        width = len(tokenizer(model_input, add_special_tokens=True)["input_ids"])
        if width > model_input_limit:
            oversized_sources.add(source)
            translated[source] = source
            latest_failures[source] = frozenset({"input_oversized"})
        else:
            kept_sources.append(source)
            kept_inputs.append(model_input)
            kept_protected_inputs.append(protected_input)
            kept_protection.append(mapping)
    all_generated_sources = list(dict.fromkeys([
        *kept_sources, *oversized_sources,
    ]))
    generated_sources = kept_sources
    generated_inputs = kept_inputs
    generated_protected_inputs = kept_protected_inputs
    protection = kept_protection

    def encode_batch(batch: list[str]) -> Mapping[str, object]:
        """Encode without silent truncation; oversized input is a hard bug."""
        encoded = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        widths = encoded.get("attention_mask").sum(dim=1).tolist()
        if any(int(width) > model_input_limit for width in widths):
            raise ValueError(
                "translation input exceeds model token limit; pre-split the "
                f"complete block first (limit={model_input_limit}, "
                f"widths={list(map(int, widths))})"
            )
        return encoded

    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def completed(sequence) -> bool:
        return bool(
            eos_token_id is None
            or int(eos_token_id) in sequence.detach().cpu().tolist()
        )

    def evaluate_raw_candidate(
        source: str, raw_value: str,
    ) -> tuple[str, frozenset[str]]:
        # Source-conditioned terminology review is the correction mechanism,
        # not evidence that a candidate must remain rejected. Validate the
        # corrected value; the semantic gate below still rejects any wrong
        # sense not covered by an exact reviewed source rule.
        candidate = normalize_candidate(source, raw_value)
        failures = candidate_failures(source, candidate)
        if not failures and not candidate_valid(source, candidate):
            failures = frozenset({"caller_gate"})
        return candidate, failures

    def select_ranked_candidate(
        source: str,
        mapping: Mapping[str, str],
        protected_text: str,
        candidates: Iterable[tuple[object, str]],
    ) -> tuple[str | None, frozenset[str]]:
        return _first_valid_ranked_candidate(
            candidates,
            completed=completed,
            restore=lambda value: _restore(
                value, mapping, protected_text=protected_text,
            ),
            evaluate=lambda value: evaluate_raw_candidate(source, value),
        )

    for start in range(0, len(generated_inputs), batch_size):
        batch = generated_inputs[start:start + batch_size]
        encoded = encode_batch(batch)
        if device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=output_budget(encoded),
                **ranked_generation,
                **generation_kwargs,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        ranked_outputs = _group_ranked_outputs(
            output, decoded, len(batch), rank_count,
        )
        for offset, candidates in enumerate(ranked_outputs):
            index = start + offset
            source = generated_sources[index]
            value, failures = select_ranked_candidate(
                source,
                protection[index],
                generated_protected_inputs[index],
                candidates,
            )
            if value is None:
                latest_failures[source] = failures
                translated[source] = source
            else:
                latest_failures.pop(source, None)
                translated[source] = value.strip() or source
            checkpoint_candidate(source, translated[source])
        checkpoint_cache()
        print(
            f"{language}: {min(start + len(batch), len(generated_inputs))}/"
            f"{len(generated_inputs)}",
            flush=True,
        )

    # A structurally damaged translation falls back to canonical English.
    # Secondary marker/fragment translations lose too much sentence context
    # for scientific API prose; the code below remains available for targeted
    # experiments but is deliberately disabled for release generation.
    allow_secondary_repairs = repair_protected

    # Marian occasionally strips angle brackets or a letter from a protected
    # token, especially in Chinese. Retry only those strings with a second,
    # independently tested numeric-X marker before accepting English fallback.
    # Keeping this inside the loaded-model lifetime makes the retry cheap.
    mechanical_failures = frozenset({"marker_restore", "protected_syntax"})
    retry_sources = [
        source for source in all_generated_sources
        if latest_failures.get(source)
        and latest_failures[source] <= mechanical_failures
        and not candidate_valid(source, translated.get(source, source))
    ]
    if retry_sources and not is_m2m and allow_secondary_repairs:
        retry_inputs: list[str] = []
        retry_protected_inputs: list[str] = []
        retry_maps: list[dict[str, str]] = []
        for source in retry_sources:
            protected, mapping = _protect(source, marker_style="numeric")
            retry_inputs.append(prefix + protected)
            retry_protected_inputs.append(protected)
            retry_maps.append(mapping)
        for start in range(0, len(retry_inputs), batch_size):
            batch = retry_inputs[start:start + batch_size]
            encoded = encode_batch(batch)
            if device == "cuda":
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=output_budget(encoded),
                    **ranked_generation,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            ranked_outputs = _group_ranked_outputs(
                output, decoded, len(batch), rank_count,
            )
            for offset, candidates in enumerate(ranked_outputs):
                index = start + offset
                source = retry_sources[index]
                value, failures = select_ranked_candidate(
                    source,
                    retry_maps[index],
                    retry_protected_inputs[index],
                    candidates,
                )
                if value is None:
                    latest_failures[source] = failures
                    translated[source] = source
                else:
                    latest_failures.pop(source, None)
                    translated[source] = value.strip() or source
                checkpoint_candidate(source, translated[source])
            checkpoint_cache()
        print(
            f"{language}: retried protected strings={len(retry_sources)}",
            flush=True,
        )

    # A final sentence-sized pass handles dense help paragraphs where even the
    # numeric retry asks one model sequence to carry too many protected values.
    # Only accept the recomposed translation when every chunk restores and the
    # complete paragraph retains its structural/API tokens.
    chunk_sources = [] if not allow_secondary_repairs else (
        _current_invalid_sources(
            all_generated_sources, translated, candidate_valid,
        )
    )
    if chunk_sources:
        chunk_inputs: list[str] = []
        chunk_protected_inputs: list[str] = []
        chunk_maps: list[dict[str, str]] = []
        chunk_owners: list[tuple[str, int]] = []
        chunks_by_source: dict[str, list[str]] = {
            source: split_oversized_source(source) for source in chunk_sources
        }
        for source, chunks in chunks_by_source.items():
            for index, chunk in enumerate(chunks):
                protected, mapping = _protect(chunk, marker_style="numeric")
                chunk_inputs.append(prefix + protected)
                chunk_protected_inputs.append(protected)
                chunk_maps.append(mapping)
                chunk_owners.append((source, index))
        restored_chunks: dict[str, list[list[str | None]]] = {
            source: [[None] * rank_count for _chunk in chunks]
            for source, chunks in chunks_by_source.items()
        }
        chunk_failures: dict[str, list[set[str]]] = {
            source: [set() for _rank in range(rank_count)]
            for source in chunks_by_source
        }
        for start in range(0, len(chunk_inputs), batch_size):
            batch = chunk_inputs[start:start + batch_size]
            encoded = encode_batch(batch)
            if device == "cuda":
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded,
                    max_new_tokens=output_budget(encoded),
                    **ranked_generation,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            ranked_outputs = _group_ranked_outputs(
                output, decoded, len(batch), rank_count,
            )
            for offset, candidates in enumerate(ranked_outputs):
                owner, chunk_index = chunk_owners[start + offset]
                restored = restored_chunks[owner][chunk_index]
                for rank, (sequence, value) in enumerate(candidates):
                    if not completed(sequence):
                        chunk_failures[owner][rank].add("eos")
                        continue
                    try:
                        restored[rank] = _restore(
                            value,
                            chunk_maps[start + offset],
                            protected_text=(
                                chunk_protected_inputs[start + offset]
                            ),
                        )
                    except ValueError:
                        chunk_failures[owner][rank].add("marker_restore")
        accepted = 0
        for source, values in restored_chunks.items():
            joined_candidates = _rank_aligned_joins(
                values,
                lambda pieces: " ".join(piece.strip() for piece in pieces),
            )
            candidate = None
            rank_failures = [set(value) for value in chunk_failures[source]]
            for rank, raw_candidate in enumerate(joined_candidates):
                if raw_candidate is None:
                    continue
                ranked_candidate, failures = evaluate_raw_candidate(
                    source, raw_candidate,
                )
                if not failures:
                    candidate = ranked_candidate
                    break
                rank_failures[rank].update(failures)
            if candidate is not None:
                translated[source] = candidate
                checkpoint_candidate(source, candidate)
                latest_failures.pop(source, None)
                accepted += 1
            else:
                latest_failures[source] = frozenset(
                    set().union(*rank_failures) or {"marker_restore"}
                )
                translated[source] = source
                cache.pop(cache_key(source), None)
        checkpoint_cache()
        print(
            f"{language}: sentence retry accepted={accepted}/"
            f"{len(chunk_sources)}",
            flush=True,
        )

    # Sentence chunks are deliberately conservative and retain whole API
    # blocks whenever they fit.  If a current failure remains, make one final
    # contextual attempt at strong clause boundaries.  Punctuation and
    # whitespace separators are reconstructed byte-for-byte, while each prose
    # clause uses the already-tested numeric marker contract.  Clause outputs
    # are never cached independently: only a fully joined candidate that passes
    # every shared and caller gate can become a checkpoint.
    clause_sources = [] if not allow_secondary_repairs else (
        _current_invalid_sources(
            all_generated_sources, translated, candidate_valid,
        )
    )
    clause_plans: dict[str, list[tuple[str, bool]]] = {}
    clause_inputs: list[str] = []
    clause_protected_inputs: list[str] = []
    clause_maps: list[dict[str, str]] = []
    clause_owners: list[tuple[str, int]] = []
    for source in clause_sources:
        plan = _context_clause_plan(source)
        if not plan:
            continue
        pending: list[
            tuple[str, str, dict[str, str], tuple[str, int]]
        ] = []
        oversized = False
        for index, (piece, translate_piece) in enumerate(plan):
            if not translate_piece:
                continue
            protected, mapping = _protect(piece, marker_style="numeric")
            model_input = prefix + protected
            width = len(tokenizer(
                model_input, add_special_tokens=True,
            )["input_ids"])
            if width > model_input_limit:
                oversized = True
                break
            pending.append((model_input, protected, mapping, (source, index)))
        if oversized or len(pending) < 2:
            continue
        clause_plans[source] = plan
        for model_input, protected, mapping, owner in pending:
            clause_inputs.append(model_input)
            clause_protected_inputs.append(protected)
            clause_maps.append(mapping)
            clause_owners.append(owner)

    clause_values: dict[str, list[list[str | None]]] = {
        source: [
            [piece] * rank_count if not translate_piece
            else [None] * rank_count
            for piece, translate_piece in plan
        ]
        for source, plan in clause_plans.items()
    }
    clause_failures: dict[str, list[set[str]]] = {
        source: [set() for _rank in range(rank_count)]
        for source in clause_plans
    }
    for start in range(0, len(clause_inputs), batch_size):
        batch = clause_inputs[start:start + batch_size]
        encoded = encode_batch(batch)
        if device == "cuda":
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
        with torch.inference_mode():
            output = model.generate(
                **encoded,
                max_new_tokens=output_budget(encoded),
                **ranked_generation,
                **generation_kwargs,
            )
        decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
        ranked_outputs = _group_ranked_outputs(
            output, decoded, len(batch), rank_count,
        )
        for offset, candidates in enumerate(ranked_outputs):
            owner, plan_index = clause_owners[start + offset]
            restored = clause_values[owner][plan_index]
            for rank, (sequence, value) in enumerate(candidates):
                if not completed(sequence):
                    clause_failures[owner][rank].add("eos")
                    continue
                try:
                    restored[rank] = _restore(
                        value,
                        clause_maps[start + offset],
                        protected_text=(
                            clause_protected_inputs[start + offset]
                        ),
                    )
                except ValueError:
                    clause_failures[owner][rank].add("marker_restore")

    clause_accepted = 0
    for source, values in clause_values.items():
        joined_candidates = _rank_aligned_joins(
            values, lambda pieces: "".join(pieces),
        )
        candidate = None
        rank_failures = [set(value) for value in clause_failures[source]]
        for rank, raw_candidate in enumerate(joined_candidates):
            if raw_candidate is None:
                continue
            ranked_candidate, failures = evaluate_raw_candidate(
                source, raw_candidate,
            )
            if not failures:
                candidate = ranked_candidate
                break
            rank_failures[rank].update(failures)
        if candidate is not None:
            translated[source] = candidate
            checkpoint_candidate(source, candidate)
            latest_failures.pop(source, None)
            clause_accepted += 1
        else:
            latest_failures[source] = frozenset(
                set().union(*rank_failures) or {"marker_restore"}
            )
            translated[source] = source
            cache.pop(cache_key(source), None)
    if clause_plans:
        checkpoint_cache()
        print(
            f"{language}: clause retry accepted={clause_accepted}/"
            f"{len(clause_plans)} pieces={len(clause_inputs)}",
            flush=True,
        )

    # Some target tokenizers (especially Devanagari and Hangul models) can
    # still mutate an otherwise simple marker.  As a deterministic last
    # resort, translate only the prose spans *between* protected API values
    # and then splice the untouched values back in.  No synthetic marker ever
    # reaches the model in this pass, so HTML/identifiers cannot leak or drift.
    fragment_sources = [] if not allow_secondary_repairs else (
        _fragment_retry_sources(
            all_generated_sources,
            translated,
            candidate_valid,
            latest_failures,
        )
    )
    if fragment_sources:
        def fragment_is_prose(piece: str) -> bool:
            stripped = piece.strip()
            if not stripped or re.fullmatch(r"[\W\d_]+", stripped):
                return False
            # UI extraction deliberately rejects slash-delimited strings as
            # likely paths. API prose often says "plate/well/field", so that
            # heuristic is too broad for a protected prose fragment.
            if re.fullmatch(r"[\\/]\S+", stripped):
                return False
            return bool(re.search(r"[A-Za-zÀ-ÖØ-öø-ÿ]{2,}", stripped))

        pieces_by_source: dict[str, list[str]] = {}
        fragment_inputs: list[str] = []
        fragment_owners: list[tuple[str, int]] = []
        fragment_totals: defaultdict[str, int] = defaultdict(int)
        for source in fragment_sources:
            protected, mapping = _protect(
                source, pattern=_FRAGMENT_PROTECT_RE
            )
            pieces = re.split(r"(<x\d+>)", protected)
            for index, piece in enumerate(pieces):
                if piece in mapping:
                    pieces[index] = mapping[piece]
                elif fragment_is_prose(piece):
                    fragment_inputs.append(prefix + piece.strip())
                    fragment_owners.append((source, index))
                    fragment_totals[source] += 1
            pieces_by_source[source] = pieces

        owner_order = {
            source: index for index, source in enumerate(fragment_sources)
        }
        owner_max_length: defaultdict[str, int] = defaultdict(int)
        for fragment_input, (owner, _piece_index) in zip(
            fragment_inputs, fragment_owners
        ):
            owner_max_length[owner] = max(
                owner_max_length[owner], len(fragment_input)
            )
        packed_fragments = sorted(
            zip(fragment_inputs, fragment_owners),
            key=lambda item: (
                owner_max_length[item[1][0]],
                owner_order[item[1][0]],
                item[1][1],
            ),
        )
        fragment_inputs = [item[0] for item in packed_fragments]
        fragment_owners = [item[1] for item in packed_fragments]
        fragment_done: defaultdict[str, int] = defaultdict(int)
        fragment_failures: defaultdict[str, set[str]] = defaultdict(set)
        finalized: set[str] = set()
        accepted = 0
        rejected_reasons: defaultdict[str, int] = defaultdict(int)
        rejected_sample: tuple[str, str, str] | None = None

        def finalize_fragment(source: str) -> None:
            nonlocal accepted, rejected_sample
            if source in finalized:
                return
            pieces = pieces_by_source[source]
            raw_candidate = _join_completed_fragments(
                pieces, fragment_failures[source],
            )
            if raw_candidate is None:
                reason = sorted(fragment_failures[source])[0]
                rejected_reasons[reason] += 1
                cache.pop(cache_key(source), None)
                translated[source] = source
                finalized.add(source)
                return
            candidate = _contextualize(raw_candidate, language, source)
            if language == "zh_CN":
                candidate = _simplify_chinese_prose(candidate)
            if candidate == source:
                rejected_reasons["exact"] += 1
                cache.pop(cache_key(source), None)
                translated[source] = source
            elif not _syntax_preserved(source, candidate):
                rejected_reasons["syntax"] += 1
                rejected_sample = rejected_sample or (
                    "syntax", source, candidate
                )
                cache.pop(cache_key(source), None)
                translated[source] = source
            elif _looks_degenerate(source, candidate, language):
                rejected_reasons["degenerate"] += 1
                rejected_sample = rejected_sample or (
                    "degenerate", source, candidate
                )
                cache.pop(cache_key(source), None)
                translated[source] = source
            elif _semantic_false_friends(
                source, candidate, language,
            ):
                rejected_reasons["semantic"] += 1
                rejected_sample = rejected_sample or (
                    "semantic", source, candidate
                )
                cache.pop(cache_key(source), None)
                translated[source] = source
            elif not _has_expected_script(
                source,
                candidate,
                language,
                force=source in forced,
            ):
                rejected_reasons["target_script"] += 1
                rejected_sample = rejected_sample or (
                    "target_script", source, candidate
                )
                cache.pop(cache_key(source), None)
                translated[source] = source
            elif not candidate_valid(source, candidate):
                rejected_reasons["caller_gate"] += 1
                rejected_sample = rejected_sample or (
                    "caller_gate", source, candidate
                )
                cache.pop(cache_key(source), None)
                translated[source] = source
            else:
                translated[source] = candidate
                checkpoint_candidate(source, candidate)
                accepted += 1
            finalized.add(source)

        for source in fragment_sources:
            if fragment_totals[source] == 0:
                finalize_fragment(source)
        if finalized:
            checkpoint_cache()

        # Fragment retries are independent sequences; a modest batch of eight
        # keeps the CPU model busy without raising the worker/thread count or
        # changing beam-search results. The caller can still request less for
        # a memory-constrained host.
        fragment_batch_size = max(1, min(batch_size, 8))
        for start in range(0, len(fragment_inputs), fragment_batch_size):
            batch = fragment_inputs[start:start + fragment_batch_size]
            encoded = encode_batch(batch)
            if device == "cuda":
                encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                output = model.generate(
                    **encoded, max_new_tokens=output_budget(encoded), num_beams=beams,
                    **generation_kwargs,
                )
            decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
            touched: set[str] = set()
            for offset, value in enumerate(decoded):
                owner, piece_index = fragment_owners[start + offset]
                if not completed(output[offset]):
                    fragment_failures[owner].add("eos")
                    value = ""
                original = pieces_by_source[owner][piece_index]
                leading = " " if original[:1].isspace() else ""
                trailing = " " if original[-1:].isspace() else ""
                pieces_by_source[owner][piece_index] = (
                    leading + value.strip() + trailing
                )
                fragment_done[owner] += 1
                touched.add(owner)
            for owner in touched:
                if fragment_done[owner] == fragment_totals[owner]:
                    finalize_fragment(owner)
            checkpoint_cache()
            print(
                f"{language}: fragment pieces="
                f"{min(start + len(batch), len(fragment_inputs))}/"
                f"{len(fragment_inputs)} finalized={len(finalized)}/"
                f"{len(fragment_sources)}",
                flush=True,
            )

        for source in fragment_sources:
            finalize_fragment(source)
        checkpoint_cache()
        print(
            f"{language}: fragment retry accepted={accepted}/"
            f"{len(fragment_sources)} rejected={dict(rejected_reasons)}",
            flush=True,
        )
        if rejected_sample is not None:
            reason, sample_source, sample_candidate = rejected_sample
            missing_literals = Counter(
                re.sub(r"\s+", " ", match.group(0))
                for match in _PROTECT_RE.finditer(sample_source)
            ) - Counter(
                re.sub(r"\s+", " ", match.group(0))
                for match in _PROTECT_RE.finditer(sample_candidate)
            )
            print(
                f"{language}: fragment {reason} sample "
                f"source={sample_source[:600]!r} candidate="
                f"{sample_candidate[:600]!r} missing="
                f"{dict(missing_literals)}",
                flush=True,
            )

    # Primary checkpoints are intentionally conservative around scientific
    # prose. A model can be fluent yet repeatedly drop the last literal or use
    # a technically wrong everyday sense (for example ``stale`` as an
    # impasse). Retry only the still-invalid hard tail with MADLAD-400 when the
    # reviewed local checkpoint is present. This is not a gate bypass: every
    # recomposed paragraph goes back through ``evaluate_raw_candidate``, which
    # includes the caller's API-context contract when one was supplied.
    secondary_path = model_root / SECONDARY_MODEL_FOLDER
    secondary_sources = [] if not (
        allow_secondary_repairs and secondary_path.is_dir()
    ) else _current_invalid_sources(
        all_generated_sources, translated, candidate_valid,
    )
    primary_model_loaded = True
    if secondary_sources:
        del model
        primary_model_loaded = False
        if device == "cuda":
            torch.cuda.empty_cache()

        secondary_tokenizer = AutoTokenizer.from_pretrained(
            secondary_path, local_files_only=True,
        )
        secondary_load_kwargs: dict[str, object] = {
            "local_files_only": True,
        }
        if device == "cuda":
            # Load the 7B checkpoint directly at inference precision. Loading
            # fp32 and converting afterwards needlessly doubles peak host
            # memory and briefly materializes a second 14 GiB parameter copy.
            secondary_load_kwargs["torch_dtype"] = torch.float16
        secondary_model = AutoModelForSeq2SeqLM.from_pretrained(
            secondary_path, **secondary_load_kwargs,
        )
        if device == "cuda":
            secondary_model = secondary_model.to("cuda")
        secondary_model.eval()
        secondary_tag = SECONDARY_LANGUAGE_TAGS[language]

        secondary_inputs: list[str] = []
        secondary_protected: list[str] = []
        secondary_maps: list[dict[str, str]] = []
        secondary_owners: list[tuple[str, int]] = []
        secondary_chunks: dict[str, list[str]] = {}
        for source in secondary_sources:
            chunks = _translation_chunks(source, limit=280)
            secondary_chunks[source] = chunks
            for index, chunk in enumerate(chunks):
                protected, mapping = _protect(
                    chunk, marker_style="numeric",
                )
                secondary_inputs.append(f"<2{secondary_tag}> {protected}")
                secondary_protected.append(protected)
                secondary_maps.append(mapping)
                secondary_owners.append((source, index))

        secondary_values: dict[str, list[list[str | None]]] = {
            source: [[None] * rank_count for _chunk in chunks]
            for source, chunks in secondary_chunks.items()
        }
        secondary_failures: defaultdict[str, set[str]] = defaultdict(set)
        # The 7B checkpoint occupies most of a 24 GiB card in fp16. Two inputs
        # with four ranked beams leave enough headroom for the encoder states
        # of the longest admitted chunks without host offload or OOM retries.
        secondary_batch_size = max(1, min(batch_size, 2))
        for start in range(0, len(secondary_inputs), secondary_batch_size):
            batch = secondary_inputs[start:start + secondary_batch_size]
            encoded = secondary_tokenizer(
                batch, return_tensors="pt", padding=True, truncation=False,
            )
            if device == "cuda":
                encoded = {
                    key: value.to("cuda") for key, value in encoded.items()
                }
            input_width = int(encoded["input_ids"].shape[1])
            with torch.inference_mode():
                output = secondary_model.generate(
                    **encoded,
                    max_new_tokens=min(480, max(64, input_width * 3 + 48)),
                    **ranked_generation,
                    **generation_kwargs,
                )
            decoded = secondary_tokenizer.batch_decode(
                output, skip_special_tokens=True,
            )
            ranked_outputs = _group_ranked_outputs(
                output, decoded, len(batch), rank_count,
            )
            secondary_eos = secondary_tokenizer.eos_token_id
            for offset, candidates in enumerate(ranked_outputs):
                owner, chunk_index = secondary_owners[start + offset]
                for rank, (sequence, value) in enumerate(candidates):
                    if (
                        secondary_eos is not None
                        and int(secondary_eos)
                        not in sequence.detach().cpu().tolist()
                    ):
                        secondary_failures[owner].add("eos")
                        continue
                    try:
                        secondary_values[owner][chunk_index][rank] = _restore(
                            value,
                            secondary_maps[start + offset],
                            protected_text=secondary_protected[start + offset],
                        )
                    except ValueError:
                        secondary_failures[owner].add("marker_restore")
            print(
                f"{language}: MADLAD pieces="
                f"{min(start + len(batch), len(secondary_inputs))}/"
                f"{len(secondary_inputs)}",
                flush=True,
            )

        secondary_accepted = 0
        secondary_rejections: Counter[str] = Counter()
        secondary_rejection_samples: list[tuple[str, str, frozenset[str]]] = []
        for source, chunk_values in secondary_values.items():
            raw_candidates = _rank_aligned_joins(
                chunk_values,
                lambda pieces: " ".join(piece.strip() for piece in pieces),
            )
            # A rank can be rejected for one sentence while being the only
            # beam that preserved a literal in another. Add bounded
            # one-coordinate variants around the first complete combination;
            # never take an unbounded Cartesian product of a long tooltip.
            baseline = [
                next((value for value in values if value is not None), None)
                for values in chunk_values
            ]
            if all(value is not None for value in baseline):
                raw_candidates.append(
                    " ".join(str(value).strip() for value in baseline)
                )
                for chunk_index, values in enumerate(chunk_values):
                    for value in values[1:]:
                        if value is None:
                            continue
                        variant = list(baseline)
                        variant[chunk_index] = value
                        raw_candidates.append(
                            " ".join(str(piece).strip() for piece in variant)
                        )

            candidate = None
            best_rejected: tuple[str, frozenset[str]] | None = None
            seen: set[str] = set()
            for raw_candidate in raw_candidates:
                if raw_candidate is None or raw_candidate in seen:
                    continue
                seen.add(raw_candidate)
                evaluated, failures = evaluate_raw_candidate(
                    source, raw_candidate,
                )
                if not failures:
                    candidate = evaluated
                    break
                secondary_failures[source].update(failures)
                if (
                    best_rejected is None
                    or len(failures) < len(best_rejected[1])
                ):
                    best_rejected = (evaluated, failures)
            if candidate is None:
                translated[source] = source
                cache.pop(cache_key(source), None)
                reasons = frozenset(secondary_failures[source])
                secondary_rejections.update(reasons)
                if best_rejected is not None and len(secondary_rejection_samples) < 8:
                    secondary_rejection_samples.append(
                        (source, best_rejected[0], best_rejected[1])
                    )
                continue
            translated[source] = candidate
            checkpoint_candidate(source, candidate)
            secondary_accepted += 1
        checkpoint_cache()
        print(
            f"{language}: MADLAD retry accepted={secondary_accepted}/"
            f"{len(secondary_sources)} rejected={dict(secondary_rejections)}",
            flush=True,
        )
        for source, candidate, reasons in secondary_rejection_samples:
            print(
                f"{language}: MADLAD rejected reasons={sorted(reasons)} "
                f"source={source[:300]!r} candidate={candidate[:300]!r}",
                flush=True,
            )
        del secondary_model
        if device == "cuda":
            torch.cuda.empty_cache()

    for source, value in tuple(translated.items()):
        if not candidate_valid(source, value):
            translated[source] = source
            cache.pop(cache_key(source), None)

    if primary_model_loaded:
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
        REVIEWED_SOURCE_HASHES as reviewed_source_hashes,
    )
    reviewed = reviewed_module_summaries.get(language, {})
    module_summaries: dict[str, str] = {}
    for key, source in sources["module_summaries"].items():
        reviewed_candidate = reviewed.get(key)
        if reviewed_source_hashes.get(key) != hashlib.sha256(
            str(source).encode("utf-8")
        ).hexdigest():
            reviewed_candidate = None
        candidate = _contextualize(
            reviewed_candidate or translations[source], language, source
        )
        if language == "zh_CN":
            candidate = _simplify_chinese_prose(candidate)
        # Human review controls terminology, but it may never rewrite an API
        # literal or protected product name. Fall back to the structurally
        # validated generated value when an older reviewed summary does so.
        if (not _syntax_preserved(source, candidate)
                or _looks_degenerate(source, candidate, language)
                or _semantic_false_friends(source, candidate, language)):
            candidate = translations[source]
        module_summaries[key] = candidate
    if language == "zh_CN":
        non_normalized = [
            name for name, table in (
                ("SETTING_LABELS", setting_labels),
                ("SETTING_TOOLTIPS", setting_tooltips),
                ("CATEGORY_HELP", categories),
                ("UI", ui),
                ("MODULE_SUMMARIES", module_summaries),
            )
            if any(_has_traditional_chinese_prose(value)
                   for value in table.values())
        ]
        if non_normalized:
            raise ValueError(
                "zh_CN runtime output is not OpenCC t2s-normalized: "
                + ", ".join(non_normalized)
            )
    path = CATALOG_DIR / f"{language}.py"
    text = (
        f'"""spaCR localization catalog for {language}.\n\n'
        f"Drafted with {model_id} ({license_name}) and corrected by spaCR's "
        f"technical-context review. Rejected tails may use {SECONDARY_MODEL} "
        f"({SECONDARY_LICENSE}). Generated by tools/build_i18n_catalogs.py.\n"
        '"""\n\n'
        + f'MODEL = {model_id!r}\nLICENSE = {license_name!r}\n'
        + f'SECONDARY_MODEL = {SECONDARY_MODEL!r}\n'
        + f'SECONDARY_LICENSE = {SECONDARY_LICENSE!r}\n'
        + ('NORMALIZER = "OpenCC 1.1+ t2s"\n'
           if language == "zh_CN" else '')
        + "\n"
        + _render_assignment("SETTING_LABELS", setting_labels)
        + "\n"
        + _render_assignment("SETTING_TOOLTIPS", setting_tooltips)
        + "\n"
        + _render_assignment("CATEGORY_HELP", categories)
        + "\n"
        + _render_assignment("UI", ui)
        + "\n"
        + _render_assignment("MODULE_SUMMARIES", module_summaries)
        + "\n"
        + _render_assignment("SOURCE_HASHES", _source_hashes(sources))
    )
    _atomic_write_text(path, text)
    return path


def audit(sources: Mapping[str, object], languages: Iterable[str]) -> int:
    """Validate key coverage, source freshness and basic translation safety."""
    from string import Formatter
    from types import SimpleNamespace

    languages = tuple(languages)
    if "zh_CN" in languages:
        # Dependency availability is part of the zh_CN release contract, not
        # contingent on whether a particular catalog happens to contain a
        # Traditional character today.
        _simplify_chinese_prose("")
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
    expected_hashes = _source_hashes(sources)

    english_path = CATALOG_DIR / "en.py"
    try:
        english_namespace: dict[str, object] = {}
        exec(
            compile(
                english_path.read_text(encoding="utf-8"),
                str(english_path),
                "exec",
            ),
            english_namespace,
            english_namespace,
        )
    except FileNotFoundError:
        failures.append("en: canonical runtime catalog is missing")
    else:
        english_contract = {
            "SETTING_LABELS": sources["setting_labels"],
            "SETTING_TOOLTIPS": sources["setting_tooltips"],
            "CATEGORY_SOURCES": frozenset(sources["categories"]),
            "UI_SOURCES": frozenset(sources["ui"]),
            "MODULE_SUMMARIES": sources["module_summaries"],
            "SOURCE_HASHES": expected_hashes,
        }
        for name, expected_value in english_contract.items():
            if english_namespace.get(name) != expected_value:
                failures.append(f"en/{name}: canonical source catalog is stale")
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
        if getattr(module, "SECONDARY_MODEL", None) != SECONDARY_MODEL:
            failures.append(
                f"{language}: secondary translation model provenance is stale"
            )
        if getattr(module, "SECONDARY_LICENSE", None) != SECONDARY_LICENSE:
            failures.append(
                f"{language}: secondary translation license provenance is stale"
            )
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
            "CATEGORY_HELP": {
                source: source for source in sources["categories"]
            },
            "UI": {source: source for source in sources["ui"]},
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
            contextual_errors = [
                key for key, value in table.items()
                if _contextualize(
                    str(value), language,
                    str(source_tables[name].get(key, key)),
                ) != str(value)
            ]
            if contextual_errors:
                failures.append(
                    f"{language}/{name}: {len(contextual_errors)} unresolved "
                    "contextual false friends "
                    f"({', '.join(map(str, contextual_errors[:5]))})"
                )
            semantic_errors = [
                key for key, value in table.items()
                if _semantic_false_friends(
                    str(source_tables[name].get(key, key)),
                    str(value),
                    language,
                )
            ]
            if semantic_errors:
                failures.append(
                    f"{language}/{name}: {len(semantic_errors)} semantic "
                    "false friends "
                    f"({', '.join(map(str, semantic_errors[:5]))})"
                )
            syntax_errors = [
                key for key, value in table.items()
                if not _syntax_preserved_or_reviewed(
                    str(source_tables[name].get(key, key)),
                    str(value),
                    language,
                )
            ]
            if syntax_errors:
                failures.append(
                    f"{language}/{name}: {len(syntax_errors)} protected "
                    "literal or markup failures "
                    f"({', '.join(map(str, syntax_errors[:5]))})"
                )
        localized_hashes = getattr(module, "SOURCE_HASHES", {})
        if localized_hashes != expected_hashes:
            missing_hashes = set(expected_hashes) - set(localized_hashes)
            stale_hashes = set(localized_hashes) - set(expected_hashes)
            wrong_hashes = {
                key for key in set(expected_hashes) & set(localized_hashes)
                if localized_hashes[key] != expected_hashes[key]
            }
            failures.append(
                f"{language}/SOURCE_HASHES: missing={len(missing_hashes)} "
                f"stale={len(stale_hashes)} wrong={len(wrong_hashes)}"
            )
        if language == "zh_CN":
            if getattr(module, "NORMALIZER", None) != "OpenCC 1.1+ t2s":
                failures.append(
                    "zh_CN/NORMALIZER: expected OpenCC 1.1+ t2s provenance"
                )
            non_normalized = [
                f"{name}/{key}"
                for name in tables
                for key, value in getattr(module, name, {}).items()
                if _has_traditional_chinese_prose(str(value))
            ]
            if non_normalized:
                failures.append(
                    "zh_CN: OpenCC t2s normalization is not at a fixed point "
                    f"({', '.join(non_normalized[:5])})"
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
            reviewed = _reviewed_translation(str(source), language)
            if reviewed is not None and module.SETTING_LABELS.get(key) != reviewed:
                failures.append(
                    f"{language}/label/{key}: reviewed translation changed"
                )
        for source in sources["ui"]:
            reviewed = _reviewed_translation(str(source), language)
            if reviewed is not None and module.UI.get(source) != reviewed:
                failures.append(
                    f"{language}/ui/{source!r}: reviewed translation changed"
                )
        unchanged_tips = [
            key for key, source in sources["setting_tooltips"].items()
            if module.SETTING_TOOLTIPS.get(key) == source
            and _looks_translatable(source)
        ]
        unchanged_ui = sum(
            module.UI.get(source) == source for source in sources["ui"]
        )
        if unchanged_tips:
            failures.append(
                f"{language}: {len(unchanged_tips)} tooltip bodies remain "
                "exact English "
                f"({', '.join(unchanged_tips[:5])})"
            )
        if unchanged_ui > max(25, len(expected_ui) // 6):
            failures.append(
                f"{language}: {unchanged_ui} static UI strings remain English"
            )
        if language in script_pattern:
            missing_script = [
                key
                for key, source in sources["setting_tooltips"].items()
                if (
                len(source) >= 40
                and bool(re.search(r"[A-Za-z]{4}", source))
                and not script_pattern[language].search(
                    str(module.SETTING_TOOLTIPS.get(key, ""))
                )
                )
            ]
            if missing_script:
                failures.append(
                    f"{language}: {len(missing_script)} prose tooltips lack "
                    "target script "
                    f"({', '.join(missing_script[:5])})"
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
    parser.add_argument(
        "--repair-untranslated",
        action="store_true",
        help=(
            "re-decode exact-English tooltip prose and safely retry entries "
            "whose protected literals were damaged"
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--beams", type=int, default=4)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    sources = canonical_sources()
    if args.audit:
        return audit(sources, args.languages)

    path = write_english(sources)
    print(
        f"wrote {path}: settings={len(sources['setting_tooltips'])} "
        f"categories={len(sources['categories'])} ui={len(sources['ui'])}"
    )
    if args.sources_only:
        return 0
    values = _unique_translation_sources(sources)
    for language in args.languages:
        forced: set[str] = set()
        if args.repair_untranslated:
            namespace: dict[str, object] = {}
            catalog_path = CATALOG_DIR / f"{language}.py"
            try:
                exec(
                    compile(
                        catalog_path.read_text(encoding="utf-8"),
                        str(catalog_path),
                        "exec",
                    ),
                    namespace,
                    namespace,
                )
            except FileNotFoundError:
                pass
            current = namespace.get("SETTING_TOOLTIPS", {})
            if isinstance(current, dict):
                forced.update(
                    source
                    for key, source in sources["setting_tooltips"].items()
                    if (
                        not isinstance(current.get(key), str)
                        or (
                            current.get(key) == source
                            and _looks_translatable(source)
                        )
                        or _contextualize(
                            str(current.get(key, "")), language, source,
                        ) != str(current.get(key, ""))
                        or not _translation_candidate_valid(
                            source, str(current.get(key, "")), language,
                        )
                    )
                )
            for table_name, source_name in (
                ("CATEGORY_HELP", "categories"),
                ("UI", "ui"),
            ):
                current_table = namespace.get(table_name, {})
                if not isinstance(current_table, dict):
                    continue
                source_values = sources[source_name]
                iterable = (
                    source_values.values()
                    if isinstance(source_values, dict)
                    else source_values
                )
                forced.update(
                    source
                    for source in iterable
                    if (
                        not isinstance(current_table.get(source), str)
                        or (
                            current_table.get(source) == source
                            and _looks_translatable(source)
                        )
                        or _contextualize(
                            str(current_table.get(source, "")), language, source,
                        ) != str(current_table.get(source, ""))
                        or not _translation_candidate_valid(
                            source, str(current_table.get(source, "")), language,
                        )
                    )
                )
            print(
                f"{language}: strict runtime repairs={len(forced)}",
                flush=True,
            )
        translations = _translate_batches(
            values,
            language,
            args.model_root,
            device=args.device,
            batch_size=args.batch_size,
            beams=args.beams,
            threads=args.threads,
            force_sources=forced,
            repair_protected=args.repair_untranslated,
        )
        print(f"wrote {write_language(language, sources, translations)}")
    return audit(sources, args.languages)


if __name__ == "__main__":
    raise SystemExit(main())
