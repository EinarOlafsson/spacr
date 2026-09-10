#!/usr/bin/env python3
"""Canonical display-to-speech substitutions for tutorial narration.

Captions always retain the official spellings ``spaCR`` and ``PyPI``. TTS
engines receive reviewed spoken forms instead. The English ``spaCR`` form
follows the user's reference recording ("spacer"). English ``PyPI`` is the
single syllable "pype", with the vowel and final consonant of "pipe". It is
never ``PyPy``, ``pypie``, a sequence of letters, or two paused syllables.
"""
from __future__ import annotations

import re


PRONUNCIATION_VERSION = "2026-08-28-pype-v11"


BRAND_SPEECH = {
    "en": "spacer",
    "es": "espáquer",
    "fr": "spakeur",
    "hi": "स्पेकर",
    "it": "spàker",
    "pt-BR": "espáquer",
    "ja": "スペーカー",
    "zh-CN": "斯佩克尔",
    # Caption-first languages.  These forms are ready for a future native
    # narrator and keep the target /speɪsər/ sound explicit.
    "de": "Späiser",
    "sv": "spejser",
    "is": "speiser",
    "nb": "speiser",
    "ko": "스페이서",
    "da": "spejser",
}

PYPI_SPEECH = {
    "en": "[pype](/pˈIp/)",
    "es": "paip",
    "fr": "païpe",
    "hi": "पाइप",
    "it": "paip",
    "pt-BR": "paip",
    "ja": "パイプ",
    "zh-CN": "派普",
    "de": "Peip",
    "sv": "pajp",
    "is": "pæp",
    "nb": "paip",
    "ko": "파이프",
    "da": "pajp",
}

QT_SPEECH = {
    "en": "Q T",
    "es": "cu te",
    "fr": "ku té",
    "hi": "क्यू टी",
    "it": "cu ti",
    "pt-BR": "quê tê",
    "ja": "キュー ティー",
    "zh-CN": "Q T",
    "de": "ku te",
    "sv": "ku te",
    "is": "kú té",
    "nb": "ku te",
    "ko": "큐 티",
    "da": "ku te",
}

if set(PYPI_SPEECH) != set(BRAND_SPEECH) or set(QT_SPEECH) != set(BRAND_SPEECH):
    raise RuntimeError(
        "Every tutorial language must define reviewed PyPI and Qt speech"
    )


_DISPLAY_PYPI = re.compile(r"(?<!\w)PyPI(?!\w)")
_APPROVED_PYPI_FORMS = tuple(PYPI_SPEECH.values())
_PYPI_ALIAS_SEPARATOR_CHAR = r"[\s,./_—–-]"
_PYPI_ALIAS_SEPARATOR = rf"{_PYPI_ALIAS_SEPARATOR_CHAR}+"
_REJECTED_PYPI_ALIAS = re.compile(
    rf"(?<!\w)(?:(?:pypi|pypie)|"
    rf"(?:pie|pai|paj|pæ|paï){_PYPI_ALIAS_SEPARATOR}p"
    rf"(?:{_PYPI_ALIAS_SEPARATOR_CHAR}*[ieí])?|"
    rf"p{_PYPI_ALIAS_SEPARATOR}y{_PYPI_ALIAS_SEPARATOR}p"
    rf"{_PYPI_ALIAS_SEPARATOR}[ie])(?!\w)",
    re.IGNORECASE,
)


# These are Misaki phonemes rather than general-purpose IPA. Keeping the whole
# inflected word in one override prevents a second artificial stress and, for
# classifier(s), preserves the complete ending instead of clipping it.
ENGLISH_DIALECT_SPEECH = {
    "us": {
        "github": "/ɡˈɪthˌʌb/",
        "xgboost": "/ˌɛksʤˌibˈust/",
        "pytorch": "/pˈItˌɔɹʧ/",
        "cellpose": "/sˈɛlpˌOz/",
        "macos": "/mˈækOˌɛs/",
        "t-sne": "/tˌi snˈi/",
        "napari": "/nəpˈɑɹi/",
        "slurm": "/slˈɜɹm/",
        "unnotarized": "/ʌnnˈOTəɹIzd/",
        "untruncated": "/ʌntɹˈʌŋkATᵻd/",
        "rerunnable": "/ɹˌiɹˈʌnəbəl/",
        "multi-rater": "/mˌʌlTiɹˈATəɹ/",
        "auditable": "/ˈɔdɪTəbəl/",
        "rotatable": "/ɹˈOtATəbəl/",
        "endoplasmic": "/ˌɛndOplˈæzmɪk/",
        "assay": "/ˈæsA/",
        "assays": "/ˈæsAz/",
        "classify": "/klˈæsəfI/",
        "classifier": "/klˈæsəfIəɹ/",
        "classifiers": "/klˈæsəfIəɹz/",
    },
    "uk": {
        "github": "/ɡˈɪthˌʌb/",
        "xgboost": "/ˌɛksʤˌiːbˈuːst/",
        "pytorch": "/pˈItˌɔːʧ/",
        "cellpose": "/sˈɛlpˌQz/",
        "macos": "/mˈakQˌɛs/",
        "z": "/zˈɛd/",
        "t-sne": "/tˌiː snˈiː/",
        "napari": "/nəpˈɑːɹi/",
        "slurm": "/slˈɜːm/",
        "unnotarized": "/ʌnnˈQtəɹIzd/",
        "untruncated": "/ʌntɹʌŋkˈAtɪd/",
        "rerunnable": "/ɹˌiːɹˈʌnəbᵊl/",
        "multi-rater": "/mˌʌltiɹˈAtə/",
        "auditable": "/ˈɔːdɪtəbᵊl/",
        "rotatable": "/ɹQtˈAtəbᵊl/",
        "endoplasmic": "/ˌɛndQplˈazmɪk/",
        "assay": "/əsˈA/",
        "assays": "/əsˈAz/",
        "classify": "/klˈasɪfI/",
        "classifier": "/klˈasɪfIə/",
        "classifiers": "/klˈasɪfIəz/",
    },
}


# Kokoro's non-English phonemizers do not reliably expand Latin UI labels or
# scientific abbreviations. Captions keep the exact interface spelling; these
# substitutions are used only for synthesis.
LOCALIZED_TECHNICAL_SPEECH = {
    "es": (
        ("Gate Editor", "gueit éditor"), ("Box gate", "box gueit"),
        ("measurements.db", "measurements punto de be"),
        ("Cellpose", "sel póus"), ("t-SNE", "te ese ene e"),
        ("UMAP", "ú map"), ("PC1", "pe ce uno"),
        ("PC2", "pe ce dos"), ("PC3", "pe ce tres"),
        ("2D", "dos de"), ("3D", "tres de"), ("xD", "equis de"),
    ),
    "fr": (
        ("Gate Editor", "guéïte éditeur"), ("Box gate", "box guéïte"),
        ("measurements.db", "measurements point dé bé"),
        ("Cellpose", "celle pose"), ("t-SNE", "té esse enne e"),
        ("UMAP", "you map"), ("PC1", "pé cé un"),
        ("PC2", "pé cé deux"), ("PC3", "pé cé trois"),
        ("2D", "deux dé"), ("3D", "trois dé"), ("xD", "ixe dé"),
    ),
    "hi": (
        ("Gate Editor", "गेट एडिटर"), ("Box gate", "बॉक्स गेट"),
        ("measurements.db", "मेज़रमेंट्स डॉट डी बी"),
        ("Cellpose", "सेल पोज़"), ("t-SNE", "टी एस एन ई"),
        ("UMAP", "यू मैप"), ("PC1", "पी सी वन"),
        ("PC2", "पी सी टू"), ("PC3", "पी सी थ्री"),
        ("2D", "टू डी"), ("3D", "थ्री डी"), ("xD", "एक्स डी"),
    ),
    "it": (
        ("Gate Editor", "gheit èditor"), ("Box gate", "box gheit"),
        ("measurements.db", "measurements punto di bi"),
        ("Cellpose", "sel pòus"), ("t-SNE", "ti esse enne i"),
        ("UMAP", "iu map"), ("PC1", "pi ci uno"),
        ("PC2", "pi ci due"), ("PC3", "pi ci tre"),
        ("2D", "due di"), ("3D", "tre di"), ("xD", "ics di"),
    ),
    "pt-BR": (
        ("Gate Editor", "gueit éditor"), ("Box gate", "bóks gueit"),
        ("measurements.db", "measurements ponto dê bê"),
        ("Cellpose", "cél pouz"), ("t-SNE", "tê ésse ene ê"),
        ("UMAP", "iu mép"), ("PC1", "pê cê um"),
        ("PC2", "pê cê dois"), ("PC3", "pê cê três"),
        ("2D", "dois dê"), ("3D", "três dê"), ("xD", "xis dê"),
    ),
    "ja": (
        ("Gate Editor", "ゲートエディタ"), ("Box gate", "ボックスゲート"),
        ("measurements.db", "メジャメンツ・ドット・ディービー"),
        ("Cellpose", "セルポーズ"), ("t-SNE", "ティーエスエヌイー"),
        ("UMAP", "ユーマップ"), ("PC1", "ピーシーワン"),
        ("PC2", "ピーシーツー"), ("PC3", "ピーシースリー"),
        ("2D", "ツーディー"), ("3D", "スリーディー"),
        ("xD", "エックスディー"), ("Home", "ホーム"),
        ("Alpha", "アルファ"), ("Beta", "ベータ"),
        ("Stable", "ステーブル"),
    ),
    "zh-CN": (
        ("Gate Editor", "门控编辑器"), ("Box gate", "箱式门"),
        ("measurements.db", "测量数据库文件"),
        ("Cellpose", "塞尔波斯"), ("t-SNE", "提艾斯恩伊"),
        ("UMAP", "优麦普"), ("PC1", "主成分一"),
        ("PC2", "主成分二"), ("PC3", "主成分三"),
        ("2D", "二维"), ("3D", "三维"), ("xD", "艾克斯迪"),
        ("Home", "主页"), ("Alpha", "阿尔法"),
        ("Beta", "贝塔"), ("Stable", "稳定"),
    ),
}


def normalize_display_terms(text: str) -> str:
    """Return stable product spelling without changing surrounding prose."""
    substitutions = (
        (r"(?i)spa\s*[- ]?c\s*r", "spaCR"),
        (r"(?i)spacr", "spaCR"),
        (r"(?i)py\s*pi", "PyPI"),
        (r"(?i)cell\s*pose", "Cellpose"),
        (r"(?i)xg\s*boost", "XGBoost"),
    )
    for pattern, replacement in substitutions:
        text = re.sub(pattern, replacement, text)
    return text


def spoken_form(
    text: str,
    language: str,
    dialect: str = "us",
) -> str:
    """Return the display text rewritten only for TTS pronunciation.

    ``dialect`` selects the American (``"us"``, the default) or British
    (``"uk"``) English overrides. Cadence and complete word endings remain
    the synthesizer's responsibility.
    """
    if language not in BRAND_SPEECH or language not in PYPI_SPEECH:
        raise ValueError(f"No tutorial pronunciation profile for {language!r}")
    dialect = dialect.lower()
    if dialect not in ENGLISH_DIALECT_SPEECH:
        raise ValueError(f"Unsupported English tutorial dialect {dialect!r}")
    speech = normalize_display_terms(text)
    speech = speech.replace("spaCR", BRAND_SPEECH[language])
    speech = speech.replace("PyPI", PYPI_SPEECH[language])
    # A hyphen or the channel qualifier can make speech engines either fuse
    # the words or insert an unnatural pause. Captions keep the exact package
    # spelling and command; synthesis receives exactly two words separated by
    # one ordinary space.
    speech = re.sub(r"(?i:\bconda-forge\b)", "Conda Forge", speech)
    speech = speech.replace("::", " ")
    # Qt names two letters. It is never pronounced "cute" in spaCR narration.
    speech = re.sub(r"(?i:\bqt\b)", QT_SPEECH[language], speech)
    speech = speech.replace("LoG", "L O G")
    speech = speech.replace("CQ1", "C Q one" if language == "en" else "C Q 1")
    if language == "en":
        speech = speech.replace("measurements.db", "measurements dot db")
    for display, spoken in LOCALIZED_TECHNICAL_SPEECH.get(language, ()):
        speech = speech.replace(display, spoken)
    if language == "en":
        # CPU and NVIDIA deliberately remain as literal, single tokens.
        # Kokoro's English lexicon supplies /ˌsiː.piːˈjuː/ and /ɛnˈvɪdiə/
        # directly. Splitting either spelling into space-separated fragments
        # gives every fragment its own stress and makes the narration halting.
        technical_terms = (
            (r"\bCPU-compatible\b", "CPU compatible"),
            (r"\bPySide\b", "pie side"),
            (r"\bSQLite\b", "S Q lite"),
            (r"\bSQL\b", "S Q L"),
            (r"\bFASTQ\b", "fast cue"),
            (r"\bUMAP\b", "you map"),
            (r"\bDBSCAN\b", "D B scan"),
            (r"\bUltrack\b", "ull track"),
            (r"\bHeLa\b", "hee lah"),
            (r"\bsiRNA\b", "sigh R N A"),
            (r"\bYokogawa\b", "yoh koh gah wah"),
            (r"\bOOF\b", "O O F"),
            (r"\bCPSAM\b", "C P sam"),
            (r"\bGrad[- ]CAM\b", "grad cam"),
            # CUDA's all-caps spelling is read as four letters. Misaki's
            # one-token phoneme override preserves the official /KOO-duh/
            # pronunciation while the reduced final schwa avoids the long,
            # separately stressed second syllable produced by ``koo duh``.
            (r"\bCUDA\b", "[CUDA](/kˈuːdᵊ/)"),
            (r"\bVRAM\b", "V ram"),
            (r"\bJSON\b", "jay son"),
            (r"\bAnnData\b", "Anne data"),
            (r"\bscanpy\b", "scan pie"),
            (r"\bscvi-tools\b", "S C V I tools"),
            (r"\bh5ad\b", "H five A D"),
            (r"\bEC50\b", "E C fifty"),
            (r"\bgRNA\b", "guide R N A"),
        )
        for pattern, replacement in technical_terms:
            speech = re.sub(pattern, replacement, speech)
        word_boundaries = (
            (r"\btimelapse\b", "time lapse"),
            (r"\bbackends\b", "back ends"),
            (r"\bbackend\b", "back end"),
            (r"\bdenoising\b", "de-noising"),
            (r"\boverinterpreted\b", "over-interpreted"),
            (r"\boverfitting\b", "over-fitting"),
            (r"\bheatmaps\b", "heat maps"),
            (r"\bheatmap\b", "heat map"),
            (r"\bqueried\b", "queryd"),
            (r"\brebuildability\b", "rebuild-ability"),
            (r"\bcopied\b", "copyd"),
            (r"\bgrayscale\b", "gray scale"),
            (r"\bdb\b", "D B"),
            (r"\bpreprocessing\b", "pre-processing"),
            (r"\bpretrained\b", "pre-trained"),
            (r"\bhyperparameters\b", "hyper-parameters"),
            (r"\bhyperparameter\b", "hyper-parameter"),
            (r"\bcolocalization\b", "co-localization"),
        )
        for pattern, replacement in word_boundaries:
            speech = re.sub(pattern, replacement, speech, flags=re.IGNORECASE)
        speech = re.sub(
            r"\b(?:"
            + "|".join(
                re.escape(word)
                for word in sorted(
                    ENGLISH_DIALECT_SPEECH[dialect], key=len, reverse=True
                )
            )
            + r")\b",
            lambda match: (
                f"[{match.group(0)}]("
                f"{ENGLISH_DIALECT_SPEECH[dialect][match.group(0).lower()]})"
            ),
            speech,
            flags=re.IGNORECASE,
        )
        speech = re.sub(
            r"\bPower\s*/\s*Design\b",
            "Power and Design",
            speech,
            flags=re.IGNORECASE,
        )
        speech = speech.replace("python -m pip", "python dash m pip")
        speech = speech.replace("--upgrade", "dash dash upgrade")
        # ``backend`` is deliberately resegmented to ``back end`` above, so
        # accept either form when verbalizing the complete command-line flag.
        speech = re.sub(
            r"--torch[- ]back(?:-|\s+)end\b",
            "dash dash torch back end",
            speech,
            flags=re.IGNORECASE,
        )
        speech = speech.replace("SHA-256", "S H A two fifty six")
        speech = speech.replace("install.log", "install dot log")
        speech = speech.replace("spacer-run", "spacer run")
        speech = speech.replace("spacer-doctor", "spacer doctor")
    return speech


def assert_pronunciation_safe(display_text: str, speech_text: str) -> None:
    """Fail when a brand spelling accidentally leaks into synthesized text."""
    if "spaCR" in speech_text or "PyPI" in speech_text:
        raise ValueError(
            "Literal brand spelling reached TTS; use spoken_form() so "
            "spaCR and PyPI retain their approved pronunciations"
        )
    normalized_display = normalize_display_terms(display_text)
    if "spaCR" in normalized_display and not speech_text:
        raise ValueError("spaCR narration produced empty synthesized text")
    display_count = len(_DISPLAY_PYPI.findall(normalized_display))
    if display_count:
        exact_count = sum(
            speech_text.count(form) for form in set(_APPROVED_PYPI_FORMS)
        )
        if exact_count != display_count:
            raise ValueError(
                "Every displayed PyPI occurrence must use exactly one "
                "reviewed, single-syllable pype spoken form"
            )
        if _REJECTED_PYPI_ALIAS.search(speech_text):
            raise ValueError(
                "PyPI narration must not contain PyPy, pypie, literal PyPI, "
                "or a "
                "split, paused, hyphenated, or letter-by-letter alias"
            )
