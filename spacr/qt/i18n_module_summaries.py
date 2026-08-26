"""Localized one-line descriptions for spaCR's built-in modules.

The stable application key is the lookup key.  Keeping these longer strings
outside the compact UI catalog makes them reviewable by fluent speakers and
prevents term-by-term translation from corrupting scientific descriptions.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from .i18n_module_summaries_asia import MODULE_SUMMARIES_ASIA
from .i18n_module_summaries_other import MODULE_SUMMARIES_OTHER
from .i18n_module_summaries_west import MODULE_SUMMARIES_WEST

MODULE_SUMMARIES = {
    **MODULE_SUMMARIES_WEST,
    **MODULE_SUMMARIES_ASIA,
    **MODULE_SUMMARIES_OTHER,
}

# Hand-reviewed prose is still source-coupled data.  These hashes bind each
# reviewed row to the exact English summary it was reviewed against, just as
# the generated external catalogs do.  A changed app description therefore
# falls through to the current hashed external catalog (or safe English),
# instead of silently displaying an obsolete but fluent translation.
REVIEWED_SOURCE_HASHES = {
    "activation": "d29ba43fb6c1bfe7109a62a46cb3049ae2704dc8855389152020e392590cdc38",
    "agreement": "b38ac3b980a7dea533aa8344ced708bb68a1fc849539cc4619cdcc8e61ac2021",
    "align": "b81441d62e266e898eb510847176135ee5b9b6720e5a248d5ffc556dc24de178",
    "analyze_plaques": "1c87341fe1da7e68fbd2d88693ec4878be9b6d138b06bbc25d1a71332ad2c4c7",
    "annotate": "031c7bced143ad7be70d54c90fd284d08127f97678eaad53ef5781869c7a05f3",
    "batch": "5445eee74a5df593ac930672f0a130bc4f5e19e8c0bb6ed7099058daf935573a",
    "classifier_evaluation": "3f2f24d0bf059a02cfcfba5ba72e3c514a538c901116b54f916d7d80ebc89d50",
    "classify_merged": "2f8134245b0283f90d289cbb37bf064954caf96616efe56c749d6bff0017304d",
    "convert": "f3e27b0c3482434489e53c71b12416f2c27b93fdc8599cf24a71a1b89add3587",
    "db_browser": "863aea17872fc9936587ae8f447ee4f4f0b4c39a835b6b7b3f3beb100642d0f1",
    "distributed_jobs": "af29f8dde164cdad47acd90526c375baa194dea7d0e52c122d47a856280b18dc",
    "external_masks": "a3a7fb7a1a041e68b61fba5b5a688a82056306b35304d525179b4a8105dab099",
    "foreign": "df519d7cdc6aa6ca50207f7c1e102816c6510fa7620e48991b599a35134ccfb0",
    "invasion": "d0091f7df6a00aa4f706d6f0f85c9c970f8b926d99a1253379cf86b584d23c78",
    "map_barcodes": "17871b5accd848e33df1ee24fc94265a24fd5b3a55fbc0b42a8189fe9474871f",
    "mask": "82dfbe390074e1a296eaf57ea37efdbb140af0eb47028693d757d9d6f630ef23",
    "measure": "4422757292446213841ddf242414ed4c2cb026d8fa9b6869fdead8173d82ac73",
    "model_compare": "265f341fa32367ff8aa89aaefab16dbae5688bdb958cf064eecc62c113bfb789",
    "model_zoo": "5eee2dc596590bf7c16c9487b5f66e318a0f83bf930d1404d21baf5dfd2a5bb1",
    "motility": "b4fbf209e1f47994434be67f52fc004fb22f497e32496c3e57848d5c30848f46",
    "plate_view": "23f3bdecd32237296126b5a804bf7c10a621c59959ace50e8de3bf2895c82e3c",
    "queue": "35c1defdd9dab03b436cb9eb7c74835c6ff28478da27230de3d4ebdd8bc49ff9",
    "recruitment": "72650f131c1c3ca1a9b3ef6c0eef6e7a8ed00b0d0c94902c5255d9c14c3fc015",
    "regression": "149811406f3e2c9d0b7647d52c9e0a530538fc49c51cc773cd222483c8c80dd0",
    "replication": "1232ca3d167887314bd244142c95f2b21b2e75b0ff6098d8da9418312f331d8b",
    "report": "7be5f8e147860714cde704e80b1e8cffddf39745827d4445ab85e868bd60c664",
    "run_history": "865d24464019c54890e14c557e9816713a49db63927c3a4d39e518b1d461b852",
    "timelapse": "e3816a2eabfc60e1ad2b404ec2fbf892d87d89a575524d713dcb635f0f089e49",
    "train_compare": "27759389a3127fa4fffaa6319b6aba44ce479ee820e3140ecef4a13d47ea5da6",
    "umap": "36befbbf466da529a3698793648819bb93689e23def5086d1bee8074b10c1e53",
}


def module_summary(
    app_key: str,
    english: str,
    language: Optional[str] = None,
) -> str:
    """Return a reviewed module summary, falling back to ``english``.

    Plugin modules and future built-ins therefore remain readable until their
    own translation catalog supplies an exact description.
    """
    from .i18n import _exact_translation, current_language, normalize_language

    code = normalize_language(language or current_language())
    if code == "en":
        return str(english)
    key = str(app_key)
    reviewed = MODULE_SUMMARIES.get(code, {}).get(key)
    expected = REVIEWED_SOURCE_HASHES.get(key)
    current = hashlib.sha256(str(english).encode("utf-8")).hexdigest()
    if reviewed and expected == current:
        return reviewed
    try:
        from .i18n_catalogs import module_summary as external_summary
        translated = external_summary(app_key, english, code)
        if translated:
            return translated
    except (ImportError, AttributeError):
        pass
    # Plugins may ship exact translations in their manifest.  Do not apply
    # conservative term substitution to a scientific paragraph: either the
    # plugin supplies the whole sentence or it stays canonical English.
    return _exact_translation(str(english), code) or str(english)


def validate_module_summaries() -> None:
    """Raise if the nine non-English catalogs drift out of alignment."""
    expected_languages = {
        "sv", "de", "es", "zh_CN", "pt", "hi", "ko", "is", "fr",
    }
    assert set(MODULE_SUMMARIES) == expected_languages
    key_sets = {frozenset(items) for items in MODULE_SUMMARIES.values()}
    assert len(key_sets) == 1
    reviewed_keys = next(iter(key_sets))
    assert len(reviewed_keys) == 30
    assert set(REVIEWED_SOURCE_HASHES) == set(reviewed_keys)


validate_module_summaries()


__all__ = ["MODULE_SUMMARIES", "REVIEWED_SOURCE_HASHES", "module_summary"]
