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
    "activation": "a167170eb9e27e10d7c5700d252eb44479f6ccd9b78e835863e91754a5547a3d",
    "agreement": "6639088d05b49ab7397a325a1531538bbc2c8d3f19976dcaa4e0ba00c504d9a3",
    "align": "329c3b6981e3522986fe2b76eb7e9fe0666b7535a1b320d8c9dfbabbbbec4e51",
    "analyze_plaques": "b9b1592b4d0c236b489e3a9b91ba81e0ac7f59ba06e774ba7fa85fbcc323fde4",
    "annotate": "6df605da96fb2ea381ed2c1b135bd8bba2259fabea58640d02aa6e893890c49c",
    "batch": "16f1fdd21fc4467e4ff8f5b2996096fbcf0e5c656263e374eeebe179e94c5cd3",
    "cellpose_masks": "07d8ffd08b802c9921bc7f1224f805f311a86ca911a6ca97a3664bd6b31fed55",
    "classifier_evaluation": "3f2f24d0bf059a02cfcfba5ba72e3c514a538c901116b54f916d7d80ebc89d50",
    "classify": "64bc454c2acf4f71e86ede1feaa403915b4e600c88cc19bda35590679eb9c9f7",
    "convert": "692de961836cf4a2badc716d9bf06d4c61d01693facd1f25e10e752f836a03d7",
    "db_browser": "049241d8dc8b71dc7704359553d2438395c7b1b9e651038039dcdd39d78bb7b6",
    "distributed_jobs": "af29f8dde164cdad47acd90526c375baa194dea7d0e52c122d47a856280b18dc",
    "external_masks": "f5b01c2fcde66a99d768460bae3932036fbe64a2054077fdb8463ff2125ed7b1",
    "foreign": "e80a74db7728631e85022d4e08a7cb449107ec397d135e872dbe8be37915cd2a",
    "invasion": "0b3ef16644bd8aab8c4f1d356dada8f8c1d7a8d09a157b2eda6b3fbe8913534f",
    "map_barcodes": "17871b5accd848e33df1ee24fc94265a24fd5b3a55fbc0b42a8189fe9474871f",
    "mask": "b85d38a6265bf63680d1fc95ac7e693275643b38c090bfda7b7c2152321668cb",
    "measure": "a6b12c7d9673bcb9b2bdbfbffabf6644a11c8d900b8566ebfa2227a9f8dcf5c1",
    "ml_analyze": "4da05e4bee0b369105382c4869cf789a2053e9184906a0f3c3572e4a5068efdb",
    "model_compare": "a119662250fc95faa7c7ee1b33e90d0deb1d43f0d92c94af605a1a732d6eb6cc",
    "model_zoo": "d1ace1a31bf95abac5b31fba3d562480f8a81f989a28f1b9949a229d1e9532b2",
    "motility": "f76b7701d1122fb1405912873b4d98caabab8c91fc65652a89c69220f0ecbb54",
    "plate_view": "392c438f3755de7b7ce786d254e7504c7a5c5c9b1497b4cb37473b0253ac45d3",
    "queue": "41f1164a8dd43d4c08899147946cb82be12281567ba9103b669c78a5dbde6151",
    "recruitment": "58306ca45a506d79d3d07bc28df65d3c96bbf1943ee9122d4fb6d927842b049b",
    "regression": "149811406f3e2c9d0b7647d52c9e0a530538fc49c51cc773cd222483c8c80dd0",
    "replication": "299ffc7f9832bf3c8a0354824677e3ae6c06f33840d50a5ad414cd1fc3907b54",
    "report": "6a0e5a55d54ef33898a37ff08c5f156a738006960aff04435bad2eca0d56ce27",
    "run_history": "924c55ff2b1ab5218e7fa604e319dface9e1c93c8290e7f66755f89ef7c33823",
    "timelapse": "e3816a2eabfc60e1ad2b404ec2fbf892d87d89a575524d713dcb635f0f089e49",
    "train_cellpose": "9ca74cad76ea20c4a8f56ce41de722a8484f2ca1d506b1bd9b2fdbfb07e86c12",
    "train_compare": "21695b9522759bdede82fb4c5dc494df8ab6d4f5dba8f02f68b48d1b249c445f",
    "umap": "e3911e6d54899dc6155887fd60863a542673677b18369ca88fde9bd7bf2f9ed6",
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
    assert len(reviewed_keys) == 33
    assert set(REVIEWED_SOURCE_HASHES) == set(reviewed_keys)


validate_module_summaries()


__all__ = ["MODULE_SUMMARIES", "REVIEWED_SOURCE_HASHES", "module_summary"]
