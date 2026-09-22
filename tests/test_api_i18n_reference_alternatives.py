"""Conjunctions between protected API references still need translation."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

from build_documentation_i18n import (
    _api_block_requires_translation,
    _api_block_valid,
    _reviewed_api_block_valid,
)


@pytest.mark.parametrize("validate", [_api_block_valid, _reviewed_api_block_valid])
@pytest.mark.parametrize('language, conjunction', [
    ('de', 'oder'), ('es', 'o'), ('fr', 'ou'), ('hi', 'या'), ('is', 'eða'),
    ('ko', '또는'), ('pt', 'ou'), ('sv', 'eller'), ('zh_CN', '或'),
])
@pytest.mark.parametrize('source, protected', [
    (':data:`EXTERNAL_STATUS_SUFFIX` or :data:`EXTERNAL_SCORES_SUFFIX`.',
     'EXTERNAL_SCORES_SUFFIX'),
    ("``location_column`` — ``'columnID'`` or ``'rowID'``.", 'rowID'),
])
def test_reference_alternatives_translate_the_conjunction_and_keep_both_targets(
    validate, language, conjunction, source, protected,
):
    translated = source.replace(' or ', f' {conjunction} ')
    assert _api_block_requires_translation(source)
    assert validate(source, translated, language)
    assert not validate(source, source, language)
    assert not validate(source, translated.replace(protected, 'WRONG'), language)


@pytest.mark.parametrize('source', ['``or``', ':data:`or`', '``no``', '``(H, W, C)``.'])
def test_protected_short_words_remain_literals(source):
    assert not _api_block_requires_translation(source)
    assert _reviewed_api_block_valid(source, source, 'de')
