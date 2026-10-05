"""A GUI category change requires new complete review before tutorial publication."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from build_navigation import CATEGORY_REVIEW, LABELS, build, category_labels


def test_actual_hierarchy_routes_have_reviewed_labels_without_changing_lessons():
    catalog_path = Path(__file__).resolve().parents[3] / 'docs/source/_extra/tutorials/catalog/lessons_en.json'
    catalog = json.loads(catalog_path.read_text())
    original = deepcopy(catalog)
    navigation = build(catalog)
    main = next(section for section in navigation['sections'] if section['id'] == 'main')
    categories = [group['title'] for group in main['groups']]
    reviewed = category_labels(categories)
    assert categories == ['Core', 'Data', 'Tools', 'Assays']
    assert set(navigation['labels']) == set(LABELS)
    for language in LABELS:
        actual = [navigation['labels'][language][group['label_index']]
                  for group in main['groups']]
        assert actual == reviewed[language]
    assert catalog == original
    assert len(navigation['preserved_lesson_ids']) == 85


@pytest.mark.parametrize('failure', ['new_category', 'reordered_source', 'stale_hash',
                                   'missing_locale', 'empty_label', 'truncated_locale',
                                   'changed_english', 'invented_signoff'])
def test_incomplete_or_stale_category_review_refuses_generation(tmp_path, failure):
    review = json.loads(CATEGORY_REVIEW.read_text())
    source = list(review['source'])
    if failure == 'new_category':
        source.append('New category')
    elif failure == 'reordered_source':
        source.reverse()
    elif failure == 'stale_hash':
        review['source_sha256'] = '0' * 64
    elif failure == 'missing_locale':
        del review['labels']['ja']
    elif failure == 'empty_label':
        review['labels']['ja'][0] = ' '
    elif failure == 'truncated_locale':
        review['labels']['da'].pop()
    elif failure == 'changed_english':
        review['labels']['en'][0] = 'Changed'
    else:
        review['review_kind'] = 'Native-speaker approved'
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(review, ensure_ascii=False))
    with pytest.raises(ValueError):
        category_labels(source, path)
