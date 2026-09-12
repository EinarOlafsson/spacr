"""Real capture receipts pin scope; edited claims must fail, not become tutorials."""
from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model_promotion import MODELS, require_recorded_model, validate_scope
from stage_lesson import read

ROOT = Path(__file__).resolve().parents[1]


def test_authored_scripts_alone_remain_held_and_cannot_request_promotion():
    from coming_soon import HELD, release_catalog
    # Unit catalog only: no fake media or capture is offered as release proof.
    source = {'lessons': [read(ROOT / 'lessons' / (identity + '.json')) if identity in MODELS
                          else {'id': identity, 'scenes': []} for identity in HELD]}
    ordinary = release_catalog(source, 'en')
    assert all(item['status'] == 'coming_soon' for item in ordinary['lessons'])
    for identity in MODELS:
        with pytest.raises(ValueError, match='actual staged media'):
            release_catalog(source, 'en', model_promotions=[identity])


def test_model_promotion_cannot_be_used_to_release_another_workflow(tmp_path):
    from build_release_candidate import build
    with pytest.raises(ValueError, match='Unknown model tutorial'):
        build(tmp_path, model_promotions=['71_investigate_hit'])


@pytest.mark.parametrize('identity', MODELS)
def test_actual_recorded_scope_passes_but_metadata_alone_cannot_promote(identity):
    capture = read(ROOT / 'evidence' / MODELS[identity][1])
    validate_scope(identity, capture)
    with pytest.raises(ValueError, match='actual staged media'):
        require_recorded_model(None, 'en', read(ROOT / 'lessons' / (identity + '.json')))


@pytest.mark.parametrize('key', ['benchmark_completed', 'model_downloaded', 'training_performed',
                                 'original_checkpoint_unchanged', 'displayed_rows'])
def test_zoo_scope_rejects_changed_actual_recording_claims(key):
    capture = read(ROOT / 'evidence' / MODELS['22_model_zoo'][1])
    validate_scope('22_model_zoo', capture)
    capture[key] = 9 if key == 'displayed_rows' else not capture[key]
    with pytest.raises(ValueError):
        validate_scope('22_model_zoo', capture)


@pytest.mark.parametrize('key', ['accuracy_validated', 'different_model_weights_compared',
                                 'source_unchanged', 'same_mask_control_passed', 'comparison'])
def test_comparison_rejects_unearned_accuracy_and_wrong_pixel_results(key):
    capture = read(ROOT / 'evidence' / MODELS['21_model_compare'][1])
    validate_scope('21_model_compare', capture)
    changed = deepcopy(capture)
    run = changed['terminal']['run']
    if key == 'comparison':
        run[key]['n_matched'] = 90
    else:
        run[key] = not run[key]
    with pytest.raises(ValueError):
        validate_scope('21_model_compare', changed)
