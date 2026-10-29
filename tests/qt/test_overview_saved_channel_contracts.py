"""Persisted Overview display choices retain independent image and channel settings."""
import copy
import json

from spacr.qt.widgets import qc_field_browser as browser


def test_legacy_view_state_restores_present_maps_without_writing_missing_defaults(tmp_path):
    path = tmp_path / 'results' / 'overview_view.json'
    path.parent.mkdir()
    legacy = {
        'version': browser._OVERVIEW_VIEW_STATE_VERSION,
        'plates': {'plate': {'name': 'Original plate', 'collapsed_rows': [1]}},
        'images': {'field': {'channels': {'1': {'gamma': 1.5}}}},
        'options': {'use_subfolders': [str(tmp_path / 'raw')]},
    }
    original = json.dumps(legacy, ensure_ascii=False, indent=2).encode('utf-8')
    path.write_bytes(original)

    loaded = browser._overview_load_view_state(tmp_path)

    assert loaded == dict(legacy, sources=[], view={})
    assert loaded['plates'] == legacy['plates'] and loaded['images'] == legacy['images']
    assert path.read_bytes() == original
    assert tuple(path.parent.iterdir()) == (path,)


def test_channel_scope_setting_roundtrips_named_channels_while_ignoring_empty_fragments():
    saved = ' , =plate, 1=plate, , 2=unknown, =well, 3=well, '
    scopes = browser._overview_channel_scopes(saved)
    assert scopes == {'1': 'plate', '2': 'image', '3': 'well'}
    assert browser._overview_scope_text(scopes) == '1=plate, 2=image, 3=well'
    assert browser._overview_channel_scopes(browser._overview_scope_text(scopes)) == scopes
    assert [browser._overview_scope_of(scopes, channel) for channel in (1, 2, 3, 4)] == [
        'plate', 'image', 'well', 'image']


def test_copying_an_unoverridden_channel_clears_only_that_target_channel_then_persists(tmp_path):
    images = {
        'reference': {'saturation': .9, 'channels': {'2': {'gamma': 1.5}}},
        'target': {'saturation': .6, 'global_normalization': False,
                   'channels': {'1': {'gamma': 2.}, '2': {'brightness': .25}}},
        'untouched': {'channels': {'1': {'black': .2}}},
    }
    before = copy.deepcopy(images)
    state = browser._overview_empty_view_state()
    state['images'] = images
    path = browser._overview_save_view_state(tmp_path, state)
    original = path.read_bytes()
    loaded = browser._overview_load_view_state(tmp_path)

    copied = browser._overview_copied_overrides(
        loaded['images'], 'reference', ('reference', 'target'), channel=1)

    assert loaded['images'] == before and images == before
    assert copied['reference'] == before['reference']
    assert copied['untouched'] == before['untouched']
    assert copied['target'] == {
        'saturation': .6, 'global_normalization': False,
        'channels': {'2': {'brightness': .25}}}
    assert path.read_bytes() == original
    loaded['images'] = copied
    assert browser._overview_save_view_state(tmp_path, loaded) == path
    assert browser._overview_load_view_state(tmp_path)['images'] == copied
