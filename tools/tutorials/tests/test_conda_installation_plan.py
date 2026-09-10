"""A solve is a bounded plan, never proof that the package actually works."""
import copy
from pathlib import Path

import pytest


@pytest.fixture
def plan_checker(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]))
    from check_conda_installation import accepted_plan
    return accepted_plan


@pytest.fixture
def plan():
    return {'success': True, 'actions': {
        'LINK': [{'name': 'spacr', 'version': '1.5.0.4', 'channel': 'conda-forge'},
                 {'name': 'python', 'version': '3.12.13', 'channel': 'conda-forge'}],
        'FETCH': [{'size': 1024**3}, {'size': 2 * 1024**3}],
    }}


def test_positive_plan_reports_actual_channel_version_and_download_size(plan_checker, plan):
    assert plan_checker(plan, 100 * 1024**3) == dict(
        packages=2, version='1.5.0.4', download_bytes=3 * 1024**3)


@pytest.mark.parametrize('broken', [
    {'success': False}, {'actions': {}}, {'actions': {'LINK': []}},
])
def test_unsolved_or_empty_plan_is_not_an_installation(plan_checker, plan, broken):
    plan.update(broken)
    with pytest.raises(ValueError, match='complete installable'):
        plan_checker(plan, 100 * 1024**3)


@pytest.mark.parametrize('variant', ['missing', 'duplicate', 'wrong_channel'])
def test_spacr_must_be_unique_and_from_the_selected_channel(plan_checker, plan, variant):
    links = plan['actions']['LINK']
    if variant == 'missing':
        links.pop(0)
    elif variant == 'duplicate':
        links.append(copy.deepcopy(links[0]))
    else:
        links[0]['channel'] = 'unrelated-channel'
    with pytest.raises(ValueError, match='exactly one spaCR'):
        plan_checker(plan, 100 * 1024**3)


@pytest.mark.parametrize('download_gib,free_gib,accepted', [
    (20, 100, True), (21, 200, False), (1, 39, False), (10, 49, False),
    (10, 50, True), (0, 40, True),
])
def test_download_and_expansion_headroom_are_bounded(
        plan_checker, plan, download_gib, free_gib, accepted):
    plan['actions']['FETCH'] = [{'size': download_gib * 1024**3}]
    if accepted:
        assert plan_checker(plan, free_gib * 1024**3)['download_bytes'] == download_gib * 1024**3
    else:
        with pytest.raises(ValueError, match='download/disk budget'):
            plan_checker(plan, free_gib * 1024**3)
