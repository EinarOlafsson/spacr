"""Generated voice metadata never excuses a changed authored lesson."""
from copy import deepcopy
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from check_completed_matrix import reconcile_english_catalog,voice_matrix


def matrix():
    return voice_matrix(Path(__file__).resolve().parents[1]/'authoring/tools/render_all_voices.py')


def source():
    return {'id':'01_pypi_github','title':'Current title','scenes':[{'visual':'overview','narration':'Find spaCR on PyPI.'}]}


def test_exact_normal_fifty_voice_metadata_and_unadorned_source_pass():
    english=source();inventory=matrix();catalog=dict(english,narration_voices=deepcopy(inventory))
    reconcile_english_catalog(catalog,english,inventory)
    reconcile_english_catalog(english,english,inventory)
    assert 'narration_voices' not in english


@pytest.mark.parametrize('mutation',['missing_voice','unknown_voice','extra_language','wrong_order','not_a_matrix'])
def test_malformed_or_noncanonical_voice_declaration_fails(mutation):
    english=source();inventory=matrix();catalog=deepcopy(english);declared=deepcopy(inventory)
    if mutation=='missing_voice':declared['en'].pop()
    elif mutation=='unknown_voice':declared['en'][0]='invented_voice'
    elif mutation=='extra_language':declared['unknown']=[]
    elif mutation=='wrong_order':declared['en'].reverse()
    else:declared=list(declared)
    catalog['narration_voices']=declared
    with pytest.raises(ValueError,match='normal fifty-voice matrix'):
        reconcile_english_catalog(catalog,english,inventory)


@pytest.mark.parametrize('field',['narration','title','scene_order'])
def test_valid_metadata_does_not_hide_changed_authored_fields(field):
    english=source();inventory=matrix();catalog=deepcopy(english);catalog['narration_voices']=deepcopy(inventory)
    if field=='narration':catalog['scenes'][0]['narration']='Changed words.'
    elif field=='title':catalog['title']='Changed title'
    else:catalog['scenes'][0]['visual']='another_action'
    with pytest.raises(ValueError,match='Staged English differs'):
        reconcile_english_catalog(catalog,english,inventory)


def test_retained_bound_complete_voice_sets_accept_original_order():
    english=source();inventory=matrix();catalog=deepcopy(english)
    catalog['narration_voices']={language:list(reversed(voices)) for language,voices in inventory.items()}
    before=deepcopy(catalog)
    reconcile_english_catalog(catalog,english,inventory,retained_narration=True)
    assert catalog==before and 'narration_voices' not in english


@pytest.mark.parametrize('mutation',['missing_declaration','missing_voice','unknown_voice','duplicate_voice','missing_language','not_a_matrix'])
def test_retained_english_rejects_incomplete_or_malformed_matrix(mutation):
    english=source();inventory=matrix();catalog=deepcopy(english);declared=deepcopy(inventory)
    if mutation=='missing_declaration':pass
    else:
        if mutation=='missing_voice':declared['en'].pop()
        elif mutation=='unknown_voice':declared['en'][0]='invented_voice'
        elif mutation=='duplicate_voice':declared['en'][-1]=declared['en'][0]
        elif mutation=='missing_language':declared.pop('en')
        else:declared=list(declared)
        catalog['narration_voices']=declared
    with pytest.raises(ValueError,match='fifty-voice matrix'):
        reconcile_english_catalog(catalog,english,inventory,retained_narration=True)


def test_retained_complete_voice_sets_do_not_hide_authored_change():
    english=source();inventory=matrix();catalog=deepcopy(english)
    catalog['narration_voices']={language:list(reversed(voices)) for language,voices in inventory.items()}
    catalog['scenes'][0]['narration']='Changed words.'
    with pytest.raises(ValueError,match='Staged English differs'):
        reconcile_english_catalog(catalog,english,inventory,retained_narration=True)
