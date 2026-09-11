from copy import deepcopy
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from compose_curate_capture import require_scope


def good():
    return dict(accepted=False,mask_prior_history_preserved_after_second_save=False,
        mask_history_on_reopen={'previous_entries':3,'current_entries':0},synthetic=True,
        original_inputs_preserved=True,tracks_prior_history_preserved=True,
        external_first_mask_checkpoint={'source_unchanged':True,'edit_count':3,'copied_sha256':{'mask':'a','ledger':'b'}},
        external_second_mask_checkpoint={'source_unchanged':True,'edit_count':1,'copied_sha256':{'mask':'c','ledger':'d'}},
        external_first_history_still_preserved=True)


@pytest.mark.parametrize('key,value,message',[
    ('accepted',True,'observed mask-history'),
    ('mask_prior_history_preserved_after_second_save',True,'observed mask-history'),
    ('synthetic',False,'disclosed synthetic'),
    ('original_inputs_preserved',False,'disclosed synthetic'),
    ('tracks_prior_history_preserved',False,'disclosed synthetic'),
    ('external_first_history_still_preserved',False,'first-session checkpoint')])
def test_false_claim_or_lost_history_rejected_after_valid_scope(key,value,message):
    proof=good();require_scope(proof);proof[key]=value
    with pytest.raises(ValueError,match=message):require_scope(proof)


def test_missing_file_or_wrong_edit_count_rejected(tmp_path):
    require_scope(good())
    for key in ('external_first_mask_checkpoint','external_second_mask_checkpoint'):
        for field,value in [('source_unchanged',False),('edit_count',0),('copied_sha256',{'mask':'a'})]:
            proof=deepcopy(good());proof[key][field]=value
            with pytest.raises(ValueError,match='exact external'):require_scope(proof)
