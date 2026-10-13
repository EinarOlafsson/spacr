import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from curate_checkpoint import checkpoint,sha


def inputs(tmp_path):
    data=tmp_path/'mask.tif';data.write_bytes(b'unchanged saved demonstration bytes')
    ledger=Path(str(data)+'.curation.json')
    ledger.write_text(json.dumps({'edits':[{'kind':'paint'},{'kind':'undo'},{'kind':'paint'}]}))
    return data,ledger


def test_both_exact_files_survive_source_changes(tmp_path):
    data,ledger=inputs(tmp_path);before=[sha(data),sha(ledger)]
    destination=tmp_path/'checkpoint';report=checkpoint(data,destination)
    assert report['edit_count']==3 and report['source_unchanged']
    assert [sha(data),sha(ledger)]==before
    data.write_bytes(b'next-session output');ledger.write_text('{"edits":[]}')
    assert [sha(destination/data.name),sha(destination/ledger.name)]==before
    assert json.loads((destination/'checkpoint.json').read_text())==report


def test_existing_checkpoint_refused_after_valid_copy(tmp_path):
    data,ledger=inputs(tmp_path);destination=tmp_path/'checkpoint'
    assert checkpoint(data,destination)['edit_count']==3
    before=sha(destination/ledger.name)
    with pytest.raises(FileExistsError,match='NEW checkpoint'):checkpoint(data,destination)
    assert sha(destination/ledger.name)==before


def test_missing_or_malformed_ledger_after_positive_counterpart(tmp_path):
    data,ledger=inputs(tmp_path)
    assert checkpoint(data,tmp_path/'good')['edit_count']==3
    ledger.unlink()
    with pytest.raises(ValueError,match='Both saved'):checkpoint(data,tmp_path/'missing')
    ledger.write_text('{"edits":null}')
    with pytest.raises(ValueError,match='recorded edit list'):checkpoint(data,tmp_path/'bad')
    assert not (tmp_path/'missing').exists() and not (tmp_path/'bad').exists()
