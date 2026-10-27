from copy import deepcopy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classify_split_evidence import check_native_audit, check_predictions


def audit():
    return dict(passed=True, group_by='well', train_samples=32, validation_samples=32,
                critical_levels=[], hash_errors=[], unverifiable_counts={},
                overlap_counts={k: 0 for k in ('well', 'field', 'object', 'exact', 'content_sha256', 'augmentation_family')})


def predictions():
    truth = {'a.png': 0, 'b.png': 1}
    rows = [dict(filename=name, true_label=str(y), predicted_label='0',
                 prob_class_0='0.7', prob_class_1='0.3') for name, y in truth.items()]
    metric = dict(accuracy='0.5', Accuracy='0.5', f1_macro=str(1/3), class_support='[1, 1]')
    return rows, metric, truth


def test_positive_native_and_independent_counterparts():
    check_native_audit(audit(), (32, 32))
    proof = check_predictions(*predictions())
    assert proof['confusion_matrix'] == [[1, 0], [1, 0]]
    assert proof['accuracy'] == proof['majority_accuracy'] == .5


@pytest.mark.parametrize('kind', ['passed', 'group', 'count', 'well_overlap', 'unknown_identity'])
def test_named_audit_defects(kind):
    value = audit()
    if kind == 'passed':
        value['passed'] = False
    elif kind == 'group':
        value['group_by'] = 'cell'
    elif kind == 'count':
        value['train_samples'] = 31
    elif kind == 'well_overlap':
        value['overlap_counts']['well'] = 1
    else:
        value['unverifiable_counts'] = {'well': 1}
    with pytest.raises(ValueError):
        check_native_audit(value, (32, 32))


@pytest.mark.parametrize('kind', ['label', 'duplicate', 'probability', 'argmax', 'accuracy', 'f1', 'support'])
def test_named_saved_evaluation_defects(kind):
    rows, metric, truth = deepcopy(predictions())
    if kind == 'label':
        rows[0]['true_label'] = '1'
    elif kind == 'duplicate':
        rows[1]['filename'] = rows[0]['filename']
    elif kind == 'probability':
        rows[0]['prob_class_0'] = '.9'
    elif kind == 'argmax':
        rows[0]['predicted_label'] = '1'
    elif kind == 'accuracy':
        metric['accuracy'] = '.8'
    elif kind == 'f1':
        metric['f1_macro'] = '.8'
    else:
        metric['class_support'] = '[0, 2]'
    with pytest.raises(ValueError):
        check_predictions(rows, metric, truth)


@pytest.mark.parametrize('fault',['foreign_root','parent_escape','relative','symlink','missing'])
def test_explicit_source_root_mapping_rejects_escape_or_nonphysical_file(tmp_path,fault):
    from classify_split_evidence import mapped_source_path
    physical=tmp_path/'physical';physical.mkdir();file=physical/'source.bin';file.write_bytes(b'original')
    recorded=Path('/recorded-capture');path=recorded/'source.bin'
    if fault=='foreign_root':path=Path('/other/source.bin')
    elif fault=='parent_escape':path=recorded/'..'/'source.bin'
    elif fault=='relative':path=Path('source.bin')
    elif fault=='symlink':
        link=physical/'link';link.symlink_to(file);path=recorded/'link'
    else:path=recorded/'missing'
    with pytest.raises(ValueError):mapped_source_path(path,(recorded,physical))


def test_explicit_source_mapping_preserves_default_and_original_bytes(tmp_path):
    from classify_split_evidence import mapped_source_path
    physical=tmp_path/'physical';physical.mkdir();p=physical/'source.bin';p.write_bytes(b'original')
    recorded=Path('/recorded-capture/source.bin')
    assert mapped_source_path(recorded)==recorded
    assert mapped_source_path(recorded,(recorded.parent,physical))==p
    assert p.read_bytes()==b'original'


@pytest.fixture
def relocated_split(tmp_path):
    import json,sqlite3,hashlib
    from prepare_classify_split import MARKER,TRAIN_WELLS,TEST_WELLS
    physical=tmp_path/'physical';dataset=physical/'dataset';dataset.mkdir(parents=True)
    logical=Path('/recorded-capture');db=physical/'source.db'
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    rows=[];manifest=[]
    with sqlite3.connect(db) as con:
        con.execute('CREATE TABLE png_list(prcfo,plateID,rowID,columnID,infected)')
        for split,wells in [('train',TRAIN_WELLS),('test',TEST_WELLS)]:
            for well in sorted(wells):
                for label in (1,2):
                    folder=dataset/split/f'infected_{label}';folder.mkdir(parents=True,exist_ok=True)
                    marker=folder/MARKER;marker.write_text(json.dumps(dict(spacr_crop_format=3,channel_order='declared_rgb',narrowing='high-byte')))
                    for i in range(8):
                        identity='_'.join(well)+f'_f1_o{label*10+i}'
                        target=folder/(identity+'.png');target.write_bytes(identity.encode())
                        source=physical/'inputs'/target.name;source.parent.mkdir(exist_ok=True);source.write_bytes(target.read_bytes())
                        con.execute('INSERT INTO png_list VALUES(?,?,?,?,?)',(identity,*well,label))
                        manifest.append(dict(prcfo=identity,well=list(well),infected=label,class_name=f'infected_{label}',split=split,
                            target=str(logical/target.relative_to(physical)),source=str(logical/source.relative_to(physical)),source_sha256=digest(source),
                            source_marker=str(logical/marker.relative_to(physical)),source_marker_sha256=digest(marker)))
    data=dict(accepted=True,destination=str(logical/'dataset'),source_database=str(logical/'source.db'),source_database_sha256=digest(db),records=manifest)
    path=dataset/'tutorial_input_manifest.json';path.write_text(json.dumps(data))
    return physical,dataset,logical,path,data


def test_relocated_inputs_validate_same64_originals_without_rewriting_manifest(relocated_split):
    from classify_split_evidence import inspect_inputs
    physical,dataset,logical,path,data=relocated_split;before=path.read_bytes()
    result=inspect_inputs(dataset,path_mapping=(logical,physical))
    assert result['objects_checked']==64 and result['sample_counts']=={'train':32,'test':32}
    assert path.read_bytes()==before
    with pytest.raises(FileNotFoundError):inspect_inputs(dataset)


@pytest.mark.parametrize('fault',['changed_source','foreign_manifest_path','wrong_dataset'])
def test_relocated_inputs_do_not_waive_hashes_or_exact_dataset(relocated_split,fault):
    import json
    from classify_split_evidence import inspect_inputs
    physical,dataset,logical,path,data=relocated_split
    if fault=='changed_source':(physical/Path(data['records'][0]['source']).relative_to(logical)).write_bytes(b'changed')
    elif fault=='foreign_manifest_path':
        data['records'][0]['source']='/different-capture/file.png';path.write_text(json.dumps(data))
    else:
        data['destination']=str(logical/'other-dataset');path.write_text(json.dumps(data))
    with pytest.raises(ValueError):inspect_inputs(dataset,path_mapping=(logical,physical))
