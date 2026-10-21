"""Matrix visual reconciliation keeps published-master and raw-frame routes distinct."""
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_native_live_timing import frozen_native_stage
from test_published_native_reuse import retained_stage, write
import published_native_reuse as reuse
from check_completed_matrix import reconcile_visual_sources,digest


def test_matrix_validated_published_intervals_need_no_fake_capture_images(retained_stage):
    stage,root,baseline,repo=retained_stage
    reuse.prepare(stage,'05_home',baseline)
    result=reconcile_visual_sources(stage,'05_home',json.loads((root/'lesson.en.json').read_text()))
    assert result['fresh_capture'] is False
    assert result['raw_capture_revalidated'] is False
    assert not (root/'visual.json').exists()
    assert result['master_sha256']==digest(root/'video/05_home_silent.mp4')


@pytest.mark.parametrize('mutation',['missing_record','forged_mode','master','published_clock','current_source'])
def test_matrix_reuse_rejects_missing_or_changed_real_source_bindings(retained_stage,mutation):
    stage,root,baseline,repo=retained_stage
    reuse.prepare(stage,'05_home',baseline)
    if mutation=='missing_record':
        (root/reuse.RECORD).rename(root/'preserved-source-record.json')
    elif mutation=='forged_mode':
        path=root/'scenes.json';value=json.loads(path.read_text());value['kind']='ordinary';write(path,value)
    elif mutation=='master':
        path=root/'video/05_home_silent.mp4';path.rename(path.with_suffix('.preserved.mp4'));path.write_bytes(b'changed explicit unit fixture')
    elif mutation=='published_clock':
        path=baseline/'media_host/05_home/video/native-live-timings.json';path.write_text(path.read_text()+' ')
    else:
        path=repo/'tools/tutorials/lessons/05_home.json';value=json.loads(path.read_text());value['scenes'][0]['narration']='Different actual current instruction.';write(path,value)
    with pytest.raises(ValueError):
        reconcile_visual_sources(stage,'05_home',json.loads((root/'lesson.en.json').read_text()))


def test_matrix_ordinary_visual_still_requires_actual_image_hash_and_inside_stage(tmp_path):
    stage=tmp_path/'stage';root=stage/'production/example';root.mkdir(parents=True)
    english={'scenes':[{'narration':'Current instruction.'}]}
    image=root/'image.png';image.write_bytes(b'explicit image hash unit fixture')
    write(root/'visual.json',{'scenes':[{'image':image.name,'capture_sha256':digest(image)}]})
    assert reconcile_visual_sources(stage,'example',english) is None
    image.write_bytes(b'changed')
    with pytest.raises(ValueError,match='captured image changed'):
        reconcile_visual_sources(stage,'example',english)
    outside=tmp_path/'outside.png';outside.write_bytes(b'explicit outside path fixture')
    write(root/'visual.json',{'scenes':[{'image':str(outside),'capture_sha256':digest(outside)}]})
    with pytest.raises(ValueError,match='leaves staging'):
        reconcile_visual_sources(stage,'example',english)
