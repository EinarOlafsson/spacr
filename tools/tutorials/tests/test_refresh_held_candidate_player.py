"""Real copying/codec binding with synthetic50 byte records; no decoding/browser."""
from pathlib import Path
from copy import deepcopy
import sys,json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import refresh_held_candidate_player as r

def put(path,value):
 path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path

def rows(root):
 return [dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=r.digest(p)) for top in ('web','media_host') for p in (root/top).rglob('*') if p.is_file()]

@pytest.fixture
def prepared(tmp_path,monkeypatch):
 baseline=tmp_path/'published';stage=tmp_path/'stage';source=stage/'release-candidate-prose-original';logs=tmp_path/'build';repo=tmp_path/'repo'
 put(baseline/'release-manifest.json',dict(real_published=True));put(baseline/'publication-receipt.json',dict(published=True))
 for name in ('web/app_v2.js','web/lesson_catalog.js','web/catalog/lessons_en.json','media_host/other/video/master.mp4'):
  p=source/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(('old '+name).encode())
 tracks=[]
 for i in range(50):
  voice=f'voice{i:02}';prefix=f'media_host/{r.LESSON}/audio/en/{voice}'
  for suffix in ('.m4a','.json'):
   p=source/(prefix+suffix);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((voice+suffix).encode())
  tracks.append(dict(language='en',voice=voice,audio_sha256=r.digest(source/(prefix+'.m4a')),timing_sha256=r.digest(source/(prefix+'.json')),provenance_kind='mixed',mixed_record_sha256='mixed',original_audio_sha256='original'))
 manifest=dict(routes=84,ready_lessons=84,files=rows(source),web_bytes=1,ceiling_bytes=1000000,release_hold=True,published=False,uploaded=False,refreshed_lessons=[r.LESSON],prose_refresh=True,baseline_manifest_sha256=r.digest(baseline/'release-manifest.json'),prose_mixed_record_sha256='mixed',audio_patch_tracks=[dict(t,lesson=r.LESSON) for t in tracks])
 put(source/'release-manifest.json',manifest)
 codec=dict(lesson=r.LESSON,passed=True,tracks_checked=50,tracks=[dict(t,errors=[]) for t in tracks],manifest_sha256=r.digest(source/'release-manifest.json'),mixed_record_sha256='mixed',current_whole_track_tts_claimed=False,human_listening_review=False)
 put(source/r.CODEC,codec);put(source/'checks/placeholder-mutation-checks.json',dict(passed=True));put(source/'checks/candidate-browser-checks.json',dict(passed=True))
 logs.mkdir();(logs/'candidate.out').write_text('CANDIDATE '+str(source)+'\n');(logs/'build.rc').write_text('0\n');(logs/'candidate.rc').write_text('0\n')
 built=dict(candidate=str(source),manifest_sha256=r.digest(source/'release-manifest.json'),builder_sha256=r.BUILDER,policy_sha256='policy',build_log_sha256=r.digest(logs/'candidate.out'))
 put(logs/'candidate.json',built)
 player=repo/'tools/tutorials/authoring/web/app_v2.js';player.parent.mkdir(parents=True);player.write_bytes(b'approved replacement player')
 monkeypatch.setattr(r,'REPO',repo)
 def published(path,manifest):
  if path!=baseline or not manifest.get('real_published'):raise ValueError('Foreign publication')
 monkeypatch.setattr(r,'require_published_script_baseline',published);monkeypatch.setattr(r,'validate',lambda *args,**kwargs:None)
 calls=[]
 def components(*args):calls.append(args);return dict(exact_current_source=True,voices={'en':[f'voice{i:02}' for i in range(50)]})
 monkeypatch.setattr(r,'components',components)
 # Any attempt to re-enter an audio producer is a hard test failure.
 monkeypatch.setattr(r.prose.engine,'checked_tracks',lambda *a,**kw:pytest.fail('No repeated final AAC/PCM checking'))
 kw=dict(source_manifest_sha256=r.digest(source/'release-manifest.json'),build_record_sha256=r.digest(logs/'candidate.json'),policy_sha256='policy',player_sha256=r.digest(player))
 return dict(baseline=baseline,stage=stage,source=source,logs=logs,repo=repo,manifest=manifest,codec=codec,kw=kw,calls=calls)

def build(x):return r.refresh(x['stage'],x['baseline'],x['source'],x['logs'],**x['kw'])

def test_only_player_changes_retained_codec_is_truthful_original_is_immutable(prepared):
 x=prepared;before={p.relative_to(x['source']).as_posix():p.read_bytes() for p in x['source'].rglob('*') if p.is_file()}
 target=build(x);old=r.read(x['source']/'release-manifest.json');new=r.read(target/'release-manifest.json')
 a={v['path']:v for v in old['files']};b={v['path']:v for v in new['files']}
 assert set(a)==set(b) and [p for p in a if a[p]!=b[p]]==['web/app_v2.js']
 assert all(p.read_bytes()==before[p.relative_to(x['source']).as_posix()] for p in x['source'].rglob('*') if p.is_file())
 assert not (target/'checks/placeholder-mutation-checks.json').exists() and not (target/'checks/candidate-browser-checks.json').exists()
 proof=r.read(target/r.CODEC);lineage=proof['acceptance_reused_from']
 assert proof['tracks']==x['codec']['tracks'] and proof['manifest_sha256']==r.digest(target/'release-manifest.json')
 assert lineage['source_manifest_sha256']==x['kw']['source_manifest_sha256'] and lineage['original_codec_report_sha256']==r.digest(x['source']/r.CODEC)
 assert len(lineage['original_audio_records'])==100 and lineage['final_aac_repeated'] is False and lineage['player_browser_checks_retained'] is False
 assert (target/'checks/original-prose-audio-codec-checks.json').read_bytes()==(x['source']/r.CODEC).read_bytes()
 assert len(x['calls'])==2

@pytest.mark.parametrize('fault',['audio','timing','other_picture','codec','build_log','build0','foreign_baseline','wrong_player'])
def test_stale_or_foreign_proofs_reject_before_creating_successor(prepared,fault):
 x=prepared
 if fault in ('audio','timing','other_picture'):
  p=x['source']/({'audio':f'media_host/{r.LESSON}/audio/en/voice00.m4a','timing':f'media_host/{r.LESSON}/audio/en/voice00.json','other_picture':'media_host/other/video/master.mp4'}[fault]);p.write_bytes(b'changed')
 elif fault=='codec':
  d=r.read(x['source']/r.CODEC);d['tracks'][0]['errors']=['failed'];put(x['source']/r.CODEC,d)
 elif fault=='build_log':(x['logs']/'candidate.out').write_text('CANDIDATE wrong\n')
 elif fault=='build0':(x['logs']/'build.rc').write_text('1\n')
 elif fault=='foreign_baseline':x['baseline']=x['repo']
 elif fault=='wrong_player':x['kw']['player_sha256']='0'*64
 with pytest.raises((ValueError,FileNotFoundError)):build(x)
 assert list(x['stage'].glob('release-candidate-player-repair-*'))==[]

@pytest.mark.parametrize('fault',['duplicate','traversal','symlink'])
def test_inventory_rejects_duplicate_or_unsafe_source_paths(prepared,fault):
 x=prepared;manifest=deepcopy(x['manifest'])
 if fault=='duplicate':manifest['files'].append(deepcopy(manifest['files'][0]))
 elif fault=='traversal':manifest['files'][0]['path']='../outside'
 else:
  path=x['source']/manifest['files'][0]['path'];content=path.read_bytes();path.rename(path.with_name(path.name+'.held'));other=x['repo']/'outside';other.write_bytes(content);path.symlink_to(other)
 with pytest.raises(ValueError):r.inventory(x['source'],manifest)

def test_current_policy_guard_runs_before_any_source_or_codec_work(tmp_path,monkeypatch):
 repo=tmp_path/'repo';builder=repo/'tools/tutorials/build_prose_refresh_candidate.py';builder.parent.mkdir(parents=True);builder.write_bytes(b'original builder')
 monkeypatch.setattr(r,'REPO',repo);monkeypatch.setattr(r,'BUILDER',r.digest(builder));put(repo/'tools/tutorials/idle_tail_policy.json',dict(actual=True))
 monkeypatch.setattr(r.prose.engine,'sources',lambda *a:pytest.fail('No work after current policy drift'))
 with pytest.raises(ValueError,match='Current policy'):r.components(tmp_path,tmp_path,tmp_path,{}, {},'wrong policy')
