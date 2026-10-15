#!/usr/bin/env python3
"""Copy existing post-record selective04 evidence; never author acceptance."""
from pathlib import Path, PurePosixPath
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import argparse,datetime,hashlib,json,os,re,shutil,subprocess
W=Path(__file__).resolve().parent
CAPTURE='b929b7dc53465132322c3ca0b866f567de773b9f'
PLAYER_PATHS={'tools/tutorials/authoring/web/app_v2.js','tools/tutorials/tests/test_player_preserve_visual_intervals.py'}
BASE_SHA='98d88a909b3eeaf14f829297d965461e62a07dff7dfd25ce50d2c2bc15a25439'
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def bind(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def checked(row):
 p=Path(row['path']);assert p.is_absolute() and not p.is_symlink() and row==bind(p);return p

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--export',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
 p.add_argument('--source-commit',required=True,help='Actual committed player followup; native capture remains K4')
 p.add_argument('--final-review',type=Path,required=True);p.add_argument('--final-review-sha256',required=True)
 p.add_argument('--evidence',type=Path,action='append',required=True,help='Actual small capture/boundary JSON or PNG evidence; repeat exact files, never raw media')
 a=p.parse_args();export=a.export.resolve();out=a.output.resolve();review=a.final_review.resolve();K=a.source_commit
 assert re.fullmatch('[0-9a-f]{40}',K)
 subprocess.run(['git','merge-base','--is-ancestor',CAPTURE,K],cwd=ROOT,check=True)
 assert set(subprocess.check_output(['git','diff','--name-only',CAPTURE,K],cwd=ROOT,text=True).splitlines())==PLAYER_PATHS
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==K
 assert export.parent==W and export.is_dir() and out.parent==W and not out.exists() and not out.is_relative_to(export)
 em=export/'export-manifest.json';manifest=read(em)
 assert manifest['lesson']=='04_platform_installers' and manifest['source_commit']==K and manifest['baseline_manifest_sha256']==BASE_SHA
 assert manifest['capture_source_commit']==manifest['stage_source_commit']==CAPTURE
 assert set(manifest['player_followup_paths'])==PLAYER_PATHS
 assert read(checked(manifest['stage_preparation']))['source_commit']==CAPTURE
 count=manifest['checkpoint_file_count'];snapshot_count=manifest['snapshot_file_count']
 assert type(count) is int and count>0 and manifest['evidence_extra_count']==3 and snapshot_count==count+3
 assert manifest['translation_review_count']==13 and manifest['preserved_other_lessons']==83 and manifest['unchanged_catalog_lesson_objects']==14 and manifest['retained_audio_sidecar_files']==100
 assert manifest['checks']['record_completed'] is True and manifest['checks']['all84_hosted_cases_passed'] is True and manifest['checks']['coherent_recorded_files_plus3_snapshot_hashes_and_lengths'] is True
 assert all(manifest['claims'][k] is False for k in ['fresh_capture','whole_fresh_capture','complete_native_clips','resynthesized','current_runtime_audio_freshness','human_listening','native_speaker_acceptance'])
 checkpoint=export/'checkpoint';rows=manifest['files'];assert len(rows)==len({r['path'] for r in rows})==snapshot_count
 assert manifest['fresh_native_replacement_scenes']==[9] and manifest['generated_reference_replacement_scenes']==[10]
 assert manifest['fresh_native_replacement_frames']==851 and manifest['total_master_frames']==9167+851==10018
 assert manifest['preserve_visual_intervals'] is True
 native_approval=checked(manifest['native_acceptance'])
 native=read(native_approval);assert native['accepted'] is True and native['source_commit']==CAPTURE and native['actual_native_frames']==851 and native['requested_minimum_seconds']==28
 inventory=read(checked(native['exact_source']))['sha256']
 assert len(inventory)==502 and not(set(inventory)&PLAYER_PATHS)
 for name,expected_sha in inventory.items():
  rel=PurePosixPath(name);assert rel.parts and not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==name
  path=ROOT/rel;assert path.resolve().is_relative_to(ROOT) and sha(path)==expected_sha
 for row in rows:
  rel=PurePosixPath(row['path']);assert rel.parts and not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==row['path']
  q=checkpoint/rel;assert q.resolve().is_relative_to(checkpoint.resolve()) and not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
 assert sorted(q.relative_to(checkpoint).as_posix() for q in checkpoint.rglob('*') if q.is_file())==sorted(r['path'] for r in rows)
 for item in manifest['review_evidence']:
  checked(item['source']);snapshot=checked(item['snapshot'])
  assert snapshot.is_relative_to(export), 'Review evidence snapshot must be inside exact export tree'
  assert item['source']['sha256']==item['snapshot']['sha256'] and item['source']['bytes']==item['snapshot']['bytes']
 assert sha(review)==a.final_review_sha256
 final=read(review);assert final['lesson']=='04_platform_installers' and final['status']=='pass' and final['source_commit']==K
 assert checked(final['export_manifest'])==em and final['export_snapshot_file_count']==snapshot_count
 assert Path(final['candidate']).resolve()==Path(manifest['candidate_root']).resolve()
 actual_manifest=checked(final['manifest']);snapshot_manifest=checkpoint/'release-manifest.json'
 assert actual_manifest==Path(manifest['candidate_root']).resolve()/'release-manifest.json'
 assert final['manifest']['sha256']==manifest['release_manifest_sha256']==sha(snapshot_manifest) and actual_manifest.stat().st_size==snapshot_manifest.stat().st_size
 assert checked(final['native_acceptance'])==native_approval and final['native_acceptance']==manifest['native_acceptance']
 assert final.get('whole_fresh_capture') is not True and final.get('complete_native_clips') is not True
 for key in ['master','timing']:
  checked(final[key]);assert final[key]==manifest[key]
 for key,name in [('publication_receipt','publication-receipt.json'),('published_media_browser_checks','published-media-browser-checks.json'),('recorded_checkpoint','checkpoint.json')]:
  actual=checked(final[key]);assert sha(actual)==sha(checkpoint/name)
 assert final.get('human_listening_signoff') is not True and final.get('native_speaker_signoff') is not True and final.get('audible_word_completeness_verified') is not True
 # Closed export tree plus explicit small work receipts/helpers/logs. No media regeneration or directory-wide source scan.
 inputs={}
 for q in export.rglob('*'):
  assert not q.is_symlink()
  if q.is_file():inputs['export/'+q.relative_to(export).as_posix()]=q
 for q in sorted(W.iterdir()):
  if q.is_file() and not q.is_symlink() and q.suffix in {'.json','.py','.sh','.txt','.log'}:
   inputs['work-evidence/'+q.name]=q
 # Source/card directories are closed existing deliveries; root passes actual boundary files explicitly.
 for folder in ['source-handoff','source-handoff-playback-followup','source-handoff-pending-seek-followup','scene10-doctor-command-reference']:
  base=W/folder;assert base.is_dir() and not base.is_symlink(), 'Required actual delivery directory missing'
  for q in sorted(base.rglob('*')):
   assert not q.is_symlink(), 'Never copy a delivery symlink'
   if q.is_file():inputs['work-evidence/'+folder+'/'+q.relative_to(base).as_posix()]=q
 inputs['final-publication-review.json']=review
 for i,q in enumerate(a.evidence):
  assert not q.is_symlink();q=q.resolve();assert q.is_file();inputs[f'additional-evidence/{i:02}-{q.name}']=q
 forbidden={'.mp4','.m4a','.wav','.webm','.mov','.mkv','.flac','.aac','.mp3'}
 assert all(q.suffix.lower() not in forbidden for q in inputs.values()), 'Never copy raw or rendered video/audio media into the evidence packet'
 before={name:bind(q) for name,q in inputs.items()}
 assert len({item['snapshot']['path'] for item in manifest['review_evidence']})==len(manifest['review_evidence'])
 # Only here can an output be created: all prerequisite checks above must already hold.
 out.mkdir();copied=[]
 for name,q in inputs.items():
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target)
  b=before[name];assert bind(q)==b and target.stat().st_size==b['bytes'] and sha(target)==b['sha256']
  copied.append({'path':name,'bytes':b['bytes'],'sha256':b['sha256'],'source':b})
 report={'schema':1,'lesson':'04_platform_installers','source_commit':K,'capture_source_commit':CAPTURE,'stage_source_commit':CAPTURE,'player_followup_paths':sorted(PLAYER_PATHS),'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'export_manifest':bind(em),'final_publication_review':bind(review),'file_count':len(copied),'files':copied,'counts':{'checkpoint_snapshot':snapshot_count,'official_reviews_bound_in_export':13,'unchanged_audio_sidecars':100,'unchanged_catalog_objects':14,'preserved_other_lessons':83},'fresh_native_replacement_frames':851,'total_master_frames':10018,'preserve_visual_intervals':True,'scope':'Copies existing actual export and acceptance evidence; does not execute or author acceptance. Only scene9 full fresh851-frame Home and scene10 generated historicaldoctor reference were replaced. Capture/render remain K4; final player source is the exact two-file K4-descendant followup. Master10018frames at30fps; other12durations and selected818-frame legacyPerformance interval retained. preserve_visual_intervals is enabled; this is not a wholefreshcapture or complete-raw-native claim.','claims':manifest['claims'],'limits':manifest['limits']}
 (out/'packet-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==K
 print(json.dumps({'packet':str(out),'listed_files':len(copied),'manifest':bind(out/'packet-manifest.json')}))
if __name__=='__main__':main()
