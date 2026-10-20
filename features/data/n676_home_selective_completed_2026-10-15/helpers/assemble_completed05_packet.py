#!/usr/bin/env python3
"""Copy existing post-record selected05 evidence; never author acceptance."""
from pathlib import Path, PurePosixPath
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import argparse,datetime,hashlib,json,os,shutil,subprocess
W=Path(__file__).resolve().parent
CAPTURE='87ffcb60cab0eca8772cb1ffaccc12f3f4f48be9'
BASE_SHA='aef7a3aeab1d0c792ee368bc6fe519377fd485f386c536498d2557b1b14fbe5f'
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
 p.add_argument('--final-review',type=Path,required=True);p.add_argument('--final-review-sha256',required=True)
 p.add_argument('--evidence',type=Path,action='append',default=[])
 a=p.parse_args();K=os.environ['SOURCE_COMMIT'];export=a.export.resolve();out=a.output.resolve();review=a.final_review.resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==K
 assert export.parent==W and export.is_dir() and out.parent==W and not out.exists() and not out.is_relative_to(export)
 em=export/'export-manifest.json';manifest=read(em)
 assert manifest['lesson']=='05_home' and manifest['source_commit']==K and manifest['baseline_manifest_sha256']==BASE_SHA
 assert manifest['checkpoint_file_count']==145 and manifest['evidence_extra_count']==3 and manifest['snapshot_file_count']==148
 assert manifest['capture_and_voice_source_commit']==CAPTURE
 assert manifest['translation_review_count']==13 and manifest['preserved_other_lessons']==83
 assert manifest['native_frame_count']==23571 and manifest['native_duration_seconds']==785.7 and manifest['whole_fresh_capture'] is False
 assert manifest['reused_strict_native_clips']==30 and manifest['fresh_strict_native_clips']==1
 assert manifest['checks']['record_completed'] is True and manifest['checks']['all84_hosted_playback_cases_passed'] is True and manifest['checks']['all_checkpoint_and_snapshot_hashes_and_lengths_match'] is True
 checkpoint=export/'checkpoint';rows=manifest['files'];assert len(rows)==len({r['path'] for r in rows})==148
 for row in rows:
  rel=PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==row['path']
  q=checkpoint/rel;assert not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
 assert sorted(q.relative_to(checkpoint).as_posix() for q in checkpoint.rglob('*') if q.is_file())==sorted(r['path'] for r in rows)
 for item in manifest['review_evidence']:
  checked(item['source']);snapshot=checked(item['snapshot'])
  assert snapshot.is_relative_to(export), 'Review evidence snapshot must be inside exact export tree'
 assert sha(review)==a.final_review_sha256
 final=read(review);assert final['lesson']=='05_home' and final['status']=='pass' and final['source_commit']==K
 assert checked(final['export_manifest'])==em and final['export_snapshot_file_count']==148
 assert Path(final['candidate']).resolve()==Path(manifest['candidate_root']).resolve()
 actual_manifest=checked(final['manifest']);snapshot_manifest=checkpoint/'release-manifest.json'
 assert actual_manifest==Path(manifest['candidate_root']).resolve()/'release-manifest.json'
 assert final['manifest']['sha256']==manifest['release_manifest_sha256']==sha(snapshot_manifest) and actual_manifest.stat().st_size==snapshot_manifest.stat().st_size
 for key in ['master','timing']:
  checked(final[key]);assert final[key]==manifest[key]
 for key,name in [('publication_receipt','publication-receipt.json'),('published_media_browser_checks','published-media-browser-checks.json'),('recorded_checkpoint','checkpoint.json')]:
  actual=checked(final[key]);assert sha(actual)==sha(checkpoint/name)
 assert final.get('human_listening_signoff') is not True and final.get('native_speaker_signoff') is not True and final.get('audible_word_completeness_verified') is not True
 # Closed export tree plus explicit small work receipts/helpers/logs. No media regeneration or directory-wide source scan.
 inputs={}
 for q in export.rglob('*'):
  assert not q.is_symlink()
  if q.is_file():
   assert q.suffix not in {'.mp4','.m4a','.wav','.npy','.db','.sqlite','.ini','.conf'}
   inputs['export/'+q.relative_to(export).as_posix()]=q
 # Only explicit reviewed helpers and actual small receipts; never private diagnostic configs/raw media.
 for name in ['export_completed05_checkpoint.py','assemble_completed05_packet.py','copy_completed05_to_report.py','completed05-helper-USAGE.txt','completed05-helper-manifest.json']:
  q=W/name;assert q.is_file() and not q.is_symlink();inputs['helpers/'+name]=q
 inputs['final-publication-review.json']=review
 for i,q in enumerate(a.evidence):
  assert not q.is_symlink();q=q.resolve();assert q.is_file() and q.suffix in {'.json','.txt','.log','.patch','.py','.sh','.png','.csv','.md'} and q.stat().st_size<20_000_000;inputs[f'additional-evidence/{i:02}-{q.name}']=q
 before={name:bind(q) for name,q in inputs.items()}
 assert len({item['snapshot']['path'] for item in manifest['review_evidence']})==len(manifest['review_evidence'])
 # Only here can an output be created: all prerequisite checks above must already hold.
 out.mkdir();copied=[]
 for name,q in inputs.items():
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target)
  b=before[name];assert bind(q)==b and target.stat().st_size==b['bytes'] and sha(target)==b['sha256']
  copied.append({'path':name,'bytes':b['bytes'],'sha256':b['sha256'],'source':b})
 report={'schema':1,'lesson':'05_home','source_commit':K,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'export_manifest':bind(em),'final_publication_review':bind(review),'file_count':len(copied),'files':copied,'counts':{'checkpoint_snapshot':148,'official_reviews_bound_in_export':13,'source_bound_audio_tracks':50,'full_native_clips':31,'reused_strict_clips':30,'fresh_strict_clips':1,'preserved_other_lessons':83},'scope':'Copies existing actual export and acceptance evidence; does not execute or author acceptance. 31whole clips:30 reused strict clips plus1 fresh strict Apply/Revert;23571frames/785.7seconds at1x. All50 audio assemblies source-bound to current31scenes. No wholly fresh capture or causal product-fix claim.','claims':{'whole_fresh_capture':False,'human_listening':False,'native_speaker_acceptance':False},'limits':manifest['limits']}
 (out/'packet-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==K
 print(json.dumps({'packet':str(out),'listed_files':len(copied),'manifest':bind(out/'packet-manifest.json')}))
if __name__=='__main__':main()
