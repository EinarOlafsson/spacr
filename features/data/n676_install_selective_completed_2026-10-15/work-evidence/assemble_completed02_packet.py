#!/usr/bin/env python3
"""Copy existing post-record selective02 evidence; never author acceptance."""
from pathlib import Path, PurePosixPath
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import argparse,datetime,hashlib,json,os,shutil,subprocess
W=Path(__file__).resolve().parent
K='14ecf9d1054e15194bb7b9ff01779af061fae875'
BASE_SHA='8065f31b9a758cd0a823efbeb8e631cbc740c1c88ca0ea21d7a41e08eab2dcd4'
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
 a=p.parse_args();export=a.export.resolve();out=a.output.resolve();review=a.final_review.resolve()
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==K
 assert export.parent==W and export.is_dir() and out.parent==W and not out.exists() and not out.is_relative_to(export)
 em=export/'export-manifest.json';manifest=read(em)
 assert manifest['lesson']=='02_install_spacr' and manifest['source_commit']==K and manifest['baseline_manifest_sha256']==BASE_SHA
 assert manifest['checkpoint_file_count']==145 and manifest['evidence_extra_count']==3 and manifest['snapshot_file_count']==148
 assert manifest['translation_review_count']==13 and manifest['preserved_other_lessons']==83 and manifest['unchanged_catalog_lesson_objects']==14 and manifest['retained_audio_sidecar_files']==100
 assert manifest['checks']['record_completed'] is True and manifest['checks']['all84_hosted_cases_passed'] is True and manifest['checks']['coherent145_plus3_snapshot_hashes_and_lengths'] is True
 assert all(manifest['claims'][k] is False for k in ['fresh_capture','complete_native_clips','resynthesized','current_runtime_audio_freshness','human_listening','native_speaker_acceptance'])
 checkpoint=export/'checkpoint';rows=manifest['files'];assert len(rows)==len({r['path'] for r in rows})==148
 for row in rows:
  rel=PurePosixPath(row['path']);assert not rel.is_absolute() and '..' not in rel.parts and rel.as_posix()==row['path']
  q=checkpoint/rel;assert not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
 assert sorted(q.relative_to(checkpoint).as_posix() for q in checkpoint.rglob('*') if q.is_file())==sorted(r['path'] for r in rows)
 for item in manifest['review_evidence']:
  checked(item['source']);snapshot=checked(item['snapshot'])
  assert snapshot.is_relative_to(export), 'Review evidence snapshot must be inside exact export tree'
 assert sha(review)==a.final_review_sha256
 final=read(review);assert final['lesson']=='02_install_spacr' and final['status']=='pass' and final['source_commit']==K
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
  if q.is_file():inputs['export/'+q.relative_to(export).as_posix()]=q
 for q in sorted(W.iterdir()):
  if q.is_file() and not q.is_symlink() and q.suffix in {'.json','.py','.sh','.txt','.log'}:
   inputs['work-evidence/'+q.name]=q
 for folder in ['final-boundary-samples','final-boundary-samples-r2']:
  for name in ['ffmpeg.log','command.json','samples.json']:
   q=W/folder/name
   if q.is_file():inputs['work-evidence/'+folder+'/'+name]=q
 # Explicit existing delivery directories: source handoff, actual generated card, actual final boundary PNGs.
 for folder in ['source-handoff','scene07-command-reference','final-boundary-samples-r2']:
  base=W/folder;assert base.is_dir() and not base.is_symlink(), 'Required actual delivery directory missing'
  for q in sorted(base.rglob('*')):
   assert not q.is_symlink(), 'Never copy a delivery symlink'
   if q.is_file():inputs['work-evidence/'+folder+'/'+q.relative_to(base).as_posix()]=q
 inputs['final-publication-review.json']=review
 for i,q in enumerate(a.evidence):
  q=q.resolve();assert q.is_file() and not q.is_symlink();inputs[f'additional-evidence/{i:02}-{q.name}']=q
 before={name:bind(q) for name,q in inputs.items()}
 assert len({item['snapshot']['path'] for item in manifest['review_evidence']})==len(manifest['review_evidence'])
 # Only here can an output be created: all prerequisite checks above must already hold.
 out.mkdir();copied=[]
 for name,q in inputs.items():
  target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target)
  b=before[name];assert bind(q)==b and target.stat().st_size==b['bytes'] and sha(target)==b['sha256']
  copied.append({'path':name,'bytes':b['bytes'],'sha256':b['sha256'],'source':b})
 report={'schema':1,'lesson':'02_install_spacr','source_commit':K,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'export_manifest':bind(em),'final_publication_review':bind(review),'file_count':len(copied),'files':copied,'counts':{'checkpoint_snapshot':148,'official_reviews_bound_in_export':13,'unchanged_audio_sidecars':100,'unchanged_catalog_objects':14,'preserved_other_lessons':83},'scope':'Copies existing actual export and acceptance evidence; does not execute or author acceptance. Only scene7 generated non-terminal command guidance was replaced. Original17scene430.033333s/12901frames and selected29.533s nativeHome interval retained.','claims':manifest['claims'],'limits':manifest['limits']}
 (out/'packet-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
 assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==K
 print(json.dumps({'packet':str(out),'listed_files':len(copied),'manifest':bind(out/'packet-manifest.json')}))
if __name__=='__main__':main()
