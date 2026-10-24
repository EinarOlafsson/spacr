"""Generate a fresh player-only successor of one verified, unpublished prose78 candidate.

The genuinely published baseline remains the source baseline. Codec acceptance is
retained by exact track bytes, never repeated or represented as a fresh decode.
"""
from pathlib import Path, PurePosixPath
from copy import deepcopy
import argparse,json,re,tempfile
from stage_lesson import REPO,read,write
from check_completed_matrix import digest,reconcile_audio
from build_release_candidate import copy_checked
from build_appended_candidate import copy_preserved_checked
from build_audio_patch_candidate import require_published_script_baseline
from append_staged_lessons import parse_javascript
from audit_staged_catalogs import CATALOGS
from validate_candidate import validate
import build_prose_refresh_candidate as prose
LESSON='78_spacr_screens'
BUILDER='99202775d822f5c07188aa5fb30795243adeed6c6dd977cd1754132ec361ca4e'
CODEC='checks/prose-audio-codec-checks.json'
REPAIR='checks/held-player-repair.json'

def regular(root,relative):
 p=PurePosixPath(relative)
 if p.is_absolute() or '..' in p.parts or str(p)!=relative:raise ValueError('Unsafe candidate path')
 path=Path(root)
 for part in p.parts:
  path=path/part
  if path.is_symlink():raise ValueError('Symlink source prohibited')
 if not path.is_file():raise ValueError('Missing regular source')
 return path

def inventory(root,manifest):
 rows=manifest['files'];result={r['path']:r for r in rows}
 if len(result)!=len(rows):raise ValueError('Duplicate inventory')
 for name,row in result.items():
  path=regular(root,name)
  if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise ValueError('Changed inventoried source: '+name)
 return result

def retained_codec(root,manifest,records):
 path=regular(root,CODEC);report=read(path)
 if (report.get('manifest_sha256')!=digest(root/'release-manifest.json') or report.get('tracks_checked')!=50
     or report.get('current_whole_track_tts_claimed') is not False or report.get('human_listening_review') is not False
     or report.get('mixed_record_sha256')!=manifest['prose_mixed_record_sha256']
     or report.get('tracks')!=[dict({k:v for k,v in t.items() if k!='lesson'},errors=[]) for t in manifest['audio_patch_tracks']]):raise ValueError('Original codec proof differs')
 if any(t.get('lesson')!=LESSON for t in manifest['audio_patch_tracks']):raise ValueError('Foreign manifest track')
 keys=[(r['language'],r['voice']) for r in report['tracks']]
 if len(keys)!=50 or len(set(keys))!=50:raise ValueError('Exact original50 offerings required')
 paths={f'media_host/{LESSON}/audio/{l}/{v}{suffix}' for l,v in keys for suffix in ('.m4a','.json')}
 actual={p for p in records if p.startswith(f'media_host/{LESSON}/audio/')}
 if actual!=paths:raise ValueError('Exact100 audio records required')
 for row in report['tracks']:
  for suffix,key in (('.m4a','audio_sha256'),('.json','timing_sha256')):
   if records[f'media_host/{LESSON}/audio/{row["language"]}/{row["voice"]}{suffix}']['sha256']!=row[key]:raise ValueError('Codec track bytes differ')
 hashes={(l,v):records[f'media_host/{LESSON}/audio/{l}/{v}.m4a']['sha256'] for l,v in keys}
 reconcile_audio([report],LESSON,set(keys),hashes)
 return report,{p:records[p] for p in sorted(paths)}

def components(stage,baseline,root,manifest,records,policy_sha):
 """Current-source/visual/policy guards only; no final AAC verifier is called."""
 if digest(REPO/'tools/tutorials/build_prose_refresh_candidate.py')!=BUILDER:raise ValueError('Original builder changed')
 if digest(REPO/'tools/tutorials/idle_tail_policy.json')!=policy_sha:raise ValueError('Current policy changed')
 sources,catalogs=prose.engine.sources(stage,baseline,[LESSON]);source=sources[LESSON]
 prose.bound_source(source,baseline,LESSON)
 if manifest['baseline_manifest_sha256']!=source['baseline_manifest_sha256']:raise ValueError('Foreign published source')
 mixed=regular(stage/'production'/LESSON,prose.engine.RECORD)
 if digest(mixed)!=manifest['prose_mixed_record_sha256'] or read(mixed)['source']!=source:raise ValueError('Mixed source changed')
 for name in CATALOGS:
  if read(regular(root,'web/catalog/'+name))!=catalogs[name]:raise ValueError('Current catalog changed')
 before={r['path']:r for r in read(baseline/'release-manifest.json')['files']}
 runtime=parse_javascript((baseline/'web/lesson_catalog.js').read_text())
 extra,policy,_=prose.checked_visual(stage,baseline,LESSON,source,before,runtime)
 prospective={f'media_host/{LESSON}/audio/{l}/{v}{s}' for l,vs in source['voices'].items() for v in vs for s in ('.m4a','.json')}|{'web/catalog/'+n for n in CATALOGS}|set(extra)
 projected,clock=prose.runtime_projection(runtime,prose.engine.selected(read(baseline/'web/catalog/lessons_en.json'),LESSON),prose.engine.selected(catalogs['lessons_en.json'],LESSON),LESSON,source,prospective,before,policy)
 if parse_javascript((root/'web/lesson_catalog.js').read_text())!=projected:raise ValueError('Runtime/source policy changed')
 prose.require_union(before,records,prospective|{'web/lesson_catalog.js'},clock)
 for name,(path,h) in extra.items():
  if digest(path)!=h or records[name]['sha256']!=h:raise ValueError('Native source changed')
 for l,vs in source['voices'].items():
  for v in vs:
   for suffix in ('.m4a','.json'):
    if digest(regular(stage/'production'/LESSON,f'audio/{l}/{v}{suffix}'))!=records[f'media_host/{LESSON}/audio/{l}/{v}{suffix}']['sha256']:raise ValueError('Actual staged audio changed')
 if records['web/app_v2.js']!=before['web/app_v2.js']:raise ValueError('Original candidate player is not the published player')
 return source

def refresh(stage,baseline,source_candidate,build_logs,*,source_manifest_sha256,build_record_sha256,policy_sha256,player_sha256):
 roots=[Path(p) for p in (stage,baseline,source_candidate,build_logs)]
 if any(p.is_symlink() for p in roots):raise ValueError('Regular roots required')
 stage,baseline,root,logs=[p.resolve() for p in roots]
 if root.parent!=stage or baseline==root or baseline.is_relative_to(stage) or stage.is_relative_to(baseline):raise ValueError('Distinct stage/published baseline required')
 validate(baseline,include_hosted_media=True,require_browser=True)
 require_published_script_baseline(baseline,read(baseline/'release-manifest.json'))
 if digest(regular(root,'release-manifest.json'))!=source_manifest_sha256 or digest(regular(logs,'candidate.json'))!=build_record_sha256:raise ValueError('Actual build binding changed')
 built=read(logs/'candidate.json')
 if (built['candidate']!=str(root) or built['manifest_sha256']!=source_manifest_sha256 or built['builder_sha256']!=BUILDER
     or built['policy_sha256']!=policy_sha256 or built['build_log_sha256']!=digest(regular(logs,'candidate.out'))
     or re.findall(r'^CANDIDATE (.+)$',(logs/'candidate.out').read_text(),re.M)!=[str(root)]
     or any(regular(logs,n).read_text().strip()!='0' for n in ('build.rc','candidate.rc'))):raise ValueError('Original actual build0 required')
 manifest=read(root/'release-manifest.json')
 if (manifest.get('release_hold') is not True or manifest.get('published') is not False or manifest.get('uploaded') is not False
     or manifest.get('refreshed_lessons')!=[LESSON] or manifest.get('prose_refresh') is not True
     or manifest.get('held_player_repair') is not None or (root/'publication-receipt.json').exists()):raise ValueError('One original held unpublished prose78 candidate required')
 records=inventory(root,manifest);validate(root,include_hosted_media=True)
 report,audio_records=retained_codec(root,manifest,records)
 source=components(stage,baseline,root,manifest,records,policy_sha256)
 if {(t['language'],t['voice']) for t in report['tracks']}!={(l,v) for l,vs in source['voices'].items() for v in vs}:raise ValueError('Codec offerings differ from exact source')
 player=regular(REPO,'tools/tutorials/authoring/web/app_v2.js')
 if digest(player)!=player_sha256 or player_sha256==records['web/app_v2.js']['sha256']:raise ValueError('Explicit different current normal player required')
 codec_sha=digest(root/CODEC);frozen={n:(r['sha256'],r['bytes']) for n,r in records.items()}
 target=Path(tempfile.mkdtemp(prefix='release-candidate-player-repair-',dir=stage));new_records=[]
 for name,row in records.items():
  if name=='web/app_v2.js':copy_checked(player,target/name,new_records,target,player_sha256)
  else:copy_preserved_checked(regular(root,name),target/name,new_records,target,row['sha256'],hardlink=True)
 after={r['path']:r for r in new_records}
 if set(after)!=set(records) or any(after[n]!=records[n] for n in records if n!='web/app_v2.js'):raise ValueError('Only player may change')
 lineage=dict(source_candidate=str(root),source_manifest_sha256=source_manifest_sha256,original_codec_report_sha256=codec_sha,build_record_sha256=build_record_sha256,baseline_root=str(baseline),baseline_manifest_sha256=digest(baseline/'release-manifest.json'),publication_receipt_sha256=digest(baseline/'publication-receipt.json'),policy_sha256=policy_sha256,old_player_sha256=records['web/app_v2.js']['sha256'],new_player_sha256=player_sha256,original_audio_records=audio_records,final_aac_repeated=False,player_browser_checks_retained=False,placeholder_checks_retained=False)
 new_manifest=deepcopy(manifest);new_manifest.update(files=sorted(new_records,key=lambda r:r['path']),held_player_repair=lineage)
 new_manifest['web_bytes']=sum(r['bytes'] for r in new_records if r['path'].startswith('web/'))
 if new_manifest['web_bytes']>new_manifest['ceiling_bytes']:raise ValueError('Pages budget exceeded')
 write(target/'release-manifest.json',new_manifest)
 retained=deepcopy(report);retained.update(manifest_sha256=digest(target/'release-manifest.json'),acceptance_reused_from=lineage,scope='Exact original50 final AAC acceptance retained; only player changed; no repeated decode')
 write(target/CODEC,retained);(target/'checks/original-prose-audio-codec-checks.json').write_bytes((root/CODEC).read_bytes());write(target/REPAIR,lineage)
 validate(target,include_hosted_media=True)
 if inventory(root,manifest)!={n:records[n] for n in records} or digest(root/CODEC)!=codec_sha or digest(root/'release-manifest.json')!=source_manifest_sha256:raise ValueError('Original candidate modified')
 if digest(player)!=player_sha256 or components(stage,baseline,root,manifest,records,policy_sha256)!=source:raise ValueError('Frozen source changed during recovery')
 print('CANDIDATE',target,flush=True);return target

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('stage','baseline','source-candidate','build-logs'):p.add_argument('--'+n,type=Path,required=True)
 for n in ('source-manifest-sha256','build-record-sha256','policy-sha256','player-sha256'):p.add_argument('--'+n,required=True)
 a=p.parse_args();refresh(a.stage,a.baseline,a.source_candidate,a.build_logs,source_manifest_sha256=a.source_manifest_sha256,build_record_sha256=a.build_record_sha256,policy_sha256=a.policy_sha256,player_sha256=a.player_sha256)
if __name__=='__main__':main()
