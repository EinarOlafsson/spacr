"""One private Tutorial48 candidate from separately checked audio and pictures.

Both components must name the SAME actual post06 published baseline. This module
never promotes prose, renders, synthesizes, uploads or turns an interim candidate
into a published baseline. Existing19/legacy-only builders remain unchanged.
"""
from pathlib import Path
from copy import deepcopy
import argparse,json,os,tempfile
from stage_lesson import read,write
from check_completed_matrix import digest
from audit_staged_catalogs import CATALOGS
from append_staged_lessons import parse_javascript
from build_audio_patch_candidate import require_published_script_baseline
from build_appended_candidate import copy_preserved_checked
from build_release_candidate import copy_checked
from validate_candidate import validate
import published_scene_audio_refresh48 as audio48
import hit_list_picture_refresh as picture48
LESSON='48_hit_list'
CLOCK_PATH=f'media_host/{LESSON}/video/original-visual-timings.json'

def selected(catalog):
 rows=[r for r in catalog['lessons'] if r.get('id')==LESSON]
 if len(rows)!=1:raise ValueError('Exactly one Hit List object required')
 return rows[0]

def joined_source(stage,baseline):
 """Metadata/component provenance only; no final AAC checks occur here."""
 source,current=audio48._source(stage,baseline)
 visual,english,_=picture48.checked_source(stage)
 baseline=Path(baseline).resolve()
 if (Path(source['baseline_root']).resolve()!=baseline or Path(visual['baseline_root']).resolve()!=baseline
     or source['baseline_manifest_sha256']!=visual['manifest_sha256'] or source['publication_receipt_sha256']!=visual['publication_receipt_sha256']
     or source['original_visual_clock']!=visual['original_clock']):raise ValueError('Audio/pictures must share one actual published source and original clock')
 if (source['baseline_manifest_sha256']!=digest(baseline/'release-manifest.json') or source['publication_receipt_sha256']!=digest(baseline/'publication-receipt.json')
     or english!=read(stage/'production'/LESSON/'lesson.en.json') or audio48.common.identity(english)!=audio48.CURRENT_ENGLISH_SHA256
     or {k:v for k,v in current['lessons_en.json'].items() if k!='narration_voices'}!=english):raise ValueError('Current reviewed source/baseline binding differs')
 return source,visual,current

def checked_catalogs(stage,baseline,current):
 catalogs={}
 for name in CATALOGS:
  before=read(audio48.common.regular(baseline,'web/catalog/'+name));after=read(audio48.common.regular(stage,'catalog/'+name))
  ids=[r['id'] for r in before['lessons']]
  if len(ids)!=84 or len(set(ids))!=84 or ids!=[r['id'] for r in after['lessons']]:raise ValueError('Keep all84 catalog identities/order')
  old,new=selected(before),selected(after)
  if new!=current[name]:raise ValueError('Catalog differs from exact checked48 source/review')
  audio48.source_scope(old,new)
  expected=deepcopy(before);expected['lessons'][ids.index(LESSON)]=deepcopy(new)
  if expected!=after:raise ValueError('Other83 lesson objects or catalog metadata changed')
  catalogs[name]=after
 return catalogs

def runtime_catalog(before,old_english,new_english,original_clock,replaced_paths,inventory):
 """Replace only48 prerequisite/scenes and its explicit unmodified picture clock."""
 after=deepcopy(before);ids=[r['id'] for r in before['lessons']]
 if len(ids)!=84 or len(set(ids))!=84 or ids.count(LESSON)!=1:raise ValueError('Exact84 runtime routes required')
 old=selected(before);row=after['lessons'][ids.index(LESSON)]
 if 'native_idle_tails' in old or old.get('native_live_video'):raise ValueError('Legacy48 must not acquire native timing/policy')
 for key in ('title','description','objectives','prerequisite','scenes'):
  if old.get(key)!=old_english.get(key):raise ValueError('Original48 runtime/source text differs')
 for key in ('narration_voices','app_key','host_app_key','status'):
  if old.get(key)!=old_english.get(key):raise ValueError('Original48 runtime routing/offerings differs')
 relative=original_clock['path'];old_reference=old.get('visual_timings',f'{LESSON}/audio/en/af_heart.json')
 if relative!='media_host/'+old_reference:raise ValueError('Explicit original visual-clock path differs')
 row['prerequisite']=new_english['prerequisite'];row['scenes']=deepcopy(new_english['scenes'])
 # If originalEN timing also drove pictures, retain its OLD bytes at an explicit
 # video clock path BEFORE the changed narration sidecar replaces that audio path.
 clock_path=relative
 if relative in replaced_paths:
  clock_path=CLOCK_PATH
  if clock_path in inventory and inventory[clock_path]['sha256']!=original_clock['sha256']:raise ValueError('Existing dedicated clock has a different source')
 row['visual_timings']=clock_path.removeprefix('media_host/')
 expected=deepcopy(before);expected['lessons'][ids.index(LESSON)]=deepcopy(row)
 if expected!=after:raise ValueError('Other83 runtime objects changed')
 return after,clock_path

def require_union(before,after,replaced_paths,clock_path):
 allowed=set(replaced_paths);old=set(before);new=set(after)
 extra={clock_path}-old
 if new!=old|extra or not allowed<=old or extra-{CLOCK_PATH}:raise ValueError('Unexpected candidate inventory addition/removal')
 for path in old-allowed:
  if after[path]!=before[path]:raise ValueError('Unselected bytes changed: '+path)
 if extra and after[clock_path]['path']!=clock_path:raise ValueError('Explicit clock record mismatch')

def build(stage,baseline):
 if Path(stage).is_symlink() or Path(baseline).is_symlink():raise ValueError('Independent stage/baseline required')
 stage,baseline=Path(stage).resolve(),Path(baseline).resolve()
 if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise ValueError('Never build within published source')
 if any(stage.glob('release-candidate-*')):raise FileExistsError('Reuse an existing actual candidate on retry; never rebuild over prior work')
 validate(baseline,include_hosted_media=True)
 old=read(baseline/'release-manifest.json');require_published_script_baseline(baseline,old)
 if old.get('refreshed_lessons')!=['06_api'] or old['routes']!=84 or old['ready_lessons']!=84:raise ValueError('Actual post06 published84 baseline required')
 before={r['path']:r for r in old['files']}
 if len(before)!=len(old['files']):raise ValueError('Duplicate baseline records')
 source,visual,current=joined_source(stage,baseline);catalogs=checked_catalogs(stage,baseline,current)
 media=picture48.checked_finished_media(stage)
 if set(media)!={'master','web','poster','original_clock'}:raise ValueError('Exact finished visual outputs required')
 expected_media=dict(master=f'production/{LESSON}/video/{LESSON}_silent.mp4',web=f'web-renditions/{LESSON}/video/{LESSON}_silent.mp4',poster=f'production/{LESSON}/poster.jpg',original_clock=f'production/{LESSON}/video/original-visual-timings.json')
 for key,relative in expected_media.items():
  if Path(media[key]).resolve()!=audio48.common.regular(stage,relative).resolve():raise ValueError('Finished media belongs to another stage')
 if digest(media['original_clock'])!=source['original_visual_clock']['sha256']:raise ValueError('ChangedEN audio must never become visual clock')
 expected={(language,voice) for language,voices in current['lessons_en.json']['narration_voices'].items() for voice in voices}
 if len(expected)!=27:raise ValueError('Exact27 output offerings required')
 prospective={f'media_host/{LESSON}/audio/{language}/{voice}{suffix}' for language,voice in expected for suffix in ('.m4a','.json')}
 prospective|={'web/catalog/'+name for name in CATALOGS}
 prospective|={f'media_host/{LESSON}/video/{LESSON}_silent.mp4',f'media_host/{LESSON}/web/{LESSON}_silent.mp4',f'web/production/{LESSON}/poster.jpg'}
 old_runtime=parse_javascript(audio48.common.regular(baseline,'web/lesson_catalog.js').read_text())
 old_english=selected(read(baseline/'web/catalog/lessons_en.json'))
 runtime,clock_path=runtime_catalog(old_runtime,old_english,current['lessons_en.json'],source['original_visual_clock'],prospective,before)
 replaced_paths=prospective|{'web/lesson_catalog.js'}
 if not replaced_paths<=set(before):raise ValueError('Original audio/catalog/visual record missing')
 # Snapshot the cheap bindings just checked, before the expensive AAC gate.
 media_hashes={key:digest(path) for key,path in media.items()}
 catalog_hashes={name:digest(stage/'catalog'/name) for name in CATALOGS}
 record_hashes={str(stage/'production'/LESSON/name):digest(stage/'production'/LESSON/name) for name in (audio48.RECORD,picture48.RECORD)}
 # Exactly one call owns the27 normal final AAC decodes. No renderer, cache
 # synthesis or sourceAAC decode is allowed in this candidate step.
 declared,tracks=audio48.checked_tracks(stage,baseline,current['lessons_en.json'],catalogs,decode=True)
 if (declared!=current['lessons_en.json']['narration_voices'] or len(expected)!=27 or len(tracks)!=27
     or {(t['language'],t['voice']) for t in tracks}!=expected):raise ValueError('Exact27 checked output offerings required')
 replacements={};planned=[]
 for t in tracks:
  prefix=f'media_host/{LESSON}/audio/{t["language"]}/{t["voice"]}'
  for key,suffix in (('audio','.m4a'),('timing','.json')):
   path=audio48.common.regular(stage,f'production/{LESSON}/audio/{t["language"]}/{t["voice"]}{suffix}')
   if digest(path)!=t[key+'_sha256']:raise ValueError('Checked audio/timing changed before copy')
   replacements[prefix+suffix]=(path,t[key+'_sha256'])
  planned.append(dict(lesson=LESSON,language=t['language'],voice=t['voice'],audio_sha256=t['audio_sha256'],timing_sha256=t['timing_sha256'],original_audio_sha256=before[prefix+'.m4a']['sha256']))
 for name in CATALOGS:
  path=audio48.common.regular(stage,'catalog/'+name);replacements['web/catalog/'+name]=(path,catalog_hashes[name])
 for key,relative in [('master',f'media_host/{LESSON}/video/{LESSON}_silent.mp4'),('web',f'media_host/{LESSON}/web/{LESSON}_silent.mp4'),('poster',f'web/production/{LESSON}/poster.jpg')]:replacements[relative]=(media[key],media_hashes[key])
 if set(replacements)!=prospective:raise ValueError('Replacement scope changed after validation')
 frozen={str(path):expected_hash for path,expected_hash in replacements.values()};frozen[str(media['original_clock'])]=media_hashes['original_clock'];frozen.update(record_hashes)
 target=Path(tempfile.mkdtemp(prefix='release-candidate-hit-list-combined-',dir=stage));records=[]
 for row in old['files']:
  relative=row['path']
  if relative=='web/lesson_catalog.js':continue
  if relative in replacements:
   path,expected_hash=replacements[relative];copy_checked(path,target/relative,records,target,expected_hash)
  else:copy_preserved_checked(audio48.common.regular(baseline,relative),target/relative,records,target,row['sha256'],hardlink=True)
 runtime_path=target/'web/lesson_catalog.js';runtime_path.write_text('"use strict";\nwindow.SPACR_LESSON_CATALOG = Object.freeze('+json.dumps(runtime,ensure_ascii=False)+');\n')
 records.append(dict(path='web/lesson_catalog.js',sha256=digest(runtime_path),bytes=runtime_path.stat().st_size))
 if clock_path not in before:copy_checked(media['original_clock'],target/clock_path,records,target,source['original_visual_clock']['sha256'])
 after={r['path']:r for r in records};require_union(before,after,replaced_paths,clock_path)
 if after[clock_path]['sha256']!=source['original_visual_clock']['sha256'] or digest(target/clock_path)!=digest(media['original_clock']):raise ValueError('Explicit original clock was not retained')
 if after['web/app_v2.js']!=before['web/app_v2.js']:raise ValueError('Player changed')
 if any(digest(path)!=value for path,value in frozen.items()):raise ValueError('Checked stage inputs changed during candidate assembly')
 if digest(baseline/'release-manifest.json')!=source['baseline_manifest_sha256'] or digest(baseline/'publication-receipt.json')!=source['publication_receipt_sha256']:raise ValueError('Published baseline binding changed during assembly')
 report=deepcopy(old)
 for key in ('legacy_picture_refresh','legacy_picture_references','metadata_refresh','retained_narration','audio_patch_player','new_lesson_tracks'):report.pop(key,None)
 report.update(scope='Private explicit Hit List sentence/prerequisite and source-bound picture/header refresh',files=sorted(records,key=lambda r:r['path']),baseline_manifest_sha256=source['baseline_manifest_sha256'],refreshed_lessons=[LESSON],preserved_lessons=83,appended_lessons=[],withdrawn_lessons=[],script_refresh=True,hit_list_combined_refresh=True,fresh_capture=False,raw_capture_revalidated=False,current_whole_track_tts_claimed=False,release_hold=True,uploaded=False,published=False,audio_patch_tracks=planned,original_visual_timings=[dict(lesson=LESSON,path=clock_path.removeprefix('media_host/'),original_visual_timing_sha256=source['original_visual_clock']['sha256'])],hit_list_components=dict(audio_record_sha256=frozen[str(stage/'production'/LESSON/audio48.RECORD)],picture_record_sha256=frozen[str(stage/'production'/LESSON/picture48.RECORD)]))
 report['web_bytes']=sum(r['bytes'] for r in records if r['path'].startswith('web/'));report['media_host_bytes']=sum(r['bytes'] for r in records if r['path'].startswith('media_host/'))
 if report['web_bytes']>report['ceiling_bytes']:raise ValueError('Existing web budget exceeded')
 write(target/'release-manifest.json',report);validate(target,include_hosted_media=True)
 print('CANDIDATE',target,flush=True);return target

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);a=p.parse_args();build(a.stage,a.baseline)
if __name__=='__main__':main()
