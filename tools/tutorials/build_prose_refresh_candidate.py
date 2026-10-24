"""One allowlisted mixed-prose candidate; existing builders stay unchanged.

Source/catalog/visual/policy bindings precede the one final AAC gate. No source
promotion, synthesis, visual preparation, receipt or fresh-TTS claim occurs here.
"""
from pathlib import Path
from copy import deepcopy
import argparse,ast,json,math,tempfile
from stage_lesson import REPO,read,write
from check_completed_matrix import digest
from audit_staged_catalogs import CATALOGS
from append_staged_lessons import parse_javascript,JS_PREFIX,JS_SUFFIX
from build_audio_patch_candidate import require_published_script_baseline
from build_appended_candidate import copy_preserved_checked
from build_release_candidate import copy_checked
from validate_candidate import validate
from idle_tail_policy import bind_policy,load_policy
import published_prose_audio_refresh as engine
TEXT_FIELDS=('title','description','objectives','prerequisite','scenes')
NATIVE={'20_cellpose_masks','78_spacr_screens','80_image_analysis_pathways'}
# Root reviewed all60 indexed frames +12 exact pre-video PNGs: no in-clip actions.
APPROVED80_SOURCE_POLICY={
  "lesson": "80_image_analysis_pathways",
  "master_sha256": "14b26675d3336c30ea5bcb0dc8274d9e9e1cb80c54b956ccf76b42b6abaa30bc",
  "visual_timing_sha256": "e73521d22a21b80962d8bbe1c9ab950cdf736f0cd1d15271b0ad4ca7abf5e1be",
  "scenes": [
    {
      "visual": "00_home",
      "source_start": 0.0,
      "source_end": 23.533333333333335,
      "idle_tail_start": 0
    },
    {
      "visual": "module_foreign",
      "source_start": 23.533333333333335,
      "source_end": 45.900000000000006,
      "idle_tail_start": 0
    },
    {
      "visual": "module_mask",
      "source_start": 45.900000000000006,
      "source_end": 73.4,
      "idle_tail_start": 0
    },
    {
      "visual": "module_measure",
      "source_start": 73.4,
      "source_end": 98.46666666666667,
      "idle_tail_start": 0
    },
    {
      "visual": "module_annotate",
      "source_start": 98.46666666666667,
      "source_end": 127.63333333333334,
      "idle_tail_start": 0
    },
    {
      "visual": "module_embeddings",
      "source_start": 127.63333333333334,
      "source_end": 152.06666666666666,
      "idle_tail_start": 0
    },
    {
      "visual": "module_gate_editor",
      "source_start": 152.06666666666666,
      "source_end": 174.93333333333334,
      "idle_tail_start": 0
    },
    {
      "visual": "module_qc_dashboard",
      "source_start": 174.93333333333334,
      "source_end": 195.9,
      "idle_tail_start": 0
    },
    {
      "visual": "module_experiment_design",
      "source_start": 195.9,
      "source_end": 219.0,
      "idle_tail_start": 0
    },
    {
      "visual": "module_toxoplasma",
      "source_start": 219.0,
      "source_end": 244.33333333333334,
      "idle_tail_start": 0
    },
    {
      "visual": "module_map_barcodes",
      "source_start": 244.33333333333334,
      "source_end": 269.1333333333333,
      "idle_tail_start": 0
    },
    {
      "visual": "home_summary",
      "source_start": 269.1333333333333,
      "source_end": 293.76666666666665,
      "idle_tail_start": 0
    }
  ]
}

def catalog_scope(before,after,lesson,expected):
 ids=[r['id'] for r in before['lessons']]
 if len(ids)!=84 or len(set(ids))!=84 or ids!=[r['id'] for r in after['lessons']]:raise ValueError('Preserve exact84 catalog identities/order')
 if engine.selected(after,lesson)!=expected:raise ValueError('Current selected review projection changed')
 result=deepcopy(before);result['lessons'][ids.index(lesson)]=deepcopy(expected)
 if result!=after:raise ValueError('Other83 catalog objects or global metadata changed')

def bound_source(source,baseline,lesson):
 if (source['lesson']!=lesson or source['selected_lessons']!=[lesson] or Path(source['baseline_root']).resolve()!=baseline
     or source['baseline_manifest_sha256']!=digest(baseline/'release-manifest.json')
     or source['publication_receipt_sha256']!=digest(baseline/'publication-receipt.json')):raise ValueError('Exactly one same published baseline/lesson required')

def joined_picture(source,visual,lesson):
 if (visual.get('lesson')!=lesson or Path(visual['baseline_root']).resolve()!=Path(source['baseline_root']).resolve()
     or visual['manifest_sha256']!=source['baseline_manifest_sha256']
     or visual['publication_receipt_sha256']!=source['publication_receipt_sha256']
     or visual['original_clock']!={k:source['original_visual_clock'][k] for k in ('path','sha256')}):raise ValueError('Picture/audio component source or original clock differs')

def checked_native_policy(lesson,old_row,source,clock,clock_path,master_record,tracks,generated,canonical,*,original_clock=None):
 """Consume normal reused_policy output, never recreate its exclusive receipt."""
 original=source['original_visual_clock'];old_entry=old_row.get('native_idle_tails')
 if lesson=='80_image_analysis_pathways':
  approved=APPROVED80_SOURCE_POLICY;bindings=original.get('published_bindings',{})
  timing_relative=f'media_host/{lesson}/video/native-live-timings.json'
  master_relative=f'media_host/{lesson}/video/{lesson}_silent.mp4'
  if (original.get('kind')!='native_visual_timing' or original.get('path')!=timing_relative
      or original.get('sha256')!=approved['visual_timing_sha256']
      or set(bindings)!={'master','timing'} or bindings['master']!=master_record
      or master_record.get('path')!=master_relative or master_record.get('sha256')!=approved['master_sha256']
      or bindings['timing'].get('path')!=timing_relative or bindings['timing'].get('sha256')!=approved['visual_timing_sha256']):raise ValueError('Only exact reviewed original80 master/clock source')
  if (not isinstance(original_clock,dict) or [(s.get('visual'),s.get('speech_start'),s.get('scene_end'),s.get('native_live_video')) for s in original_clock.get('scenes',[])]
      !=[(f['visual'],f['source_start'],f['source_end'],True) for f in approved['scenes']]):raise ValueError('Only exact twelve reviewed original80 intervals')
  original_row=dict(id=lesson,silent=master_relative.removeprefix('media_host/'),visual_timings=timing_relative.removeprefix('media_host/'))
  bind_policy(dict(schema=1,lessons=[deepcopy(approved)]),list(bindings.values()),[original_row],lambda _path:original_clock)
  reviewed=dict(schema=1,**deepcopy(approved))
  if old_entry is not None and old_entry!=reviewed:raise ValueError('Unexpected prior80 idle policy')
  old_entry=reviewed
 if not isinstance(old_entry,dict) or old_entry.get('schema')!=1:raise ValueError('Original reviewed fences required')
 old_entry=deepcopy(old_entry);old_entry.pop('schema')
 expected=deepcopy(old_entry);expected['visual_timing_sha256']=digest(clock_path)
 if generated!={'schema':1,'lessons':[expected]} or canonical!=[expected]:raise ValueError('Generated/current policy must preserve exact old fences, rebound only to timing')
 records=[master_record,dict(path=original['path'],sha256=digest(clock_path),bytes=clock_path.stat().st_size)]
 row=dict(id=lesson,silent=master_record['path'].removeprefix('media_host/'),visual_timings=original['path'].removeprefix('media_host/'))
 bound=bind_policy(generated,records,[row],lambda _path:clock)[0]['native_idle_tails']
 if len(tracks)!=50:raise ValueError('All50 current speech spans required for idle fence fit')
 for fence in bound['scenes']:
  indices=[i for i,s in enumerate(clock['scenes']) if s['visual']==fence['visual'] and abs(s['speech_start']-fence['source_start'])<=1e-7 and abs(s['scene_end']-fence['source_end'])<=1e-7]
  if len(indices)!=1:raise ValueError('Indexed native fence/source clock mismatch')
  for track in tracks:
   s=track['scenes'][indices[0]];a,b=s.get('speech_start'),s.get('speech_end')
   if any(type(v) not in (int,float) or not math.isfinite(v) for v in (a,b)) or b<=a or fence['idle_tail_start']>b-a:raise ValueError('Essential action fence exceeds complete new speech')
 return bound

def checked_visual(stage,baseline,lesson,source,before,old_runtime):
 root=stage/'production'/lesson;clock=source['original_visual_clock'];original=engine.common.regular(baseline,clock['path'])
 if digest(original)!=clock['sha256']:raise ValueError('Original visual clock changed')
 extra={};frozen={str(original):clock['sha256']};policy=None
 if lesson in NATIVE:
  if source.get('native_adapter_required') is not True or clock.get('kind')!='native_visual_timing':raise ValueError('Explicit native source binding required')
  from published_native_reuse import RECORD,checked_published_native_timing,current_reuse_timing,_published_inputs
  from native_live_timing import verified_narration_tracks
  reuse=read(engine.common.regular(root,RECORD));base,old,bindings,_=_published_inputs(baseline,lesson)
  if Path(reuse['baseline_root']).resolve()!=baseline or clock['published_bindings']!=bindings:raise ValueError('Native reuse/audio published source differs')
  timing=checked_published_native_timing(stage,lesson);current=current_reuse_timing(stage,lesson)
  intervals=lambda t:[(s['scene'],s['visual'],s['speech_start'],s['scene_end'],s.get('native_live_video')) for s in t['scenes']]
  if intervals(current)!=intervals(old):raise ValueError('Original indexed native intervals/flags changed')
  master=engine.common.regular(root,f'video/{lesson}_silent.mp4')
  if digest(master)!=bindings['master']['sha256'] or master.stat().st_size!=bindings['master']['bytes']:raise ValueError('Native master must remain unchanged')
  tracks,proofs=verified_narration_tracks(stage,lesson,read(root/'lesson.en.json'))
  if len(proofs)!=50:raise ValueError('Actual current mixed50 source proofs required')
  canonical=[r for r in load_policy(REPO)['lessons'] if r.get('lesson')==lesson]
  generated=read(engine.common.regular(root,'reused-idle-tail-policy.json'))
  policy=checked_native_policy(lesson,engine.selected(old_runtime,lesson),source,current,timing,bindings['master'],tracks,generated,canonical,original_clock=old)
  if generated is not None:frozen[str(root/'reused-idle-tail-policy.json')]=digest(root/'reused-idle-tail-policy.json')
  extra[clock['path']]=(timing,digest(timing));frozen.update({str(timing):digest(timing),str(root/RECORD):digest(root/RECORD),str(root/'scenes.json'):digest(root/'scenes.json'),str(REPO/'tools/tutorials/idle_tail_policy.json'):digest(REPO/'tools/tutorials/idle_tail_policy.json')})
 elif lesson=='81_sequencing_pathways':
  import simulation_picture_refresh81 as picture
  visual,english,_,_=picture.checked_source(stage);joined_picture(source,visual,lesson)
  if english!=read(root/'lesson.en.json'):raise ValueError('Picture current English differs')
  media=picture.checked_finished_media(stage)
  expected=dict(master=f'production/{lesson}/video/{lesson}_silent.mp4',web=f'web-renditions/{lesson}/video/{lesson}_silent.mp4',poster=f'production/{lesson}/poster.jpg',original_clock=f'production/{lesson}/video/original-visual-timings.json')
  if set(media)!=set(expected):raise ValueError('Exact normal finished picture outputs required')
  for key,path in expected.items():
   if Path(media[key]).resolve()!=engine.common.regular(stage,path).resolve():raise ValueError('Finished picture belongs to another stage')
  if digest(media['original_clock'])!=clock['sha256']:raise ValueError('Changed EN audio cannot supply picture clock')
  for key,path in [('master',f'media_host/{lesson}/video/{lesson}_silent.mp4'),('web',f'media_host/{lesson}/web/{lesson}_silent.mp4'),('poster',f'web/production/{lesson}/poster.jpg')]:extra[path]=(media[key],digest(media[key]))
  frozen.update({str(root/p):digest(root/p) for p in (picture.SOURCE,picture.RECORD)})
 else:
  if source['native_adapter_required'] or clock.get('kind')!='legacy':raise ValueError('Legacy source must retain original legacy clock')
 return extra,policy,frozen

def runtime_projection(before,old_en,new_en,lesson,source,changed,inventory,policy):
 row=engine.selected(before,lesson);after=deepcopy(before);catalog_scope(before,deepcopy(before),lesson,row)
 for key in TEXT_FIELDS+('narration_voices','app_key','host_app_key','status'):
  if row.get(key)!=old_en.get(key):raise ValueError('Original runtime/source projection differs')
 for key in ('narration_voices','app_key','host_app_key','status'):
  if new_en.get(key)!=old_en.get(key):raise ValueError('Offering or route change is not prose refresh')
 target=after['lessons'][[r['id'] for r in after['lessons']].index(lesson)]
 for key in TEXT_FIELDS:
  if key in new_en:target[key]=deepcopy(new_en[key])
  else:target.pop(key,None)
 clock=source['original_visual_clock'];old_path='media_host/'+row.get('visual_timings',f'{lesson}/audio/en/af_heart.json')
 if old_path!=clock['path']:raise ValueError('Runtime must bind original visual clock')
 path=clock['path']
 if lesson in NATIVE:
  if policy is None:target.pop('native_idle_tails',None)
  else:target['native_idle_tails']=deepcopy(policy)
 else:
  if row.get('native_idle_tails') is not None or row.get('native_live_video'):raise ValueError('Never downgrade native source')
  if path in changed:
   path=f'media_host/{lesson}/video/original-visual-timings.json'
   if path in inventory and inventory[path]['sha256']!=clock['sha256']:raise ValueError('Dedicated original clock conflict')
 target['visual_timings']=path.removeprefix('media_host/')
 return after,path

def require_union(before,after,changed,clock_path):
 if set(after)!=set(before)|{clock_path} or not set(changed)<=set(before):raise ValueError('Unexpected file inventory addition/removal')
 for path in set(before)-set(changed):
  if before[path]!=after[path]:raise ValueError('Unselected record changed: '+path)

def peak_limit():
 tree=ast.parse((REPO/'tools/tutorials/authoring/tools/render_all_voices.py').read_text())
 values=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='MAX_RELEASE_TRUE_PEAK_DBFS' for t in n.targets)]
 if len(values)!=1 or type(values[0]) not in (int,float) or not math.isfinite(values[0]):raise ValueError('Exact normal producer true-peak limit required')
 return values[0]

def producer_peak_proofs(record,limit):
 for row in record['tracks']:
  if row['mode']=='source_coordinate_splice':
   peak=row.get('decoded_true_peak_dbfs')
   if type(peak) not in (int,float) or not math.isfinite(peak) or peak>limit:raise ValueError('Mandatory producer codec true-peak gate missing/failed')
  elif row['mode']!='published_audio_copy':raise ValueError('Unknown mixed audio mode')


def refresh_manifest_base(old,lesson):
 """Clear prior producer scope, keeping persistent release metadata/records."""
 report=deepcopy(old)
 # Existing19/audio, legacy-picture, native append and metadata/link producers.
 # These describe the previous refresh, not the unchanged released lessons.
 for key in ('legacy_picture_refresh','legacy_picture_references','metadata_refresh',
             'metadata_only_updates','link_only_updates','retained_narration',
             'retained_visual_timing','web_checks','new_lesson_tracks',
             'audio_patch_player','audio_patch_tracks','original_visual_timings',
             'script_refresh','hit_list_combined_refresh','hit_list_components',
             'prose_refresh','prose_components','prose_visual_source','prose_mixed_record_sha256'):
  report.pop(key,None)
 report.update(refreshed_lessons=[lesson],preserved_lessons=83,appended_lessons=[],withdrawn_lessons=[])
 return report


def build(stage,baseline,lesson):
 engine.render_selection([lesson])
 if Path(stage).is_symlink() or Path(baseline).is_symlink():raise ValueError('Regular independent roots required')
 stage,baseline=Path(stage).resolve(),Path(baseline).resolve()
 if stage==baseline or stage.is_relative_to(baseline) or baseline.is_relative_to(stage):raise ValueError('Independent stage and published baseline required')
 if any(stage.glob('release-candidate-*')):raise FileExistsError('Reuse actual candidate on retry; no rebuild/AAC repetition')
 validate(baseline,include_hosted_media=True,require_browser=True);old=read(baseline/'release-manifest.json');require_published_script_baseline(baseline,old)
 placeholder_relative='checks/placeholder-mutation-checks.json' if (baseline/'checks/placeholder-mutation-checks.json').exists() else 'placeholder-mutation-checks.json'
 placeholder=engine.common.regular(baseline,placeholder_relative);placeholder_hash=digest(placeholder)
 before={r['path']:r for r in old['files']}
 if len(before)!=len(old['files']) or old['routes']!=84 or old['ready_lessons']!=84:raise ValueError('Actual published84 source required')
 sources,catalogs=engine.sources(stage,baseline,[lesson]);source=sources[lesson];bound_source(source,baseline,lesson)
 for name in CATALOGS:catalog_scope(read(baseline/'web/catalog'/name),catalogs[name],lesson,engine.selected(catalogs[name],lesson))
 root=stage/'production'/lesson;record_path=engine.common.regular(root,engine.RECORD);mixed=read(record_path)
 if mixed.get('source')!=source or mixed.get('schema')!=1 or mixed.get('kind')!=engine.KIND:raise ValueError('Mixed audio record/source differs')
 producer_peak_proofs(mixed,peak_limit())
 metadata_voices,metadata_tracks=engine.checked_tracks(stage,baseline,lesson,decode=False)
 if metadata_voices!=source['voices']:raise ValueError('Metadata-only current offering proof differs')
 runtime=parse_javascript((baseline/'web/lesson_catalog.js').read_text());extra,policy,visual_frozen=checked_visual(stage,baseline,lesson,source,before,runtime);frozen=dict(visual_frozen);frozen[str(placeholder)]=placeholder_hash
 expected={(l,v) for l,vs in source['voices'].items() for v in vs}
 if len(expected)!=(50 if lesson in NATIVE else 27):raise ValueError('Exact original offering count required')
 prospective={f'media_host/{lesson}/audio/{l}/{v}{s}' for l,v in expected for s in ('.m4a','.json')}|{'web/catalog/'+n for n in CATALOGS}|set(extra)
 new_runtime,clock_path=runtime_projection(runtime,engine.selected(read(baseline/'web/catalog/lessons_en.json'),lesson),engine.selected(catalogs['lessons_en.json'],lesson),lesson,source,prospective,before,policy)
 changed=prospective|{'web/lesson_catalog.js'}
 if not changed<=set(before):raise ValueError('Missing original replacement records')
 replacements=dict(extra)
 for l,v in expected:
  for suffix in ('.m4a','.json'):
   path=engine.common.regular(root,f'audio/{l}/{v}{suffix}');replacements[f'media_host/{lesson}/audio/{l}/{v}{suffix}']=(path,digest(path))
 for name in CATALOGS:
  path=engine.common.regular(stage,'catalog/'+name);replacements['web/catalog/'+name]=(path,digest(path))
 frozen.update({str(path):h for path,h in replacements.values()});frozen[str(record_path)]=digest(record_path)
 # The sole final27/50 AAC acceptance gate. No fresh-TTS verifier is invoked.
 declared,tracks=engine.checked_tracks(stage,baseline,lesson,decode=True)
 if tracks!=metadata_tracks or declared!=source['voices'] or len(tracks)!=len(expected) or {(t['language'],t['voice']) for t in tracks}!=expected:raise ValueError('Incomplete actual mixed AAC checks')
 for track in tracks:
  for suffix,key in (('.m4a','audio_sha256'),('.json','timing_sha256')):
   if replacements[f'media_host/{lesson}/audio/{track["language"]}/{track["voice"]}{suffix}'][1]!=track[key]:raise ValueError('Final AAC bindings changed')
 if any(digest(path)!=h for path,h in frozen.items()):raise ValueError('Checked stage source changed before candidate construction')
 target=Path(tempfile.mkdtemp(prefix='release-candidate-prose-',dir=stage));records=[]
 for row in old['files']:
  path=row['path']
  if path=='web/lesson_catalog.js':continue
  if path in replacements:
   src,h=replacements[path];copy_checked(src,target/path,records,target,h)
  else:copy_preserved_checked(engine.common.regular(baseline,path),target/path,records,target,row['sha256'],hardlink=True)
 out=target/'web/lesson_catalog.js';out.write_text(JS_PREFIX+json.dumps(new_runtime,ensure_ascii=False)+JS_SUFFIX)
 records.append(dict(path='web/lesson_catalog.js',bytes=out.stat().st_size,sha256=digest(out)))
 if clock_path not in before:copy_checked(baseline/source['original_visual_clock']['path'],target/clock_path,records,target,source['original_visual_clock']['sha256'])
 after={r['path']:r for r in records};require_union(before,after,changed,clock_path)
 if lesson not in NATIVE and after[clock_path]['sha256']!=source['original_visual_clock']['sha256']:raise ValueError('Original legacy clock not preserved')
 if after['web/app_v2.js']!=before['web/app_v2.js']:raise ValueError('Player changed')
 if checked_visual(stage,baseline,lesson,source,before,runtime)!=(extra,policy,visual_frozen):raise ValueError('Visual source or policy changed during assembly')
 if any(digest(path)!=h for path,h in frozen.items()) or engine.sources(stage,baseline,[lesson])!=(sources,catalogs):raise ValueError('Frozen component source changed during assembly')
 report=refresh_manifest_base(old,lesson)
 report.update(scope='Explicit reviewed mixed prose refresh; original offerings and source-bound pictures',files=sorted(records,key=lambda r:r['path']),baseline_manifest_sha256=source['baseline_manifest_sha256'],refreshed_lessons=[lesson],preserved_lessons=83,appended_lessons=[],withdrawn_lessons=[],script_refresh=True,prose_refresh=True,fresh_capture=False,raw_capture_revalidated=False,current_whole_track_tts_claimed=False,release_hold=True,uploaded=False,published=False,audio_patch_tracks=[dict(lesson=lesson,**t) for t in tracks],original_visual_timings=[] if lesson in NATIVE else [dict(lesson=lesson,path=clock_path.removeprefix('media_host/'),original_visual_timing_sha256=source['original_visual_clock']['sha256'])],prose_visual_source=source['original_visual_clock'],prose_mixed_record_sha256=digest(record_path))
 report['web_bytes']=sum(r['bytes'] for r in records if r['path'].startswith('web/'));report['media_host_bytes']=sum(r['bytes'] for r in records if r['path'].startswith('media_host/'))
 if report['web_bytes']>report['ceiling_bytes']:raise ValueError('Existing Pages budget exceeded')
 write(target/'release-manifest.json',report);validate(target,include_hosted_media=True)
 # Exact retained normal mutation proof, valid only because player bytes match.
 (target/'checks').mkdir(exist_ok=True)
 with (target/'checks/placeholder-mutation-checks.json').open('xb') as output:output.write(placeholder.read_bytes())
 if digest(target/'checks/placeholder-mutation-checks.json')!=placeholder_hash:raise ValueError('Retained placeholder proof changed')
 # Truthful normal reconciliation input: every errors[] comes from the above
 # actual final AAC gate. This is not a current-TTS or listening certificate.
 write(target/'checks/prose-audio-codec-checks.json',dict(lesson=lesson,passed=True,tracks_checked=len(tracks),tracks=[dict(t,errors=[]) for t in tracks],provenance_kind=engine.KIND,mixed_record_sha256=digest(record_path),manifest_sha256=digest(target/'release-manifest.json'),scope='Actual final AAC codec acceptance only; producer true-peak verified separately',current_whole_track_tts_claimed=False,human_listening_review=False))
 print('CANDIDATE',target,flush=True);return target

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--stage',required=True,type=Path);p.add_argument('--baseline',required=True,type=Path);p.add_argument('--lesson',required=True,choices=list(engine.POLICIES));a=p.parse_args();build(a.stage,a.baseline,a.lesson)
if __name__=='__main__':main()
