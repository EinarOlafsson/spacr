from pathlib import Path
import datetime,hashlib,json,re,urllib.request,subprocess
root=Path('/tmp/spacr-implementation-20261001/suggest-capture');out=Path(__file__).parent;site='https://einarolafsson.github.io/spacr/tutorials/';lesson_id='14_make_masks'
publication=json.loads((root/'tools/tutorials/release_candidate/publication-receipt.json').read_text());manifest=json.loads((root/'tools/tutorials/release_candidate/release-manifest.json').read_text());expected={r['path']:r for r in manifest['files']};records=[]
def fetch(url,relative,sha=None,size=None):
 p=out/relative;p.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256();n=0
 with urllib.request.urlopen(url,timeout=60) as response,p.open('wb') as stream:
  while chunk:=response.read(1024*1024):
   n+=len(chunk);assert n<=64*1024*1024;h.update(chunk);stream.write(chunk)
 value=h.hexdigest();assert sha is None or value==sha,(relative,value,sha);assert size is None or n==size
 records.append({'url':url,'path':relative,'sha256':value,'bytes':n});print('MATCH',relative,flush=True);return p
index=fetch(site,'live-index.html').read_text();roots=set(re.findall(r'data-(?:audio|video4k|web)-root="([^"]+)"',index));assert roots=={publication['media_root']};media=publication['media_root']
lessons={}
for local in sorted((root/'docs/source/_extra/tutorials/catalog').glob('*.json')):
 p=fetch(site+'catalog/'+local.name,'catalog/'+local.name,hashlib.sha256(local.read_bytes()).hexdigest(),local.stat().st_size)
 lessons[local.stem.removeprefix('lessons_').removeprefix('captions_')]=next(r for r in json.loads(p.read_text())['lessons'] if r['id']==lesson_id)
assert len(lessons)==14
english=lessons['en'];source=json.loads((root/f'tools/tutorials/lessons/{lesson_id}.json').read_text());assert len(english['scenes'])==32
for scene in source['scenes']:
 if scene['visual'].startswith('restoration_'):
  published=next(x for x in english['scenes'] if x['visual']==scene['visual']);assert published['narration']==scene['narration']
original_scene_proof={}
for local in sorted((root/'docs/source/_extra/tutorials/catalog').glob('*.json')):
 prior=json.loads(subprocess.check_output(['git','show','648b87bb5^:'+str(local.relative_to(root))],cwd=root));old=next(x for x in prior['lessons'] if x['id']==lesson_id);lang=local.stem.removeprefix('lessons_').removeprefix('captions_');new=lessons[lang]
 remaining=[x for i,x in enumerate(new['scenes']) if i not in (21,22,23,24,25)];assert old['scenes']==remaining
 original_scene_proof[lang]={'prior_scene_count':len(old['scenes']),'unchanged_scenes':len(remaining),'comparison_commit':'648b87bb5^'}
indices=[i for i,s in enumerate(english['scenes']) if s['visual'].startswith('restoration_')];assert indices==[21,22,23,24,25]
for suffix in [f'web/{lesson_id}_silent.mp4','audio/en/af_heart.m4a','audio/en/af_heart.json']:
 rel=lesson_id+'/'+suffix;r=expected['media_host/'+rel];fetch(media+'/'+rel,rel,r['sha256'],r['bytes'])
timing=json.loads((out/lesson_id/'audio/en/af_heart.json').read_text());assert len(timing['scenes'])==32
for i in indices:assert timing['scenes'][i]['text']==english['scenes'][i]['narration']
voice_counts={lang:len(v) for lang,v in english['narration_voices'].items()};assert sum(voice_counts.values())==27
result={'date_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'site':site,'media_root':media,'published_scene_count':32,'current_authoring_restoration_scenes_match_published':True,'authoring_scene_count':len(source['scenes']),'scope_limit':'Other later authoring scenes are not claimed published by F489','original_scene_preservation':original_scene_proof,'restoration_indices':indices,'restoration_scenes':[dict(index=i,catalog=english['scenes'][i],timing=timing['scenes'][i]) for i in indices],'catalog_languages':sorted(lessons),'published_voice_counts':voice_counts,'fresh_artifacts':records,'historical_full_media_readback':{'receipt':'tools/tutorials/release_candidate/publication-receipt.json','commit':publication['commit'],'files':publication['readback']['downloaded_sha256_matched'],'not_repeated':True},'playback_pending':True}
(out/'readback.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('PASS',len(records),'current hosted/immutable artifacts')
