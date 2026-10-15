#!/usr/bin/env python3
"""Freeze lesson11 English first, then separately promote approved13 reviews."""
from pathlib import Path
import sys
ROOT=Path('/media/carruthers/mnt3/claude/spacr-illumination-performance-20261014')
sys.path[:0]=[str(ROOT), str(ROOT/'tools'), str(ROOT/'tools/tutorials'), str(ROOT/'tools/tutorials/authoring/tools')]
import argparse, hashlib, json, os
from build_appended_candidate import append_catalogs, CATALOGS
from stage_lesson import write
from render_all_voices import LANGUAGES
p=argparse.ArgumentParser();p.add_argument('phase',choices=['english','translations']);args=p.parse_args()
stage=Path(os.environ['VOICES']);lesson_id='11_classify_ml'
lesson=json.loads((ROOT/'tools/tutorials/lessons'/f'{lesson_id}.json').read_text())
# Guard direct path execution as strictly as the common run_tool entry point.
import subprocess
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
assert head==os.environ['SOURCE_COMMIT'], 'Final source HEAD changed'
lesson_path=ROOT/'tools/tutorials/lessons'/f'{lesson_id}.json'
raw_english_hash=hashlib.sha256(lesson_path.read_bytes()).hexdigest()
assert raw_english_hash==os.environ['APPROVED_ENGLISH_SHA256'], 'Approved raw English changed'
relevant=['spacr','tools/tutorials/lessons/'+lesson_id+'.json',
          *[str(p.relative_to(ROOT)) for p in sorted((ROOT/'tools/tutorials/lessons/reviews').glob(lesson_id+'.*.json'))]]
assert not subprocess.check_output(['git','status','--porcelain','--',*relevant],cwd=ROOT,text=True).strip(), 'App or canonical/review source is dirty'
accepted_path=Path(os.environ['ACCEPTED_CAPTURE_EVIDENCE'])
assert hashlib.sha256(accepted_path.read_bytes()).hexdigest()==os.environ['ACCEPTED_CAPTURE_SHA256']
accepted=json.loads(accepted_path.read_text())
assert accepted['lesson']==lesson_id and accepted['accepted'] is True and accepted['whole_capture_accepted'] is True
assert accepted['english_sha256']==raw_english_hash and accepted['source_commit']==head
capture_source=accepted['capture_source_commit']
assert len(capture_source)==40 and all(c in '0123456789abcdef' for c in capture_source)
assert subprocess.check_output(['git','cat-file','-t',capture_source],cwd=ROOT,text=True).strip()=='commit'
subprocess.run(['git','diff','--quiet',capture_source,'--','spacr'],cwd=ROOT,check=True)

def seed_published(english_only):
    # Preserve already published exact narration; current owning renderer still
    # validates its own fingerprint before skipping. No acceptance is inferred.
    import shutil
    baseline=Path('/media/carruthers/mnt3/claude/n676_stages/host/stage-09-r41-20261014/release-candidate-append-q6mh6nzw')
    manifest_path=baseline/'release-manifest.json'
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest()=='daf0d6db958a00487a7dd3764637ded51d664826fb50d77e51629db7261e6009'
    records={r['path']:r for r in json.loads(manifest_path.read_text())['files']}
    paths=sorted((baseline/'media_host/11_classify_ml/audio').glob('*/*.m4a'))
    assert len(paths)==27
    copied=[]
    for source_audio in paths:
        language,voice=source_audio.parent.name,source_audio.stem
        if english_only and language!='en': continue
        filename=f'lessons_{language}.json'
        old=next(r for r in json.loads((baseline/'web/catalog'/filename).read_text())['lessons'] if r['id']==lesson_id)
        new=next(r for r in json.loads((stage/'catalog'/filename).read_text())['lessons'] if r['id']==lesson_id)
        def authored(r):
            r=dict(r);r.pop('narration_voices',None);return r
        assert authored(old)==authored(new), 'Published narration no longer matches approved text'
        timing=json.loads(source_audio.with_suffix('.json').read_text())
        assert timing['render_inputs']['lesson']==lesson_id and timing['language']==language and timing['voice']==voice
        assert [r['narration'] for r in timing['render_inputs']['scenes']]==[r['narration'] for r in new['scenes']]
        for suffix in ('.m4a','.json'):
            source=source_audio.with_suffix(suffix);relative=source.relative_to(baseline).as_posix();record=records[relative]
            data=source.read_bytes();assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256']
            target=stage/'production'/lesson_id/'audio'/language/(voice+suffix)
            if target.exists():
                # Owning renderer may have refreshed validated timing metadata.
                if suffix=='.m4a':assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
                else:
                    current=json.loads(target.read_text());assert current['render_inputs']['lesson']==lesson_id and current['language']==language and current['voice']==voice
                    assert [r['narration'] for r in current['render_inputs']['scenes']]==[r['narration'] for r in new['scenes']]
            else:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as f:f.write(data)
            copied.append({'source':str(source),'target':str(target),'published_sha256':record['sha256']})
    assert len(copied)==(2 if english_only else 54)
    receipt=stage/('published-English-seed.json' if english_only else'published-27-track-seed.json')
    assert not receipt.exists();write(receipt,{'scope':'Exact published27-track reuse seed; normal current renderer and final staged audio QA still required','baseline_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'files':copied,'new_synthesis':False,'native_speaker_signoff':False,'publication_eligible':False})
if args.phase=='english':
    assert not stage.exists(), 'Preserve existing voice stage'
    published={n:json.loads((ROOT/'docs/source/_extra/tutorials/catalog'/n).read_text()) for n in CATALOGS}
    planned={lesson_id:{language:list(pair[1]) for language,pair in LANGUAGES.items()}}
    catalogs,_=append_catalogs(published,[lesson],planned,{},replace=True)
    stage.mkdir(parents=True)
    for name,catalog in catalogs.items():write(stage/'catalog'/name,catalog)
    write(stage/'production'/lesson_id/'lesson.en.json',lesson)
    write(stage/'source-freeze.json',{'source_commit':os.environ['SOURCE_COMMIT'],'selected':[lesson_id],'lesson_sha256':{lesson_id:os.environ['APPROVED_ENGLISH_SHA256']},'voice_count':50,'track_count':50,'source_dirty':bool(__import__('subprocess').check_output(['git','status','--porcelain','tools/tutorials/lessons'],cwd=ROOT,text=True).strip()),'catalog_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((stage/'catalog').glob('*.json'))},'partitions':[[v for _,items in LANGUAGES.values() for v in items][i::4] for i in range(4)],'native_speaker_signoff':False,'listening_review':False,'publication_eligible':False})
    seed_published(True)
else:
    assert os.environ.get('TRANSLATIONS_APPROVED')=='1', 'Root must approve refreshed13 source-bound reviews after English'
    from apply_translation_review import promote_many, translated_lesson, SPOKEN, CAPTION_ONLY
    files=ROOT/'tools/tutorials/lessons/reviews';reviews={lang:json.loads((files/f'{lesson_id}.{lang}.json').read_text()) for lang in sorted(SPOKEN|CAPTION_ONLY)}
    assert len(reviews)==13
    # Validate every review first; do not leave a partially promoted invalid batch.
    english=json.loads((stage/'catalog/lessons_en.json').read_text())
    actual=next(x for x in english['lessons'] if x['id']==lesson_id)
    for lang,review in reviews.items():
        name=f"{'lessons' if lang in SPOKEN else 'captions'}_{lang}.json"
        previous=json.loads((stage/'catalog'/name).read_text())
        old=next(x for x in previous['lessons'] if x['id']==lesson_id)
        translated_lesson(review,actual,old)
    for review in reviews.values():promote_many([review],stage)

    seed_published(False)
