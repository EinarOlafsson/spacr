"""Normal local caption translator applied only to changed strings, review-only."""
from pathlib import Path
import hashlib,json,os
from narration_audio import split_english_sentences,split_sentences
from translation_delta import plan_delta,finish_delta,localized_sentences,alignment_splitter


def run_delta(args, spoken, captions):
    source_path=args.stage/'catalog/lessons_en.json'
    source_bytes=source_path.read_bytes()
    baseline_bytes=args.baseline_english.read_bytes()
    source=json.loads(source_bytes);baseline=json.loads(baseline_bytes)
    selected=args.lessons
    alignment_bytes=args.sentence_alignments.read_bytes() if args.sentence_alignments else b'[]'
    entries=json.loads(alignment_bytes)
    model=Path(os.environ.get('SPACR_NLLB_MODEL',captions.LOCAL_MODEL))
    if not model.is_dir():raise ValueError(f'Local translation model unavailable: {model}')
    # Same official NLLB routes; reuse the existing caption translation engine
    # for a collection of sentence/metadata units without hidden extra fields.
    captions.LANGUAGES={**captions.LANGUAGES,**spoken.LANGUAGES}
    output=args.stage/'catalog-drafts'
    output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'review-manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for language in args.languages:
        filename=f"{'lessons' if language in spoken.LANGUAGES else 'captions'}_{language}.json"
        local_path=args.baseline_catalogs/filename
        local_bytes=local_path.read_bytes()
        localized=json.loads(local_bytes)
        target=output/filename
        bindings={'source_sha256':hashlib.sha256(source_bytes).hexdigest(),
            'baseline_english_sha256':hashlib.sha256(baseline_bytes).hexdigest(),
            'baseline_localized_sha256':hashlib.sha256(local_bytes).hexdigest(),
            'sentence_alignments_sha256':hashlib.sha256(alignment_bytes).hexdigest(),
            'selected_lessons':selected}
        old=manifest.get(language,{})
        if (target.is_file() and all(old.get(k)==v for k,v in bindings.items())
                and old.get('draft_sha256')==hashlib.sha256(target.read_bytes()).hexdigest()):
            print(f'{language}: exact delta draft retained; semantic review pending',flush=True)
            continue
        split_local=lambda text:localized_sentences(text,language,split_english_sentences)
        if args.sentence_alignments:
            split_local=alignment_splitter(baseline,localized,language,entries,split_local)
        result,strings,destinations,plans=plan_delta(source,baseline,localized,selected,
            split_english_sentences,split_local)
        if any(len(text.split())>120 for text in strings):
            raise ValueError('Split overlong delta sentences before translation')
        units={'schema':1,'lessons':[{'id':'changed_sentence_units','scenes':[{'narration':text} for text in strings]}]}
        translated=captions.translate(units,language,model,args.batch_size,args.threads,args.device) if strings else None
        values=[s['narration'] for s in translated['lessons'][0]['scenes']] if translated else []
        result=finish_delta(result,values,destinations,plans,language,
                            spoken.speech_text if language in spoken.LANGUAGES else None)
        if source_path.read_bytes()!=source_bytes or args.baseline_english.read_bytes()!=baseline_bytes or local_path.read_bytes()!=local_bytes or (args.sentence_alignments and args.sentence_alignments.read_bytes()!=alignment_bytes):
            raise ValueError('Translation source/baseline changed during work')
        target=output/filename
        content=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
        target.write_text(content)
        manifest[language]={'state':'machine_draft_requires_semantic_review',
            'translation_protocol':'visual-key-sentence-delta-v1','selected_lessons':selected,
            **bindings,
            'translated_strings':len(strings),
            'draft_sha256':hashlib.sha256(content.encode()).hexdigest(),'device':args.device}
        manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
        print(f'{language}: translated {len(strings)} changed/new strings; preserved unchanged reviewed text; review pending',flush=True)
