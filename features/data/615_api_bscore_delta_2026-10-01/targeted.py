"""Resume completed checkpoints with exact-input caches and unchanged gates."""
import sys
import os
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
import threading
import time
from functools import lru_cache, wraps
from pathlib import Path

root = Path('/tmp/spacr-implementation-20261001/suggest-capture')
sys.path.insert(0, str(root / 'tools'))
import build_i18n_catalogs as catalogs

catalogs.SECONDARY_MODEL_FOLDER = '/tmp/spacr-translation-no-7b-fallback'
import build_documentation_i18n as builder

context_cache = lru_cache(maxsize=65536)(builder._api_translation_source)
normalization_cache = lru_cache(maxsize=65536)(catalogs._simplify_chinese_prose)
builder._api_translation_source = context_cache
catalogs._simplify_chinese_prose = normalization_cache
if hasattr(builder, '_simplify_chinese_prose'):
    builder._simplify_chinese_prose = normalization_cache
assert builder._has_traditional_chinese_prose.__globals__['_simplify_chinese_prose'] is normalization_cache
assert builder._contextualize.__globals__['_simplify_chinese_prose'] is normalization_cache

stages = []
started = time.monotonic()
finished = threading.Event()


def cache_status():
    return f'context={context_cache.cache_info()} OpenCC={normalization_cache.cache_info()}'


def log_stage(name):
    original = getattr(builder, name)

    @wraps(original)
    def logged(*args, **kwargs):
        locale = f' {args[1]}' if len(args) > 1 else ''
        stage = name + locale
        stages.append(stage)
        begin = time.monotonic()
        print(f'BEGIN {stage}', flush=True)
        try:
            result = original(*args, **kwargs)
            if name == 'write_language':
                print(f'{args[1]}: atomic catalog write complete; {cache_status()}', flush=True)
            return result
        finally:
            print(f'END {stage} elapsed={time.monotonic() - begin:.1f}s', flush=True)
            stages.pop()

    setattr(builder, name, logged)


for name in ('public_docstrings', '_proven_api_history', 'reviewed_api_block_translations',
             'repair_api_translations', '_translate_blocks', 'write_language', 'audit'):
    log_stage(name)


def heartbeat():
    while not finished.wait(30):
        snapshot = tuple(stages)
        print(f'HEARTBEAT elapsed={time.monotonic() - started:.1f}s '
              f'stage={" > ".join(snapshot) or "between stages"}; {cache_status()}', flush=True)


threading.Thread(target=heartbeat, daemon=True).start()
import argparse, json, subprocess
folder=Path(__file__).parent
keys={'spacr.sp_stats'}
docs=builder.public_docstrings()
subset={key:docs[key] for key in sorted(keys)}
original_english=json.loads((folder/'en-before.json').read_text())['symbols']
assert set(docs)==set(original_english)
assert {key for key,value in docs.items() if value!=original_english[key]['text']}==keys
args=argparse.Namespace(device='cpu',threads=1,batch_size=8,beams=4,force=False)
proof={}
try:
    for language in builder.MODEL_SPECS:
        relative=f'docs/source/_static/i18n/api/{language}.json'
        before=json.loads(subprocess.check_output(['git','show','HEAD:'+relative],cwd=root))
        current=json.loads((root/relative).read_text())
        if True:
            repaired=builder.repair_api_translations(subset,language,builder.default_model_root(),args)
            assert set(repaired)==keys
            translations={key:entry['text'] for key,entry in current['symbols'].items()}
            translations.update(repaired)
            builder.write_language(docs,language,translations)
        after=json.loads((root/relative).read_text())
        assert set(before['symbols'])==set(after['symbols'])
        changed={key for key in before['symbols'] if before['symbols'][key]!=after['symbols'][key]}
        assert changed==keys, (language,changed)
        assert {key:value for key,value in before.items() if key!='symbols'}=={key:value for key,value in after.items() if key!='symbols'}
        before_blocks,_=builder.translatable_blocks(before['symbols']['spacr.sp_stats']['text'])
        after_blocks,_=builder.translatable_blocks(after['symbols']['spacr.sp_stats']['text'])
        assert len(before_blocks)==len(after_blocks)
        changed_blocks=[i for i,(a,b) in enumerate(zip(before_blocks,after_blocks)) if a!=b]
        assert changed_blocks==[5],(language,changed_blocks)
        proof[language]={'changed_blocks':changed_blocks,'repair_mode':'unchanged repair API with intentional one-symbol subset','changed_symbols':sorted(changed),'unrelated_records_unchanged':len(before['symbols'])-len(changed)}
        print(f'{language}: exact one-symbol diff verified; all other records unchanged',flush=True)
    (folder/'catalog-diff-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    builder._write_english_api_manifest(docs)
    print('All nine catalogs written; beginning unchanged full strict audit.',flush=True)
    status=builder.audit(docs,tuple(builder.MODEL_SPECS))
    (folder/'audit-result.json').write_text(json.dumps({'status':status,'languages':list(builder.MODEL_SPECS),'symbols':len(docs),'resource_limit':'4 GiB, CPU 0, zero swap','source_binding':'normal reviewed_api_block_translations loader; current source/hash/context and target gates unchanged'},indent=2)+'\n')
finally:
    finished.set()
    print(f'Final cache counts: {cache_status()}',flush=True)
sys.exit(status)
