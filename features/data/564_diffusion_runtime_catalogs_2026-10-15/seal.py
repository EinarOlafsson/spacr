import gzip, hashlib, json, shutil, subprocess, types
from pathlib import Path
root = Path.cwd()
scratch = Path('/mnt/wd4tb/scratch/home-b-f564-runtime-20261015')
r = json.loads((scratch / 'runtime-receipt.json').read_text())
assert r['full_nine_locale_audit'] == 0
owned = {(dict(setting_labels='SETTING_LABELS', setting_tooltips='SETTING_TOOLTIPS', ui='UI')[x['table']], x['key']) for x in r['owned_sources']}
tables = ['SETTING_LABELS','SETTING_TOOLTIPS','CATEGORY_HELP','UI','MODULE_SUMMARIES']
english_tables = ['SETTING_LABELS','SETTING_TOOLTIPS','CATEGORY_SOURCES','UI_SOURCES','MODULE_SUMMARIES']
counts = {}
for lang in ['en', *r['locales']]:
    path = 'spacr/qt/i18n_catalogs/' + lang + '.py'
    old, new = {}, {}
    exec(compile(subprocess.check_output(['git','show',r['base']+':'+path],text=True),'old','exec'), old)
    exec(compile((root/path).read_text(),'new','exec'), new)
    old_meta = {k:v for k,v in old.items() if not k.startswith('__') and k not in set(tables+english_tables+['SOURCE_HASHES'])}
    new_meta = {k:v for k,v in new.items() if not k.startswith('__') and k not in set(tables+english_tables+['SOURCE_HASHES'])}
    assert old_meta == new_meta
    changed=[]
    for table in (english_tables if lang == 'en' else tables):
        actual_table = {'CATEGORY_SOURCES':'CATEGORY_HELP','UI_SOURCES':'UI'}.get(table,table)
        left, right = old[table], new[table]
        if not isinstance(left,dict):
            left = {k:k for k in left};right={k:k for k in right}
        for key in set(left)|set(right):
            if key not in left or key not in right or left[key] != right[key]:
                assert (actual_table,key) in owned,(lang,table,key)
                changed.append([actual_table,key])
    assert set(old['SOURCE_HASHES']) - set(new['SOURCE_HASHES']) <= owned
    for ident in set(old['SOURCE_HASHES'])|set(new['SOURCE_HASHES']):
        if ident not in owned:
            assert old['SOURCE_HASHES'][ident] == new['SOURCE_HASHES'][ident]
    counts[lang]={'changed_records':changed,'changed_count':len(changed)}
modified=subprocess.check_output(['git','diff','--name-only'],text=True).splitlines()
assert set(modified) == {'spacr/qt/i18n_catalogs/'+lang+'.py' for lang in ['en',*r['locales']]},modified
out=root/'features/data/564_diffusion_runtime_catalogs_2026-10-15'
out.mkdir(parents=True,exist_ok=True)
for name in ['generation.log','generation.time','runtime-receipt.json','canonical-snapshot.json.gz']:
    data=(scratch/name).read_bytes()
    (out/(name if name.endswith('.gz') else name+'.gz')).write_bytes(data if name.endswith('.gz') else gzip.compress(data,mtime=0))
for name in ['scoped-runtime.py','seal.py']:
    shutil.copyfile(scratch/name,out/name)
receipt={'status':'PASS normal reviewed-only F564 runtime closure and exact unrelated record readback',
         'base':r['base'],'scope':'Seven identities per locale: two setting labels, two setting tooltips, three UI captions; 63 reviewed targets.',
         'canonical_calls':r['original_canonical_calls'],'fresh_canonical_equal':True,
         'normal_full_nine_locale_audit':r['full_nine_locale_audit'],
         'elapsed_seconds':r['elapsed_seconds'],'locales':r['locales'],
         'changed_identity_readback':counts,'metadata_retained':True,
         'source_sha256':r['source_sha256'],'catalog_sha256':r['catalog_sha256'],
         'limits':['No model decoding/downloads, backend training, GUI test repetition, API/Help/settings-flow generation or publication','Source and target independent peers are prior evidence; this receipt records normal generation and readback, not a new independent semantic review'],
         'evidence_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'changed_per_catalog':{k:v['changed_count'] for k,v in counts.items()},'elapsed_seconds':r['elapsed_seconds'],'output':str(out)}))
