from pathlib import Path
import ast,hashlib,json,os,sqlite3,subprocess,sys
repo=Path('/tmp/spacr-implementation-20261001/suggest-capture');stage=Path(sys.argv[1]).resolve();sys.path.insert(0,str(repo))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
gui=json.loads((stage/'gui-result.json').read_text());journal=Path(gui['journal']);before=json.loads((stage/'gui-journal-hashes.json').read_text())
gui_root=stage/'gui/plate1';replay_root=stage/'replay/plate1'
with sqlite3.connect(gui_root/'measurements/measurements.db') as conn:
    for name in ['cell','nucleus']:
        count=conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0];assert count>0,(name,count);print('GUI_ROWS',name,count,flush=True)
workflow=stage/'replay-workflow'
validate_only='--validate-only' in sys.argv
if not validate_only:
    subprocess.run([sys.executable,'-m','spacr.cli_repro',str(journal),'--export','snakemake','--out',str(workflow),'--plates',str(replay_root/'merged')],check=True)
exported=json.loads(next((workflow/'settings').glob('*.json')).read_text());original=json.loads((journal/'settings.json').read_text());differences={k:[original.get(k),exported.get(k)] for k in original.keys()|exported.keys() if original.get(k)!=exported.get(k)}
assert set(differences)=={'src'},differences
env=os.environ.copy();env['PATH']=str(stage/'bin')+':'+env['PATH']
engine='/tmp/spacr-implementation-20261001/f575-acceptance/engine-env/bin/snakemake'
command=[engine,'--cores','1','--snakefile',str(workflow/'Snakefile'),'--directory',str(workflow),'--printshellcmds']
if not validate_only:
    with (stage/'snakemake.log').open('w') as log:subprocess.run(command,check=True,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=240)
assert list((workflow/'done').glob('*.ok'))
def tables(folder):
    with sqlite3.connect(folder/'measurements/measurements.db') as conn:
        assert conn.execute('PRAGMA integrity_check').fetchone()==('ok',)
        names=[r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
        result={}
        for name in names:
            quote='"'+name.replace('"','""')+'"';cursor=conn.execute('SELECT * FROM '+quote);columns=[r[0] for r in cursor.description]
            def value(v):
                if isinstance(v,str):return v.replace(str(folder),'<PLATE>')
                if isinstance(v,bytes):return {'bytes_hex':v.hex()}
                return v
            rows=[]
            for raw in cursor.fetchall():
                row=dict(zip(columns,raw))
                for key in {'run_status':['run_id','started_utc','stamped_utc'],'settings_history':['run_id','stamped_utc']}.get(name,[]):
                    assert row[key],(name,key)
                    row[key]='<execution-specific>'
                if name in ('settings','settings_history') and row['setting_key']=='png_channel_mapping':
                    mapping=ast.literal_eval(row['setting_value']);assert isinstance(mapping,dict)
                    row['setting_value']=json.dumps(mapping,sort_keys=True)
                rows.append(tuple(value(row[c]) for c in columns))
            rows.sort(key=repr)
            result[name]={'columns':columns,'rows':rows}
        return result
reference=tables(gui_root);actual=tables(replay_root);assert reference==actual,[(name,len(reference[name]['rows']),len(actual.get(name,{}).get('rows',[]))) for name in reference if reference[name]!=actual.get(name)]
def products(folder):
    return {str(p.relative_to(folder)):digest(p) for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in ('.png','.npy','.tif','.tiff') and 'merged' not in p.relative_to(folder).parts}
images=products(gui_root);assert images, 'No actual image products were written';assert images==products(replay_root)
assert before=={str(p.relative_to(journal)):digest(p) for p in journal.rglob('*') if p.is_file()},'Original GUI journal changed'
source=json.loads((stage/'source.json').read_text());assert digest(Path(source['source']))==source['source_sha256'];assert digest(gui_root/'merged'/Path(source['source']).name)==source['private_input_sha256'];assert digest(replay_root/'merged'/Path(source['source']).name)==source['private_input_sha256']
proof={name:{'row_count':len(data['rows']),'columns':data['columns'],'canonical_sha256':hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()} for name,data in reference.items()}
receipt={'passed':True,'engine_version':subprocess.check_output([engine,'--version'],text=True).strip(),'engine_command':command,'settings_differences':differences,'sqlite_tables':proof,'normalization':['Only the explicitly different private plate-root prefix in textual paths becomes <PLATE>.','run_status run_id/started_utc/stamped_utc and settings_history run_id/stamped_utc are execution-specific; their nonempty values are retained in original SQLite files.','The png_channel_mapping dictionary is parsed and key-sorted; recording a journal sorts dictionary keys, so textual repr order is not scientific content.','Every other value and every column match exactly; rows are sorted without relying on insertion order. No numeric tolerance or row-count-only comparison.'],'image_outputs':images,'original_gui_journal_unchanged':True,'original_cached_source_unchanged':True,'private_inputs_unchanged':True,'execution':'Local CPU Snakemake, no container, cluster, GPU or per-well execution.'}
(stage/'acceptance.json').write_text(json.dumps(receipt,indent=2)+'\n');print('REPLAY PASS',len(proof),'tables',len(images),'image products',flush=True)
