import ast,hashlib,json
from pathlib import Path
r=Path('/tmp/spacr-implementation-20261001/suggest-capture');out=Path(__file__).parent
p=r/'spacr/qt/widgets/restoration_controls.py';tree=ast.parse(p.read_text());sources=sorted({n.args[0].value for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='tr' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str)})
def tables(p):
 result={}
 for n in ast.parse(p.read_text()).body:
  if isinstance(n,ast.Assign):
   for target in n.targets:
    if isinstance(target,ast.Name) and target.id in ('UI','SOURCE_HASHES'):result[target.id]=ast.literal_eval(n.value)
 return result
en=tables(r/'spacr/qt/i18n_catalogs/en.py');locales=['sv','de','es','zh_CN','pt','hi','ko','is','fr'];coverage={}
row_tree=ast.parse((r/'spacr/qt/i18n.py').read_text());rows=next(n.value for n in row_tree.body if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='_ROWS');off=next(v for k,v in zip(rows.keys,rows.values) if isinstance(k,ast.Constant) and k.value=='Off');assert isinstance(off,ast.Call) and off.func.id=='_row';off_values=[ast.literal_eval(v) for v in off.args];assert len(off_values)==9
for lang in locales:
 d=tables(r/f'spacr/qt/i18n_catalogs/{lang}.py');missing=[s for s in sources if s!='Off' and (not d['UI'].get(s) or d['SOURCE_HASHES'].get(('UI',s))!=en['SOURCE_HASHES'].get(('UI',s)))];assert not missing,(lang,missing);assert off_values[locales.index(lang)];coverage[lang]={'current_caption_entries':len(sources),'generated_source_bound_entries':len(sources)-1,'existing_static_row':{'Off':off_values[locales.index(lang)]},'missing_or_stale':0}
a=json.loads((r/'docs/source/_static/i18n/api/en.json').read_text())['symbols'];keys=[k for k in a if k.startswith(('spacr.qt.detect_chain','spacr.qt.widgets.restoration_controls'))]
api={}
for lang in locales:
 b=json.loads((r/f'docs/source/_static/i18n/api/{lang}.json').read_text())['symbols']
 for k in keys:assert b[k]['source_sha256']==a[k]['source_sha256'] and b[k]['source_blocks_sha256']==a[k]['source_blocks_sha256']
 api[lang]={'current_chain_controls_symbols':len(keys)}
guide=r/'docs/source/make_masks.rst';text=guide.read_text();section=text.split('Restore an image with Cellpose 3\n',1)[1].split('\nGrow secondary objects',1)[0]
for phrase in ['separate','Compare','Apply','floating-point','original','checkpoint hash','Noise2Void']:assert phrase in section
result={'caption_sources':sources,'caption_coverage':coverage,'api_scope':keys,'api_source_contracts':api,'full_current_api_audit':'features/data/615_api_bscore_delta_2026-10-01.json','guide':{'path':str(guide.relative_to(r)),'file_sha256':hashlib.sha256(guide.read_bytes()).hexdigest(),'restoration_section_sha256':hashlib.sha256(section.encode()).hexdigest(),'content':section},'scope':'Focused current caption and API source-binding inventory; no new native linguistic or biological acceptance claim'}
(out/'support.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print('PASS',len(sources),'captions ×9;',len(keys),'chain/control API symbols ×9; current guide requirements present')
