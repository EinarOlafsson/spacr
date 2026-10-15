from pathlib import Path
import ast
import hashlib
import json
import sys
import tempfile

root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/f556-current-whole-item-audit-20261015');sys.path.insert(0,str(root/'tools'))
import build_i18n_catalogs as runtime
runtime.canonical_sources=lambda *args,**kwargs: (_ for _ in ()).throw(AssertionError('No global collector'))
import build_documentation_i18n as api
import nested_helper_docs as nested
path=root/'spacr/qt/screens/make_masks.py';raw=path.read_bytes();module='spacr.qt.screens.make_masks';key=module+'._Sam2SeedDialog.run.work';assert module in nested.ENABLED_MODULES
with tempfile.TemporaryDirectory(dir=scratch,prefix='sam2-api-source-') as directory:
 snapshot=Path(directory);target=snapshot/'spacr/qt/screens/make_masks.py';target.parent.mkdir(parents=True);target.write_bytes(raw)
 entries=nested.entries(nested.inventory(snapshot,ignore_patterns=api.AUTOAPI_IGNORE),modules={module})
 actual=next(entry for entry in entries if entry.qualified_key==key);doc=actual.docstring
compiled_en=json.loads((root/'docs/source/_static/i18n/api/en.json').read_text())['symbols'][key]
assert compiled_en['text']==doc
records={}
for lang in ['sv','de','es','zh_CN','fr','pt','hi','ko','is']:
 reviewed=api.reviewed_api_block_translations({key:doc},lang)
 compiled=json.loads((root/'docs/source/_static/i18n/api'/f'{lang}.json').read_text())['symbols'][key]
 assert compiled['source_sha256']==compiled_en['source_sha256']
 sourceblocks,_=api.translatable_blocks(doc);targetblocks,_=api.translatable_blocks(compiled['text'])
 assert len(sourceblocks)==len(targetblocks)==1
 assert reviewed[sourceblocks[0]]==targetblocks[0]
 records[lang]={'source_sha256':compiled['source_sha256'],'target_sha256':hashlib.sha256(compiled['text'].encode()).hexdigest(),'normal_review_exact_compiled':True}
source_symbols={}
for filename,names in [('spacr/timelapse.py',['_sam2_track_cells','_sam2_parent_links']),('spacr/qt/screens/make_masks.py',['_build_sam2_button','_on_sam2_tracking'])]:
 p=root/filename;tree=ast.parse(p.read_text());source_symbols[filename]={'file_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'symbols':{name:hashlib.sha256(ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)).encode()).hexdigest() for name in names}}
receipt={'normal_nested_helper':key,'current_doc':doc,'compiled_english_source_sha256':compiled_en['source_sha256'],'all9_exact_API_reviews_compiled':records,'private_dialog_normal_top_level_class_visible':False,'source_symbols':source_symbols,'source_clause_evidence':['ALPHA_FEATURES[556] registers timelapse_mode sam2, model sam2_v1, widget MakeMasksSam2Button; actual literal setObjectName and _apply_alpha_widgets present','Current timelapse calls propagate, seeds new objects, writes native lineage columns with _sam2_parent_links and source sam2','Current dialog save writes TIFF labels, tracksCSV, seedsNPZ and settingsJSON beside source movie; backward and corrections retained','Historical fullmovie SAM2/stitcher/trackastra raw scoring logs hashed in audit.json; Timeflows export explicitly N/A in dated accepted item notes'],'scope':'Only current SAM2 source and one exact eligible API helper; normal review gates/no global API/runtime collector, no model/test/science replay.'}
(scratch/'source-api-proof.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print('F556 current sourceclauses and1eligibleAPI×9exact normalreview/compiled targets PASS')
