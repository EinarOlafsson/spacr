from pathlib import Path
import ast,copy,gzip,hashlib,io,json,subprocess,tokenize
root=Path.cwd();scratch=Path(__file__).parent;rel='spacr/qt/widgets/model_zoo_picker.py';before=subprocess.check_output(['git','show','HEAD:'+rel]);after=(root/rel).read_bytes();old=ast.parse(before);new=ast.parse(after)
sha=lambda b:hashlib.sha256(b).hexdigest()
comments=lambda raw:[t.string for t in tokenize.tokenize(io.BytesIO(raw).readline) if t.type==tokenize.COMMENT]
assert comments(before)==comments(after)
docs=lambda tree:[ast.get_docstring(n,clean=False) for n in ast.walk(tree) if isinstance(n,(ast.Module,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef))]
assert docs(old)==docs(new)
normalized=copy.deepcopy(new)
for node in normalized.body:
 if isinstance(node,ast.ImportFrom) and node.module=='i18n':node.names=[ast.alias(name='tr')]
 if isinstance(node,ast.ClassDef) and node.name=='SourceStrip':
  init=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
  loop=next(n for n in init.body if isinstance(n,ast.For))
  assert isinstance(loop.body[2],ast.If) and isinstance(loop.body[3],ast.If)
  assert ast.unparse(loop.body[2].test)=="name == 'spaCR community'"
  assert ast.unparse(loop.body[3].test)=='name in _SOURCE_TOOLTIPS'
  del loop.body[2:4]
assert ast.dump(old,include_attributes=False)==ast.dump(normalized,include_attributes=False)
rawdict=lambda tree:ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_SOURCE_TOOLTIPS' for t in n.targets)))
assert rawdict(old)==rawdict(new)
assert '10 failed in 4.35s' in (scratch/'before.log').read_text();assert '10 passed in 2.72s' in (scratch/'after.log').read_text()
out=root/'features/data/440_source_picker_rendering_2026-10-15';out.mkdir(parents=True,exist_ok=True)
for name in ['before.log','before.time','after.log','after.time','extractor-owner-request.json']:(out/(name+'.gz')).write_bytes(gzip.compress((scratch/name).read_bytes(),mtime=0))
receipt={'result':'Targeted renderer10FAIL-before to10PASS-after','source_base':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_file':rel,'before_sha256':sha(before),'after_sha256':sha(after),'source_scope':'Only localization imports and two SourceStrip constructor clauses; all other executable AST and comments exact.','all_docstrings_unchanged':True,'public_or_enabled_helper_arrivals':0,'raw_tooltip_dictionary_unchanged':rawdict(new),'canonical_heading_source':'spaCR community','normal_render_helpers':['set_translatable_text','retranslate_widget_tree'],'test_file':'tests/qt/test_model_zoo_source_localization.py','test_sha256':sha((root/'tests/qt/test_model_zoo_source_localization.py').read_bytes()),'test_scope':['Nine current-language actual SourceStrip constructions,5 tooltip+1community labels, unchanged brand names/default2/source IDs/rawdictionary','Real rejected and accepted mouse clicks use canonical guard name and persist canonical settings','Same-widget de/ko/fr/en/de translation roundtrip retains selection without extra changed signals; reopening retains selection'],'test_catalog_boundary':'Synthetic exact-catalog values through actual i18n/Qt; does not claim nine-language authored/catalog acceptance before Easy54 and normal writer.','remaining':['Normal exact five-tooltip dictionary extraction owner; literal community already exposed through existing set_translatable_text helper','Easy54 reviewed inputs and normal runtime generation','Actual nine-language compiled output acceptance'], 'unchanged_warning_sources_and_behavior':True,'no_network_models_downloads_or_api_generation':True}
(out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(out/'seal-source.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({'source_sha256':sha(after),'docs':'unchanged','tests':'10PASS2.72s','owner_request':str(scratch/'extractor-owner-request.json')}))
