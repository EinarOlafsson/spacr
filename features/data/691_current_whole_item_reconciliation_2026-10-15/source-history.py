from pathlib import Path
import ast,hashlib,json,subprocess
root=Path.cwd();scratch=Path('/mnt/wd4tb/scratch/n691-current-whole-item-audit-20261015');sha=lambda x:hashlib.sha256(x).hexdigest();old='8fc2aa5601847ccd8f7cff287c36c6a9ee00ba53'
owned={'spacr/model_zoo.py':['is_third_party','_mitonet_record','mitonet_weights','_mitonet_model','_mitonet_input','_mitonet_heads','mitonet_instances','segment_mitonet'],'spacr/qt/screens/make_masks.py':['_mitonet_segmenter','_count_mitonet_tile','_mitonet_visible','_build_mitonet_card','_gate_mitonet_mode'],'spacr/qt/widgets/model_zoo_picker.py':['_third_party_heading_visible','SourceStrip']}
proof={}
for path,names in owned.items():
 current=(root/path).read_bytes();before=subprocess.check_output(['git','show',old+':'+path]);a=ast.parse(current);b=ast.parse(before);items={}
 for name in names:
  aa=next(n for n in ast.walk(a) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);bb=next(n for n in ast.walk(b) if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name);dump=ast.dump(aa);other=ast.dump(bb)
  items[name]={'current_ast_sha256':sha(dump.encode()),'hosted_ast_sha256':sha(other.encode()),'identical':dump==other}
 proof[path]={'current_file_sha256':sha(current),'hosted_file_sha256':sha(before),'file_identical':current==before,'owned':items}
for path in ['tests/test_691_mitonet_backend.py','tests/qt/test_691_mitonet_make_masks.py']:
 current=(root/path).read_bytes();before=subprocess.check_output(['git','show',old+':'+path]);proof[path]={'current_sha256':sha(current),'hosted_sha256':sha(before),'identical':current==before};assert current==before
log=Path('/mnt/wd4tb/scratch/ci-38699600639-current-triage-20261015/failed.log');lines=[l for l in log.read_text().splitlines() if 'PASSED tests/' in l and ('test_691_mitonet_backend.py::' in l or 'test_691_mitonet_make_masks.py::' in l)]
science={}
for p in [Path('/mnt/firecuda2/Claude/scratch/n691')/n for n in ['f1_port.json','f1_port_tile512.json','eval_port.log','eval_port.py','eval_tile.py']]+[Path('/mnt/firecuda2/Claude/scratch/n690/mitonet_f1.json')]:
 science[str(p)]={'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size,'text':p.read_text()}
assert json.loads(science['/mnt/firecuda2/Claude/scratch/n691/f1_port.json']['text'])=={'TP':142,'FP':8,'FN':53,'F1':.823,'dice':.842}
receipt={'current_public':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'hosted_source_commit':old,'source_comparison':proof,'hosted_log':{'path':str(log),'sha256':sha(log.read_bytes()),'exact_owning_PASS_lines':lines,'count':len(lines),'scope':'Historical hosted outcome only; no newtest execution.'},'historical_cpu_science':science,'limits':'Historical source receipts not fresh dataset/model evaluation; no image/weights scanning/network or replay. Whole-item decision pending root.'}
(scratch/'source-history.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'changed_owned_AST':{p:[n for n,x in v.get('owned',{}).items() if not x['identical']] for p,v in proof.items()},'hosted_PASS_count':len(lines)}))
