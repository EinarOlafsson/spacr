import importlib.util,sys,pathlib,subprocess,ast,json,dataclasses
root=pathlib.Path.cwd();out=pathlib.Path(__file__).parent
class DenyPackage:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='spacr'or fullname.startswith('spacr.'):raise RuntimeError('No package import: '+fullname)
sys.meta_path.insert(0,DenyPackage())
spec=importlib.util.spec_from_file_location('_private_dataclass_boundary',root/'tests/test_docstring_correctness.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
current={v.symbol:v for v in m._public_callable_inventory()}
oldsource=subprocess.check_output(['git','show','b8984d5e45^:tests/test_docstring_correctness.py']);oldtree=ast.parse(oldsource);helpers=[n for n in oldtree.body if isinstance(n,ast.FunctionDef)and n.name in {'_local_generated_fields','_dataclass_constructor_parameters'}];assert len(helpers)==2
exec(compile(ast.Module(body=helpers,type_ignores=[]),'<exact-old-dataclass-helpers>','exec',dont_inherit=True),m.__dict__)
m._public_callable_inventory.cache_clear();prior={v.symbol:v for v in m._public_callable_inventory()}
assert prior.keys()==current.keys()
changes=[]
for key in current:
    a,b=prior[key],current[key]
    if a!=b:
        changes.append({'symbol':key,'added':sorted(b.parameters-a.parameters),'removed':sorted(a.parameters-b.parameters),'before_parameters':sorted(a.parameters),'after_parameters':sorted(b.parameters),'before_required':sorted(a.required_parameters),'after_required':sorted(b.required_parameters),'category':b.category,'prose_unchanged':a.docstring==b.docstring})
r={'helper_change':'b8984d5e45','identical_source':subprocess.check_output(['git','rev-parse','HEAD']).decode().strip(),'changes':changes,'added_count':sum(len(v['added'])for v in changes),'removed_count':sum(len(v['removed'])for v in changes),'package_runtime_imports':[v for v in sys.modules if v=='spacr'or v.startswith('spacr.')]}
assert not r['package_runtime_imports'];(out/'private-field-helper-delta.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
