import importlib.util,sys,json,dataclasses,traceback,pathlib,hashlib,subprocess
root=pathlib.Path.cwd();out=pathlib.Path(__file__).parent
class DenyPackage:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=='spacr' or fullname.startswith('spacr.'):
            raise RuntimeError('Package runtime import refused: '+fullname)
sys.meta_path.insert(0,DenyPackage())
spec=importlib.util.spec_from_file_location('_callable_boundary_readonly',root/'tests/test_docstring_correctness.py');m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
def serial(x):
    if dataclasses.is_dataclass(x):return serial(dataclasses.asdict(x))
    if isinstance(x,(set,frozenset)):return sorted(serial(v)for v in x)
    if isinstance(x,(tuple,list)):return [serial(v)for v in x]
    if isinstance(x,dict):return {str(k):serial(v)for k,v in x.items()}
    return x
results={}
for name in ['test_public_callable_inventory_is_source_derived_not_docstring_derived','test_callable_boundary_is_cross_checked_with_i18n_extractor']:
    fn=getattr(m,name);captured={}
    def trace(frame,event,arg):
        if event=='exception' and frame.f_code is fn.__code__:
            for k,v in frame.f_locals.items():
                if isinstance(v,(set,frozenset,dict,list,tuple)):
                    try:captured[k]=serial(v)
                    except(TypeError,ValueError):pass
            captured['failure_line']=frame.f_lineno
        return trace
    def global_trace(frame,event,arg):return trace if frame.f_code is fn.__code__ else None
    sys.settrace(global_trace)
    try:
        fn();results[name]={'result':'PASS'}
    except Exception as e:
        results[name]={'result':type(e).__name__,'error':str(e),'failure_line':captured.get('failure_line')};traceback.print_exc()
    finally:sys.settrace(None)
    (out/(name+'.json')).write_text(json.dumps(captured,sort_keys=True,indent=2)+'\n')
    print(name,results[name],flush=True)
assert not any(x=='spacr'or x.startswith('spacr.')for x in sys.modules)
(out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
