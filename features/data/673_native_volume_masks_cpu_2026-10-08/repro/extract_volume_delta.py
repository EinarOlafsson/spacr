import sys; sys.path.insert(0, str(__import__("pathlib").Path.cwd()/"tools"))
import ast, gzip, hashlib, importlib.util, json, subprocess, tempfile
from pathlib import Path
root=Path.cwd(); scratch=Path('/mnt/wd4tb/scratch/gate-anchors-20261008')
paths=['spacr/qt/mask_engine.py','spacr/qt/screens/make_masks.py']
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,root/path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
docs=load('build_documentation_i18n','tools/build_documentation_i18n.py'); ui=load('build_i18n_catalogs','tools/build_i18n_catalogs.py')
results={}
for label,commit in [('before','ec50d0bbb1e'),('after','2720b462e8a')]:
    with tempfile.TemporaryDirectory(dir=scratch) as folder:
        base=Path(folder)
        for path in paths:
            target=base/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(subprocess.check_output(['git','show',commit+':'+path]))
        for extra in ['spacr/model_compare.py','spacr/embeddings.py']:(base/extra).write_text('')
        docs.ROOT=base;docs.API_DOC_ALIASES={};docs.nested_helper_docs.active_entries=lambda *args,**kwargs:[]
        ui.ROOT=base;ui._indirect_runtime_ui_sources=lambda:set()
        api=docs.public_docstrings(); strings=ui.extract_static_ui_sources()
        objects=[]
        for path in paths:
            for node in ast.walk(ast.parse((base/path).read_text())):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='setObjectName' and node.args and isinstance(node.args[0],ast.Constant):objects.append(node.args[0].value)
        results[label]={'api':api,'ui':strings,'object_names':sorted(set(objects))}
b,a=results['before'],results['after']
packet={'scope':'Normal public_docstrings/extract_static_ui_sources policies applied to two changed files only; unrelated API aliases, nested helper inventory and indirect runtime registries excluded equally in both passes. Not global canonical counts or regenerated catalogs.','before':'ec50d0bbb1e','after':'2720b462e8a','api_added':{k:v for k,v in a['api'].items() if k not in b['api']},'api_changed':{k:{'before':b['api'][k],'after':v} for k,v in a['api'].items() if k in b['api'] and b['api'][k]!=v},'ui_added':sorted(set(a['ui'])-set(b['ui'])),'ui_removed':sorted(set(b['ui'])-set(a['ui'])),'object_names_added':sorted(set(a['object_names'])-set(b['object_names']))}
(scratch/'volume-api-ui-delta.json').write_text(json.dumps(packet,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:len(packet[k]) for k in ['api_added','api_changed','ui_added','ui_removed','object_names_added']}))
