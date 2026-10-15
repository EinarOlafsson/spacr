"""Source-exact caching counterexample with factory doubles; no model imports."""
import ast
from functools import lru_cache, partial
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from unittest.mock import patch

pin='686793f8642cbff9f9f6d3a25770864139e9089e'
active={'device':'cpu'}
calls=[]

def factory(*args,**kwargs):
    result=types.SimpleNamespace(device=active['device'])
    calls.append({'device':active['device'],'args':list(map(str,args))})
    return result

fake_submodules=types.ModuleType('spacr.submodules')
fake_submodules._plaque_cellpose_model=factory
fake_object=types.ModuleType('spacr.object')
fake_object._robustness_segmenter=lambda *args: partial(lambda model:None,factory(*args))
fake_settings=types.ModuleType('spacr.settings')
fake_settings.set_default_settings_preprocess_generate_masks=lambda settings:settings
results=[]
with patch.dict(sys.modules,{'spacr.submodules':fake_submodules,'spacr.object':fake_object,'spacr.settings':fake_settings}):
    for file,symbol,args in [
        ('tools/web_mask_generator/plaque.py','_load_model',('/operator/approved.pt','same-approved-hash')),
        ('tools/web_mask_generator/masks.py','_load_mask_model',('/operator/approved.pt','same-approved-hash','cell'))]:
        source=subprocess.check_output(['git','show',pin+':'+file]).decode()
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==symbol)
        ns={'lru_cache':lru_cache};exec(compile(ast.Module(body=[node],type_ignores=[]),file,'exec'),ns)
        start=len(calls);active['device']='cpu';first=ns[symbol](*args)
        active['device']='cuda:0';second=ns[symbol](*args)
        assert first is second and second.device=='cpu' and len(calls)==start+1
        results.append({'file':file,'symbol':symbol,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'factory_calls':calls[start:],'second_context':'cuda:0','second_model_device':second.device,'same_cached_instance':first is second,'cache_key_args':list(args)})
output={'source_pin':pin,'method':'Execute exact AST-extracted existing cache wrappers with only stdlib lru_cache/controlled desktop factory doubles. CPU/cuda strings are declared simulated contexts, not actual device probes.','counterexamples':results,'scope':'Actual wrapper caching ignores changed factory execution context; no actual model constructor/decoder, GPU, torch import, network, server or deployment was executed.','source_chain':['build_app(preload=zero_gpu) calls preload_default before defining spaces.GPU execute','preload_default warms plaque/mask loader outside request GPU context','same checkpoint/digest(/role) cache key reused inside inference','accelerator.resolve caches first probe unless refresh=True'],'proposal':'Existing web files only: explicit hosted execution-context device binding/refresh before model construction and context-aware cache keys or controlled transfer. Do not mutate generic desktop accelerator defaults; preserve digest identity. Need native host contract before implementation.','acceptance':'Not a ZeroGPU failure reproduction or hosted acceptance. Deployment packaging still lacks a reviewed operator artifact/provisioned endpoint in this bounded source scope.'}
path=Path('/mnt/wd4tb/scratch/n698-device-context-cache-20261015/counterexample.json');path.write_text(json.dumps(output,indent=2)+'\n');print(path)
