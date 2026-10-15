"""Pure file/hash plan binding; does not import any collector or writer."""
import ast
import hashlib
import json
import subprocess
from pathlib import Path
ROOT=Path.cwd()
OUT=Path('/mnt/wd4tb/scratch/home-b-n180-figure-producer-20261015')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
source_plan=Path('features/data/180_session_figure_inputs_2026-10-15/sources.json')
source=json.loads(source_plan.read_text())
for name,want in source['source_files'].items():assert sha(Path(name).read_bytes())==want,name
languages=['sv','de','es','zh_CN','pt','hi','ko','is','fr']
tracked=subprocess.check_output(['git','ls-files','-z','spacr','tools','docs/i18n/reviewed'],text=True).split('\0')
inputs={}
for name in tracked:
 if not name:continue
 if name.startswith('spacr/qt/i18n_catalogs/') or name=='spacr/qt/help_api_index.py':continue
 if name.endswith('.py') or name.startswith('docs/i18n/reviewed/') and name.endswith('.json'):
  inputs[name]=sha(Path(name).read_bytes())
inputs[str(source_plan)]=sha(source_plan.read_bytes())
inputs['docs/setting_consumers.json']=sha(Path('docs/setting_consumers.json').read_bytes())
outputs=['docs/source/_static/i18n/api/'+lang+'.json' for lang in ['en',*languages]]
outputs+=['spacr/qt/i18n_catalogs/'+lang+'.py' for lang in ['en',*languages]]
outputs+=['docs/i18n/api_source_history.json','spacr/qt/help_api_index.py','tests/data/release_contracts/runtime_caption_arrivals_2026_10_08.json.gz']
base=subprocess.check_output(['git','rev-parse','origin/nightly'],text=True).strip()
plan={'schema':1,'status':'PREPARED ONLY; execution requires coordinator writer-window release and independent driver review','public_base':base,'private_preparation_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_commits':['288f17d7a2','bf9b5dd0ba'],'input_commit':'c9e3dfadf3','languages':languages,'source_files':source['source_files'],'source_docs':source['source_docs'],'new_symbols':source['new_symbols'],'changed_blocks':source['changed_blocks'],'runtime_source':source['runtime_arrivals'][0]['source'],'input_pins':inputs,'baseline_outputs':{name:sha(Path(name).read_bytes()) for name in outputs},'driver_sha256':sha((OUT/'generate.py').read_bytes()),'normal_help':'python tools/build_help_search_index.py after coherent API/runtime catalogs and runtime arrival refresh','followup_acceptance':'Actual Help/compiled-locale consumer script prepared by accessibility peer. Strict named 2-symbol API arrival fixture owned by Original; no historical floor change.'}
(OUT/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
(OUT/'before-help_api_index.py').write_bytes(Path('spacr/qt/help_api_index.py').read_bytes())
ast.parse((OUT/'generate.py').read_text())
print(json.dumps({'status':'PREPARED_NOT_RUN','public_base':base,'inputs':len(inputs),'baseline_outputs':len(outputs),'driver_sha256':plan['driver_sha256'],'plan_sha256':sha((OUT/'plan.json').read_bytes())}))
