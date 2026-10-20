import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

root=Path.cwd();pin='098f3edabb7712b66ef43df0ccfa4f5d8dc26a37'
assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==pin
sys.path[:0]=[str(root),str(root/'tools')]
from spacr.qt.help_api_index import API_ENTRIES
from spacr.qt.help_index import api_entries, search
from spacr.qt.help_search import api_url, local_docstring
from build_help_search_index import _summary
from build_documentation_i18n import _autoapi_class_doc, _clean_doc

specs=[('spacr.qt.bridge.make_thread','spacr.qt.bridge','spacr/qt/bridge.py',['make_thread'],'bridge make_thread'),('spacr.qt.bridge.RunRegistry','spacr.qt.bridge','spacr/qt/bridge.py',['RunRegistry'],'bridge RunRegistry'),('spacr.qt.bridge.RunRegistry.active','spacr.qt.bridge','spacr/qt/bridge.py',['RunRegistry','active'],'bridge RunRegistry active'),('spacr.qt.widgets.activity_spinner.ActivitySpinner.is_spinning','spacr.qt.widgets.activity_spinner','spacr/qt/widgets/activity_spinner.py',['ActivitySpinner','is_spinning'],'activity_spinner ActivitySpinner is_spinning')]
modules={r[1] for r in specs};assert not modules.intersection(sys.modules),'Help imports unexpectedly loaded target modules'
generation=json.loads((root/'features/data/678_bridge_spinner_coordinator_review_2026-10-15/help-generation.json').read_text());assert generation['owned4_exact_English_summaries']==[r[0] for r in specs]
index_hash=hashlib.sha256((root/'spacr/qt/help_api_index.py').read_bytes()).hexdigest();assert index_hash==generation['output_sha256']
source_hashes={'spacr/qt/bridge.py':'79c9b2e72259fcd18ac7d02707c4bffcba588bd55b154e2026e089863772df5d','spacr/qt/widgets/activity_spinner.py':'909e079fafe5cecd461abe0f404c57f0a4f9faa1d2cbd0d520ccdb6987806bae'}
for path,want in source_hashes.items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==want
manifest=json.loads((root/'docs/source/_static/i18n/api/en.json').read_text())['symbols'];rows=api_entries();assert len(rows)>=len(API_ENTRIES);checked=[]
for key,module,path,names,qualified in specs:
 node=ast.parse((root/path).read_text())
 for name in names:node=next(n for n in node.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and n.name==name)
 expected_local=ast.get_docstring(node) or '';expected_english=_autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else _clean_doc(node)
 summaries=[summary for symbol,summary in API_ENTRIES if symbol==key];assert len(summaries)==1;summary=summaries[0];actual_rows=[row for row in rows if row.title==key];assert len(actual_rows)==1 and actual_rows[0].payload['symbol']==key and actual_rows[0].kind=='api';assert actual_rows[0].description.startswith(summary)
 local=local_docstring(key);assert local==expected_local and local;assert _summary(local)==summary==_summary(manifest[key]['text']);assert manifest[key]['text']==expected_english
 assert not modules.intersection(sys.modules),'Offline AST source reader imported its target'
 ranks=[]
 for query in (key,qualified):
  found=search(rows,query,limit=12);assert found and found[0].title==key,(key,query,[r.title for r in found]);ranks.append({'query':query,'rank':1,'top_three':[r.title for r in found[:3]]})
 checked.append({'symbol':key,'module':module,'summary':summary,'local_doc_sha256':hashlib.sha256(local.encode()).hexdigest(),'local_prose_equals_full_direct_AST_docstring':True,'English_equals_current_normal_AST_rendering':True,'class_constructor_prose_appended_in_English':isinstance(node,ast.ClassDef),'production_provider_rows':len(actual_rows),'rankings':ranks})
before_urls=sorted(modules.intersection(sys.modules))
for row in checked:
 key=row['symbol'];suffix='/'+row['module'].replace('.','/')+'/index.html';fragment='#'+key;english=api_url(key,'en');korean=api_url(key,'ko');assert english.endswith(suffix+fragment),(key,english);assert korean.endswith(suffix+'?lang=ko'+fragment),(key,korean);row.update(english_url=english,korean_url=korean)
after_urls=sorted(modules.intersection(sys.modules));assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==pin
print(json.dumps({'result':'PASS','exact_source_pin':pin,'checked_symbols':checked,'help_index_sha256':index_hash,'source_files_sha256':source_hashes,'actual_production_api_rows':len(rows),'exact_ranking_queries':8,'target_modules_imported_before_URLs':before_urls,'target_modules_imported_after_URL_dependency_loading':after_urls,'scope':'Actual production api_entries/search across complete provider; four unique rows, current full direct-AST local prose, exact current English summaries, normal class-plus-constructor English rendering and en/ko module/class/function links. No provider fabrication, generator execution, canonical collection, model, network, Sphinx or native GUI claim.','reviewer_boundary':'Consumer execution by API-target author, not independent source or target semantic review; those have separate existing source and target peers.'},indent=2))
