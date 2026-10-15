import ast,hashlib,json,sys
from pathlib import Path
root=Path.cwd();sys.path.insert(0,str(root/'tools'))
import build_documentation_i18n as api
scratch=Path('/mnt/wd4tb/scratch/n678-shortcut-api-inputs-20261015'); receipt=json.loads((scratch/'collected.json').read_text()); path=root/'spacr/qt/shortcuts.py';tree=ast.parse(path.read_text());docs={}
nodes={'spacr.qt.shortcuts':tree}
for node in tree.body:
 if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
  name='spacr.qt.shortcuts.'+node.name;nodes[name]=node
  if isinstance(node,ast.ClassDef):
   for child in node.body:
    if isinstance(child,(ast.FunctionDef,ast.AsyncFunctionDef)):nodes[name+'.'+child.name]=child
for key in receipt['rendered_symbols']:
 node=nodes[key];docs[key]=api._autoapi_class_doc(node) if isinstance(node,ast.ClassDef) else api._clean_doc(node)
for row in receipt['blocks']:
 key,_,index=row['label'].rpartition('#');blocks,_=api.translatable_blocks(docs[key]);assert blocks[int(index)]==row['source'];assert api._api_translation_source(row['source'])==row['context']
assert hashlib.sha256(path.read_bytes()).hexdigest()==receipt['source_sha256']
(scratch/'scoped-docstrings.json').write_text(json.dumps(docs,ensure_ascii=False,indent=2)+'\n')
print('Seven source-bound canonical bodies; all22 collector blocks match. No full collector or loader.')
