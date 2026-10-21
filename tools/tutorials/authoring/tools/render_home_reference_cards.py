#!/usr/bin/env python3
"""Generate six explicitly authored Home reference cards; no GUI capture claim."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys

DEFINITIONS = {
 'home_module_embeddings': ('Embeddings', ['Object crops → numerical features', 'Explore and compare encoded objects'], ['spacr/qt/screens/embeddings.py']),
 'home_module_experiment_design': ('Experiment Design', ['Plan conditions and wells', 'Assign sample identities before the experiment'], ['spacr/qt/screens/experiment_design.py']),
 'home_module_dose_response': ('Dose-Response', ['Fit responses across concentrations', 'Keep the endpoint and controls explicit'], ['spacr/qt/screens/dose_response.py']),
 'home_module_qc_dashboard': ('Quality control', ['Inspect saved project quality reports', 'Identify reports that are missing'], ['spacr/qt/screens/qc_dashboard.py']),
 'home_module_gate_editor': ('Gate Editor', ['Define selections on real data columns', 'Review selected objects before creating labels'], ['spacr/qt/screens/gate_editor.py']),
 '24_spaceout_field': ('Spaceout field controls', ['Switch eight field effects independently', 'Appearance and settings animation controls remain available', 'Apply previews • Keep accepts • Revert restores'], ['spacr/qt/preferences.py']),
}

def digest(path):
 return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def identity(value):
 return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def validate_cards(folder, lesson, repo):
 """Validate explicit card inputs, canonical text, source files and actual PNGs."""
 from PIL import Image
 folder, repo = Path(folder), Path(repo)
 contract=json.loads((folder/'cards.json').read_text())
 if (contract.get('schema') != 1 or contract.get('lesson') != lesson['id']
     or contract.get('english_sha256') != identity(lesson)
     or contract.get('definitions_sha256') != identity(DEFINITIONS)
     or contract.get('generator_sha256') != digest(__file__)
     or contract.get('style_helper_sha256') != digest(repo/'tools/tutorials/authoring/tools/render_install_keyframes.py')):
  raise ValueError('Reference card source binding changed')
 rows=contract.get('cards',[])
 if len(rows)!=6 or {r.get('visual') for r in rows}!=set(DEFINITIONS):
  raise ValueError('Exactly six distinct authored Home cards required')
 for row in rows:
  visual=row['visual']; scene=next(s for s in lesson['scenes'] if s['visual']==visual)
  expected_refs=[{'path':p,'sha256':digest(repo/p)} for p in DEFINITIONS[visual][2]]
  if (row.get('kind')!='authored_reference_card' or row.get('narration_sha256')!=identity(scene['narration'])
      or row.get('references')!=expected_refs or row.get('image')!=visual+'.png'
      or row.get('sha256')!=digest(folder/row['image'])):
   raise ValueError('Home reference card bytes, source or narration changed')
  with Image.open(folder/row['image']) as image:
   if image.size!=(3840,2160) or image.format!='PNG':raise ValueError('Reference card must be native 4K PNG')
   image.verify()
 return {row['visual']:row for row in rows}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--repo',type=Path,required=True)
 p.add_argument('--lesson',type=Path,required=True)
 p.add_argument('--destination',type=Path,required=True)
 a=p.parse_args();lesson=json.loads(a.lesson.read_text())
 if lesson['id']!='05_home' or len(lesson['scenes'])!=39:raise ValueError('Use the current 39-scene Home lesson')
 if a.destination.exists():raise ValueError('Preserve existing cards; choose a fresh destination')
 sys.path.insert(0,str(a.repo/'tools/tutorials/authoring/tools'))
 import render_install_keyframes as r
 from PIL import ImageDraw
 a.destination.mkdir(parents=True)
 cards=[]
 for visual,(title,lines,refs) in DEFINITIONS.items():
  scene=next(s for s in lesson['scenes'] if s['visual']==visual)
  image=r.gradient_background();draw=ImageDraw.Draw(image)
  # Flat typography, without simulated application windows or terminal output.
  r.draw_text(draw,(150,100),'Home · Feature reference',r.SUBTITLE,r.CYAN)
  r.draw_text(draw,(150,190),title,r.TITLE,r.TEXT)
  for index,line in enumerate(lines):
   if r.BODY.getlength(line)>r.scaled(1580):raise ValueError('Card line exceeds safe text width')
   r.draw_text(draw,(170,380+index*130),line,r.BODY,r.TEXT)
  r.draw_text(draw,(150,1000),'spaCR tutorial series',r.SMALL,r.MUTED)
  r.draw_text(draw,(1770,1000),'Feature reference',r.SMALL,r.MUTED,anchor='ra')
  path=a.destination/(visual+'.png');image.save(path)
  cards.append({'visual':visual,'image':path.name,'sha256':digest(path),'kind':'authored_reference_card',
   'narration_sha256':identity(scene['narration']), 'references':[{'path':f,'sha256':digest(a.repo/f)} for f in refs]})
 contract={'schema':1,'lesson':lesson['id'],'english_sha256':identity(lesson),
  'definitions_sha256':identity(DEFINITIONS),'generator_sha256':digest(__file__),
  'style_helper_sha256':digest(a.repo/'tools/tutorials/authoring/tools/render_install_keyframes.py'),'cards':cards}
 (a.destination/'cards.json').write_text(json.dumps(contract,ensure_ascii=False,indent=2)+'\n')
 validate_cards(a.destination,lesson,a.repo)
 print('Six authored feature-reference cards; no raw GUI capture or execution claim',flush=True)

if __name__=='__main__':main()
