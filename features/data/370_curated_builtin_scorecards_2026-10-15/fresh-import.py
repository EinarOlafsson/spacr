import sys
from pathlib import Path
sys.path.insert(0,'/mnt/wd4tb/spacr-worktrees/codex-n703-cache-cleanup-gui-20261015')
class Forbidden:
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in {'torch','cellpose','PySide6','huggingface_hub','requests','transformers'}:
   raise AssertionError('Forbidden provider/model/Qt import:'+name)
sys.meta_path.insert(0,Forbidden())
import urllib.request
urllib.request.urlopen=lambda *a,**k:(_ for _ in ()).throw(AssertionError('No network authorized'))
from spacr import model_zoo as zoo
raw=next(row for row in zoo.BUNDLED_REMOTE_MODELS if row['key']=='toxoplasma_pv_v4')
entry=zoo._builtin_entry_from_mapping(raw)
assert entry._curated_card and entry.metrics['n_fields']==209
assert entry.metrics['boundary_f1_2px']['finetuned']==0.8053199803663135
assert 'boundary_f1_1px' not in entry.metrics
assert len(entry.scorecard_lines())==7
assert 'Boundary F1 (2 px)' in entry.describe() and '209' in entry.describe()
assert 'historical_scorecards' not in entry.describe()
assert 'Boundary F1 (2 px)' in zoo.scorecard_html(entry)
assert not zoo._entry_from_mapping(dict(raw))._curated_card
assert 'torch' not in sys.modules and 'PySide6' not in sys.modules
print('Actual builtin/describe/tooltip six-row2px/209field identity PASS; no model/Qt/provider/network import')
