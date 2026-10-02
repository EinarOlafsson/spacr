import os,sys,json,hashlib
from pathlib import Path
root=Path(__file__).parent;repo=Path('/tmp/spacr-implementation-20261001/suggest-capture');sys.path.insert(0,str(repo))
os.environ['XDG_CONFIG_HOME']=str(root/'config');os.environ['MPLCONFIGDIR']=str(root/'mpl')
import matplotlib
matplotlib.use('Agg')
import pandas as pd
from PySide6.QtWidgets import QApplication
from spacr.qt import preferences
assert Path(preferences._settings().fileName()).resolve().is_relative_to(root/'config')
app=QApplication([])
preferences.set_figure_format('png');preferences.set_figure_png_dpi(100)
from spacr.sp_stats import screen_wells,score_screen,write_hit_report
source=pd.read_csv(root/'screen.csv');files={}
for feature in ['cell_count_proxy','nuclear_area']:
    wells=screen_wells(source,feature,plate_column='plateID',control_column='well_type',negative_levels=['negcon'])
    result=score_screen(wells,rank_by='b_score')
    assert len(result.plates)==4
    outputs=write_hit_report(result,root/'plots'/feature,methods=['b_score'],target='print')
    files[feature]={k:{'path':str(v),'sha256':hashlib.sha256(Path(v).read_bytes()).hexdigest()} for k,v in outputs.items()}
(root/'plots.json').write_text(json.dumps(files,indent=2)+'\n')
print(json.dumps(files,indent=2))
