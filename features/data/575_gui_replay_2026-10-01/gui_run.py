from pathlib import Path
import hashlib,json,os,shutil,sqlite3,sys,time
import numpy as np

repo=Path('/tmp/spacr-implementation-20261001/suggest-capture');stage=Path(sys.argv[1]).resolve();sys.path.insert(0,str(repo))
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(name,value):(stage/name).write_text(json.dumps(value,indent=2,default=str)+'\n')
source=Path('/home/carruthers/.cache/spacr/example_data/plate1/merged/plate1_E01_10_1.npy')
source_hash=digest(source);image=np.load(source,mmap_mode='r');assert image.ndim==3 and image.shape[-1]==7
choices=[]
for y in range(0,image.shape[0]-511,256):
    for x in range(0,image.shape[1]-511,256):
        labels=np.unique(image[y:y+512,x:x+512,4]);choices.append((len(labels[labels>0]),y,x))
count,y,x=max(choices);assert count>0
crop=np.array(image[y:y+512,x:x+512]);merged=stage/'gui/plate1/merged';merged.mkdir(parents=True)
np.save(merged/source.name,crop);replay=stage/'replay/plate1/merged';replay.mkdir(parents=True);shutil.copyfile(merged/source.name,replay/source.name)
write('source.json',dict(source=str(source),source_sha256=source_hash,source_shape=image.shape,crop_yx=[y,x],crop_shape=crop.shape,original_mask_labels_preserved=True,positive_cell_labels=count,private_input_sha256=digest(merged/source.name)))
del image,crop
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout
from PySide6.QtCore import Qt,QEvent
from PySide6.QtTest import QTest
from spacr.qt import preferences
actual=Path(preferences._settings().fileName()).resolve();assert actual.is_relative_to(stage/'config'),actual
write('profile-guard.json',dict(preferences=str(actual),private=True))
preferences.set_theme('dark');preferences.set_ambient_enabled(False)
from spacr.qt.theme import apply_qpalette,stylesheet
from spacr.qt.screens.app_screen import AppScreen
from spacr.qt.screens.run_history import RunHistoryScreen
from spacr.run_journal import runs_root
assert runs_root()==Path('/home/carruthers/.spacr/runs')
assert (stage/'app-state/runs').is_dir(),'Namespace did not route actual run journal privately'
app=QApplication([]);app.setQuitOnLastWindowClosed(False);apply_qpalette(app,theme='dark');app.setStyleSheet(stylesheet(theme='dark'))
screen=AppScreen('measure');host=QWidget();layout=QVBoxLayout(host);layout.addWidget(screen);host.resize(1450,950)
settings=dict(src=str(merged),channels=[0,1,2,3],cell_mask_dim=4,nucleus_mask_dim=5,pathogen_mask_dim=6,organelle_mask_dim=None,
 cell_chann_dim=2,nucleus_chann_dim=0,pathogen_chann_dim=3,cell_min_size=0,nucleus_min_size=0,pathogen_min_size=0,cytoplasm_min_size=0,
 uninfected=True,cytoplasm=False,save_measurements=True,save_png=True,save_arrays=True,png_dims=[0,2,3],png_size=[[128,128]],crop_mode=['cell'],
 representative_images=False,plot=False,save=False,verbose=False,n_jobs=1,timelapse=False,test_mode=False,normalize=False,
 use_bounding_box=False,homogeneity=False,radial_dist=False,calculate_correlation=False,merge_edge_pathogen_cells=True,hash_inputs=True)
screen.apply_settings_dict(settings);collected=screen._settings_model.collect();write('gui-collected-settings.json',collected)
for key in ['src','channels','cell_mask_dim','nucleus_mask_dim','pathogen_mask_dim','normalize','save_measurements','save_png','n_jobs']:
    assert collected[key]==settings[key],(key,collected[key],settings[key])
assert not screen._crop_choice_warnings(collected)
host.show();QTest.qWait(150);assert screen._btn_run.isEnabled()
print('CLICK actual Measure Run',flush=True);QTest.mouseClick(screen._btn_run,Qt.LeftButton)
start=time.monotonic()
try:
    while getattr(screen,'_thread',None) is not None:
        app.processEvents();time.sleep(.03)
        if time.monotonic()-start>180:raise TimeoutError('Measure run exceeded180 seconds')
    app.processEvents();assert not screen._last_error_text,screen._last_error_text
    runs=list((stage/'app-state/runs').glob('*/manifest.json'));assert len(runs)==1,runs
    manifest=json.loads(runs[0].read_text());write('gui-manifest.json',manifest);assert manifest['status']=='success',manifest
    assert manifest['app_key']=='measure'
    assert (stage/'gui/plate1/measurements/measurements.db').is_file()
    with sqlite3.connect(stage/'gui/plate1/measurements/measurements.db') as connection:
        assert connection.execute('SELECT COUNT(*) FROM cell').fetchone()[0] > 0
        assert connection.execute('SELECT COUNT(*) FROM nucleus').fetchone()[0] > 0
    host.grab().save(str(stage/'measure-complete.png'))
    history=RunHistoryScreen(threaded=False);history.refresh();assert history.select_run(runs[0].parent)
    workflow=history._export_selected_workflow('snakemake',str(stage/'export'));assert workflow is not None
    history.close();history.deleteLater()
    write('gui-result.json',dict(success=True,seconds=time.monotonic()-start,journal=str(runs[0].parent),workflow=str(workflow),original_source_unchanged=digest(source)==source_hash))
    write('gui-journal-hashes.json',{str(p.relative_to(runs[0].parent)):digest(p) for p in runs[0].parent.rglob('*') if p.is_file()})
    print('GUI PASS',workflow,flush=True)
finally:
    screen.close();host.close();screen.deleteLater();host.deleteLater();app.sendPostedEvents(None,QEvent.DeferredDelete);app.processEvents()
