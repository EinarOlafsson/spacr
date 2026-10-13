import os,sys,json,hashlib
from pathlib import Path
os.environ['XDG_CONFIG_HOME']='/tmp/spacr-implementation-20261001/f572-gui-warning/private-config'
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from matplotlib.figure import Figure
from PySide6.QtWidgets import QApplication,QPlainTextEdit
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt,QEvent
from spacr.qt.theme import apply_qpalette,stylesheet
from spacr.qt.widgets import save_figure_dialog as saving
os.environ['SPACR_FIGURE_INTEGRITY']='1'
app=QApplication([])
out=Path('/tmp/spacr-implementation-20261001/f572-gui-warning')
results=[]
for theme in ('dark','light'):
 apply_qpalette(app,theme=theme);app.setStyleSheet(stylesheet(theme=theme))
 folder=out/theme;folder.mkdir(exist_ok=True)
 fig=Figure(figsize=(4,2),dpi=80)
 for i in range(2):
  data=np.random.default_rng(i).integers(100,900,(64,64),dtype=np.uint16)
  fig.add_subplot(1,2,i+1).imshow(data,cmap='gray',vmin=0,vmax=1000 if i==0 else 600)
 target=folder/'comparison.png'
 dlg=saving.SaveFigureDialog(fig);dlg.dpi.setValue(80);dlg.show();app.processEvents()
 saving.QFileDialog.getSaveFileName=lambda *a,**k:(str(target),'')
 QTest.mouseClick(dlg._save,Qt.MouseButton.LeftButton);app.processEvents()
 notice,=saving._integrity_notices;QTest.qWait(150)
 assert notice.isVisible() and not notice.isModal() and target.is_file()
 body=notice.findChild(QPlainTextEdit);assert 'use one range' in body.toPlainText()
 path=folder/'notice.png';assert notice.grab().save(str(path))
 sidecar=Path(str(target)+'.provenance.json');report=json.loads(sidecar.read_text())
 assert report['figure_sha256']==hashlib.sha256(target.read_bytes()).hexdigest()
 results.append({'theme':theme,'screenshot':str(path),'screenshot_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'figure_sha256':report['figure_sha256'],'warning_count':report['integrity']['warnings'],'modeless':True})
 notice.close();dlg.close();app.sendPostedEvents(None,QEvent.Type.DeferredDelete)
(out/'capture-receipt.json').write_text(json.dumps(results,indent=2)+'\n')
print('PASS actual Save button, PNG/sidecar, modeless notice, dark/light capture')
