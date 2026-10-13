import hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
out=Path('/tmp/spacr-implementation-20261001/preferences-native-recheck')
real=Path.home()/'.config/spacr/qt.conf'
before=hashlib.sha256(real.read_bytes()).hexdigest() if real.exists() else None
os.environ['XDG_CONFIG_HOME']=tempfile.mkdtemp(prefix='spacr-native-recheck-')
os.environ.pop('QT_QPA_PLATFORM',None)
from PySide6.QtCore import Qt,QPoint,QTimer
from PySide6.QtGui import QPainter,QColor
from PySide6.QtWidgets import QApplication,QWidget
from spacr.qt import preferences,theme
from spacr.qt.widgets.glass import install_glass_everywhere
mode=sys.argv[1]
app=QApplication([])
preferences.set_theme_choice(mode)
theme.apply_qpalette(app,mode)
app.setStyleSheet(theme.stylesheet(mode))
def active():
 return subprocess.check_output(['xprop','-root','_NET_ACTIVE_WINDOW'],text=True).strip()
active_before=active()
class Ground(QWidget):
 def paintEvent(self,event):
  p=QPainter(self)
  for x in range(0,self.width(),40):
   for y in range(0,self.height(),40):p.fillRect(x,y,40,40,QColor('#ee66bb' if (x//40+y//40)%2 else '#44ccaa'))
  p.end()
flags=Qt.FramelessWindowHint|Qt.WindowStaysOnTopHint|Qt.WindowDoesNotAcceptFocus
host=Ground();host.setWindowFlags(flags);host.setAttribute(Qt.WA_ShowWithoutActivating)
host.resize(1350,1100)
avail=app.primaryScreen().availableGeometry();host.move(max(avail.x(),avail.right()-1370),avail.y()+30)
host.setWindowTitle('spaCR controlled corner check');host.show()
install_glass_everywhere(app)
d=preferences.PreferencesDialog(host)
d.setModal(False);d.setWindowFlag(Qt.WindowDoesNotAcceptFocus,True);d.setWindowFlag(Qt.WindowStaysOnTopHint,True)
d.setAttribute(Qt.WA_ShowWithoutActivating);d.move(host.x()+30,host.y()+30);d.show()
rows=[];errors=[]
def capture(stage=0):
 try:
  wh=d.windowHandle();screen=wh.screen()
  im=screen.grabWindow(0,d.frameGeometry().x(),d.frameGeometry().y(),d.width(),d.height()).toImage()
  corners=[]
  for x,y in ((0,0),(d.width()-1,0),(0,d.height()-1),(d.width()-1,d.height()-1)):
   pt=host.mapFromGlobal(d.mapToGlobal(QPoint(x,y)))
   expected='#ee66bb' if (pt.x()//40+pt.y()//40)%2 else '#44ccaa'
   corners.append([im.pixelColor(x,y).name(),expected])
  record={'theme':mode,'stage':stage,'alpha':wh.format().alphaBufferSize(),'mask_empty':d.mask().isEmpty(),'window_id':int(d.winId()),'size':[d.width(),d.height()],'corners':corners,'active_window':active()}
  rows.append(record)
  assert record['alpha']>=8,record
  assert all(a==b for a,b in corners),record
  assert im.pixelColor(im.width()//2,im.height()//2).name() not in ('#ee66bb','#44ccaa')
  im.save(str(out/f'preferences-{mode}-{stage}.png'))
  if stage==0:
   d.resize(d.width()+41,d.height()+31);QTimer.singleShot(300,lambda:capture(1));return
  if stage==1:
   d.hide();d.show();d.raise_();QTimer.singleShot(300,lambda:capture(2));return
 except Exception as exc:errors.append(str(exc))
 d.close();host.close();app.quit()
QTimer.singleShot(600,capture)
QTimer.singleShot(6000,app.quit)
app.exec()
after=hashlib.sha256(real.read_bytes()).hexdigest() if real.exists() else None
result={'rows':rows,'errors':errors,'real_settings_before':before,'real_settings_after':after,'active_before':active_before,'active_after':active(),'native_platform':app.platformName(),'config':os.environ['XDG_CONFIG_HOME']}
(out/f'{mode}.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert before==after
assert not errors and len(rows)==3
assert len({r['window_id'] for r in rows})==1
assert result['active_before']==result['active_after']
