import hashlib,json,sys
from pathlib import Path
from PySide6.QtCore import QRectF,QSettings
from PySide6.QtGui import QImage,QPainter
from PySide6.QtWidgets import QApplication
from spacr.qt import preferences,theme
from spacr.qt.widgets import workflow_diagram as wd
root=Path('/mnt/wd4tb/scratch/home-b-n350-workflow-title-20261015')
app=QApplication([])
store=QSettings(str(root/'settings.ini'),QSettings.IniFormat)
preferences._settings=lambda:store
app.setStyleSheet(theme.stylesheet('dark'))
class Painter(QPainter):
    def __init__(self,image):
        super().__init__(image)
        self.calls=[]
    def drawText(self,*args):
        result=super().drawText(*args)
        if len(args)==3 and isinstance(args[0],QRectF):
            rect,flags,text=args
            measured=self.boundingRect(rect,flags,text)
            self.calls.append({'text':text,'box':[rect.x(),rect.y(),rect.width(),rect.height()],'measured':[measured.x(),measured.y(),measured.width(),measured.height()],'font_px':self.font().pixelSize(),'fits':rect.adjusted(-.5,-.5,.5,.5).contains(measured)})
        return result
rows=[]
data=wd.workflow_map()
keys=['cellpose_masks','simulation','ml_analyze','host_pathogen']
for language in ['en','de','is','pt']:
    preferences.set_language(language)
    view=wd.WorkflowView(data,keys=keys)
    for key,node in view.nodes.items():
        image=QImage(280,200,QImage.Format_ARGB32);image.fill(0)
        painter=Painter(image)
        try: node.paint(painter,None)
        finally: painter.end()
        rows.append({'language':language,'key':key,**painter.calls[0]})
    view.close()
failed=[r for r in rows if not r['fits']]
result={'source_sha256':hashlib.sha256(Path(wd.__file__).read_bytes()).hexdigest(),'scope':'Four current long title candidates in real noncompact WorkflowView nodes; production paint drawText geometry, no map/data changes','rows':rows,'failed':failed}
(root/'result.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'cases':len(rows),'failed':failed},indent=2,ensure_ascii=False))
sys.exit(bool(failed))
