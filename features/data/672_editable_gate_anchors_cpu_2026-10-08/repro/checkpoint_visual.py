import importlib.util, json, os
from pathlib import Path
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QEvent
from shiboken6 import isValid
from spacr.qt.widgets.gate_spec import CylinderGate
from dataclasses import replace
root=Path.cwd(); target=Path('/mnt/wd4tb/scratch/gate-anchors-20261008')
spec=importlib.util.spec_from_file_location('gate_checks', root/'tests/qt/test_gate_editable_anchors_and_surfaces.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
app=QApplication.instance() or QApplication([])
class Owner:
    def addWidget(self, widget): self.widget=widget
owner=Owner(); screen=m.screen.__wrapped__(owner)
canvas=m._install(screen,m._box()); screen.apply_settings(replace(screen._settings,anchor_density=3))
canvas._view_angles=(31.,-47.,0.); canvas.render_now()
for _ in range(4): app.processEvents()
records=[]
for name in ['box-anchors']:
    path=target/(name+'.png'); screen.grab().save(str(path)); records.append({'path':path.name,'width':screen.width(),'height':screen.height()})
canvas._selected_surfaces={1,5}; canvas._surface_gate='box'; canvas._edit_axis='z'
canvas._volume_scroll(m._event(canvas,[300,300],step=1,button='up'))
for _ in range(4): app.processEvents()
path=target/'edited-adjacent-surfaces.png';screen.grab().save(str(path));records.append({'path':path.name,'gate':screen.gates.gates.get('box').to_dict()})
gate=CylinderGate(name='cylinder',u_column='a',v_column='b',axis_column='c',u_radius=1,v_radius=1,axis_low=-1,axis_high=1)
canvas=m._install(screen,gate)
for _ in range(4): app.processEvents()
path=target/'cylinder-depth-anchors.png';screen.grab().save(str(path));records.append({'path':path.name,'analytic_gate_preserved':screen.gates.gates.get('cylinder') is gate})
(target/'visual-receipt.json').write_text(json.dumps({'source_checkpoint':'1ad258e625f','scope':'Actual 1100x800 offscreen GUI captures; no animation/FPS/GPU acceptance.','images':records},indent=2)+'\n')
screen.close(); screen.deleteLater();app.sendPostedEvents(None,QEvent.Type.DeferredDelete);app.processEvents()
print(json.dumps({'screen_native_destroyed':not isValid(screen),'top_level_widgets':len(app.topLevelWidgets()),'images':len(records)}))
