import hashlib
import json
import os
from pathlib import Path
import sys
root=Path('/mnt/wd4tb/spacr-worktrees/codex-home-b-n180-final-help-20261015')
out=Path('/mnt/wd4tb/scratch/home-b-n655-plate-focus-20261015')
sys.path.insert(0,str(root))
from PySide6.QtCore import QSettings,Qt
from PySide6.QtGui import QAccessible
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from spacr.qt import i18n,preferences,theme
from spacr.qt.widgets.plate_map_picker import PlateMapPicker
app=QApplication([])
preferences._settings=lambda:QSettings(str(out/'settings.ini'),QSettings.IniFormat)
preferences._SAFE_MODE=False
records=[]
for scheme in ('dark','light','high_contrast'):
 preferences.set_theme_choice(scheme)
 preferences.set_strong_focus(False)
 theme._forget_painting_theme()
 theme.apply_qpalette(app)
 app.setStyleSheet(theme.stylesheet())
 picker=PlateMapPicker(layout=6)
 i18n.retranslate_widget_tree(picker,'en')
 picker.resize(500,300)
 picker.show();picker.activateWindow();app.processEvents()
 well=picker._wells[(1,1)]
 picker.plate_button.setFocus(Qt.TabFocusReason);app.processEvents()
 reached=False
 for n in range(24):
  QTest.keyClick(app.focusWidget(),Qt.Key_Tab);app.processEvents()
  if well.hasFocus():reached=True;break
 interface=QAccessible.queryAccessibleInterface(well)
 name=interface.text(QAccessible.Name)
 assert reached and name=='A01', (scheme,reached,name)
 assert not well.isChecked()
 QTest.keyClick(well,Qt.Key_Space)
 assert well.isChecked() and (1,1) in picker.selection()
 QTest.keyClick(well,Qt.Key_Space)
 assert not well.isChecked() and (1,1) not in picker.selection()
 picker.plate_button.setFocus(Qt.TabFocusReason);app.processEvents()
 plain=well.grab().toImage()
 well.setFocus(Qt.TabFocusReason);app.processEvents()
 ordinary=well.grab().toImage()
 preferences.set_strong_focus(True)
 app.setStyleSheet(theme.stylesheet());app.processEvents()
 stronger=well.grab().toImage()
 record={'theme':scheme,'tab_reached':reached,'tab_steps':n+1,'accessible_name':name,'space_selects_deselects':True,'ordinary_differs_unfocused':ordinary!=plain,'strong_differs_ordinary':stronger!=ordinary,'size':[well.width(),well.height()]}
 records.append(record)
 picker.close();picker.deleteLater();app.processEvents()
source='spacr/qt/widgets/plate_map_picker.py'
result={'source':source,'source_sha256':hashlib.sha256((root/source).read_bytes()).hexdigest(),'records':records}
(out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
assert all(r['ordinary_differs_unfocused'] for r in records),'Ordinary well keyboard focus is invisible'
assert all(r['strong_differs_ordinary'] for r in records),'Stronger well keyboard focus is invisible'
