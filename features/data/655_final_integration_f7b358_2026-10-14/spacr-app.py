"""Show real spaCR windows against private settings for external AT-SPI."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(os.environ['N655_SOURCE_ROOT'])
OUT=Path(os.environ['N655_SCRATCH'])
sys.path.insert(0,str(ROOT))
os.environ['SPACR_HOME']=str(OUT/'spacr-home')
os.environ['SPACR_LOG_DIR']=str(OUT/'spacr-logs')
os.environ['SPACR_NETWORK_CONFIG']=str(OUT/'spacr-network.json')
from PySide6.QtCore import QCoreApplication,QSettings,QTimer,Qt
from PySide6.QtGui import QAccessible
from PySide6.QtWidgets import QApplication,QTabWidget,QWidget
QSettings.setPath(QSettings.IniFormat,QSettings.UserScope,str(OUT/'spacr-settings'))
QSettings.setPath(QSettings.NativeFormat,QSettings.UserScope,str(OUT/'spacr-settings'))
from spacr.qt import preferences,theme,i18n
preferences.set_ambient_animation('none')
preferences.set_preload_policy('on_demand')
preferences._set_restore_session(False)
preferences.set_theme_choice('dark')
preferences._settings().setValue('onboarding/first_run_tour_seen',True)
from spacr.qt.app import MainWindow
from spacr.qt.widgets.section import Section
app=QApplication([])
app.setApplicationName('spaCR N655 external acceptance')
theme.apply_qpalette(app)
app.setStyleSheet(theme.stylesheet())
window=MainWindow(initial_app='__home__');window.resize(1200,800);window.show()
dialog=preferences.PreferencesDialog(window);dialog.resize(1100,950);dialog.show()
tabs=dialog.findChild(QTabWidget,'PreferencesTabs')
appearance=dialog.findChild(QWidget,'PreferencesTabAppearance')
for index in range(tabs.count()):
    if tabs.tabText(index)=='Appearance':tabs.setCurrentIndex(index);break
for section in dialog.findChildren(Section):
    if section.title()=='ACCESSIBILITY':section.set_expanded(True)
i18n.retranslate_widget_tree(window,'en')
i18n.retranslate_widget_tree(dialog,'en')
control=dialog.findChild(QWidget,'AccessibilityHighContrast')
control.toggled.connect(lambda checked:print('HIGH_CONTRAST_TOGGLED '+str(checked),flush=True))
QAccessible.setActive(True)
def ready():
    assert control.isVisible(), 'Accessibility switch must be shown for external acceptance'
    hashes={}
    for name,module in list(sys.modules.items()):
        path=getattr(module,'__file__',None)
        if name.startswith('spacr') and path:
            path=Path(path)
            if path.is_relative_to(ROOT) and path.is_file():hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT/'spacr-process-source.json').write_text(json.dumps(hashes,indent=2)+'\n')
    print('READY '+json.dumps({'pid':os.getpid(),'application_children':QAccessible.queryAccessibleInterface(app).childCount(),'high_contrast_visible':control.isVisible(),'source_files':len(hashes)}),flush=True)
QTimer.singleShot(1500,ready)
QTimer.singleShot(22000,app.quit)
app.exec()
dialog.reject();window.close()
