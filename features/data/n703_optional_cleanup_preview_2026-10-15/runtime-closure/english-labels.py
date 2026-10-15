from pathlib import Path
import tempfile
from PySide6.QtWidgets import QApplication,QTreeWidget,QPushButton
from spacr import install_cleanup as backend
from spacr.qt import i18n,install_consent
app=QApplication.instance() or QApplication([])
i18n.current_language=lambda:'en'
with tempfile.TemporaryDirectory(dir='/mnt/wd4tb/scratch/n703-coalesced-runtime-20261015') as folder:
 home=Path(folder)
 news=home/'.spacr'/'news';news.mkdir(parents=True)
 (news/'releases.json').write_bytes(b'keep')
 preview=backend.preview_optional_cleanup('english-labels',system=backend._Machine(environ={'HOME':str(home)},platform='linux'))
 dialog=install_consent._InstallerCleanupDialog(preview)
 tree=dialog.findChild(QTreeWidget,'InstallerCleanupFolders')
 for index,label in [(1,'Background image cache'),(6,'Hugging Face cache'),(7,'Torch cache')]:
  assert tree.topLevelItem(index).text(0)==label,(index,tree.topLevelItem(index).text(0))
 assert not any(row['selectable'] for row in preview['folders'])
 keep=dialog.findChild(QPushButton,'InstallerCleanupKeep')
 assert keep.text()=='Keep all files' and keep.isDefault()
 assert dialog.choices()['keys']==[]
 assert (news/'releases.json').read_bytes()==b'keep'
 dialog.close();dialog.deleteLater();app.processEvents()
print('English actual native dialog:3labels/defaultKeep/no selection/protected preview/private bytes PASS')
