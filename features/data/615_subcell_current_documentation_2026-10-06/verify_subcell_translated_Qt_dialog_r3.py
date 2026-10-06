from pathlib import Path
import hashlib
import json
import os
import numpy as np
from PySide6.QtWidgets import QApplication, QComboBox, QDialogButtonBox, QLabel
from spacr.embeddings import EmbeddingError
from spacr.qt import preferences
from spacr.qt.screens.embeddings import EmbeddingsScreen

scratch = Path('/media/carruthers/mnt3/codex/scratch/docs-completion-20261005')
out = scratch / 'subcell-translated-Qt-dialog-r3'
out.mkdir(exist_ok=False)
reviews = json.loads((scratch / 'subcell-rybg-reviewed-runtime-r4.json').read_text())
application = QApplication.instance() or QApplication([])
preferences._get_show_alpha_features = lambda: True
captured = []
names = ('EmbeddingsSubCellMicrotubulesChannel', 'EmbeddingsSubCellErChannel', 'EmbeddingsSubCellDnaChannel', 'EmbeddingsSubCellProteinChannel')
for language, rows in reviews.items():
    os.environ['SPACR_LANGUAGE'] = language
    targets = {row['source']: row['translation'] for row in rows}
    covered = set()
    def verify(actual, source, **values):
        assert actual == targets[source].format(**values), (language, source, actual)
        covered.add(source)
    screen = EmbeddingsScreen(threaded=False)
    screen.resize(1360, 900)
    screen.show()
    screen.set_crops(np.zeros((1, 16, 16, 5), dtype=np.float32))
    screen._foundation.setCurrentIndex(screen._foundation.findData('subcell_rybg'))
    verify(screen._foundation.currentText(), 'SubCell (CZI / Lundberg lab ViT-B/16, R/Y/B/G)')
    tooltip = next(source for source in targets if source.startswith('A model trained on microscopy'))
    verify(screen._foundation._spacr_setting_label.toolTip(), tooltip)
    verify(screen._subcell_button.text(), 'Channels…')
    verify(screen._subcell_button.toolTip(), "Map four crop channels to microtubules, ER, DNA and protein before running SubCell's four-plane model.")
    verify(screen._policy.currentText(), 'Four mapped planes (one pass)')
    try:
        screen.spec()
        raise AssertionError('Unmapped source was accepted')
    except EmbeddingError as error:
        verify(str(error), "Choose four distinct crop channels before running SubCell's four-plane model.")
    dialog = screen._subcell_channels_dialog()
    dialog.resize(760, 400)
    dialog.show()
    application.processEvents()
    verify(dialog.windowTitle(), 'SubCell four-plane channels')
    labels = {label.text() for label in dialog.findChildren(QLabel)}
    for source in ('Microtubules (R)', 'ER (Y)', 'DNA (B)', 'Protein (G)', "Assign four different crop channels in SubCell's official order. No stain identity is guessed from channel position."):
        assert targets[source] in labels, (language, source, labels)
        covered.add(source)
    selectors = [dialog.findChild(QComboBox, name) for name in names]
    for selector in selectors:
        verify(selector.itemText(0), 'Choose channel…')
        assert selector.currentData() is None
        for index in range(5):
            verify(selector.itemText(index + 1), 'Channel {position} (index {index})', position=index + 1, index=index)
    actions = dialog.findChild(QDialogButtonBox)
    problem = dialog.findChild(QLabel, 'EmbeddingsSubCellChannelsProblem')
    actions.button(QDialogButtonBox.Ok).click()
    verify(problem.text(), 'Choose a crop channel for each SubCell plane: microtubules, ER, DNA and protein.')
    for selector in selectors:
        selector.setCurrentIndex(selector.findData(0))
    actions.button(QDialogButtonBox.Ok).click()
    verify(problem.text(), 'Choose four different crop channels for SubCell.')
    source = 'A selected SubCell channel is outside the loaded crops. Open Channels… and map the four planes again.'
    verify(screen._subcell_mapping_error((0, 1, 2, 5), 5), source)
    if language in ('sv', 'zh_CN', 'ko'):
        assert dialog.grab().save(str(out / ('dialog-' + language + '.png')))
    for selector, index in zip(selectors, (4, 0, 2, 1)):
        selector.setCurrentIndex(selector.findData(index))
    actions.button(QDialogButtonBox.Ok).click()
    assert screen.spec().channels == (4, 0, 2, 1)
    verify(screen._status.text(), 'SubCell channel mapping saved in microtubules, ER, DNA, protein order.')
    screen.set_crops(np.zeros((1, 16, 16, 4), dtype=np.float32))
    verify(screen._status.text(), "The new crops have fewer channels. Reopen Channels… to map SubCell's four planes.")
    screen.set_crops(np.zeros((1, 16, 16, 3), dtype=np.float32))
    verify(screen._subcell_button.toolTip(), "Load crops with at least four channels before mapping SubCell's microtubules, ER, DNA and protein planes.")
    screen.embed()
    verify(screen._status.text(), "SubCell's four-plane model needs crops with at least four channels. Load a suitable crop source first.")
    assert covered == set(targets), (language, set(targets) - covered)
    captured.append({'language': language, 'all_21_reviewed_captions_exercised_in_actual_Qt': True, 'no_guessed_mapping': True, 'raw_selected_mapping': [4, 0, 2, 1]})
    dialog.close()
    dialog.deleteLater()
    screen.close()
    screen.deleteLater()
    application.processEvents()
    print('PASS actual translated Qt dialog and mapping refusals', language, flush=True)
(out / 'acceptance.json').write_text(json.dumps({'passed': True, 'languages': captured, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, indent=2) + '\n')
