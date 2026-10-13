from pathlib import Path
import sys

sys.meta_path = [finder for finder in sys.meta_path
                 if '__editable__' not in (getattr(finder, '__module__', '') or type(finder).__module__)]
sys.path.insert(0, str(Path.cwd()))
from spacr.qt.night_themes import DATA_ART_THEMES
import spacr.qt.night_themes as registry

assert Path(registry.__file__).resolve().is_relative_to(Path.cwd().resolve())

path = Path('docs/source/features.rst')
text = path.read_text()
anchor = 'Arranging the window\n--------------------\n'
assert text.count(anchor) == 1 and 'Data-art themes\n' not in text
section = '''Data-art themes
---------------

Under **Preferences** → **Appearance** → **Theme**, twelve data-art presets
each select an interface palette and an animated background. The original
ten night themes and seven background animations remain available.
**Animation** lets you choose a background independently; select **None**
for a static background. These backgrounds are decorative and do not
display project measurements.

'''
for theme in DATA_ART_THEMES.values():
    section += f'- **{theme.label}** — {theme.description}\n'
section += '\n'
path.write_text(text.replace(anchor, section + anchor))
print('Added actual twelve-theme registry descriptions and selection guidance', flush=True)
