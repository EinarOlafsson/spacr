"""Explicit identities for the requested installation tutorial consolidation."""
from copy import deepcopy

INSTALL_ID = '02_install_spacr'
INSTALL_ARCHIVES = ('02_conda_install', '03_pip_install')
LESSON_ALIASES = dict.fromkeys(INSTALL_ARCHIVES, INSTALL_ID)


def installation_placeholder(catalog, original):
    """Keep lesson2's position while replacing its explicitly retired identity."""
    result = deepcopy(catalog)
    originals = {item['id']: item for item in original['lessons']}
    if INSTALL_ID in {item['id'] for item in result['lessons']}:
        return result
    if not all(key in originals for key in INSTALL_ARCHIVES):
        raise ValueError('Installation merge requires both archival originals')
    old = originals[INSTALL_ARCHIVES[0]]
    if old['number'] != 2 or old.get('app_key') is not None:
        raise ValueError('Installation merge cannot renumber or change a module route')
    replacement = {**deepcopy(old), 'id': INSTALL_ID, 'slug': 'install_spacr'}
    result['lessons'] = [item for item in result['lessons'] if item['id'] not in INSTALL_ARCHIVES]
    index = next((i for i, item in enumerate(result['lessons']) if item['number'] > 2), len(result['lessons']))
    result['lessons'].insert(index, replacement)
    result['lesson_aliases'] = {**result.get('lesson_aliases', {}), **LESSON_ALIASES}
    return result
