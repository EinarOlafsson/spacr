"""Fail closed on a field/channel tour that changed analysis or crop identity."""


def check_controls(proof):
    for key in ('source_unchanged', 'batch_settings_restored'):
        if proof.get(key) is not True:
            raise ValueError('The tutorial must preserve input and batch settings')
    for key in ('batch_started', 'crop_dialog_opened', 'application_layout_fixed'):
        if proof.get(key) is not False:
            raise ValueError('Only the usable preview controls are in this scope')
    original, single, restored, second = [proof[k] for k in
                                         ('original', 'single_channel', 'restored', 'second_field')]
    if original != restored or not original['objects'] or not second['objects']:
        raise ValueError('Restoration must recover a nonempty actual crop grid')
    if original['source'] != single['source'] or original['source'] == second['source']:
        raise ValueError('Expected one channel change and one different source field')
    if original['params']['display_channel'] is not None or single['params']['display_channel'] != 0:
        raise ValueError('The actual channel selector must reach zero and All channels')
    identity = lambda row: [(c['label'], c['area'], c['category'], c['shape']) for c in row['objects']]
    if identity(original) != identity(single):
        raise ValueError('A display channel must not change crop identity')
    if not all(c['equal_rgb'] and c['max'] > 0 for c in single['objects']):
        raise ValueError('Channel zero must show nonempty grayscale crops')
    if [c['sha256'] for c in original['objects']] == [c['sha256'] for c in single['objects']]:
        raise ValueError('The channel change must actually change rendered pixels')
