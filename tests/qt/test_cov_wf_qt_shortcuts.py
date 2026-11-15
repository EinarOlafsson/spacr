"""Tooltip prose spells the keys it names for the keyboard in use."""
from __future__ import annotations


def test_mac_tooltips_spell_native_keys_and_preserve_unparseable_gestures(monkeypatch):
    from spacr.qt import shortcuts

    monkeypatch.setattr(shortcuts.sys, 'platform', 'darwin')
    translations = {'Ctrl+A': '⌘A', 'Ctrl+Shift+N': '⇧⌘N'}
    monkeypatch.setattr(shortcuts, 'native', lambda keys: translations.get(keys, keys))
    assert shortcuts._native_prose('Add (Ctrl+Shift+N); drag Ctrl+L+right-click.') == (
        'Add (⇧⌘N); drag ⌘L+right-click.')
    assert shortcuts._native_prose('No shortcut here.') == 'No shortcut here.'
