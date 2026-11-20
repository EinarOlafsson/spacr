# Item 1001: a Preferences choice for the settings view a form opens on

- **Number:** 1001, assigned by the master session on 2026-11-20.
- **Requested by the maintainer (2026-11-20).**
- **What:** Preferences gets a "Settings forms open on" choice, **Essentials** or **All**. It is the view (`spacr/qt/settings_search.py`, the Essentials/All switch above every settings form) a module opens on when it has no remembered choice of its own.
- **Default:** Essentials, today's behaviour, so nothing changes unless the user changes it.
- **Changing it** forgets every module's remembered Essentials/All choice, so the new default reaches every module at once; switching a form afterwards is remembered for that module as before. Easy mode's Essentials-only visits are unchanged.
- **Storage:** a Preferences (QSettings) key, `settings/disclosure_default`. It is not a module setting, so no settings-file default changes and no `SETTINGS_SCHEMA_VERSION` row is needed.
- **Acceptance:** offscreen tests: the preference defaults to Essentials; with All chosen, a module without a remembered choice opens on All; a remembered choice still wins until the preference is changed again; the Preferences row reads and writes it.
