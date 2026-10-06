Keyboard shortcuts and settings templates
=========================================

Keyboard shortcuts
------------------

Press **F1** or **?** to open the shortcut cheat sheet. It lists the keys
that work anywhere in the window, such as **Ctrl+K** for the command palette,
**Ctrl+P** for Preferences, **Ctrl+F** to search the current module's
settings and **F11** for full screen, followed by the keys of the screen you
are on.

To change a key, choose **Change shortcuts…** in the cheat sheet. The
**Change shortcuts** window has one row per window-wide action, with its
**Default** key and the **Shortcut** in use. Click a shortcut and press the
new key, or clear it to leave the action without a key. **Restore defaults**
puts every row back to its default key.

If a key is already used by another action, or by a key of one of the
screens, the line under the table names the actions that share it, and
**Save** stays unavailable until the conflict is resolved. Saved keys apply
straight away, also to the matching menu items, and the cheat sheet shows
them. They are kept with your preferences and read the same on Linux, macOS
and Windows.

Keys that belong to one screen, such as Annotate's labelling keys or the
Make Masks editing keys, are listed in the cheat sheet but cannot be changed.

Settings templates
------------------

A settings template is one module's settings saved under a name you choose,
for example ``Toxo PVM, 40x``. Open a module with a settings panel and press
**Ctrl+Shift+R** to open **Settings templates** for that module.

* **Save current settings…** stores the form as a new template, selects it
  in the list and shows its full saved path as text you can select and copy.
* **Apply** fills the form from the selected template. A template saved with
  another spaCR version lists the settings that no longer exist and the ones
  added since. A template for a different module is refused.
* **Share…** writes the template to a JSON file you can send to someone else.
* **Import…** reads a shared template file, or a settings CSV such as the one
  every run writes beside its results. A CSV becomes a template named after
  the file. A file for another module, or one without settings, is refused.
* **Rename…** gives the selected template a new name. A name that another
  template of the module already uses is refused.
* **Delete** removes the selected template.

Templates are stored in ``~/.spacr/recipes/<module>/``, one JSON file per
template. Set the ``SPACR_RECIPE_DIR`` environment variable to keep them in
another folder, for example a shared lab folder.
