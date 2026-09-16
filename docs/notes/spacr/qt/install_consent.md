# Notes from `spacr/qt/install_consent.py`

Prose lifted out of `spacr/qt/install_consent.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [InstallerConsentDialog.__init__](#installerconsentdialog__init__) (2 entries)
- [maybe_show_installer_consent](#maybe_show_installer_consent) (1 entry)

## InstallerConsentDialog.__init__

### lines 67-73

```python
self.setMinimumWidth(scaled_px(640))
```

SIZED IN SCALED PIXELS, NOT RAW ONES. A dialog size set from

Python does not grow when the stylesheet's font size does, so at the 200%% font scale the prose inside this window wrapped to more height than the window had and the last line was cut off. The size-policy fix on the label was necessary and not sufficient: a policy stops a parent handing a label less than it asks for, but it cannot make a window grow that has no room to give.

### lines 86-91

```python
explanation.setSizePolicy(QSizePolicy.Preferred,
```

A WRAPPED LABEL NEEDS (Preferred, Minimum): with Qt's default Preferred height a parent is free to hand it less than its heightForWidth. This is the house rule `prerun._label` documents. NECESSARY BUT NOT SUFFICIENT HERE -- 350's sweep still reports this label clipped at 2.0x, because the container above it does not grow either. See 350; the remaining fix is the dialog's layout, not this.

## maybe_show_installer_consent

### lines 180-181

```python
sign_in = apply_choices(choices)
```

Mark before opening another modal. If that dialog or a vendor CLI fails, the privacy page must not reappear and rewrite the user's choices.
