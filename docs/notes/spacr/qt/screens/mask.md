# Notes from `spacr/qt/screens/mask.py`

Prose lifted out of `spacr/qt/screens/mask.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [_OfferedPreview.set_offered](#_offeredpreviewset_offered) (1 entry)
- [install_example_data_button](#install_example_data_button) (1 entry)
- [_OpsPage.set_shown](#_opspageset_shown) (1 entry)
- [_OpsPage._page_closed](#_opspage_page_closed) (1 entry)
- [install_ops_switch](#install_ops_switch) (1 entry)
- [install_folds](#install_folds) (4 entries)

## _OfferedPreview.set_offered

### line 170  _(unsure)_

```python
LOG.debug("the folded preview is gone", exc_info=True)
```

Qt deleted the card under us; nothing left to offer.

## install_example_data_button

### lines 366-368

```python
button = QPushButton(EXAMPLE_BUTTON_TEXT)
```

The English source, the way every other caption in the tool is written: the language pass walks the widget tree and renders it from the catalog, and a caption translated here would be rendered twice.

## _OpsPage.set_shown

### lines 461-462

```python
page.hide()
```

Shown as a window instead, because this host had no page strip to put it on. Hidden, not closed: same screen back.

## _OpsPage._page_closed

### line 497, trailing  _(unsure)_

```python
except RuntimeError:
```

Qt deleted the page under us

## install_ops_switch

### line 542

```python
screen._ops_page = page
```

The installation outlives this call only because the screen holds it.

## install_folds

### lines 565-567

```python
try:
```

BEFORE the folds, and outside their guard: the example plate is what a user with no data of their own presses first, and a fold that cannot be mounted must not take it away.

### line 595

```python
screen._category_folds = folds
```

The set outlives this call only because the screen holds it.

### lines 598-600

```python
screen._fold_previews = _offer_fold_previews(screen, folds, strip)
```

The panels the folded modules brought with them, offered by their own switches. After the strip is on the masthead: a preview that could not be built must not cost the switches.

### lines 602-604

```python
mark_fold_sources(screen)
```

And the folded modules' icons, on the headings of the settings they became. After the strip, for the same reason the previews are: a mark that cannot be drawn must not cost the switches.
