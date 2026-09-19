# Notes from `spacr/qt/widgets/setup_slides.py`

Prose lifted out of `spacr/qt/widgets/setup_slides.py` by `tools/extract_source_notes.py`.
The module itself carries no comments now, so this file is where its reasons live; the path mirrors the source path, which is how it is found.

Entries are grouped by the function or class they sat in and carry the line they came from. Line numbers are from the state of the module when the notes were taken, so they drift; the quoted code line is the durable anchor.

## Contents

- [Module level](#module-level) (3 entries)
- [graphics_card](#graphics_card) (3 entries)
- [_let_go_of](#_let_go_of) (3 entries)
- [_held_at_the_top](#_held_at_the_top) (2 entries)
- [SetupSlides.__init__](#setupslides__init__) (7 entries)
- [SetupSlides._build_pages](#setupslides_build_pages) (8 entries)
- [SetupSlides._go_frameless](#setupslides_go_frameless) (1 entry)
- [SetupSlides._github_row](#setupslides_github_row) (2 entries)
- [SetupSlides._refresh_github](#setupslides_refresh_github) (2 entries)
- [SetupSlides._sign_in_to_github](#setupslides_sign_in_to_github) (3 entries)
- [SetupSlides._read_github_output](#setupslides_read_github_output) (1 entry)
- [SetupSlides._closing_page](#setupslides_closing_page) (2 entries)
- [SetupSlides._terms_page](#setupslides_terms_page) (7 entries)
- [SetupSlides._draw_the_terms_gate](#setupslides_draw_the_terms_gate) (1 entry)
- [SetupSlides._refuse_to_leave_the_terms](#setupslides_refuse_to_leave_the_terms) (1 entry)
- [SetupSlides._row](#setupslides_row) (1 entry)
- [SetupSlides._editor](#setupslides_editor) (3 entries)
- [SetupSlides._animation_row](#setupslides_animation_row) (2 entries)
- [SetupSlides._apply_animation](#setupslides_apply_animation) (1 entry)
- [SetupSlides._provider_buttons](#setupslides_provider_buttons) (3 entries)
- [SetupSlides._choose_provider](#setupslides_choose_provider) (2 entries)
- [SetupSlides._say_hello](#setupslides_say_hello) (1 entry)
- [SetupSlides._show_the_greeting](#setupslides_show_the_greeting) (2 entries)
- [SetupSlides._fade_the_greeting_away](#setupslides_fade_the_greeting_away) (3 entries)
- [SetupSlides._place_the_gpu_note](#setupslides_place_the_gpu_note) (4 entries)
- [SetupSlides._say_what_the_gpu_is](#setupslides_say_what_the_gpu_is) (3 entries)
- [SetupSlides._what_this_machine_can_do](#setupslides_what_this_machine_can_do) (4 entries)
- [SetupSlides.retranslate](#setupslidesretranslate) (1 entry)
- [SetupSlides._apply_look](#setupslides_apply_look) (1 entry)
- [SetupSlides._show_slide](#setupslides_show_slide) (5 entries)
- [SetupSlides._fade_in](#setupslides_fade_in) (2 entries)
- [SetupSlides.next](#setupslidesnext) (1 entry)
- [SetupSlides._advance_after_the_greeting](#setupslides_advance_after_the_greeting) (1 entry)
- [SetupSlides._record_the_agreement](#setupslides_record_the_agreement) (1 entry)
- [SetupSlides._install_backdrop](#setupslides_install_backdrop) (1 entry)
- [SetupSlides.resizeEvent](#setupslidesresizeevent) (3 entries)
- [_catalogue_this_screen](#_catalogue_this_screen) (1 entry)
- [open_setup_if_needed](#open_setup_if_needed) (2 entries)

## Module level

### lines 50-65

```python
("How it runs",
```

FIVE LEVELS, NOT THREE. This text described the old three-value posture -- Extra Performance, Performance, Balanced -- and stayed behind when the screen was pointed at PERFORMANCE_LEVELS, so it explained a control the reader was not looking at.

SHORT ON PURPOSE, AND THIS IS THE CEILING. The first version of this ran to 667 characters and named all five levels with a clause each. Nobody reads a 667-character caption in any language, and every one of these strings is translated into nine -- so length here is a cost paid nine times over, in text no reviewer can check against the English at a glance. The five levels are listed in the control right beside this sentence; the caption says what the control DOES and what it does not affect, which is the part the list cannot say.

ORDERED AS THE CONTROL IS, least of the machine kept to most, so reading the sentence and reading the list agree.

### lines 81-84

```python
("Terms of use",
```

THE ONE SLIDE THAT IS NOT A PREFERENCE. Every other question here has a working default and can be answered by dismissing the screen; this one is the condition the licence names, so it is the one slide that has to be answered before the screen can finish.

### lines 90-93

```python
("Done", "Welcome to spaCR", ()),
```

THE LAST SLIDE SAYS TWO THINGS AND NO MORE. "Done" is the answer to the six questions; "Welcome to spaCR" is what the screen is for. The paragraph that used to sit here explained where the settings live, which is a thing to find out when you go looking, not on the way in.

## graphics_card

### lines 334-336

```python
found = inspect_torch(_torch_module)
```

PROBED FOR THIS torch, not read from the cached answer for the machine: the slide is exercised against a stand-in torch, and a cached global reports the developer's own card instead.

### lines 338-342

```python
if found.is_gpu:
```

ANY VENDOR, NOT ONLY NVIDIA. This used to ask

`torch.cuda.is_available()`, so an AMD card driven perfectly well through Metal reported as "No compatible GPU" -- the machine this was fixed on segments 139x faster on the card the slide was denying. See instruction 319.

### lines 346-347  _(unsure)_

```python
return False, found.name or found.label
```

Found and not usable is its own answer, and the label carries which accelerator it was.

## _let_go_of

### lines 409-412

```python
import warnings
```

PER SIGNAL, not `QObject.disconnect()`: the argument-less form is about connections FROM this object made through the QObject overload, and it left the `finished` lambda connected -- measured, not assumed.

### lines 418-424

```python
with warnings.catch_warnings():
```

PySide6 WARNS BEFORE IT RAISES. Disconnecting a signal that was never connected prints "libpyside: Failed to disconnect" through the warnings machinery and then raises RuntimeError, so catching the exception alone still left the user reading a warning about the ordinary case -- a process that never emitted. Suppressing it here and nowhere wider keeps every other libpyside warning visible.

### lines 431-432

```python
pass
```

RuntimeError is Qt's "nothing was connected", which is the ordinary case for a process that never emitted.

## _held_at_the_top

### lines 449-452

```python
holder.setAttribute(Qt.WA_TranslucentBackground, True)
```

THE WRAPPER PAINTS NOTHING. A bare QWidget with no rule of its own takes the blanket window fill, which over this card reads as a black box behind the caption -- the wrapper exists to position the label, not to put a surface under it.

### lines 458-461

```python
try:
```

AS TALL AS THE ROW IT NAMES. The label then centres its own text in that height -- which is a QLabel's default -- and the caption lands level with the middle of the marks rather than at the top of a tile eighty pixels tall.

## SetupSlides.__init__

### lines 495-505

```python
self._go_frameless()
```

NO TITLE BAR HERE EITHER. The card this screen builds has rounded corners, and a square window frame around it -- with a close and a minimise button on top -- is the box the settings dialogs had until they went frameless. This screen builds its own card, so the glass filter deliberately leaves it alone, and leaving it alone left it with its frame.

Same order as `glass.make_frameless`: the attribute BEFORE the flags, because the flags recreate the native window and a translucency asked for afterwards applies to one that no longer exists.

### lines 515-516

```python
self.card.setMouseTracking(True)
```

THE RIM FOLLOWS THE POINTER, so the card has to see it move even when no button is down -- which is not the default.

### lines 523-526

```python
column.addStretch(1)
```

CENTRED, NOT TOP-ALIGNED. One question in a card this size sat against the ceiling with a void under it, which reads as a page that failed to load the rest of itself. A slide is one thing, and one thing belongs in the middle.

### lines 537-550

```python
self._greeting = QLabel("", self.card)
```

THE GREETING HAS ITS OWN LINE. It used to be prepended to the explanation, so choosing a language rewrote the paragraph under the title and the one word that changed was buried in it. THE GREETING IS THE ANSWER TO THE LANGUAGE QUESTION, so it comes AFTER the question is answered rather than sitting under it while it is still being decided. It is hidden until the first Next. NOT IN THE COLUMN. The greeting used to be a row in the layout, so it took space on the language slide and had to be switched off the moment the next slide arrived -- and switched off is what it looked like: "the transition away from Hello is abrupt and bad".

It floats over the card instead, low and centred, in a band the question rows never reach on any slide. Nothing has to move out of its way, so it can take its time leaving.

### lines 557-560

```python
self._gpu_note = QLabel("", self.card)
```

WHAT THIS MACHINE CAN RUN, in the row the greeting moved up out of. It floats over the card the same way, for the same reason: the question rows never reach this band, so nothing has to move for it and it can stay while the greeting comes and goes.

### lines 566-569

```python
self._gpu_note.setVisible(False)
```

THE FIRST SLIDE ONLY. It answers "can this machine run spaCR", which is a question the reader has once, at the start; carried down the rest of the slides it would be a banner that stopped being read on slide two and took the space anyway.

### lines 592-601

```python
self._clear_the_containers()
```

NO BLACK BOXES INSIDE THE CARD (reported 2026-08-22). The card paints itself translucent over the drifting backdrop, but every plain QWidget between the two -- the page stack, each page, the provider strip -- is caught by the blanket `QWidget` rule and paints an opaque `bg`, which is a solid dark rectangle sitting on top of the animation the dialog just installed.

THE CONTAINERS ONLY. The combos and buttons stay opaque: they are the readable surface, and a control you can see through is a control you cannot read.

## SetupSlides._build_pages

### lines 663-665

```python
self._pages.addWidget(self._terms_page())
```

NOT A FORM AND NOT THE CLOSING WORD. It writes no preference; it records an acceptance, and it is the one page the sequence will not let go of unanswered.

### lines 669-671

```python
self._pages.addWidget(self._closing_page(title, blurb))
```

THE CLOSING SLIDE IS NOT A FORM, so it is not laid out like one. It says one word, in the middle, with the sentence that qualifies it underneath.

### lines 675-688

```python
form = QFormLayout(page)
```

ONE FORM PER PAGE, NOT ONE LAYOUT PER ROW.

Every row used to be its own QHBoxLayout with a stretch between the label and the control, which does two bad things at once. It pushes the pair to opposite edges -- measured at 771 px apart on the language slide, so the eye has to travel the width of the card to find out what it is answering -- and because each row is an independent layout, nothing lines up with the row above it: every label starts in a different place and so does every control.

A QFormLayout is two columns for the whole page. Labels align with labels, controls with controls, and the gap between them is a number rather than whatever is left over.

### lines 690-693

```python
form.setContentsMargins(0, 8, 8, 0)
```

A MARGIN ON THE RIGHT. The controls are right-aligned, so with none they finish exactly on the card's content edge and their drop-down arrow is drawn flush against it -- which reads as a clipped control rather than as a control that fits.

### lines 697-700

```python
form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
```

LABELS SIT AGAINST THEIR CONTROL, vertically centred on it. The AI provider row is a strip of logo marks and is taller than a combo box; top-aligned, its caption floated above the marks while every other caption sat beside its control.

### lines 703-705

```python
form.setFieldGrowthPolicy(QFormLayout.FieldsStayAtSizeHint)
```

The control column takes what it needs and no more, so a combo does not stretch to the card edge while a slider does not.

### lines 710-713

```python
continue
```

A QUESTION THAT REMOVED ITSELF LEAVES NO GAP. The provider question is absent when no CLI is installed, and an empty labelled row would read as a broken control rather than as a question that does not apply.

### lines 717-720

```python
animation = self._animation_row()
```

IMMEDIATELY UNDER THE THEME, which is where it was asked for and where it belongs: the backdrop is part of what spaCR looks like, and a reader deciding on the look decides on both in one place.

## SetupSlides._go_frameless

### lines 742-744

```python
from .glass import _DragByBackground, _paint_nothing_behind_the_card
```

The window's own body paints nothing: `WA_TranslucentBackground` stops Qt filling it from the palette, and this stops the application stylesheet's `QDialog` rule doing it anyway.

## SetupSlides._github_row

### lines 773-777

```python
from .provider_marks import ProviderMark
```

THE MARK, WHICH IS THE STATE. "if spacr detects the login there should also be a github logo that gains colour" -- signed out it is drawn in the muted ink, signed in in GitHub's own black. The mark is the same widget the AI providers use, so one rule covers every sign-in on this screen.

### lines 780-786

```python
self._gh_mark = ProviderMark("github", "GitHub", False, holder)
```

THE LOGO IS THE BUTTON, the way the three AI marks are. It used to be an indicator beside a "Sign in" push button, which the user asked to collapse into one thing on 2026-08-23: "i want the github button to also be a github logo just like the AI icons work". One control, so there is nothing for a second one to disagree with; what the click will DO is in the tooltip and spelled out in the status line beside it.

## SetupSlides._refresh_github

### lines 850-855

```python
mark = getattr(self, "_gh_mark", None)
```

THE LOGO IS NEVER DEAD. All three states have something to do sign in, sign in again, or install `gh` -- and two of them used to be greyed out, so on a machine where `gh` is already signed in the row read "Signed in" beside a control nothing happened on. Reported 2026-08-22 as "i cant click the github sign in", which is exactly what a disabled control looks like from the outside.

### lines 872-875

```python
if mark is not None:
```

NAMED, not "sign-in failed". The CLI being absent and the CLI being logged out need different things from the user -- and what the absent one needs is the install page, which is something this button can actually do.

## SetupSlides._sign_in_to_github

### lines 965-966

```python
process.setProcessChannelMode(QProcess.MergedChannels)
```

ONE STREAM. `gh` prints the code on stderr and the prompt on stdout, and reading only one of them loses half the exchange.

### lines 971-975

```python
self.destroyed.connect(lambda *_a: _let_go_of(process))
```

AND IT IS CLEANED UP IF THE DIALOG GOES FIRST. A QProcess destroyed with its child still running prints "QProcess: Destroyed while process is still running" and leaves `gh` parented to nothing -- so the dialog's destruction detaches it rather than taking it down mid-login.

### lines 990-992

```python
return True
```

The logo stays live while `gh` runs. There is no second control to disable now, and disabling the only one would leave a user whose browser never opened with nothing to click.

## SetupSlides._read_github_output

### lines 1025-1026

```python
try:
```

AND ANSWER THE PROMPT `gh` is sitting on, so it proceeds to poll GitHub. Without this it waits on Enter forever.

## SetupSlides._closing_page

### line 1048  _(unsure)_

```python
self._done_word = QLabel(str(title))
```

AS IT IS WRITTEN, not shouted: "Done", not "DONE".

### lines 1051-1058

```python
try:
```

THE SIZE GOES IN A STYLESHEET, not through setFont. The application sheet already gives every QLabel a font-size, and QSS beats a font set on the widget -- so setPointSize was overruled and the word came out the size of the sentence beneath it.

THE ACCENT BLUE, which is the blue the wordmark uses and the same one the greeting arrives in -- one blue for the things this screen is saying, rather than a second one invented for the last slide.

## SetupSlides._terms_page

### lines 1104-1108

```python
body = QLabel(terms_module.terms_text())
```

NOT TRANSLATED, and deliberately. A translated licence summary is not the licence, and offering one as though it were would have the screen promise something the document does not. The terms are shown in the language the licence is written in, with its name and its URL beside them; everything else on this page IS translated.

### lines 1115-1117

```python
scroll = QScrollArea(page)
```

SCROLLED, BECAUSE THE TERMS MAY OUTGROW THE CARD. A card that clips the last clause is a card asking for agreement to something it did not show.

### lines 1125-1129

```python
bar = scroll.verticalScrollBar()
```

BOTH SIGNALS, because there are two ways to arrive at the end. `valueChanged` is the reader scrolling; `rangeChanged` is the viewport growing until the whole document fits in it, which is the case a gate written as "the scroll bar moved" turns into a trap on a large monitor.

### lines 1145-1147

```python
self._scroll_hint = QLabel(_say(terms_module.SCROLL_HINT), page)
```

WHY THE SWITCH IS DEAD, said before the reader has to ask. It is visible from the moment the slide opens and goes when the gate does, so the greyed control is never unexplained.

### lines 1153-1154  _(unsure)_

```python
self._agree = Toggle(_say(terms_module.AGREE_LABEL), page)
```

A SLIDER, like every other boolean on this screen. A tick box is a form control and this is not a form.

### lines 1156-1157

```python
self._agree.toggled.connect(self._on_agreement_toggled)
```

AGREEING IS ITSELF THE ANSWER, so ticking the box clears the complaint rather than leaving it standing under a satisfied form.

### lines 1172-1175

```python
self._terms_read = False
```

CLOSED UNTIL PROVEN READ. The gate is drawn shut here rather than measured: the page has not been on screen yet, so there is nothing for "the end is on screen" to be true of, and the safe direction for a licence is the one that asks.

## SetupSlides._draw_the_terms_gate

### lines 1212-1214

```python
body.setStyleSheet("" if read else f"color: {self._dim_ink()};")
```

THE TEXT IS GREYED TOO, not only the switch. A live-looking document over a dead control reads as a broken control; one greyed page reads as a page waiting for something.

## SetupSlides._refuse_to_leave_the_terms

### lines 1263-1266

```python
target = getattr(self, "_agree" if read else "_terms_scroll", None)
```

THE KEYBOARD GOES WHERE THE WORK IS. A disabled switch cannot take focus, so a Next pressed before the end would leave the caret nowhere; the terms take it instead and Page Down carries on from where the reader is.

## SetupSlides._row

### lines 1284-1292

```python
return _held_at_the_top(label), editor
```

A CAPTION LINES UP WITH THE CONTROL, NOT WITH THE MIDDLE OF A COLUMN. Every other field here is one line tall, so centring the caption on it is right. The provider field is not: it is a row of logo marks with a status note underneath, and centred on the pair the caption landed in the gap between them level with nothing, and 42 px below the marks it names.

Held at the top of its cell, it sits beside the marks, which is the row it is the caption for.

## SetupSlides._editor

### lines 1313-1318

```python
box.currentIndexChanged.connect(self._apply_language)
```

AND THE WHOLE SCREEN CHANGES LANGUAGE WITH IT. Reported 2026-08-23: "language in the startup is also not implemented (other than english...)". A screen whose first question is the language and which then goes on asking the rest of them in English is asking the user to take the setting on faith.

### lines 1321-1322

```python
box.currentIndexChanged.connect(
```

APPLIED AS CHOSEN, for the same reason the greeting is:

the only way to know a look took is to see it.

### lines 1329-1330

```python
slider = Toggle()
```

THE APPLICATION'S OWN SLIDER, not a second one, so the gesture and the look are the ones the user meets everywhere else.

## SetupSlides._animation_row

### lines 1381-1383

```python
LOG.debug("no %s among the animations offered", DEFAULT_THEME)
```

THE DEFAULT IS NOT IN THE LIST, which means the ambient module and its own default disagree. There is nothing better left to show than the first entry, and it is worth a line in the log.

### lines 1387-1388

```python
box.currentIndexChanged.connect(self._apply_animation)
```

APPLIED AS CHOSEN, like the theme above it: a backdrop is a look, and the only way to know a look took is to see it.

## SetupSlides._apply_animation

### line 1416

```python
LOG.debug("could not store the animation choice", exc_info=True)
```

A BACKDROP THAT WILL NOT APPLY IS NOT A REASON TO STOP SETUP.

## SetupSlides._provider_buttons

### lines 1431-1432  _(unsure)_

```python
holder.setProperty("spacrProviderStrip", True)
```

Tagged so `_clear_the_containers` can find it without knowing the shape of the page it ended up on.

### lines 1434-1436

```python
column = QVBoxLayout(holder)
```

A COLUMN: the marks on one line, and under them a note, so that choosing a provider can say what it started. Without somewhere to say it, the sign-in would begin with no sign of it.

### lines 1451-1453

```python
state = self.provider_status(code, command)
```

SIGNED IN, not merely installed -- that is what the colour is for. An installed-but-signed-out provider used to look exactly like one that was ready to answer.

## SetupSlides._choose_provider

### lines 1701-1703

```python
status.setText(note)
```

SET EVEN WHEN EMPTY. A provider that is ready has nothing to say, and leaving the previous provider's note standing under it says something untrue about the one now selected.

### line 1705

```python
self._refresh_provider_marks(holder)
```

The mark's colour is the login state, so it has to be re-asked.

## SetupSlides._say_hello

### lines 1738-1740

```python
self._greeting.setText(greeting_for(box.currentData()))
```

AS THE WORD IS WRITTEN in each language -- "Hello", "Hej", "Hallo", 你好 -- rather than shouted. GREETINGS already holds each in its own conventional form, so there is nothing to do to it.

## SetupSlides._show_the_greeting

### lines 1755-1758

```python
self._greeting.setStyleSheet(
```

THE SIZE GOES IN THE SHEET TOO. The application stylesheet gives every QLabel a font-size and QSS beats a font set on the widget, so setPointSize here would be overruled and the word would come out the size of the prose above it.

### line 1777

```python
LOG.debug("no fade for the greeting", exc_info=True)
```

INVARIANTS 10: without an animation it is simply there.

## SetupSlides._fade_the_greeting_away

### lines 1788-1800

```python
greeting = getattr(self, "_greeting", None)
```

`isVisibleTo`, NOT `isVisible`. The latter is False whenever an ancestor is hidden -- a dialog built but not yet shown -- so the guard skipped the fade and left the word marked visible for whenever the dialog did appear. The question here is whether this widget is marked visible, which is what isVisibleTo answers. `getattr`, FOR THE SAME REASON `_place_the_greeting` FETCHES THE CARD THAT WAY. A resize can arrive while the dialog is still being built, and `_greeting` is created AFTER `card` -- so there is a window in which the card exists and this label does not. `_place_the_greeting` already survived it and this did not, which is the asymmetry instruction 310 A33 reports: the two methods disagreed about whether the label may be absent, and only one of them was right.

### lines 1815-1816

```python
animation.finished.connect(
```

HIDDEN AT THE END, not at the start: a hidden widget does not animate, so hiding first is the abrupt cut with extra steps.

### line 1822

```python
LOG.debug("no fade for the greeting", exc_info=True)
```

INVARIANTS 10: without an animation it simply goes.

## SetupSlides._place_the_gpu_note

### lines 1853-1855

```python
margin = 28
```

ACROSS THE CARD, INSIDE ITS MARGINS. The greeting is one word and can be centred in the full width; this is two lines of prose and would otherwise run into the rounded corners.

### lines 1859-1865

```python
height = note.heightForWidth(width)
```

HEIGHT FOR THIS WIDTH, not the bare size hint. A word-wrapped QLabel's `sizeHint()` is the height it would like if it could choose its own width, which for two sentences of prose is far taller than the wrapped text -- 323 px against a 700 px card. The box was then centred inside that, which put the words below the card's bottom edge: the verdict was written, coloured and placed, and simply not on screen.

### lines 1870-1879

```python
floor = card.height() - margin
```

AND IT CANNOT HANG OFF THE BOTTOM, NOR OVER THE BUTTONS. A card short enough that the band plus the wrapped height overflows lifts the note instead of losing it -- the note is the answer to "can this machine run spaCR", so a small window must not be the reason it is missed.

The floor is the NAV ROW, not the card's edge. The note grew a capability table on 2026-08-31 and the extra height put it straight over Back and the step counter, which is how a label that is only decoration ends up eating a button.

### lines 1892-1901

```python
try:
```

THE BUTTON IS NOT LAID OUT YET, AND THAT IS THE COMMON

CASE RATHER THAN THE EDGE ONE. Every caller of this method runs during slide setup, before the nav row has a geometry, so `mapTo` answers 0 and the clamp above was SKIPPED -- leaving the floor at the card's own edge and the note 18 px over Back at 900x640. The clamp read as protection and was inert exactly when it was needed.

A sizeHint is available before layout, so the floor can be computed without waiting for one.

## SetupSlides._say_what_the_gpu_is

### lines 1921-1925

```python
library = _gpu_library()
```

"GPU: <card>: <library>", with only the CARD coloured. Asked for on 2026-08-31. The eye should land on the thing that varies between machines; the word "GPU" and the library name are the same on every machine with that card, so colouring them too would just be more red or more green.

### lines 1931-1934

```python
hint = _say(GPU_DOCTOR_HINT)
```

NAMED THE CARD, SO SAY WHAT TO DO ABOUT IT. Finding an

NVIDIA card that torch cannot reach is a CUDA problem, not a hardware one, and the reader should not have to guess that from a red line.

### lines 1948-1958

```python
self._place_the_gpu_note()
```

RE-PLACED, BECAUSE THE TEXT JUST CHANGED ITS HEIGHT. The note is laid out by `_place_the_gpu_note`, which clamps it off the nav row using the height it has AT THAT MOMENT. This runs afterwards and makes it taller -- 94 px of prose became 125 with the capability table -- and the clamp had already been applied to the old height, so the note's BOTTOM grew back down over the Back button by 18 px. The clamp was doing its job on a number that then changed.

Measured at 900x640: placed at y=487 for a floor of 581 and a height of 94; the final geometry is 125 tall, bottom 612, against a button top of 593.

## SetupSlides._what_this_machine_can_do

### lines 2018-2021

```python
LOG.debug("no capability row starts with %r", prefix)
```

A capability row was renamed. Drop the table row rather than draw an empty cell: a blank middle column reads as "spaCR does not know", which is a worse thing to say than nothing.

### lines 2030-2035

```python
f'<td align="right" style="padding-right:14px;">'
```

RIGHT-ALIGNED, asked for 2026-09-01. The library names differ in length -- "UMAP / t-SNE / cluster" against "Torch models" -- so ragging them left puts the GPU/CPU column a different distance from each one. Aligned right, the verdicts line up against a straight edge and read as a column.

### lines 2043-2046

```python
rows.append(
```

CENTRED. Qt's rich text does not honour `margin:auto`, so the table is centred by wrapping it in a block that is, which is the one construction that works in QLabel's subset of HTML.

### lines 2052-2054

```python
rows.append(
```

FOUND AND NOT USED, said in as many words. There is no portable torch device for a neural engine, so silence here would read as "spaCR did not look".

## SetupSlides.retranslate

### lines 2100-2102

```python
self._show_slide(self._index)
```

THE TITLE AND THE BLURB ARE COMPOSED, so the walker sees the composition rather than the sentence, and the slide has to put them back itself.

## SetupSlides._apply_look

### lines 2117-2118

```python
LOG.debug("could not apply %s live", key, exc_info=True)
```

A LOOK THAT WILL NOT APPLY MUST NOT LOSE THE ANSWER. It is still written with the rest on accept.

## SetupSlides._show_slide

### lines 2138-2140

```python
self._title.setText(f"<b>{_say(title)}</b>")
```

TRANSLATED HERE, not in the table. SLIDES holds the English the catalog is keyed on; a slide re-shown after the language changes picks up the new rendering because this runs again.

### lines 2143-2148

```python
note = getattr(self, "_gpu_note", None)
```

THE GREETING BELONGS TO THE LANGUAGE SLIDE and nowhere else: a "Hello" left standing over the theme question is a word with no job on that page. THE GREETING BELONGS TO THE MOMENT THE LANGUAGE IS CONFIRMED, not to a slide. It is shown by the first Next and hidden again by anything that leaves that moment behind.

### lines 2152-2154

```python
self._place_the_gpu_note()
```

PLACED THE MOMENT IT IS SHOWN. Nothing else lays this label out, so a note made visible and left unplaced sits in the corner it was born in.

### lines 2158-2161

```python
self._where.setText(
```

NOT ON THE FIRST SLIDE. Slide one carries the greeting and the capability table and has no room for a counter under them; "1 of 7" was landing on top of the note. It also says least there nobody needs telling they are at the beginning.

### lines 2168-2172

```python
self._drop_the_fade()
```

A LEFTOVER FADE IS DROPPED WHETHER OR NOT A NEW ONE STARTS. The effect belongs to the page STACK, not to a page, so one still running when the slide changes again leaves the new page wearing an opacity that stopped part-way -- which draws an empty card and is indistinguishable from a page that failed to build.

## SetupSlides._fade_in

### lines 2207-2210

```python
animation = QPropertyAnimation(effect, b"opacity", self)
```

`setGraphicsEffect` takes ownership and deletes whatever was there, so the old animation is now driving a dead object. `_drop_the_fade` above has already stopped it; this is the note for anyone who moves that call.

### line 2220

```python
LOG.debug("no cross-fade on this platform", exc_info=True)
```

INVARIANTS 10: the slide is shown either way.

## SetupSlides.next

### lines 2263-2264

```python
self.card.circuit(clockwise=True)
```

THE PRESS IS ANSWERED, just not with a page change. The button is live so the reader learns what is missing by using it.

## SetupSlides._advance_after_the_greeting

### lines 2292-2293

```python
LOG.debug("no timer for the greeting pause", exc_info=True)
```

INVARIANTS 10: without a timer the slides still advance, they just do not wait.

## SetupSlides._record_the_agreement

### lines 2391-2393

```python
LOG.warning("the terms acceptance could not be recorded",
```

A STORE THAT WILL NOT TAKE THE RECORD ASKS AGAIN NEXT TIME, which is the safe direction: the alternative is treating an unwritten acceptance as given.

## SetupSlides._install_backdrop

### lines 2409-2413

```python
return install_ambient(self, theme=BACKDROP_THEME,
```

ROUNDED TO THE CARD'S RADIUS. The dialog is frameless and translucent and holds exactly one card; a square backdrop behind a rounded card is a second surface, and looked like one -- "there is a square window with the theme and in front of that window is a dark square with rounded edges".

## SetupSlides.resizeEvent

### lines 2427-2432

```python
self.card.setGeometry(self.rect())
```

NO MARGIN. The card used to be inset by 44px inside the ambient backdrop, which put a themed square around a rounded card and made the dialog read as two windows. The card now IS the window: same rectangle as the backdrop, same corner radius, so there is one rounded translucent surface and the settings sit on it rather than in a container floating over it.

### lines 2435-2436

```python
self._look_at_the_terms_gate()
```

A WINDOW MADE TALLER CAN PUT THE END OF THE TERMS ON SCREEN, and that is the whole of the gate's question.

### lines 2438-2440

```python
try:
```

THE WINDOW IS CUT TO THE CARD'S SHAPE, the same way every glassed popup is, so the two surfaces are the same surface and not two takes on one idea.

## _catalogue_this_screen

### lines 2466-2469

```python
add_translation(ANIMATION_LABEL, (
```

THE ONE ROW THIS MODULE OWNS. Every other caption on the screen comes from `setup_screen.questions()` or from `terms`; the animation question is asked here, so its caption is catalogued here, through the same seam.

## open_setup_if_needed

### lines 2494-2496

```python
if skipped_on_purpose():
```

WHETHER THIS PROFILE IS DUE and whether THIS LAUNCH CAN ASK are two different questions. `should_open` answers the first; a batch job on a server can be due and still have nobody to answer.

### lines 2499-2502

```python
if not should_open() and not needs_agreement():
```

UNACCEPTED TERMS ARE THEIR OWN REASON TO ASK. Dismissing the screen marks the questions answered -- they all have defaults -- but a licence is not answered by a default, so terms that were never accepted, or accepted at an older version, bring the screen back.

## The Install button and the GitHub account button, 2026-09-19 (item 420)

```python
act_install = (box.addButton(_say("Install"),
```

"in the startup spacr when the use clicks an AI provider there should be an aditional button, install which automatically downloads and installs the chosen ai provider and asks the user for the needed information and sets up the AI provider" and "same for the github cli, add that to and a button that links to generating a github account" (maintainer, 2026-09-16).

The prompt that already opened for a provider that is not set up gets Install, and the GitHub mark with no `gh` now opens the same kind of prompt instead of going straight to cli.github.com. NOTHING RUNS UNTIL INSTALL IS PRESSED, and the prompt shows the exact command first. The install runs in the `CliSetupPanel` under the marks (or under the GitHub row); when it finishes, the provider's sign-in starts -- in a terminal for the AI CLIs, in spaCR's own `gh auth login` flow for GitHub -- and the panel asks the tool's status command every 3 s until it says yes. When no install row can run here, the prompt names the missing program and keeps the page and the command.

`_github_sign_in_ended` replaces the bare refresh on `gh`'s `finished`: a non-zero exit while the panel is waiting is a sign-in that was cancelled or expired, and the panel says so instead of waiting out its ten minutes.

"Create a GitHub account" opens https://github.com/signup and is hidden once a token is found.
