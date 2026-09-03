"""Picture-rendering controls shared by cell and image views.

Defaults come from :func:`spacr.settings.set_annotate_default_settings`, and
mode applicability comes from :mod:`spacr.picture_settings`. The dialog thus
uses the same values and availability rules as non-GUI callers.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ...crops import LOAD_IMAGES
from ...picture_settings import ALL_KEYS, applies_to, categories, why_not

__all__ = ["PictureSettingsDialog", "picture_defaults"]


def picture_defaults() -> Dict[str, Any]:
    """Return typed defaults for every setting offered by the dialog."""
    from ...settings import set_annotate_default_settings

    try:
        filled = set_annotate_default_settings({})
    except Exception:                                        # noqa: BLE001
        filled = {}
    if not isinstance(filled, dict):
        filled = {}
    from ...picture_settings import OWN_DEFAULTS

    # OWN_DEFAULTS WINS WHERE IT SPEAKS, and that is not a preference for
    # our own table: it is where a shipped default is the wrong TYPE for a
    # control. `set_annotate_default_settings` ships the STRING 'False' for
    # `edge_image`, and a non-empty string is TRUE -- so the flag read as on
    # everywhere it was used as one, and this dialog drew a text box
    # containing the word False instead of a checkbox. The annotator's value
    # is still what fills every key OWN_DEFAULTS does not name.
    out = {}
    for key in ALL_KEYS:
        if key in OWN_DEFAULTS:
            out[key] = OWN_DEFAULTS[key]
        else:
            out[key] = filled.get(key)
    return out


#: Settings that name channels rather than choosing an item from a list.
#: ``outline`` uses the same red/green/blue vocabulary and parsing rules as
#: ``normalize_channels``, so the dialog presents them with the same control.
#: ``_as_channel_list`` parses the
#: comma-separated string either control produces. Only the widget differed.
CHANNEL_KEYS = ("channels", "normalize_channels", "outline")

#: Settings that are a PAIR of numbers rather than one value.
#:
#: A window is two numbers, so it is asked for as two numbers. As one text
#: box it was a parsing problem handed to the user -- `[1, 99]` and `[1 99]`
#: are one intent, and only one of them survived the trip to the renderer.
PAIR_KEYS = ("percentiles",)


def _editor(value: Any, parent: Optional[QWidget] = None,
            choices: Any = ()) -> QWidget:
    """A control suited to ``value``'s type.

    Deliberately small: a float gets a step that follows its magnitude, for
    the reason the settings panel had to be taught the same thing -- a spin
    box left at Qt's default step of 1.0 turns 0.05 into -0.95 on one wheel
    tick.
    """
    if choices:
        # BUILT FROM THE SCREEN, not typed. Offering `object_array` as free
        # text asks the user to remember what their own screen contains and
        # to spell it the way `measure` did -- and every other chooser in
        # spaCR is built from the data.
        combo = QComboBox(parent)
        for option in choices:
            # A chooser may offer (value, label) or a bare value. The STORED
            # value is always the first, so a label can be renamed without
            # changing what any settings file already on disk means.
            #
            # `stored`, NOT `value`: the first version of this loop unpacked
            # into `value` and so clobbered the parameter it was about to
            # search for -- every dropdown then opened on its LAST entry,
            # whatever the setting actually was.
            if isinstance(option, tuple) and len(option) == 2:
                stored, label = option
            else:
                stored = label = option
            combo.addItem(str(label), stored)
        current = combo.findData(value)
        if current < 0:
            current = combo.findText(str(value))
        combo.setCurrentIndex(max(current, 0))
        return combo
    if isinstance(value, bool):
        box = QCheckBox(parent)
        box.setChecked(value)
        return box
    if isinstance(value, int):
        spin = QSpinBox(parent)
        spin.setRange(0, 1_000_000)
        spin.setValue(int(value))
        return spin
    if isinstance(value, float):
        spin = QDoubleSpinBox(parent)
        spin.setDecimals(4)
        spin.setRange(-1e6, 1e6)
        spin.setSingleStep(0.01 if abs(value) < 1 else 0.1)
        spin.setValue(float(value))
        return spin
    edit = QLineEdit(parent)
    edit.setText("" if value is None else str(value))
    return edit


def _value_of(widget: QWidget) -> Any:
    from .percentile_pair import PercentilePair

    if isinstance(widget, PercentilePair):
        # THE PAIR STAYS A PAIR. Read through `text()` like any other
        # unfamiliar editor it would come back as the string "2, 98", and
        # every settings file already on disk holds a two-element list.
        return widget.value()
    if isinstance(widget, QComboBox):
        data = widget.currentData()
        return widget.currentText() if data is None else data
    if isinstance(widget, QCheckBox):
        return widget.isChecked()
    if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
        return widget.value()
    return widget.text()


def _attach_picking_help(editor, key: str) -> None:
    """Give each annotation method's dropdown entry its own explanation.

    A no-op for every other setting and for a control that is not a combo,
    so the caller does not have to know which is which.
    """
    if str(key) != "cell_picking":
        return
    from PySide6.QtWidgets import QComboBox

    if not isinstance(editor, QComboBox):
        return
    from ...picture_settings import PICKING_HELP

    for index in range(editor.count()):
        value = editor.itemData(index)
        if value is None:
            value = editor.itemText(index).split(" ")[0]
        help_text = PICKING_HELP.get(str(value))
        if help_text:
            editor.setItemData(index, help_text, Qt.ToolTipRole)


class PictureSettingsDialog(QDialog):
    """Edit picture settings while retaining mode-inapplicable values.

    :param values: the settings to open with. Keys outside the picture
        vocabulary are IGNORED rather than carried, and keys the dialog knows
        but the current mode does not apply are kept untouched -- which is
        what "retaining" in the summary above means, and why editing PNG
        settings does not silently drop the array ones.
    :param mode: which crop SOURCE the pictures come from. It decides which
        settings apply -- a plane index means something to the array route
        and nothing to the database route, which finds its rows by
        coordinate columns -- so an inapplicable setting is greyed with a
        reason rather than hidden.
    :param parent: parent widget.
    :param source: where the pictures come from, for the previews.
    :param objects: the objects available to crop, for the previews.
    """

    def __init__(self, values: Optional[Dict[str, Any]] = None,
                 mode: str = "png", parent: Optional[QWidget] = None, *,
                 source: Any = None, objects: Any = None):
        super().__init__(parent)
        self.setWindowTitle("Picture settings")
        self._mode = str(mode or "png")
        self._editors: Dict[str, QWidget] = {}
        self._labels: Dict[str, QLabel] = {}
        #: The cap label's help WITHOUT the cost sentence, so re-stating the
        #: cost replaces it instead of stacking another copy on the end.
        self._cap_help: Optional[str] = None

        start = dict(picture_defaults())
        start.update({k: v for k, v in (values or {}).items() if k in ALL_KEYS})

        self._tabs = QTabWidget(self)
        self._tab_of: Dict[str, str] = {}
        layout = QVBoxLayout(self)
        from ...picture_settings import offered_values

        # THE R,G,B SYSTEM FOR THE TWO CHANNEL SETTINGS (188 B). A dropdown
        # of the eight combinations made "which channels are on" a question
        # you had to open a list to answer, and turning one channel off --
        # the thing a user does constantly here -- two clicks.
        from .channel_picker import ChannelPicker
        from .percentile_pair import PercentilePair

        # ONE TAB PER GROUP OF QUESTIONS. Twenty-eight controls in one column
        # made the reader scroll past every question they were not asking to
        # reach the one they were; the module screens group their settings
        # the same way and `picture_settings.categories` is the one table
        # both read.
        for title, keys in categories():
            page = QWidget(self._tabs)
            form = QFormLayout(page)
            form.setLabelAlignment(Qt.AlignRight)
            for key in keys:
                value = start.get(key)
                if key == "cap" and isinstance(value, float):
                    # CAP IS A COUNT even when a settings table round-trip
                    # stores it as a float. Normalise at the boundary, using
                    # the same truncation the montage already applies, so the
                    # dialog cannot hand a fractional object count back to its
                    # caller and the control remains the live-wired QSpinBox.
                    value = int(value)
                if key in CHANNEL_KEYS:
                    editor = ChannelPicker(
                        value, page,
                        # `channels` with nothing on is a blank picture;
                        # `normalize_channels` with nothing on means
                        # "normalise nothing" and `outline` with nothing on
                        # means "outline nothing" -- both real answers, and
                        # `outline`'s default is off.
                        allow_none=(key != "channels"))
                elif key in PAIR_KEYS:
                    editor = PercentilePair(value, page)
                else:
                    editor = _editor(value, page,
                                     choices=offered_values(key, source=source,
                                                            frame=objects))
                label = QLabel(key.replace("_", " "), page)
                # THE TOOLTIP IS ON THE LABEL, not the field: a tooltip on
                # the control fires while the user is editing it, which is
                # the one moment they did not ask for it.
                # EACH ANNOTATION METHOD EXPLAINS ITSELF, ON ITS OWN ENTRY
                # (208 C). "The API should be verry specific and evplain
                # exactly how cells are cjhoosen for annotation" -- and five
                # methods cannot be explained in one tooltip that has to fit
                # in 600 characters. Per-entry help is the only place with
                # room, and it is where the choice is actually made.
                _attach_picking_help(editor, key)
                self._editors[key] = editor
                self._labels[key] = label
                self._tab_of[key] = title
                form.addRow(label, editor)
            self._tabs.addTab(page, title)
        layout.addWidget(self._tabs)

        # THE COST FOLLOWS THE NUMBER. Written once at build time it would
        # describe the cap the dialog opened on, which is the one value the
        # user is not asking about while they change it.
        cap = self._editors.get("cap")
        if isinstance(cap, QSpinBox):
            cap.valueChanged.connect(
                lambda _v: self._say_what_the_cap_costs())

        # AND THE GREYING FOLLOWS THE MODE CHOSEN *HERE*. `crop_source` is one
        # of the annotator's own controls, so it is a control IN this window
        # as well as on the toolbar that opened it -- and a mode read once at
        # build time described the mode the window OPENED on. A user who
        # switched to streaming here was then told the array and channel
        # selectors their chosen mode does use were unavailable, with a
        # reason naming the mode they had just left.
        source_editor = self._editors.get("crop_source")
        if isinstance(source_editor, QComboBox):
            source_editor.currentIndexChanged.connect(
                lambda _i: self.set_mode(
                    str(source_editor.currentData()
                        or source_editor.currentText() or LOAD_IMAGES)))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   parent=self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # THE SAME API HELP THE SETTINGS PANEL GIVES, not a plainer copy.
        # Reported: "there are still no tooltips with api guides". These
        # labels carried the description string and nothing else -- no
        # `settingKey`, no rendered `apiTooltipHtml`, so no link to the API
        # page and none of the typed metadata every other reader keys on.
        #
        try:
            from ..screens.settings_model import install_api_tooltips

            install_api_tooltips(
                self, "annotate",
                {editor: key for key, editor in self._editors.items()})
        except Exception:                                    # noqa: BLE001
            # A dialog that cannot decorate its help is still a dialog that
            # sets the picture. The plain descriptions installed below remain.
            pass

        self.set_mode(self._mode)
        # HOVER HELP BELONGS TO THE SETTING'S NAME, never to the box
        # you type in. Built here on the field, it is moved onto the
        # label as the last step, so every panel in the application
        # explains itself the same way.
        from ..screens.settings_model import retarget_field_tooltips
        retarget_field_tooltips(self)

    # ------------------------------------------------------------------ mode

    def set_mode(self, mode: str) -> None:
        """Update control availability for an image-source mode.

        Inapplicable controls remain visible and explain why they are disabled,
        so switching modes does not hide or discard configured values.
        """
        self._mode = str(mode or "png")
        from ...settings import tooltips

        for key, editor in self._editors.items():
            usable = applies_to(key, self._mode)
            editor.setEnabled(usable)
            label = self._labels[key]
            label.setEnabled(usable)
            if not usable:
                # THE REASON BEATS THE DESCRIPTION when the control is
                # greyed: what the user is asking at that moment is why they
                # cannot touch it, not what it would have done.
                label.setToolTip(why_not(key, self._mode))
                continue
            # THE RICH API HELP IF IT WAS INSTALLED, and the plain
            # description otherwise. This line used to write the plain text
            # unconditionally, straight over the HTML `install_api_tooltips`
            # had just rendered -- so every label carried the API metadata
            # and showed none of it. Reported as "there are still no
            # tooltips with api guides".
            rich = str(label.property("apiTooltipHtml") or "")
            label.setToolTip(rich or str(tooltips.get(key, "") or ""))
            if key == "cap":
                self._cap_help = str(label.toolTip() or "")
        self._say_which_tabs_this_mode_uses()
        self._say_what_the_cap_costs()

    def _say_what_the_cap_costs(self) -> None:
        """Put the measured cost of the chosen cap beside the cap itself.

        A CAP IS A DECISION ABOUT WHERE A LIMIT SITS, and the numbers that
        decide it -- how many pages a reader has to walk, how much memory the
        tab holds while they do, and how long the cut takes -- are on screen
        nowhere else. Raising it without them is raising it blind.
        """
        from ...picture_settings import montage_cap_cost

        label = self._labels.get("cap")
        editor = self._editors.get("cap")
        if label is None or editor is None:
            return
        # THE HELP IS REMEMBERED, not read back off the label. Re-reading it
        # appends the new sentence to the last one, so a reader who tried
        # three caps got three of them.
        base = self._cap_help
        if base is None:
            base = str(label.toolTip() or "")
            self._cap_help = base
        cost = montage_cap_cost(_value_of(editor))
        if not cost:
            # ZERO HAS NO PRICE. Remove the previous count's sentence rather
            # than leaving a tooltip that describes a value no longer in the
            # control; the setting's original help remains truthful at zero.
            label.setToolTip(base)
            return
        # PLAIN OR RICH, whichever the label ended up with: the API help is
        # HTML, so the sentence is appended as a paragraph there and as a
        # blank line on a plain tooltip.
        joiner = "<p>{0}</p>" if base.lstrip().startswith("<") else "\n\n{0}"
        label.setToolTip(base + joiner.format(cost) if base else cost)

    def _say_which_tabs_this_mode_uses(self) -> None:
        """Put the greyed count for each tab on the tab itself.

        A GREYED CONTROL IS ONLY AN EXPLANATION IF IT IS FOUND. Behind a tab
        the reason a control cannot be touched is a hover the reader has to
        go looking for, so the tab says how many of its settings this mode
        does not use -- and it stays selectable, because the reason lives on
        the labels inside it.
        """
        for index in range(self._tabs.count()):
            title = self._tabs.tabText(index)
            keys = [k for k, tab in self._tab_of.items() if tab == title]
            greyed = [k for k in keys if not applies_to(k, self._mode)]
            if not greyed:
                self._tabs.setTabToolTip(index, "")
                continue
            self._tabs.setTabToolTip(
                index,
                f"{len(greyed)} of {len(keys)} settings here are not used by "
                f"the chosen image source. They stay on the tab, and each "
                f"one's label says why.")

    def mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------------ tabs

    def tab_titles(self) -> tuple:
        """The tabs, in the order they are shown."""
        return tuple(self._tabs.tabText(i) for i in range(self._tabs.count()))

    def tab_of(self, key: str) -> str:
        """Which tab ``key``'s control is on, or ``""`` if it has none."""
        return self._tab_of.get(str(key or "").strip(), "")

    def show_tab(self, title: str) -> bool:
        """Bring the tab named ``title`` to the front. False if there is none.

        A caller that wants one question answered can open the panel on it
        rather than on whichever tab happened to be first.
        """
        for index in range(self._tabs.count()):
            if self._tabs.tabText(index) == str(title):
                self._tabs.setCurrentIndex(index)
                return True
        return False

    # ---------------------------------------------------------------- values

    def values(self) -> Dict[str, Any]:
        """Return every configured value, including disabled controls.

        Values for the current mode's disabled controls are preserved so that
        switching away from a mode and back restores the prior configuration.
        """
        return {key: _value_of(editor)
                for key, editor in self._editors.items()}

    def applied_values(self) -> Dict[str, Any]:
        """Only the settings the current mode actually uses."""
        return {key: value for key, value in self.values().items()
                if applies_to(key, self._mode)}
