"""The Sound tab of Preferences: every switch off until somebody asks.

Built into the dialog by :class:`spacr.qt.preferences.PreferencesDialog`
as its LAST tab -- it is the least important thing in Preferences -- and
kept in its own module so the dialog only has to create the page, save it
and reset it. Built ONLY IN SPACEOUT MODE since 2026-09-21: ordinary spaCR
has no Sound tab and plays nothing (see
:func:`spacr.qt.preferences.sound_is_offered`).

Every event has its own switch and a Preview button beside it. The
switches, the volume, the sound set and the Previews are all disabled while
the master switch is off, so nothing on this page can make a sound until
the user has turned sound on; a Preview is the user asking to hear one.

Every caption is a literal at the line that shows it, rather than a row in
a table handed to a helper: the translation extractor reads literals at
the calls it knows, and a caption routed through a helper's parameter is a
caption it never sees (``tests/test_a_helper_does_not_hide_a_caption_from_the_catalog.py``).
"""
from __future__ import annotations

from typing import Dict

__all__ = ["SOUND_PAGE_EVENTS", "SoundPage"]

#: The events the page offers, in the order it lists them: the two interface
#: sounds, the two run sounds, then the music bed.
SOUND_PAGE_EVENTS = ("click", "hover", "run_finished", "run_failed", "bed")


class SoundPage:
    """The Sound tab's controls, and the two things the dialog asks of them.

    :param form: the tab's form layout, from the dialog's ``_page``.
    :param dialog: the Preferences dialog; closing it ends any preview.
    """

    def __init__(self, form, dialog) -> None:
        """Build every row, reading the stored values."""
        from PySide6.QtCore import Qt
        from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel,
                                       QLineEdit, QPushButton, QSlider,
                                       QWidget)

        from . import preferences as prefs
        from .i18n import tr
        from .sound_synth import SOUND_THEMES
        from .widgets.toggle import Toggle

        self._dialog = dialog
        self.events: Dict[str, object] = {}
        self.previews: Dict[str, object] = {}

        help_label = QLabel(tr(
            "Everything here is off until you switch sound on. The sounds "
            "are made by spaCR on this computer, nothing is recorded or "
            "downloaded, and they play on a thread of their own, so a "
            "missing or busy audio device only ever means silence."))
        help_label.setWordWrap(True)
        help_label.setObjectName("SoundTabHelp")
        form.addRow(help_label)

        self.enabled = Toggle()
        self.enabled.setObjectName("SoundEnabled")
        self.enabled.setToolTip(
            "Off by default. While it is off spaCR loads no audio library "
            "and plays nothing at all. On, spaCR plays the sounds switched "
            "on below through your computer's default audio output.")
        self.enabled.setChecked(prefs.get_sound_enabled())
        form.addRow(tr("Play sounds"), self.enabled)

        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setObjectName("SoundVolume")
        self.volume.setRange(0, 100)
        self.volume.setSingleStep(5)
        self.volume.setPageStep(10)
        self.volume.setValue(int(round(prefs.get_sound_volume() * 100)))
        volume_value = QLabel()
        volume_value.setObjectName("SoundVolumeValue")
        self.volume.valueChanged.connect(
            lambda value: volume_value.setText(f"{int(value)}%"))
        volume_value.setText(f"{self.volume.value()}%")
        volume_row = QWidget()
        volume_layout = QHBoxLayout(volume_row)
        volume_layout.setContentsMargins(0, 0, 0, 0)
        volume_layout.addWidget(self.volume, 1)
        volume_layout.addWidget(volume_value)
        volume_row.setToolTip(
            "How loud every spaCR sound is, from silent to full. The music "
            "bed always plays quieter than the other sounds.")
        form.addRow(tr("Volume"), volume_row)

        self.theme = QComboBox()
        self.theme.setObjectName("SoundTheme")
        for key, theme in SOUND_THEMES.items():
            self.theme.addItem(tr(theme.label), key)
            self.theme.setItemData(self.theme.count() - 1,
                                   tr(theme.description),
                                   Qt.ItemDataRole.ToolTipRole)
        self.theme.setCurrentIndex(
            max(0, self.theme.findData(prefs.get_sound_theme())))
        self.theme.setToolTip(
            "Which set of sounds plays. A set is synthesized on this "
            "computer the first time it is needed and kept under "
            "~/.spacr/sounds, so it costs nothing until it is used.")
        form.addRow(tr("Sound set"), self.theme)

        row = self._event_row("click", "SoundClick")
        row.setToolTip(
            "A short, quiet pluck in the sound set's key when you press a "
            "button, a switch, a tab, a list or a slider: confirmation that "
            "the press registered, felt more than heard.")
        form.addRow(tr("Click"), row)

        row = self._event_row("hover", "SoundHover")
        row.setToolTip(
            "An even quieter pluck when the pointer rests on an enabled "
            "control. It waits for the pointer to settle and then stays "
            "silent for a moment, so moving across a panel never turns into "
            "a stream of sounds. Off until you switch it on here.")
        form.addRow(tr("Hover"), row)

        row = self._event_row("run_finished", "SoundRunFinished")
        row.setToolTip(
            "A rising arpeggio that resolves onto the key's home note when a "
            "module's run finishes successfully, for when spaCR is in "
            "another window.")
        form.addRow(tr("Run finished"), row)

        row = self._event_row("run_failed", "SoundRunFailed")
        row.setToolTip(
            "A slower falling figure when a run stops with an error, so a "
            "finished run and a failed one never sound alike. A run you "
            "stop yourself is silent.")
        form.addRow(tr("Run failed"), row)

        row = self._event_row("bed", "SoundMusicBed")
        row.setToolTip(
            "A quiet looping piece in the sound set's key: pads, a plucked "
            "arpeggio with a dotted-eighth echo, a soft sub, and a quiet "
            "four-on-the-floor kick and shaker that come and go with the "
            "arrangement. Separate from the other sounds, so you can have "
            "feedback without music. It rests at the Laptop and Extra "
            "Performance levels.")
        self.previews["bed"].setToolTip(tr(
            "Play a few seconds of the music bed at the volume above."))
        form.addRow(tr("Music bed"), row)

        self.music = QLineEdit()
        self.music.setObjectName("SoundMusicFile")
        self.music.setPlaceholderText(tr("spaCR's own music"))
        self.music.setText(prefs.get_sound_music_file())
        browse = QPushButton(tr("Browse"))
        browse.setObjectName("SoundMusicFileBrowse")
        browse.clicked.connect(self._choose_music)
        self.browse = browse
        music_row = QWidget()
        music_layout = QHBoxLayout(music_row)
        music_layout.setContentsMargins(0, 0, 0, 0)
        music_layout.addWidget(self.music, 1)
        music_layout.addWidget(browse)
        music_row.setToolTip(
            "A WAV file of your own for the music bed to play instead of "
            "spaCR's. Leave it empty for spaCR's own music. Nothing is "
            "uploaded or copied: the file is read from where it is, and it "
            "is also what the Resonance background moves to.")
        form.addRow(tr("Music file"), music_row)

        self.enabled.toggled.connect(self._follow_the_master)
        self._follow_the_master(self.enabled.isChecked())
        try:
            dialog.finished.connect(self._closed)
        except (AttributeError, RuntimeError, TypeError):
            pass

    def _event_row(self, event: str, name: str):
        """One event's switch and its Preview button, in a row widget.

        The Preview button carries ``spacrSilentPress``
        (:data:`spacr.qt.sound.SILENT_PRESS_PROPERTY`), so pressing it makes
        no click sound of its own: a Preview is the user asking to hear one
        sound, and the Click row's Preview would otherwise play the same
        pluck twice. The name is written out rather than imported because
        building this page must not pull the sound engine in for somebody
        who has sound switched off.

        :param event: the event the row controls.
        :param name: object name of the switch; the button is
            ``<name>Preview``.
        :returns: the row widget, for the caller to caption and explain.
        """
        from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget

        from . import preferences as prefs
        from .i18n import tr
        from .widgets.toggle import Toggle

        toggle = Toggle()
        toggle.setObjectName(name)
        toggle.setChecked(prefs.get_sound_event_enabled(event))
        preview = QPushButton(tr("Preview"))
        preview.setObjectName(f"{name}Preview")
        preview.setProperty("spacrSilentPress", True)
        preview.setToolTip(tr("Play this sound once at the volume above."))
        preview.clicked.connect(
            lambda _checked=False, which=event: self._preview(which))
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(toggle)
        layout.addStretch(1)
        layout.addWidget(preview)
        self.events[event] = toggle
        self.previews[event] = preview
        return row

    def _follow_the_master(self, on: bool) -> None:
        """Every other control on the page is live only while sound is on.

        Switching the master off also ends any preview still playing.
        Greying the Preview buttons stops the user starting another one and
        does nothing about the nine seconds of music bed already running,
        which is the one sound on this page long enough to outlive the
        switch that started it.
        """
        for widget in ([self.volume, self.theme, self.music, self.browse]
                       + list(self.events.values())
                       + list(self.previews.values())):
            widget.setEnabled(bool(on))
        if not on:
            self._end_any_preview()

    @staticmethod
    def _end_any_preview() -> None:
        """End a preview, without loading the engine to find there is none.

        Nothing can be playing unless something has already imported
        :mod:`spacr.qt.sound`, so an unloaded module is the answer rather
        than a reason to load it: this page is built every time anybody
        opens Preferences, sound or no sound.
        """
        import sys

        if sys.modules.get(f"{__package__}.sound") is None:
            return
        from .sound import stop_sound_preview
        stop_sound_preview()

    def _choose_music(self) -> str:
        """Ask for a WAV for the music bed and put it in the field.

        Cancelling leaves the field alone, which is what a user who opened
        the dialog to look at a folder expects; clearing the field by hand
        is how you go back to spaCR's own music.

        :returns: the path chosen, or ``""``.
        """
        from PySide6.QtWidgets import QFileDialog

        from .i18n import tr

        chosen, _filter = QFileDialog.getOpenFileName(
            self._dialog, tr("Choose music for the bed"),
            self.music.text().strip(), tr("WAV audio (*.wav)"))
        if chosen:
            self.music.setText(str(chosen))
        return str(chosen or "")

    def _preview(self, event: str) -> bool:
        """Play ``event`` with the page's own, unsaved, set and volume."""
        if not self.enabled.isChecked():
            return False
        from .sound import preview_sound
        return preview_sound(event, str(self.theme.currentData() or ""),
                             self.volume.value() / 100.0,
                             music=self.music.text().strip())

    def select_theme(self, key: str) -> bool:
        """Show ``key`` in the Sound set control, without saving anything.

        The Appearance tab calls this when one of the ten night themes is
        chosen, so the sound set that goes with the colours is visible on
        the Sound tab before Save is pressed rather than appearing there
        afterwards. It moves one combo box and nothing else: the master
        switch, the volume and the per-event switches are the user's and
        are not touched, so a theme can never turn sound on.

        :param key: a key of :data:`spacr.qt.sound_synth.SOUND_THEMES`.
        :returns: whether a set of that name was found and selected.
        """
        index = self.theme.findData(str(key))
        if index < 0:
            return False
        self.theme.setCurrentIndex(index)
        return True

    def _closed(self, *_result) -> None:
        """The dialog closed: no preview outlives it."""
        self._end_any_preview()

    def save(self) -> None:
        """Write every control on the page to the preference store."""
        from . import preferences as prefs

        prefs.set_sound_enabled(self.enabled.isChecked())
        prefs.set_sound_volume(self.volume.value() / 100.0)
        key = self.theme.currentData()
        if key:
            prefs.set_sound_theme(str(key))
        prefs.set_sound_music_file(self.music.text())
        for event, toggle in self.events.items():
            prefs.set_sound_event_enabled(event, toggle.isChecked())

    def reset(self) -> None:
        """Show what a fresh install has.

        Called while the dialog has swapped in its empty defaults store, so
        every getter answers with its default.
        """
        from . import preferences as prefs

        self.enabled.setChecked(prefs.get_sound_enabled())
        self.volume.setValue(int(round(prefs.get_sound_volume() * 100)))
        self.theme.setCurrentIndex(
            max(0, self.theme.findData(prefs.get_sound_theme())))
        self.music.setText(prefs.get_sound_music_file())
        for event, toggle in self.events.items():
            toggle.setChecked(prefs.get_sound_event_enabled(event))
        self._follow_the_master(self.enabled.isChecked())
