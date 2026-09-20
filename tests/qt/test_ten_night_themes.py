"""The ten night themes: ten of them, all legible, and each one three things.

Asked 2026-09-19: "make 10 themes for spacr that should all try and capture
a space house theme vibe", and separately "The theme should sound like
melodic space house". The instruction spells out what a theme has to be:
a coherent palette that stays legible, an ambient backdrop, and a sound
identity of its own, "so choosing a theme changes look AND sound together"
-- and it spells out what a theme must NOT be: named after any of the four
artists it names, or after anything trademarked.

Every one of those clauses is turned into something that can be measured.
Legibility is not an opinion here: it is
``theme.contrast_failures`` and ``theme.page_separation_failures``, the
same two published rules the four older themes are judged by, run over the
ten. "Changes look AND sound together" is the Preferences dialog, driven
with the mouse, with the other three controls read afterwards (HANDOFF
0b). The artist names are a ratchet over every string the ten put on the
screen.

The bed-seam sweep is marked ``heavy``: it renders eleven minutes of audio.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QComboBox, QDialogButtonBox

from spacr.qt import night_themes as night
from spacr.qt import sound_synth as ss
from spacr.qt import theme as T
from spacr.qt.widgets import ambient

#: The four artists the request names, and the words a theme name or a
#: blurb would most plausibly borrow from them. None of these may appear
#: anywhere a user can read.
FORBIDDEN = ("worakls", "nto", "riviere", "rivière", "monk", "birrd",
             "hungry music", "afterlife", "anjunadeep")


@pytest.fixture
def store(monkeypatch, tmp_path):
    """A preference store of this test's own, never the user's."""
    from spacr.qt import preferences as prefs

    path = str(tmp_path / "prefs.ini")
    monkeypatch.setattr(prefs, "_settings",
                        lambda: QSettings(path, QSettings.IniFormat))
    monkeypatch.setenv(ss.CACHE_ENV, str(tmp_path / "sounds"))
    return prefs


def _theme_combo(dlg):
    """The Theme control, found the way a user finds it: by its entries."""
    for combo in dlg.findChildren(QComboBox):
        data = [combo.itemData(i) for i in range(combo.count())]
        if "glass" in data and "dark" in data:
            return combo
    raise AssertionError("no Theme combo in the dialog")


class TestThereAreTenOfThem:
    def test_ten_themes_with_ten_distinct_names(self):
        assert len(night.NIGHT_THEMES) == 10, "the request said ten"
        keys = list(night.NIGHT_THEME_KEYS)
        assert len(set(keys)) == 10
        labels = [t.label for t in night.NIGHT_THEMES.values()]
        assert len(set(labels)) == 10, f"two themes share a name: {labels}"
        blurbs = [t.description for t in night.NIGHT_THEMES.values()]
        assert len(set(blurbs)) == 10, "two themes share a description"

    def test_every_one_is_a_real_theme_the_application_offers(self):
        """A theme nothing can select is not a theme."""
        from spacr.qt.preferences import PALETTE_THEMES, theme_choices

        tokens = {token for _label, token in theme_choices()}
        for key in night.NIGHT_THEME_KEYS:
            assert key in T.THEMES, f"{key} is not in theme.THEMES"
            assert key in PALETTE_THEMES, f"{key} is not a palette theme"
            assert key in tokens, f"{key} is in no menu"

    def test_the_two_theme_lists_cannot_drift(self):
        """`preferences` restates `theme.THEMES` and must still agree."""
        from spacr.qt.preferences import PALETTE_THEMES

        assert tuple(T.THEMES) == tuple(PALETTE_THEMES)

    def test_no_theme_borrows_an_artist(self):
        """"WITHOUT using artist names or trademarks in theme names or UI"."""
        offenders = []
        for key, theme in night.NIGHT_THEMES.items():
            text = f"{key} {theme.label} {theme.description}".lower()
            offenders += [(key, word) for word in FORBIDDEN if word in text]
        for key in night.NIGHT_THEME_KEYS:
            sound = ss.SOUND_THEMES[key]
            text = f"{sound.label} {sound.description}".lower()
            offenders += [(key, word) for word in FORBIDDEN if word in text]
        assert not offenders, f"an artist reached the interface: {offenders}"

    def test_the_ratchet_would_notice(self):
        """The sweep above passes trivially if it reads nothing."""
        seen = " ".join(t.description for t in night.NIGHT_THEMES.values())
        assert len(seen) > 400, "the descriptions were not read"


class TestEveryPaletteStaysLegible:
    """The claim is "each must stay legible and pass the existing
    theme-sweep / contrast tests", so it is those tests that judge it."""

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_contrast_sweep_passes(self, key):
        assert T.contrast_failures(key) == []

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_panels_stay_off_the_page(self, key):
        assert T.page_separation_failures(key) == []

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_every_role_the_dark_palette_has(self, key):
        """A missing role renders a widget with an empty colour string,
        which Qt draws black on black."""
        assert set(T.palette_for(key)) == set(T.palette_for("dark"))

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_it_is_a_night_palette(self, key):
        """"light-on-dark night palettes are expected"."""
        palette = T.palette_for(key)
        page = T.relative_luminance(palette["page"])
        ink = T.relative_luminance(palette["fg"])
        assert page < 0.12, f"{key}'s page is not dark: {page:.3f}"
        assert ink > page, f"{key} is not light-on-dark"

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_status_hues_do_not_follow_the_theme(self, key):
        """A red that means "this run failed" is not a theme decision."""
        palette = T.palette_for(key)
        first = T.palette_for(night.NIGHT_THEME_KEYS[0])
        for role in ("success", "warning", "error"):
            assert palette[role] == first[role], (
                f"{key} moved {role} away from the rest of the family")

    def test_the_ten_are_ten_different_colours(self):
        """A family, not one palette with the hue nudged twice."""
        accents = {T.palette_for(k)["accent"] for k in night.NIGHT_THEME_KEYS}
        assert len(accents) == 10, "two themes share an accent"
        pages = {T.palette_for(k)["page"] for k in night.NIGHT_THEME_KEYS}
        assert len(pages) == 10, "two themes share a page colour"

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_stylesheet_builds(self, key):
        """A palette that cannot be turned into QSS is not a theme."""
        assert len(T.stylesheet(key)) > 1000


class TestEveryThemeHasItsOwnBackdrop:
    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_animation_is_one_that_exists(self, key):
        animation, palette = night.ambient_for(key)
        assert animation in ambient.AMBIENT_THEMES
        assert palette in ambient.PALETTE_SETS

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_that_animation_offers_that_palette(self, key):
        """Otherwise `get_ambient_palette` would validate it away on the
        next read and the theme would quietly paint in something else."""
        animation, palette = night.ambient_for(key)
        assert palette in ambient.palettes_for(animation)

    def test_no_two_themes_put_the_same_picture_on_the_screen(self):
        pairs = [night.ambient_for(k) for k in night.NIGHT_THEME_KEYS]
        assert len(set(pairs)) == 10, f"a backdrop is used twice: {pairs}"

    def test_the_backdrops_reuse_the_producers_that_were_there(self):
        """"reuse and parameterise the existing producers" -- so no new
        engine was added for this, and the count is the check."""
        assert len(ambient.AMBIENT_THEMES) == 7

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_backdrop_builds_and_paints(self, key, qapp):
        animation, palette = night.ambient_for(key)
        engine = ambient.make_engine(animation, palette,
                                     T.palette_for(key)["page"])
        assert engine is not None


class TestEveryThemeHasItsOwnSound:
    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_there_is_a_sound_set_of_the_same_name(self, key):
        assert key in ss.SOUND_THEMES
        assert night.sound_for(key) == key

    def test_the_ten_sets_are_not_presets_of_one_another(self):
        """Key, mode and tempo are what separate two records in this
        music, so no two of the ten may share all three."""
        signatures = [(t.tonic % 12, t.scale, t.tempo)
                      for k, t in ss.SOUND_THEMES.items()
                      if k in night.NIGHT_THEMES]
        assert len(set(signatures)) == 10, f"two sets match: {signatures}"

    def test_they_stay_inside_the_genre(self):
        """"melodic space house": the tempo range the music occupies, and
        a mode that is minor or modal rather than plain major."""
        for key in night.NIGHT_THEME_KEYS:
            theme = ss.SOUND_THEMES[key]
            assert 105.0 <= theme.tempo <= 130.0, (
                f"{key} at {theme.tempo} BPM is not house")
            assert theme.scale in (ss.NATURAL_MINOR, ss.DORIAN, ss.LYDIAN,
                                   ss.HARMONIC_MINOR)
            assert theme.pad_pump > 0.0, f"{key} has no pump"
            assert theme.delay_beats > 0.0, f"{key} has no delay"

    def test_the_four_modes_are_all_seven_notes(self):
        """`chord_tones` stacks thirds by index, so a scale of another
        length would build chords nobody chose."""
        for scale in (ss.NATURAL_MINOR, ss.DORIAN, ss.LYDIAN,
                      ss.HARMONIC_MINOR):
            assert len(scale) == 7
            assert scale[0] == 0
            assert list(scale) == sorted(scale)
            assert max(scale) < 12

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_progression_is_diatonic_and_reachable(self, key):
        theme = ss.SOUND_THEMES[key]
        assert theme.progression, f"{key} has no progression"
        for degree in theme.progression:
            assert 0 <= degree < len(theme.scale)

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_arpeggio_indexes_tones_that_exist(self, key):
        """"indexes into the five arpeggio tones of a chord"."""
        theme = ss.SOUND_THEMES[key]
        assert theme.arp_pattern
        assert all(0 <= step <= 4 for step in theme.arp_pattern)

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_a_click_and_a_hover_render_and_are_quiet(self, key):
        """The two sounds a user meets most often, on the samples."""
        theme = ss.SOUND_THEMES[key]
        click = ss.render(theme, "click-0").audio
        hover = ss.render(theme, "hover-0").audio
        click_peak = 20 * np.log10(max(1e-9, float(np.abs(click).max())))
        hover_peak = 20 * np.log10(max(1e-9, float(np.abs(hover).max())))
        assert -20.0 < click_peak <= -10.0, f"{key} click at {click_peak:.1f}"
        assert hover_peak < click_peak, f"{key}'s hover is not quieter"

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_failure_sounds_darker_than_success(self, key):
        """A user who hears "done" and finds a traceback stops trusting
        the sound, so the two must not be near neighbours."""
        theme = ss.SOUND_THEMES[key]
        done = ss.render(theme, "run_finished").audio.mean(axis=0)
        failed = ss.render(theme, "run_failed").audio.mean(axis=0)
        assert _centroid(failed) < _centroid(done), (
            f"{key}'s failure is brighter than its success")

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_every_set_caches_somewhere_of_its_own(self, key, tmp_path):
        """Two sets that fingerprint the same would answer for each
        other out of the cache."""
        theme = ss.SOUND_THEMES[key]
        others = {ss.theme_fingerprint(t)
                  for k, t in ss.SOUND_THEMES.items() if k != key}
        assert ss.theme_fingerprint(theme) not in others


def _centroid(mono: np.ndarray) -> float:
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / ss.SAMPLE_RATE)
    return float((power * freqs).sum() / power.sum())


class TestChoosingAThemeChangesLookAndSound:
    def test_the_setter_writes_all_three(self, store):
        store.set_theme_choice("nocturne")
        assert store.get_theme() == "nocturne"
        assert store.get_ambient_animation() == "drift"
        assert store.get_ambient_palette() == "midnight"
        assert store.get_sound_theme() == "nocturne"

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_it_writes_what_the_theme_says_for_all_ten(self, store, key):
        store.set_theme_choice(key)
        animation, palette = night.ambient_for(key)
        assert store.get_ambient_animation() == animation
        assert store.get_ambient_palette() == palette
        assert store.get_sound_theme() == night.sound_for(key)

    def test_it_never_switches_sound_on(self, store):
        """"all off-sound by default"."""
        assert store.get_sound_enabled() is False
        for key in night.NIGHT_THEME_KEYS:
            store.set_theme_choice(key)
            assert store.get_sound_enabled() is False, (
                f"{key} turned the sound on")

    def test_it_never_restarts_a_backdrop_the_user_turned_off(self, store):
        """"degrade to static ... under reduced motion or when disabled"."""
        store.set_ambient_animation("none")
        assert store.backdrop_is_switched_off() is True
        for key in night.NIGHT_THEME_KEYS:
            store.set_theme_choice(key)
            assert store.get_ambient_animation() == "none", (
                f"{key} started the backdrop moving again")
            assert store.get_ambient_enabled() is False

    def test_the_off_switch_alone_is_enough(self, store):
        """The separate on/off key, with an animation still named."""
        store.set_ambient_animation("blobs")
        store.set_ambient_enabled(False)
        assert store.backdrop_is_switched_off() is True
        store.set_theme_choice("pulsar")
        assert store.get_ambient_animation() == "blobs"

    def test_the_four_older_themes_bring_nothing(self, store):
        """Dark has never had an opinion about the backdrop."""
        store.set_theme_choice("nocturne")
        store.set_theme_choice("dark")
        assert store.get_ambient_animation() == "drift"
        assert store.get_sound_theme() == "nocturne"

    def test_a_night_theme_round_trips_through_the_token(self, store):
        for key in night.NIGHT_THEME_KEYS:
            store.set_theme_choice(key)
            assert store.get_theme_choice() == key
            assert store.resolve_effective_theme() == key


class TestTheDialogMovesTheOtherThreeControls:
    """HANDOFF 0b: press the control a user presses, then read what moved."""

    @pytest.fixture
    def dialog(self, store, qtbot, qt_theme_applied):
        from spacr.qt.preferences import PreferencesDialog

        dlg = PreferencesDialog()
        qtbot.addWidget(dlg)
        dlg.resize(900, 700)
        dlg.show()
        qtbot.waitExposed(dlg)
        return dlg

    def test_picking_a_night_theme_moves_animation_palette_and_sound(
            self, dialog):
        theme = _theme_combo(dialog)
        animation = dialog.findChild(QComboBox, "AmbientTheme")
        palette = dialog.findChild(QComboBox, "AmbientPalette")
        sound = dialog.findChild(QComboBox, "SoundTheme")
        theme.setCurrentIndex(theme.findData("aphelion"))
        assert animation.currentData() == "resonance"
        assert palette.currentData() == "midnight"
        assert sound.currentData() == "aphelion"

    def test_picking_dark_moves_nothing(self, dialog):
        theme = _theme_combo(dialog)
        animation = dialog.findChild(QComboBox, "AmbientTheme")
        sound = dialog.findChild(QComboBox, "SoundTheme")
        theme.setCurrentIndex(theme.findData("undertow"))
        was = (animation.currentData(), sound.currentData())
        theme.setCurrentIndex(theme.findData("dark"))
        assert (animation.currentData(), sound.currentData()) == was

    def test_the_dialog_leaves_none_alone(self, dialog):
        """A user with motion off keeps it off, whatever theme they pick."""
        animation = dialog.findChild(QComboBox, "AmbientTheme")
        sound = dialog.findChild(QComboBox, "SoundTheme")
        animation.setCurrentIndex(animation.findData("none"))
        theme = _theme_combo(dialog)
        theme.setCurrentIndex(theme.findData("cirrus"))
        assert animation.currentData() == "none"
        assert sound.currentData() == "cirrus", (
            "the sound set decides WHICH sounds, not whether any play")

    def test_save_writes_what_the_dialog_shows(self, dialog, store):
        theme = _theme_combo(dialog)
        theme.setCurrentIndex(theme.findData("vesper"))
        buttons = dialog.findChild(QDialogButtonBox)
        buttons.button(QDialogButtonBox.Save).click()
        assert store.get_theme() == "vesper"
        assert store.get_ambient_animation() == "aurora"
        assert store.get_ambient_palette() == "dusk"
        assert store.get_sound_theme() == "vesper"
        assert store.get_sound_enabled() is False

    def test_the_user_can_put_the_backdrop_back(self, dialog, store):
        """A preset, not an override: the theme suggests and the user
        decides, on the same two controls."""
        theme = _theme_combo(dialog)
        animation = dialog.findChild(QComboBox, "AmbientTheme")
        theme.setCurrentIndex(theme.findData("nocturne"))
        animation.setCurrentIndex(animation.findData("bokeh"))
        dialog.findChild(QDialogButtonBox).button(
            QDialogButtonBox.Save).click()
        assert store.get_theme() == "nocturne"
        assert store.get_ambient_animation() == "bokeh"


class TestTheSpaceoutDressingStaysCheapAndStaysTheSame:
    """Ten more themes made `enable_spaceout()` solve ten more dressings
    at launch, for a process that is in one theme. The solve is now lazy,
    and the only claim worth testing about a performance change is that it
    changed nothing else."""

    @pytest.fixture
    def dressed(self):
        T.enable_spaceout()
        yield
        T.disable_spaceout()

    def test_only_the_four_are_solved_when_the_dressing_goes_on(self,
                                                                dressed):
        assert set(T._DRESSED) == set(T.DRESSED_EAGERLY)
        for key in night.NIGHT_THEME_KEYS:
            assert key not in T._INK_BANDS

    def test_asking_for_a_palette_solves_that_theme_and_only_it(self,
                                                                dressed):
        T.palette_for("nocturne")
        assert "nocturne" in T._DRESSED
        assert "vesper" not in T._DRESSED

    def test_the_lazy_solve_is_the_eager_solve(self, dressed):
        """Dress lazily, then solve eagerly, and compare. A performance
        change that moves a colour is not a performance change."""
        lazy_palettes = {name: T.palette_for(name) for name in T.THEMES}
        lazy_bands = {k: dict(v) for k, v in T._INK_BANDS.items()}
        lazy_damping = {k: dict(v) for k, v in T._PAGE_DAMPING.items() if v}
        eager_bands = T._solve_ink_bands()
        eager_damping = {k: v for k, v in T._solve_page_damping().items()
                         if v}
        assert lazy_bands == eager_bands
        assert lazy_damping == eager_damping
        assert {name: T.palette_for(name)
                for name in T.THEMES} == lazy_palettes

    def test_every_theme_stays_readable_at_every_offset(self, dressed):
        """The dressing rotates the hue sixty ways; all ten must survive
        all sixty. This is what caught Vesper's accent at 4.49:1."""
        failures = []
        for key in night.NIGHT_THEME_KEYS:
            for drift in T._drift_grid():
                with T._dressed_at(drift):
                    failures += [f"{key} at {drift}: {line}"
                                 for line in T.contrast_failures(key)]
        assert failures == []

    def test_taking_the_dressing_off_restores_the_ten(self):
        plain = {name: T.palette_for(name) for name in night.NIGHT_THEME_KEYS}
        T.enable_spaceout()
        try:
            for name in night.NIGHT_THEME_KEYS:
                T.palette_for(name)
        finally:
            T.disable_spaceout()
        assert {name: T.palette_for(name)
                for name in night.NIGHT_THEME_KEYS} == plain


@pytest.mark.heavy
class TestTheMusicBeds:
    """Eleven minutes of audio, so this is the slow one."""

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_bed_loops_without_a_seam(self, key):
        """The same rule the reference set is judged by: the loop may not
        jump further at the join than the music jumps anywhere else."""
        audio = ss.render(ss.SOUND_THEMES[key], ss.BED).audio
        steps = np.abs(np.diff(audio, axis=1))
        seam = np.abs(audio[:, 0] - audio[:, -1])
        assert (seam <= np.percentile(steps, 99.5, axis=1)).all(), (
            f"{key}'s loop jumps by {seam} at the seam")

    @pytest.mark.parametrize("key", night.NIGHT_THEME_KEYS)
    def test_the_bed_is_quieter_than_anything_mastered(self, key):
        """It plays under somebody's work."""
        theme = ss.SOUND_THEMES[key]
        audio = ss.render(theme, ss.BED).audio
        assert ss.loudness_lufs(audio) == pytest.approx(theme.bed_lufs,
                                                        abs=0.5)
        peak = 20 * np.log10(max(1e-9, float(np.abs(audio).max())))
        assert peak <= theme.bed_peak_db + 0.5
