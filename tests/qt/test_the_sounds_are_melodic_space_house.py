"""The reference sound set sounds like what was asked for, measured.

Asked 2026-09-19: "The theme should sound like melodic space house", and the
accepted recommendation spelled the timbre out -- warm detuned-saw pads,
plucked arpeggio notes with a dotted-eighth delay, a soft sub, in a minor
key; click and hover as short in-key plucks; run-finished a rising arpeggio
resolving to the tonic with a pad swell; run-failed a falling minor figure;
tasteful, quiet, never jarring.

None of that can be listened to by a test, so each clause is turned into
something that can be measured on the samples themselves: the pitch a click
actually sounds, where an echo actually lands, whether the loop has a seam.
The note schedule each render returns is used for the shape of a figure,
and the AUDIO is checked against it wherever the two could disagree.

Nothing here needs Qt. Every file is written under ``tmp_path``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from spacr.qt import sound_synth as ss

SR = ss.SAMPLE_RATE
A_MINOR = {9, 11, 0, 2, 4, 5, 7}


@pytest.fixture(scope="module")
def rendered():
    """Every sound of the reference set, rendered once for the module."""
    names = (ss.sound_names("click") + ss.sound_names("hover")
             + ["run_finished", "run_failed", ss.BED])
    return {name: ss.render(ss.ORBIT, name) for name in names}


def _pitch(audio: np.ndarray, start: float = 0.0, seconds: float = 0.08) -> float:
    """The strongest frequency in a window, from a zero-padded FFT."""
    mono = audio.mean(axis=0)[int(start * SR):int((start + seconds) * SR)]
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size), 1 << 18))
    freqs = np.fft.rfftfreq(1 << 18, 1.0 / SR)
    spectrum[freqs < 60.0] = 0.0
    return float(freqs[int(spectrum.argmax())])


def _midi(freq: float) -> float:
    return 69.0 + 12.0 * math.log2(freq / 440.0)


def _centroid(audio: np.ndarray) -> float:
    mono = audio.mean(axis=0)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / SR)
    return float((power * freqs).sum() / power.sum())


def test_the_reference_set_is_a_minor_at_a_house_tempo():
    assert ss.ORBIT.tonic % 12 == 9, "the tonic is A"
    assert ss.ORBIT.scale == ss.NATURAL_MINOR
    assert 118 <= ss.ORBIT.tempo <= 126
    assert ss.ORBIT.delay_beats == pytest.approx(0.75), "a dotted eighth"
    assert ss.DEFAULT_THEME == ss.ORBIT.key


@pytest.mark.parametrize("name", ["click-0", "hover-0", "run_finished",
                                  "run_failed", "bed"])
def test_every_sound_is_finite_stereo_and_never_at_full_scale(rendered, name):
    audio = rendered[name].audio
    assert audio.ndim == 2 and audio.shape[0] == 2
    assert np.isfinite(audio).all()
    peak = float(np.abs(audio).max())
    assert 0.05 < peak <= 0.51, f"{name} peaks at {20 * math.log10(peak):.1f} dBFS"


def test_the_interface_sounds_are_quieter_than_the_run_sounds(rendered):
    """Tasteful and quiet: a hover under a click under a run ending."""
    def peak(name):
        return float(np.abs(rendered[name].audio).max())

    assert peak("hover-0") < peak("click-0") < peak("run_finished")
    assert 20 * math.log10(peak("hover-0")) <= -18.0
    assert 20 * math.log10(peak("click-0")) <= -10.0


@pytest.mark.parametrize("name", ["click-0", "click-1", "click-2", "click-3",
                                  "hover-0", "hover-1", "hover-2", "hover-3"])
def test_a_click_or_hover_sounds_a_note_of_the_key(rendered, name):
    """Measured on the audio: the loudest pitch is in tune and in A minor."""
    midi = _midi(_pitch(rendered[name].audio))
    assert abs(midi - round(midi)) < 0.2, f"{name} is {midi:.2f}, out of tune"
    assert round(midi) % 12 in A_MINOR, f"{name} sounds {round(midi)}"
    assert rendered[name].notes[0][1] == round(midi)


def test_successive_clicks_walk_a_phrase_rather_than_repeat(rendered):
    notes = [rendered[f"click-{i}"].notes[0][1] for i in range(4)]
    assert len(set(notes)) == 4


def test_clicks_and_hovers_are_short_plucks(rendered):
    """A pluck is loud at once and gone within a fraction of a second."""
    for name in ("click-0", "hover-0"):
        audio = np.abs(rendered[name].audio.mean(axis=0))
        attack = int(np.argmax(audio))
        assert attack < int(0.01 * SR), f"{name} does not start as a pluck"
        after = audio[int(0.25 * SR):int(0.30 * SR)].max()
        assert after < 0.1 * audio.max(), f"{name} rings too long"
    assert rendered["hover-0"].audio.shape[1] < rendered["click-0"].audio.shape[1]


def test_the_click_echo_lands_a_dotted_eighth_later(rendered):
    """The delay is the genre's dotted eighth, measured where it lands."""
    audio = np.abs(rendered["click-0"].audio.mean(axis=0))
    window = int(0.005 * SR)
    smooth = np.convolve(audio, np.ones(window) / window, mode="same")
    delay = ss.ORBIT.delay_beats * ss.ORBIT.beat
    lo, hi = int((delay - 0.03) * SR), int((delay + 0.03) * SR)
    echo_at = (lo + int(np.argmax(smooth[lo:hi]))) / SR
    before = smooth[int((delay - 0.08) * SR):int((delay - 0.04) * SR)].max()
    assert abs(echo_at - delay) < 0.02
    assert smooth[lo:hi].max() > 1.5 * before, "no echo stands out at the delay"


def test_the_echo_alternates_sides():
    """Ping-pong: the first repeat on one side, the second on the other."""
    impulse = np.zeros((2, SR))
    impulse[:, 0] = 1.0
    echoes = ss._ping_pong(impulse, 0.1, 0.5)
    d = int(0.1 * SR)
    first = np.abs(echoes[:, d:d + 200]).sum(axis=1)
    second = np.abs(echoes[:, 2 * d:2 * d + 200]).sum(axis=1)
    assert first[1] > 10 * first[0]
    assert second[0] > 10 * second[1]


def test_the_pads_are_detuned_saws():
    """Several voices a few cents apart, each a sawtooth: split lines at
    every harmonic, with the harmonics falling as 1/n."""
    rng = np.random.default_rng(0)
    seconds = 8.0
    pad = ss._supersaw(220.0, int(seconds * SR), 5, 16.0, 0.0, rng)
    mono = pad.mean(axis=0)
    spectrum = np.abs(np.fft.rfft(mono * np.hanning(mono.size)))
    freqs = np.fft.rfftfreq(mono.size, 1.0 / SR)

    band = (freqs > 216.0) & (freqs < 224.0)
    local = spectrum[band]
    peaks = [i for i in range(1, local.size - 1)
             if local[i] > local[i - 1] and local[i] > local[i + 1]
             and local[i] > 0.2 * local.max()]
    assert len(peaks) >= 4, "the voices are not detuned from each other"

    def around(f):
        return spectrum[(freqs > f - 6) & (freqs < f + 6)].max()

    ratio = around(440.0) / around(220.0)
    assert 0.35 < ratio < 0.65, f"second harmonic at {ratio:.2f}, not a saw"


def test_run_finished_rises_and_resolves_on_the_tonic(rendered):
    plucks = [n for _t, n, part in rendered["run_finished"].notes
              if part == "pluck"]
    climb, landing = plucks[:-2], plucks[-2:]
    assert climb == sorted(climb) and len(set(climb)) == len(climb)
    top = max(landing)
    assert top % 12 == 9, "it does not land on A"
    assert top > climb[-1], "the resolution is not above the climb"
    assert all(n % 12 in A_MINOR for n in plucks)

    arrival = [t for t, n, part in rendered["run_finished"].notes
               if part == "pluck" and n == top][0]
    heard = _midi(_pitch(rendered["run_finished"].audio, arrival + 0.02, 0.12))
    assert round(heard) % 12 == 9, f"the audio lands on {heard:.2f}, not A"


def test_run_finished_swells_a_pad_into_the_tonic_chord(rendered):
    notes = rendered["run_finished"].notes
    pads = [(t, n) for t, n, part in notes if part == "pad"]
    late = {n % 12 for t, n in pads if t > 0}
    assert {9, 0, 4} <= late, "the arriving pad is not A minor"
    audio = rendered["run_finished"].audio
    arrival = max(t for t, _n in pads)
    early = np.sqrt((audio[:, :int(0.2 * SR)] ** 2).mean())
    later = np.sqrt((audio[:, int((arrival + 0.3) * SR):
                           int((arrival + 0.6) * SR)] ** 2).mean())
    assert later > 1.5 * early, "no swell (measured +5.4 dB when written)"


def test_run_failed_falls_is_minor_and_is_darker(rendered):
    notes = rendered["run_failed"].notes
    plucks = [n for _t, n, part in notes if part == "pluck"]
    assert plucks == sorted(plucks, reverse=True)
    assert len(set(plucks)) == len(plucks) >= 3
    assert all(n % 12 in A_MINOR for n in plucks)
    pad = sorted(n % 12 for _t, n, part in notes if part == "pad")
    assert sorted({2, 5, 9, 0}) == pad, "the pad is not D minor seven"
    assert (_centroid(rendered["run_failed"].audio)
            < 0.8 * _centroid(rendered["run_finished"].audio))
    assert plucks[0] - plucks[-1] >= 5, "it barely falls"
    onsets = [t for t, _n, part in notes if part == "pluck"]
    up = [t for t, _n, part in rendered["run_finished"].notes if part == "pluck"]
    assert onsets[1] - onsets[0] > up[1] - up[0], "it is not slower"


def test_finished_and_failed_never_sound_alike(rendered):
    up = [n for _t, n, p in rendered["run_finished"].notes if p == "pluck"]
    down = [n for _t, n, p in rendered["run_failed"].notes if p == "pluck"]
    assert up[-1] > up[0] and down[-1] < down[0]


def test_the_bed_walks_the_progression_in_seventh_chords(rendered):
    notes = rendered[ss.BED].notes
    bar = 4 * ss.ORBIT.beat
    expected = [{9, 0, 4, 7}, {5, 9, 0, 4}, {0, 4, 7, 11}, {7, 11, 2, 5}]
    for b in range(ss.ORBIT.bed_bars):
        chord = {n % 12 for t, n, part in notes
                 if part == "pad" and abs(t - b * bar) < 1e-6}
        assert chord == expected[b % 4], f"bar {b + 1}"


def test_the_bed_arpeggio_stays_in_the_key_and_its_sub_is_the_root(rendered):
    notes = rendered[ss.BED].notes
    bar = 4 * ss.ORBIT.beat
    plucks = [n for _t, n, part in notes if part == "pluck"]
    assert len(plucks) == ss.ORBIT.bed_bars * 4 * ss.ORBIT.arp_division
    assert all(n % 12 in A_MINOR for n in plucks)
    roots = [9, 5, 0, 7]
    for t, n, part in notes:
        if part == "sub":
            b = int(round(t / bar))
            assert n % 12 == roots[b % 4]
            assert n <= ss.ORBIT.tonic - 17, "the sub is not a sub"


def test_the_sub_is_soft(rendered):
    """Felt, not a boom: present below 90 Hz, well under the pads."""
    mono = rendered[ss.BED].audio.mean(axis=0)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / SR)
    low = power[(freqs > 35) & (freqs < 90)].sum()
    mid = power[(freqs > 150) & (freqs < 1500)].sum()
    assert 0.02 < low / mid < 0.8


def test_the_bed_is_exactly_its_bars_long_and_loops_without_a_seam(rendered):
    audio = rendered[ss.BED].audio
    bars = ss.ORBIT.bed_bars * 4 * ss.ORBIT.beat
    assert audio.shape[1] == int(round(bars * SR))
    steps = np.abs(np.diff(audio, axis=1))
    seam = np.abs(audio[:, 0] - audio[:, -1])
    assert (seam <= np.percentile(steps, 99.5, axis=1)).all(), (
        f"the loop jumps by {seam} at the seam")
    assert rendered[ss.BED].loop is True


def test_the_bed_sits_under_the_other_sounds(rendered):
    bed = float(np.abs(rendered[ss.BED].audio).max())
    finished = float(np.abs(rendered["run_finished"].audio).max())
    assert bed < finished


def test_one_shots_start_and_end_in_silence(rendered):
    """No click at either end of a file, at any volume."""
    for name in ("click-0", "hover-0", "run_finished", "run_failed"):
        audio = rendered[name].audio
        assert np.abs(audio[:, 0]).max() < 1e-3
        assert np.abs(audio[:, -1]).max() < 1e-3


def test_a_theme_always_renders_the_same_samples():
    first = ss.render(ss.ORBIT, "run_failed").audio
    second = ss.render(ss.ORBIT, "run_failed").audio
    assert np.array_equal(first, second)


def test_an_unknown_sound_is_refused():
    with pytest.raises(KeyError):
        ss.render(ss.ORBIT, "applause")
    with pytest.raises(KeyError):
        ss.sound_names("applause")


def test_a_wav_round_trips_at_48_khz_16_bit(tmp_path, rendered):
    path = ss.write_wav(tmp_path / "x.wav", rendered["click-0"].audio)
    back, rate = ss.read_wav(path)
    assert rate == SR and back.shape == rendered["click-0"].audio.shape
    assert np.abs(back - rendered["click-0"].audio).max() < 1.0 / 16000
    assert not list(tmp_path.glob(".*.tmp")), "a temporary file was left"


class TestTheCache:
    """Rendered once per set per machine, and never answered stale."""

    def test_the_cache_honours_its_environment_variable(self, tmp_path,
                                                        monkeypatch):
        monkeypatch.setenv(ss.CACHE_ENV, str(tmp_path / "elsewhere"))
        assert ss.sound_cache_root() == tmp_path / "elsewhere"
        monkeypatch.delenv(ss.CACHE_ENV)
        assert ss.sound_cache_root().parts[-2:] == (".spacr", "sounds")

    def test_a_second_request_renders_nothing(self, tmp_path, monkeypatch):
        calls = []
        real = ss.render

        def counting(theme, name, sr=SR):
            calls.append(name)
            return real(theme, name, sr)

        monkeypatch.setattr(ss, "render", counting)
        names = ss.sound_names("hover")
        first = ss.ensure_rendered(ss.ORBIT, names, root=tmp_path)
        assert sorted(first) == sorted(names) and len(calls) == 4
        assert all(path.is_file() for path in first.values())
        second = ss.ensure_rendered(ss.ORBIT, names, root=tmp_path)
        assert second == first and len(calls) == 4

    def test_a_changed_theme_gets_a_new_folder_and_the_old_one_goes(
            self, tmp_path):
        import dataclasses

        old = ss.ensure_rendered(ss.ORBIT, ["hover-0"], root=tmp_path)
        unrelated = tmp_path / "orbit-notahexname"
        unrelated.mkdir()
        other_theme = tmp_path / "nebula-0123456789ab"
        other_theme.mkdir()
        brighter = dataclasses.replace(ss.ORBIT, pluck_brightness=0.7)
        assert (ss.theme_fingerprint(brighter)
                != ss.theme_fingerprint(ss.ORBIT))
        new = ss.ensure_rendered(brighter, ["hover-0"], root=tmp_path)
        assert new["hover-0"].parent != old["hover-0"].parent
        assert not old["hover-0"].parent.exists(), "the stale set stayed"
        assert unrelated.exists() and other_theme.exists()

    def test_the_synth_version_is_part_of_the_fingerprint(self, monkeypatch):
        before = ss.theme_fingerprint(ss.ORBIT)
        monkeypatch.setattr(ss, "SYNTH_VERSION", ss.SYNTH_VERSION + 1)
        assert ss.theme_fingerprint(ss.ORBIT) != before

    def test_a_render_can_be_abandoned_between_files(self, tmp_path):
        ready = ss.ensure_rendered(ss.ORBIT, ss.sound_names("click"),
                                   root=tmp_path, should_stop=lambda: True)
        assert ready == {}
