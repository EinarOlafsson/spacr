"""The music bed is a composed piece, and every claim about it is measured.

Asked 2026-09-19: "The theme should sound like melodic space house", and
part B of instruction 427 spells out what that has to mean for the bed --
thirty-two bars at 120 to 124 BPM from parameters, a minor-key progression,
an arpeggio, a pad, a sub, a soft four-on-the-floor kick and a shaker THAT
CAN BE TURNED DOWN TO NEAR-SILENT, filter movement, delay and reverb; and
then: "It must sound musical: check your render by analysing it (spectral
balance, no clipping, loudness around -18 LUFS-ish, loop seam click-free)."

Nobody can listen to a test, so each clause becomes something measurable on
the samples: where the kicks fall, how much energy the shaker puts above
4 kHz, what the gated BS.1770 loudness comes out at, how big the step at
the join is against the steps everywhere else.

Nothing here needs Qt, and nothing is written outside ``tmp_path``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from spacr.qt import sound_synth as ss

SR = ss.SAMPLE_RATE

#: A short theme for the tests that only care about the arrangement. The
#: reference bed is sixty-three seconds long and takes three and a half to
#: render; four bars and a short tail take a fifth of a second and answer
#: every question about WHICH PART PLAYS WHEN just as well.
SHORT = dataclasses.replace(ss.ORBIT, key="short", bed_bars=4,
                            reverb_seconds=0.6)


@pytest.fixture(scope="module")
def bed():
    """The reference bed, rendered once for the whole module."""
    return ss.render(ss.ORBIT, ss.BED)


def _band(audio: np.ndarray, low: float, high: float) -> float:
    """Power in a band, as a share of the whole."""
    mono = audio.mean(axis=0)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / SR)
    return float(power[(freqs >= low) & (freqs < high)].sum() / power.sum())


# ---------------------------------------------------------------------------
# The shape that was asked for
# ---------------------------------------------------------------------------

def test_the_bed_is_thirty_two_bars_at_a_house_tempo():
    assert ss.ORBIT.bed_bars == 32
    assert 120.0 <= ss.ORBIT.tempo <= 124.0
    assert ss.ORBIT.progression == (0, 5, 2, 6), "i - VI - III - VII"
    assert len(ss.bed_plan(ss.ORBIT.bed_bars)) == 32


def test_the_bed_is_exactly_its_bars_long(bed):
    seconds = ss.ORBIT.bed_bars * 4 * ss.ORBIT.beat
    assert bed.audio.shape[1] == int(round(seconds * SR))
    assert bed.loop is True


def test_the_arrangement_closes_where_it_opens():
    """The last bar's levels and the first bar's are the same levels.

    That is what makes the loop a circle rather than a line: every part
    ramps toward its section's level over the section's first half, and the
    first section ramps from the LAST one -- so bar one continues bar
    thirty-two instead of restarting it.
    """
    plan = ss.bed_plan(32)
    first, last = plan[0], plan[-1]
    for part in ("arp", "kick", "shaker", "sub"):
        assert getattr(first, part) == pytest.approx(getattr(last, part),
                                                     abs=0.02), part


def test_every_section_of_the_plan_is_reached_and_none_is_flat():
    plan = ss.bed_plan(32)
    assert [row.section for row in plan[:1] + plan[8:9] + plan[16:17]
            + plan[24:25]] == ["drift", "pulse", "lift", "return"]
    for part in ("arp", "kick", "shaker"):
        values = [getattr(row, part) for row in plan]
        assert max(values) - min(values) > 0.3, f"{part} never changes"


def test_a_shorter_loop_still_gets_every_section():
    """A theme may ask for fewer bars and still gets the whole shape."""
    plan = ss.bed_plan(4)
    assert [row.section for row in plan] == [name
                                             for name, *_ in ss.BED_SECTIONS]
    assert len(ss.bed_plan(1)) == len(ss.BED_SECTIONS)


# ---------------------------------------------------------------------------
# The drums
# ---------------------------------------------------------------------------

def test_the_kick_is_four_on_the_floor(bed):
    beat = ss.ORBIT.beat
    hits = sorted(t for t, _n, part in bed.notes if part == "kick")
    assert hits, "there is no kick at all"
    for at in hits:
        assert (at / beat) == pytest.approx(round(at / beat), abs=1e-6), \
            f"a kick at {at:.3f}s is not on a beat"


def test_the_shaker_is_on_the_eighths_and_the_offbeat_is_the_loud_one(bed):
    beat = ss.ORBIT.beat
    hits = sorted(t for t, _n, part in bed.notes if part == "shaker")
    assert hits
    for at in hits:
        assert (at / (beat / 2.0)) == pytest.approx(
            round(at / (beat / 2.0)), abs=1e-6)
    loud = ss._shaker(0.14, ss._rng(ss.ORBIT, "shaker"), SR)
    assert float(np.abs(loud).max()) == pytest.approx(1.0, abs=1e-6), \
        "the hit is normalised, so the accent in the bed is the only level"


def test_the_kick_is_felt_and_never_ticks():
    """A kick under a settings form is body and not click."""
    kick = ss._kick(0.62, SR)
    mono = kick.mean(axis=0)
    power = np.abs(np.fft.rfft(mono)) ** 2
    freqs = np.fft.rfftfreq(mono.size, 1.0 / SR)
    below = power[freqs < 160.0].sum() / power.sum()
    assert below > 0.9, f"only {below:.1%} of the kick is below 160 Hz"
    assert float(np.abs(mono[0])) < 1e-6 and float(np.abs(mono[-1])) < 1e-6


def test_the_drums_can_be_turned_down_to_nothing_and_the_bed_still_plays():
    """The request's own words, measured: the parts leave the render.

    Not merely quieter -- the kick and the shaker are not synthesized at
    all below :data:`spacr.qt.sound_synth.PART_FLOOR`, so a theme that
    wants the pad-and-arpeggio piece pays nothing for drums it does not
    want and the loudness normaliser makes no room for them.
    """
    silent = dataclasses.replace(SHORT, key="silent", kick_level=0.0,
                                 shaker_level=0.0)
    quiet = ss.render(silent, ss.BED)
    assert not [row for row in quiet.notes if row[2] in ("kick", "shaker")]
    assert quiet.audio.shape == ss.render(SHORT, ss.BED).audio.shape
    assert float(np.abs(quiet.audio).max()) > 0.05, "the bed went silent too"


def test_the_drums_are_audible_when_they_are_asked_for(bed):
    """The other half of the switch, and the one that is easy to get wrong.

    A shaker mixed too low is a setting that does nothing, which is worse
    than no shaker: measured as the energy above 4 kHz, where the pads and
    the sub have nothing at all.
    """
    drumless = ss.render(dataclasses.replace(ss.ORBIT, key="drumless",
                                             kick_level=0.0,
                                             shaker_level=0.0), ss.BED)
    with_drums = _band(bed.audio, 4000.0, 12000.0)
    without = _band(drumless.audio, 4000.0, 12000.0)
    assert with_drums > 4.0 * without, (
        f"the shaker adds only {with_drums / without:.1f}x the air")


# ---------------------------------------------------------------------------
# "It must sound musical", measured
# ---------------------------------------------------------------------------

def test_the_loudness_measurement_agrees_with_the_standard():
    """BS.1770's own calibration tone: 1 kHz at full scale is -3.01 LKFS.

    The measurement has to be right before anything measured with it is
    worth reading, and this is the case the standard states outright. It
    also pins the one thing that is easy to get wrong by arithmetic: the
    -0.691 offset does NOT show up at 1 kHz, because the K-weighting
    filter has +0.691 dB of gain there by construction and the two cancel.
    Writing this test from "K-weighting is flat at 1 kHz" gave an expected
    value 0.7 dB out and the code was right.
    """
    t = np.arange(int(3.0 * SR)) / SR
    tone = np.sin(2.0 * math.pi * 1000.0 * t)
    one_channel = np.vstack([tone, np.zeros_like(tone)])
    assert ss.loudness_lufs(one_channel, SR) == pytest.approx(-3.01, abs=0.1)

    quiet = np.vstack([0.1 * tone, 0.1 * tone])
    assert ss.loudness_lufs(quiet, SR) == pytest.approx(-20.0, abs=0.1)
    assert ss.loudness_lufs(np.zeros((2, SR)), SR) == -math.inf


def test_a_peak_is_not_a_loudness():
    """The reason the bed is levelled by loudness and not by its peak."""
    t = np.arange(int(2.0 * SR)) / SR
    steady = np.vstack([0.5 * np.sin(2.0 * math.pi * 440.0 * t)] * 2)
    spiky = np.zeros((2, t.size))
    spiky[:, ::SR // 4] = 0.5
    assert float(np.abs(steady).max()) == pytest.approx(
        float(np.abs(spiky).max()), abs=1e-9)
    assert ss.loudness_lufs(steady, SR) > ss.loudness_lufs(spiky, SR) + 10.0


def test_the_bed_lands_where_the_request_asked_for(bed):
    """"loudness around -18 LUFS-ish", and no clipping anywhere."""
    measured = ss.loudness_lufs(bed.audio, SR)
    assert ss.ORBIT.bed_lufs == -18.0
    assert measured == pytest.approx(-18.0, abs=1.0), f"{measured:.2f} LUFS"
    peak = 20.0 * math.log10(float(np.abs(bed.audio).max()))
    assert peak <= ss.ORBIT.bed_peak_db + 0.1, f"{peak:.2f} dBFS"
    assert not np.any(np.abs(bed.audio) >= 0.999), "the bed clips"


def test_the_bed_is_quieter_than_the_sound_it_plays_under(bed):
    """It plays under somebody's work, so it is under the run sounds too."""
    finished = ss.render(ss.ORBIT, "run_finished")
    assert float(np.abs(bed.audio).max()) < float(np.abs(finished.audio).max())
    assert ss.loudness_lufs(bed.audio, SR) < ss.loudness_lufs(
        finished.audio, SR)


def test_the_spectral_balance_is_a_bed_and_not_a_rumble(bed):
    """Every octave from the sub to the air carries something, and the
    bottom does not swamp the middle -- which it does the moment a kick
    and a sub are added without anybody looking."""
    assert 0.05 < _band(bed.audio, 20.0, 120.0) < 0.45
    assert _band(bed.audio, 250.0, 1000.0) > 0.3
    assert _band(bed.audio, 1000.0, 4000.0) > 0.001
    low = _band(bed.audio, 35.0, 90.0)
    mid = _band(bed.audio, 150.0, 1500.0)
    assert 0.02 < low / mid < 0.8, f"low/mid is {low / mid:.2f}"


def test_the_loop_seam_is_click_free(bed):
    """The step at the join, against the steps everywhere else.

    A seam is audible as a click when the join is a bigger step than the
    music itself ever takes; measured against the 99.5th percentile of
    every sample-to-sample step in the loop, which is what the loudest
    transient in it actually does.
    """
    audio = bed.audio
    steps = np.abs(np.diff(audio, axis=1))
    seam = np.abs(audio[:, 0] - audio[:, -1])
    ceiling = np.percentile(steps, 99.5, axis=1)
    assert (seam <= ceiling).all(), f"the loop jumps by {seam} at the seam"
    assert float(np.abs(audio[:, :64]).max()) > 1e-3, \
        "a loop must not fade in, or every repeat dips"


def test_the_filter_breathes_over_the_whole_loop_and_comes_back(bed):
    """A sweep that did not return would be a step at the seam.

    Measured as the spectral centroid over the loop: it has to MOVE, and
    it has to be back where it started by the end.
    """
    mono = bed.audio.mean(axis=0)
    window = int(2.0 * SR)
    centres = []
    for start in range(0, mono.size - window, window):
        block = mono[start:start + window]
        power = np.abs(np.fft.rfft(block)) ** 2
        freqs = np.fft.rfftfreq(block.size, 1.0 / SR)
        centres.append(float((power * freqs).sum() / power.sum()))
    assert max(centres) > 1.25 * min(centres), "nothing moves"
    assert abs(centres[0] - centres[-1]) < 0.45 * max(centres), \
        "the loop does not end where it began"


def test_the_arrangement_is_audible_and_not_only_a_table(bed):
    """The sections differ in the SAMPLES, which is the only place it
    counts. The quiet section is quieter than the loud one by a margin a
    listener would notice."""
    bar = 4 * ss.ORBIT.beat
    spans = {}
    for index, (name, *_rest) in enumerate(ss.BED_SECTIONS):
        start, stop = int(index * 8 * bar * SR), int((index + 1) * 8 * bar * SR)
        spans[name] = ss.loudness_lufs(bed.audio[:, start:stop], SR)
    assert spans["lift"] > spans["drift"] + 0.8, spans
    assert spans["pulse"] > spans["drift"], spans


def test_the_same_theme_always_renders_the_same_bed():
    first = ss.render(SHORT, ss.BED).audio
    second = ss.render(SHORT, ss.BED).audio
    assert np.array_equal(first, second)


def test_a_changed_drum_level_is_a_different_cache_folder(tmp_path):
    """The fingerprint covers every field, so a theme that sounds
    different can never be answered from the old file."""
    loud = dataclasses.replace(SHORT, kick_level=0.9)
    assert ss.theme_cache_dir(SHORT, tmp_path) != \
        ss.theme_cache_dir(loud, tmp_path)
    assert ss.SYNTH_VERSION >= 2, "the 32-bar bed retired version 1's files"
