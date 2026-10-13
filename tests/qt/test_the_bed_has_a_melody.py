"""The bed has a tune in it, and the beat is something you can hear.

Asked 2026-09-19: "The theme should sound like melodic space house". Parts
A and B built the space house -- the key, the tempo, the progression, the
pads, the arpeggio, the kick, the shaker, the arrangement. Two things the
samples said were still missing, and both of them are the word MELODIC and
the word HOUSE doing their work:

  * THERE WAS NO MELODY. Every note in the bed came from the arpeggio,
    which is a texture: eight notes a bar, no rests, no shape. What this
    genre is remembered for is a phrase -- long notes, a rise, an answer,
    and a bar of silence before it comes round again.

  * AND THE SIDE-CHAIN WAS NOT AUDIBLE. The pads were ducked on the beat
    but the reverb they fed was not, so the tail filled the dip straight
    back in: 1.2 dB measured on the mix where the pads themselves were
    down 3.1. A house record's pump is not a subtlety, it is the pulse.

Nobody can listen to a test, so each of those becomes a number: where the
lead's notes fall and what they spell, how much of the 0.5-2.5 kHz band
belongs to the lead, and how deep the music bus dips on the beat.

THE BEAT IS 0.4918 SECONDS AND A HOP IS NOT. Folding an envelope on a
whole number of hops drifts half a beat across thirty-two bars and smears
the pump to nothing -- which is exactly how the first measurement of it
reported 1.2 dB and looked like a finding. Every phase here is computed
from the sample index and binned.

Nothing here needs Qt, and nothing is written outside the cache the
conftest points at ``tmp_path``.
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from spacr.qt import sound_synth as ss

SR = ss.SAMPLE_RATE
BEAT = ss.ORBIT.beat
BAR = 4.0 * BEAT

#: The motif as MIDI notes, worked out by hand from :data:`ss.LEAD_MOTIF`
#: rather than through :func:`ss.scale_note`, so the test would still fail
#: if the scale lookup itself went wrong. A minor, one octave above the
#: pads' tonic: E5 D5 C5, D5 C5 A4, E5 G5 A5, G5 E5, rest.
MOTIF_NOTES = [76, 74, 72, 74, 72, 69, 76, 79, 81, 79, 76]

#: Lead notes in the reference bed. Pinned because a note can only go
#: missing silently: the first version of the scheduler added the note
#: length in SECONDS each time, so a note landing on a bar line arrived a
#: float's breadth early, was priced at the PREVIOUS bar's level, and was
#: dropped where that bar was silent. It cost exactly one note of
#: seventy-two, and nothing but a count would have caught it.
LEAD_NOTES_IN_THE_BED = 72


@pytest.fixture(scope="module")
def bed():
    """The reference bed, rendered once for the whole module."""
    return ss.render(ss.ORBIT, ss.BED)


def _lead_notes(rendered):
    """Every ``(onset, note)`` the melody plays, in order."""
    return [(onset, note) for onset, note, part in rendered.notes
            if part == "lead"]


def _band(mono, low, high):
    """``mono`` through a fourth-order band-pass."""
    return sosfilt(butter(4, [low, high], btype="band", fs=SR, output="sos"),
                   mono)


def _rms_db(signal):
    return 20.0 * math.log10(max(float(np.sqrt((signal ** 2).mean())), 1e-12))


def _pump_db(audio, beat, low=300.0, high=3000.0, hop=240, bins=24):
    """How far the band dips and recovers across one beat, in dB.

    Every frame is placed by the phase of its own centre rather than by a
    whole number of frames per beat, which is the only way the dip survives
    thirty-two bars of averaging.
    """
    signal = _band(audio.mean(axis=0), low, high)
    frames = signal[:signal.size // hop * hop].reshape(-1, hop)
    energy = np.sqrt((frames ** 2).mean(axis=1))
    centres = (np.arange(frames.shape[0]) * hop + hop / 2.0) / SR
    index = np.minimum((np.mod(centres / beat, 1.0) * bins).astype(int),
                       bins - 1)
    profile = np.array([energy[index == b].mean() for b in range(bins)])
    loud = 20.0 * math.log10(max(float(profile.max()), 1e-12))
    quiet = 20.0 * math.log10(max(float(profile.min()), 1e-12))
    return loud - quiet, int(np.argmin(profile)) / bins


# ---------------------------------------------------------------------------
# There is a melody
# ---------------------------------------------------------------------------

def test_the_bed_has_a_melody_and_not_only_an_arpeggio(bed):
    """A part of its own, playing far fewer notes than the arpeggio."""
    parts = {part for _onset, _note, part in bed.notes}
    assert "lead" in parts, sorted(parts)
    lead = _lead_notes(bed)
    arp = [row for row in bed.notes if row[2] == "pluck"]
    assert len(lead) == LEAD_NOTES_IN_THE_BED
    assert len(lead) < len(arp) / 3, \
        f"{len(lead)} lead notes against {len(arp)} arpeggio notes is a " \
        "second arpeggio, not a tune"


def test_the_melody_is_the_motif_the_theme_asks_for(bed):
    """In the section where it plays in full, the lead spells the motif.

    Two phrases of it, back to back, which is what makes it a tune the
    second time round rather than a line that wanders.
    """
    lift_from, lift_to = 16.0 * BAR, 24.0 * BAR
    played = [note for onset, note in _lead_notes(bed)
              if lift_from - 1e-9 <= onset < lift_to]
    assert played == MOTIF_NOTES + MOTIF_NOTES, played


def test_the_melody_leaves_a_bar_of_silence_every_phrase(bed):
    """The rest in the motif is a real rest in the samples' schedule.

    A phrase that never stops is a pad with a rhythm. The last two beats
    of every four-bar phrase have no lead note starting in them.
    """
    for onset, _note in _lead_notes(bed):
        into = round(onset / BEAT * 2.0) / 2.0 % 16.0
        assert into <= 14.0 - 0.5, \
            f"a lead note starts {into} beats into its phrase, in the rest"


def test_every_lead_note_lands_on_the_beat_grid(bed):
    """Every onset is a whole number of half-beats from the loop's start.

    The motif is written in halves, so any onset that is not one is drift
    in the scheduler -- and drift is what silently dropped a note. See
    :data:`LEAD_NOTES_IN_THE_BED`.
    """
    for onset, note in _lead_notes(bed):
        halves = onset / (BEAT / 2.0)
        assert abs(halves - round(halves)) < 1e-9, \
            f"{note} at {onset!r}s is {halves:.9f} half-beats in"


# ---------------------------------------------------------------------------
# It can be heard, and it can be turned off
# ---------------------------------------------------------------------------

def test_the_melody_is_audible_over_the_arpeggio(bed):
    """The lift's 0.5-2.5 kHz band with the melody in it, against without.

    The one band where a lead in this register, an arpeggio an octave up
    and a pad an octave down all compete: if the melody cannot be measured
    there it cannot be heard there either.
    """
    silent = ss.render(dataclasses.replace(ss.ORBIT, lead_level=0.0), ss.BED)
    lift = slice(int(16 * BAR * SR), int(24 * BAR * SR))
    with_lead = _rms_db(_band(bed.audio.mean(axis=0)[lift], 500.0, 2500.0))
    without = _rms_db(_band(silent.audio.mean(axis=0)[lift], 500.0, 2500.0))
    assert with_lead - without > 2.5, \
        f"the melody adds {with_lead - without:.2f} dB, which is furniture"


def test_the_melody_arrives_and_leaves_with_the_arrangement(bed):
    """It is not simply on: it is away for the quiet section and full in
    the lift, which is what stops a tune under somebody's work from being
    the reason they turn the music off."""
    plan = ss.bed_plan(32)
    by_section = {}
    for row in plan:
        by_section.setdefault(row.section, []).append(row.lead)
    assert min(by_section["drift"]) == pytest.approx(0.0, abs=1e-9)
    assert max(by_section["lift"]) == pytest.approx(1.0, abs=1e-9)
    assert max(by_section["drift"]) < min(by_section["lift"])

    onsets = [onset for onset, _note in _lead_notes(bed)]
    quiet = [onset for onset in onsets if 4.0 * BAR <= onset < 8.0 * BAR]
    assert quiet == [], f"the melody plays through the quiet bars: {quiet}"


def test_a_theme_can_be_asked_for_without_a_melody():
    """Both ways of saying no leave the part out of the render, rather
    than playing it at nothing -- the same rule the drums already keep."""
    for quiet in (dataclasses.replace(ss.ORBIT, key="q1", bed_bars=4,
                                      reverb_seconds=0.6, lead_level=0.0),
                  dataclasses.replace(ss.ORBIT, key="q2", bed_bars=4,
                                      reverb_seconds=0.6, lead_pattern=())):
        assert _lead_notes(ss.render(quiet, ss.BED)) == []

    short = dataclasses.replace(ss.ORBIT, key="q3", bed_bars=4,
                                reverb_seconds=0.6)
    assert _lead_notes(ss.render(short, ss.BED)) != []


def test_a_changed_melody_is_a_different_cache_folder(tmp_path):
    """The fingerprint covers the new fields too, so a bed that has gained
    a tune can never be answered from the file that had none."""
    base = dataclasses.replace(ss.ORBIT, bed_bars=4)
    for changed in (dataclasses.replace(base, lead_level=0.2),
                    dataclasses.replace(base, lead_octave=2),
                    dataclasses.replace(base, lead_pattern=((0, 4.0),))):
        assert ss.theme_cache_dir(base, tmp_path) != \
            ss.theme_cache_dir(changed, tmp_path)
    assert ss.SYNTH_VERSION >= 3, "the melody retired version 2's files"


def test_the_melody_sustains_where_a_pluck_has_already_gone():
    """Why the lead is its own voice and not the arpeggio played slower.

    A quarter of a second later the lead is still all of what it was and
    the pluck is a fraction of it. That difference is the only reason a
    tune is audible over eight plucks a bar.

    MEASURED OVER QUARTER-SECOND WINDOWS, not over fifty milliseconds. The
    lead is three saws seven cents apart, which beat against each other
    with a period of about a third of a second; two short windows sample
    that beat rather than the note, and the first version of this test read
    the lead as decaying almost as fast as the pluck because of it.
    """
    freq = ss.midi_to_hz(76)
    rng = np.random.default_rng(0)
    lead = ss._lead_voice(ss.ORBIT, freq, 1.0, rng)
    pluck = ss._pluck(freq, 1.0, ss.ORBIT.pluck_brightness,
                      ss.ORBIT.pluck_decay)
    early = slice(0, int(0.25 * SR))
    late = slice(int(0.30 * SR), int(0.55 * SR))
    lead_held = _rms_db(lead[:, late]) - _rms_db(lead[:, early])
    pluck_held = _rms_db(pluck[:, late]) - _rms_db(pluck[:, early])
    assert lead_held > pluck_held + 5.0, \
        f"lead holds {lead_held:.1f} dB, pluck {pluck_held:.1f} dB"


# ---------------------------------------------------------------------------
# The pulse
# ---------------------------------------------------------------------------

def test_the_music_bus_ducks_on_the_beat(bed):
    """The pump, measured on the mix and not on the pads alone.

    The dip has to survive the reverb, because the reverb is most of what
    is heard: ducking the pads and then convolving a three-second room
    over the top put the energy straight back into the dip.
    """
    depth, trough = _pump_db(bed.audio, BEAT)
    assert depth > 3.5, f"the mix only dips {depth:.2f} dB on the beat"
    assert trough < 0.12 or trough > 0.88, \
        f"the dip is at phase {trough:.2f}, which is not the beat"


def test_the_duck_is_the_theme_s_own_number_and_can_be_switched_off():
    """The same bed with the pump at the theme's number and at zero.

    MEASURED WITH THE DRUMS OFF, because a kick on every beat and a shaker
    accented off it both put a bump at the same phase as the duck. The
    reference mix dips 4.1 dB on the beat; with the drums taken out it is
    3.8, and with the pump taken out as well it is 1.7 -- and that 1.7 is
    the arpeggio's own accent on the downbeat, which is music rather than
    side-chain. Anything that measures the pump through the drums is
    measuring the drums.

    BOTH RENDERS KEEP ``ORBIT``'s KEY, because the key is what seeds the
    theme's generator: the shaker's noise, the pads' saw phases and the
    reverb's impulse are all drawn from it, so two variants under two keys
    are two different pieces of music and the difference between them is
    not the field that was changed. Only ``pad_pump`` differs here.
    """
    dry = dataclasses.replace(ss.ORBIT, bed_bars=8, reverb_seconds=0.8,
                              kick_level=0.0, shaker_level=0.0)
    flat = dataclasses.replace(dry, pad_pump=0.0)
    deep_depth, _ = _pump_db(ss.render(dry, ss.BED).audio, BEAT)
    flat_depth, _ = _pump_db(ss.render(flat, ss.BED).audio, BEAT)
    assert flat_depth < 2.5, \
        f"a theme that asked for no pump dips {flat_depth:.2f} dB"
    assert deep_depth > flat_depth + 1.5, (flat_depth, deep_depth)


def test_no_theme_can_take_the_shaker_s_air():
    """A melody asked for as bright as the dial goes still leaves the top
    of the mix to the percussion.

    The guard, not the reference theme: ``Orbit``'s lead is warm enough
    that the wall at :data:`ss.LEAD_AIR_HZ` never binds on it. Part C's
    ten themes will each pick their own brightness, and the first melody
    built here -- before the wall -- took 2.1 dB of the shaker's air and
    turned ``test_the_drums_are_audible_when_they_are_asked_for`` red.
    """
    bright = dataclasses.replace(ss.ORBIT, bed_bars=8, reverb_seconds=0.8,
                                 lead_brightness=0.9)
    silent = dataclasses.replace(bright, shaker_level=0.0)
    with_shaker = _rms_db(_band(ss.render(bright, ss.BED).audio.mean(axis=0),
                                4000.0, 12000.0))
    without = _rms_db(_band(ss.render(silent, ss.BED).audio.mean(axis=0),
                            4000.0, 12000.0))
    assert with_shaker - without > 4.0, \
        f"the shaker only adds {with_shaker - without:.2f} dB over the melody"


def test_the_pump_s_depth_is_mostly_the_theme_s_number_and_not_the_routing():
    """Which of the two changes made the beat audible, measured apart.

    THE ROUTING WAS NOT THE WHOLE OF IT, and the note that said so was
    written before this was measured. Moving the side-chain from the pads
    to the whole music bus is one change; taking ``pad_pump`` from the 0.3
    the pads used to 0.6 is another, and it is the larger of the two. The
    ladder below is the reference bed with nothing altered but that one
    field, so a theme in part C that copies ``Orbit`` and then trims the
    depth back toward 0.3 can see here what it is giving up.

    The three renders share ``ORBIT``'s key, so the shaker, the pad phases
    and the reverb are identical in all three and the only difference is
    the depth.
    """
    base = dataclasses.replace(ss.ORBIT, bed_bars=8, reverb_seconds=0.8)
    depths = {}
    for pump in (0.0, 0.3, 0.6):
        theme = dataclasses.replace(base, pad_pump=pump)
        depths[pump], _ = _pump_db(ss.render(theme, ss.BED).audio, BEAT)
    assert depths[0.0] < depths[0.3] < depths[0.6], depths
    assert depths[0.6] - depths[0.3] > depths[0.3] - depths[0.0], (
        f"the depth doubling buys {depths[0.6] - depths[0.3]:.2f} dB and "
        f"part B's own depth buys {depths[0.3] - depths[0.0]:.2f} dB; the "
        "note claiming the routing did the work would then be right")
    assert depths[0.3] < 3.5 < depths[0.6], (
        f"at part B's depth the bus dips {depths[0.3]:.2f} dB, which "
        "test_the_music_bus_ducks_on_the_beat would have to be re-read")


def test_a_note_of_no_length_cannot_run_the_renderer_forever():
    """A pattern a theme hands in is walked forward by a bounded step.

    The scheduler advances by the length of the note it has just read, so
    a length of zero -- a typo, or a rhythm computed with integer division
    -- would never reach the end of the loop and would grow the note list
    until the process was killed, on the audio thread, with no error. Part
    C's ten themes each hand in their own pattern, which is exactly where
    such a length arrives. :data:`ss.LEAD_MIN_BEATS` is the floor, and it
    is the note's length as well as the step, so the note sounds as the
    shortest note rather than as nothing.
    """
    broken = dataclasses.replace(
        ss.ORBIT, bed_bars=4, reverb_seconds=0.8,
        lead_pattern=((0, 1.0), (2, 0.0), (4, -1.0), (ss.REST, 0.5),
                      (1, 4.0e9)))
    bed = ss.render(broken, ss.BED)
    lead = _lead_notes(bed)
    assert lead, "the melody went missing rather than being made finite"
    onsets = [onset for onset, _note in lead]
    assert onsets == sorted(onsets)
    assert len(lead) < 4.0 * 4 / ss.LEAD_MIN_BEATS
    assert 0.0 < ss.LEAD_MIN_BEATS <= 0.25
    assert bed.audio.shape[1] < 3 * int(4.0 * 4 * ss.ORBIT.beat * ss.SAMPLE_RATE), \
        "a note asked for in billions of beats was rendered at its own length"
