"""Synthesize spaCR's interface sounds and music bed with numpy.

Every sound spaCR can play is made here, in code, from a
:class:`SoundTheme`: nothing is recorded, sampled or downloaded, so no
sound carries a licence question. A theme is a key, a scale, a tempo, a
chord progression and a handful of timbre numbers; the same theme always
renders the same samples, because every random choice is drawn from a
generator seeded by the theme itself.

The reference theme, :data:`ORBIT`, is melodic space house in A minor:
warm detuned-saw pads, plucked arpeggio notes with a dotted-eighth
ping-pong delay, a soft sub, a long dark reverb, and — in the music bed
alone, never in an interface sound — a quiet four-on-the-floor kick and
an eighth-note shaker that come and go with the arrangement. Clicks and hovers
are short plucks on notes of the scale, a finished run is an arpeggio
rising through the seventh chord into the tonic under a pad swell, and a
failed run is a slower falling figure over the minor fourth.

This module is Qt-free on purpose. It writes 16-bit PCM WAV files, which
is all :class:`PySide6.QtMultimedia.QSoundEffect` needs, into a cache
under the spaCR data directory (:func:`sound_cache_root`), one folder per
theme and fingerprint so a changed theme or a new synthesiser version can
never be answered from a stale file. :mod:`spacr.qt.sound` renders
through :func:`ensure_rendered` on its own audio thread and never on the
GUI thread.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import shutil
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import (Callable, Dict, Iterable, List, NamedTuple, Optional,
                    Sequence, Tuple)

import numpy as np

__all__ = [
    "BED",
    "BED_SECTIONS",
    "BedBar",
    "CACHE_ENV",
    "CLICK_VARIANTS",
    "DEFAULT_THEME",
    "DORIAN",
    "EVENTS",
    "FEEDBACK_EVENTS",
    "HARMONIC_MINOR",
    "HOVER_VARIANTS",
    "LEAD_AIR_HZ",
    "LEAD_MOTIF",
    "LYDIAN",
    "NATURAL_MINOR",
    "NIGHT_SETS",
    "ORBIT",
    "REST",
    "Rendered",
    "SAMPLE_RATE",
    "SOUND_THEMES",
    "SYNTH_VERSION",
    "SoundTheme",
    "bed_plan",
    "chord_tones",
    "ensure_rendered",
    "loudness_lufs",
    "midi_to_hz",
    "read_wav",
    "render",
    "scale_note",
    "sound_cache_root",
    "sound_names",
    "theme_cache_dir",
    "theme_fingerprint",
    "write_wav",
]

#: Samples per second of every file written. 48 kHz is what PipeWire,
#: PulseAudio, CoreAudio and WASAPI run at by default, so the audio device
#: plays the file without resampling it.
SAMPLE_RATE = 48000

#: Raised whenever a change here alters what a theme sounds like. It is part
#: of every cache fingerprint, so raising it retires every cached file.
#:
#: Version 3 adds the melody -- :data:`LEAD_MOTIF` on its own sustaining
#: voice, in and out with the arrangement -- moves the side-chain from the
#: pads alone to the whole music bus ahead of the reverb, gives that
#: side-chain a finite fall (:data:`PUMP_ATTACK`) and takes the reference
#: theme's ``pad_pump`` from 0.3 to 0.6. Version 2 was the 32-bar
#: arrangement with sections, a soft four-on-the-floor kick, a shaker and
#: a loudness-normalised master; version 1 was an eight-bar phrase with no
#: drums.
SYNTH_VERSION = 3

#: Environment variable that moves the cache. Tests and probes point it at
#: a scratch folder so they never write into a real ``~/.spacr``.
CACHE_ENV = "SPACR_SOUND_CACHE"

#: The natural minor (Aeolian) scale, in semitones above the tonic.
NATURAL_MINOR: Tuple[int, ...] = (0, 2, 3, 5, 7, 8, 10)

#: Minor with a raised sixth. Its fourth degree carries a MAJOR triad,
#: which is the lift the whole genre leans on: the same minor room with
#: one window open. :data:`HALCYON` and :data:`MERIDIAN` use it.
DORIAN: Tuple[int, ...] = (0, 2, 3, 5, 7, 9, 10)

#: Major with a raised fourth. Every seventh chord on it is major or
#: dominant, so nothing in it settles; :data:`SOLSTICE` uses it for the
#: one theme of the ten that never quite lands.
LYDIAN: Tuple[int, ...] = (0, 2, 4, 6, 7, 9, 11)

#: Minor with a raised seventh, so the fifth degree carries a DOMINANT
#: seventh and the tonic a minor-major seventh. It is the tension in
#: cinematic melodic techno, and :data:`APHELION` is built on it.
HARMONIC_MINOR: Tuple[int, ...] = (0, 2, 3, 5, 7, 8, 11)

#: Where the melody stops, in hertz. ABOVE THIS THE MIX BELONGS TO THE
#: SHAKER, and the eighth-order wall is a guard on every theme rather than
#: a tone control on this one.
#:
#: A sustained voice playing all the way up competes with the one part
#: whose whole job is the top of the spectrum, and it wins, because it is
#: sustained and the shaker is not. Measured with the theme key held fixed,
#: because the shaker IS noise drawn from the key's own generator and a
#: comparison across two keys compares two different shakers: the shaker
#: adds 5.36 dB above 4 kHz with no melody at all, and the first melody
#: built here left it 3.23 dB, which turned part B's
#: ``test_the_drums_are_audible_when_they_are_asked_for`` red. Warming the
#: melody to ``lead_brightness`` 0.18 gives the shaker 5.44 dB back, and
#: this wall is what stops a theme from asking for the brightness that
#: takes it away again.
LEAD_AIR_HZ = 3000.0

#: A rest in a lead pattern, in the place a scale degree would be. It is
#: far outside any scale so a pattern that reaches it by arithmetic rather
#: than on purpose cannot be mistaken for a note.
REST = -1000

#: The reference lead melody: twelve notes over four bars, in scale
#: degrees of the theme's key and beats.
#:
#: THIS IS THE LINE SOMEBODY HUMS. Pads, an arpeggio, a kick and a shaker
#: are a texture; what makes the genre MELODIC space house is a phrase with
#: a shape, and one that leaves room -- long notes, and a whole bar of rest
#: before it comes round again. The phrase falls E-D-C, falls again D-C-A,
#: climbs E-G-A, and answers G-E; over i-VI-III-VII every one of those is a
#: chord tone or the ninth, so it sings over the progression instead of
#: being led by it, which is how the melody of this music is written.
LEAD_MOTIF: Tuple[Tuple[int, float], ...] = (
    (4, 1.5), (3, 0.5), (2, 2.0),
    (3, 1.5), (2, 0.5), (0, 2.0),
    (4, 1.0), (6, 1.0), (7, 2.0),
    (6, 1.5), (4, 0.5), (REST, 2.0),
)

#: How many different in-key plucks a click, and a hover, rotates through.
#: Successive clicks walk a short phrase rather than repeating one note.
CLICK_VARIANTS = 4
HOVER_VARIANTS = 4

#: The events a sound set answers, in the order Preferences lists them.
FEEDBACK_EVENTS: Tuple[str, ...] = ("click", "hover", "run_finished",
                                    "run_failed")
#: The looping music bed. Rendered only when somebody asks for it.
BED = "bed"
EVENTS: Tuple[str, ...] = FEEDBACK_EVENTS + (BED,)


@dataclass(frozen=True)
class SoundTheme:
    """Everything that decides how one sound set sounds.

    :param key: stable identifier, used in the cache path and the stored
        preference.
    :param label: name shown in Preferences.
    :param description: one sentence shown as the choice's explanation.
    :param tonic: MIDI note of the key's tonic, in the octave the pads sit
        in (57 is A3).
    :param scale: semitones above the tonic of each scale degree.
    :param tempo: beats per minute; sets the arpeggio rate and the delay.
    :param progression: one scale degree per bar of the music bed, each
        the root of a diatonic seventh chord (0-based: 0 is the tonic).
    :param pad_voices: detuned sawtooth oscillators per pad note.
    :param pad_detune_cents: spread of those oscillators, either side.
    :param pad_cutoff_hz: low-pass cutoff of the pad at rest; it breathes
        up to roughly twice this.
    :param pad_level: pad level in the mix.
    :param pad_pump: depth of the dip on every beat of the music bus --
        pads, arpeggio, lead and their reverb, everything but the kick and
        half of the sub. The side-chain of house music, with or without a
        kick drum to cause it (0 to 1). THIS NUMBER, NOT THE ROUTING, IS
        MOST OF HOW DEEP THE PUMP SOUNDS: measured on the reference bed,
        300-3000 Hz, the mix dips 1.81 dB with this at 0, 2.46 dB at the
        0.3 the pads alone used, and 4.59 dB at the 0.6 the reference now
        asks for. A theme that sets it back toward 0.3 is asking for a
        pump shallower than
        ``test_the_music_bus_ducks_on_the_beat`` accepts.
    :param pluck_brightness: how slowly the pluck's harmonics fall away
        (0 to 1); higher is brighter.
    :param pluck_decay: time constant of the pluck's fundamental, seconds.
    :param pluck_level: arpeggio level in the mix.
    :param lead_pattern: the melody, as ``(scale degree, beats)`` pairs
        read in order and repeated for the length of the bed. A degree of
        :data:`REST` is a silence of that many beats. An empty pattern, or
        a ``lead_level`` at or below :data:`PART_FLOOR`, leaves the melody
        out of the render entirely.
    :param lead_level: melody level in the mix.
    :param lead_octave: whole octaves above the tonic the melody is played
        in.
    :param lead_attack: seconds the melody's notes take to speak. Slow
        enough that it sings rather than plucks.
    :param lead_brightness: how far above each note's own pitch the
        melody's filter opens; higher is brighter. Capped so that no theme
        can turn the melody into a saw lead that takes the air away from
        the shaker -- this genre's lead is warm, and the top of the mix
        belongs to the percussion.
    :param arp_pattern: indexes into the five arpeggio tones of a chord
        (four chord tones and the root an octave up), one per step.
    :param arp_division: arpeggio steps per beat (2 is eighth notes).
    :param delay_beats: echo spacing in beats; 0.75 is the dotted eighth.
    :param delay_feedback: level of each echo relative to the one before.
    :param delay_mix: how much of the echo reaches the output.
    :param sub_level: level of the sine sub under each chord.
    :param space: reverb send (0 to 1).
    :param reverb_seconds: time for the reverb tail to fall 60 dB.
    :param width: stereo spread of the pad voices (0 is mono).
    :param bed_bars: bars in one loop of the music bed. Rounded up to a
        whole number of :data:`BED_SECTIONS` sections by :func:`bed_plan`,
        so the arrangement always closes where it opened.
    :param kick_level: level of the four-on-the-floor kick in the bed, 0 to
        1. ``0.0`` leaves the kick out of the render entirely.
    :param shaker_level: level of the eighth-note shaker, 0 to 1.
        ``0.0`` leaves it out entirely.
    :param bed_lufs: programme loudness the finished bed is normalised to,
        in LUFS (:func:`loudness_lufs`). Quieter than anything mastered for
        release, because this plays under somebody's work.
    :param bed_peak_db: the bed's peak ceiling in dBFS. Reached with a
        memoryless soft knee, which is what lets the ceiling be applied
        after the loop is folded without putting a seam back in.
    :param seed: seeds every random choice, so a theme always renders the
        same samples.
    """

    key: str
    label: str
    description: str
    tonic: int = 57
    scale: Tuple[int, ...] = NATURAL_MINOR
    tempo: float = 122.0
    progression: Tuple[int, ...] = (0, 5, 2, 6)
    pad_voices: int = 5
    pad_detune_cents: float = 16.0
    pad_cutoff_hz: float = 1100.0
    pad_level: float = 0.5
    pad_pump: float = 0.6
    pluck_brightness: float = 0.62
    pluck_decay: float = 0.32
    pluck_level: float = 0.5
    lead_pattern: Tuple[Tuple[int, float], ...] = LEAD_MOTIF
    lead_level: float = 0.68
    lead_octave: int = 1
    lead_attack: float = 0.045
    lead_brightness: float = 0.18
    arp_pattern: Tuple[int, ...] = (0, 2, 1, 3, 2, 4, 3, 1)
    arp_division: int = 2
    delay_beats: float = 0.75
    delay_feedback: float = 0.42
    delay_mix: float = 0.38
    sub_level: float = 0.45
    space: float = 0.32
    reverb_seconds: float = 3.2
    width: float = 0.7
    bed_bars: int = 32
    kick_level: float = 0.5
    shaker_level: float = 0.34
    bed_lufs: float = -18.0
    bed_peak_db: float = -9.0
    seed: int = 427

    @property
    def beat(self) -> float:
        """Seconds per beat at this theme's tempo."""
        return 60.0 / float(self.tempo)


#: The reference sound set: melodic space house in A minor at 122 BPM.
#: The bed walks i - VI - III - VII (Am7, Fmaj7, Cmaj7, G7), the
#: progression most of the genre is built on.
ORBIT = SoundTheme(
    key="orbit",
    label="Orbit",
    description=("Melodic space house in A minor: a singing lead over warm "
                 "detuned pads, plucked arpeggios with a dotted-eighth "
                 "echo, a soft sub, and a quiet four-on-the-floor kick and "
                 "shaker that come and go with the arrangement."),
)


#: One section of the music bed's arrangement, as fractions of the parts
#: available: ``(name, bars, arp, kick, shaker, sub, lead)``.
#:
#: THE ARRANGEMENT IS WHAT MAKES A LOOP BEARABLE. Eight bars of the same
#: four chords with everything playing is a phrase; thirty-two bars with a
#: shape is a piece, and a piece can run for an hour behind somebody's work
#: without being noticed, which is the whole requirement.
#:
#: AND THE SEAM IS PLACED, NOT PATCHED. The loop opens and closes on the
#: quietest section -- pads and sub, no drums, the arpeggio barely in -- so
#: the join lands where the music has least to give away. The tails are
#: carried over it by :func:`_fold` and the level is set by a memoryless
#: curve, so nothing puts a step back in. See
#: ``test_the_bed_is_exactly_its_bars_long_and_loops_without_a_seam``.
#: AND THE MELODY IS THE SHAPE, not another part that is simply on. It
#: fades out across the opening section, is away for four bars, comes back
#: under the drums and is full only in the lift -- so the loop has a tune
#: that arrives rather than a tune that never stops. A melody playing for
#: sixty-three seconds out of every sixty-three is the one thing that would
#: make this unbearable to work over.
BED_SECTIONS: Tuple[
    Tuple[str, int, float, float, float, float, float], ...] = (
    ("drift", 8, 0.45, 0.0, 0.0, 0.85, 0.0),
    ("pulse", 8, 0.85, 0.85, 0.80, 1.00, 0.60),
    ("lift", 8, 1.00, 1.00, 1.00, 1.00, 1.00),
    ("return", 8, 0.62, 0.15, 0.20, 0.90, 0.28),
)

#: THE TEN NIGHT SOUND SETS, one per theme in
#: :data:`spacr.qt.night_themes.NIGHT_THEMES` and keyed the same, so
#: choosing a theme in Preferences moves the sound with the colours
#: without a lookup table between them.
#:
#: They are all the same genre and none of them is the same piece. What
#: separates them is what separates two records in it: the key and the
#: mode (four modes across the ten), the tempo (108 to 128 BPM, the range
#: the music actually occupies), how far open the pad's filter sits, how
#: hard the pad is pumped on every beat, how bright and how long the
#: plucks are, how fast the arpeggio runs and in what shape, and how much
#: room is around all of it.
#:
#: THE TWO ENDS ARE THE ARGUMENT. :data:`NOCTURNE` is 108 BPM, quarter-note
#: arpeggios, a pluck that decays for six tenths of a second like a
#: damped piano string, a 4.8 s tail and almost no drum. :data:`PULSAR` is
#: 128 BPM, sixteenth notes, a pluck a third as long, the deepest pump of
#: the ten and the shortest reverb. Neither is a preset of the other.
#:
#: ONE SEED WAS CHOSEN BY MEASUREMENT RATHER THAN PICKED. Lantern's bed is
#: the smoothest of the eleven -- a pad filtered to 900 Hz makes small
#: sample-to-sample steps -- so the seam rule, which asks that the loop
#: jump no further at the join than the music jumps anywhere else, is
#: tightest here. At seed 4271 the right channel jumped 0.0388 against a
#: 99.5th-percentile step of 0.0376 and failed it. 4291 renders the same
#: theme with different noise in the shaker and the reverb tail and lands
#: at 0.0112 against 0.0332. Nothing else about Lantern was changed, and
#: the other nine night seeds run 4272 to 4280 in menu order because they
#: never needed to be anything else. :data:`ORBIT`, which predates the
#: ten, keeps its own seed of 427.
LANTERN = SoundTheme(
    key="lantern",
    label="Lantern",
    description=("G minor at 118 BPM: a close, warm room — the pad's "
                 "filter barely open, wide detuning, plucks that decay "
                 "slowly, and a short tail."),
    tonic=55,
    tempo=118.0,
    progression=(0, 5, 3, 4),
    pad_detune_cents=20.0,
    pad_cutoff_hz=900.0,
    pluck_brightness=0.55,
    pluck_decay=0.40,
    arp_pattern=(0, 2, 1, 3, 1, 4, 2, 0),
    space=0.28,
    reverb_seconds=2.6,
    width=0.65,
    kick_level=0.50,
    shaker_level=0.30,
    seed=4291,
)

HALCYON = SoundTheme(
    key="halcyon",
    label="Halcyon",
    description=("F Dorian at 120 BPM: minor with the sixth raised, so "
                 "the fourth chord of the bed arrives major and the room "
                 "opens for a bar."),
    tonic=53,
    scale=DORIAN,
    tempo=120.0,
    progression=(0, 3, 6, 4),
    pad_cutoff_hz=1250.0,
    pluck_brightness=0.68,
    pluck_decay=0.30,
    arp_pattern=(0, 2, 4, 2, 1, 3, 2, 0),
    space=0.34,
    reverb_seconds=3.0,
    kick_level=0.46,
    shaker_level=0.36,
    seed=4272,
)

SOLSTICE = SoundTheme(
    key="solstice",
    label="Solstice",
    description=("G Lydian at 112 BPM: the raised fourth leaves every "
                 "chord major, so nothing in it settles — the slowest "
                 "bright set of the ten, with six voices in the pad."),
    tonic=55,
    scale=LYDIAN,
    tempo=112.0,
    progression=(0, 4, 5, 1),
    pad_voices=6,
    pad_detune_cents=12.0,
    pad_cutoff_hz=1500.0,
    pluck_brightness=0.72,
    pluck_decay=0.34,
    arp_pattern=(0, 2, 4, 1, 3, 2, 4, 3),
    delay_beats=0.5,
    space=0.40,
    reverb_seconds=3.6,
    width=0.80,
    kick_level=0.34,
    shaker_level=0.40,
    seed=4273,
)

UNDERTOW = SoundTheme(
    key="undertow",
    label="Undertow",
    description=("E minor at 116 BPM: the heaviest sub of the ten under "
                 "the most closed pad filter — weight rather than "
                 "brightness."),
    tonic=52,
    tempo=116.0,
    progression=(0, 5, 3, 6),
    pad_cutoff_hz=780.0,
    pad_pump=0.42,
    pluck_brightness=0.44,
    pluck_decay=0.50,
    arp_pattern=(0, 1, 2, 1, 3, 2, 4, 2),
    delay_feedback=0.50,
    sub_level=0.72,
    space=0.30,
    reverb_seconds=3.4,
    kick_level=0.55,
    shaker_level=0.22,
    seed=4274,
)

MERIDIAN = SoundTheme(
    key="meridian",
    label="Meridian",
    description=("A Dorian at 124 BPM: a wide stereo pad with a bright, "
                 "short pluck running eighth notes across it."),
    tonic=57,
    scale=DORIAN,
    tempo=124.0,
    progression=(0, 6, 3, 4),
    pad_cutoff_hz=1300.0,
    pluck_brightness=0.70,
    pluck_decay=0.28,
    arp_pattern=(0, 2, 1, 4, 2, 3, 1, 2),
    space=0.36,
    reverb_seconds=3.0,
    width=0.85,
    kick_level=0.48,
    shaker_level=0.38,
    seed=4275,
)

CIRRUS = SoundTheme(
    key="cirrus",
    label="Cirrus",
    description=("B minor at 126 BPM: the most open set of the ten — the "
                 "widest pad filter, the widest stereo image, the "
                 "lightest sub and the softest pump."),
    tonic=59,
    tempo=126.0,
    progression=(0, 4, 5, 6),
    pad_voices=6,
    pad_detune_cents=10.0,
    pad_cutoff_hz=1900.0,
    pad_pump=0.22,
    pluck_brightness=0.80,
    pluck_decay=0.26,
    arp_pattern=(0, 3, 2, 4, 1, 3, 2, 4),
    delay_feedback=0.48,
    delay_mix=0.46,
    sub_level=0.28,
    space=0.48,
    reverb_seconds=4.2,
    width=0.90,
    kick_level=0.30,
    shaker_level=0.42,
    seed=4276,
)

NOCTURNE = SoundTheme(
    key="nocturne",
    label="Nocturne",
    description=("C sharp minor at 108 BPM: the slowest of the ten, the "
                 "arpeggio down to quarter notes, plucks that ring like a "
                 "damped piano string, a long tail and barely a drum."),
    tonic=49,
    tempo=108.0,
    progression=(0, 5, 2, 6),
    pad_detune_cents=18.0,
    pad_cutoff_hz=850.0,
    pluck_brightness=0.38,
    pluck_decay=0.62,
    pluck_level=0.44,
    arp_pattern=(0, 2, 4, 1),
    arp_division=1,
    delay_feedback=0.46,
    delay_mix=0.34,
    sub_level=0.50,
    space=0.46,
    reverb_seconds=4.8,
    width=0.72,
    kick_level=0.18,
    shaker_level=0.12,
    bed_lufs=-19.0,
    seed=4277,
)

APHELION = SoundTheme(
    key="aphelion",
    label="Aphelion",
    description=("D harmonic minor at 128 BPM: the raised seventh puts a "
                 "dominant chord in the bed and a minor-major seventh on "
                 "the tonic — the tense end of the family."),
    tonic=50,
    scale=HARMONIC_MINOR,
    tempo=128.0,
    progression=(0, 5, 3, 4),
    pad_cutoff_hz=1050.0,
    pad_pump=0.38,
    pluck_brightness=0.66,
    pluck_decay=0.28,
    arp_pattern=(0, 3, 2, 4, 1, 3, 2, 4),
    delay_beats=0.5,
    delay_feedback=0.40,
    space=0.30,
    reverb_seconds=2.8,
    kick_level=0.52,
    shaker_level=0.36,
    seed=4278,
)

PULSAR = SoundTheme(
    key="pulsar",
    label="Pulsar",
    description=("F sharp minor at 128 BPM: sixteenth-note arpeggios, the "
                 "brightest and shortest plucks of the ten, the hardest "
                 "pump and the least room around any of it."),
    tonic=54,
    tempo=128.0,
    progression=(0, 6, 5, 4),
    pad_cutoff_hz=1150.0,
    pad_pump=0.55,
    pluck_brightness=0.86,
    pluck_decay=0.22,
    arp_pattern=(0, 2, 4, 2, 1, 3, 2, 4, 0, 3, 2, 4, 1, 2, 3, 4),
    arp_division=4,
    delay_beats=0.5,
    delay_feedback=0.36,
    delay_mix=0.32,
    sub_level=0.55,
    space=0.24,
    reverb_seconds=2.4,
    width=0.78,
    kick_level=0.60,
    shaker_level=0.45,
    seed=4279,
)

VESPER = SoundTheme(
    key="vesper",
    label="Vesper",
    description=("E flat minor at 114 BPM: the most detuned pad of the "
                 "ten, a soft mid-length pluck, a forward shaker and a "
                 "kick kept well back."),
    tonic=51,
    tempo=114.0,
    progression=(0, 3, 5, 6),
    pad_detune_cents=22.0,
    pad_cutoff_hz=950.0,
    pluck_brightness=0.50,
    pluck_decay=0.44,
    pluck_level=0.42,
    arp_pattern=(0, 1, 2, 1, 3, 2, 4, 2),
    delay_feedback=0.44,
    delay_mix=0.42,
    sub_level=0.42,
    space=0.42,
    reverb_seconds=3.8,
    width=0.75,
    kick_level=0.26,
    shaker_level=0.48,
    bed_lufs=-18.5,
    seed=4280,
)

#: The ten, in the order :data:`spacr.qt.night_themes.NIGHT_THEMES` lists
#: their themes, so the Sound set menu and the Theme menu read the same way
#: down.
NIGHT_SETS: Tuple[SoundTheme, ...] = (LANTERN, HALCYON, SOLSTICE, UNDERTOW,
                                      MERIDIAN, CIRRUS, NOCTURNE, APHELION,
                                      PULSAR, VESPER)

#: Every sound set spaCR can play, by key. Orbit first because it is the
#: default and the reference the other ten were measured against.
SOUND_THEMES: Dict[str, SoundTheme] = {ORBIT.key: ORBIT}
SOUND_THEMES.update({theme.key: theme for theme in NIGHT_SETS})

#: The sound set a fresh install uses.
DEFAULT_THEME = ORBIT.key


@dataclass
class Rendered:
    """One rendered sound and the notes it was made from.

    :param name: the sound's file stem, for example ``"click-2"``.
    :param audio: float samples, shape ``(2, n)``, peak at most 1.
    :param notes: ``(onset seconds, MIDI note, part)`` for every note
        played, in onset order. ``part`` is ``"pluck"``, ``"pad"`` or
        ``"sub"``. The schedule is what a test reads to check a figure
        rises or falls without having to transcribe the audio.
    :param loop: whether the sound is meant to repeat seamlessly.
    """

    name: str
    audio: np.ndarray
    notes: List[Tuple[float, int, str]]
    loop: bool = False


def midi_to_hz(note: float) -> float:
    """Frequency of a MIDI note number in equal temperament, A4 = 440 Hz.

    :param note: MIDI note number (69 is A4).
    :returns: frequency in hertz.
    """
    return 440.0 * 2.0 ** ((float(note) - 69.0) / 12.0)


def scale_note(theme: SoundTheme, degree: int, octave: int = 0) -> int:
    """The MIDI note of a scale degree, any number of octaves away.

    :param theme: supplies the tonic and the scale.
    :param degree: 0-based scale degree; values past the scale wrap into
        the next octave and negative values into the one below.
    :param octave: whole octaves added on top.
    :returns: a MIDI note number.
    """
    size = len(theme.scale)
    extra, step = divmod(int(degree), size)
    return int(theme.tonic + 12 * (extra + octave) + theme.scale[step])


def chord_tones(theme: SoundTheme, degree: int, count: int = 4) -> List[int]:
    """The diatonic chord built in thirds on a scale degree.

    :param theme: supplies the tonic and the scale.
    :param degree: 0-based scale degree of the chord's root.
    :param count: chord tones to stack; 4 is a seventh chord.
    :returns: MIDI notes, root first, ascending.
    """
    return [scale_note(theme, degree + 2 * k) for k in range(count)]


def _rooted(theme: SoundTheme, degree: int, count: int, centre: int) -> List[int]:
    """A chord in root position, its root in the octave nearest ``centre``.

    Stacked thirds keep every neighbouring pair of notes three or four
    semitones apart, so a detuned pad never smears two notes a semitone
    apart into a beat; and moving the whole block in parallel from chord to
    chord is how the pads of this genre move anyway.

    :param theme: supplies the tonic and the scale.
    :param degree: 0-based scale degree of the root.
    :param count: chord tones, 4 for a seventh chord.
    :param centre: MIDI note the root should sit closest to; a tie goes to
        the lower octave.
    :returns: MIDI notes, ascending.
    """
    tones = chord_tones(theme, degree, count)
    root = tones[0]
    shift = math.floor((centre - root) / 12.0 + 0.5)
    if abs(root + 12 * (shift - 1) - centre) <= abs(root + 12 * shift - centre):
        shift -= 1
    return [int(t + 12 * shift) for t in tones]


def _rng(theme: SoundTheme, salt: str) -> np.random.Generator:
    """A generator seeded by the theme and a label, stable across runs."""
    digest = hashlib.sha256(f"{theme.seed}:{theme.key}:{salt}".encode())
    return np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))


def _saw(freq: float, n: int, phase: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A band-limited sawtooth, corrected at each wrap by PolyBLEP.

    A naive sawtooth aliases audibly at pad pitches; the two-sample
    polynomial correction removes most of that for almost no cost.
    """
    dt = float(freq) / sr
    t = np.mod(phase + dt * np.arange(n, dtype=np.float64), 1.0)
    y = 2.0 * t - 1.0
    low = t < dt
    x = t[low] / dt
    y[low] -= x + x - x * x - 1.0
    high = t > 1.0 - dt
    x = (t[high] - 1.0) / dt
    y[high] -= x * x + x + x + 1.0
    return y


def _lowpass(signal: np.ndarray, cutoff: float, order: int = 2,
             sr: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth low-pass along the last axis."""
    from scipy.signal import butter, sosfilt

    cutoff = float(min(max(cutoff, 20.0), 0.45 * sr))
    sos = butter(order, cutoff, btype="low", fs=sr, output="sos")
    return sosfilt(sos, signal, axis=-1)


def _highpass(signal: np.ndarray, cutoff: float, order: int = 2,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """Butterworth high-pass along the last axis."""
    from scipy.signal import butter, sosfilt

    sos = butter(order, float(cutoff), btype="high", fs=sr, output="sos")
    return sosfilt(sos, signal, axis=-1)


def _envelope(n: int, attack: float, release: float, hold: Optional[float] = None,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """A raised-cosine attack, a flat hold, and a raised-cosine release.

    :param n: length in samples.
    :param attack: seconds to rise from silence.
    :param release: seconds to fall back to silence at the end.
    :param hold: seconds from the start at which the release begins;
        ``None`` releases at the very end.
    """
    env = np.ones(n, dtype=np.float64)
    a = max(1, min(n, int(attack * sr)))
    env[:a] = 0.5 - 0.5 * np.cos(np.linspace(0.0, math.pi, a))
    r = max(1, int(release * sr))
    start = n - r if hold is None else int(hold * sr)
    start = max(0, min(n, start))
    stop = min(n, start + r)
    if stop > start:
        env[start:stop] *= 0.5 + 0.5 * np.cos(
            np.linspace(0.0, math.pi, stop - start))
    env[stop:] = 0.0
    return env


def _pan(mono: np.ndarray, position: float) -> np.ndarray:
    """Equal-power pan of a mono signal; -1 is hard left, +1 hard right."""
    angle = (float(np.clip(position, -1.0, 1.0)) + 1.0) * math.pi / 4.0
    return np.vstack([mono * math.cos(angle), mono * math.sin(angle)])


def _pluck(freq: float, seconds: float, brightness: float, decay: float,
           sr: int = SAMPLE_RATE) -> np.ndarray:
    """A plucked note, built from harmonics that die faster the higher they are.

    That is what a sawtooth through a closing low-pass filter sounds like --
    the pluck of melodic house -- and it is exact in pitch, which a
    Karplus-Strong string at these frequencies is not. The two channels are
    detuned by a few cents so a single note already has width.

    :returns: stereo samples, shape ``(2, n)``.
    """
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    out = np.zeros((2, n))
    limit = 0.45 * sr
    tilt = float(np.clip(brightness, 0.05, 0.95))
    for channel, cents in ((0, -3.0), (1, 3.0)):
        f0 = freq * 2.0 ** (cents / 1200.0)
        harmonic = 1
        while harmonic * f0 < limit and harmonic <= 32:
            amp = tilt ** (harmonic - 1) / harmonic
            if amp < 2e-4:
                break
            tau = decay / (1.0 + 0.55 * (harmonic - 1))
            span = min(n, int(tau * 9.0 * sr) + 1)
            out[channel, :span] += (
                amp * np.exp(-t[:span] / tau)
                * np.sin(2.0 * math.pi * harmonic * f0 * t[:span]))
            harmonic += 1
    attack = max(1, int(0.0015 * sr))
    out[:, :attack] *= np.linspace(0.0, 1.0, attack)
    tail = max(1, int(0.01 * sr))
    out[:, -tail:] *= np.linspace(1.0, 0.0, tail)
    return out


def _supersaw(freq: float, n: int, voices: int, detune_cents: float,
              width: float, rng: np.random.Generator,
              sr: int = SAMPLE_RATE) -> np.ndarray:
    """Detuned sawtooth voices spread across the stereo field.

    :returns: stereo samples, shape ``(2, n)``, roughly unit RMS per voice.
    """
    voices = max(1, int(voices))
    out = np.zeros((2, n))
    for v in range(voices):
        spread = 0.0 if voices == 1 else 2.0 * v / (voices - 1) - 1.0
        detuned = freq * 2.0 ** (spread * detune_cents / 1200.0)
        wave_ = _saw(detuned, n, float(rng.random()), sr)
        out += _pan(wave_, spread * width)
    return out / math.sqrt(voices)


def _lead_voice(theme: SoundTheme, freq: float, seconds: float,
                rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """One note of the melody: a small supersaw under a closing filter.

    A pluck cannot carry a tune under an arpeggio of plucks -- it is the
    same attack and the same decay, so it joins the texture instead of
    rising out of it. This voice is told apart by SUSTAIN: three gently
    detuned saws, an attack slow enough to be a breath rather than a hit,
    and a filter that opens on the note and closes across it, so the note
    is still there when the next one arrives.

    :param theme: supplies the attack, the brightness and the width.
    :param freq: the note, in hertz.
    :param seconds: how long the note sounds, tail included.
    :param rng: seeds the saw phases, so a theme always renders the same.
    :returns: stereo samples, shape ``(2, n)``.
    """
    n = max(1, int(seconds * sr))
    tone = _supersaw(freq, n, 3, theme.pad_detune_cents * 0.45,
                     theme.width * 0.55, rng, sr)
    cutoff = float(np.clip(freq * theme.lead_brightness * 9.0, 400.0, 2600.0))
    closing = np.exp(-np.arange(n, dtype=np.float64) / max(1.0, 0.30 * sr))
    tone = _lowpass(_swelling_filter(tone, cutoff, closing, sr),
                    LEAD_AIR_HZ, order=8, sr=sr)
    release = float(min(0.4, max(0.08, seconds * 0.45)))
    return tone * _envelope(n, theme.lead_attack, release, sr=sr)


def _sub(freq: float, n: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    """A sine sub with a little second harmonic so small speakers hear it."""
    t = np.arange(n, dtype=np.float64) / sr
    tone = np.sin(2.0 * math.pi * freq * t) + 0.18 * np.sin(
        4.0 * math.pi * freq * t)
    return np.vstack([tone, tone])


def _ping_pong(signal: np.ndarray, delay_s: float, feedback: float,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """Echoes of ``signal`` alternating right and left, each one darker.

    Computed tap by tap rather than as a recursive filter: the delay is
    thousands of samples long, so a handful of shifted, filtered copies is
    both exact and fast.

    :param signal: stereo input, shape ``(2, n)``; echoes are taken from
        its mono sum.
    :returns: the echoes alone, shape ``(2, n + taps * delay)``.
    """
    feedback = float(np.clip(feedback, 0.0, 0.9))
    d = max(1, int(round(delay_s * sr)))
    if feedback <= 0.0:
        return np.zeros((2, signal.shape[1]))
    taps = int(min(10, max(1, math.ceil(math.log(0.02) / math.log(feedback)))))
    n = signal.shape[1]
    out = np.zeros((2, n + taps * d))
    echo = signal.mean(axis=0)
    for k in range(1, taps + 1):
        echo = _lowpass(echo, 4200.0 / (1.0 + 0.25 * k), order=1, sr=sr) * feedback
        channel = 1 if k % 2 else 0
        out[channel, k * d:k * d + n] += echo
    return out


def _reverb_ir(seconds: float, rng: np.random.Generator,
               sr: int = SAMPLE_RATE) -> np.ndarray:
    """A synthetic stereo impulse response: decaying, darkening noise.

    The two channels are independent noise, which is what makes the tail
    wide; the tail darkens as it decays, as a real room's does, by fading
    from a bright to a dark filtered copy.
    """
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    decay = np.exp(-6.9 * t / max(seconds, 0.05))
    noise = rng.standard_normal((2, n))
    bright = _lowpass(noise, 7000.0, sr=sr)
    dark = _lowpass(noise, 1800.0, sr=sr)
    blend = np.clip(t / max(seconds * 0.5, 1e-3), 0.0, 1.0)
    ir = (bright * (1.0 - blend) + dark * blend) * decay
    onset = max(1, int(0.03 * sr))
    ir[:, :onset] *= np.linspace(0.0, 1.0, onset)
    predelay = np.zeros((2, int(0.018 * sr)))
    ir = np.concatenate([predelay, ir], axis=1)
    energy = np.sqrt((ir ** 2).sum(axis=1, keepdims=True))
    return ir / np.maximum(energy, 1e-9)


def _convolve(signal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """Convolve each channel of ``signal`` with the same channel of ``ir``."""
    from scipy.signal import fftconvolve

    return np.vstack([fftconvolve(signal[c], ir[c]) for c in range(2)])


def _add(bus: np.ndarray, part: np.ndarray, at: int) -> None:
    """Mix ``part`` into ``bus`` starting at sample ``at``, clipped to fit."""
    if at >= bus.shape[1]:
        return
    stop = min(bus.shape[1], at + part.shape[1])
    bus[:, at:stop] += part[:, :stop - at]


def _pad(length: int, extra: int = 0) -> np.ndarray:
    """An empty stereo bus."""
    return np.zeros((2, int(length) + int(extra)))


def _trim(audio: np.ndarray, floor_db: float = -66.0,
          sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cut the silent end of a one-shot sound and fade its last 20 ms.

    A tail below ``floor_db`` of the peak cannot be heard at any volume the
    slider allows, and every second of it is a second the file takes to
    load and hold in memory.
    """
    peak = float(np.max(np.abs(audio))) or 1.0
    floor = peak * 10.0 ** (floor_db / 20.0)
    loud = np.nonzero(np.max(np.abs(audio), axis=0) > floor)[0]
    end = int(loud[-1]) + 1 if loud.size else audio.shape[1]
    end = min(audio.shape[1], end + int(0.02 * sr))
    audio = audio[:, :end].copy()
    fade = max(1, min(end, int(0.02 * sr)))
    audio[:, -fade:] *= np.linspace(1.0, 0.0, fade)
    return audio


def _level(audio: np.ndarray, peak_db: float, fade_in: bool = True,
           sr: int = SAMPLE_RATE) -> np.ndarray:
    """Round the peaks and set the level, sample by sample.

    ``tanh`` rounds only what is already loud, so the level can be set by
    the peak without one transient deciding it. Everything here is
    memoryless, which is what lets the music bed be levelled AFTER it is
    folded into a loop without putting a seam back in.

    :param fade_in: ramp the first 2 ms up from silence. A one-shot wants
        it; a loop must not have it, or every repeat would dip.
    """
    peak = float(np.max(np.abs(audio))) or 1.0
    audio = np.tanh(1.2 * audio / peak) / math.tanh(1.2)
    target = 10.0 ** (peak_db / 20.0)
    audio = audio * (target / (float(np.max(np.abs(audio))) or 1.0))
    if fade_in:
        fade = max(1, int(0.002 * sr))
        audio[:, :fade] *= np.linspace(0.0, 1.0, fade)
    return audio


def _master(audio: np.ndarray, peak_db: float,
            sr: int = SAMPLE_RATE) -> np.ndarray:
    """High-pass the rumble out of a one-shot sound, then level it."""
    return _level(_highpass(audio, 28.0, sr=sr), peak_db, True, sr)


def _with_space(dry: np.ndarray, theme: SoundTheme, send: float,
                seconds: float, salt: str,
                sr: int = SAMPLE_RATE) -> np.ndarray:
    """``dry`` plus its reverb, the reverb scaled by ``send``."""
    if send <= 0.0:
        return dry
    wet = _convolve(dry, _reverb_ir(seconds, _rng(theme, salt), sr))
    out = _pad(wet.shape[1])
    out[:, :dry.shape[1]] += dry
    return out + send * wet


def _click(theme: SoundTheme, index: int, sr: int = SAMPLE_RATE) -> Rendered:
    """A short in-key pluck with one quiet echo, for a pressed control.

    The four variants are the tonic, third, fifth and second of the key in
    the octave above the pads, so a run of clicks spells a small phrase.
    """
    degree = (0, 2, 4, 1)[index % 4]
    note = scale_note(theme, degree, octave=1)
    dry = _pluck(midi_to_hz(note), 0.30, theme.pluck_brightness * 0.8, 0.06, sr)
    echo = _ping_pong(dry, theme.delay_beats * theme.beat, 0.18, sr)
    bus = _pad(echo.shape[1])
    _add(bus, dry, 0)
    bus += 0.6 * echo
    audio = _with_space(bus, theme, 0.10, 0.9, f"click-{index}", sr)
    audio = _trim(_master(audio, -12.0, sr), sr=sr)
    return Rendered(f"click-{index}", audio, [(0.0, note, "pluck")])


def _hover(theme: SoundTheme, index: int, sr: int = SAMPLE_RATE) -> Rendered:
    """A muted, very short pluck an octave above the click, and quieter.

    No echo: a hover can follow another within a fraction of a second, and
    echoes of echoes are what turn hover sounds into a stream.
    """
    degree = (4, 6, 7, 9)[index % 4]
    note = scale_note(theme, degree, octave=1)
    dry = _pluck(midi_to_hz(note), 0.16, theme.pluck_brightness * 0.5, 0.035, sr)
    audio = _with_space(dry, theme, 0.06, 0.6, f"hover-{index}", sr)
    audio = _trim(_master(audio, -20.0, sr), sr=sr)
    return Rendered(f"hover-{index}", audio, [(0.0, note, "pluck")])


def _pad_chord(theme: SoundTheme, notes: Sequence[int], n: int,
               rng: np.random.Generator, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Every note of a chord as a detuned-saw pad, summed, unfiltered."""
    bus = _pad(n)
    for note in notes:
        bus += _supersaw(midi_to_hz(note), n, theme.pad_voices,
                         theme.pad_detune_cents, theme.width, rng, sr)
    return bus / math.sqrt(max(1, len(notes)))


def _swelling_filter(pad: np.ndarray, cutoff: float, opening: np.ndarray,
                     sr: int = SAMPLE_RATE) -> np.ndarray:
    """Cross-fade a dark and an open copy of ``pad`` by ``opening`` (0 to 1).

    A time-varying low-pass without a sample-by-sample loop: two static
    filters, blended, sound like one filter sweeping between them.
    """
    closed = _lowpass(pad, cutoff, order=4, sr=sr)
    opened = _lowpass(pad, cutoff * 2.0, order=4, sr=sr)
    return closed * (1.0 - opening) + opened * opening


def _run_finished(theme: SoundTheme, sr: int = SAMPLE_RATE) -> Rendered:
    """A rising arpeggio through the seventh-degree chord into the tonic.

    Sixteenth notes climb the VII triad (G B D in A minor) for two octaves
    and step up a whole tone onto the tonic, where they stop; under them a
    pad swells from the VII chord into the tonic chord with its ninth, and
    a sub enters on the resolution. VII to i is the cadence this genre
    resolves with.
    """
    rng = _rng(theme, "run_finished")
    step = theme.beat / 4.0
    climb = [scale_note(theme, 6 + 2 * k, octave=-1) for k in range(3)]
    climb = climb + [n + 12 for n in climb] + [climb[0] + 24]
    tonic = int(theme.tonic)
    while tonic <= climb[-1]:
        tonic += 12
    arrival = step * len(climb)
    length = arrival + 2.4
    n = int(length * sr)

    plucks = _pad(n)
    notes: List[Tuple[float, int, str]] = []
    for i, note in enumerate(climb):
        at = i * step
        velocity = 0.55 + 0.45 * i / max(1, len(climb) - 1)
        plucks_note = _pluck(midi_to_hz(note), 0.9, theme.pluck_brightness,
                             theme.pluck_decay * 0.6, sr) * velocity
        _add(plucks, plucks_note, int(at * sr))
        notes.append((at, note, "pluck"))
    final = _pluck(midi_to_hz(tonic), 2.2, theme.pluck_brightness,
                   theme.pluck_decay * 1.8, sr)
    _add(plucks, final, int(arrival * sr))
    notes.append((arrival, tonic, "pluck"))
    body = _pluck(midi_to_hz(tonic - 12), 1.6, theme.pluck_brightness * 0.7,
                  theme.pluck_decay * 1.4, sr) * 0.45
    _add(plucks, body, int(arrival * sr))
    notes.append((arrival, tonic - 12, "pluck"))
    echoes = _ping_pong(plucks, theme.delay_beats * theme.beat,
                        theme.delay_feedback, sr)

    before = [scale_note(theme, d, octave=-1) for d in (6, 8, 10)]
    after = [scale_note(theme, d) for d in (0, 2, 4, 8)]
    a_n = int(arrival * sr)
    pad = _pad(n)
    first = _pad_chord(theme, before, a_n + int(0.25 * sr), rng, sr)
    first *= _envelope(first.shape[1], arrival * 0.9, 0.3, sr=sr) * 0.55
    _add(pad, first, 0)
    second = _pad_chord(theme, after, n - a_n, rng, sr)
    second *= _envelope(second.shape[1], 0.35, 1.5, sr=sr)
    _add(pad, second, a_n)
    opening = np.clip(np.arange(n) / max(1, a_n + int(0.4 * sr)), 0.0, 1.0)
    pad = _swelling_filter(pad, theme.pad_cutoff_hz, opening, sr)
    for note in before:
        notes.append((0.0, note, "pad"))
    for note in after:
        notes.append((arrival, note, "pad"))

    sub_note = theme.tonic - 24
    sub = _sub(midi_to_hz(sub_note), n - a_n, sr)
    sub *= _envelope(sub.shape[1], 0.08, 1.4, sr=sr)
    notes.append((arrival, sub_note, "sub"))

    bus = _pad(echoes.shape[1])
    _add(bus, plucks * theme.pluck_level, 0)
    bus += echoes * theme.pluck_level * theme.delay_mix * 1.6
    _add(bus, pad * theme.pad_level * 0.8, 0)
    wet_send = _with_space(bus, theme, theme.space, theme.reverb_seconds * 0.8,
                           "run_finished-space", sr)
    _add(wet_send, sub * theme.sub_level * 0.7, a_n)
    audio = _trim(_master(wet_send, -6.0, sr), sr=sr)
    notes.sort(key=lambda row: (row[0], row[2], row[1]))
    return Rendered("run_finished", audio, notes)


def _run_failed(theme: SoundTheme, sr: int = SAMPLE_RATE) -> Rendered:
    """A slower figure falling from the fifth to the tonic over the iv chord.

    Eighth notes step down five, four, three, one of the key (E D C A in
    A minor), darker and softer than the finished sound, over a closed
    minor-fourth pad (Dm7) that never resolves. It says "stopped" without
    an alarm in it.
    """
    rng = _rng(theme, "run_failed")
    step = theme.beat / 2.0
    fall = [scale_note(theme, d, octave=1) for d in (4, 3, 2, 0)]
    length = step * len(fall) + 2.2
    n = int(length * sr)
    plucks = _pad(n)
    notes: List[Tuple[float, int, str]] = []
    for i, note in enumerate(fall):
        at = i * step
        last = i == len(fall) - 1
        velocity = 0.9 - 0.12 * i
        note_audio = _pluck(midi_to_hz(note), 1.8 if last else 0.8,
                            theme.pluck_brightness * 0.6,
                            theme.pluck_decay * (1.4 if last else 0.8), sr)
        _add(plucks, note_audio * velocity, int(at * sr))
        notes.append((at, note, "pluck"))
    echoes = _ping_pong(plucks, theme.delay_beats * theme.beat,
                        theme.delay_feedback * 0.7, sr)

    chord = _rooted(theme, 3, 4, theme.tonic)
    pad = _pad_chord(theme, chord, n, rng, sr)
    pad *= _envelope(n, 0.5, 1.6, sr=sr)
    pad = _lowpass(pad, theme.pad_cutoff_hz * 0.7, order=4, sr=sr)
    for note in chord:
        notes.append((0.0, note, "pad"))
    sub_note = scale_note(theme, 3, octave=-2)
    sub = _sub(midi_to_hz(sub_note), n, sr) * _envelope(n, 0.3, 1.2, sr=sr)
    notes.append((0.0, sub_note, "sub"))

    bus = _pad(echoes.shape[1])
    _add(bus, plucks * theme.pluck_level, 0)
    bus += echoes * theme.pluck_level * theme.delay_mix * 1.2
    _add(bus, pad * theme.pad_level * 0.7, 0)
    audio = _with_space(bus, theme, theme.space, theme.reverb_seconds * 0.8,
                        "run_failed-space", sr)
    _add(audio, sub * theme.sub_level * 0.6, 0)
    audio = _trim(_master(audio, -7.0, sr), sr=sr)
    notes.sort(key=lambda row: (row[0], row[2], row[1]))
    return Rendered("run_failed", audio, notes)


#: Below this a part is left out of the render rather than played quietly.
#: A kick at two per cent is a sample nobody can hear and a transient the
#: loudness normaliser still has to make room for.
PART_FLOOR = 0.05


class BedBar(NamedTuple):
    """How loudly each part plays in one bar of the music bed.

    :param section: which entry of :data:`BED_SECTIONS` this bar is in.
    :param arp: arpeggio level, 0 to 1.
    :param kick: kick level, 0 to 1, before the theme's ``kick_level``.
    :param shaker: shaker level, 0 to 1, before ``shaker_level``.
    :param sub: sub level, 0 to 1.
    :param lift: semitones the arpeggio rises by in this bar's second half.
    :param lead: melody level, 0 to 1, before the theme's ``lead_level``.
        Last, because this is a public tuple and a field added anywhere but
        the end changes what every position after it means.
    """

    section: str
    arp: float
    kick: float
    shaker: float
    sub: float
    lift: int
    lead: float


def bed_plan(bars: int) -> List[BedBar]:
    """The arrangement, bar by bar, for a loop of ``bars`` bars.

    Every part reaches its section's level over the section's first half
    and holds it, so nothing arrives as a step; the ramp for the FIRST
    section starts from the LAST one's levels, which is what makes the
    arrangement circular rather than merely long. ``bars`` is stretched
    proportionally across :data:`BED_SECTIONS`, so a theme can ask for a
    shorter or longer loop and still get the same shape.

    :param bars: bars in the loop; at least one per section.
    :returns: one :class:`BedBar` per bar.
    """
    count = len(BED_SECTIONS)
    bars = max(count, int(bars))
    lengths = [max(1, int(round(bars * spec[1]
                                / sum(s[1] for s in BED_SECTIONS))))
               for spec in BED_SECTIONS]
    while sum(lengths) > bars:
        lengths[lengths.index(max(lengths))] -= 1
    while sum(lengths) < bars:
        lengths[lengths.index(min(lengths))] += 1

    plan: List[BedBar] = []
    for index, (name, _weight, arp, kick, shaker, sub,
                lead) in enumerate(BED_SECTIONS):
        before = BED_SECTIONS[index - 1]
        span = lengths[index]
        for b in range(span):
            amount = min(1.0, 2.0 * b / span)
            ease = 0.5 - 0.5 * math.cos(math.pi * amount)
            bar = len(plan)
            plan.append(BedBar(
                section=name,
                arp=before[2] + (arp - before[2]) * ease,
                kick=before[3] + (kick - before[3]) * ease,
                shaker=before[4] + (shaker - before[4]) * ease,
                sub=before[5] + (sub - before[5]) * ease,
                lead=before[6] + (lead - before[6]) * ease,
                lift=12 if bar % 4 == 3 else 0,
            ))
    return plan


def _kick(seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """One soft four-on-the-floor kick: a dropping sine with a short knock.

    The pitch falls from about 150 Hz to the fundamental in thirty
    milliseconds, which is what a kick drum is; the knock is a very short
    burst an octave and a half above it rather than a click of noise, so
    the drum stays round enough to sit under a settings form. It is
    low-passed at 1.2 kHz: this has to be FELT and never TICK.
    """
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    sweep = 48.0 + 106.0 * np.exp(-t / 0.028)
    body = np.sin(2.0 * math.pi * np.cumsum(sweep) / sr) * np.exp(-t / 0.17)
    knock = 0.16 * np.sin(2.0 * math.pi * 128.0 * t) * np.exp(-t / 0.006)
    mono = _lowpass(body + knock, 1200.0, order=2, sr=sr)
    attack = max(1, int(0.0012 * sr))
    mono[:attack] *= np.linspace(0.0, 1.0, attack)
    tail = max(1, int(0.008 * sr))
    mono[-tail:] *= np.linspace(1.0, 0.0, tail)
    return np.vstack([mono, mono]) / (float(np.max(np.abs(mono))) or 1.0)


def _shaker(seconds: float, rng: np.random.Generator,
            sr: int = SAMPLE_RATE) -> np.ndarray:
    """One shaker hit: a very short burst of high-passed noise.

    Two channels of independent noise, so the shaker is wide where the kick
    is dead centre and the two never mask each other.
    """
    n = max(1, int(seconds * sr))
    t = np.arange(n, dtype=np.float64) / sr
    burst = rng.standard_normal((2, n)) * np.exp(-t / 0.021)
    out = _highpass(burst, 3800.0, order=2, sr=sr)
    attack = max(1, int(0.0008 * sr))
    out[:, :attack] *= np.linspace(0.0, 1.0, attack)
    tail = max(1, int(0.004 * sr))
    out[:, -tail:] *= np.linspace(1.0, 0.0, tail)
    return out / (float(np.max(np.abs(out))) or 1.0)


#: The two K-weighting stages of ITU-R BS.1770, as direct-form coefficients
#: at 48 kHz: a high shelf that stands for the head's own response, then a
#: high-pass that discards what is felt rather than heard.
_K_SHELF = ((1.53512485958697, -2.69169618940638, 1.19839281085285),
            (1.0, -1.69065929318241, 0.73248077421585))
_K_HIGHPASS = ((1.0, -2.0, 1.0),
               (1.0, -1.99004745483398, 0.99007225036621))


def loudness_lufs(audio: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Programme loudness in LUFS, by ITU-R BS.1770-4.

    The gated integrated measurement: K-weight both channels, take the
    mean square over 400 ms blocks overlapping by three quarters, drop
    every block below -70 LUFS, then drop every block more than 10 LU under
    the mean of what is left and take the mean of the rest.

    A PEAK IS NOT A LOUDNESS, which is the reason this exists. The music
    bed and the run sounds have similar peaks and are nothing like as loud
    as each other, and "quiet enough to work under" is a statement about
    loudness. The coefficients are the standard's own and are written for
    48 kHz; another rate is measured with them anyway and the answer drifts
    by a fraction of a LU, which is inside what anybody can hear.

    :param audio: shape ``(channels, n)``.
    :param sr: sample rate.
    :returns: LUFS, or ``-inf`` for silence.
    """
    from scipy.signal import lfilter

    data = np.atleast_2d(np.asarray(audio, dtype=np.float64))
    weighted = lfilter(*_K_SHELF, data, axis=-1)
    weighted = lfilter(*_K_HIGHPASS, weighted, axis=-1)
    block = int(0.4 * sr)
    hop = max(1, block // 4)
    if weighted.shape[1] < block:
        power = float((weighted ** 2).mean(axis=-1).sum())
        return -0.691 + 10.0 * math.log10(power) if power > 0 else -math.inf
    starts = range(0, weighted.shape[1] - block + 1, hop)
    powers = np.array([float((weighted[:, s:s + block] ** 2)
                             .mean(axis=-1).sum()) for s in starts])
    loud = np.where(powers > 0.0,
                    -0.691 + 10.0 * np.log10(np.maximum(powers, 1e-30)),
                    -np.inf)
    keep = loud > -70.0
    if not keep.any():
        return -math.inf
    absolute = -0.691 + 10.0 * math.log10(float(powers[keep].mean()))
    keep = keep & (loud > absolute - 10.0)
    if not keep.any():
        return absolute
    return -0.691 + 10.0 * math.log10(float(powers[keep].mean()))


#: How many gain-then-knee rounds :func:`_to_loudness` takes. See its
#: docstring for the measurement that chose it.
LOUDNESS_ROUNDS = 6


def _to_loudness(audio: np.ndarray, target_lufs: float, peak_db: float,
                 sr: int = SAMPLE_RATE) -> np.ndarray:
    """Set a loop's programme loudness, then hold it under a peak ceiling.

    Two memoryless operations and nothing else -- a gain and a ``tanh``
    knee -- because this runs AFTER the loop has been folded and anything
    with a memory would put a step back in at the seam. The knee costs
    loudness, so the gain is re-derived after it.

    HOW MANY ROUNDS, measured rather than assumed. This said "two rounds is
    enough to land inside a tenth of a LU in every theme measured", and for
    eight of the ten night themes it is. It is not for the two whose bed
    still breaks the ceiling on the second round: the loop applies the knee
    and then RETURNS, so the loudness the knee just cost is never taken back.
    Measured on the ten themes' beds, declared -18.0 LUFS:

        rounds   meridian   aphelion   peak after
             2    -18.562    -18.579   -11.36 dB
             4    -18.000    -18.000   -10.80 dB
             6    -18.000    -18.000   -10.80 dB

    Four is where both land exactly and the peak stops moving; six is what
    runs, for the margin, and costs nothing when the loop returns early --
    which it does on the first round for every theme that never reaches the
    ceiling at all.

    :param audio: the folded loop, shape ``(2, n)``.
    :param target_lufs: programme loudness to aim for.
    :param peak_db: the ceiling no sample may pass, in dBFS.
    :param sr: sample rate.
    :returns: the levelled loop.
    """
    ceiling = 10.0 ** (float(peak_db) / 20.0)
    out = np.asarray(audio, dtype=np.float64)
    for _ in range(LOUDNESS_ROUNDS):
        measured = loudness_lufs(out, sr)
        if not math.isfinite(measured):
            return out
        out = out * 10.0 ** ((float(target_lufs) - measured) / 20.0)
        peak = float(np.max(np.abs(out)))
        if peak <= ceiling:
            return out
        out = np.tanh(out / ceiling) * ceiling
    return out


#: How long the side-chain takes to duck, as a fraction of the beat. About
#: forty milliseconds at 122 BPM, which is a compressor's attack.
PUMP_ATTACK = 0.08

#: The shortest step a lead pattern can advance by, in beats -- a
#: thirty-second note. A pattern is data a theme hands in, and part C's ten
#: themes each hand in their own, so a length of zero arrives sooner or
#: later from a typo or from a rhythm computed with integer division. The
#: scheduler walks forward by the length it is given, so zero or a negative
#: length would never reach the end of the loop and would grow the note
#: list until the process was killed. Clamping here keeps the walk finite
#: for any pattern at all; the same clamped value is the note's length, so
#: a zero-length note sounds as the shortest note rather than as nothing.
#: The step is capped at the length of the loop at the other end, so a note
#: asked for in millions of beats renders one loop's worth of samples
#: instead of asking for an array nothing can hold.
LEAD_MIN_BEATS = 0.125


def _pump(n: int, beat: float, depth: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """The side-chain dip: down INTO every beat, back up over half of it.

    AND IT GOES DOWN IN A FINITE TIME, which is not a detail. Recovering
    over half a beat and then snapping back to the bottom at the beat
    leaves a step in the gain -- eight decibels at the reference depth,
    a hundred and twenty-eight times over the loop -- and a step in a gain
    is a click. On the pads alone it was quiet enough to miss. Applied to
    the whole music bus it was loud enough to measure: it raised the mix
    above 4 kHz by more than the shaker puts there, which is how it was
    found. :data:`PUMP_ATTACK` is the fall, so the curve is continuous
    where it wraps and the dip is still fast enough to be felt.

    :param n: length in samples.
    :param beat: seconds per beat.
    :param depth: how far the gain falls, 0 to 1.
    :param sr: sample rate.
    :returns: a gain curve, one value per sample.
    """
    t = np.arange(n, dtype=np.float64) / sr
    phase = np.mod(t / beat, 1.0)
    shape = 0.5 - 0.5 * np.cos(math.pi * np.clip(phase / 0.5, 0.0, 1.0))
    into = np.clip((phase - (1.0 - PUMP_ATTACK)) / PUMP_ATTACK, 0.0, 1.0)
    shape = shape * (0.5 + 0.5 * np.cos(math.pi * into))
    return 1.0 - float(np.clip(depth, 0.0, 1.0)) * (1.0 - shape)


def _fold(audio: np.ndarray, length: int) -> np.ndarray:
    """Wrap everything past ``length`` back onto the start.

    That is what makes the bed loop without a seam: the reverb and the
    echoes still ringing at the end of bar eight are exactly what should be
    sounding under bar one the second time round.
    """
    out = audio[:, :length].copy()
    at = length
    while at < audio.shape[1]:
        chunk = audio[:, at:at + length]
        out[:, :chunk.shape[1]] += chunk
        at += length
    return out


def _bed(theme: SoundTheme, sr: int = SAMPLE_RATE) -> Rendered:
    """The looping music bed: a composed arrangement, not a repeated phrase.

    The one sound long enough to be listened to rather than noticed, so it
    is the one that has to carry the sound set's genre on its own.

    WHAT IS PLAYED. One diatonic seventh chord per bar from
    ``theme.progression`` (i - VI - III - VII in the reference set), the
    pads pumping on every beat and breathing open and shut over the whole
    loop; ``theme.arp_pattern`` an octave above them through a dotted-eighth
    ping-pong delay, lifted an octave in the second half of every fourth
    bar; ``theme.lead_pattern`` sung over the top of both on its own
    sustaining voice, in and out with the arrangement; a sine sub on each
    chord's root two octaves down; and, from :func:`bed_plan`, a soft
    four-on-the-floor kick and an eighth-note shaker that come in and go
    out with the sections.

    THE WHOLE MUSIC BUS DUCKS ON THE BEAT, not the pads alone. The pump is
    applied to the summed pads, arpeggio, lead, shaker and echoes, so the
    REVERB is ducked with what feeds it, which is what a side-chain in this
    genre actually does and what the pads alone could not do: the tail of a
    three-second room filled the dip straight back in, and the pump
    measured a little over a decibel where the pads themselves were down
    three. The kick and half of the sub stay out of it, because they are
    what the rest is ducking for.

    WHAT MAKES IT A LOOP AND NOT A PHRASE. :data:`BED_SECTIONS` gives the
    thirty-two bars a shape that closes where it opened: the quiet section
    is at both ends, so the seam falls where the music has least to give
    away, and every part ramps to its section's level over four bars rather
    than arriving as a step. The reverb and the echoes still ringing at the
    end are folded onto the start by :func:`_fold`, and the level is set by
    :func:`_to_loudness`, which is two memoryless operations and therefore
    cannot put a step back in.

    THE DRUMS CAN BE TURNED OFF AND THAT IS NOT A SETTING WITH NOTHING
    BEHIND IT. ``theme.kick_level`` and ``theme.shaker_level`` at 0 leave
    the kick and the shaker out of the render altogether
    (:data:`PART_FLOOR`), so the bed is the pad-and-arpeggio piece it was
    before they existed, and a theme that wants to sit under a talk can ask
    for exactly that.

    THREE MORE THINGS THE CODE CANNOT SAY. The shaker's OFFBEAT is the
    loud one, because that is where the shaker of house music lives and a
    pattern with every hit at the same weight reads as a hiss. Both filter breaths are
    PERIODIC OVER THE LOOP -- one opening across the whole thirty-two bars
    and one four times, each a raised cosine that is 0 at both ends -- so a
    filter can move for a minute and still arrive back where it started.
    And the kick and the sub STAY DRY: low frequencies through a
    three-second tail are what turn a quiet bed into a rumble, and the
    kick's job is to be felt on the beat rather than to fill the room.

    :param theme: the sound set.
    :param sr: sample rate.
    :returns: the loop and every note in it.
    """
    rng = _rng(theme, "bed")
    beat = theme.beat
    bar = 4.0 * beat
    plan = bed_plan(theme.bed_bars)
    bars = len(plan)
    length = int(round(bars * bar * sr))
    tail = int((theme.reverb_seconds + 2.0) * sr)
    pads = _pad(length, tail)
    plucks = _pad(length, tail)
    leads = _pad(length, tail)
    subs = _pad(length, tail)
    drums = _pad(length, tail)
    shakers = _pad(length, tail)
    notes: List[Tuple[float, int, str]] = []
    pluck_cache: Dict[int, np.ndarray] = {}
    step = beat / max(1, int(theme.arp_division))
    steps_per_bar = int(round(bar / step))
    pattern = theme.arp_pattern or (0,)
    kick = (_kick(0.62, sr) if theme.kick_level > PART_FLOOR else None)
    shaker = (_shaker(0.14, _rng(theme, "shaker"), sr)
              if theme.shaker_level > PART_FLOOR else None)

    for b, row in enumerate(plan):
        degree = theme.progression[b % len(theme.progression)]
        start = b * bar
        chord = _rooted(theme, degree, 4, theme.tonic + 2)
        span = int((bar + 0.35) * sr)
        chord_audio = _pad_chord(theme, chord, span, rng, sr)
        chord_audio *= _envelope(span, 0.25, 0.45, sr=sr)
        _add(pads, chord_audio, int(start * sr))
        for note in chord:
            notes.append((start, note, "pad"))

        tones = [note + 12 for note in chord]
        tones = tones + [tones[0] + 12]
        if row.arp > PART_FLOOR:
            for s in range(steps_per_bar):
                at = start + s * step
                index = pattern[(b * steps_per_bar + s) % len(pattern)]
                note = tones[int(index) % len(tones)]
                if row.lift and s >= steps_per_bar // 2:
                    note += row.lift
                accent = 1.0 if s % max(1, int(theme.arp_division)) == 0 else 0.72
                accent *= 1.0 + 0.08 * (float(rng.random()) - 0.5)
                if note not in pluck_cache:
                    pluck_cache[note] = _pluck(
                        midi_to_hz(note), theme.pluck_decay * 6.0,
                        theme.pluck_brightness, theme.pluck_decay, sr)
                _add(plucks, pluck_cache[note] * accent * row.arp,
                     int(at * sr))
                notes.append((at, note, "pluck"))

        root = scale_note(theme, degree, octave=-2)
        while root > theme.tonic - 17:
            root -= 12
        sub_span = int((bar + 0.1) * sr)
        sub_audio = _sub(midi_to_hz(root), sub_span, sr)
        sub_audio *= _envelope(sub_span, 0.06, 0.12, sr=sr) * row.sub
        _add(subs, sub_audio, int(start * sr))
        notes.append((start, root, "sub"))

        if kick is not None and row.kick > PART_FLOOR:
            for hit in range(4):
                at = start + hit * beat
                _add(drums, kick * row.kick, int(at * sr))
                notes.append((at, 24, "kick"))
        if shaker is not None and row.shaker > PART_FLOOR:
            for hit in range(8):
                at = start + hit * beat / 2.0
                accent = 1.0 if hit % 2 else 0.42
                _add(shakers, shaker * row.shaker * accent, int(at * sr))
                notes.append((at, 42, "shaker"))

    if theme.lead_level > PART_FLOOR and theme.lead_pattern:
        lead_rng = _rng(theme, "lead")
        position = 0.0
        index = 0
        while position < 4.0 * bars:
            degree, beats = theme.lead_pattern[index % len(theme.lead_pattern)]
            index += 1
            step = min(max(LEAD_MIN_BEATS, float(beats)), 4.0 * bars)
            row = plan[min(bars - 1, int(position // 4.0))]
            if int(degree) != REST and row.lead > PART_FLOOR:
                note = scale_note(theme, int(degree), octave=theme.lead_octave)
                voice = _lead_voice(theme, midi_to_hz(note),
                                    step * beat + 0.3, lead_rng, sr)
                _add(leads, voice * row.lead, int(position * beat * sr))
                notes.append((position * beat, note, "lead"))
            position += step

    total = pads.shape[1]
    t = np.arange(total, dtype=np.float64) / sr
    loop_seconds = bars * bar
    slow = 0.5 - 0.5 * np.cos(2.0 * math.pi * t / loop_seconds)
    quick = 0.5 - 0.5 * np.cos(8.0 * math.pi * t / loop_seconds)
    pads = _swelling_filter(pads, theme.pad_cutoff_hz,
                            0.62 * slow + 0.22 * quick, sr)
    subs *= 0.5 + 0.5 * _pump(total, beat, theme.pad_pump, sr)

    echoes = _ping_pong(plucks, theme.delay_beats * beat, theme.delay_feedback, sr)
    lead_echoes = _ping_pong(leads, theme.delay_beats * beat,
                             theme.delay_feedback * 0.6, sr)
    bus = _pad(max(echoes.shape[1], lead_echoes.shape[1]))
    _add(bus, pads * theme.pad_level, 0)
    _add(bus, plucks * theme.pluck_level * 0.55, 0)
    _add(bus, leads * theme.lead_level, 0)
    _add(bus, shakers * theme.shaker_level * 0.75, 0)
    _add(bus, echoes * theme.pluck_level * 0.55 * theme.delay_mix * 1.6, 0)
    _add(bus, lead_echoes * theme.lead_level * theme.delay_mix, 0)
    bus *= _pump(bus.shape[1], beat, theme.pad_pump, sr)
    wet = _with_space(bus, theme, theme.space, theme.reverb_seconds, "bed-space",
                      sr)
    _add(wet, subs * theme.sub_level * 0.5, 0)
    _add(wet, drums * theme.kick_level * 0.62, 0)
    looped = _fold(_highpass(wet, 40.0, sr=sr), length)
    audio = _to_loudness(looped, theme.bed_lufs, theme.bed_peak_db, sr)
    notes.sort(key=lambda row: (row[0], row[2], row[1]))
    return Rendered(BED, audio, notes, loop=True)


def sound_names(event: str) -> List[str]:
    """The file stems an event plays, in the order they rotate.

    :param event: one of :data:`EVENTS`.
    :returns: for example ``["click-0", ..., "click-3"]`` for a click.
    :raises KeyError: for an event this module does not know.
    """
    if event == "click":
        return [f"click-{i}" for i in range(CLICK_VARIANTS)]
    if event == "hover":
        return [f"hover-{i}" for i in range(HOVER_VARIANTS)]
    if event in ("run_finished", "run_failed", BED):
        return [event]
    raise KeyError(event)


def render(theme: SoundTheme, name: str, sr: int = SAMPLE_RATE) -> Rendered:
    """Synthesize one named sound of a theme.

    :param theme: the sound set.
    :param name: a stem from :func:`sound_names`.
    :param sr: sample rate.
    :returns: the rendered audio and its note schedule.
    :raises KeyError: for a name no event plays.
    """
    if name.startswith("click-"):
        return _click(theme, int(name.split("-", 1)[1]), sr)
    if name.startswith("hover-"):
        return _hover(theme, int(name.split("-", 1)[1]), sr)
    if name == "run_finished":
        return _run_finished(theme, sr)
    if name == "run_failed":
        return _run_failed(theme, sr)
    if name == BED:
        return _bed(theme, sr)
    raise KeyError(name)


def write_wav(path, audio: np.ndarray, sr: int = SAMPLE_RATE) -> Path:
    """Write stereo float samples as 16-bit PCM, atomically.

    The file appears under its final name only once it is complete, so a
    player that looks while it is being written sees the old file or none,
    never half of one.

    :param path: destination.
    :param audio: shape ``(2, n)``, values in -1 to 1.
    :param sr: sample rate.
    :returns: the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
    frames = np.round(pcm.T * 32767.0).astype("<i2").tobytes()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with wave.open(str(temporary), "wb") as handle:
        handle.setnchannels(int(pcm.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(int(sr))
        handle.writeframes(frames)
    os.replace(temporary, path)
    return path


def read_wav(path) -> Tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV written by :func:`write_wav`.

    :param path: the file.
    :returns: ``(audio, sample rate)``, audio shaped ``(channels, n)`` in
        -1 to 1.
    """
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        rate = handle.getframerate()
        raw = handle.readframes(handle.getnframes())
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32767.0
    return data.reshape(-1, channels).T, rate


def sound_cache_root() -> Path:
    """Where rendered sounds are kept: ``~/.spacr/sounds`` by default.

    :data:`CACHE_ENV` overrides it. Read on every call, so a test that sets
    the variable is honoured without any reload.
    """
    override = os.environ.get(CACHE_ENV, "").strip()
    if override:
        return Path(override)
    return Path.home() / ".spacr" / "sounds"


def theme_fingerprint(theme: SoundTheme, sr: int = SAMPLE_RATE) -> str:
    """A short hash of everything that decides what a theme's files hold.

    :param theme: the sound set.
    :param sr: sample rate.
    :returns: twelve hex digits.
    """
    payload = json.dumps({"synth": SYNTH_VERSION, "rate": int(sr),
                          "theme": dataclasses.asdict(theme)},
                         sort_keys=True, default=list)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def theme_cache_dir(theme: SoundTheme, root: Optional[Path] = None,
                    sr: int = SAMPLE_RATE) -> Path:
    """The folder a theme's files live in, named by key and fingerprint.

    :param theme: the sound set.
    :param root: cache root; :func:`sound_cache_root` when omitted.
    :param sr: sample rate.
    :returns: the folder (not created here).
    """
    base = Path(root) if root is not None else sound_cache_root()
    return base / f"{theme.key}-{theme_fingerprint(theme, sr)}"


def _forget_other_versions(theme: SoundTheme, keep: Path) -> int:
    """Delete this theme's folders from earlier fingerprints.

    Only folders named ``<key>-<twelve hex digits>`` beside ``keep`` are
    touched, so nothing that is not a sound cache can be removed.
    """
    removed = 0
    prefix = f"{theme.key}-"
    try:
        siblings = list(keep.parent.iterdir())
    except OSError:
        return 0
    for folder in siblings:
        name = folder.name
        tail = name[len(prefix):]
        if (folder == keep or not folder.is_dir()
                or not name.startswith(prefix) or len(tail) != 12
                or any(ch not in "0123456789abcdef" for ch in tail)):
            continue
        shutil.rmtree(folder, ignore_errors=True)
        removed += 1
    return removed


def ensure_rendered(theme: SoundTheme, names: Iterable[str],
                    root: Optional[Path] = None,
                    sr: int = SAMPLE_RATE,
                    should_stop: Optional[Callable[[], bool]] = None,
                    ) -> Dict[str, Path]:
    """Make sure each named sound exists on disk, rendering what is missing.

    A file already in the theme's folder is trusted: the folder name is the
    fingerprint of everything that decides its contents. Folders left by an
    earlier fingerprint of the same theme are removed once the current one
    is in use.

    :param theme: the sound set.
    :param names: stems from :func:`sound_names`.
    :param root: cache root; :func:`sound_cache_root` when omitted.
    :param sr: sample rate.
    :param should_stop: polled between sounds; returning True abandons the
        rest, so a render can be cancelled at the next file boundary.
    :returns: stem -> path, for every file that exists afterwards.
    """
    folder = theme_cache_dir(theme, root, sr)
    ready: Dict[str, Path] = {}
    wrote = False
    for name in names:
        path = folder / f"{name}.wav"
        if path.is_file():
            ready[name] = path
            continue
        if should_stop is not None and should_stop():
            break
        write_wav(path, render(theme, name, sr).audio, sr)
        ready[name] = path
        wrote = True
    if wrote:
        _forget_other_versions(theme, folder)
    return ready
