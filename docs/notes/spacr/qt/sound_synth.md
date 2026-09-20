# Notes from `spacr/qt/sound_synth.py`

The module carries no comments; this is where its reasons live. Item 427,
part A, built 2026-09-19.

## What was asked, and how each clause became a number

"The theme should sound like melodic space house" (maintainer, 2026-09-19),
with the accepted recommendation: warm detuned-saw pads, plucked arpeggio
notes with a dotted-eighth delay, a soft sub, a minor key; clicks and
hovers as short in-key plucks; run-finished a rising arpeggio resolving to
the tonic with a pad swell; run-failed a falling minor figure; quiet, never
jarring. Artists named as the reference: Worakls, N'to, Rivière Monk,
BIRRD. No sample, recording or preset from anyone is used -- everything is
arithmetic on sine and sawtooth waves.

`tests/qt/test_the_sounds_are_melodic_space_house.py` measures each clause
on the rendered samples. Measured on the reference set, `ORBIT`, when it was
written:

| sound | length | peak | RMS | spectral centroid |
|---|---|---|---|---|
| click (4 variants) | 1.26-1.29 s | -12.0 dBFS | -29 dBFS | 458 Hz |
| hover (4 variants) | 0.57-0.59 s | -20.0 dBFS | -36 dBFS | 672 Hz |
| run finished | 4.89 s | -6.0 dBFS | -19.1 dBFS | 409 Hz |
| run failed | 4.80 s | -7.0 dBFS | -19.3 dBFS | 256 Hz |
| music bed (8 bars, 122 BPM) | 15.74 s | -13.0 dBFS | -23.1 dBFS | 334 Hz |

The click pitches measured from the audio were 440.6, 523.5, 659.5 and
494.6 Hz (A4, C5, E5, B4) and the hovers E5, G5, A5, C6: every one within
0.03 of a semitone of a note of A natural minor. The click's echo lands
0.369 s after the attack, a dotted eighth at 122 BPM. The loop seam of the
bed jumps by 0.013, against a median sample-to-sample step of 0.005 and a
99th percentile of 0.025: no audible click where it repeats.

## Why the pluck is additive rather than Karplus-Strong

A melodic-house pluck is a sawtooth through a low-pass filter whose cutoff
falls fast. Harmonics that each decay faster the higher they are sound the
same and are exact in pitch; a Karplus-Strong string rounds its delay to
whole samples, which puts high notes audibly out of tune at 48 kHz.

## Why the pad voicings are root position

The first version voiced each chord's notes to the octave nearest a centre
note. That put E beside F in Fmaj7 and B beside C in Cmaj7, a semitone
apart, and five detuned saws per note turn a semitone into a smear. Stacked
thirds in root position (`_rooted`) keep every neighbouring pair three or
four semitones apart, and the whole block moving in parallel from chord to
chord is how the genre's pads move anyway. The pad filter is a fourth-order
low-pass at 1.1 kHz breathing up to 2.2 kHz; the first version's
second-order filter at 1.5 kHz was measurably brighter than "warm".

## Why the bed is levelled after it is folded, and filtered before

The bed is rendered for eight bars plus its reverb and echo tails, and the
tails are folded back onto the start, which is what sounds under bar one
the second time round. Everything LINEAR (delay, reverb, filters, the
28 Hz high-pass) is done before the fold, on the whole unfolded signal:
folding a filtered signal is the same as filtering the endless loop, while
filtering a folded one restarts the filter at the seam. Everything
MEMORYLESS (the `tanh` peak rounding, the gain) is done after the fold,
because it cannot put a seam back. The bed has no fade-in; a one-shot has
2 ms.

## Cost

Rendering the whole reference set on the workstation: 506 ms for the first
click (it pays for importing `scipy.signal`), 15-19 ms for each later click,
6 ms per hover, 255 ms for run finished, 119 ms for run failed and 760 ms
for the eight-bar bed; peak resident memory of the probe 317 MB, most of it
numpy and scipy themselves. It happens once per set per machine, on the
audio thread, and the files are kept under `~/.spacr/sounds/<key>-<hash>`.

## The cache

The folder name carries a fingerprint of `SYNTH_VERSION`, the sample rate
and every field of the theme, so a changed theme or synthesiser can never be
answered from an old file. Older folders of the same theme are removed once
a new one is written; only folders named `<key>-<12 hex digits>` are ever
touched. Files are written under a temporary name and renamed into place.

--------------------------------------------------------------------------------

## 2026-09-19 — part B: the bed became a composed piece

Part A's bed was eight bars of the same four chords with everything
playing: a phrase, and a phrase heard for the eleventh time is a phrase
somebody turns off. Part B's brief was specific — "a seamless loop (e.g.
32 bars at 120-124 BPM) ... key/mode, chord progression (minor-key, e.g.
i-VI-III-VII), arpeggio pattern, pad timbre, sub, a soft four-on-the-floor
kick and shaker that can be turned down to near-silent, filter movement,
delay/reverb ... It must sound musical: check your render by analysing it
(spectral balance, no clipping, loudness around -18 LUFS-ish, loop seam
click-free)."

`SYNTH_VERSION` went to 2, so every cached version-1 folder retires itself
the first time the new bed is asked for.

### The arrangement, and where the seam is put

`BED_SECTIONS` gives the thirty-two bars four eight-bar sections — drift,
pulse, lift, return — and `bed_plan` ramps every part to its section's
level over the section's FIRST HALF. The first section ramps from the
LAST one's levels, which is what makes the plan a circle rather than a
long line: bar one continues bar thirty-two.

**The seam is placed, not patched.** The loop opens and closes on the
quietest section, so the join falls where the music has least to give
away; `_fold` carries the reverb and delay tails over it; and
`_to_loudness` is two memoryless operations, so the level can be set after
the fold without putting a step back in. Measured on the reference bed:
the join steps by 0.0024 and 0.0034 (left, right) against a 99.5th
percentile sample-to-sample step of 0.0397 — the loudest transient in the
loop moves eleven times further than the join does.

Both filter breaths are periodic over the whole loop (one opening across
the thirty-two bars, one four times), so a filter can move for a minute
and still arrive back where it started. The spectral centroid measured in
two-second blocks moves by a factor of more than 1.25 and returns.

### The drums, and "turned down to near-silent"

`kick_level` and `shaker_level` are theme fields. Below `PART_FLOOR`
(0.05) the part is not synthesized at all rather than mixed quietly: a
kick at two per cent is a sample nobody can hear and a transient the
loudness normaliser still has to make room for. At 0 the bed is exactly
the pad-and-arpeggio piece part A shipped, which is what a theme that has
to sit under a talk should ask for.

The kick is a sine falling from about 150 Hz to 48 Hz in thirty
milliseconds with a very short knock an octave and a half above it, then
low-passed at 1.2 kHz: 98.4 % of its energy is below 160 Hz. It stays DRY —
low frequencies through a three-second tail are what turn a quiet bed into
a rumble.

The shaker's offbeat is the loud one, because that is where the shaker of
house music lives and a pattern with every hit at the same weight reads as a hiss.

**THE SHAKER WAS INAUDIBLE ON THE FIRST PASS AND THE TEST THAT CAUGHT IT
IS THE ONE WORTH KEEPING.** Mixed at 0.22 and high-passed at 5.2 kHz it
added 0.8 dB to the 4-12 kHz band over a drumless render — a setting that
does nothing, which is worse than no shaker. At 0.75 and 3.8 kHz it adds
6.2 dB. `test_the_drums_are_audible_when_they_are_asked_for` renders both
ways and requires a factor of four.

### Loudness, because a peak is not a loudness

`loudness_lufs` is the gated integrated measurement of ITU-R BS.1770-4:
K-weighting, 400 ms blocks overlapping by three quarters, an absolute gate
at -70 LUFS and a relative gate 10 LU under the mean of what survives.

Checked against the standard's own calibration tone before anything was
measured with it: 1 kHz at full scale in one channel is -3.01 LKFS.
**The test for that was written wrong first**, from "K-weighting is flat
at 1 kHz", which gives an expected value 0.7 dB out — the filter has
+0.691 dB of gain at 1 kHz by construction and it cancels the -0.691
offset exactly. The code was right and the arithmetic in the test was
wrong.

Measured on the reference bed, 32 bars at 122 BPM:

| | |
|---|---|
| length | 62.95 s, 12.1 MB as 48 kHz 16-bit stereo |
| render time | 3.3 s (once, cached) |
| programme loudness | -18.00 LUFS |
| peak | -9.13 dBFS, no sample at or past full scale |
| low (35-90 Hz) against mid (150-1500 Hz) | 0.44 |
| loop seam | 0.0024 / 0.0034 against a p99.5 step of 0.0397 |
| by section (LUFS) | drift -18.8, pulse -18.0, lift -17.3, return -18.0 |

**The spectrum is deliberately bottom-heavy and was nearly too much so.**
The first render put 30 % of the power below 60 Hz and 0.01 % above 4 kHz,
with `low/mid` at 0.82 — outside what part A's own `test_the_sub_is_soft`
allows. Raising the final high-pass from 28 Hz to 40 Hz, trimming the sub
mix from 0.6 to 0.5 and lifting the shaker brought it to 20 % below 60 Hz
and `low/mid` 0.44.

**Nobody has listened to it.** The numbers say what was asked for; whether
it sounds like Worakls or like a test tone is still the maintainer's ear.

**The copy did not say there were drums, and now it does.** Found on
review before part B landed. `ORBIT.description` — the Sound-set combo's
tooltip — and the Music bed row's own tooltip both still described the
version-1 bed: pads, a plucked arpeggio with a dotted-eighth echo and a
soft sub. Version 2 has `kick_level` 0.5 and `shaker_level` 0.34 under
that, and the `_bed` docstring it replaced said outright "No drums: this
plays under somebody's work, not over it". A user who switched the bed on
under the old sentence would have got a kick under their work with no
notice, so both strings now name the kick and the shaker and say that they
come and go with the arrangement.

What they still cannot do is turn them down on their own: `kick_level` and
`shaker_level` are theme fields, which is what lets part C's ten sound sets
be a table of numbers, and the only control a user has is the bed's own
volume and the bed's own switch. If a per-user drum level is wanted it is
part C's to add, and it is written down here rather than left to be
discovered.

**The shaker plays eighths, and three docstrings said sixteenths.** `_bed`
emits `for hit in range(8)` over a four-beat bar — 208 hits over 26 sounding
bars, 8 a bar — and the part-B test is named
`test_the_shaker_is_on_the_eighths_and_the_offbeat_is_the_loud_one`. Part C
is meant to be written from the `:param:` docs of `SoundTheme`, so a wrong
note value there is a wrong note value in ten sound sets.
