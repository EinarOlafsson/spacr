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
