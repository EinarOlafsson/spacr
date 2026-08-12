# Naturalness contract

This reference records the current, measured English narration behavior. It is
not a general prescription for every synthesizer. Before changing thresholds,
inspect the live values and tests in the tutorial publishing workspace named by
`instructions/open/48_optimize_the_tutorials.txt`.

## Source map

Relative to that publishing workspace:

- `catalog/lessons_en.json`: display narration and scene inventory.
- `tools/pronunciation.py`: central display-to-synthesis transformations and
  US/UK technical-word phonemes.
- `tools/narration_audio.py`: voice speeds, sentence splitting, edge trimming,
  pauses, source freezing, fingerprint helpers, and atomic writes.
- `tools/english_scene_cadence.json`: measured scene-rate audit; currently audit
  data, not an enabled scene-speed controller.
- `tools/render_all_voices.py`: supported language/voice matrix and primary
  renderer.
- `tools/verify_audio_release.py`: strict media, timing, activity, freshness,
  and identity gates.
- `web/app_v2.js`: narration-led playback and sentence-caption behavior.
- `tests/test_pronunciation.py`, `tests/test_narration_audio.py`,
  `tests/test_verify_audio_release.py`, and `tests/test_web_player_contract.py`:
  executable contract.

The derived repository bundle is `docs/source/_extra/tutorials/`. Generate it
through the publisher; never repair narration directly there.

## Display text and synthesis text are different products

Captions retain official spelling and punctuation. The TTS engine receives a
separate speech string. This permits `PyPI`, `classifier`, `assay`,
`preprocessing`, `CUDA`, and similar terms to sound correct without misspelling
the captions or accessibility text.

The current approved English PyPI synthesis form is the cohesive `pie-P-I`
token. CPU and NVIDIA remain cohesive dictionary tokens. CUDA uses one cohesive
phoneme override. Whole-word US and UK overrides preserve the unstressed endings
of `assay`, `assays`, `classify`, `classifier`, and `classifiers`.

Rules:

1. Normalize product spelling before transforming it for speech.
2. Keep the transformation central and deterministic for every English voice.
3. Preserve the complete inflected word in a phoneme override when a suffix is
   at risk.
4. Reject unknown and empty phoneme output across the full US/UK catalog.
5. Keep the display sentence alongside its synthesis sentence in timing data.

## Sentence-sized rendering protects cadence and endings

English is synthesized one sentence at a time, then assembled. This gives the
engine a natural prosodic unit without asking one long scene render to invent all
pauses, and it provides exact sentence intervals for captions.

Edge trimming may remove only one contiguous slice of redundant model silence.
It must preserve active samples exactly: no active-speech crossfade, resampling,
splicing, vowel shortening, or time compression.

The current measured defaults in `tools/narration_audio.py` are:

- retain 100 ms before first detected activity;
- retain 180 ms after last detected activity;
- insert 120 ms of explicit silence between sentence clips, producing about
  400 ms from the previous audible ending to the next audible start;
- clamp authored scene holds to 320–520 ms. With retained sentence edges,
  current delivered scene transitions generally target about 730–800 ms.

These values describe this Kokoro tutorial pipeline. Re-measure them if the
model, sample rate, activity detector, mastering, or content changes. The
implementation constants and decoded output win over copied documentation.

The activity detector deliberately uses a low, peak-based threshold so quiet
initial fricatives and final consonants remain active. Tests require the trimmed
output to be a byte-for-byte-equivalent contiguous sample slice.

## Speed is a voice property, not a scene property

Kokoro voices have different native phone rates, so each supported English voice
has one measured base speed. Keep that speed constant for all catalog scenes.

The scene WPM profile in `tools/english_scene_cadence.json` is intentionally
disabled. Independent phone-rate review found that scene-level WPM correction
increased acoustic variation, dragged syllables in some short-word scenes, and
shortened technical endings. Word count does not measure phonetic density,
stress, punctuation, or a voice's internal pauses.

Use WPM to find candidates for listening, not to calculate per-scene synthesis
speed. Review active-span phone rate, pause structure, and ending completeness.

## Retire pathological voices

A scalar speed cannot naturally repair a voice that inserts long, inconsistent
pauses inside otherwise correct sentences. Destructive waveform surgery creates
new rhythm and boundary risks. Remove such a voice from the supported matrix,
keep the reason in tests/configuration, and let the publisher ignore retired
artifacts.

The current snapshot supports 24 English voices and explicitly excludes
`af_alloy`, `af_nicole`, `af_kore`, and `af_nova`. The full current test matrix
is 69 lessons by 50 voices, or 3,450 selected tracks. Always derive these counts
again from `tools/render_all_voices.py` and the current catalog before release.

## Playback contract

Narration at 1x is the master clock. The player adjusts only the silent video to
scene timing. If the visual stream reaches its end first, park it on its final
frame without pausing or seeking the still-audible narration. A native Pause
pauses both clocks, and replay resets both.

English caption cues use exact sentence intervals from timing sidecars. The
browser and renderer sentence splitters must agree on every catalog scene,
including dotted filenames, command labels, abbreviations, and labels such as
`Model B.`

## Reproducibility and atomicity

The fingerprint must bind the exact narration/sentence plan, language and
dialect, pronunciation and cadence sources, effective speed, model config and
weights, voice tensor, library/tool versions, CPU/GPU device, FFmpeg identity,
assembly settings, and mastering settings.

Freeze source bytes at process start. Abort if those sources change during
render or verification. Stage media and sidecars under `.part` names that a
publisher cannot select. Atomically replace public targets only after strict
decode, peak, duration/sample-count, byte-count, and SHA-256 checks succeed.
