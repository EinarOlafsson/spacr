---
name: natural-voice-narration
description: Build, repair, and release natural English tutorial narration. Use when generating or auditing TTS media, fixing technical-word pronunciation or clipped endings, tuning cadence without scene-level speed swings, retiring unsuitable voices, synchronizing captions and silent video, or verifying narration artifacts before publication.
---

# Natural Voice Narration

Treat naturalness as a reproducible release contract spanning source text,
phonemes, rendering, signal boundaries, timing, media identity, and playback.
Use the repository's current configuration as authority; historical voice and
track counts in audits can become stale.

## Load the right reference

- Read [references/naturalness-contract.md](references/naturalness-contract.md)
  before changing pronunciation, rendering, trimming, pauses, speed, or voice
  support.
- Read [references/diagnostics-and-release.md](references/diagnostics-and-release.md)
  when diagnosing a defect or preparing a release.

## Workflow

### 1. Freeze scope and ownership

1. Read `instructions/open/48_optimize_the_tutorials.txt`, especially its latest
   authoritative English narration contract and ownership note.
2. Locate the publishing workspace named there. Treat its `production/`,
   `catalog/`, and `tools/` as source; never hand-edit the derived
   `docs/source/_extra/tutorials/` bundle.
3. Inspect the current English catalog, renderer language/voice matrix, cadence
   profile, tests, and publisher before quoting counts or changing media.
4. Preserve any concurrent-session ownership boundary. Audit read-only if the
   workspace is assigned elsewhere.

### 2. Separate what readers see from what TTS receives

Keep captions and catalog narration in official written form. Produce a separate
synthesis-only string through the central pronunciation layer. Apply each
English fix to both US and UK profiles when their phonemes differ, and test the
whole inflected word when its ending matters.

Prefer cohesive technical tokens or a one-token phoneme override. Do not insert
spaces that give every letter or syllable a separate stress unless the term is
intentionally lettered. Render English sentence by sentence while retaining the
unchanged display sentence for captions.

If the current lexicon has no approved form, do not invent phonemes from
spelling. Compare authoritative pronunciation evidence with renders in the
actual engine and dialect, measure the resulting phone sequence and duration,
listen in context, and add the chosen form plus a regression together.

### 3. Establish a natural baseline

Render a small, representative set containing technical terms, fricatives,
plural endings, short sentences, long sentences, and sentence transitions.
Choose one measured base speed per voice, then keep that speed uniform across
all scenes. Use WPM as an audit signal only; judge phone rate, word endings,
pauses, and listening evidence together.

Inspect the current timing constants in the publishing workspace instead of
treating measured defaults as universal TTS advice. Do not enable scene-level
word-WPM scaling to make individual scenes fit.

### 4. Diagnose before repairing

Classify the defect as pronunciation, missing phonemes, clipped ending, edge
trim, internal pause, cadence, boundary click, decode, timing, caption, or
player-clock behavior. Compare display text, synthesis text, recorded phonemes,
timing metadata, and decoded PCM for the same sentence.

Fix the earliest layer that is wrong. Never repair speech by shortening a
rendered vowel, splicing a word, cross-fading active speech, or applying a
different scalar speed to individual scenes. If a voice keeps pathological
internal pauses or damaged endings after correct text and a reasonable uniform
speed, retire it from the supported matrix and leave existing artifacts outside
that matrix untouched.

### 5. Verify the complete affected matrix

Run focused pronunciation and assembly tests first, then the strict release
verifier for every affected supported track. Require phoneme, ending, tail,
pause, click/pop, decode, sample-count, timing, peak, dead-air, caption, hash,
byte-count, and fingerprint checks. Listen to adversarial terms and sentence
boundaries across every supported English voice, not only the default voice.

Rerender timing-dependent silent masters from the final master-voice timings.
In a phone-sized browser, verify narration-led playback, native pause, visual
end parking, replay, seeking, and exact English sentence captions.

### 6. Publish reproducibly

Bind every audible input into the render fingerprint and freeze source bytes at
process start. Write media and sidecars to non-publishable `.part` paths;
atomically replace the public targets only after validation. A failed render or
sidecar write must leave the previous known-good target intact.

Publish locally first, inspect the staged player and full artifact inventory,
then upload the exact verified bytes. Recheck hashes, range requests, captions,
and mobile playback against the published URLs before declaring success.

## Release refusal conditions

Do not release while any supported track has an unknown or empty phoneme token,
an incomplete word ending, an unnatural internal pause, scene-specific speed
variation, an audible boundary click, a decoder warning, a media/sidecar/hash
mismatch, non-monotonic timing, stale fingerprints, incorrect caption text, or
video behavior that pauses or seeks the narration incorrectly.
