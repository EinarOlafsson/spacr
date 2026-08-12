# Diagnostics and release QA

Work from a single failing sentence and voice first, but validate fixes across
the complete affected matrix. The earliest incorrect representation identifies
the layer to repair.

## Diagnostic ladder

| Symptom | Inspect | Correct response | Reject |
| --- | --- | --- | --- |
| Technical term is wrong | Display text, synthesis-only text, dialect, recorded phonemes, catalog-wide unknown-token scan | Add or refine one central cohesive whole-word form; cover US and UK; add focused and full-catalog tests | Caption misspellings, per-voice text forks, arbitrary letter spacing |
| Ending sounds cut off | Final word's phonemes, decoded last consonant/fricative, activity boundary, kept 180 ms tail, sidecar sentence end | Fix missing phonemes or activity detection; retain active samples and conservative tail | Vowel shortening, word splicing, time compression, crossfading active speech |
| Scene sounds rushed or dragged | Uniform voice speed, phone rate, punctuation, sentence split, internal pauses, adversarial listening | Adjust one base speed for that voice only after broad measurement; improve sentence structure | Per-scene WPM speed multipliers |
| Pause feels robotic | Sentence audible gap, explicit pause, retained edges, scene transition, punctuation | Tune assembly constants as one versioned contract and rerender all dependent media | Editing individual pauses into rendered words |
| Long pauses occur inside words or clauses | Decoded PCM pause distribution across many scenes for that voice | Confirm text/phonemes and uniform speed, then retire a persistently pathological voice | Destructive silence excision presented as natural speech |
| Click or pop is audible | Decoded PCM around every sentence/scene boundary, sample discontinuity, source render, mastering output | Preserve contiguous slices, locate the stage introducing the transient, rerender from clean PCM | Hiding clicks with untracked fades or splices |
| Narration and visual click diverge | Sentence/scene timing, rendered frame at click transition, spotlight/pointer target | Rerender timing-dependent visual master; inspect start/mid/end and each actual click | Stretching or seeking narration to follow video |
| Caption drifts or changes spelling | Display sentence, sidecar sentence intervals, VTT, browser splitter | Keep display text exact and use sidecar intervals; add splitter regression | Captions made from synthesis spellings |
| File plays but release verifier fails | Strict decoder stderr/exit, sample count, timing, media bytes/hash, fingerprint freshness | Regenerate atomically from frozen inputs and recheck the full track | Accepting a zero decoder exit with warnings or updating a hash by hand |

## Independent naturalness audit

For each supported English voice, sample every risky term and enough surrounding
context to hear stress and endings. Include singular/plural and verb/noun forms,
US/UK voices, sentence-final uses, and consecutive technical terms.

Record at least:

- display text, synthesis text, and non-empty phonemes;
- expected and decoded duration/sample count;
- first/last activity, retained lead/tail, and sentence-to-sentence gap;
- longest internal pause and whether it follows punctuation;
- end-window energy or an equivalent final-phone check;
- boundary discontinuities/click candidates;
- per-scene effective speed and confirmation it equals the voice base speed;
- listening disposition and any retirement decision.

Automated WPM, RMS, tail energy, or pause thresholds are candidate finders. They
do not independently prove natural speech. Listen to every flagged case and a
stratified clean sample.

## Focused test pass

From the tutorial publishing workspace, run the executable contracts before a
large render:

```bash
python -m pytest \
  tests/test_pronunciation.py \
  tests/test_narration_audio.py \
  tests/test_verify_audio_release.py \
  tests/test_visual_master.py \
  tests/test_web_player_contract.py
```

Add tests for every newly repaired token, sentence-split edge case, trim
boundary, fingerprint input, or player-clock regression.

## Strict release pass

Derive the expected lesson/voice inventory from the current catalog and
renderer. Force-render changed English voices; do not let stale-looking output
skip merely because a target file exists. Then run:

```bash
python tools/verify_audio_release.py --workers 16 --freshness-languages en
```

The current verifier and authoritative contract require:

- mono, 24 kHz AAC for supported narration tracks;
- strict decode with no warnings and no nonzero exit;
- decoded duration/sample count within 75 ms of the timing sidecar;
- exact catalog scene count, contiguous monotonic scene/sentence timing, a
  silent end hold, and credible activity in every scene;
- no unintended silence interval of eight seconds or longer;
- media byte count and SHA-256 equal the sidecar;
- current render inputs and fingerprint equal a fresh reconstruction;
- non-empty per-sentence phonemes for fresh English media;
- true peak at or below -1 dBFS after AAC decode.

The eight-second dead-air threshold is a gross-corruption gate, not a
naturalness allowance. A reported random pause shorter than that still requires
a focused waveform/timing regression and explicit listening approval. If a
voice repeatedly inserts such clause-internal pauses after text, phonemes, and
uniform speed are correct, retire it even though the coarse dead-air check
passes.

The current mastering path begins with a -3 dBTP render target. If decoded AAC
exceeds the -1 dBFS ceiling, retry from the original synthesized PCM at -5 dBTP
and use the documented limiter fallbacks; never repeatedly transcode AAC. A
complete zero-change peak pass must precede publication.

Also run the tutorial release verifier and publisher/media tests selected by the
current workspace. In the spaCR repository, run the media budget report and its
tests before staging the derived documentation bundle:

```bash
python tools/docs_media_budget.py --report
```

Inspect `--help` for the current publisher and use its documented lesson filter
and upload-disable mode for the first local pass rather than copying a historical
command line.

## Visual and mobile pass

Rerender every timing-dependent silent master from the final default/master
voice timing. Inspect scene start, midpoint, and end frames plus every click
transition. A click pointer/ripple must represent an actual scripted click;
passive explanation uses a spotlight.

In a phone-sized browser, check:

1. narration remains at 1x and leads the silent video;
2. the final visual frame parks while narration finishes;
3. native Pause pauses both clocks;
4. replay resets both clocks and does not inherit the parked state;
5. seeks map through scene timings without corrupting narration position;
6. English captions use exact sidecar sentence intervals and display spelling;
7. resizing and keyboard/player controls remain usable.

## Publication pass

Publish locally with upload disabled. Confirm the selected inventory contains
one media file and one timing sidecar for every supported lesson/voice pair,
plus the expected visual masters and captions. Verify strict hashes, byte counts,
CORS/range behavior, and staged player playback.

Only then upload the exact verified artifacts and generate the derived
repository bundle. Repeat pronunciation, pause/end/replay, caption, range, and
hash checks against live URLs. Publication success is not inferred from an
upload command's exit code.
