# Tutorials v2 — silent video, client-side voice

**Deferred by explicit instruction: do this LAST.** Recorded now so the design is
not lost; no work has started.

The HTML shell is liked and stays. What changes is how the media is produced and
delivered, and how much of the app the lessons actually show.

---

## 1. The architecture change

**Today:** one pre-rendered audio file per lesson × language × voice, shipped as
static assets, plus a video that carries the narration timing.

**Proposed:** one **silent** video per lesson with a caption/timing track, and the
voice synthesised **in the browser** from that track at play time. Language and
voice become a runtime choice rather than a build-time multiplication.

### The size argument, measured

| Asset | Now | After |
|---|---|---|
| audio | **674 MB** (4,320 files) | 0 |
| video | 36 MB (40 files) | ~36 MB |
| posters | 12 MB | ~12 MB |
| **tutorial library** | **723 MB** | **~48 MB** |
| **whole built site** | **876 MB** | **~200 MB** |

Audio is **93% of the payload**. This is also the fix for the GitHub Pages
**1 GB** ceiling the site is currently at 88% of — a wall that would otherwise be
hit mid-lesson-batch with no warning, and the reason `docs/` makes a `git clone`
cost 1.45 GiB.

It also removes the combinatorial build: adding a 9th language today means
rendering 40 more lesson-audio sets. After, it means adding a voice entry.

### Voice delivery — the open decision

Two options, and they trade differently:

1. **Web Speech API** (`speechSynthesis`). Zero bytes shipped, works offline
   once the page is cached, but the available voices are the *operating
   system's* — so quality and language coverage vary per visitor and cannot be
   controlled or previewed. A reviewer on a bare Linux box may get nothing
   usable.
2. **A WASM/ONNX model in the browser** (Kokoro — already a dependency here as
   `kokoro-onnx` — or Piper, via `onnxruntime-web` or `transformers.js`). One
   model file, cached once, shared by all 40 lessons in every language it
   supports. Consistent everywhere, at the cost of a one-time download
   (~80 MB quantised, versus 674 MB today).

**Recommendation: option 2 with option 1 as fallback.** Ship the ONNX voice,
fall back to `speechSynthesis` where WASM is blocked or the download is refused,
and captions alone where neither works. Captions are the accessibility floor and
must never depend on either.

The current media is marked non-commercial in
`docs/source/_extra/tutorials/TUTORIAL_MEDIA_NOTICE.md`; a re-render is the
natural moment to settle that licence.

---

## 2. Production quality — what the lessons should show

The requested changes, in the order they raise the ceiling:

**Highlight and dim.** When a step refers to part of the screen, dim everything
else and highlight the target. `spacr/qt/tutorial/engine.py` already has
`_draw_highlight_on` and a newer `_draw_spotlight_on` — the spotlight is the
shape wanted; it needs to become the default and to animate the transition
rather than cutting.

**A visible pointer on click.** A cursor glyph that moves to the target and
shows a click ripple. Without it the viewer sees state change with no cause,
which is the single most common complaint about screen-recorded tutorials.

**Real data wherever possible.** Synthetic demo data teaches the mechanics but
not the judgement — what a good mask looks like versus an over-segmented one.
Depends on the demo repair already committed, and on choosing a small public or
publishable field set that can ship without licence problems.

**Live views of settings and modules.** Show the actual `AppScreen` with its
settings panel and live preview responding, not a static screenshot of one. This
is the item that makes the lessons age with the software instead of against it:
a re-render picks up the current UI automatically.

---

## 3. Sequencing when it starts

1. Caption/timing track format, and the renderer emitting silent video against it.
2. Voice runtime in the page (model + fallback chain), behind a feature flag so
   the current audio can stay live until the new path is proven.
3. Re-render the 40 lessons silent; delete the 674 MB of audio in the same
   commit that proves the new path works, not before.
4. Spotlight-by-default, pointer, and click ripple in the recorder.
5. Real data and live views, lesson by lesson.

Steps 1-3 are the ones that pay for themselves immediately: they cut the repo,
clear the Pages ceiling, and make every later language free.

---

## 4. Known issues to fold in

- **Lesson 1 reads "In production — coming soon"** on the live site while
  `production/01_pypi_github/` has video, audio and a poster on disk. Either a
  catalog status flag was never flipped or the published copy is stale.
- `qt/tutorial/scripts.py` narrates **"cyto for cells"**, which is wrong under
  Cellpose 4 — every model resolves to `cpsam`.
- Every lesson has exactly 5 scenes and one scene opening is byte-identical
  across all 40; 38 of 40 share the same objective line. A re-render is the
  moment to give them real per-lesson structure.
- Lesson titles are untranslated (`07_mask` is "Mask" in every language) and NLLB
  artefacts survive in some descriptions.
