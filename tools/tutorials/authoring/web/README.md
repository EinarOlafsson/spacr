# spaCR tutorial portal

This is a dependency-free static tutorial framework. It reads the videos,
posters, scene descriptions and timing metadata from `../production/`.

Run it from the tutorial workspace so browsers can load the lesson metadata
and video captions:

```bash
cd /mnt/firecuda2/Claude/toxoplasma_projects/tutorials
python -m http.server 8765
```

Then open:

```text
http://localhost:8765/web/
```

The portal includes:

- the complete ordered 73-lesson learning path;
- one 4K silent master for every tutorial;
- dynamically generated WebVTT captions in 14 languages;
- language-first selection for 50 curated Kokoro voices across eight languages;
- a silent-master fallback for narration tracks that are not rendered yet;
- chapter seeking and a searchable transcript;
- stored watch position and completion state;
- previous, next and continue controls;
- deep links such as `#lesson=07_mask`; and
- responsive desktop and mobile navigation.

`voice_catalog.js` is the player catalogue. `app_v2.js` combines the selected
audio and timing manifest with the shared silent master, including scene-level
time mapping for voices whose delivery is faster or slower than the reference.
External narration is fetched as one complete file and attached to the hidden
audio element through a Blob URL. This keeps phone playback independent of
host-specific MP4/M4A byte-range behavior; release tracks are deliberately
small enough for that download strategy (currently under 1.5 MB each).
During playback the narration is the master clock and remains at its natural
rate. Scene-duration mapping, drift correction, and any corrective seeks are
applied only to the silent video. Narration is sought only when the user seeks
or when a lesson/voice is loaded or resumed; repeatedly seeking audible audio
from coarse mobile video time updates produces skips and stretched syllables.
