================================================================================
UPDATING THE spaCR TUTORIALS
================================================================================

Live site:  https://einarolafsson.github.io/spacr/tutorials/
Media host: https://huggingface.co/datasets/einarolafsson/spacr-tutorials
Videos:     https://www.youtube.com/channel/UCft7M--R8Vb2W1Q0sFOSOGA

--------------------------------------------------------------------------------
1. THE ROUTINE UPDATE
--------------------------------------------------------------------------------

After changing anything -- a lesson video, narration, the player, the catalog:

    cd /media/carruthers/mnt3/claude/tutorials
    python tools/publish_tutorials.py

That does all four halves of the job:
  * copies the player (web/) into the docs tree
  * re-encodes changed 4K masters to the published 1440p
  * uploads narration + 4K masters to Hugging Face
  * prints the commit to run

Then commit the part that lives in git:

    cd /media/carruthers/mnt3/claude/repo
    git add -A docs/source/_extra/tutorials
    git commit -m 'tutorials: refresh lesson media' && git push

Narration and the 4K video go live the moment the upload lands -- no deploy
needed. The 1440p video and the player need the push, and only a push to
`main` publishes (see section 6).

Useful flags:

    --lessons 07_mask,24_plaque   only these lessons
    --skip-hf                     docs tree only, do not touch the media host
    --force-encode                re-encode even if the master looks unchanged
    --youtube-masters DIR         also build narrated 4K files to upload

Re-running is cheap and safe. Encoding is skipped when a master's size and
mtime are unchanged (tracked in docs/source/_extra/tutorials/production/
.encode-manifest.json), and the Hugging Face upload hashes before it
transfers. A no-op run takes seconds.

--------------------------------------------------------------------------------
2. THE ONE RULE THAT WILL BITE YOU
--------------------------------------------------------------------------------

    EDIT  /media/carruthers/mnt3/claude/tutorials/web/
    NEVER /media/carruthers/mnt3/claude/repo/docs/source/_extra/tutorials/

The repo copy is derived output. publish_tutorials.py overwrites it every
run. Editing the repo copy appears to work -- it even deploys -- and is then
silently reverted by the next publish, with no error and no conflict.

This already happened once: the YouTube links, the quality selector and the
Hugging Face roots were all written into the repo copy first and had to be
moved back into web/.

Files that flow web/ -> repo:
    index.html  styles.css  app_v2.js  voice_catalog.js  lesson_catalog.js
    youtube_links.js  logo_spacr.png  favicon.svg  TUTORIAL_MEDIA_NOTICE.md
    fonts/  catalog/

index.html is rewritten in transit: web/ uses data-production-root="../production"
so it can be previewed in place; the published copy uses "production".

--------------------------------------------------------------------------------
3. WHERE THE MEDIA LIVES, AND WHY IT IS SPLIT
--------------------------------------------------------------------------------

Three homes, because one has a hard cap.

  docs/source/_extra  (GitHub Pages -- 1 GB HARD LIMIT)
      player, posters, 1440p video.  ~185 MiB.  committed to git.

  Hugging Face dataset
      all 54 narration voices + timing sidecars, and the 4K masters.  ~3.5 GB.
      NOT in git.  Uploaded by publish_tutorials.py.

  YouTube
      one narrated 4K cut per lesson, linked from each lesson page.
      Uploaded by hand.  Ids live in web/youtube_links.js.

Why narration is not on Pages: all 54 voices are 2,662 MiB. While the site
carried the audio it could only publish ONE voice per language, so 27 of the
28 English voices were listed nowhere and reachable never. Moving the audio
to the host removed the constraint instead of rationing it.

Pages is the dangerous one. It does not fail loudly when a site exceeds 1 GB
-- it refuses the deployment and keeps serving the last build that fitted,
which is indistinguishable from a site that simply did not rebuild. That is
exactly how the tutorials appeared frozen for four days.

Check the budget any time:

    cd /media/carruthers/mnt3/claude/repo
    python tools/docs_media_budget.py --report

Guard rails, in tools/docs_media_budget.py and tests/test_docs_media_budget.py:
    PUBLISHED_MEDIA_CEILING = 300 MiB   payload ceiling
    a second test keeps the whole site under 85% of the 1 GB limit

--------------------------------------------------------------------------------
4. CHANGING A LESSON VIDEO
--------------------------------------------------------------------------------

Replace the 4K master:

    /media/carruthers/mnt3/claude/tutorials/production/<lesson>/video/<lesson>_silent.mp4

then run publish_tutorials.py. It re-encodes to 1440p and uploads the 4K.

The master MUST be silent (no audio stream) and MUST keep the same timeline
as its narration. The player overlays narration on silent video and maps the
two timelines onto each other at a continuous playback rate, so:

    * do not change the frame rate
    * do not trim
    * do not add an audio track

The encoder is scale-only and refuses to publish a file whose frame count
changed -- a dropped or duplicated frame desyncs every voice in every
language. Several lessons are variable frame rate (05_home is 147 frames
across 48.8 s while declaring 30 fps), which is why -vsync 0 is used.

If the video's LENGTH changes, the narration must be re-rendered too, or
every voice will drift.

--------------------------------------------------------------------------------
5. NARRATION AND VOICES
--------------------------------------------------------------------------------

Rendered with Kokoro-82M via tools/render_all_voices.py into:

    production/<lesson>/audio/<language>/<voice>.m4a    the narration
    production/<lesson>/audio/<language>/<voice>.json   word timings

Both files must exist for a voice. A .json with no .m4a gives the player
timings for narration it cannot load; an .m4a with no .json plays with no
scene highlighting. Either is a broken lesson.

The picker is built from web/voice_catalog.js. Every voice listed there must
exist on the media host, or the picker offers a voice whose audio 404s.
After adding voices: update voice_catalog.js, then publish.

There is no per-language limit any more -- the host has room. Currently:
    en 28,  zh-CN 8,  ja 5,  hi 4,  es 3,  pt-BR 3,  it 2,  fr 1   = 54

--------------------------------------------------------------------------------
6. HOW IT DEPLOYS
--------------------------------------------------------------------------------

.github/workflows/docs.yml publishes GitHub Pages from `main` ONLY. This is
deliberate: nightly pushes used to republish the public docs over main's
build. nightly still BUILDS (so a broken docstring is caught early) but
never deploys.

So: work on nightly, but the tutorials do not change on the public site
until it reaches main.

    git checkout main && git merge nightly && git push origin main

Deploy takes roughly 15-25 minutes. The build stages the media subset, runs
sphinx, and uploads ~237 MiB to Pages.

CAUTION -- do not push to main while a release workflow is running. The
release job commits the installer README and pushes; a push that lands first
makes its push non-fast-forward and the whole release fails after the
installers have already built. This has happened.

--------------------------------------------------------------------------------
7. YOUTUBE 4K LINKS
--------------------------------------------------------------------------------

Build the uploadable files (4K + English narration muxed in):

    python tools/publish_tutorials.py --youtube-masters /path/to/out

Never upload the raw masters -- they have no audio stream at all, so they
would publish 40 silent videos.

Upload them, then put the 11-character video ids in web/youtube_links.js:

    "07_mask": "nlErufRyrtU",

A lesson with an empty id renders no button, so the catalogue can be filled
a few at a time without ever showing a dead link.

ALWAYS verify the id belongs to the lesson you are mapping it to. Pasted
lists have been wrong three times. Check with:

    curl -s "https://www.youtube.com/oembed?url=https://youtu.be/<ID>&format=json"

and read the title back.

--------------------------------------------------------------------------------
8. THE QUALITY TOGGLE
--------------------------------------------------------------------------------

The player offers 1440p (from Pages) and 4K (from the media host). Both are
the same silent cut on the same timeline, so switching quality does not
interrupt narration -- the audio keeps playing and is only re-synced to the
new video clock.

This is why the 4K option is a plain <video src> and not a YouTube embed:
the player drives the element at an arbitrary playback rate to match the
selected voice, and the YouTube IFrame API only accepts a fixed set of
rates. YouTube also keys alternate audio by LANGUAGE, not by voice, so the
54-voice picker could never work there. Hence: YouTube for a 4K link,
Hugging Face for 4K playback.

Roots are declared on <html> in web/index.html:
    data-production-root  video, posters, captions   (the site itself)
    data-audio-root       narration + timings        (the media host)
    data-video4k-root     4K masters                 (the media host)

They must match NARRATION_HOST in tools/docs_media_budget.py and HF_DATASET
in tools/publish_tutorials.py.

--------------------------------------------------------------------------------
9. VERIFYING A DEPLOY
--------------------------------------------------------------------------------

    B=https://einarolafsson.github.io/spacr/tutorials

    # the player picked up the new build
    curl -s "$B/youtube_links.js" | md5sum
    md5sum docs/source/_extra/tutorials/youtube_links.js

    # narration is NOT on Pages (should be 404 -- it comes from the host)
    curl -sI "$B/production/07_mask/audio/en/af_heart.m4a" | head -1

    # the media host answers cross-origin, with range support
    curl -sI -H "Origin: https://einarolafsson.github.io" -H "Range: bytes=0-99" -L \
      https://huggingface.co/datasets/einarolafsson/spacr-tutorials/resolve/main/07_mask/audio/en/af_heart.m4a \
      | grep -iE "HTTP/|access-control-allow-origin|content-range"

Expect 206 with an access-control-allow-origin echoing the site origin.
Range support is not optional -- audio seeking and narration/video sync both
depend on it.

Browsers cache the old build aggressively. Hard-refresh (Ctrl+Shift+R)
before concluding a deploy did not work.

--------------------------------------------------------------------------------
10. THINGS THAT HAVE ALREADY GONE WRONG
--------------------------------------------------------------------------------

* Editing the repo copy instead of web/. Silently reverted on next publish.

* Pages refusing an oversized deploy without saying so. The site served a
  four-day-old build while every push looked green.

* Running ffmpeg inside a `while read` loop without -nostdin. ffmpeg ate the
  loop's input and silently skipped most of the list -- 2 of 40 files
  processed, exit code 0.

* CFR conversion on variable-frame-rate lessons, which moved their duration.

* A run of pushes queueing a full test suite per commit until ~45 jobs were
  queued, zero running, and the docs deploy starved behind them. Fixed with
  per-ref concurrency in tests.yml and compat-matrix.yml, but it is worth
  watching after a batch of commits:

      gh run list --repo EinarOlafsson/spacr --limit 30

* Pushing to main during a release run, which broke the release's own push.

* A >300-file push not triggering a paths-filtered workflow. release.yml
  watches setup.py; a push touching 4618 files did not match it, so version
  1.5.0.0 never built or published anything. Keep release commits small.
