"""What the published documentation site is allowed to weigh.

The editable tutorial production library is dominated by one narration
``.m4a`` per lesson x language x voice. The derived library under
``docs/source/_extra`` deliberately is not: copying all narration through
``html_extra_path`` would put the site far past the GitHub Pages **1 GB**
limit — a wall that gets hit mid-lesson-batch, from a build that has never
once warned about it.

``tools/docs_media_budget.py`` now publishes no narration at all: the audio and
its timing sidecars are served from ``NARRATION_HOST`` instead, which is what
lets the picker offer all 50 voices rather than the one per language the site
could afford to carry. This file is the guard on that: it asserts the payload
stays under budget, that the reduction takes only narration (never a lesson, a
video, a caption or a language), and that the catalog the site ships still
offers every voice the host serves.

Read-only and fast: the whole-library assertions ``stat`` files and never
copy one. Staging is exercised against a miniature library built in
``tmp_path``, so hardlinking 743 files is not on the critical path of the
suite.
"""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
BUDGET_PATH = REPO_ROOT / "tools" / "docs_media_budget.py"


def _load_budget():
    spec = importlib.util.spec_from_file_location(
        "spacr_docs_media_budget", BUDGET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


budget = _load_budget()

#: The library is only present in a checkout, not in an installed wheel.
_LIBRARY = budget.extra_root(REPO_ROOT)
requires_library = pytest.mark.skipif(
    not (_LIBRARY / "tutorials" / "voice_catalog.js").is_file(),
    reason="docs/source/_extra is not in this checkout")


@pytest.fixture(scope="module")
def real_plan():
    """``(published, dropped, kept)`` for the shipped library."""
    return budget.plan(_LIBRARY)


def _bytes(paths):
    return sum(path.stat().st_size for path in paths)


# ---------------------------------------------------------------------------
# The budget itself
# ---------------------------------------------------------------------------

@requires_library
def test_the_published_tutorial_library_is_under_its_ceiling(real_plan):
    """The number the whole exercise exists to move."""
    published, dropped, _keep = real_plan
    after = _bytes(published)
    before = after + _bytes(dropped)
    mib = 1024 * 1024

    assert after <= budget.PUBLISHED_MEDIA_CEILING, (
        f"the published tutorial library is {after / mib:.1f} MiB against a "
        f"ceiling of {budget.PUBLISHED_MEDIA_CEILING / mib:.0f} MiB. Either "
        f"the lessons grew or VOICES_PER_LANGUAGE went up; do not raise the "
        f"ceiling without checking the whole site against the 1 GB Pages "
        f"limit first.")
    voice_assets = [
        path for path in published
        if "audio" in path.parts
        and path.suffix.lower() in budget.VOICE_ASSET_SUFFIXES
    ]
    assert not voice_assets, (
        f"the derived tree still publishes {len(voice_assets)} narration "
        "asset(s); narration belongs on the configured host")
    # Older checkouts can still contain the pre-policy narration tree. In
    # those, prove that staging makes a material reduction. The current
    # publisher removes those derived copies entirely, so there is correctly
    # nothing left for the staging filter to drop.
    if dropped:
        assert after < before / 3, (
            f"the filter is not doing its job: {after / mib:.1f} MiB "
            f"published out of {before / mib:.1f} MiB")


@requires_library
def test_the_whole_site_clears_the_pages_limit_with_room(real_plan):
    """Payload plus HTML, against the limit that actually bites.

    The tutorial library is not the whole site: autoapi's HTML, ``_modules``,
    ``_static`` and ``resources`` are ~52 MiB on top, and that half grows with
    the codebase rather than with the lessons. The allowance below is over
    twice what it measures today, because the failure being prevented is a
    *silent* one — GitHub Pages refuses the deployment, and the live site keeps
    serving the last build that fitted.

    The bound was "half the limit" when the course was much smaller. Audio
    does not respond to video encoding, so the complete voice matrix and 4K
    masters live on Hugging Face while Pages carries the 1440p video.

    85% rather than 50% retains room for the growing module catalog while still
    failing before Pages would silently refuse the deployment. Narration must
    remain external; the remaining levers are video encoding and poster size.
    """
    published, _dropped, _keep = real_plan
    html_allowance = 120 * 1024 * 1024
    pages_limit = 1000 * 1000 * 1000          # GitHub states 1 GB

    total = _bytes(published) + html_allowance
    assert total < pages_limit * 0.85, (
        f"site would be {total / (1024 * 1024):.0f} MiB including a "
        f"{html_allowance / (1024 * 1024):.0f} MiB HTML allowance, against a "
        f"{pages_limit * 0.85 / (1024 * 1024):.0f} MiB bound (85% of the "
        f"{pages_limit / 1e9:.0f} GB Pages limit). Drop a narration language "
        f"or re-encode the videos harder; do not raise this past ~90% — the "
        f"failure it prevents is silent.")


# ---------------------------------------------------------------------------
# What the reduction is allowed to take
# ---------------------------------------------------------------------------

@requires_library
def test_only_narration_is_ever_dropped(real_plan):
    """A lesson, a video, a poster or a caption must never be filtered out.

    This is the assertion that keeps a size fix from becoming a content
    edit. Anything the filter drops has to be under an ``audio/`` directory
    and has to be a voice asset.
    """
    _published, dropped, _keep = real_plan
    strays = [str(p.relative_to(_LIBRARY)) for p in dropped
              if "audio" not in p.relative_to(_LIBRARY).parts
              or p.suffix.lower() not in budget.VOICE_ASSET_SUFFIXES]
    assert not strays, (
        f"the filter dropped {len(strays)} file(s) that are not narration: "
        f"{strays[:5]}")


@requires_library
def test_every_lesson_keeps_its_video_and_its_poster(real_plan):
    published, _dropped, _keep = real_plan
    kept = {p.relative_to(_LIBRARY) for p in published}
    lessons = sorted({p.parts[2] for p in kept
                      if len(p.parts) > 2 and p.parts[1] == "production"})
    catalog = json.loads(
        (_LIBRARY / "tutorials" / "catalog" / "lessons_en.json").read_text())
    expected = [lesson["id"] for lesson in catalog["lessons"]]
    assert lessons == expected, (
        f"published lesson media differs from the {len(expected)}-lesson "
        "catalog")

    for lesson in lessons:
        videos = [p for p in kept
                  if p.parts[2:3] == (lesson,) and p.suffix == ".mp4"]
        posters = [p for p in kept
                   if p.parts[2:3] == (lesson,) and p.suffix == ".jpg"]
        assert videos, f"{lesson}: no video survived the filter"
        assert posters, f"{lesson}: no poster survived the filter"


@requires_library
def test_narration_is_external_not_duplicated_on_pages(real_plan):
    """Hugging Face serves narration; Pages keeps video and captions."""
    _published, _dropped, keep = real_plan
    catalog = budget.parse_voice_catalog(
        budget.voice_catalog_path(_LIBRARY).read_text())

    assert budget.NARRATION_HOST
    assert set(keep) == set(catalog)
    assert all(not voices for voices in keep.values())


@requires_library
def test_external_narration_is_downloaded_before_media_playback():
    """Do not expose mobile media elements directly to the Xet range path.

    Some mobile browsers request multiple byte ranges for a media-element
    source. The hosted narration endpoint does not reliably satisfy that
    request shape, so the player fetches the complete file with CORS and gives
    the decoder a same-origin Blob URL instead.
    """
    player = (_LIBRARY / "tutorials" / "app_v2.js").read_text(
        encoding="utf-8")

    assert "fetchNarrationAudio" in player
    assert "response.blob()" in player
    assert "URL.createObjectURL(result.blob)" in player
    assert "elements.audio.src = audioSource()" not in player


@requires_library
def test_narration_is_the_stable_mobile_clock():
    """Normal playback must never repair drift by editing audible speech.

    Mobile video clocks can advance in coarse bursts. Seeking or rate-shifting
    narration from those updates is heard as skipped, repeated, or stretched
    syllables. Narration therefore drives the silent visual master; only an
    explicit seek/load boundary may move the audio clock.

    Pin the current cache key too: the 2026-08-11 player keeps narration in
    charge through visual EOF and derives caption cues from measured sentence
    timings, so phones must not reuse the superseded mobile-smooth asset.
    """
    player = (_LIBRARY / "tutorials" / "app_v2.js").read_text(
        encoding="utf-8")

    assert "function syncVideoToNarration(force = false)" in player
    assert 'elements.audio.addEventListener("timeupdate"' in player
    assert "VIDEO_CLOCK_DRIFT_SECONDS" in player
    assert "function seekNarrationToVideo()" in player
    assert "function syncAudio(" not in player
    assert "elements.audio.playbackRate = userPlaybackRate" not in player
    index = (_LIBRARY / "tutorials" / "index.html").read_text(encoding="utf-8")
    assert 'app_v2.js?v=20260811-audio-end-park-captions' in index
    assert "20260810-mobile-smooth" not in index


@requires_library
def test_the_catalog_offers_every_hosted_voice(real_plan):
    """``app_v2.js`` reads this catalog to build the picker.

    While the site served narration the catalog had to be trimmed to what was
    staged, or the picker offered voices whose audio 404'd. The host serves
    all 50 now, so trimming the catalog to the empty staged narration set would
    hide working voices.
    """
    _published, _dropped, keep = real_plan
    catalog = budget.parse_voice_catalog(
        budget.voice_catalog_path(_LIBRARY).read_text())

    assert budget.NARRATION_HOST, (
        "no host is configured, so the catalog would be offering voices "
        "nothing serves")
    offered = sum(len(v) for v in catalog.values())
    staged = sum(len(v) for v in keep.values())
    assert offered > staged, (
        f"the catalog offers {offered} voices and {staged} are staged; if "
        f"those are equal the host is buying nothing")


@requires_library
def test_no_narration_is_staged_on_pages(real_plan):
    """External narration must not consume the GitHub Pages budget."""
    published, _dropped, _keep = real_plan
    audio, timings = set(), set()
    for path in published:
        if "audio" not in path.relative_to(_LIBRARY).parts:
            continue
        if path.suffix == ".m4a":
            audio.add(path.with_suffix(""))
        elif path.suffix == ".json":
            timings.add(path.with_suffix(""))
    assert not audio
    assert not timings


def test_audio_and_its_timing_track_travel_together(tiny_library, tmp_path):
    """The pairing rule still governs the offline build.

    ``SPACR_DOCS_FULL_AUDIO=1`` publishes the whole library, which is how a
    site is built with no access to the narration host. A ``.json`` left
    behind for a voice with no ``.m4a`` gives the player scene timings for
    narration it cannot load; an ``.m4a`` with no ``.json`` plays with no
    scene highlighting. Either mismatch is a broken lesson.
    """
    for per_language in (0, 1):
        published, _dropped, _keep = budget.plan(tiny_library, per_language)
        audio, timings = set(), set()
        for path in published:
            if "audio" not in path.relative_to(tiny_library).parts:
                continue
            if path.suffix == ".m4a":
                audio.add(path.with_suffix(""))
            elif path.suffix == ".json":
                timings.add(path.with_suffix(""))
        assert audio, f"per_language={per_language}: no narration published"
        assert audio == timings, (
            f"per_language={per_language}: "
            f"{len(audio ^ timings)} voice(s) published without their pair: "
            f"{sorted(p.name for p in (audio ^ timings))[:5]}")


# ---------------------------------------------------------------------------
# The staged tree
# ---------------------------------------------------------------------------

VOICE_CATALOG_JS = """\
"use strict";

window.SPACR_VOICE_CATALOG = Object.freeze([
  {
    id: "en",
    label: "English",
    locale: "en",
    voices: [
      { id: "af_heart", name: "Heart", variant: "American female", engineCode: "a" },
      { id: "am_puck", name: "Puck", variant: "American male", engineCode: "a" },
      { id: "bf_lily", name: "Lily", variant: "British female", engineCode: "b" }
    ]
  },
  {
    id: "ja",
    label: "Japanese",
    locale: "ja",
    voices: [
      { id: "jf_alpha", name: "Alpha", variant: "Female", engineCode: "j" },
      { id: "jm_kumo", name: "Kumo", variant: "Male", engineCode: "j" }
    ]
  }
]);
"""


@pytest.fixture
def tiny_library(tmp_path):
    """A miniature ``_extra``: 2 lessons, 2 languages, 5 voices."""
    extra = tmp_path / "_extra"
    tutorials = extra / "tutorials"
    tutorials.mkdir(parents=True)
    (tutorials / "voice_catalog.js").write_text(VOICE_CATALOG_JS)
    (tutorials / "index.html").write_text("<p>player</p>")
    (tutorials / "catalog").mkdir()
    (tutorials / "catalog" / "captions_de.json").write_text("{}")
    voices = {"en": ["af_heart", "am_puck", "bf_lily"],
              "ja": ["jf_alpha", "jm_kumo"]}
    for lesson in ("07_mask", "08_measure"):
        root = tutorials / "production" / lesson
        (root / "video").mkdir(parents=True)
        (root / "video" / f"{lesson}_silent.mp4").write_bytes(b"\0" * 64)
        (root / "poster.jpg").write_bytes(b"\0" * 16)
        for language, ids in voices.items():
            folder = root / "audio" / language
            folder.mkdir(parents=True)
            for voice in ids:
                (folder / f"{voice}.m4a").write_bytes(b"\0" * 128)
                (folder / f"{voice}.json").write_text("{}")
    return extra


def test_staging_publishes_the_default_voice_and_nothing_else(tiny_library,
                                                              tmp_path):
    dest = tmp_path / "staged"
    budget.stage(dest, tiny_library, per_language=1)

    kept = sorted(p.relative_to(dest).as_posix()
                  for p in dest.rglob("*") if p.is_file())
    audio = [p for p in kept if "/audio/" in p]
    assert audio == [
        "tutorials/production/07_mask/audio/en/af_heart.json",
        "tutorials/production/07_mask/audio/en/af_heart.m4a",
        "tutorials/production/07_mask/audio/ja/jf_alpha.json",
        "tutorials/production/07_mask/audio/ja/jf_alpha.m4a",
        "tutorials/production/08_measure/audio/en/af_heart.json",
        "tutorials/production/08_measure/audio/en/af_heart.m4a",
        "tutorials/production/08_measure/audio/ja/jf_alpha.json",
        "tutorials/production/08_measure/audio/ja/jf_alpha.m4a",
    ]
    # Everything that is not narration came through untouched.
    assert "tutorials/index.html" in kept
    assert "tutorials/catalog/captions_de.json" in kept
    assert "tutorials/production/07_mask/poster.jpg" in kept
    assert "tutorials/production/07_mask/video/07_mask_silent.mp4" in kept


def test_the_staged_catalog_offers_exactly_what_was_staged(tiny_library,
                                                           tmp_path,
                                                           monkeypatch):
    """With no host configured, the picker must not list what is not staged.

    Left unfiltered, the English dropdown offers 28 voices of which 27 are
    404s, and choosing one produces the player's "narration is unavailable"
    toast — a broken control is a worse answer than a control with one entry.

    This is the ``NARRATION_HOST = ""`` configuration: the site is the only
    thing serving narration, so what it stages is the whole offering.
    """
    monkeypatch.setattr(budget, "NARRATION_HOST", "")
    dest = tmp_path / "staged"
    budget.stage(dest, tiny_library, per_language=1)

    staged_js = (dest / "tutorials" / "voice_catalog.js").read_text()
    offered = budget.parse_voice_catalog(staged_js)
    assert offered == {"en": ["af_heart"], "ja": ["jf_alpha"]}

    for language, voices in offered.items():
        for voice in voices:
            for lesson in ("07_mask", "08_measure"):
                path = (dest / "tutorials" / "production" / lesson / "audio"
                        / language / f"{voice}.m4a")
                assert path.is_file(), f"{path} is offered but not published"

    # The filter is line-based, so the JS it emits has to still be JS: no
    # trailing comma before a `]`, and the same bracket balance as the input.
    assert ",\n    ]" not in staged_js and ",\n]" not in staged_js
    assert staged_js.count("[") == staged_js.count("]")
    assert staged_js.count("{") == staged_js.count("}")


def test_a_configured_host_leaves_the_catalog_alone(tiny_library, tmp_path,
                                                    monkeypatch):
    """The inverse: with a host set, trimming would hide working voices.

    Staged audio in this synthetic policy check is not the offering. Filtering
    the picker down to it would drop voices the host serves — which is the
    entire reason the real policy keeps narration external.
    """
    monkeypatch.setattr(budget, "NARRATION_HOST", "https://example.invalid/x")
    dest = tmp_path / "staged"
    budget.stage(dest, tiny_library, per_language=1)

    source = budget.parse_voice_catalog(
        budget.voice_catalog_path(tiny_library).read_text())
    offered = budget.parse_voice_catalog(
        (dest / "tutorials" / "voice_catalog.js").read_text())
    assert offered == source, (
        "the catalog was trimmed even though a narration host is configured")
    assert sum(len(v) for v in offered.values()) > 2, (
        "this fixture should offer more voices than the one per language "
        "staged, or the test proves nothing")


def test_full_audio_publishes_every_voice(tiny_library, tmp_path):
    """The escape hatch has to actually be one."""
    published, dropped, keep = budget.plan(tiny_library, per_language=0)
    assert not dropped
    assert keep == {"en": ["af_heart", "am_puck", "bf_lily"],
                    "ja": ["jf_alpha", "jm_kumo"]}
    assert len([p for p in published if p.suffix == ".m4a"]) == 10


def test_the_full_audio_env_var_selects_that_policy(monkeypatch):
    monkeypatch.delenv(budget.FULL_AUDIO_ENV, raising=False)
    assert budget.per_language_setting() == budget.VOICES_PER_LANGUAGE
    monkeypatch.setenv(budget.FULL_AUDIO_ENV, "1")
    assert budget.per_language_setting() == 0


def test_staging_is_rebuilt_rather_than_merged(tiny_library, tmp_path):
    """A voice retired by a policy change must not survive in the output.

    ``dest`` is a build directory that is written on every run; merging into
    it would keep publishing whatever the previous policy published, forever,
    and the ceiling test above would go on passing because it measures the
    plan rather than the directory.
    """
    dest = tmp_path / "staged"
    budget.stage(dest, tiny_library, per_language=0)
    assert (dest / "tutorials" / "production" / "07_mask" / "audio" / "en"
            / "am_puck.m4a").is_file()

    budget.stage(dest, tiny_library, per_language=1)
    assert not (dest / "tutorials" / "production" / "07_mask" / "audio" / "en"
                / "am_puck.m4a").exists()


def test_an_unknown_language_is_published_whole(tiny_library, tmp_path):
    """A tree the catalog does not describe is not evidence to delete by.

    If a lesson gains a language before ``voice_catalog.js`` does, the safe
    reading is "the catalog is stale", not "these files are surplus".
    """
    stray = (tiny_library / "tutorials" / "production" / "07_mask"
             / "audio" / "is")
    stray.mkdir(parents=True)
    (stray / "isf_one.m4a").write_bytes(b"\0" * 32)
    (stray / "isf_two.m4a").write_bytes(b"\0" * 32)

    published, dropped, _keep = budget.plan(tiny_library, per_language=1)
    assert {p.name for p in published if p.parent == stray} == {
        "isf_one.m4a", "isf_two.m4a"}
    assert not [p for p in dropped if p.parent == stray]


# ---------------------------------------------------------------------------
# The wiring
# ---------------------------------------------------------------------------

def test_conf_py_publishes_the_staged_tree_rather_than_the_raw_one():
    """The regression guard for the whole item.

    ``html_extra_path = ['_extra']`` is one line, it is the obvious thing to
    write, and it is what put the site at 88% of the Pages limit. If it comes
    back, everything above goes on passing while the build ships 712 MiB.
    """
    conf = (REPO_ROOT / "docs" / "source" / "conf.py").read_text()
    assert "html_extra_path = ['_extra']" not in conf
    assert "docs_media_budget" in conf, (
        "conf.py no longer stages the tutorial media through "
        "tools/docs_media_budget.py")
    assert "_staged_extra" in conf


def test_the_budget_module_runs_as_a_script():
    """``--report`` is what a build log and a commit message quote."""
    report = budget.report(_LIBRARY if _LIBRARY.is_dir() else None)
    assert "published:" in report
    assert "ceiling:" in report
    assert "OVER" not in report


@requires_library
def test_the_docs_workflow_publishes_from_main_only():
    """Nightly stopped republishing production Pages; keep it that way.

    The push trigger used to run every step on ``nightly`` too, so each
    nightly push overwrote the public site with unreviewed docs. The gate is
    a one-line ``if:`` and deleting it is silent.
    """
    workflow = (REPO_ROOT / ".github" / "workflows" / "docs.yml")
    if not workflow.is_file():
        pytest.skip("no docs workflow in this checkout")
    text = workflow.read_text()
    assert text.count("github.ref == 'refs/heads/main'") >= 3, (
        "every Pages-touching step (configure, upload, deploy) needs the "
        "main-only gate")
    assert "upload-pages-artifact" in text
