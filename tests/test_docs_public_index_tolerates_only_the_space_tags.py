"""The public-index check allows Hugging Face's own tags and nothing else.

A static Space serves the page with a creator script after ``<head>`` and,
since October 2026, Open Graph and Twitter card tags before ``</head>``. The
nightly verification must accept exactly those, so a deploy is not reported
unverified when the Space is correct, and must still refuse a real change.
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://einarolafsson-spacr-docs-nightly-v2.static.hf.space/"


def _tool():
    name = "publish_docs_channels_for_the_index_test"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "tools" / "publish_docs_channels.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


EXPECTED = (b'<!doctype html><html><head><meta charset="utf-8">'
            b'<style>p{}</style></head><body><p>hello</p></body></html>')
SCRIPT = (b'<script>window.huggingface={variables:{"SPACE_CREATOR_USER_ID":'
          b'"651dd0c2e956b4858bd2f10e"}};</script>')
CARD = (b'<meta property="og:title" content="spaCR nightly documentation">'
        b'<meta property="og:description" content="Nightly preview">'
        b'<meta property="og:image" content="https://cdn-thumbnails.huggingface.co/'
        b'social-thumbnails/spaces/einarolafsson/spacr-docs-nightly-v2.png">'
        b'<meta name="twitter:card" content="summary_large_image">'
        b'<meta name="twitter:image" content="https://cdn-thumbnails.huggingface.co/'
        b'social-thumbnails/spaces/einarolafsson/spacr-docs-nightly-v2.png">')


def _served(script=True, card=True):
    page = EXPECTED
    if card:
        page = page.replace(b"</head>", CARD + b"</head>", 1)
    if script:
        page = page.replace(b"<head>", b"<head>" + SCRIPT, 1)
    return page


def test_the_creator_script_and_the_card_tags_are_both_accepted():
    tool = _tool()
    for script in (True, False):
        for card in (True, False):
            assert tool._public_index_matches(EXPECTED, _served(script, card), URL)


def test_any_other_difference_is_refused():
    tool = _tool()
    assert not tool._public_index_matches(
        EXPECTED, _served().replace(b"hello", b"stale"), URL)
    assert not tool._public_index_matches(
        EXPECTED, _served().replace(b"og:title", b"og:titel"), URL)
    assert not tool._public_index_matches(
        EXPECTED, _served().replace(b"cdn-thumbnails.huggingface.co", b"evil.example"), URL)
    assert not tool._public_index_matches(
        EXPECTED, _served(), "https://example.org/")
