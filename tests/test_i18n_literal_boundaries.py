"""Quoted API literals remain visible beside target-language grammar."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

from build_i18n_catalogs import _syntax_preserved  # noqa: E402


def test_korean_particle_may_follow_an_exact_quoted_literal():
    source = "Use 'load_images' or 'stream_images'."
    translated = "'load_images'로 읽고 'stream_images'에서 자릅니다."

    assert _syntax_preserved(source, translated)
    assert not _syntax_preserved(
        source, translated.replace("'load_images'", "'load_image'")
    )


def test_english_apostrophes_do_not_become_quoted_api_literals():
    source = "Don't replace the user's selected source."
    translated = "사용자가 선택한 소스를 바꾸지 않습니다."

    assert _syntax_preserved(source, translated)
