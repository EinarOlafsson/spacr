"""The 3-page model cards of the Model Zoo: data, repository mapping, images.

Each published segmentation model has three card pages on Hugging Face
(``cards/page1.png`` to ``page3.png``): its scores against the stock base, its
training facts, and an about-and-links page. The same data ships with spaCR as
``resources/model_cards/<repo>.json``, so a card can be shown as text when the
images cannot be fetched.

Nothing here imports Qt; the viewer lives in
:mod:`spacr.qt.widgets.model_card_viewer`.
"""
from __future__ import annotations

import html
import json
import os
import shutil
import urllib.request
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .logging_util import _spacr_home

#: Number of pages of a card.
PAGE_COUNT = 3

_CARD_DIR = Path(__file__).resolve().parent / "resources" / "model_cards"
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


@lru_cache(maxsize=1)
def _cards() -> Dict[str, Dict[str, Any]]:
    """Every shipped card, keyed by zoo key."""
    out: Dict[str, Dict[str, Any]] = {}
    for path in sorted(_CARD_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        out[str(data["key"])] = data
    return out


def card_keys() -> Tuple[str, ...]:
    """The zoo keys that have a 3-page card."""
    return tuple(_cards())


def repo_of_key(key: Any) -> str:
    """The ``owner/repo`` of a zoo key that has a card, else ``''``.

    :param key: a model zoo key.
    """
    card = _cards().get(str(key or ""))
    return str(card["repo_id"]) if card else ""


def card_for_entry(entry: Any) -> Optional[Dict[str, Any]]:
    """The card data of a :class:`~spacr.model_zoo.ModelEntry`, or ``None``.

    Matched on the zoo key, then on the repository the entry is fetched from.

    :param entry: the model zoo entry.
    """
    card = _cards().get(str(getattr(entry, "key", "") or ""))
    if card is not None:
        return card
    from .model_zoo import hf_repo_of

    repo = hf_repo_of(getattr(entry, "uri", ""))
    for candidate in _cards().values():
        if repo and candidate["repo_id"] == repo:
            return candidate
    return None


def page_url(repo: str, page: int) -> str:
    """The Hugging Face URL of card page ``page`` (1-based) of ``repo``.

    :param repo: ``owner/repo``.
    :param page: the page number, from 1.
    """
    return f"https://huggingface.co/{repo}/resolve/main/cards/page{int(page)}.png"


def repo_url(repo: str) -> str:
    """The Hugging Face page of a model repository.

    :param repo: ``owner/repo``.
    """
    return f"https://huggingface.co/{repo}"


def cache_dir(repo: str) -> Path:
    """Where the page images of ``repo`` are cached, under the spaCR home.

    :param repo: ``owner/repo``.
    """
    return Path(_spacr_home()) / "model_cards" / repo.split("/")[-1]


def cached_pages(repo: str) -> List[Optional[Path]]:
    """The cached image of each page, ``None`` where it is not cached.

    :param repo: ``owner/repo``.
    """
    folder = cache_dir(repo)
    return [folder / f"page{n}.png" if (folder / f"page{n}.png").is_file()
            else None for n in range(1, PAGE_COUNT + 1)]


def _fetch(url: str, timeout: float = 20.0) -> bytes:
    """Download ``url`` and return its bytes."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def fetch_pages(repo: str,
                fetch: Optional[Callable[[str], bytes]] = None
                ) -> List[Path]:
    """Download the missing page images of ``repo`` into the cache.

    :param repo: ``owner/repo``.
    :param fetch: byte source taking a URL; defaults to ``urlopen``.
    :returns: the three cached paths.
    :raises OSError: when a page cannot be fetched or is not a PNG. Pages
        fetched before the failure stay cached.
    """
    get = fetch or _fetch
    folder = cache_dir(repo)
    folder.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for n in range(1, PAGE_COUNT + 1):
        target = folder / f"page{n}.png"
        if not target.is_file():
            data = get(page_url(repo, n))
            if not data.startswith(_PNG_MAGIC):
                raise OSError(f"page {n} of {repo} is not a PNG image")
            part = target.with_suffix(".part")
            part.write_bytes(data)
            os.replace(part, target)
        paths.append(target)
    return paths


def clear_cache(repo: str) -> None:
    """Remove the cached images of ``repo``.

    :param repo: ``owner/repo``.
    """
    shutil.rmtree(cache_dir(repo), ignore_errors=True)


def _num(value: Any) -> str:
    """``value`` to three decimals, or as text when it is not a number.

    :param value: a score from the card data.
    :returns: the printed value.
    """
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def score_rows(card: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Page 1 as ``(metric, "this (stock x)")`` pairs.

    :param card: the card data.
    """
    rows = (card.get("scores") or {}).get("rows") or []
    return [(str(r.get("label", r.get("key", ""))),
             f"{_num(r.get('this'))} ({_num(r.get('stock'))})")
            for r in rows]


def fact_rows(card: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Page 2 as ``(fact, value)`` pairs.

    :param card: the card data.
    """
    rows = (card.get("facts") or {}).get("rows") or []
    return [(str(r.get("label", "")), str(r.get("this", ""))) for r in rows]


def about_lines(card: Dict[str, Any]) -> List[str]:
    """Page 3 as plain sentences, without the empty fields.

    :param card: the card data.
    """
    about = card.get("about") or {}
    lines = [str(about.get(k)) for k in
             ("segments", "trained_on", "status", "load_line")
             if about.get(k)]
    if about.get("leak_note"):
        lines.append(str(about["leak_note"]))
    return lines


def page_html(card: Dict[str, Any], page: int,
              headers: Tuple[str, str, str, str] = ("Score", "Value",
                                                    "Fact", "Value")) -> str:
    """The text fallback of one card page (1-based) as a small HTML body.

    :param card: the card data.
    :param page: the page number, from 1.
    :param headers: the already translated column headings for the scores
        and facts tables.
    """
    e = html.escape

    def table(rows, head):
        """One two-column HTML table with a heading row.

        :param rows: ``(left, right)`` text pairs, one per body row.
        :param head: the two column headings.
        """
        body = "".join(f"<tr><td>{e(a)}</td><td>{e(b)}</td></tr>"
                       for a, b in rows)
        return (f"<table cellspacing='4'><tr><th align='left'>{e(head[0])}"
                f"</th><th align='left'>{e(head[1])}</th></tr>{body}</table>")

    if page == 1:
        return table(score_rows(card), headers[:2])
    if page == 2:
        return table(fact_rows(card), headers[2:])
    return "".join(f"<p>{e(line)}</p>" for line in about_lines(card))


def model_link(card: Dict[str, Any]) -> str:
    """The Hugging Face URL of the model the card describes.

    :param card: the card data.
    """
    links = (card.get("about") or {}).get("links") or {}
    return str(links.get("model") or repo_url(str(card.get("repo_id", ""))))
