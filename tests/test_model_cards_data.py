"""The model card data layer: shipped JSON, lookups, URLs, cache and fallback text.

Nothing here needs Qt. The spaCR home is a temporary folder and every byte
source is a stand-in, so no test reaches Hugging Face.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from spacr import model_cards as cards

PNG = cards._PNG_MAGIC + b"pixels"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    import socket

    def _boom(*_a, **_k):
        raise AssertionError("network access in a card test")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom, raising=False)
    monkeypatch.setattr(cards, "_spacr_home", lambda: tmp_path / "home")
    cards._cards.cache_clear()
    yield
    cards._cards.cache_clear()


@pytest.fixture
def card_folder(monkeypatch, tmp_path):
    folder = tmp_path / "cards"
    folder.mkdir()
    monkeypatch.setattr(cards, "_CARD_DIR", folder)
    cards._cards.cache_clear()
    return folder


def write_card(folder: Path, name: str, **fields) -> dict:
    data = {"key": name, "repo_id": f"owner/{name}"}
    data.update(fields)
    (folder / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def test_every_shipped_card_is_a_three_page_card_named_after_its_repo():
    folder = Path(cards.__file__).resolve().parent / "resources" / "model_cards"
    files = sorted(folder.glob("*.json"))
    assert files
    assert cards.card_keys() == tuple(
        json.loads(p.read_text(encoding="utf-8"))["key"] for p in files)
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["repo_id"].split("/")[-1] == path.stem
        assert len(data["pages"]) == cards.PAGE_COUNT == 3
        assert cards.repo_of_key(data["key"]) == data["repo_id"]
        assert cards.score_rows(data) and cards.fact_rows(data)
        assert cards.about_lines(data)


def test_cards_are_read_once_in_file_name_order(card_folder):
    write_card(card_folder, "b-model")
    write_card(card_folder, "a-model")
    assert cards.card_keys() == ("a-model", "b-model")
    assert cards._cards() is cards._cards()
    write_card(card_folder, "c-model")
    assert cards.card_keys() == ("a-model", "b-model")
    cards._cards.cache_clear()
    assert cards.card_keys() == ("a-model", "b-model", "c-model")


def test_a_card_key_is_read_as_text(card_folder):
    (card_folder / "x.json").write_text(
        json.dumps({"key": 7, "repo_id": "o/x"}), encoding="utf-8")
    assert cards.card_keys() == ("7",)
    assert cards.repo_of_key(7) == "o/x"


def test_no_card_files_means_no_keys(card_folder):
    assert cards.card_keys() == ()
    assert cards.repo_of_key("anything") == ""


def test_invalid_json_is_an_error_not_an_empty_card(card_folder):
    (card_folder / "bad.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        cards.card_keys()


def test_a_card_without_a_key_is_an_error(card_folder):
    (card_folder / "nokey.json").write_text(
        json.dumps({"repo_id": "o/nokey"}), encoding="utf-8")
    with pytest.raises(KeyError):
        cards.card_keys()


def test_a_missing_card_folder_means_no_cards(monkeypatch, tmp_path):
    monkeypatch.setattr(cards, "_CARD_DIR", tmp_path / "does-not-exist")
    cards._cards.cache_clear()
    assert cards.card_keys() == ()


def test_repo_of_key_accepts_odd_input(card_folder):
    write_card(card_folder, "m")
    assert cards.repo_of_key("m") == "owner/m"
    for odd in (None, "", 0, "M", "m "):
        assert cards.repo_of_key(odd) == ""


def test_card_for_entry_prefers_the_key_then_the_repo(card_folder, monkeypatch):
    first = write_card(card_folder, "first")
    second = write_card(card_folder, "second")
    by_key = SimpleNamespace(key="second",
                             uri="https://huggingface.co/owner/first/x")
    assert cards.card_for_entry(by_key)["key"] == second["key"]
    seen = []

    def hf_repo_of(uri):
        seen.append(uri)
        return "owner/first" if "first" in uri else ""

    monkeypatch.setattr("spacr.model_zoo.hf_repo_of", hf_repo_of)
    by_uri = SimpleNamespace(key="renamed",
                             uri="https://huggingface.co/owner/first/x")
    assert cards.card_for_entry(by_uri)["key"] == first["key"]
    assert seen == [by_uri.uri]
    assert cards.card_for_entry(SimpleNamespace(key="renamed", uri="")) is None
    assert cards.card_for_entry(SimpleNamespace(key=None)) is None
    assert cards.card_for_entry(object()) is None
    assert cards.card_for_entry(None) is None
    monkeypatch.setattr("spacr.model_zoo.hf_repo_of",
                        lambda uri: "owner/unknown")
    other = SimpleNamespace(key="x",
                            uri="https://huggingface.co/owner/unknown/x")
    assert cards.card_for_entry(other) is None


def test_hugging_face_urls():
    assert cards.page_url("o/r", 1) == (
        "https://huggingface.co/o/r/resolve/main/cards/page1.png")
    assert cards.page_url("o/r", "3").endswith("/cards/page3.png")
    assert cards.page_url("o/r", 2.0).endswith("/cards/page2.png")
    assert cards.repo_url("o/r") == "https://huggingface.co/o/r"


def test_the_cache_sits_under_the_spacr_home_by_repo_name(tmp_path):
    assert cards.cache_dir("owner/repo") == (
        tmp_path / "home" / "model_cards" / "repo")
    assert cards.cache_dir("bare") == tmp_path / "home" / "model_cards" / "bare"


def test_cached_pages_reports_each_page_on_its_own():
    assert cards.cached_pages("o/r") == [None, None, None]
    folder = cards.cache_dir("o/r")
    folder.mkdir(parents=True)
    (folder / "page2.png").write_bytes(PNG)
    (folder / "page3.png").mkdir()
    assert cards.cached_pages("o/r") == [None, folder / "page2.png", None]


def test_fetch_pages_downloads_in_order_and_leaves_no_part_files():
    urls = []

    def fetch(url):
        urls.append(url)
        return PNG + url[-5:-4].encode()

    paths = cards.fetch_pages("o/r", fetch)
    assert urls == [cards.page_url("o/r", n) for n in (1, 2, 3)]
    assert [p.name for p in paths] == ["page1.png", "page2.png", "page3.png"]
    assert [p.read_bytes() for p in paths] == [PNG + b"1", PNG + b"2",
                                               PNG + b"3"]
    assert not list(cards.cache_dir("o/r").glob("*.part"))


def test_fetch_pages_only_asks_for_the_missing_pages():
    folder = cards.cache_dir("o/r")
    folder.mkdir(parents=True)
    (folder / "page1.png").write_bytes(PNG + b"old")
    urls = []

    def fetch(url):
        urls.append(url)
        return PNG

    paths = cards.fetch_pages("o/r", fetch)
    assert urls == [cards.page_url("o/r", 2), cards.page_url("o/r", 3)]
    assert paths[0].read_bytes() == PNG + b"old"


def test_a_failure_keeps_the_pages_fetched_before_it():
    def fetch(url):
        if url.endswith("page3.png"):
            raise OSError("connection reset")
        return PNG

    with pytest.raises(OSError, match="connection reset"):
        cards.fetch_pages("o/r", fetch)
    assert [p is not None for p in cards.cached_pages("o/r")] == [
        True, True, False]


def test_a_page_that_is_not_a_png_is_refused_and_not_cached():
    def fetch(url):
        return PNG if url.endswith("page1.png") else b"<html>Not found</html>"

    with pytest.raises(OSError, match="page 2 of o/r is not a PNG image"):
        cards.fetch_pages("o/r", fetch)
    assert [p is not None for p in cards.cached_pages("o/r")] == [
        True, False, False]
    assert not list(cards.cache_dir("o/r").glob("*.part"))


def test_the_default_byte_source_is_urlopen_with_a_timeout(monkeypatch):
    calls = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return PNG

    def urlopen(url, timeout):
        calls.append((url, timeout))
        return Response()

    monkeypatch.setattr(cards.urllib.request, "urlopen", urlopen)
    paths = cards.fetch_pages("o/r")
    assert len(paths) == 3
    assert calls == [(cards.page_url("o/r", n), 20.0) for n in (1, 2, 3)]
    assert cards._fetch("https://x", timeout=3) == PNG
    assert calls[-1] == ("https://x", 3)


def test_the_default_byte_source_reports_an_unreachable_host(monkeypatch):
    def urlopen(url, timeout):
        raise OSError("name resolution failed")

    monkeypatch.setattr(cards.urllib.request, "urlopen", urlopen)
    with pytest.raises(OSError, match="name resolution"):
        cards.fetch_pages("o/r")
    assert cards.cached_pages("o/r") == [None, None, None]


def test_clear_cache_removes_the_images_and_tolerates_a_missing_folder():
    cards.clear_cache("o/never-cached")
    cards.fetch_pages("o/r", lambda url: PNG)
    other = cards.fetch_pages("o/keep", lambda url: PNG)
    cards.clear_cache("o/r")
    assert cards.cached_pages("o/r") == [None, None, None]
    assert all(p.is_file() for p in other)


@pytest.mark.parametrize("value,text", [
    (0.75, "0.750"), ("0.5", "0.500"), (1, "1.000"), (0.12345, "0.123"),
    (None, "None"), ("n/a", "n/a"), ([], "[]"),
])
def test_numbers_are_three_decimals_and_text_is_left_alone(value, text):
    assert cards._num(value) == text


def test_score_rows_pair_this_model_with_the_stock_base():
    card = {"scores": {"rows": [
        {"label": "Object F1", "key": "f1", "this": 0.75, "stock": 0.5},
        {"key": "iou", "this": "0.6", "stock": None},
        {"this": 1, "stock": 2},
    ]}}
    assert cards.score_rows(card) == [
        ("Object F1", "0.750 (0.500)"),
        ("iou", "0.600 (None)"),
        ("", "1.000 (2.000)"),
    ]
    for empty in ({}, {"scores": None}, {"scores": {}},
                  {"scores": {"rows": None}}):
        assert cards.score_rows(empty) == []


def test_fact_rows_are_label_and_value_text():
    card = {"facts": {"rows": [{"label": "Training FOV", "this": 120},
                               {"this": None}, {"label": "Epochs"}]}}
    assert cards.fact_rows(card) == [("Training FOV", "120"), ("", "None"),
                                     ("Epochs", "")]
    for empty in ({}, {"facts": None}, {"facts": {"rows": []}}):
        assert cards.fact_rows(empty) == []


def test_about_lines_skip_empty_fields_and_end_with_the_leak_note():
    card = {"about": {"load_line": "load it", "segments": "vacuoles",
                      "status": "", "trained_on": "plates",
                      "leak_note": "careful"}}
    assert cards.about_lines(card) == ["vacuoles", "plates", "load it",
                                       "careful"]
    assert cards.about_lines({"about": {"status": "x", "leak_note": ""}}) == [
        "x"]
    for empty in ({}, {"about": None}, {"about": {}}):
        assert cards.about_lines(empty) == []


def test_page_html_builds_each_page_with_its_headings_and_escapes_text():
    card = {"scores": {"rows": [{"label": "A <b>", "this": 1, "stock": 2}]},
            "facts": {"rows": [{"label": "F & G", "this": "<i>x</i>"}]},
            "about": {"segments": "5 < 6", "status": "ok"}}
    one = cards.page_html(card, 1)
    assert one.startswith("<table") and one.endswith("</table>")
    assert ">Score</th>" in one and ">Value</th>" in one
    assert "A &lt;b&gt;" in one and "<b>" not in one
    assert "1.000 (2.000)" in one
    two = cards.page_html(card, 2, ("S", "V", "Fact", "Val"))
    assert ">Fact</th><th align='left'>Val</th>" in two
    assert "F &amp; G" in two and "&lt;i&gt;x&lt;/i&gt;" in two
    three = cards.page_html(card, 3)
    assert three == "<p>5 &lt; 6</p><p>ok</p>"
    assert cards.page_html(card, 9) == three
    custom = cards.page_html(card, 1, ("Metrica", "Valor", "x", "y"))
    assert ">Metrica</th><th align='left'>Valor</th>" in custom
    assert cards.page_html({}, 1).count("<tr>") == 1
    assert cards.page_html({}, 3) == ""


def test_model_link_prefers_the_card_link_then_the_repo():
    assert cards.model_link({"about": {"links": {"model": "https://x/y"}},
                             "repo_id": "a/b"}) == "https://x/y"
    assert cards.model_link({"about": {"links": {"model": ""}},
                             "repo_id": "a/b"}) == "https://huggingface.co/a/b"
    assert cards.model_link({"about": {"links": None}, "repo_id": "a/b"}
                            ) == "https://huggingface.co/a/b"
    assert cards.model_link({"about": None}) == "https://huggingface.co/"
