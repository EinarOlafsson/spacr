"""Pinned media retries transient HTTP failures without relaxing verification."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import traceback

import pytest
import requests


TREE_PATH = "docs/source/i18n/api/de.json"
REVISION = "9" * 40


@pytest.fixture
def pinned_media(tmp_path, monkeypatch):
    import huggingface_hub

    source = Path(__file__).resolve().parents[1] / "tools" / "fetch_docs_media.py"
    spec = importlib.util.spec_from_file_location("media_retry_control", source)
    media = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(media)
    payload = tmp_path / "cached.json"
    payload.write_bytes(b'{"verified":true}')
    root = tmp_path / "repo"
    pin = root / "docs" / "media_revision.json"
    pin.parent.mkdir(parents=True)
    pin.write_text(json.dumps({"groups": {"api": {
        "repo": "owner/public-data", "repo_type": "dataset", "revision": REVISION,
        "files": {TREE_PATH: {"path_in_repo": "i18n/api/de.json",
                               "sha256": hashlib.sha256(payload.read_bytes()).hexdigest()}}}}}))
    waits = []
    class Clock:
        elapsed = 0.0

        def monotonic(self):
            return self.elapsed

        def time(self):
            return 1000.0 + self.elapsed

        def sleep(self, seconds):
            waits.append(seconds)
            self.elapsed += seconds

    monkeypatch.setattr(media, "time", Clock())
    monkeypatch.setattr(media, "random", SimpleNamespace(uniform=lambda low, high: 0.25))
    return media, huggingface_hub, root, pin, payload, waits


def http_failure(status, retry_after=None):
    response = requests.Response()
    response.status_code = status
    if retry_after is not None:
        response.headers["Retry-After"] = retry_after
    return requests.HTTPError("https://cdn.invalid/data?Signature=SIGNED_SECRET", response=response)


def test_transient_statuses_retry_the_same_pin_before_verified_copy(pinned_media, monkeypatch):
    media, hub, root, pin, payload, waits = pinned_media
    original_pin = pin.read_bytes()
    for status in (429, 500, 502, 503, 504):
        calls = []
        waits.clear()
        media.time.elapsed = 0.0
        target = root / TREE_PATH
        target.unlink(missing_ok=True)

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise http_failure(status)
            return str(payload)

        monkeypatch.setattr(hub, "hf_hub_download", download)
        assert media.fetch("api", root)[TREE_PATH] == target
        assert target.read_bytes() == payload.read_bytes()
        assert waits == [1.25]
        assert calls == [(("owner/public-data", "i18n/api/de.json"),
                          {"repo_type": "dataset", "revision": REVISION})] * 2
        assert pin.read_bytes() == original_pin


def test_retry_after_seconds_dates_and_invalid_values_are_bounded(pinned_media, monkeypatch):
    media, hub, root, pin, payload, waits = pinned_media
    for header, expected in (("3", 3.0), ("60", 60.0),
                             ("Thu, 01 Jan 1970 00:17:00 GMT", 20.0),
                             ("Thu, 01 Jan 1970 00:17:40 GMT", 60.0),
                             ("-1", 1.25), ("nan", 1.25), ("inf", 1.25),
                             ("not a date", 1.25)):
        calls = []
        waits.clear()
        media.time.elapsed = 0.0

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            if len(calls) == 1:
                raise http_failure(429, header)
            return str(payload)

        monkeypatch.setattr(hub, "hf_hub_download", download)
        assert media.media_path(TREE_PATH, "api", pin) == payload
        assert waits == [expected]
        assert len(calls) == 2


def test_exhaustion_is_finite_and_traceback_contains_no_signed_url(pinned_media, monkeypatch, capsys):
    media, hub, root, pin, payload, waits = pinned_media
    for header, expected in (("30", [30.0] * 3), (None, [1.25, 2.25, 4.25])):
        waits.clear()
        media.time.elapsed = 0.0
        calls = []

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            raise http_failure(429, header)

        monkeypatch.setattr(hub, "hf_hub_download", download)
        with pytest.raises(RuntimeError, match="could not fetch.*HTTP 429") as caught:
            media.fetch("api", root)
        assert len(calls) == 4 and waits == expected
        assert not (root / TREE_PATH).exists()
        rendered = "".join(traceback.format_exception(type(caught.value), caught.value, caught.value.__traceback__))
        assert "SIGNED_SECRET" not in rendered and "cdn.invalid" not in rendered
        assert caught.value.__cause__ is None and caught.value.__suppress_context__
        captured = capsys.readouterr()
        assert "SIGNED_SECRET" not in captured.out + captured.err


def test_retry_after_beyond_remaining_budget_fails_without_retrying_early(pinned_media, monkeypatch):
    media, hub, root, pin, payload, waits = pinned_media
    scenarios = ((["121"], 0.0, []), (["120"], 0.0, [120.0]),
                 (["Thu, 01 Jan 1970 00:18:41 GMT"], 0.0, []),
                 (["80", "60"], 0.0, [80.0]),
                 (["30"], 100.0, []))
    for headers, download_seconds, expected_waits in scenarios:
        calls = []
        waits.clear()
        media.time.elapsed = 0.0

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            media.time.elapsed += download_seconds
            if len(calls) > len(headers):
                pytest.fail("retried before the server minimum or beyond the retry budget")
            raise http_failure(429, headers[len(calls) - 1])

        monkeypatch.setattr(hub, "hf_hub_download", download)
        with pytest.raises(RuntimeError, match="could not fetch.*HTTP 429"):
            media.media_path(TREE_PATH, "api", pin)
        assert len(calls) == len(headers) and waits == expected_waits


def test_permanent_and_non_http_failures_never_retry(pinned_media, monkeypatch):
    media, hub, root, pin, payload, waits = pinned_media
    for error in (http_failure(401), http_failure(403), http_failure(404), http_failure(422), OSError("offline")):
        calls = []
        waits.clear()
        media.time.elapsed = 0.0

        def download(*args, **kwargs):
            calls.append((args, kwargs))
            raise error

        monkeypatch.setattr(hub, "hf_hub_download", download)
        with pytest.raises(RuntimeError, match="could not fetch"):
            media.media_path(TREE_PATH, "api", pin)
        assert len(calls) == 1 and waits == []


def test_a_digest_mismatch_is_not_retried_or_copied(pinned_media, monkeypatch):
    media, hub, root, pin, payload, waits = pinned_media
    payload.write_bytes(b"tampered cache")
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        return str(payload)

    monkeypatch.setattr(hub, "hf_hub_download", download)
    with pytest.raises(RuntimeError, match="does not match the pin"):
        media.fetch("api", root)
    assert len(calls) == 1 and waits == []
    assert not (root / TREE_PATH).exists()
