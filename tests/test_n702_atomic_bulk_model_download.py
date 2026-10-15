from pathlib import Path

import pytest


@pytest.fixture
def pack(tmp_path, monkeypatch):
    from spacr import utils

    package = tmp_path / "package"
    package.mkdir()
    monkeypatch.setattr(utils, "spacr_path", str(package / "__init__.py"))
    monkeypatch.setattr(utils.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(utils, "list_repo_files", lambda repo_id, repo_type: ["first.pth", "nested/second.pth"])
    return utils, package / "resources" / "models"


class Response:
    status_code = 200

    def __init__(self, fail=False):
        self.fail = fail

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield b"complete-first-chunk"
        if self.fail:
            raise OSError("interrupted native stream")
        yield b"-tail"


def test_later_interrupted_stream_never_publishes_partial_pack(pack, monkeypatch):
    utils, destination = pack
    monkeypatch.setattr(utils.requests, "get", lambda url, stream: Response("second" in url))
    with pytest.raises(OSError, match="interrupted native stream"):
        utils.download_models(retries=1, delay=0)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert list(destination.parent.iterdir()) == [destination]
    monkeypatch.setattr(utils.requests, "get", lambda url, stream: Response())
    assert Path(utils.download_models(retries=1, delay=0)) == destination
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == {
        "first.pth": b"complete-first-chunk-tail", "second.pth": b"complete-first-chunk-tail"}


def test_all_final_files_appear_only_after_last_stream_finishes(pack, monkeypatch):
    utils, destination = pack

    def get(url, stream):
        assert list(destination.iterdir()) == []
        return Response()

    monkeypatch.setattr(utils.requests, "get", get)
    utils.download_models(retries=1, delay=0)
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: pytest.fail("cached pack fetched"))
    utils.download_models(retries=1, delay=0)


def test_existing_private_cache_is_preserved_without_network(pack, monkeypatch):
    utils, destination = pack
    destination.mkdir(parents=True)
    private = destination / "my-trained-checkpoint.pth"
    private.write_bytes(b"private-local-model")
    monkeypatch.setattr(utils, "list_repo_files", lambda *args, **kwargs: pytest.fail("cache listed"))
    assert Path(utils.download_models()) == destination
    assert private.read_bytes() == b"private-local-model"


def test_collision_refused_before_fetch_or_publication(pack, monkeypatch):
    utils, destination = pack
    monkeypatch.setattr(utils, "list_repo_files", lambda *args, **kwargs: ["a/same.pth", "b/same.pth"])
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: pytest.fail("ambiguous file fetched"))
    with pytest.raises(ValueError):
        utils.download_models(retries=1, delay=0)
    assert list(destination.iterdir()) == []


def test_private_file_added_during_stream_is_never_overwritten(pack, monkeypatch):
    utils, destination = pack

    def get(url, stream):
        (destination / "private.pth").write_bytes(b"user-file-arrived-during-download")
        return Response()

    monkeypatch.setattr(utils.requests, "get", get)
    with pytest.raises(OSError):
        utils.download_models(retries=1, delay=0)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == {
        "private.pth": b"user-file-arrived-during-download"}
    assert list(destination.parent.iterdir()) == [destination]


def test_stream_retry_truncates_staging_and_closes_each_response(pack, monkeypatch):
    utils, destination = pack
    monkeypatch.setattr(utils, "list_repo_files", lambda *args, **kwargs: ["model.pth"])
    responses = []

    class RetryingResponse(Response):
        closed = False

        def iter_content(self, chunk_size):
            if len(responses) == 1:
                yield b"partial-old-long-prefix"
                raise utils.requests.ConnectionError("interrupted")
            yield b"new"

        def close(self):
            self.closed = True

    def get(url, stream):
        response = RetryingResponse()
        responses.append(response)
        return response

    monkeypatch.setattr(utils.requests, "get", get)
    utils.download_models(retries=2, delay=0)
    assert (destination / "model.pth").read_bytes() == b"new"
    assert len(responses) == 2
    assert all(response.closed for response in responses)


def test_failed_directory_publish_restores_empty_cache_and_cleans_staging(pack, monkeypatch):
    utils, destination = pack
    monkeypatch.setattr(utils.requests, "get", lambda *args, **kwargs: Response())

    def fail_replace(*args):
        raise OSError("controlled publication failure")

    monkeypatch.setattr(utils.os, "replace", fail_replace)
    with pytest.raises(OSError, match="controlled publication failure"):
        utils.download_models(retries=1, delay=0)
    assert destination.is_dir()
    assert list(destination.iterdir()) == []
    assert list(destination.parent.iterdir()) == [destination]
