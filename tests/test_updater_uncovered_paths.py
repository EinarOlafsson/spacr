"""What the updater does when the network, the stdlib, or the tool lets it down.

Nothing here reaches PyPI, GitHub, or a packaging tool. The transport is
replaced at its boundary -- ``urllib.request.urlopen`` for the two version
checks, an injected ``runner`` for the resolver -- so the real parsing,
comparison, and error-reporting code runs against real strings while no
socket is opened and no install is ever performed.
"""
from __future__ import annotations

import io
import sys
import types
import urllib.error
import urllib.request

import pytest

from spacr import updater


# ---------------------------------------------------------------------------
# Is there an upgrade?
# ---------------------------------------------------------------------------

def test_a_check_that_learned_no_release_offers_no_upgrade():
    """With PyPI unreachable there is no version to compare against."""
    info = updater.UpdateInfo(installed_version="1.5.0.4",
                              latest_release=None,
                              nightly_sha=None,
                              error="pypi: timed out")
    assert info.upgrade_available is False


def test_a_release_newer_than_the_installed_version_is_an_upgrade():
    """``1.5.0.4`` against a published ``1.5.1`` is behind."""
    info = updater.UpdateInfo(installed_version="1.5.0.4",
                              latest_release="1.5.1",
                              nightly_sha="abc1234")
    assert info.upgrade_available is True


def test_the_installed_version_matching_the_release_is_not_an_upgrade():
    """Being on the current release is not an offer to reinstall it."""
    same = updater.UpdateInfo(installed_version="1.5.0.4",
                              latest_release="1.5.0.4", nightly_sha=None)
    ahead = updater.UpdateInfo(installed_version="1.5.1",
                               latest_release="1.5.0.4", nightly_sha=None)
    assert same.upgrade_available is False
    assert ahead.upgrade_available is False


# ---------------------------------------------------------------------------
# The two fetches, without a network
# ---------------------------------------------------------------------------

class _Response:
    """The slice of an ``http.client.HTTPResponse`` the updater reads."""

    def __init__(self, body: bytes):
        self._body = io.BytesIO(body)

    def read(self):
        return self._body.read()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _route(monkeypatch, handler):
    """Answer every ``urlopen`` from ``handler(url)`` instead of the network."""
    def _urlopen(req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        return handler(url)

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)


def test_an_offline_check_reports_the_pypi_failure_and_still_answers(
        monkeypatch):
    """Both fetches down: the version check reports, it does not raise.

    The first failure is the one named, and both remote fields stay ``None``
    so nothing downstream can mistake a failed check for "up to date".
    """
    def _dead(_url):
        raise urllib.error.URLError("Name or service not known")

    _route(monkeypatch, _dead)

    info = updater.check_for_updates(timeout=0.01)

    assert info.latest_release is None
    assert info.nightly_sha is None
    assert info.error is not None
    assert info.error.startswith("pypi: ")
    assert "Name or service not known" in info.error
    # The check never claims an upgrade it could not verify.
    assert info.upgrade_available is False
    assert info.installed_version


def test_a_reachable_pypi_with_a_dead_github_names_github_as_the_failure(
        monkeypatch):
    """The nightly hash is optional; its failure is reported on its own terms.

    The PyPI answer must survive intact -- the GitHub error is only recorded
    because nothing had claimed the ``error`` slot before it.
    """
    def _handler(url):
        if url == updater.PYPI_URL:
            return _Response(b'{"info": {"version": "9.9.9"}}')
        raise urllib.error.HTTPError(url, 503, "Service Unavailable", {}, None)

    _route(monkeypatch, _handler)

    info = updater.check_for_updates(timeout=0.01)

    assert info.latest_release == "9.9.9"
    assert info.nightly_sha is None
    assert info.error is not None
    assert info.error.startswith("github: ")
    assert "503" in info.error


def test_a_dead_pypi_does_not_let_github_overwrite_the_first_failure(
        monkeypatch):
    """``error`` names the first thing that went wrong, not the last."""
    def _handler(url):
        if url == updater.PYPI_URL:
            raise urllib.error.URLError("pypi is down")
        raise urllib.error.URLError("github is down too")

    _route(monkeypatch, _handler)

    info = updater.check_for_updates(timeout=0.01)

    assert "pypi is down" in info.error
    assert "github is down too" not in info.error


def test_a_successful_pair_of_fetches_keeps_the_short_nightly_hash(
        monkeypatch):
    """The commit hash is abbreviated to seven characters, as git prints it."""
    def _handler(url):
        if url == updater.PYPI_URL:
            return _Response(b'{"info": {"version": "1.5.1"}}')
        return _Response(b'{"sha": "0123456789abcdef0123456789abcdef01234567"}')

    _route(monkeypatch, _handler)

    info = updater.check_for_updates(timeout=0.01)

    assert info.error is None
    assert info.latest_release == "1.5.1"
    assert info.nightly_sha == "0123456"


# ---------------------------------------------------------------------------
# The stdlib the version lookup depends on
# ---------------------------------------------------------------------------

def test_an_interpreter_without_importlib_metadata_reports_no_version(
        monkeypatch):
    """A build whose ``importlib.metadata`` lacks the API answers ``None``.

    The frozen desktop builds are assembled by a bundler that only ships the
    modules it can see being imported, so this import is not guaranteed. The
    lookup has to degrade to "I do not know" rather than take the process
    down; ``pytest`` itself is installed here, so a working lookup would
    return a version string.
    """
    assert updater.installed_version("pytest") is not None  # the control

    stripped = types.ModuleType("importlib.metadata")  # no version(), no error
    monkeypatch.setitem(sys.modules, "importlib.metadata", stripped)

    assert updater.installed_version("pytest") is None


def test_a_package_that_is_not_installed_has_no_version():
    """The ordinary absent-package answer, distinct from a broken stdlib."""
    assert updater.installed_version(
        "a-distribution-nobody-published-42") is None


# ---------------------------------------------------------------------------
# A resolver that cannot be run at all
# ---------------------------------------------------------------------------

def test_a_resolver_the_os_refuses_to_launch_is_reported_not_raised(
        monkeypatch):
    """``PermissionError`` from the launch is a failed dry run, not a crash.

    ``FileNotFoundError`` and a timeout have their own messages; everything
    else the operating system can raise on exec -- a non-executable
    interpreter, a mount without exec permission, ``OSError`` out of
    ``posix_spawn`` -- lands here and must still produce a ``DryRun`` that
    says the plan is unknown, because :func:`install_decision` refuses an
    install whose consequences are unknown.
    """
    monkeypatch.setattr(updater, "pip_available", lambda: True)

    def _refused(*_args, **_kwargs):
        raise PermissionError(13, "Permission denied")

    result = updater.dry_run_install("cuml-cu12", runner=_refused)

    assert result.ok is False
    assert result.requirement == "cuml-cu12"
    assert result.changes == ()
    assert "Permission denied" in result.error
    # And nothing may be installed on the strength of it.
    assert updater.install_decision(result)["allowed"] is False


def test_a_resolver_that_dies_mid_read_is_still_a_readable_refusal(
        monkeypatch):
    """Any other exception out of the runner becomes the reported reason."""
    monkeypatch.setattr(updater, "pip_available", lambda: True)

    def _broken(*_args, **_kwargs):
        raise OSError("Cannot allocate memory")

    result = updater.dry_run_install("torch", runner=_broken)

    assert result.ok is False
    assert result.error == "Cannot allocate memory"
    assert result.raw == ""
    assert "Could not work out what installing torch would change" \
        in result.summary()


# ---------------------------------------------------------------------------
# Offer text
# ---------------------------------------------------------------------------

def test_an_offer_with_no_recipe_is_exactly_its_message():
    """No recipe means no trailing blank line and no second paragraph."""
    offer = updater.offer_ready("GPU UMAP", "cuML is already installed.")

    assert offer.recipe == ""
    assert offer.as_text() == "cuML is already installed."


def test_an_offer_with_no_message_does_not_open_with_a_blank_paragraph():
    """An empty message is dropped rather than becoming a leading gap.

    ``as_text`` is what a dialog body and a log line are both built from, so
    a missing half must leave no separator behind it.
    """
    offer = updater.offer_impossible("GPU UMAP", "  ",
                                     "There is no CUDA device on this host.")

    assert offer.as_text() == "There is no CUDA device on this host."


def test_an_offer_with_a_recipe_puts_it_in_a_second_paragraph():
    """The recipe is appended, separated by one blank line."""
    offer = updater.offer_elsewhere(
        "GPU UMAP", "cuML needs its own environment.",
        "conda create -n rapids -c rapidsai cuml")

    assert offer.as_text() == (
        "cuML needs its own environment.\n\n"
        "conda create -n rapids -c rapidsai cuml")
