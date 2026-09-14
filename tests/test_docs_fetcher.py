"""Tests for the pinned document fetcher.

The HTTP layer is replaced at the transport boundary rather than by patching the
module: what is asserted is that a request was or was not made, which is the whole
point of an idempotent fetch.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import httpx
import pytest
from pydantic import HttpUrl

from rail_rag.ingestion.docs.exceptions import FetchError
from rail_rag.ingestion.docs.fetcher import (
    artefact_path,
    fetch_source,
    manifest_path,
    probe_digest,
)
from rail_rag.ingestion.docs.sources import DocumentSource

_BODY = b"%PDF-1.7 pretend this is a glossary"
_DIGEST = hashlib.sha256(_BODY).hexdigest()


def _source(digest: str = _DIGEST) -> DocumentSource:
    return DocumentSource(
        source_id="example_glossary",
        title="Example Glossary",
        publisher="Example Publisher",
        landing_page=HttpUrl("https://example.org/landing"),
        canonical_url=HttpUrl("https://example.org/files/example_20260101.pdf"),
        version="20260101",
        language="en",
        document_type="glossary",
        sha256=digest,
    )


def _client(
    calls: list[httpx.Request],
    *,
    status: int = 200,
    body: bytes = _BODY,
) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(status, content=body)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_a_new_source_is_downloaded_with_its_manifest(tmp_path: Path) -> None:
    calls: list[httpx.Request] = []
    with _client(calls) as client:
        outcome = fetch_source(_source(), tmp_path, client=client)

    assert outcome.downloaded is True
    assert outcome.path == tmp_path / "example_glossary__20260101.pdf"
    assert outcome.path.read_bytes() == _BODY
    assert len(calls) == 1

    manifest = outcome.manifest
    assert manifest.sha256 == _DIGEST
    assert manifest.bytes == len(_BODY)
    assert manifest.version == "20260101"
    assert manifest.source_id == "example_glossary"
    assert manifest.canonical_url.endswith("example_20260101.pdf")


def test_the_manifest_timestamp_is_timezone_aware_utc(tmp_path: Path) -> None:
    """A naive timestamp is unreadable across machines; provenance must be absolute."""
    with _client([]) as client:
        outcome = fetch_source(_source(), tmp_path, client=client)
    assert outcome.manifest.fetched_at.tzinfo is not None
    assert outcome.manifest.fetched_at.utcoffset() == dt.timedelta(0)


def test_the_manifest_survives_a_round_trip_on_disk(tmp_path: Path) -> None:
    with _client([]) as client:
        outcome = fetch_source(_source(), tmp_path, client=client)
    reread = manifest_path(outcome.path).read_text(encoding="utf-8")
    assert _DIGEST in reread


def test_a_second_run_makes_no_request(tmp_path: Path) -> None:
    """Idempotence is observable at the transport, not only in the log."""
    with _client([]) as first:
        fetch_source(_source(), tmp_path, client=first)

    calls: list[httpx.Request] = []
    with _client(calls) as second:
        outcome = fetch_source(_source(), tmp_path, client=second)

    assert calls == []
    assert outcome.downloaded is False
    assert outcome.manifest.sha256 == _DIGEST


def test_a_non_200_response_leaves_nothing_behind(tmp_path: Path) -> None:
    with _client([], status=404) as client, pytest.raises(FetchError, match="HTTP 404"):
        fetch_source(_source(), tmp_path, client=client)
    assert list(tmp_path.iterdir()) == []


def test_a_digest_mismatch_leaves_nothing_behind(tmp_path: Path) -> None:
    """A partial file that survived a rejected fetch is a file something will parse."""
    with (
        _client([], body=b"a different document") as client,
        pytest.raises(FetchError, match="does not match the pinned"),
    ):
        fetch_source(_source(), tmp_path, client=client)
    assert list(tmp_path.iterdir()) == []


def test_an_altered_artefact_is_refused_and_left_alone(tmp_path: Path) -> None:
    """A new upstream version is declared in the registry; it is never inferred here."""
    with _client([]) as client:
        outcome = fetch_source(_source(), tmp_path, client=client)
    outcome.path.write_bytes(b"tampered")

    calls: list[httpx.Request] = []
    with (
        _client(calls) as client,
        pytest.raises(FetchError, match="Nothing was overwritten"),
    ):
        fetch_source(_source(), tmp_path, client=client)

    assert calls == []
    assert outcome.path.read_bytes() == b"tampered"


def test_an_artefact_without_a_manifest_is_refused(tmp_path: Path) -> None:
    """Provenance invented after the fact reads as true and is not."""
    with _client([]) as client:
        outcome = fetch_source(_source(), tmp_path, client=client)
    manifest_path(outcome.path).unlink()

    with (
        _client([]) as client,
        pytest.raises(FetchError, match="provenance cannot be reconstructed"),
    ):
        fetch_source(_source(), tmp_path, client=client)


def test_a_transport_failure_becomes_a_fetch_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    with (
        httpx.Client(transport=httpx.MockTransport(handler)) as client,
        pytest.raises(FetchError, match="Could not download"),
    ):
        fetch_source(_source(), tmp_path, client=client)
    assert list(tmp_path.iterdir()) == []


def test_the_probe_returns_the_digest_and_keeps_nothing(tmp_path: Path) -> None:
    """Bootstrap must not seed the artefact: the pin is reviewed before it is trusted."""
    unpinned = _source(digest="0" * 64)
    with _client([]) as client:
        assert probe_digest(unpinned, client=client) == _DIGEST
    assert list(tmp_path.iterdir()) == []
    assert not artefact_path(unpinned, tmp_path).exists()
