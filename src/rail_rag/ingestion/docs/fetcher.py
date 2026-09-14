"""Reproducible acquisition of the documents declared in the source registry.

The fetch is pinned, not trusting. The digest a download must produce is declared
in ``config/sources.yaml`` and reviewed in a commit, so a republished file fails
loudly instead of entering the corpus under the old version's name. Bytes reach
their final path only after the digest matches - the download lands in a sibling
``.part`` file first - so an interrupted or rejected fetch never leaves something
parseable behind.

The manifest written next to the artefact is provenance, not authority:
verification is always against the registry, never against the manifest.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from rail_rag.ingestion.docs.exceptions import FetchError
from rail_rag.ingestion.docs.sources import DocumentSource

logger = logging.getLogger(__name__)

#: Repository-relative default; the artefacts themselves are git-ignored.
DEFAULT_RAW_DIR = Path("data/docs/raw")

#: Generous on read: these are multi-megabyte PDFs on a public web server.
_TIMEOUT = httpx.Timeout(60.0, connect=10.0)
#: Bounds peak memory independently of the file size.
_CHUNK_BYTES = 64 * 1024


class FetchManifest(BaseModel):
    """Provenance of one downloaded artefact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    canonical_url: str
    sha256: str
    bytes: int
    fetched_at: dt.datetime
    version: str


@dataclass(frozen=True)
class FetchOutcome:
    """What the fetch did, for the caller to report on."""

    path: Path
    manifest: FetchManifest
    downloaded: bool


def artefact_path(source: DocumentSource, dest_dir: Path) -> Path:
    """Where the pinned artefact for this source lives."""
    return dest_dir / f"{source.file_stem}{source.file_extension}"


def manifest_path(artefact: Path) -> Path:
    """The provenance file that sits beside an artefact."""
    return artefact.with_suffix(".manifest.json")


def sha256_of(path: Path) -> str:
    """Digest a file without reading it whole into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_source(
    source: DocumentSource,
    dest_dir: Path = DEFAULT_RAW_DIR,
    *,
    client: httpx.Client | None = None,
) -> FetchOutcome:
    """Ensure the pinned artefact is on disk, downloading only when it is absent.

    Raises:
        FetchError: if the download fails, if the bytes do not match the pinned
            digest, or if an artefact is present without its manifest.
    """
    artefact = artefact_path(source, dest_dir)
    if artefact.exists():
        return _verified_outcome(source, artefact)

    dest_dir.mkdir(parents=True, exist_ok=True)
    partial = artefact.with_name(artefact.name + ".part")
    with _http_client(client) as http:
        digest, size = _stream_to(str(source.canonical_url), partial, http)

    if digest != source.sha256:
        partial.unlink(missing_ok=True)
        raise FetchError(
            f"{source.source_id}: downloaded digest {digest} does not match the pinned "
            f"{source.sha256}. Declare the new version in the registry before fetching it."
        )

    manifest = FetchManifest(
        source_id=source.source_id,
        canonical_url=str(source.canonical_url),
        sha256=digest,
        bytes=size,
        fetched_at=dt.datetime.now(dt.UTC),
        version=source.version,
    )
    os.replace(partial, artefact)
    manifest_path(artefact).write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    logger.info("Fetched %s (%d bytes) into %s", source.source_id, size, artefact)
    return FetchOutcome(path=artefact, manifest=manifest, downloaded=True)


def probe_digest(source: DocumentSource, *, client: httpx.Client | None = None) -> str:
    """Download the source to a temporary file and return its digest, keeping nothing.

    The bootstrap step for a new entry: a pin is a human decision, so the digest is
    printed for review rather than written into the registry by the program itself.

    Raises:
        FetchError: if the download fails.
    """
    with tempfile.TemporaryDirectory() as scratch:
        target = Path(scratch) / "probe"
        with _http_client(client) as http:
            digest, size = _stream_to(str(source.canonical_url), target, http)
    logger.info("%s: %d bytes, sha256=%s", source.source_id, size, digest)
    return digest


@contextmanager
def _http_client(client: httpx.Client | None) -> Iterator[httpx.Client]:
    """Use the caller's client, or own one for the duration of the call."""
    if client is not None:
        yield client
        return
    with httpx.Client(timeout=_TIMEOUT, follow_redirects=True) as owned:
        yield owned


def _stream_to(url: str, destination: Path, client: httpx.Client) -> tuple[str, int]:
    """Download ``url`` into ``destination``, returning its digest and its size.

    Raises:
        FetchError: on any transport failure or non-200 response.
    """
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream("GET", url, follow_redirects=True) as response:
            if response.status_code != httpx.codes.OK:
                raise FetchError(f"GET {url} returned HTTP {response.status_code}")
            with destination.open("wb") as handle:
                for chunk in response.iter_bytes(_CHUNK_BYTES):
                    digest.update(chunk)
                    size += len(chunk)
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        destination.unlink(missing_ok=True)
        raise FetchError(f"Could not download {url}: {exc}") from exc
    return digest.hexdigest(), size


def _verified_outcome(source: DocumentSource, artefact: Path) -> FetchOutcome:
    """Accept an artefact already on disk, or refuse to work with it.

    Raises:
        FetchError: if the file no longer matches its pin, or has no manifest.
    """
    digest = sha256_of(artefact)
    if digest != source.sha256:
        raise FetchError(
            f"{source.source_id}: {artefact} has digest {digest}, not the pinned "
            f"{source.sha256}. Nothing was overwritten."
        )
    return FetchOutcome(path=artefact, manifest=_read_manifest(artefact), downloaded=False)


def _read_manifest(artefact: Path) -> FetchManifest:
    """Load the provenance of an artefact.

    A missing manifest is fatal rather than regenerated: a ``fetched_at`` invented
    now would be provenance that reads as true and is not.

    Raises:
        FetchError: if the manifest is missing, unreadable or invalid.
    """
    path = manifest_path(artefact)
    try:
        return FetchManifest.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FetchError(
            f"{artefact} is present but {path.name} is missing; its provenance cannot be "
            f"reconstructed. Delete the artefact and fetch it again."
        ) from exc
    except ValidationError as exc:
        raise FetchError(f"Invalid manifest at {path}: {exc}") from exc
