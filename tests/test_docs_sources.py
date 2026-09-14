"""Tests for the external source registry."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from rail_rag.core.exceptions import ConfigError
from rail_rag.ingestion.docs.sources import load_registry, load_source

#: Resolved from ``__file__``: the autouse settings fixture chdirs into a tmp_path.
REPO_REGISTRY = Path(__file__).resolve().parent.parent / "config" / "sources.yaml"

_DIGEST = "a" * 64
_OTHER_DIGEST = "b" * 64

_MINIMAL = f"""
sources:
  - source_id: example_glossary
    title: Example Glossary
    publisher: Example Publisher
    landing_page: https://example.org/landing
    canonical_url: https://example.org/files/example_20260101.pdf
    version: "20260101"
    language: en
    document_type: glossary
    sha256: "{_DIGEST}"
"""

_SECOND_ENTRY = f"""  - source_id: example_prose
    title: Example Prose
    publisher: Example Publisher
    landing_page: https://example.org/landing
    canonical_url: https://example.org/files/example_prose.pdf
    version: "1.0"
    language: fr
    document_type: prose
    sha256: "{_OTHER_DIGEST}"
"""

_PARSE_BLOCK = """    parse:
      sections:
        - heading: Definitions
          columns: [term, definition, source]
        - heading: Explanation of abbreviations
          columns: [abbreviation, expansion]          
"""


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "sources.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_repo_registry_is_valid() -> None:
    """The committed file must always load; it is what the CLI boots with."""
    registry = load_registry(REPO_REGISTRY)
    assert registry.names


def test_a_source_is_selected_by_identifier(tmp_path: Path) -> None:
    source = load_source(_write(tmp_path, _MINIMAL), "example_glossary")
    assert source.publisher == "Example Publisher"
    assert source.document_type == "glossary"


def test_the_stored_name_carries_the_version(tmp_path: Path) -> None:
    """The artefact on disk must say which version it is, not just which source."""
    source = load_source(_write(tmp_path, _MINIMAL), "example_glossary")
    assert source.file_stem == "example_glossary__20260101"
    assert source.file_extension == ".pdf"


def test_an_unknown_source_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Unknown source"):
        load_source(_write(tmp_path, _MINIMAL), "absent")


def test_a_duplicated_identifier_is_rejected(tmp_path: Path) -> None:
    """A list makes duplicates visible; a YAML mapping would keep the last silently."""
    with pytest.raises(ConfigError, match="duplicate source_id"):
        load_registry(_write(tmp_path, _MINIMAL + _MINIMAL.split("sources:\n")[1]))


def test_two_distinct_sources_coexist(tmp_path: Path) -> None:
    """Multi-source is the point of the registry; one entry must not be a special case."""
    registry = load_registry(_write(tmp_path, _MINIMAL + _SECOND_ENTRY))
    assert registry.names == ["example_glossary", "example_prose"]


def test_a_malformed_url_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace("https://example.org/landing", "not-a-url")
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, body))


def test_a_url_without_a_file_extension_is_rejected(tmp_path: Path) -> None:
    """The extension names the artefact on disk; guessing one would be silent."""
    body = _MINIMAL.replace("/files/example_20260101.pdf", "/files/latest")
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, body))


def test_an_empty_version_is_rejected(tmp_path: Path) -> None:
    body = _MINIMAL.replace('version: "20260101"', 'version: ""')
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, body))


def test_a_version_that_would_escape_the_directory_is_rejected(tmp_path: Path) -> None:
    """``version`` becomes part of a path, so a separator in it is a traversal."""
    body = _MINIMAL.replace('version: "20260101"', 'version: "../../etc/passwd"')
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, body))


@pytest.mark.parametrize("digest", ["deadbeef", "A" * 64, "z" * 64])
def test_a_malformed_digest_is_rejected(tmp_path: Path, digest: str) -> None:
    body = _MINIMAL.replace(_DIGEST, digest)
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, body))


def test_an_empty_registry_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, "sources: []\n"))


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """``extra=forbid`` turns a typo into a load failure, not a silent default."""
    body = _MINIMAL + "    publisherr: Typo\n"
    with pytest.raises(ConfigError, match="Invalid source registry"):
        load_registry(_write(tmp_path, body))


def test_missing_file_raises_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_registry(tmp_path / "absent.yaml")


def test_non_mapping_document_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_registry(_write(tmp_path, "- just\n- a list\n"))


def test_malformed_yaml_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="Could not read"):
        load_registry(_write(tmp_path, "sources: [unclosed\n"))


def test_a_source_is_immutable(tmp_path: Path) -> None:
    """Frozen entries keep the pin from drifting after it has been validated."""
    source = load_source(_write(tmp_path, _MINIMAL), "example_glossary")
    with pytest.raises(ValidationError):
        source.sha256 = _OTHER_DIGEST


def test_a_source_may_declare_no_structure(tmp_path: Path) -> None:
    """Fetching and parsing are separate capabilities: a source can be pinned and
    downloaded before anyone has described what is inside it."""
    assert load_source(_write(tmp_path, _MINIMAL), "example_glossary").parse is None


def test_parsing_an_undescribed_source_is_a_loud_error(tmp_path: Path) -> None:
    """The absence must surface where the parse is attempted, not as a None later."""
    source = load_source(_write(tmp_path, _MINIMAL), "example_glossary")
    with pytest.raises(ConfigError, match="declares no 'parse' block"):
        source.parse_spec()


def test_sections_are_declared_in_document_order(tmp_path: Path) -> None:
    source = load_source(_write(tmp_path, _MINIMAL + _PARSE_BLOCK), "example_glossary")
    assert source.parse_spec().headings == ["Definitions", "Explanation of abbreviations"]


def test_each_section_carries_its_own_columns(tmp_path: Path) -> None:
    """The two tables in the real document have different shapes; one column list
    for the whole file would have to be wrong about one of them."""
    spec = load_source(_write(tmp_path, _MINIMAL + _PARSE_BLOCK), "example_glossary").parse_spec()
    assert spec.select("Definitions").columns == ["term", "definition", "source"]
    assert spec.select("Explanation of abbreviations").columns == ["abbreviation", "expansion"]


def test_an_undeclared_section_is_rejected(tmp_path: Path) -> None:
    spec = load_source(_write(tmp_path, _MINIMAL + _PARSE_BLOCK), "example_glossary").parse_spec()
    with pytest.raises(ConfigError, match="Undeclared section"):
        spec.select("Appendix B")


def test_a_duplicated_column_name_is_rejected(tmp_path: Path) -> None:
    """Duplicates would collapse two columns into one key and lose a column silently."""
    body = (
        _MINIMAL
        + "    parse:\n      sections:\n        - heading: D\n          columns: [a, b, a]\n"
    )
    with pytest.raises(ConfigError, match="duplicate column name: a"):
        load_registry(_write(tmp_path, body))


def test_a_duplicated_section_heading_is_rejected(tmp_path: Path) -> None:
    body = (
        _MINIMAL
        + "    parse:\n      sections:\n        - heading: D\n          columns: [a]\n"
        + "        - heading: D\n          columns: [b]\n"
    )
    with pytest.raises(ConfigError, match="duplicate section heading: D"):
        load_registry(_write(tmp_path, body))


def test_the_repo_registry_describes_the_glossary() -> None:
    """The committed file is what the CLI boots with; its parse block must be usable."""
    spec = load_registry(REPO_REGISTRY).select("infrabel_ns_glossary").parse_spec()
    assert spec.select("Definitions").columns == ["term", "definition", "source"]
