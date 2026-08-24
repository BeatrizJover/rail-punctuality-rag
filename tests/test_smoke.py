"""Smoke tests.

Placeholder tests until real unit tests arrive alongside the modules they
cover. This one also verifies the package installs and imports.
"""

from __future__ import annotations


def test_package_importable() -> None:
    """The top-level package imports and exposes a version string."""
    import rail_rag

    assert rail_rag.__version__
