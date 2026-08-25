"""Custom exception hierarchy for the project."""
from __future__ import annotations


class RailRagError(Exception):
    """Base class for all application-specific errors."""


class ConfigError(RailRagError):
    """Raised when configuration is missing or invalid."""


class DatabaseError(RailRagError):
    """Raised when a database operation fails."""
