"""Wyndle ingestion package.

This package contains modules responsible for fetching data from Slack and loading it
into the project database. It provides a clean separation between
API interactions (`slack`), transformation/ingestion logic (`ingest`),
and storage layers (`storage`).
"""

from __future__ import annotations

__all__: list[str] = [
    "loader",
]
