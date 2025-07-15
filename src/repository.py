"""Database repository wrapping DuckDB database for type-safe operations."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import AbstractContextManager

import duckdb

from data.duckdb_database import DuckDBConversationDB, get_optimized_db
from models import SlackConversation, SlackMessage

SYNC_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sync_state (
    conversation_id TEXT PRIMARY KEY,
    last_ts REAL
);
"""


class StorageRepository:
    """High-level database facade used by ingestion layer."""

    def __init__(self, lazy_resolver: bool = False) -> None:
        self._db: DuckDBConversationDB = get_optimized_db(lazy_resolver=lazy_resolver)
        self._ensure_sync_table()

    # ------------------------------------------------------------------
    # Sync-state helpers
    # ------------------------------------------------------------------
    def _ensure_sync_table(self) -> None:
        conn: duckdb.DuckDBPyConnection
        with self._db.get_connection() as conn:
            conn.execute(SYNC_TABLE_SQL)

    def get_last_synced_ts(self, conversation_id: str) -> float:
        """Return the newest timestamp we have stored for a conversation."""
        conn: duckdb.DuckDBPyConnection
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT last_ts FROM sync_state WHERE conversation_id = ?",
                [conversation_id],
            ).fetchone()
            return float(row[0]) if row else 0.0

    def update_last_synced_ts(self, conversation_id: str, last_ts: float) -> None:
        """Upsert latest timestamp after successful load."""
        conn: duckdb.DuckDBPyConnection
        with self._db.get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sync_state(conversation_id, last_ts) "
                "VALUES(?, ?)",
                [conversation_id, last_ts],
            )

    # ------------------------------------------------------------------
    # Data insertion
    # ------------------------------------------------------------------
    def bulk_insert_messages(self, messages: Sequence[SlackMessage]) -> None:
        """Insert many messages using DuckDB's optimized add_message method."""
        if not messages:
            return

        # Use DuckDB's add_message method which handles the new schema properly
        for m in messages:
            # Convert SlackMessage to raw message format expected by DuckDB
            raw_message = {
                "ts": str(m.timestamp),
                "user": m.user_id,
                "text": m.text,
                "thread_ts": m.thread_ts,
            }

            # Include the original raw data if available
            if m.raw:
                raw_message.update(m.raw)

            # Determine conversation type based on conversation_id
            conversation_type = "dm" if m.conversation_id.startswith("D") else "channel"

            # Use DuckDB's add_message method which properly handles schema
            # and name resolution
            self._db.add_message(
                raw_message=raw_message,
                conversation_id=m.conversation_id,
                conversation_type=conversation_type,
                participants=None,  # Will be resolved by name resolver if needed
            )

    # ------------------------------------------------------------------
    # Conversations
    # ------------------------------------------------------------------
    def upsert_conversation(self, conv: SlackConversation) -> None:
        # Use DuckDB's add_conversation method which handles the new schema
        self._db.add_conversation(
            conversation_id=conv.id,
            conversation_type=conv.type,
            participants=conv.participants,
            conversation_name=conv.name,
        )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def get_connection(
        self,
    ) -> AbstractContextManager[duckdb.DuckDBPyConnection]:  # pragma: no cover
        """Return a raw database connection (caller must close)."""
        return self._db.get_connection()

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------
    def reset_sync_state(self) -> None:
        """Delete all sync_state rows."""
        conn: duckdb.DuckDBPyConnection
        with self._db.get_connection() as conn:
            conn.execute("DELETE FROM sync_state")
