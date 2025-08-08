"""
DuckDB Database for Slack PA
Uses flat, human-readable schema for maximum performance and searchability with columnar storage.
"""

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb

from config import get_project_root
from data.name_resolver import NameResolver, get_name_resolver

logger = logging.getLogger(__name__)

# Database location
PROJECT_ROOT = get_project_root()
DB_PATH = PROJECT_ROOT / "data" / "conversations.duckdb"

# Load optimized schema
SCHEMA_FILE = PROJECT_ROOT / "src" / "data" / "optimized_schema.sql"

# Schedule table for continuous loader
SCHEDULE_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule (
    conversation_id TEXT PRIMARY KEY,
    type TEXT,                -- 'channel' | 'dm'
    interval_sec INTEGER,     -- polling cadence in seconds
    next_run_epoch BIGINT     -- unix timestamp when next fetch should happen
);
"""


class DuckDBConversationDB:
    """DuckDB database with flat, human-readable schema and columnar performance."""

    def __init__(self, db_path: Path | None = None, lazy_resolver: bool = False):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._name_resolver = None
        self._lazy_resolver = lazy_resolver
        self._init_db()
        # Initialize name resolver after database is ready
        if not lazy_resolver:
            self._name_resolver = get_name_resolver(db=self)

    def _init_db(self) -> None:
        """Initialize database with optimized schema."""
        with self.get_connection() as conn:
            # Check if database already exists
            existing_tables = conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()

            if not existing_tables:
                # Load and execute schema only if database is empty
                if SCHEMA_FILE.exists():
                    schema_sql = SCHEMA_FILE.read_text()
                    # DuckDB uses BOOL instead of BOOLEAN
                    schema_sql = schema_sql.replace("BOOLEAN", "BOOL")
                    conn.execute(schema_sql)
                    logger.debug("DuckDB database schema created")
                else:
                    logger.error(f"Schema file not found: {SCHEMA_FILE}")
            else:
                logger.debug(
                    f"Database already exists with {len(existing_tables)} tables"
                )

            # Always ensure schedule table exists
            conn.execute(SCHEDULE_SCHEMA)

            # Note: DuckDB FTS extension is not needed for basic LIKE search
            # We'll use LIKE-based search which is sufficient for this use case
            logger.debug("Using LIKE-based search instead of FTS for compatibility")

        logger.debug(f"DuckDB database initialized at {self.db_path}")

    @property
    def name_resolver(self) -> NameResolver:
        """Get name resolver with database cache, loading lazily if needed."""
        if self._name_resolver is None:
            self._name_resolver = get_name_resolver(db=self)
        return self._name_resolver

    @contextmanager
    def get_connection(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Get database connection with proper error handling."""
        conn = duckdb.connect(str(self.db_path))
        try:
            yield conn
        finally:
            conn.close()

    def add_conversation(
        self,
        conversation_id: str,
        conversation_type: str,
        participants: list[str] | None = None,
        conversation_name: str | None = None,
    ) -> str:
        """Add or update a conversation with human-readable name."""

        # Use provided name if available, otherwise create one via name resolver
        if conversation_name is None:
            conversation_name = self.name_resolver.create_conversation_name(
                conversation_id, conversation_type, participants
            )

        # Resolve participant names
        participant_names = None
        if participants:
            participant_names = [
                self.name_resolver.resolve_user_id(uid) for uid in participants
            ]
            participant_names = [name for name in participant_names if name]

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations
                (id, name, type, participants, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                [
                    conversation_id,
                    conversation_name,
                    conversation_type,
                    json.dumps(participant_names) if participant_names else None,
                    datetime.now().isoformat(),
                    datetime.now().isoformat(),
                ],
            )

        return conversation_name

    def add_message(
        self,
        raw_message: dict,
        conversation_id: str,
        conversation_type: str,
        participants: list[str] | None = None,
    ) -> str:
        """Add a message using optimized flat structure."""

        # First ensure conversation exists
        conversation_name = self.add_conversation(
            conversation_id, conversation_type, participants
        )

        # Process message to optimized format
        processed_msg = self.name_resolver.process_message(
            raw_message, conversation_name, conversation_type
        )

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO messages
                (id, conversation_name, conversation_type, timestamp, datetime,
                 user_name, author_name, text, clean_text, message_type, subtype,
                 thread_ts, is_thread_parent, reply_count, has_attachments,
                 has_reactions, is_edited, edited_ts, mentions_users,
                 contains_links, word_count, search_text)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    processed_msg["id"],
                    processed_msg["conversation_name"],
                    processed_msg["conversation_type"],
                    processed_msg["timestamp"],
                    processed_msg["datetime"],
                    processed_msg["user_name"],
                    processed_msg["author_name"],
                    processed_msg["text"],
                    processed_msg["clean_text"],
                    processed_msg["message_type"],
                    processed_msg["subtype"],
                    processed_msg["thread_ts"],
                    processed_msg["is_thread_parent"],
                    processed_msg["reply_count"],
                    processed_msg["has_attachments"],
                    processed_msg["has_reactions"],
                    processed_msg["is_edited"],
                    processed_msg["edited_ts"],
                    processed_msg["mentions_users"],
                    processed_msg["contains_links"],
                    processed_msg["word_count"],
                    processed_msg["search_text"],
                ],
            )

            # Update conversation stats
            conn.execute(
                """
                UPDATE conversations
                SET
                    updated_at = ?,
                    message_count = (
                        SELECT COUNT(*) FROM messages WHERE conversation_name = ?
                    ),
                    latest_message_ts = ?
                WHERE name = ?
            """,
                [
                    datetime.now().isoformat(),
                    conversation_name,
                    processed_msg["timestamp"],
                    conversation_name,
                ],
            )

        # Process attachments if any
        if processed_msg["has_attachments"]:
            self._add_attachments(raw_message, processed_msg["id"])

        # Process reactions if any
        if processed_msg["has_reactions"]:
            self._add_reactions(raw_message, processed_msg["id"])

        return str(processed_msg["id"])

    def _add_attachments(self, raw_message: dict, message_id: str) -> None:
        """Add attachments for a message."""
        files = raw_message.get("files", [])
        attachments = raw_message.get("attachments", [])

        with self.get_connection() as conn:
            # Process files
            for i, file_info in enumerate(files):
                if not isinstance(file_info, dict):
                    continue

                conn.execute(
                    """
                    INSERT OR REPLACE INTO attachments
                    (id, message_id, file_name, file_type, file_size, url_private,
                     title, is_image, is_video, is_document)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"{message_id}:file:{i}",
                        message_id,
                        file_info.get("name"),
                        file_info.get("filetype"),
                        file_info.get("size"),
                        file_info.get("url_private"),
                        file_info.get("title"),
                        file_info.get("mimetype", "").startswith("image/"),
                        file_info.get("mimetype", "").startswith("video/"),
                        file_info.get("filetype") in ["pdf", "doc", "docx", "txt"],
                    ],
                )

            # Process other attachments
            for i, attachment in enumerate(attachments):
                if not isinstance(attachment, dict):
                    continue

                conn.execute(
                    """
                    INSERT OR REPLACE INTO attachments
                    (id, message_id, file_name, file_type, title, url_private)
                    VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        f"{message_id}:attachment:{i}",
                        message_id,
                        attachment.get("title"),
                        "link",
                        attachment.get("title"),
                        attachment.get("title_link"),
                    ],
                )

    def _add_reactions(self, raw_message: dict, message_id: str) -> None:
        """Add reactions for a message."""
        reactions = raw_message.get("reactions", [])

        with self.get_connection() as conn:
            for reaction in reactions:
                if not isinstance(reaction, dict):
                    continue

                emoji = reaction.get("name")
                users = reaction.get("users", [])

                for user_id in users:
                    user_name = self.name_resolver.resolve_user_id(user_id)
                    if user_name:
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO reactions
                            (message_id, emoji, emoji_name, user_name, timestamp)
                            VALUES (?, ?, ?, ?, ?)
                        """,
                            [
                                message_id,
                                f":{emoji}:",
                                emoji,
                                user_name,
                                datetime.now().timestamp(),
                            ],
                        )

    def get_conversation(self, conversation_name: str) -> dict[str, Any] | None:
        """Get conversation by human-readable name or channel ID."""
        with self.get_connection() as conn:
            # First try exact match by name or ID
            result = conn.execute(
                """
                SELECT * FROM conversations WHERE name = ? OR id = ?
            """,
                [conversation_name, conversation_name],
            ).fetchone()

            if result and conn.description:
                # Convert DuckDB result to dict
                columns = [desc[0] for desc in conn.description]
                row_dict = dict(zip(columns, result, strict=False))
                if row_dict["participants"]:
                    row_dict["participants"] = json.loads(row_dict["participants"])
                return row_dict

            # If not found, try to resolve channel name to ID
            channel_result = conn.execute(
                """
                SELECT slack_id FROM channels WHERE name = ?
                """,
                [conversation_name],
            ).fetchone()

            if channel_result:
                channel_id = channel_result[0]
                # Try again with the resolved channel ID
                result = conn.execute(
                    """
                    SELECT * FROM conversations WHERE name = ? OR id = ?
                """,
                    [channel_id, channel_id],
                ).fetchone()

                if result and conn.description:
                    columns = [desc[0] for desc in conn.description]
                    row_dict = dict(zip(columns, result, strict=False))
                    if row_dict["participants"]:
                        row_dict["participants"] = json.loads(row_dict["participants"])
                    return row_dict

            return None

    def get_messages(
        self,
        conversation_name: str,
        limit: int = 100,
        offset: int = 0,
        since_timestamp: float | None = None,
        until_timestamp: float | None = None,
    ) -> list[dict[str, Any]]:
        """Get messages with human-readable conversation name or channel ID."""
        with self.get_connection() as conn:
            # First try direct lookup
            query = """
                SELECT * FROM messages
                WHERE conversation_name = ?
            """
            params: list[Any] = [conversation_name]

            if since_timestamp:
                query += " AND timestamp >= ?"
                params.append(since_timestamp)

            if until_timestamp:
                query += " AND timestamp <= ?"
                params.append(until_timestamp)

            query += " ORDER BY timestamp ASC LIMIT ? OFFSET ?"
            params.extend([limit, offset])

            results = conn.execute(query, params).fetchall()

            # If no results, try to resolve channel name to ID and search again
            if not results:
                # Look up channel ID by name
                channel_result = conn.execute(
                    "SELECT slack_id FROM channels WHERE name = ?", [conversation_name]
                ).fetchone()

                if channel_result:
                    channel_id = channel_result[0]
                    # Try again with channel ID
                    query = """
                        SELECT * FROM messages
                        WHERE conversation_name = ?
                    """
                    channel_params: list[Any] = [channel_id]

                    if since_timestamp:
                        query += " AND timestamp >= ?"
                        channel_params.append(since_timestamp)

                    if until_timestamp:
                        query += " AND timestamp <= ?"
                        channel_params.append(until_timestamp)

                    query += " ORDER BY timestamp ASC LIMIT ? OFFSET ?"
                    channel_params.extend([limit, offset])

                    results = conn.execute(query, channel_params).fetchall()

            if conn.description:
                columns = [desc[0] for desc in conn.description]
                return [dict(zip(columns, row, strict=False)) for row in results]
            return []

    def search_messages(
        self,
        query_text: str,
        conversation_name: str | None = None,
        user_name: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Fast search using DuckDB's built-in FTS or LIKE if FTS unavailable."""

        # Use LIKE search (FTS extension disabled for compatibility)
        sql_query = """
            SELECT * FROM messages
            WHERE search_text LIKE ? OR clean_text LIKE ?
        """
        search_term = f"%{query_text.lower()}%"
        params: list[Any] = [search_term, search_term]

        if conversation_name:
            sql_query += " AND conversation_name = ?"
            params.append(conversation_name)

        if user_name:
            sql_query += " AND user_name LIKE ?"
            params.append(f"%{user_name}%")

        sql_query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self.get_connection() as conn:
            results = conn.execute(sql_query, params).fetchall()
            if conn.description:
                columns = [desc[0] for desc in conn.description]
                return [dict(zip(columns, row, strict=False)) for row in results]
            return []

    def list_conversations(
        self, conversation_type: str | None = None
    ) -> list[dict[str, Any]]:
        """List conversations with stats."""
        query = """
            SELECT
                c.id,
                c.name,
                c.type,
                c.participants,
                c.created_at,
                c.updated_at,
                c.message_count,
                c.latest_message_ts,
                COUNT(m.id) as actual_message_count,
                MIN(m.datetime) as start_date,
                MAX(m.datetime) as end_date,
                MAX(m.timestamp) as last_message_ts
            FROM conversations c
            LEFT JOIN messages m ON c.name = m.conversation_name
        """
        params = []

        if conversation_type:
            query += " WHERE c.type = ?"
            params.append(conversation_type)

        query += """
            GROUP BY c.id, c.name, c.type, c.participants, c.created_at, c.updated_at, c.message_count, c.latest_message_ts
            ORDER BY last_message_ts DESC NULLS LAST
        """

        with self.get_connection() as conn:
            results = conn.execute(query, params).fetchall()
            if not conn.description:
                return []
            columns = [desc[0] for desc in conn.description]
            conversations = []
            for row in results:
                result = dict(zip(columns, row, strict=False))
                if result["participants"]:
                    result["participants"] = json.loads(result["participants"])
                conversations.append(result)
            return conversations

    def get_stats(self) -> dict[str, Any]:
        """Get database statistics."""
        with self.get_connection() as conn:
            stats: dict[str, Any] = {}

            # Conversation counts by type
            conv_results = conn.execute(
                """
                SELECT type, COUNT(*) as count
                FROM conversations
                GROUP BY type
            """
            ).fetchall()
            conversation_counts = {row[0]: row[1] for row in conv_results}
            stats["conversations"] = conversation_counts

            # Calculate totals for CLI display
            total_convs = sum(conversation_counts.values())
            stats["total_conversations"] = total_convs
            stats["channels"] = conversation_counts.get("channel", 0)
            stats["dms"] = conversation_counts.get("dm", 0)

            # Total users count
            user_total = conn.execute("SELECT COUNT(*) FROM users").fetchone()
            stats["total_users"] = user_total[0] if user_total else 0

            # Message counts
            msg_count = conn.execute(
                "SELECT COUNT(*) as count FROM messages"
            ).fetchone()
            stats["total_messages"] = msg_count[0] if msg_count else 0

            # Recent activity (last 24 hours)
            recent_count = conn.execute(
                """
                SELECT COUNT(*) as count FROM messages
                WHERE timestamp >= ?
            """,
                [datetime.now().timestamp() - 24 * 3600],
            ).fetchone()
            stats["messages_last_24h"] = recent_count[0] if recent_count else 0

            # Thread stats
            thread_stats = conn.execute(
                """
                SELECT
                    COUNT(CASE WHEN is_thread_parent THEN 1 END) as thread_count,
                    COUNT(CASE WHEN thread_ts IS NOT NULL AND NOT is_thread_parent THEN 1 END) as reply_count
                FROM messages
            """
            ).fetchone()
            stats["threads"] = thread_stats[0] if thread_stats else 0
            stats["thread_replies"] = thread_stats[1] if thread_stats else 0

            # User activity
            user_count = conn.execute(
                """
                SELECT COUNT(DISTINCT user_name) as count FROM messages
                WHERE user_name IS NOT NULL
            """
            ).fetchone()
            stats["active_users"] = user_count[0] if user_count else 0

            return stats

    # Name mapping management
    def sync_user_mappings(self, users_data: dict[str, dict[str, Any]]) -> None:
        """Sync user mappings from Slack API to database."""
        with self.get_connection() as conn:
            for slack_id, user_info in users_data.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO users
                    (slack_id, name, display_name, real_name, email, is_bot, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        slack_id,
                        user_info.get("profile", {}).get("display_name")
                        or user_info.get("real_name")
                        or user_info.get("name", ""),
                        user_info.get("name", ""),
                        user_info.get("real_name", ""),
                        user_info.get("profile", {}).get("email", ""),
                        user_info.get("is_bot", False),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ],
                )
        logger.info(f"Synced {len(users_data)} user mappings to database")

    def sync_channel_mappings(self, channels_data: dict[str, dict[str, Any]]) -> None:
        """Sync channel mappings from Slack API to database."""
        with self.get_connection() as conn:
            for slack_id, channel_info in channels_data.items():
                conn.execute(
                    """
                    INSERT OR REPLACE INTO channels
                    (slack_id, name, display_name, purpose, topic, is_private, member_count, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        slack_id,
                        channel_info.get("name", ""),
                        f"#{channel_info.get('name', '')}",
                        channel_info.get("purpose", {}).get("value", ""),
                        channel_info.get("topic", {}).get("value", ""),
                        channel_info.get("is_private", False),
                        channel_info.get("num_members", 0),
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ],
                )
        logger.info(f"Synced {len(channels_data)} channel mappings to database")

    def get_user_mappings(self) -> dict[str, str]:
        """Get all user ID -> name mappings from database."""
        with self.get_connection() as conn:
            results = conn.execute("SELECT slack_id, name FROM users").fetchall()
            return {row[0]: row[1] for row in results}

    def get_channel_mappings(self) -> dict[str, str]:
        """Get all channel ID -> name mappings from database."""
        with self.get_connection() as conn:
            results = conn.execute("SELECT slack_id, name FROM channels").fetchall()
            return {row[0]: row[1] for row in results}

    def resolve_user_id(self, user_id: str) -> str | None:
        """Resolve user ID to name using database cache."""
        if not user_id:
            return None
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT name FROM users WHERE slack_id = ?", [user_id]
            ).fetchone()
            return result[0] if result else user_id

    def resolve_channel_id(self, channel_id: str) -> str | None:
        """Resolve channel ID to name using database cache."""
        if not channel_id:
            return None
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT name FROM channels WHERE slack_id = ?", [channel_id]
            ).fetchone()
            return result[0] if result else channel_id

    def resolve_channel_name_to_id(self, channel_name: str) -> str | None:
        """Resolve channel name to ID using database cache."""
        if not channel_name:
            return None
        with self.get_connection() as conn:
            # Try exact match first
            result = conn.execute(
                "SELECT slack_id FROM channels WHERE name = ?", [channel_name]
            ).fetchone()
            if result:
                return str(result[0])

            # Try with # prefix removed if present
            clean_name = channel_name.lstrip("#")
            result = conn.execute(
                "SELECT slack_id FROM channels WHERE name = ?", [clean_name]
            ).fetchone()
            if result:
                return str(result[0])

            # Try fuzzy matching for common variations
            result = conn.execute(
                """SELECT slack_id FROM channels
                   WHERE name LIKE ? OR name LIKE ?
                   ORDER BY CASE
                       WHEN name = ? THEN 1
                       WHEN name LIKE ? THEN 2
                       ELSE 3
                   END
                   LIMIT 1""",
                [f"%{clean_name}%", f"{clean_name}%", clean_name, f"{clean_name}%"],
            ).fetchone()

            return result[0] if result else None

    def get_mapping_stats(self) -> dict[str, int]:
        """Get statistics about cached mappings."""
        with self.get_connection() as conn:
            user_result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
            user_count = user_result[0] if user_result else 0

            channel_result = conn.execute("SELECT COUNT(*) FROM channels").fetchone()
            channel_count = channel_result[0] if channel_result else 0

            bot_result = conn.execute(
                "SELECT COUNT(*) FROM users WHERE is_bot = true"
            ).fetchone()
            bot_count = bot_result[0] if bot_result else 0

            private_channel_result = conn.execute(
                "SELECT COUNT(*) FROM channels WHERE is_private = true"
            ).fetchone()
            private_channel_count = (
                private_channel_result[0] if private_channel_result else 0
            )

            return {
                "total_users": user_count,
                "total_channels": channel_count,
                "bot_users": bot_count,
                "private_channels": private_channel_count,
            }

    # Schedule management for continuous loader
    def get_next_scheduled_conversation(self) -> dict[str, Any] | None:
        """Get the next conversation that should be fetched."""
        with self.get_connection() as conn:
            result = conn.execute(
                """
                SELECT conversation_id, type, interval_sec
                FROM schedule
                WHERE next_run_epoch <= ?
                ORDER BY next_run_epoch ASC
                LIMIT 1
            """,
                [int(time.time())],
            ).fetchone()

            if result and conn.description:
                columns = [desc[0] for desc in conn.description]
                return dict(zip(columns, result, strict=False))
            return None

    def update_schedule(self, conversation_id: str, interval_sec: int) -> None:
        """Update the schedule for a conversation."""
        next_run = int(time.time()) + interval_sec
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO schedule (conversation_id, type, interval_sec, next_run_epoch)
                VALUES (?, ?, ?, ?)
            """,
                [conversation_id, "unknown", interval_sec, next_run],
            )

    def update_schedule_id(self, old_id: str, new_id: str) -> None:
        """Update conversation_id in schedule table (for resolving channel names to IDs)."""
        with self.get_connection() as conn:
            conn.execute(
                """
                UPDATE schedule
                SET conversation_id = ?
                WHERE conversation_id = ?
            """,
                [new_id, old_id],
            )

    def seed_schedule(self, conversation_schedules: list[dict[str, Any]]) -> None:
        """Seed the schedule table with initial conversation intervals."""
        with self.get_connection() as conn:
            for sched in conversation_schedules:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO schedule (conversation_id, type, interval_sec, next_run_epoch)
                    VALUES (?, ?, ?, ?)
                """,
                    [
                        sched["conversation_id"],
                        sched["type"],
                        sched["interval_sec"],
                        0,  # Start immediately
                    ],
                )


# Global DuckDB database instance
_duckdb_instance: DuckDBConversationDB | None = None


def get_duckdb() -> DuckDBConversationDB:
    """Get global DuckDB database instance."""
    global _duckdb_instance
    if _duckdb_instance is None:
        _duckdb_instance = DuckDBConversationDB()
    return _duckdb_instance


# Compatibility function to maintain existing API
def get_optimized_db(lazy_resolver: bool = False) -> DuckDBConversationDB:
    """Get DuckDB database instance (maintains compatibility with old API)."""
    global _duckdb_instance
    if _duckdb_instance is None:
        _duckdb_instance = DuckDBConversationDB(lazy_resolver=lazy_resolver)
    return _duckdb_instance


async def sync_mappings_to_database() -> dict[str, int]:
    """Sync user and channel mappings from Slack API to database for proper name resolution."""
    import asyncio

    from slack_client.async_client import AsyncSlackClient

    logger.info("Syncing user and channel mappings from Slack API...")
    client = AsyncSlackClient()
    db = get_optimized_db()

    # Fetch users from Slack API
    logger.info("Fetching users from Slack API...")
    users_data = {}
    cursor = None
    while True:
        response = await client._client.users_list(limit=200, cursor=cursor)
        users: list[dict[str, Any]] = response.get("members", [])

        # Convert to ID -> user_info mapping
        batch = {user["id"]: user for user in users if user.get("id")}
        users_data.update(batch)

        metadata: dict[str, Any] = response.get("response_metadata", {})
        cursor = metadata.get("next_cursor") if isinstance(metadata, dict) else None
        if not cursor:
            break

        # Rate limiting courtesy pause
        await asyncio.sleep(1)

    # Fetch channels from Slack API
    logger.info("Fetching channels from Slack API...")
    channels_data = {}
    cursor = None
    while True:
        response = await client._client.conversations_list(
            types="public_channel,private_channel", limit=200, cursor=cursor
        )
        channels: list[dict[str, Any]] = response.get("channels", [])

        # Convert to ID -> channel_info mapping
        batch = {channel["id"]: channel for channel in channels if channel.get("id")}
        channels_data.update(batch)

        metadata2: dict[str, Any] = response.get("response_metadata", {})
        cursor = metadata2.get("next_cursor") if isinstance(metadata2, dict) else None
        if not cursor:
            break

        # Rate limiting courtesy pause
        await asyncio.sleep(1)

    logger.info(
        f"Syncing {len(users_data)} users and {len(channels_data)} channels to database..."
    )
    db.sync_user_mappings(users_data)
    db.sync_channel_mappings(channels_data)

    # Show stats
    stats = db.get_mapping_stats()
    logger.info("Mapping sync complete:")
    logger.info(
        f"  - Total users: {stats['total_users']} (including {stats['bot_users']} bots)"
    )
    logger.info(
        f"  - Total channels: {stats['total_channels']} (including {stats['private_channels']} private)"
    )

    return stats


def update_dm_names_with_participants() -> int:
    """Update existing DM conversation names using participant information and user mappings."""
    from data.name_resolver import get_name_resolver

    logger.info("Updating DM conversation names with participant information...")
    db = get_optimized_db()
    name_resolver = get_name_resolver(db=db)

    updated_count = 0

    with db.get_connection() as conn:
        # Get all DM conversations that need name updates (where name equals ID)
        dm_conversations = conn.execute("""
            SELECT id, name, participants
            FROM conversations
            WHERE type = 'dm' AND (name = id OR name LIKE 'C%' OR name LIKE 'D%')
        """).fetchall()

        logger.info(
            f"Found {len(dm_conversations)} DM conversations that need name updates"
        )

        for dm_id, current_name, participants_json in dm_conversations:
            try:
                # Parse existing participants if available
                participant_ids = []
                if participants_json:
                    import json

                    try:
                        participant_names = json.loads(participants_json)
                        # These might be names, try to get IDs from them
                        for name in participant_names:
                            # Look up user ID by name
                            user_result = conn.execute(
                                "SELECT slack_id FROM users WHERE name = ? OR real_name = ? OR display_name = ?",
                                [name, name, name],
                            ).fetchone()
                            if user_result:
                                participant_ids.append(user_result[0])
                    except (json.JSONDecodeError, TypeError):
                        pass

                # If we don't have participant IDs, try to get them from Slack API conversation info
                if not participant_ids:
                    logger.debug(
                        f"No participant info available for DM {dm_id}, skipping"
                    )
                    continue

                # Create human-readable name using name resolver
                new_name = name_resolver.create_conversation_name(
                    dm_id, "dm", participants=participant_ids
                )

                if new_name != current_name and new_name != dm_id:
                    # Update the conversation name
                    conn.execute(
                        "UPDATE conversations SET name = ?, updated_at = ? WHERE id = ?",
                        [new_name, datetime.now().isoformat(), dm_id],
                    )

                    # Update any existing messages to use the new conversation name
                    conn.execute(
                        "UPDATE messages SET conversation_name = ? WHERE conversation_name = ?",
                        [new_name, current_name],
                    )

                    logger.info(f"Updated DM name: {current_name} -> {new_name}")
                    updated_count += 1

            except Exception as e:
                logger.warning(f"Failed to update DM {dm_id}: {e}")
                continue

    logger.info(f"Updated {updated_count} DM conversation names")
    return updated_count


def fix_message_conversation_names() -> int:
    """Fix message conversation names to match the human-readable names from conversations table."""

    logger.info("Fixing message conversation names to use human-readable names...")
    db = get_optimized_db()

    updated_count = 0

    with db.get_connection() as conn:
        # Get all mismatched messages where conversation_name is a Slack ID but should be human name
        mismatched_messages = conn.execute("""
            SELECT DISTINCT m.conversation_name, m.conversation_type, c.name as correct_name
            FROM messages m
            LEFT JOIN conversations c ON (
                (m.conversation_type = 'channel' AND m.conversation_name = c.id AND c.type = 'channel') OR
                (m.conversation_type = 'dm' AND m.conversation_name LIKE 'DM:%' AND c.name = m.conversation_name)
            )
            WHERE c.name IS NOT NULL
               AND m.conversation_name != c.name
               AND (m.conversation_name LIKE 'C%' OR m.conversation_name LIKE 'D%')
        """).fetchall()

        logger.info(
            f"Found {len(mismatched_messages)} conversation name mismatches to fix"
        )

        for old_name, conv_type, correct_name in mismatched_messages:
            try:
                # Update messages to use correct conversation name
                result = conn.execute(
                    "UPDATE messages SET conversation_name = ? WHERE conversation_name = ?",
                    [correct_name, old_name],
                )

                # Get number of affected rows (DuckDB specific)
                _ = result.fetchall() if hasattr(result, "fetchall") else 0

                logger.info(
                    f"Updated {conv_type} messages: '{old_name}' -> '{correct_name}'"
                )
                updated_count += 1

            except Exception as e:
                logger.warning(f"Failed to update messages for {old_name}: {e}")
                continue

    logger.info(f"Fixed {updated_count} message conversation name mismatches")
    return updated_count


def fix_dm_conversation_names() -> int:
    """Fix DM conversation names to match message naming format and add participant information."""

    logger.info("Fixing DM conversation names to match message format...")
    db = get_optimized_db()

    updated_count = 0

    with db.get_connection() as conn:
        # Get all DM conversations that need fixing
        broken_dms = conn.execute("""
            SELECT c.id, c.name, c.participants
            FROM conversations c
            WHERE c.type = 'dm'
              AND c.name NOT LIKE 'DM:%'
              AND (c.name LIKE 'U%' OR c.name LIKE 'D%')
        """).fetchall()

        logger.info(f"Found {len(broken_dms)} DM conversations with incorrect names")

        for dm_id, current_name, _participants in broken_dms:
            try:
                # Create proper DM conversation name in "DM: D[ID]" format
                correct_dm_name = f"DM: {dm_id}"

                # Try to get participant information from messages
                message_users = conn.execute(
                    """
                    SELECT DISTINCT user_name
                    FROM messages
                    WHERE conversation_name LIKE ? AND user_name IS NOT NULL
                    ORDER BY user_name
                """,
                    [f"%{dm_id}%"],
                ).fetchall()

                participant_names = [
                    user[0] for user in message_users if user[0] and user[0] != "None"
                ]

                # Update conversation with correct name and participants
                participant_json = (
                    json.dumps(participant_names) if participant_names else None
                )

                conn.execute(
                    """
                    UPDATE conversations
                    SET name = ?, participants = ?, updated_at = ?
                    WHERE id = ?
                """,
                    [
                        correct_dm_name,
                        participant_json,
                        datetime.now().isoformat(),
                        dm_id,
                    ],
                )

                logger.info(
                    f"Fixed DM: {current_name} -> {correct_dm_name} (participants: {participant_names})"
                )
                updated_count += 1

            except Exception as e:
                logger.warning(f"Failed to fix DM {dm_id}: {e}")
                continue

    logger.info(f"Fixed {updated_count} DM conversation names")
    return updated_count


def create_human_readable_dm_names() -> int:
    """Create human-readable DM names using participant information."""

    logger.info("Creating human-readable DM names from participant data...")
    db = get_optimized_db()

    updated_count = 0

    with db.get_connection() as conn:
        # Get DMs with participant data
        dms_with_participants = conn.execute("""
            SELECT id, name, participants
            FROM conversations
            WHERE type = 'dm'
              AND participants IS NOT NULL
              AND participants != 'null'
              AND participants != '[]'
        """).fetchall()

        logger.info(f"Found {len(dms_with_participants)} DMs with participant data")

        for dm_id, current_name, participants_json in dms_with_participants:
            try:
                # Parse participant user IDs
                participant_ids = json.loads(participants_json)
                if not participant_ids:
                    continue

                # Resolve user IDs to human names
                participant_names = []
                for user_id in participant_ids:
                    if user_id == "USLACKBOT":
                        continue  # Skip Slackbot for cleaner names

                    human_name = db.resolve_user_id(user_id)
                    if human_name and human_name != user_id:
                        participant_names.append(human_name)
                    else:
                        participant_names.append(user_id)  # Fallback to ID

                if not participant_names:
                    continue  # Skip if no valid participants

                # Create human-readable DM name (clean format without "DM:" prefix)
                if len(participant_names) == 1:
                    human_dm_name = participant_names[0]
                elif len(participant_names) == 2:
                    human_dm_name = f"{participant_names[0]} & {participant_names[1]}"
                else:
                    human_dm_name = f"{participant_names[0]}, {participant_names[1]} + {len(participant_names) - 2} others"

                # Update conversation name if it's different
                if human_dm_name != current_name:
                    conn.execute(
                        """
                        UPDATE conversations
                        SET name = ?, updated_at = ?
                        WHERE id = ?
                    """,
                        [human_dm_name, datetime.now().isoformat(), dm_id],
                    )

                    # Update any existing messages to use the new conversation name
                    conn.execute(
                        """
                        UPDATE messages
                        SET conversation_name = ?
                        WHERE conversation_name = ?
                    """,
                        [human_dm_name, current_name],
                    )

                    logger.info(f"Updated DM: {current_name} -> {human_dm_name}")
                    updated_count += 1

            except Exception as e:
                logger.warning(f"Failed to create human name for DM {dm_id}: {e}")
                continue

    logger.info(f"Created {updated_count} human-readable DM names")
    return updated_count
