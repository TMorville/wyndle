"""Incremental Slack data loader.

Fetches Slack messages since the last stored timestamp per conversation and writes
results directly to the optimized database via the StorageRepository.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from models import SlackConversation, SlackMessage
from repository import StorageRepository
from slack_client.async_client import AsyncSlackClient

log = logging.getLogger(__name__)


async def ensure_mappings_synced(
    client: AsyncSlackClient, repo: StorageRepository
) -> None:
    """Ensure user and channel mappings are synced to enable proper name resolution."""
    from data.duckdb_database import get_duckdb

    db = get_duckdb()

    # Check if we already have mappings
    with db.get_connection() as conn:
        user_result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        user_count = user_result[0] if user_result else 0
        channel_result = conn.execute("SELECT COUNT(*) FROM channels").fetchone()
        channel_count = channel_result[0] if channel_result else 0

    if user_count > 0 and channel_count > 0:
        log.info(
            f"Mappings already synced: {user_count} users, {channel_count} channels"
        )
        return

    log.info("Syncing user and channel mappings for proper name resolution...")

    # Fetch users from Slack API
    log.info("Fetching users from Slack API...")
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
    log.info("Fetching channels from Slack API...")
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

    user_count = len(users_data)
    channel_count = len(channels_data)
    log.info(f"Syncing {user_count} users and {channel_count} channels to database...")
    db.sync_user_mappings(users_data)
    db.sync_channel_mappings(channels_data)

    # Show stats
    stats = db.get_mapping_stats()
    log.info("Mapping sync complete:")
    total_users = stats["total_users"]
    bot_users = stats["bot_users"]
    log.info(f"  - Total users: {total_users} (including {bot_users} bots)")
    total_channels = stats["total_channels"]
    private_channels = stats["private_channels"]
    log.info(
        f"  - Total channels: {total_channels} (including {private_channels} private)"
    )


class LoaderConfig(BaseModel):
    channels: list[str] = []  # noqa: RUF012 (pydantic handles default)
    sync_days: int = 30
    parallel_requests: int = 6
    save_threads: bool = True


async def _fetch_and_store(
    client: AsyncSlackClient,
    repo: StorageRepository,
    conversation_id: str,
    conversation_name: str,
    conversation_type: str,
    cfg: LoaderConfig,
) -> None:
    """Fetch messages for a single conversation and persist them."""
    oldest = 0.0
    if cfg.sync_days:
        oldest = (datetime.now() - timedelta(days=cfg.sync_days)).timestamp()

    # Incremental sync state check
    last_ts = repo.get_last_synced_ts(conversation_id)
    oldest = max(oldest, last_ts)

    try:
        messages_raw = await client.fetch_history(
            conversation_id,
            oldest=oldest,
            fetch_threads=cfg.save_threads and conversation_type == "channel",
        )
    except Exception as e:
        # Handle API errors gracefully - log and continue
        error_msg = str(e)
        if "channel_not_found" in error_msg or "not_in_channel" in error_msg:
            log.warning(
                "Skipping %s '%s' (%s): channel not found or no access",
                conversation_type,
                conversation_name,
                conversation_id,
            )
        elif "account_inactive" in error_msg:
            log.warning(
                "Skipping %s '%s' (%s): account inactive",
                conversation_type,
                conversation_name,
                conversation_id,
            )
        else:
            log.error(
                "Failed to fetch %s '%s' (%s): %s",
                conversation_type,
                conversation_name,
                conversation_id,
                error_msg,
            )
        return

    messages: list[SlackMessage] = []
    for m in messages_raw:
        ts_val = m.get("ts")
        if ts_val is None:
            # Skip malformed message
            continue
        msg_ts = float(ts_val)
        message = SlackMessage(
            message_id=str(ts_val),
            conversation_id=conversation_id,
            user_id=m.get("user"),
            text=m.get("text", ""),
            timestamp=msg_ts,
            thread_ts=m.get("thread_ts"),
            raw=m,
        )
        messages.append(message)

    repo.bulk_insert_messages(messages)

    if messages:
        repo.update_last_synced_ts(conversation_id, max(m.timestamp for m in messages))

    log.info("Synced %s messages for %s", len(messages), conversation_name)


async def async_run_loader(
    config_path: str | Path, reset_sync_state: bool = False
) -> None:
    """Asynchronous entry point for the incremental loader."""
    import yaml

    with open(config_path) as f:
        config_yaml: dict[str, Any] = yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # Back-compat: merge legacy top-level `channels:` key if present
    # ------------------------------------------------------------------
    slack_cfg = config_yaml.get("slack", {})
    legacy_channels: list[str] = config_yaml.get("channels", [])

    if legacy_channels:
        import warnings

        msg = (
            "Top-level 'channels' key in config.yaml is deprecated; "
            "move it under 'slack.channels'"
        )
        warnings.warn(msg, stacklevel=2)
        merged = list({*slack_cfg.get("channels", []), *legacy_channels})
        slack_cfg["channels"] = merged

    cfg = LoaderConfig(**slack_cfg)

    repo = StorageRepository()
    if reset_sync_state:
        repo.reset_sync_state()

    client = AsyncSlackClient()
    await client.test_auth()

    # Ensure user and channel mappings are available for name resolution
    await ensure_mappings_synced(client, repo)

    # Build channel and DM maps using the async client
    channel_map, dm_participants = await asyncio.gather(
        client.get_all_channels_map(), client.get_all_dms_with_participants()
    )

    tasks = []

    # Channels
    for ch_name in cfg.channels:
        ch_id = channel_map.get(ch_name)

        # If not found in API response, try database cache as fallback
        if not ch_id:
            from data.duckdb_database import get_optimized_db

            db = get_optimized_db()
            ch_id = db.resolve_channel_name_to_id(ch_name)
            if ch_id:
                log.info(
                    "Resolved channel '%s' to ID '%s' using database cache",
                    ch_name,
                    ch_id,
                )

        if not ch_id:
            log.warning(
                "Channel '%s' not found in Slack workspace, database cache, "
                "or bot lacks access",
                ch_name,
            )
            continue
        repo.upsert_conversation(
            SlackConversation(id=ch_id, name=ch_name, type="channel", participants=[])
        )
        tasks.append(_fetch_and_store(client, repo, ch_id, ch_name, "channel", cfg))

    # DMs (incremental sync for all DMs with participant information)
    for dm_id, participant_ids in dm_participants.items():
        # Always store user IDs in participants
        participants = participant_ids or []
        # For display, resolve names for the DM name
        participant_names = []
        if participants:
            from data.duckdb_database import get_optimized_db

            db = get_optimized_db()
            for participant_id in participants:
                name = db.resolve_user_id(participant_id)
                participant_names.append(
                    name if name and name != participant_id else participant_id
                )

        # Create human-readable DM name
        if len(participant_names) == 1:
            dm_name = participant_names[0]
        elif len(participant_names) == 2:
            dm_name = f"{participant_names[0]} & {participant_names[1]}"
        elif len(participant_names) > 2:
            others_count = len(participant_names) - 2
            dm_name = (
                f"{participant_names[0]}, {participant_names[1]} + "
                f"{others_count} others"
            )
        else:
            dm_name = dm_id  # Fallback to ID if no participants

        repo.upsert_conversation(
            SlackConversation(
                id=dm_id, name=dm_name, type="dm", participants=participants
            )
        )
        tasks.append(_fetch_and_store(client, repo, dm_id, dm_name, "dm", cfg))

    # Run with semaphore concurrency control
    sem = asyncio.Semaphore(cfg.parallel_requests)

    async def _sem_task(coro: Awaitable[None]) -> None:
        async with sem:
            await coro

    await asyncio.gather(*[_sem_task(t) for t in tasks])

    log.info("Incremental loader finished")


def run_loader(config_path: str, reset_sync_state: bool = False) -> None:
    """Synchronous wrapper to call from CLI."""
    asyncio.run(async_run_loader(config_path, reset_sync_state=reset_sync_state))
