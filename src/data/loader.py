import json
import logging
import os
from datetime import datetime

from config import get_project_root
from data.duckdb_database import get_optimized_db
from slack_client.client import (
    fetch_history,
    get_all_channels,
    get_all_users,
    get_dms,
    load_config,
    test_slack_auth,
)

# Setup logging
logger = logging.getLogger(__name__)

# Use absolute path relative to project root
PROJECT_ROOT = get_project_root()
CONFIG_FILE = PROJECT_ROOT / "meta.json"


def fetch_and_process_conversation(
    conversation_id: str,
    conversation_type: str,
    participants: list[str] | None = None,
    days_to_fetch: int = 7,
) -> int:
    """Fetch conversation history from Slack and directly insert into optimized database."""
    db = get_optimized_db()

    # Get human-readable name for logging
    readable_name = db.name_resolver.create_conversation_name(
        conversation_id, conversation_type, participants
    )

    # For channels, resolve natural name to Slack ID if needed
    actual_conversation_id = conversation_id
    if conversation_type == "channel" and not conversation_id.startswith("C"):
        # Try to resolve natural name to ID
        resolved_id = db.resolve_channel_name_to_id(conversation_id)
        if resolved_id:
            actual_conversation_id = resolved_id
            logger.info(
                f"Resolved channel name '{conversation_id}' to ID '{resolved_id}'"
            )
        else:
            # Try API lookup as fallback
            from slack_client.client import get_all_channels

            channel_map = get_all_channels()
            resolved_id = channel_map.get(conversation_id)
            if resolved_id:
                actual_conversation_id = resolved_id
                logger.info(
                    f"Resolved channel name '{conversation_id}' to ID '{resolved_id}' via API"
                )
            else:
                logger.error(
                    f"Could not resolve channel name '{conversation_id}' to ID"
                )
                return 0

    # Get last message timestamp to avoid duplicates
    existing_messages = db.get_messages(
        conversation_name=f"{conversation_type}:{conversation_id}", limit=1, offset=0
    )

    oldest_timestamp: float
    if existing_messages:
        # Get timestamp of last message and fetch only newer messages
        oldest_timestamp = existing_messages[-1].get("timestamp", 0)
        logger.info(f"Found existing messages, fetching since {oldest_timestamp}")
    else:
        # Calculate oldest timestamp for initial fetch
        oldest_timestamp = datetime.now().timestamp() - (days_to_fetch * 24 * 3600)
        logger.debug(f"New conversation, fetching last {days_to_fetch} days")

    try:
        # Fetch conversation history using the robust fetch_history function
        messages = fetch_history(
            channel_id=actual_conversation_id,
            oldest=oldest_timestamp,
        )
        # Only log when we actually fetched messages
        if len(messages) > 0:
            logger.info(
                f"Fetched {len(messages)} messages from {conversation_type} {readable_name}"
            )
        else:
            logger.debug(
                f"No messages fetched from {conversation_type} {readable_name}"
            )

        # Process and insert messages directly into optimized database
        new_messages = 0
        for raw_message in messages:
            try:
                # Skip if message is too old
                msg_timestamp = float(raw_message.get("ts", 0))
                if msg_timestamp <= oldest_timestamp:
                    continue

                db.add_message(
                    raw_message=raw_message,
                    conversation_id=actual_conversation_id,
                    conversation_type=conversation_type,
                    participants=participants,
                )
                new_messages += 1

            except Exception as e:
                logger.warning(
                    f"Error processing message from {conversation_type} {readable_name}: {e}"
                )
                continue

        # Only log when we actually added new messages
        if new_messages > 0:
            logger.info(
                f"Added {new_messages} new messages from {conversation_type} {readable_name}"
            )
        else:
            logger.debug(
                f"No new messages to add from {conversation_type} {readable_name}"
            )
        return new_messages

    except Exception as e:
        logger.error(f"Error fetching {conversation_type} {readable_name}: {e}")
        return 0


def sync_mappings_if_needed() -> None:
    """Sync user and channel mappings if they don't exist yet."""
    from slack_client.client import get_all_channels, get_all_users

    db = get_optimized_db()

    # Check if we already have mappings
    with db.get_connection() as conn:
        user_result = conn.execute("SELECT COUNT(*) FROM users").fetchone()
        user_count = user_result[0] if user_result else 0
        channel_result = conn.execute("SELECT COUNT(*) FROM channels").fetchone()
        channel_count = channel_result[0] if channel_result else 0

    if user_count > 0 and channel_count > 0:
        logger.info(
            f"Mappings already synced: {user_count} users, {channel_count} channels"
        )
        return

    logger.info("Syncing user and channel mappings for proper name resolution...")

    # Fetch and sync users
    logger.info("Fetching users from Slack API...")
    users_simple = get_all_users()

    # Convert user ID->name map to ID->user_info map
    # The get_all_users() returns ID->name, but sync_user_mappings expects ID->user_info
    users_data = {}
    for user_id, user_name in users_simple.items():
        users_data[user_id] = {
            "id": user_id,
            "name": user_name,
            "real_name": user_name,
            "is_bot": False,  # We'll assume not bot for simple mapping
            "email": "",
            "display_name": user_name,
        }

    # Fetch and sync channels
    logger.info("Fetching channels from Slack API...")
    channels_data = get_all_channels()

    # Convert channel name->ID map to ID->info map
    # The get_all_channels() returns name->ID, but we need ID->channel_info
    channel_info_map = {}
    for name, channel_id in channels_data.items():
        channel_info_map[channel_id] = {
            "id": channel_id,
            "name": name,
            "is_private": False,  # We'll assume public for now
            "purpose": {"value": ""},
            "topic": {"value": ""},
            "num_members": 0,
        }

    logger.info(
        f"Syncing {len(users_data)} users and {len(channel_info_map)} channels to database..."
    )
    db.sync_user_mappings(users_data)
    db.sync_channel_mappings(channel_info_map)

    # Show stats
    stats = db.get_mapping_stats()
    logger.info("Mapping sync complete:")
    logger.info(
        f"  - Total users: {stats['total_users']} (including {stats['bot_users']} bots)"
    )
    logger.info(
        f"  - Total channels: {stats['total_channels']} (including {stats['private_channels']} private)"
    )


def run_dataloader(config_path: str = "config.yaml") -> None:
    """Main function to fetch Slack data and load directly into optimized database."""
    if not test_slack_auth():
        logger.error("Slack authentication failed")
        return

    # Ensure user and channel mappings are synced for proper name resolution
    sync_mappings_if_needed()

    config = load_config(config_path)
    channels_to_fetch = config.get("channels", [])
    days = config.get("days_to_fetch", 7)

    total_new_messages = 0

    # Fetch all channels once and create a name-to-ID map
    logger.info("Fetching all available channels...")
    all_channels_map = get_all_channels()
    logger.info(f"Found {len(all_channels_map)} channels.")

    # Fetch channel conversations
    logger.info(f"Processing {len(channels_to_fetch)} channels from config...")
    for channel_name in channels_to_fetch:
        channel_id = all_channels_map.get(channel_name)

        # If not found in API response, try database cache as fallback
        if not channel_id:
            db = get_optimized_db()
            channel_id = db.resolve_channel_name_to_id(channel_name)
            if channel_id:
                logger.info(
                    f"Resolved channel '{channel_name}' to ID '{channel_id}' using database cache"
                )

        if not channel_id:
            logger.warning(
                f"Channel '{channel_name}' not found in Slack workspace, database cache, or bot lacks access."
            )
            continue

        try:
            new_messages = fetch_and_process_conversation(
                conversation_id=channel_id,
                conversation_type="channel",
                days_to_fetch=days,
            )
            total_new_messages += new_messages

        except Exception as e:
            logger.error(f"Error processing channel {channel_name}: {e}")
            continue

    # Fetch all DM conversations
    logger.info("Fetching all DMs...")
    dm_map = get_dms()
    user_map = get_all_users()

    for dm_id, participant_ids in dm_map.items():
        try:
            # Convert participant IDs to names
            participant_names: list[str] = []
            if participant_ids:
                participant_names = [user_map.get(pid, pid) for pid in participant_ids]

            new_messages = fetch_and_process_conversation(
                conversation_id=dm_id,
                conversation_type="dm",
                participants=participant_names,
                days_to_fetch=days,
            )
            total_new_messages += new_messages

        except Exception as e:
            logger.error(f"Error processing DM {dm_id}: {e}")
            continue

    # Get final database stats
    db = get_optimized_db()
    stats = db.get_stats()

    logger.info("Dataloader completed!")
    logger.info(f"Total new messages added: {total_new_messages}")
    logger.info(f"Final database stats: {stats}")

    # Update meta.json
    meta = {
        "dataloader_last_run": datetime.now().isoformat(),
        "user_id": os.environ.get("SLACK_USER_ID"),
        "user_name": os.environ.get("SLACK_USER_NAME"),
        "total_messages_processed": total_new_messages,
        "database_stats": stats,
    }
    with open(CONFIG_FILE, "w") as f:
        json.dump(meta, f, indent=2)
