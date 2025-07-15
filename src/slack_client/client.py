import logging
import os
import time
from typing import Any, cast

import yaml
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_handlers import RateLimitErrorRetryHandler
from slack_sdk.web.slack_response import SlackResponse

# Load environment variables
load_dotenv()

SLACK_USER_KEY = os.environ.get("SLACK_USER_KEY")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Debug log to confirm key is loaded
if SLACK_USER_KEY:
    logger.debug(f"Loaded SLACK_USER_KEY: {SLACK_USER_KEY[:6]}...{SLACK_USER_KEY[-4:]}")
else:
    logger.warning("SLACK_USER_KEY not found in environment!")

# Configure the retry handler
rate_limit_handler = RateLimitErrorRetryHandler(max_retry_count=3)

# Initialize the client with the retry handler
client = WebClient(token=SLACK_USER_KEY)
client.retry_handlers.append(rate_limit_handler)


def test_slack_auth() -> bool:
    """Tests authentication with Slack API."""
    try:
        resp = client.auth_test()
        logger.info(
            f"Slack auth OK. User: {resp.get('user')} Team: {resp.get('team')}\n"
        )
        return True
    except Exception as e:
        logger.error(f"Slack auth failed: {e}\n")
        return False


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            return config_data if isinstance(config_data, dict) else {}
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return {}


def get_all_users() -> dict[str, str]:
    """Fetches all users from Slack and returns a mapping of user ID to user name."""
    users: dict[str, str] = {}
    try:
        for page in client.users_list(limit=200):
            for user in page.get("members", []):
                if isinstance(user, dict) and user.get("id"):
                    users[user["id"]] = user.get("real_name") or user.get(
                        "name", "Unknown"
                    )
    except SlackApiError as e:
        logger.error(f"Error fetching users: {e}")
    return users


def fetch_history(
    channel_id: str, oldest: float = 0.0, fetch_threads: bool = True
) -> list[dict[str, Any]]:
    """Fetch all messages from a channel or DM, optionally including thread replies."""
    messages: list[dict[str, Any]] = []
    next_cursor = None

    # First, fetch all main messages
    while True:
        try:
            result: SlackResponse = client.conversations_history(
                channel=channel_id, limit=200, oldest=str(oldest), cursor=next_cursor
            )
            messages.extend(result.get("messages", []))
            if not result.get("has_more"):
                break
            response_metadata: dict[str, Any] = result.get("response_metadata", {})
            if isinstance(response_metadata, dict):
                next_cursor = response_metadata.get("next_cursor")
        except SlackApiError as e:
            logger.error(f"Slack API error fetching history for {channel_id}: {e}")
            break

    # Conditionally fetch thread replies with optimization
    if fetch_threads:
        thread_parents = [
            msg
            for msg in messages
            if (
                msg.get("thread_ts")
                and msg.get("thread_ts") == msg.get("ts")
                and msg.get("reply_count", 0) > 0
            )
        ]

        if thread_parents:
            # Limit thread fetching to avoid rate limits
            default_max = 10  # Conservative default
            max_threads = min(len(thread_parents), default_max)
            if len(thread_parents) > max_threads:
                # Sort by timestamp and take most recent threads
                thread_parents.sort(key=lambda x: float(x.get("ts", 0)), reverse=True)
                thread_parents = thread_parents[:max_threads]
                total_threads = len(thread_parents)
                logger.info(
                    f"Limited thread fetching to {max_threads} most recent threads "
                    f"(out of {total_threads} total)"
                )

            total_thread_replies = 0
            for i, message in enumerate(thread_parents):
                thread_replies = fetch_thread_replies(channel_id, message["ts"])
                if thread_replies:
                    messages.extend(thread_replies)
                    total_thread_replies += len(thread_replies)

                # Add progressive delay to prevent rate limiting
                if i < len(thread_parents) - 1:  # Don't sleep after last thread
                    time.sleep(2)  # 2 second delay between thread requests

            num_threads = len(thread_parents)
            logger.info(
                f"Fetched {total_thread_replies} thread replies from {num_threads} "
                f"threads in {channel_id}"
            )

    return messages


def fetch_thread_replies(channel_id: str, thread_ts: str) -> list[dict[str, Any]]:
    """Fetch all replies for a specific thread using conversations.replies API."""
    replies: list[dict[str, Any]] = []
    try:
        result: SlackResponse = client.conversations_replies(
            channel=channel_id, ts=thread_ts, limit=200
        )
        # Skip the first message (thread parent) and return only replies
        all_messages: list[dict[str, Any]] = result.get("messages", [])
        if len(all_messages) > 1:
            replies = all_messages[1:]  # Skip parent message
            logger.debug(f"Fetched {len(replies)} replies for thread {thread_ts}")
    except SlackApiError as e:
        if e.response and e.response.get("error") == "ratelimited":
            delay = int(e.response.headers.get("Retry-After", "10"))
            logger.warning(f"Rate limited on thread fetch. Waiting {delay} seconds.")
            time.sleep(delay)
            # Skip retry to avoid cascading rate limits
            logger.warning(f"Skipping thread {thread_ts} due to rate limit")
        else:
            logger.error(f"Error fetching thread replies for {thread_ts}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching thread replies for {thread_ts}: {e}")
    return replies


def get_dms(limit: int = 1000) -> dict[str, list[str]]:
    """Fetches all DMs the user is a part of."""
    dm_map: dict[str, list[str]] = {}
    next_cursor = None
    while True:
        try:
            result: SlackResponse = client.conversations_list(
                types="im", limit=limit, cursor=next_cursor
            )
            im_data: Any
            for im_data in result.get("channels", []):
                im = cast(dict[str, Any], im_data)
                # User ID is available directly in the conversations_list response
                if isinstance(im, dict) and im.get("user") and im.get("id"):
                    dm_map[im["id"]] = [im["user"]]

            response_metadata: dict[str, Any] = result.get("response_metadata", {})
            if isinstance(response_metadata, dict):
                next_cursor = response_metadata.get("next_cursor")
            if not next_cursor:
                break
            time.sleep(1)  # Respect rate limits
        except SlackApiError as e:
            if e.response and e.response.get("error") == "ratelimited":
                delay = int(e.response.headers.get("Retry-After", "1"))
                logger.warning(f"Rate limited. Retrying after {delay} seconds.")
                time.sleep(delay)
            else:
                if e.response:
                    error_msg = e.response.get("error")
                    logger.error(f"Slack API error (conversations_list): {error_msg}")
                break
        except Exception as e:
            logger.error(f"Error fetching DM list: {e}")
            break
    return dm_map


def get_all_channels() -> dict[str, str]:
    """Fetches all public and private channels and returns a name-to-ID map."""
    channel_map: dict[str, str] = {}
    next_cursor = None
    while True:
        try:
            result: SlackResponse = client.conversations_list(
                types="public_channel,private_channel", limit=200, cursor=next_cursor
            )
            channel_data: Any
            for channel_data in result.get("channels", []):
                channel = cast(dict[str, Any], channel_data)
                if (
                    isinstance(channel, dict)
                    and channel.get("name")
                    and channel.get("id")
                ):
                    channel_map[channel["name"]] = channel["id"]
            response_metadata: dict[str, Any] = result.get("response_metadata", {})
            if isinstance(response_metadata, dict):
                next_cursor = response_metadata.get("next_cursor")
            if not next_cursor:
                break
            time.sleep(1)  # Respect rate limits
        except SlackApiError as e:
            if e.response and e.response.get("error") == "ratelimited":
                delay = int(e.response.headers.get("Retry-After", "1"))
                logger.warning(f"Rate limited. Retrying after {delay} seconds.")
                time.sleep(delay)
            else:
                if e.response:
                    logger.error(
                        f"Slack API error (channels_list): {e.response.get('error')}"
                    )
                break
        except Exception as e:
            logger.error(f"Error fetching channel list: {e}")
            break
    return channel_map


def get_all_dm_ids() -> list[str]:
    """Get a list of all DM conversation IDs (both individual and group DMs)."""
    dm_ids = []
    next_cursor = None
    while True:
        try:
            result = client.conversations_list(
                types="im,mpim", limit=200, cursor=next_cursor
            )
            for conv in result["channels"]:
                dm_ids.append(conv["id"])
            next_cursor = result.get("response_metadata", {}).get("next_cursor")  # type: ignore
            if not next_cursor:
                break
        except SlackApiError as e:
            logger.error(f"Slack API error fetching DMs: {e}")
            break
    return dm_ids


def get_channel_id_map() -> dict[str, str]:
    """Get a mapping of channel names to channel IDs."""
    channel_map = {}
    next_cursor = None
    while True:
        try:
            result = client.conversations_list(
                types="public_channel,private_channel", limit=200, cursor=next_cursor
            )
            for channel in result["channels"]:
                if "name" in channel:
                    channel_map[channel["name"]] = channel["id"]
            next_cursor = result.get("response_metadata", {}).get("next_cursor")  # type: ignore
            if not next_cursor:
                break
            time.sleep(1)  # Respect rate limits
        except SlackApiError as e:
            if e.response and e.response.get("error") == "ratelimited":
                delay = int(e.response.headers.get("Retry-After", "1"))
                logger.warning(f"Rate limited. Retrying after {delay} seconds.")
                time.sleep(delay)
            else:
                logger.error(f"Slack API error fetching channels: {e}")
                break
    return channel_map
