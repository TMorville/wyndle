"""
Enhanced MCP server using DuckDB database for fast queries.
Provides all the same functionality as the original server but with
O(log n) performance.
"""

import json
import logging
import os
import re
import sys
import warnings
from datetime import datetime
from typing import Any

import yaml

from config import get_project_root  # type: ignore[attr-defined]
from data.duckdb_database import DuckDBConversationDB, get_optimized_db

# Completely suppress warnings that interfere with MCP JSON protocol
warnings.filterwarnings("ignore")

# Set environment variables to disable provider warnings
os.environ["FASTMCP_DISABLE_PROVIDER_WARNINGS"] = "1"
os.environ["GOOGLE_DISABLE_WARNINGS"] = "1"


class SilentWarnings:
    def __init__(self) -> None:
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self) -> "SilentWarnings":
        # Redirect all output to stderr during imports
        sys.stdout = sys.stderr
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> None:
        # Restore stdout
        sys.stdout = self.original_stdout


# Import with warnings redirected to stderr
with SilentWarnings():
    from fastmcp import Context, FastMCP

# Setup logging to stderr to avoid interfering with MCP JSON protocol
logging.basicConfig(
    level=logging.WARNING,  # Reduce log level for MCP
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stderr,  # Send logs to stderr instead of stdout
)
logger = logging.getLogger(__name__)

# Create the FastMCP server with proper paths
PROJECT_ROOT = get_project_root()
STYLEGUIDE_PATH = PROJECT_ROOT / "styleguide.md"
META_PATH = PROJECT_ROOT / "meta.json"
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


# User identity constants for mention/DM-reply detection
def load_user_identity() -> tuple[str | None, str]:
    """Load user identity from meta.json with fallbacks."""
    try:
        meta = load_meta()
        user_id: str | None = meta.get("owner_id") or meta.get("user_id")
        user_name = str(
            meta.get("user_name")
            or meta.get("owner_name")
            or os.getenv("WYNDLE_USER_NAME", "User")
        )
        return user_id, user_name
    except Exception:
        return None, os.getenv("WYNDLE_USER_NAME", "User")


# We'll load this lazily to avoid circular dependency
USER_ID: str | None = None
USER_NAME: str | None = None


# Helpers to load resources
def load_styleguide() -> str:
    try:
        with open(STYLEGUIDE_PATH) as f:
            return f.read()
    except Exception:
        return "Styleguide not found. Please run persona.py to generate it."


def load_meta() -> dict[str, Any]:
    try:
        with open(META_PATH) as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_config() -> dict[str, Any]:
    """Load config.yaml and return the configuration."""
    try:
        with open(CONFIG_PATH) as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_ignored_bots() -> set[str]:
    """Get the list of ignored bots from config, normalized to lowercase."""
    config = load_config()
    slack_config = config.get("slack", {})
    ignored_bots = slack_config.get("ignored_bots", [])

    # Normalize to lowercase for case-insensitive matching
    return {bot.lower() for bot in ignored_bots if isinstance(bot, str)}


# Load resources at startup
STYLEGUIDE_RESOURCE = load_styleguide()
META_RESOURCE = load_meta()

mcp: FastMCP = FastMCP(
    name="Slack Personal Assistant",
    instructions="""
You are a Personal Assistant that helps the user understand and manage their
Slack communications.

🎯 **PRIMARY ROLE**: Act as an intelligent personal assistant, not a raw data provider.
When users ask questions like "summarize my latest interaction with Emil Bunk", provide
meaningful, contextual summaries rather than listing raw messages.

📋 **RESPONSE STYLE**:
- **CONCISE & ACTION-FIRST**: Lead with what needs to be done, skip items with
  no action required
- **Smart Filtering**: Only mention interactions that need user attention -
  ignore concluded conversations
- **Clean Formatting**: Use emojis, bullet points, and clear structure for readability
- **Priority Focus**: Highlight urgent items, skip low-priority background information
- **No Fluff**: Avoid verbose explanations - be direct and helpful

🛠️ **AVAILABLE TOOLS**:

**DISCOVERY Tools** (explore available data):
- discovery_list_channels: Find available channels with activity stats
- discovery_list_dms: Find available DMs with conversation metadata

**CONTENT RETRIEVAL Tools** (get conversation data for analysis):
- content_get_channel_activity: Get channel conversations with full context and
  thread structure
- content_get_user_interactions: Get ALL interactions with a specific person
  (perfect for relationship summaries)

**PRODUCTIVITY Tools** (actionable insights):
- productivity_list_followups: Find pending @-mentions and unanswered DMs that
  need attention
- productivity_get_styleguide: Get user's writing style for consistent communication

🧠 **INTELLIGENT ANALYSIS**:
- **Relationship Insights**: When asked about interactions with someone, analyze
  the conversation tone, outcomes, and current status
- **Action Items**: Identify what needs the user's attention vs. what's just
  informational
- **Context Awareness**: Consider conversation patterns, timing, and
  relationship dynamics
- **Smart Summaries**: Extract key takeaways, decisions made, and next steps
  from conversations

🔧 **TECHNICAL CAPABILITIES**:
- **Smart Name Resolution**: Handles fuzzy matching (e.g., "emil" → "Emil Bunk")
- **Comprehensive Search**: Finds interactions across all channels, DMs, and mentions
- **Thread-Aware**: Understands conversation flow including replies and context
- **Bot Filtering**: Automatically excludes noise from automated systems
- **Human-Readable Output**: All data uses real names instead of IDs

📚 **RESOURCES**:
- resource://styleguide: User's writing style (use for drafting responses)
- resource://meta: Workspace metadata and sync status

💡 **EXAMPLE INTERACTIONS**:

**User**: "What needs my attention?"
**Good Response**:
🔥 **Urgent**: Marta needs receipts attached to Pleo transactions
📅 **This Week**: Diego wants to confirm your availability for bi-weekly data
sharing meetings

**User**: "Summarize my latest interaction with Emil"
**Good Response**:
✅ **All good with Emil** - he said everything's running smoothly. No action needed.

**Bad Response**: Avoid verbose explanations, numbered lists, and mentioning
concluded conversations with no action items.

🎯 **ALWAYS REMEMBER**: You're a personal assistant, not a database query tool.
Focus on helping the user understand their communication landscape and take
appropriate action.
""",
)


# Load ignored bots from config
IGNORED_BOTS = get_ignored_bots()


def is_bot_user(user: str) -> bool:
    """Check if a user should be filtered out as a bot."""
    if not user:
        return False

    user_lc = user.lower()

    # Check against config-based ignored bots list
    for ignored_bot in IGNORED_BOTS:
        if ignored_bot in user_lc:
            return True

    # Legacy hardcoded bot detection for backwards compatibility
    is_slackbot = "slackbot" in user_lc
    is_bot_colon = "bot:" in user_lc and "slackbot" in user_lc
    is_unknown_bot = re.search(r"unknownuser\\(uslackbot\\)", user_lc)

    return is_slackbot or is_bot_colon or bool(is_unknown_bot)


def filter_bots_from_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter out bot messages from results."""
    filtered = []
    for item in results:
        # Check various user name fields for bot detection
        user_fields = [
            "user_name",
            "user",
            "display_user_name",
            "display_name",
            "name",
            "thread_starter",
        ]
        is_bot = False

        for field in user_fields:
            if item.get(field) and is_bot_user(item[field]):
                is_bot = True
                break

        if is_bot:
            continue

        # For followup items, skip DMs with only bot participants
        if (
            item.get("type") == "dm"
            and "location" in item
            and all(
                is_bot_user(p.strip())
                for p in item["location"]
                .replace("DM:", "")
                .replace("(", "")
                .replace(")", "")
                .split(",")
            )
        ):
            continue

        # Filter context messages if present
        if "context_messages" in item and isinstance(item["context_messages"], list):
            item = item.copy()
            item["context_messages"] = [
                cm
                for cm in item["context_messages"]
                if not any(is_bot_user(cm.get(field, "")) for field in user_fields)
            ]

        # Filter thread replies if present
        if (
            "thread" in item
            and "replies" in item["thread"]
            and isinstance(item["thread"]["replies"], list)
        ):
            item = item.copy()
            item["thread"] = item["thread"].copy()
            item["thread"]["replies"] = [
                reply
                for reply in item["thread"]["replies"]
                if not any(is_bot_user(reply.get(field, "")) for field in user_fields)
            ]

        # Filter participants list if present
        if "participants" in item and isinstance(item["participants"], list):
            item = item.copy()
            item["participants"] = [
                participant
                for participant in item["participants"]
                if not any(
                    is_bot_user(
                        participant.get(field, participant)
                        if isinstance(participant, dict)
                        else participant
                    )
                    for field in user_fields + [""]
                )
            ]

        # Filter recent_replies if present (for thread discovery)
        if "recent_replies" in item and isinstance(item["recent_replies"], list):
            item = item.copy()
            item["recent_replies"] = [
                reply
                for reply in item["recent_replies"]
                if not any(is_bot_user(reply.get(field, "")) for field in user_fields)
            ]

        filtered.append(item)

    return filtered


@mcp.resource("resource://styleguide")
def styleguide_resource() -> str:
    """Return the user's writing styleguide for drafting messages."""
    return load_styleguide()


@mcp.resource("resource://meta")
def meta_resource() -> dict[str, Any]:
    """Return MCP metadata (project title, description, etc)."""
    return load_meta()


@mcp.tool
async def productivity_get_styleguide(ctx: Context) -> str:
    """Get the user's writing styleguide for drafting messages."""
    content_list = await ctx.read_resource("resource://styleguide")
    if not content_list:
        return "Styleguide not found."
    content = content_list[0].content
    return content.decode("utf-8") if isinstance(content, bytes) else content


@mcp.tool
def discovery_list_channels() -> list[dict[str, Any]]:
    """Discover all available channels with metadata and activity stats."""
    db = get_optimized_db(
        lazy_resolver=True
    )  # Avoid API calls for read-only operations
    conversations = db.list_conversations(conversation_type="channel")

    results = []

    # Get channel name mappings for resolving IDs to human names
    with db.get_connection() as conn:
        channel_mappings = conn.execute(
            "SELECT slack_id, name FROM channels WHERE slack_id LIKE ?", ["C%"]
        ).fetchall()
        channel_map = {channel[0]: channel[1] for channel in channel_mappings}

    for conv in conversations:
        # Use human-readable name if available, otherwise use the stored name
        display_name = conv["name"]

        # Check if this conversation name is a channel ID that needs mapping
        if conv["name"] in channel_map:
            display_name = channel_map[conv["name"]]
        elif conv["id"] in channel_map:
            display_name = channel_map[conv["id"]]

        results.append(
            {
                "id": conv["id"],
                "name": display_name,
                "message_count": conv.get("actual_message_count", 0),
                "start_date": conv.get("start_date"),
                "end_date": conv.get("end_date"),
                "last_activity": conv.get("last_message_ts"),
                "created_at": conv.get("created_at"),
                "updated_at": conv.get("updated_at"),
            }
        )

    return filter_bots_from_results(results)


@mcp.tool
def discovery_list_dms() -> list[dict[str, Any]]:
    """Discover all available DMs with metadata and activity stats."""
    db = get_optimized_db(
        lazy_resolver=True
    )  # Avoid API calls for read-only operations
    conversations = db.list_conversations(conversation_type="dm")

    results = []
    for conv in conversations:
        results.append(
            {
                "id": conv["id"],
                "name": conv["name"],  # Human-readable name like "DM: John, Alice"
                "participants": conv.get("participants", []),
                "message_count": conv.get("actual_message_count", 0),
                "start_date": conv.get("start_date"),
                "end_date": conv.get("end_date"),
                "last_activity": conv.get("last_message_ts"),
                "created_at": conv.get("created_at"),
                "updated_at": conv.get("updated_at"),
            }
        )

    return filter_bots_from_results(results)


@mcp.tool
def content_get_channel_activity(
    channel_name: str, days_back: int = 14
) -> dict[str, Any]:
    """
    📢 CHANNEL INTELLIGENCE: Get channel activity for meaningful project/team updates.

    🎯 PURPOSE: This tool provides channel conversation data for you to analyze
    and summarize.
    When users ask about channel activity, provide insights like:
    - Key decisions made and their impact
    - Important discussions and outcomes
    - Who's driving what initiatives
    - Action items or deadlines mentioned
    - Overall team/project health and momentum

    📊 DATA PROVIDED: Complete channel conversations with thread structure and
    participant activity.

    🧠 YOUR JOB: Transform this into executive-style briefings that help the
    user stay informed
    about team dynamics and project progress without reading every message.

    Args:
        channel_name: Channel name (e.g., "data-squad", "#general", "C046SG5QTA7")
        days_back: How many days back to analyze (default: 14)
    """
    db = get_optimized_db(lazy_resolver=True)

    # Calculate time cutoff
    cutoff_timestamp = datetime.now().timestamp() - (days_back * 24 * 3600)

    # Resolve channel name to conversation name
    resolved_name = resolve_conversation_name(channel_name, db)
    conversation = db.get_conversation(resolved_name)

    if not conversation:
        # Try to find by searching messages directly
        with db.get_connection() as conn:
            channel_search = conn.execute(
                """
                SELECT DISTINCT conversation_name
                FROM messages
                WHERE conversation_type = 'channel'
                  AND (conversation_name LIKE ? OR conversation_name = ?)
                LIMIT 1
                """,
                [f"%{channel_name}%", channel_name],
            ).fetchone()

            if channel_search:
                resolved_name = channel_search[0]
            else:
                # Provide helpful suggestions
                matching_channels = find_matching_channels(channel_name, db)
                if matching_channels:
                    suggestions = [ch["name"] for ch in matching_channels[:5]]
                    return {
                        "error": f"Channel '{channel_name}' not found",
                        "suggestions": suggestions,
                        "help": (
                            "Did you mean one of these channels? "
                            "Try using the exact channel name."
                        ),
                    }
                else:
                    return {
                        "error": (
                            f"Channel '{channel_name}' not found. "
                            "Use discovery_list_channels to see available channels."
                        )
                    }

    with db.get_connection() as conn:
        # Get channel metadata and stats
        channel_stats = conn.execute(
            """
            SELECT
                COUNT(*) as total_messages,
                COUNT(DISTINCT user_name) as unique_participants,
                MIN(timestamp) as first_message_ts,
                MAX(timestamp) as last_message_ts,
                COUNT(CASE WHEN is_thread_parent THEN 1 END) as thread_count,
                COUNT(
                    CASE WHEN thread_ts IS NOT NULL AND NOT is_thread_parent THEN 1 END
                ) as thread_reply_count
            FROM messages
            WHERE conversation_name = ? AND timestamp > ?
            """,
            [resolved_name, cutoff_timestamp],
        ).fetchone()

        if not channel_stats or channel_stats[0] == 0:
            return {
                "error": (
                    f"No recent activity in channel '{channel_name}' "
                    f"in the last {days_back} days"
                )
            }

        # Get active participants with message counts
        participants = conn.execute(
            """
            SELECT
                m.user_name,
                COALESCE(u.name, u.real_name, m.user_name) as display_name,
                COUNT(*) as message_count,
                MAX(m.timestamp) as last_activity
            FROM messages m
            LEFT JOIN users u ON m.user_name = u.slack_id
            WHERE m.conversation_name = ? AND m.timestamp > ?
            GROUP BY m.user_name, u.name, u.real_name
            ORDER BY message_count DESC
            """,
            [resolved_name, cutoff_timestamp],
        ).fetchall()

        # Get recent main channel messages (non-thread)
        main_messages = conn.execute(
            """
            SELECT m.timestamp, m.datetime, m.user_name, m.clean_text,
                   m.is_thread_parent, m.reply_count, m.thread_ts,
                   COALESCE(u.name, u.real_name, m.user_name) as display_user_name
            FROM messages m
            LEFT JOIN users u ON m.user_name = u.slack_id
            WHERE m.conversation_name = ?
              AND m.timestamp > ?
              AND (m.thread_ts IS NULL OR m.is_thread_parent = true)
            ORDER BY m.timestamp ASC
            """,
            [resolved_name, cutoff_timestamp],
        ).fetchall()

        results: dict[str, Any] = {
            "channel": {
                "name": channel_name,
                "resolved_name": resolved_name,
                "type": "channel",
            },
            "analysis_period": {
                "days_back": days_back,
                "from_date": datetime.fromtimestamp(cutoff_timestamp).isoformat(),
                "to_date": datetime.now().isoformat(),
            },
            "summary": {
                "total_messages": channel_stats[0],
                "unique_participants": channel_stats[1],
                "thread_count": channel_stats[4],
                "thread_replies": channel_stats[5],
                "first_activity": (
                    datetime.fromtimestamp(channel_stats[2]).isoformat()
                    if channel_stats[2]
                    else None
                ),
                "last_activity": (
                    datetime.fromtimestamp(channel_stats[3]).isoformat()
                    if channel_stats[3]
                    else None
                ),
            },
            "participants": [
                {
                    "name": p[1],  # display_name is now at index 1
                    "message_count": p[2],
                    "last_activity": datetime.fromtimestamp(p[3]).isoformat(),
                }
                for p in participants
                if p[1]
            ],
            "conversation_flow": [],
        }

        # Format main messages with nested threads
        msg_columns = [
            "timestamp",
            "datetime",
            "user_name",
            "clean_text",
            "is_thread_parent",
            "reply_count",
            "thread_ts",
            "display_user_name",
        ]
        for msg_tuple in main_messages:
            msg = dict(zip(msg_columns, msg_tuple, strict=False))

            message_entry = {
                "timestamp": msg["timestamp"],
                "datetime": msg["datetime"],
                "user": msg["display_user_name"],
                "text": msg["clean_text"],
                "type": "thread_parent" if msg["is_thread_parent"] else "message",
            }

            # If this is a thread parent, get and include the replies
            if (
                msg["is_thread_parent"]
                and msg["reply_count"]
                and msg["reply_count"] > 0
            ):
                thread_replies = conn.execute(
                    """
                    SELECT m.timestamp, m.datetime, m.user_name, m.clean_text,
                           COALESCE(u.name, u.real_name, m.user_name) as
                           display_user_name
                    FROM messages m
                    LEFT JOIN users u ON m.user_name = u.slack_id
                    WHERE m.conversation_name = ?
                      AND m.thread_ts = ?
                      AND m.is_thread_parent = false
                    ORDER BY m.timestamp ASC
                    """,
                    [resolved_name, msg["thread_ts"]],
                ).fetchall()

                reply_columns = [
                    "timestamp",
                    "datetime",
                    "user_name",
                    "clean_text",
                    "display_user_name",
                ]
                formatted_replies = []
                for reply_tuple in thread_replies:
                    reply = dict(zip(reply_columns, reply_tuple, strict=False))
                    formatted_replies.append(
                        {
                            "timestamp": reply["timestamp"],
                            "datetime": reply["datetime"],
                            "user": reply["display_user_name"],
                            "text": reply["clean_text"],
                            "type": "thread_reply",
                        }
                    )

                message_entry["thread"] = {
                    "reply_count": msg["reply_count"],
                    "replies": formatted_replies,
                }

            results["conversation_flow"].append(message_entry)

    return results


@mcp.tool
def content_get_user_interactions(
    user_name: str, days_back: int = 30, include_mentions_only: bool = False
) -> dict[str, Any]:
    """
    🤝 RELATIONSHIP ANALYSIS: Get all interactions with a specific person for
    intelligent summaries.

    🎯 PURPOSE: This tool provides conversation data for you to analyze and
    summarize meaningfully.
    When users ask "summarize my interaction with [person]", use this data to
    provide insights like:
    - Current relationship status and tone
    - Key outcomes or decisions from recent conversations
    - Whether any action is needed from the user
    - Personal touches (e.g., "they wished you well", "waiting for your response")

    📊 DATA PROVIDED: Complete interaction history across DMs, channels, and mentions.

    🧠 YOUR JOB: Transform this raw data into assistant-style summaries that
    help the user understand
    their relationship dynamics and any needed actions.

    Args:
        user_name: Name of the user to find interactions with (e.g., "Emil Bunk")
        days_back: How many days back to search (default: 30)
        include_mentions_only: If True, only include messages where user_name is
                                mentioned or is the author
    """
    db = get_optimized_db(lazy_resolver=True)

    # Check for ambiguous user names first
    matching_users = find_matching_users(user_name, db)

    # If we have multiple high-scoring matches, suggest clarification
    high_score_matches = [u for u in matching_users if u["score"] >= 70]
    if len(high_score_matches) > 1:
        exact_matches = [u for u in high_score_matches if u["score"] == 100]
        if len(exact_matches) != 1:
            return {
                "error": f"Multiple users found matching '{user_name}'",
                "suggestions": [u["name"] for u in high_score_matches[:5]],
                "help": "Please be more specific. Which user did you mean?",
            }
    elif not matching_users:
        return {
            "error": f"No users found matching '{user_name}'",
            "help": (
                "Try using discovery_list_dms to see available users or check "
                "the spelling."
            ),
        }

    # Calculate cutoff timestamp
    cutoff_timestamp = datetime.now().timestamp() - (days_back * 24 * 3600)

    with db.get_connection() as conn:
        # Build search criteria
        if include_mentions_only:
            # Only messages where user is mentioned or is the author
            search_query = """
                SELECT m.timestamp, m.datetime, m.user_name, m.clean_text,
                       m.conversation_name,
                       m.conversation_type, m.thread_ts, m.is_thread_parent,
                       COALESCE(u.name, u.real_name, m.user_name) as display_user_name,
                       c.participants
                FROM messages m
                LEFT JOIN users u ON m.user_name = u.slack_id
                LEFT JOIN conversations c ON m.conversation_name = c.name
                WHERE m.timestamp > ?
                  AND (LOWER(COALESCE(u.name, u.real_name, m.user_name)) LIKE LOWER(?)
                       OR LOWER(m.clean_text) LIKE LOWER(?))
                ORDER BY m.timestamp DESC
                LIMIT 200
            """
            search_params = [cutoff_timestamp, f"%{user_name}%", f"%{user_name}%"]
        else:
            # All messages in conversations where user participated
            search_query = """
                SELECT m.timestamp, m.datetime, m.user_name, m.clean_text,
                       m.conversation_name,
                       m.conversation_type, m.thread_ts, m.is_thread_parent,
                       COALESCE(u.name, u.real_name, m.user_name) as display_user_name,
                       c.participants
                FROM messages m
                LEFT JOIN users u ON m.user_name = u.slack_id
                LEFT JOIN conversations c ON m.conversation_name = c.name
                WHERE m.timestamp > ?
                  AND m.conversation_name IN (
                      SELECT DISTINCT conversation_name
                      FROM messages m2
                      LEFT JOIN users u2 ON m2.user_name = u2.slack_id
                      WHERE LOWER(COALESCE(u2.name, u2.real_name, m2.user_name))
                            LIKE LOWER(?)
                  )
                ORDER BY m.timestamp DESC
                LIMIT 500
            """
            search_params = [cutoff_timestamp, f"%{user_name}%"]

        messages = conn.execute(search_query, search_params).fetchall()

    if not messages:
        return {
            "error": (
                f"No interactions found with '{user_name}' in the last {days_back} days"
            )
        }

    # Group by conversation
    conversations = {}
    msg_columns = [
        "timestamp",
        "datetime",
        "user_name",
        "clean_text",
        "conversation_name",
        "conversation_type",
        "thread_ts",
        "is_thread_parent",
        "display_user_name",
        "participants",
    ]

    for msg_tuple in messages:
        msg = dict(zip(msg_columns, msg_tuple, strict=False))
        conv_name = msg["conversation_name"]

        if conv_name not in conversations:
            conversations[conv_name] = {
                "name": conv_name,
                "type": msg["conversation_type"],
                "participants": (
                    json.loads(msg["participants"]) if msg["participants"] else None
                ),
                "messages": [],
                "interaction_count": 0,
            }

        # Format message
        formatted_msg = {
            "timestamp": msg["timestamp"],
            "datetime": msg["datetime"],
            "user": msg["display_user_name"],
            "text": msg["clean_text"],
            "thread_info": "",
        }

        if msg["thread_ts"]:
            if msg["is_thread_parent"]:
                formatted_msg["thread_info"] = "thread_parent"
            else:
                formatted_msg["thread_info"] = "thread_reply"

        conversations[conv_name]["messages"].append(formatted_msg)
        conversations[conv_name]["interaction_count"] += 1

    # Sort conversations by most recent activity
    sorted_conversations = sorted(
        conversations.values(),
        key=lambda x: max(msg["timestamp"] for msg in x["messages"]),
        reverse=True,
    )

    # Calculate summary stats
    total_messages = sum(conv["interaction_count"] for conv in conversations.values())
    dm_count = sum(1 for conv in conversations.values() if conv["type"] == "dm")
    channel_count = sum(
        1 for conv in conversations.values() if conv["type"] == "channel"
    )

    return {
        "_assistant_guidance": {
            "purpose": (
                "Provide a concise relationship summary focused on actionable items"
            ),
            "format": (
                "Use emojis and bullet points. Lead with action items if any exist."
            ),
            "examples": {
                "action_needed": (
                    "🔔 **Diego** wants to confirm your availability for "
                    "bi-weekly meetings"
                ),
                "concluded": (
                    "✅ **All good with Emil** - project complete, wished you good "
                    "holidays"
                ),
                "no_action": "Don't mention if there's nothing actionable",
            },
            "avoid": [
                "Verbose explanations and context sections",
                "Numbered lists with detailed breakdowns",
                "Mentioning conversations with no action items",
                "Background information unless essential",
            ],
        },
        "user": user_name,
        "search_period": {
            "days_back": days_back,
            "from_date": datetime.fromtimestamp(cutoff_timestamp).isoformat(),
            "to_date": datetime.now().isoformat(),
            "include_mentions_only": include_mentions_only,
        },
        "summary": {
            "total_interactions": total_messages,
            "conversations_count": len(conversations),
            "dm_conversations": dm_count,
            "channel_conversations": channel_count,
        },
        "conversations": filter_bots_from_results(sorted_conversations),
    }


def get_user_identity() -> tuple[str | None, str]:
    """Get the current user identity, loading lazily if needed."""
    global USER_ID, USER_NAME
    if USER_ID is None and USER_NAME is None:
        USER_ID, USER_NAME = load_user_identity()
    return USER_ID, USER_NAME or "Tobias Morville"


def find_matching_channels(
    search_term: str, db: DuckDBConversationDB
) -> list[dict[str, Any]]:
    """Find channels that match the search term with fuzzy matching."""
    with db.get_connection() as conn:
        matches = conn.execute(
            """
            SELECT c.name, c.slack_id,
                   CASE
                       WHEN c.name = ? THEN 100
                       WHEN c.name LIKE ? THEN 90
                       WHEN c.name LIKE ? THEN 80
                       WHEN c.name LIKE ? THEN 70
                       ELSE 0
                   END as match_score
            FROM channels c
            WHERE c.name LIKE ? OR c.name LIKE ? OR c.name LIKE ?
            ORDER BY match_score DESC, c.name
            LIMIT 10
            """,
            [
                search_term,
                f"{search_term}%",
                f"%{search_term}%",
                f"%{search_term}",
                f"{search_term}%",
                f"%{search_term}%",
                f"%{search_term}",
            ],
        ).fetchall()

        return [
            {"name": match[0], "slack_id": match[1], "score": match[2]}
            for match in matches
        ]


def find_matching_users(
    search_term: str, db: DuckDBConversationDB
) -> list[dict[str, Any]]:
    """Find users that match the search term with fuzzy matching."""
    with db.get_connection() as conn:
        matches = conn.execute(
            """
            SELECT COALESCE(u.name, u.real_name, u.slack_id) as display_name,
                   u.slack_id,
                   CASE
                       WHEN LOWER(COALESCE(u.name, u.real_name)) = LOWER(?) THEN 100
                       WHEN LOWER(COALESCE(u.name, u.real_name)) LIKE LOWER(?) THEN 90
                       WHEN LOWER(COALESCE(u.name, u.real_name)) LIKE LOWER(?) THEN 80
                       WHEN LOWER(COALESCE(u.name, u.real_name)) LIKE LOWER(?) THEN 70
                       ELSE 0
                   END as match_score
            FROM users u
            WHERE LOWER(COALESCE(u.name, u.real_name)) LIKE LOWER(?)
               OR LOWER(COALESCE(u.name, u.real_name)) LIKE LOWER(?)
               OR LOWER(COALESCE(u.name, u.real_name)) LIKE LOWER(?)
            ORDER BY match_score DESC, display_name
            LIMIT 10
            """,
            [
                search_term,
                f"{search_term}%",
                f"%{search_term}%",
                f"%{search_term}",
                f"{search_term}%",
                f"%{search_term}%",
                f"%{search_term}",
            ],
        ).fetchall()

        return [
            {"name": match[0], "slack_id": match[1], "score": match[2]}
            for match in matches
        ]


def resolve_conversation_name(user_friendly_name: str, db: DuckDBConversationDB) -> str:
    """
    Resolve user-friendly conversation name to database conversation name with
    fuzzy matching.

    Args:
        user_friendly_name: Human name like "data", "Jesper Christiansen"
        db: Database instance

    Returns:
        Database conversation name (might be same as input or resolved ID)
    """
    # First try direct lookup
    conversation = db.get_conversation(user_friendly_name)
    if conversation:
        return user_friendly_name

    # For channels, try reverse mapping: readable name -> channel ID
    with db.get_connection() as conn:
        # Look up channel by readable name to get its ID
        channel_result = conn.execute(
            "SELECT slack_id FROM channels WHERE name = ?", [user_friendly_name]
        ).fetchone()

        if channel_result:
            channel_id = channel_result[0]
            # Check if conversation exists with this channel ID as the name
            conv_result = conn.execute(
                "SELECT name FROM conversations WHERE name = ? OR id = ?",
                [channel_id, channel_id],
            ).fetchone()

            if conv_result:
                return str(
                    conv_result[0]
                )  # Return the actual database conversation name

        # Try fuzzy matching for channel names
        fuzzy_matches = conn.execute(
            """
            SELECT c.name, c.slack_id,
                   CASE
                       WHEN c.name = ? THEN 100
                       WHEN c.name LIKE ? THEN 90
                       WHEN c.name LIKE ? THEN 80
                       WHEN c.name LIKE ? THEN 70
                       ELSE 0
                   END as match_score
            FROM channels c
            WHERE c.name LIKE ? OR c.name LIKE ? OR c.name LIKE ?
            ORDER BY match_score DESC, c.name
            LIMIT 5
            """,
            [
                user_friendly_name,
                f"{user_friendly_name}%",
                f"%{user_friendly_name}%",
                f"%{user_friendly_name}",
                f"{user_friendly_name}%",
                f"%{user_friendly_name}%",
                f"%{user_friendly_name}",
            ],
        ).fetchall()

        if fuzzy_matches:
            # Check if top match exists as conversation
            best_match = fuzzy_matches[0]
            conv_result = conn.execute(
                "SELECT name FROM conversations WHERE name = ? OR id = ?",
                [best_match[0], best_match[1]],
            ).fetchone()

            if conv_result:
                return str(conv_result[0])

    # If no mapping found, return original name
    return user_friendly_name


@mcp.tool
def productivity_list_followups(max_age_hours: int = 72) -> dict[str, Any]:
    """
    ⚡ ATTENTION MANAGER: Find items that need the user's response or
    attention.

    🎯 PURPOSE: This tool finds pending items for you to summarize as
    actionable insights.
    When users ask "what needs my attention?" or "what's pending?", transform
    this data into:
    - Clear priority levels (urgent vs. can wait)
    - Who is waiting and for how long
    - Context about why it matters
    - Suggested next actions

    📊 DATA PROVIDED: Unanswered @mentions and DMs with context and timing.

    🧠 YOUR JOB: Present this as an executive summary focusing on what the
    user should do next,
    not a raw list of notifications.

    Args:
        max_age_hours: Maximum age in hours to include in results
    """
    db = get_optimized_db(
        lazy_resolver=True
    )  # Avoid API calls for read-only operations
    user_id, user_name = get_user_identity()

    if not user_name:
        return {
            "_assistant_guidance": {
                "purpose": "No user identity found - unable to find followups",
                "example_response": "❌ Unable to identify user for followup tracking.",
            },
            "status": "error",
            "followups": [],
        }

    cutoff_timestamp = datetime.now().timestamp() - (max_age_hours * 3600)
    context_cutoff = cutoff_timestamp - (14 * 24 * 3600)  # 14 days for context

    followups = []

    # Find unanswered mentions in channels
    with db.get_connection() as conn:
        mention_query = """
            SELECT m1.timestamp, m1.datetime, m1.user_name, m1.clean_text,
                   m1.conversation_name,
                   COALESCE(u.name, u.real_name, m1.user_name) as display_user_name
            FROM messages m1
            LEFT JOIN users u ON m1.user_name = u.slack_id
            WHERE m1.conversation_type = 'channel'
              AND m1.timestamp > ?
              AND COALESCE(u.name, u.real_name, m1.user_name) != ?
              AND (m1.clean_text LIKE ? OR m1.clean_text LIKE ?)
              AND NOT EXISTS (
                  SELECT 1 FROM messages m2
                  LEFT JOIN users u2 ON m2.user_name = u2.slack_id
                  WHERE m2.conversation_name = m1.conversation_name
                    AND m2.timestamp > m1.timestamp
                    AND COALESCE(u2.name, u2.real_name, m2.user_name) = ?
              )
            ORDER BY m1.timestamp DESC
        """

        mention_params = [
            cutoff_timestamp,
            user_name,
            f"%@{user_name}%",
            f"%{user_name}%",
            user_name,
        ]

        mentions = conn.execute(mention_query, mention_params).fetchall()
        mention_columns = [
            "timestamp",
            "datetime",
            "user_name",
            "clean_text",
            "conversation_name",
            "display_user_name",
        ]

        for mention_tuple in mentions:
            mention = dict(zip(mention_columns, mention_tuple, strict=False))
            age_hours = (datetime.now().timestamp() - mention["timestamp"]) / 3600
            snippet = mention["clean_text"][:100] + (
                "..." if len(mention["clean_text"]) > 100 else ""
            )

            # Get context messages
            context_query = """
                SELECT m.timestamp, m.datetime, m.user_name, m.clean_text as text,
                       COALESCE(u.name, u.real_name, m.user_name) as display_user_name
                FROM messages m
                LEFT JOIN users u ON m.user_name = u.slack_id
                WHERE m.conversation_name = ?
                  AND m.timestamp >= ?
                  AND m.timestamp < ?
                ORDER BY m.timestamp ASC
            """
            context_results = conn.execute(
                context_query,
                (mention["conversation_name"], context_cutoff, mention["timestamp"]),
            ).fetchall()

            context_columns = [
                "timestamp",
                "datetime",
                "user_name",
                "text",
                "display_user_name",
            ]
            context_messages = [
                dict(zip(context_columns, ctx_tuple, strict=False))
                for ctx_tuple in context_results
            ]

            followups.append(
                {
                    "type": "mention",
                    "location": f"#{mention['conversation_name']}",
                    "ts": str(mention["timestamp"]),
                    "snippet": snippet,
                    "age_hours": round(age_hours, 1),
                    "user": mention["display_user_name"],
                    "context_messages": context_messages,
                }
            )

    # Find unanswered DMs
    with db.get_connection() as conn:
        dm_query = """
            SELECT c.name, m.timestamp, m.datetime, m.user_name, m.clean_text as text,
                   COALESCE(u.name, u.real_name, m.user_name) as display_user_name
            FROM conversations c
            JOIN messages m ON c.name = m.conversation_name
            LEFT JOIN users u ON m.user_name = u.slack_id
            WHERE c.type = 'dm'
              AND m.timestamp = (
                  SELECT MAX(timestamp) FROM messages m2
                  WHERE m2.conversation_name = c.name
              )
              AND COALESCE(u.name, u.real_name, m.user_name) != ?
              AND m.timestamp > ?
              AND (? - m.timestamp) > 86400  -- More than 24 hours old
        """

        dm_params = [user_name, cutoff_timestamp, datetime.now().timestamp()]
        dms = conn.execute(dm_query, dm_params).fetchall()
        dm_columns = [
            "name",
            "timestamp",
            "datetime",
            "user_name",
            "text",
            "display_user_name",
        ]

        for dm_tuple in dms:
            dm = dict(zip(dm_columns, dm_tuple, strict=False))
            age_hours = (datetime.now().timestamp() - dm["timestamp"]) / 3600
            snippet = dm["text"][:100] + ("..." if len(dm["text"]) > 100 else "")

            # Use human-readable name directly
            location = dm["name"]  # Already in format "DM: John, Alice"

            # Get context messages
            context_query = """
                SELECT m.timestamp, m.datetime, m.user_name, m.clean_text as text,
                       COALESCE(u.name, u.real_name, m.user_name) as display_user_name
                FROM messages m
                LEFT JOIN users u ON m.user_name = u.slack_id
                WHERE m.conversation_name = ?
                  AND m.timestamp >= ?
                  AND m.timestamp < ?
                ORDER BY m.timestamp ASC
            """
            context_results = conn.execute(
                context_query,
                (dm["name"], context_cutoff, dm["timestamp"]),
            ).fetchall()

            context_columns = [
                "timestamp",
                "datetime",
                "user_name",
                "text",
                "display_user_name",
            ]
            context_messages = [
                dict(zip(context_columns, ctx_tuple, strict=False))
                for ctx_tuple in context_results
            ]

            followups.append(
                {
                    "type": "dm",
                    "location": location,
                    "ts": str(dm["timestamp"]),
                    "snippet": snippet,
                    "age_hours": round(age_hours, 1),
                    "user": dm["display_user_name"],
                    "context_messages": context_messages,
                }
            )

    # Sort by age (oldest first) and filter
    followups = [f for f in followups if f["age_hours"] <= max_age_hours]
    followups.sort(key=lambda x: x["age_hours"], reverse=True)

    # Add assistant guidance for better responses
    if not followups:
        return {
            "_assistant_guidance": {
                "purpose": "No pending items found - keep it brief and positive",
                "example_response": (
                    "✅ **All clear!** No pending items need your attention."
                ),
            },
            "status": "all_clear",
            "followups": [],
        }

    return {
        "_assistant_guidance": {
            "purpose": (
                "Create a concise priority list using emojis and action-focused "
                "language"
            ),
            "format": (
                "🔥 **Urgent**: [action needed]\n📅 **This Week**: [action needed]"
            ),
            "rules": [
                "ONLY mention items that require user action",
                "Skip items with no clear action (e.g., 'everything running smoothly')",
                "Use priority emojis: 🔥 urgent, 📅 this week, 📋 when convenient",
                (
                    "Be direct: 'Sarah needs feedback on mockups' not 'Sarah is "
                    "waiting for feedback'"
                ),
                "No numbered lists or verbose context sections",
            ],
            "examples": {
                "good": "🔥 **Marta** needs receipts attached to Pleo transactions",
                "bad": (
                    "1. **Marta Hernández Gonzalo**: Context: Request to attach "
                    "receipts..."
                ),
            },
        },
        "summary": {
            "total_pending": len(followups),
            "mentions": len([f for f in followups if f["type"] == "mention"]),
            "dms": len([f for f in followups if f["type"] == "dm"]),
        },
        "followups": filter_bots_from_results(followups),
    }


if __name__ == "__main__":
    mcp.run()
