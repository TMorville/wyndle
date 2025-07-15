"""
Name Resolution System for Slack PA
Converts all Slack IDs to human-readable names for optimal searchability.
"""

import json
import logging
import re
from typing import Any

# Note: Avoiding bulk API calls to prevent rate limits
# from slack_client.client import get_all_channels, get_all_users

logger = logging.getLogger(__name__)


class NameResolver:
    """Resolves Slack IDs to human-readable names across all content."""

    def __init__(self, db: Any | None = None) -> None:
        self.db = db  # DuckDB database instance for cached mappings
        self.user_map: dict[str, str] = {}
        self.channel_map: dict[str, str] = {}
        self.reverse_user_map: dict[str, str] = {}
        self.reverse_channel_map: dict[str, str] = {}
        self._load_mappings()

    def _load_mappings(self) -> None:
        """Load mappings from database cache if available, otherwise initialize empty."""
        if self.db:
            try:
                self.user_map = self.db.get_user_mappings()
                self.channel_map = self.db.get_channel_mappings()
                self.reverse_user_map = {
                    name: uid for uid, name in self.user_map.items()
                }
                self.reverse_channel_map = {
                    name: cid for cid, name in self.channel_map.items()
                }
                logger.debug(
                    f"Loaded {len(self.user_map)} users and {len(self.channel_map)} channels from database cache"
                )
            except Exception as e:
                logger.warning(f"Failed to load mappings from database: {e}")
                self._initialize_empty_mappings()
        else:
            self._initialize_empty_mappings()

    def _initialize_empty_mappings(self) -> None:
        """Initialize empty mappings."""
        self.user_map = {}
        self.channel_map = {}
        self.reverse_user_map = {}
        self.reverse_channel_map = {}
        logger.debug("Name resolver initialized with empty mappings")

    def resolve_user_id(self, user_id: str | None) -> str | None:
        """Convert user ID to human name using database cache or in-memory fallback."""
        if not user_id:
            return None

        # Try database first if available
        if self.db:
            try:
                result: str | None = self.db.resolve_user_id(user_id)
                if result and result != user_id:  # Found a real name, not just the ID
                    return result
            except Exception as e:
                logger.debug(f"Database user lookup failed: {e}")

        # Fallback to in-memory cache, then return ID as-is
        return self.user_map.get(user_id, user_id)

    def resolve_channel_id(self, channel_id: str | None) -> str | None:
        """Convert channel ID to human name using database cache or in-memory fallback."""
        if not channel_id:
            return None

        # Try database first if available
        if self.db:
            try:
                result: str | None = self.db.resolve_channel_id(channel_id)
                if (
                    result and result != channel_id
                ):  # Found a real name, not just the ID
                    return result
            except Exception as e:
                logger.debug(f"Database channel lookup failed: {e}")

        # Fallback to in-memory cache, then return ID as-is
        return self.channel_map.get(channel_id, channel_id)

    def resolve_text_mentions(self, text: str) -> str:
        """Replace all @mentions and #channels with human names in text."""
        if not text:
            return text

        # Replace user mentions: <@U01234> -> @John Doe
        def replace_user_mention(match: re.Match[str]) -> str:
            user_id = match.group(1)
            user_name = self.user_map.get(user_id, user_id)
            return f"@{user_name}"

        text = re.sub(r"<@([UW][A-Z0-9]+)>", replace_user_mention, text)

        # Replace channel mentions: <#C01234|general> -> #general
        def replace_channel_mention(match: re.Match[str]) -> str:
            channel_id = match.group(1)
            channel_name = match.group(2) or self.channel_map.get(
                channel_id, channel_id
            )
            return f"#{channel_name}"

        text = re.sub(r"<#([C][A-Z0-9]+)(?:\|([^>]+))?>", replace_channel_mention, text)

        # Replace special mentions
        text = re.sub(r"<!everyone>", "@everyone", text)
        text = re.sub(r"<!channel>", "@channel", text)
        text = re.sub(r"<!here>", "@here", text)

        return text

    def extract_mentioned_users(self, text: str) -> list[str]:
        """Extract list of mentioned user names from text."""
        if not text:
            return []

        mentioned = []

        # Find @mentions in text
        for match in re.finditer(r"<@([UW][A-Z0-9]+)>", text):
            user_id = match.group(1)
            user_name = self.user_map.get(user_id)
            if user_name:
                mentioned.append(user_name)

        return list(set(mentioned))  # Remove duplicates

    def create_conversation_name(
        self,
        conversation_id: str,
        conversation_type: str,
        participants: list[str] | None = None,
        participant_names: list[str] | None = None,
    ) -> str:
        """Create human-readable conversation name."""
        if conversation_type == "channel":
            # For channels, use the channel name
            return self.resolve_channel_id(conversation_id) or conversation_id
        elif conversation_type == "dm":
            # For DMs, create "User1, User2" or "User1 + N others" format
            if participant_names:
                user_names = sorted(participant_names)
            elif participants:
                resolved_names = [self.resolve_user_id(uid) for uid in participants]
                user_names = sorted([name for name in resolved_names if name])
            else:
                user_names = []

            if not user_names:
                return f"DM: {conversation_id}"

            # Return single name, "A & B", or "A, B + N"
            if len(user_names) == 1:
                return user_names[0]
            elif len(user_names) == 2:
                return f"{user_names[0]} & {user_names[1]}"
            else:
                return (
                    f"{user_names[0]}, {user_names[1]} + {len(user_names) - 2} others"
                )
        else:
            return conversation_id

    def process_message(
        self, raw_message: dict, conversation_name: str, conversation_type: str
    ) -> dict[str, Any]:
        """Convert raw Slack message to optimized, human-readable format."""

        # Basic message info
        timestamp = float(raw_message.get("ts", 0))
        user_id = raw_message.get("user")
        user_name = self.resolve_user_id(user_id)

        # Extract and clean text
        text = self._extract_text_content(raw_message)
        clean_text = self.resolve_text_mentions(text)

        # Thread information
        thread_ts = raw_message.get("thread_ts")
        is_thread_parent = thread_ts and thread_ts == raw_message.get("ts")

        # Message type and content analysis
        message_type = self._determine_message_type(raw_message)
        subtype = raw_message.get("subtype")

        # Content flags
        has_attachments = bool(
            raw_message.get("files") or raw_message.get("attachments")
        )
        has_reactions = bool(raw_message.get("reactions"))
        is_edited = bool(raw_message.get("edited"))

        # Mentioned users
        mentioned_users = self.extract_mentioned_users(text)

        # Word count and links
        word_count = len(clean_text.split()) if clean_text else 0
        contains_links = bool(re.search(r"https?://", clean_text))

        # Create optimized message record
        return {
            "id": f"{conversation_name}:{timestamp}",
            "conversation_name": conversation_name,
            "conversation_type": conversation_type,
            "timestamp": timestamp,
            "datetime": self._format_datetime(timestamp),
            "user_name": user_name,
            "author_name": user_name,  # Consistency
            "text": text,  # Original text
            "clean_text": clean_text,  # Processed text
            "message_type": message_type,
            "subtype": subtype,
            "thread_ts": float(thread_ts) if thread_ts else None,
            "is_thread_parent": is_thread_parent,
            "reply_count": raw_message.get("reply_count", 0) if is_thread_parent else 0,
            "has_attachments": has_attachments,
            "has_reactions": has_reactions,
            "is_edited": is_edited,
            "edited_ts": self._get_edit_timestamp(raw_message),
            "mentions_users": json.dumps(mentioned_users) if mentioned_users else None,
            "contains_links": contains_links,
            "word_count": word_count,
            "search_text": self._create_search_text(
                clean_text, user_name, conversation_name
            ),
        }

    def _extract_text_content(self, message: dict) -> str:
        """Extract text from various Slack message formats."""
        # Try simple text field first
        text = message.get("text", "")

        # If no text, try to extract from blocks
        if not text and message.get("blocks"):
            text = self._extract_text_from_blocks(message["blocks"])

        # Handle file shares
        if not text and message.get("files"):
            files = message["files"]
            if isinstance(files, list) and files:
                file_info = files[0]
                text = f"[File: {file_info.get('name', 'Unknown')}]"
                if file_info.get("title"):
                    text += f" {file_info['title']}"

        return text or ""

    def _extract_text_from_blocks(self, blocks: list) -> str:
        """Extract text content from Slack blocks format."""
        text_parts = []

        for block in blocks:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "rich_text":
                elements = block.get("elements", [])
                for element in elements:
                    if (
                        isinstance(element, dict)
                        and element.get("type") == "rich_text_section"
                    ):
                        sub_elements = element.get("elements", [])
                        for sub_elem in sub_elements:
                            if isinstance(sub_elem, dict):
                                if sub_elem.get("type") == "text":
                                    text_parts.append(sub_elem.get("text", ""))
                                elif sub_elem.get("type") == "emoji":
                                    emoji_name = sub_elem.get("name", "")
                                    text_parts.append(f":{emoji_name}:")
                                elif sub_elem.get("type") == "user":
                                    user_id = sub_elem.get("user_id")
                                    user_name = self.resolve_user_id(user_id)
                                    text_parts.append(
                                        f"@{user_name}"
                                        if user_name
                                        else f"<@{user_id}>"
                                    )

        return "".join(text_parts)

    def _determine_message_type(self, message: dict) -> str:
        """Determine the type of message for categorization."""
        subtype = message.get("subtype")

        if subtype:
            if subtype in ["file_share", "file_comment"]:
                return "file_share"
            elif subtype in ["channel_join", "channel_leave"]:
                return "system"
            elif subtype == "bot_message":
                return "bot"
            else:
                return "system"

        # Check if it's a bot message
        if message.get("bot_id") or message.get("app_id"):
            return "bot"

        # Check for files
        if message.get("files") or message.get("attachments"):
            return "file_share"

        return "message"

    def _format_datetime(self, timestamp: float) -> str:
        """Convert timestamp to readable datetime string."""
        try:
            from datetime import datetime

            return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, OSError):
            return "Invalid Timestamp"

    def _get_edit_timestamp(self, message: dict) -> float | None:
        """Extract edit timestamp if message was edited."""
        edited = message.get("edited")
        if isinstance(edited, dict) and edited.get("ts"):
            return float(edited["ts"])
        return None

    def _create_search_text(
        self, clean_text: str, user_name: str | None, conversation_name: str
    ) -> str:
        """Create optimized search text combining content and metadata."""
        parts = []

        if clean_text:
            parts.append(clean_text.lower())
        if user_name:
            parts.append(user_name.lower())
        if conversation_name:
            parts.append(conversation_name.lower())

        return " ".join(parts)


# Global resolver instance
_resolver_instance: NameResolver | None = None


def get_name_resolver(db: Any | None = None) -> NameResolver:
    """Get global name resolver instance with optional database cache."""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = NameResolver(db=db)
    return _resolver_instance


def refresh_name_mappings() -> None:
    """Refresh the global name resolver mappings."""
    global _resolver_instance
    _resolver_instance = None  # Force reload on next access
