"""Domain models used across Slack PA.

These models provide strict typing and validation for data exchanged between layers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, validator


class SlackUser(BaseModel):
    """Represents a Slack user profile."""

    id: str = Field(..., description="Slack user ID, e.g., U123456")
    name: str = Field(..., description="Display or real name of the user.")


class SlackMessage(BaseModel):
    """Represents a single Slack message."""

    message_id: str = Field(..., description="Unique Slack message timestamp (ts)")
    conversation_id: str = Field(..., description="Channel or DM ID")
    user_id: str | None = Field(None, description="User ID of the sender")
    text: str = Field(..., description="Raw text content")
    timestamp: float = Field(..., description="Unix epoch timestamp")
    thread_ts: str | None = Field(None, description="Thread parent ts if applicable")
    raw: dict[str, Any] | None = Field(None, description="Original message payload")

    @validator("timestamp", pre=True, always=True)
    def _ensure_float(cls, v: Any) -> float:  # noqa: D401
        """Ensure timestamp is stored as float."""
        return float(v)

    @property
    def datetime(self) -> datetime:
        """Return human-readable datetime for convenience."""
        return datetime.fromtimestamp(self.timestamp)


class SlackConversation(BaseModel):
    """Represents a Slack channel or DM."""

    id: str = Field(..., description="Conversation ID, e.g., C123456 or D123456")
    name: str = Field(..., description="Human readable name")
    type: str = Field(..., description="'channel' or 'dm'")
    participants: list[str] = Field(
        default_factory=list, description="User IDs involved in the conversation"
    )
