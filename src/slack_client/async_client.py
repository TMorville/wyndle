"""Asynchronous Slack client wrapper used by ingestion layer."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from dotenv import load_dotenv
from slack_sdk.errors import SlackApiError
from slack_sdk.http_retry.builtin_async_handlers import AsyncRateLimitErrorRetryHandler
from slack_sdk.web.async_client import AsyncWebClient

load_dotenv()

logger = logging.getLogger(__name__)
SLACK_USER_KEY = os.environ.get("SLACK_USER_KEY")


class AsyncSlackClient:
    """Thin async wrapper around slack_sdk.AsyncWebClient with helpful utilities."""

    def __init__(self) -> None:
        if not SLACK_USER_KEY:
            raise RuntimeError("SLACK_USER_KEY env var not set")

        self._client = AsyncWebClient(token=SLACK_USER_KEY)
        self._client.retry_handlers.append(
            AsyncRateLimitErrorRetryHandler(max_retry_count=3)
        )

    # ------------------------------------------------------------------
    async def test_auth(self) -> bool:
        try:
            resp = await self._client.auth_test()
            logger.info(
                "Slack auth OK async: user=%s team=%s",
                resp.get("user"),
                resp.get("team"),
            )
            return True
        except SlackApiError as exc:
            logger.error("Slack auth failed async: %s", exc)
            return False

    # ------------------------------------------------------------------
    async def fetch_history(
        self,
        channel_id: str,
        oldest: float | str = "0",
        fetch_threads: bool = True,
    ) -> list[dict[str, Any]]:
        """Fetch channel/DM messages since `oldest`.

        Returns list of raw message dicts.
        """
        messages: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            try:
                result: Any = await self._client.conversations_history(
                    channel=channel_id,
                    limit=200,
                    oldest=str(oldest),
                    cursor=cursor,
                )
                messages.extend(result.get("messages", []))
                if not result.get("has_more"):
                    break
                cursor = result.get("response_metadata", {}).get("next_cursor")
            except SlackApiError as exc:
                if exc.response and exc.response.get("error") == "ratelimited":
                    delay = int(exc.response.headers.get("Retry-After", "1"))
                    logger.warning("Rate limited (history). Sleeping %s s", delay)
                    await asyncio.sleep(delay)
                    continue
                raise

        if fetch_threads:
            await self._augment_with_threads(channel_id, messages)

        return messages

    # ------------------------------------------------------------------
    async def _augment_with_threads(
        self, channel_id: str, messages: list[dict[str, Any]]
    ) -> None:
        parents = [
            m
            for m in messages
            if m.get("thread_ts") and m.get("thread_ts") == m.get("ts")
        ]
        sem = asyncio.Semaphore(4)

        async def _fetch_and_extend(parent_ts: str) -> None:
            async with sem:
                try:
                    result: Any = await self._client.conversations_replies(
                        channel=channel_id, ts=parent_ts, limit=200
                    )
                    replies: list[dict[str, Any]] = result.get("messages", [])[1:]
                    messages.extend(replies)
                except SlackApiError as exc:
                    logger.debug("Thread fetch failed for %s: %s", parent_ts, exc)

        await asyncio.gather(*[_fetch_and_extend(p["ts"]) for p in parents])

    async def get_conversations(self, conv_type: str) -> list[dict[str, Any]]:
        """Fetch all conversations of a given type (e.g.,
        'public_channel,private_channel', 'im').
        """
        conversations = []
        cursor = None
        while True:
            try:
                result: Any = await self._client.conversations_list(
                    types=conv_type, limit=200, cursor=cursor
                )
                conversations.extend(result.get("channels", []))
                cursor = result.get("response_metadata", {}).get("next_cursor")
                if not cursor:
                    break
                # A short sleep to be kind to the API, even with retry handlers
                await asyncio.sleep(1)
            except SlackApiError as e:
                if e.response and e.response.get("error") == "ratelimited":
                    delay = int(e.response.headers.get("Retry-After", "5"))
                    logger.warning(
                        f"Rate limited on conversations.list. Waiting {delay} seconds."
                    )
                    await asyncio.sleep(delay)
                    continue
                logger.error(f"Error fetching conversations list: {e}")
                break
        return conversations

    async def get_all_channels_map(self) -> dict[str, str]:
        """Fetches all public and private channels and returns a name-to-ID map."""
        channels = await self.get_conversations("public_channel,private_channel")
        channel_map = {}
        for channel in channels:
            if isinstance(channel, dict) and channel.get("name") and channel.get("id"):
                channel_map[channel["name"]] = channel["id"]
        return channel_map

    async def get_all_dm_ids(self) -> list[str]:
        """Get a list of all DM conversation IDs (both individual and group DMs)."""
        dms = await self.get_conversations(
            "im,mpim"
        )  # Include both individual and group DMs
        return [dm["id"] for dm in dms if dm.get("id")]

    async def get_all_dms_with_participants(self) -> dict[str, list[str]]:
        """Get all DM conversations with their participant information."""
        dms = await self.get_conversations("im,mpim")
        dm_participants = {}

        for dm in dms:
            dm_id = dm.get("id")
            if not dm_id:
                continue

            participants = []

            # For individual DMs (type 'im'), get the user from the conversation
            if dm.get("is_im"):
                user_id = dm.get("user")
                if user_id:
                    participants = [user_id]

            # For group DMs (type 'mpim'), we need to fetch members
            elif dm.get("is_mpim"):
                try:
                    # Get conversation members
                    result = await self._client.conversations_members(channel=dm_id)
                    participants = result.get("members", [])
                except Exception as e:
                    logger.warning(f"Failed to get members for group DM {dm_id}: {e}")
                    participants = []

            dm_participants[dm_id] = participants

        return dm_participants
