import json
import logging
from contextlib import suppress
from datetime import datetime
from typing import Any

from anthropic import Anthropic
from anthropic.types import TextBlock

from data.duckdb_database import get_optimized_db

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Constants
STYLEGUIDE_PATH = "styleguide.md"
META_PATH = "meta.json"
MODEL_NAME = "claude-3-opus-20240229"
MAX_TOKENS = 4000
import os
USER_NAME = os.getenv("WYNDLE_USER_NAME", "User")  # Configurable user name

client = Anthropic()


def collect_user_messages(user_name: str) -> list[str]:
    """Load all DM messages from optimized database for the specified user."""
    messages: list[str] = []

    # Get database instance
    db = get_optimized_db(lazy_resolver=True)

    # Get all DM conversations
    dm_conversations = db.list_conversations(conversation_type="dm")

    for conv in dm_conversations:
        # Get messages from this DM conversation
        conv_messages = db.get_messages(
            conversation_name=conv["name"],
            limit=1000,  # Get recent messages for analysis
        )

        for msg in conv_messages:
            if msg.get("user_name") == user_name:
                text = msg.get("clean_text") or msg.get("text", "")
                if text:
                    messages.append(text)

    logger.info(
        f"Collected {len(messages)} messages from "
        f"{len(dm_conversations)} DM conversations"
    )
    return messages


def chunk_messages(messages: list[str], max_length: int = 15000) -> list[str]:
    """Combine messages into larger chunks without exceeding max_length."""
    chunks: list[str] = []
    current_chunk = ""
    for message in messages:
        if len(current_chunk) + len(message) + 1 > max_length:
            chunks.append(current_chunk)
            current_chunk = ""
        current_chunk += message + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return chunks


def styleguide_prompt(user_name: str) -> str:
    """Generate the system prompt for the styleguide creation task."""
    return f"""
You are a writing analyst. Based on the following messages from a user named
{user_name}, extract a styleguide that describes their writing style, tone, typical
phrases, and any notable patterns. The styleguide should be actionable, so an AI can
use it to write new messages that sound like {user_name}.
"""


def main() -> None:
    """Main function to generate the styleguide."""
    logger.info("Collecting messages for %s from optimized database...", USER_NAME)
    messages = collect_user_messages(USER_NAME)
    if not messages:
        logger.warning("No messages found for user.")
        return
    logger.info("Collected %d messages. Chunking for context window...", len(messages))
    chunks = chunk_messages(messages)
    styleguides: list[str] = []
    prompt = styleguide_prompt(USER_NAME)
    for i, chunk in enumerate(chunks):
        logger.info(
            "Requesting styleguide for chunk %d/%d using %s (Responses API)...",
            i + 1,
            len(chunks),
            MODEL_NAME,
        )
        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=MAX_TOKENS,
            system=prompt,
            messages=[{"role": "user", "content": chunk}],
        )
        if response.content and isinstance(response.content[0], TextBlock):
            styleguides.append(response.content[0].text.strip())

    logger.info("\n=== STYLEGUIDE FOR TOBIAS MORVILLE ===\n")
    with open(STYLEGUIDE_PATH, "w") as f:
        for i, sg in enumerate(styleguides):
            chunk_header = f"--- Chunk {i + 1} Analysis ---\n"
            f.write(f"{chunk_header}{sg}\n\n")
            logger.info("%s%s\n", chunk_header, sg)
    # Update meta.json with styleguide_last_run
    now_iso = datetime.now().isoformat()
    meta: dict[str, Any] = {}
    with open(META_PATH) as f, suppress(json.JSONDecodeError, FileNotFoundError):
        meta = json.load(f)

    meta["styleguide_last_run"] = now_iso
    with open(META_PATH, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
