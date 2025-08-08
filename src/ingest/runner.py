"""
Continuous Slack Data Loader with Rate Limiting

Runs continuously, fetching messages from Slack API while respecting rate limits.
Uses a token bucket to spread API calls over time.
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from config import load_config  # type: ignore[attr-defined]
from data.duckdb_database import get_duckdb
from data.loader import fetch_and_process_conversation
from slack_client.client import (
    test_slack_auth,
)

logger = logging.getLogger(__name__)

# Rate limiting: Slack allows ~50 conversations.history calls per minute
RATE_LIMIT_CALLS_PER_MINUTE = 45  # Conservative limit
BUCKET_SIZE = 50  # Allow bursts
TOKEN_REFILL_INTERVAL = 60.0 / RATE_LIMIT_CALLS_PER_MINUTE  # seconds per token


class TokenBucket:
    """Token bucket for rate limiting API calls."""

    def __init__(self, size: int, refill_interval: float) -> None:
        self.size = size
        self.refill_interval = refill_interval
        self.tokens: float = float(size)
        self.last_refill = time.monotonic()
        self.lock = asyncio.Lock()

    async def consume(self) -> None:
        """Consume a token, waiting if necessary."""
        async with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_refill

            # Add tokens based on elapsed time
            tokens_to_add = elapsed / self.refill_interval
            self.tokens = min(self.size, self.tokens + tokens_to_add)
            self.last_refill = now

            # Wait if no tokens available
            while self.tokens < 1:
                wait_time = self.refill_interval
                logger.debug(f"Rate limit: waiting {wait_time:.1f}s for token")
                await asyncio.sleep(wait_time)

                now = time.monotonic()
                elapsed = now - self.last_refill
                tokens_to_add = elapsed / self.refill_interval
                self.tokens = min(self.size, self.tokens + tokens_to_add)
                self.last_refill = now

            # Consume token
            self.tokens -= 1
            logger.debug(f"Token consumed, {self.tokens:.1f} remaining")


@asynccontextmanager
async def get_db_connection() -> AsyncGenerator[Any, None]:
    """Get database connection context manager."""
    db = get_duckdb()
    try:
        yield db
    finally:
        # DuckDB connections are automatically managed
        pass


async def fetch_conversation_safe(
    bucket: TokenBucket,
    conversation_id: str,
    conversation_type: str,
    participants: list[str] | None = None,
) -> int:
    """Safely fetch a conversation with rate limiting."""
    try:
        # Wait for rate limit token
        await bucket.consume()

        # Fetch messages synchronously (blocking I/O)
        new_messages: int = await asyncio.to_thread(
            fetch_and_process_conversation,
            conversation_id=conversation_id,
            conversation_type=conversation_type,
            participants=participants,
            days_to_fetch=1,  # Incremental fetch
        )

        # Only log when we actually found new messages
        if new_messages > 0:
            logger.info(
                f"Fetched {new_messages} new messages from {conversation_type} "
                f"{conversation_id}"
            )
        else:
            logger.debug(
                f"No new messages from {conversation_type} {conversation_id}"
            )
        return new_messages

    except Exception as e:
        logger.error(f"Error fetching {conversation_type} {conversation_id}: {e}")
        return 0


async def worker(bucket: TokenBucket, worker_id: int) -> None:
    """Worker that continuously processes scheduled conversations."""
    logger.info(f"Worker {worker_id} starting")

    while True:
        try:
            async with get_db_connection() as db:
                # Get next conversation to process
                scheduled_conv = db.get_next_scheduled_conversation()

                if not scheduled_conv:
                    # No conversations ready, wait a bit
                    await asyncio.sleep(5)
                    continue

                conversation_id = scheduled_conv["conversation_id"]
                conversation_type = scheduled_conv["type"]
                interval_sec = scheduled_conv["interval_sec"]

                # Resolve conversation ID if it's not already resolved
                original_id = conversation_id
                needs_resolution = False
                participants = None  # Will be set for DM conversations

                if (
                    conversation_type == "channel"
                    and not conversation_id.startswith("C")
                    or conversation_type == "dm"
                    and conversation_id.startswith("dm:")
                ):
                    needs_resolution = True

                if needs_resolution:
                    try:
                        # Apply rate limiting before any API resolution
                        await bucket.consume()

                        if conversation_type == "channel":
                            logger.debug(
                                f"Using channel name '{conversation_id}' as-is "
                                "(skipping bulk resolution)"
                            )
                            # Use channel name directly instead of resolving to ID
                            # This avoids the expensive get_all_channels() API call
                            # The fetch_and_process_conversation will handle
                            # name-to-ID resolution if needed

                        elif conversation_type == "dm":
                            # Resolve DM user name to user ID using database cache
                            user_name = conversation_id.replace("dm:", "")

                            # Look up user ID in database (no API call needed!)
                            async with get_db_connection() as db_lookup:
                                with db_lookup.get_connection() as conn:
                                    query = (
                                        "SELECT slack_id FROM users WHERE name = ? "
                                        "OR display_name = ? OR real_name = ?"
                                    )
                                    result = conn.execute(
                                        query, [user_name, user_name, user_name]
                                    ).fetchone()

                            if result:
                                user_id = result[0]

                                # Now we need to get the DM conversation ID
                                # (starts with 'D')
                                # This requires one API call to conversations.open
                                try:
                                    from slack_client.client import client

                                    # Open/get DM conversation with this user
                                    response = client.conversations_open(
                                        users=[user_id]
                                    )
                                    if (
                                        response
                                        and response.get("ok")
                                        and response.get("channel")
                                    ):
                                        dm_conversation_id = response.get(
                                            "channel", {}
                                        ).get("id")
                                        conversation_id = dm_conversation_id
                                        participants = [user_id]

                                        logger.info(
                                            f"Resolved DM '{user_name}' "
                                            f"(user {user_id}) to conversation "
                                            f"{dm_conversation_id}"
                                        )

                                        # Update the schedule to use the resolved
                                        # conversation ID
                                        async with get_db_connection() as db_update:
                                            db_update.update_schedule_id(
                                                original_id, dm_conversation_id
                                            )
                                    else:
                                        error_msg = response.get(
                                            "error", "Unknown error"
                                        )
                                        logger.error(
                                            f"Failed to open DM with {user_name}: "
                                            f"{error_msg}"
                                        )
                                        continue
                                except Exception as e:
                                    logger.error(
                                        f"Error getting DM conversation for "
                                        f"{user_name}: {e}"
                                    )
                                    continue

                                logger.info(
                                    f"Resolved DM '{user_name}' to conversation "
                                    f"{conversation_id}"
                                )
                            else:
                                logger.warning(
                                    f"Could not find user '{user_name}' in database. "
                                    "Try running --sync-mappings"
                                )
                                continue

                    except Exception as e:
                        logger.warning(f"Could not resolve '{original_id}': {e}")
                        # Delay before retrying to avoid hammering the API
                        await asyncio.sleep(30)
                        continue

                logger.debug(
                    f"Worker {worker_id} processing {conversation_type} "
                    f"{conversation_id}"
                )

                # participants is already set above during resolution if needed

                # Fetch the conversation
                await fetch_conversation_safe(
                    bucket, conversation_id, conversation_type, participants
                )

                # Update schedule for next run
                db.update_schedule(conversation_id, interval_sec)

        except Exception as e:
            logger.error(f"Worker {worker_id} error: {e}")
            await asyncio.sleep(10)  # Back off on errors


async def bootstrap_schedule_safe(config: dict[str, Any], bucket: TokenBucket) -> None:
    """
    Bootstrap the schedule table with conversations from config,
    respecting rate limits.
    """
    logger.info("Bootstrapping conversation schedule with rate limiting")

    schedule_config = config.get("schedule", {})
    default_interval = schedule_config.get("default_interval_minutes", 30) * 60

    conversation_schedules = []

    try:
        # Skip bulk channel fetching to avoid rate limits
        channels_to_fetch = config.get("channels", [])

        for channel_name in channels_to_fetch:
            # Use channel name directly to avoid API calls
            interval = default_interval

            conversation_schedules.append(
                {
                    "conversation_id": channel_name,
                    "type": "channel",
                    "interval_sec": interval,
                }
            )

        # Schedule all DMs for continuous monitoring
        # This will fetch all DM conversation IDs and schedule them

        # Insert into database
        db = get_duckdb()
        db.seed_schedule(conversation_schedules)

        schedule_count = len(conversation_schedules)
        logger.info(f"Bootstrapped {schedule_count} conversations to schedule")

    except Exception as e:
        logger.error(f"Error bootstrapping schedule: {e}")


async def bootstrap_schedule_minimal(config: dict[str, Any]) -> None:
    """Minimal bootstrap that only schedules explicitly configured channels and
    DM users (no API calls).
    """
    logger.info("Using explicit config bootstrap (no API calls to avoid rate limits)")

    schedule_config = config.get("schedule", {})
    default_interval = schedule_config.get("default_interval_minutes", 30) * 60

    conversation_schedules = []

    # Schedule explicitly configured channels
    channels_to_fetch = config.get("channels", [])
    for channel_name in channels_to_fetch:
        conversation_schedules.append(
            {
                "conversation_id": channel_name,  # Will be resolved during first fetch
                "type": "channel",
                "interval_sec": default_interval,
            }
        )

    # Schedule all DMs automatically
    # Note: DM IDs will be fetched and scheduled dynamically during runtime
    # This ensures all DMs are monitored without requiring explicit configuration

    # Insert into database
    try:
        db = get_duckdb()
        db.seed_schedule(conversation_schedules)
        channel_count = len(channels_to_fetch)
        logger.info(
            f"Bootstrap: scheduled {channel_count} channels "
            "(DMs will be discovered and scheduled automatically)"
        )
        logger.info("Channel and DM IDs will be resolved during first fetch cycle")

        # Schedule all DMs automatically
        await schedule_all_dms(default_interval)

    except Exception as e:
        logger.error(f"Error in bootstrap: {e}")


async def schedule_all_dms(interval_sec: int) -> None:
    """Automatically discover and schedule all DM conversations."""
    try:
        from slack_client.async_client import AsyncSlackClient

        client = AsyncSlackClient()
        dm_ids = await client.get_all_dm_ids()

        if not dm_ids:
            logger.info("No DMs found to schedule")
            return

        # Create schedule entries for all DMs
        dm_schedules = []
        for dm_id in dm_ids:
            dm_schedules.append(
                {
                    "conversation_id": dm_id,
                    "type": "dm",
                    "interval_sec": interval_sec,
                }
            )

        # Add DM schedules to database
        db = get_duckdb()
        db.seed_schedule(dm_schedules)

        logger.info(f"Automatically scheduled {len(dm_ids)} DMs for monitoring")

    except Exception as e:
        logger.warning(f"Could not automatically schedule DMs: {e}")
        logger.info("DMs will be scheduled when first encountered during fetch")


async def main() -> None:
    """Main continuous loader function."""
    logger.info("Starting Slack PA continuous loader")

    # Test Slack auth
    if not test_slack_auth():
        logger.error("Slack authentication failed")
        return

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        return

    # Bootstrap schedule if needed (using minimal approach to avoid rate limits)
    await bootstrap_schedule_minimal(config)

    # Create token bucket for rate limiting
    bucket = TokenBucket(BUCKET_SIZE, TOKEN_REFILL_INTERVAL)

    # Start worker tasks
    num_workers = 3  # Conservative number of workers
    logger.info(f"Starting {num_workers} workers")

    try:
        # Use gather instead of TaskGroup for compatibility
        tasks = [worker(bucket, i) for i in range(num_workers)]
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")




def get_pid_file() -> Path:
    """Get the PID file path for the daemon."""
    return Path.home() / ".wyndle" / "loader.pid"


def is_daemon_running() -> bool:
    """Check if the daemon is currently running."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        return False

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        # Check if process is still running
        os.kill(pid, 0)  # This will raise OSError if process doesn't exist
        return True
    except (OSError, ValueError):
        # Process not found or invalid PID file - clean up
        with suppress(OSError):
            pid_file.unlink()
        return False


def start_daemon(num_workers: int, verbose: bool = False) -> None:
    """Start the loader as a background daemon."""
    if is_daemon_running():
        print("Daemon is already running. Use 'wyndle-loader status' to check status.")
        sys.exit(1)

    # Create PID directory
    pid_file = get_pid_file()
    pid_file.parent.mkdir(parents=True, exist_ok=True)

    # Fork to background
    if os.fork() != 0:
        # Parent process - exit
        sys.exit(0)

    # Child process - become daemon
    os.setsid()

    # Second fork to prevent zombie processes
    if os.fork() != 0:
        sys.exit(0)

    # Write PID file
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))

    # Redirect standard streams to /dev/null
    with open('/dev/null', 'w') as devnull:
        os.dup2(devnull.fileno(), sys.stdout.fileno())
        os.dup2(devnull.fileno(), sys.stderr.fileno())

    with open('/dev/null') as devnull:
        os.dup2(devnull.fileno(), sys.stdin.fileno())

    # Set up logging to file with appropriate level
    log_file = Path.home() / ".wyndle" / "loader.log"
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
        ]
    )

    # Set our logger to INFO level to show meaningful activity
    logger.setLevel(logging.INFO)

    # Store num_workers globally for main() to use
    global NUM_WORKERS
    NUM_WORKERS = num_workers

    # Handle signals for graceful shutdown
    def signal_handler(signum: int, frame: Any) -> None:
        logger.info(f"Received signal {signum}, shutting down...")
        with suppress(OSError):
            pid_file.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        asyncio.run(main_daemon())
    except Exception as e:
        logger.error(f"Daemon error: {e}")
    finally:
        with suppress(OSError):
            pid_file.unlink()


def stop_daemon() -> None:
    """Stop the running daemon."""
    pid_file = get_pid_file()
    if not pid_file.exists():
        print("Daemon is not running.")
        sys.exit(1)

    try:
        with open(pid_file) as f:
            pid = int(f.read().strip())

        print(f"Stopping daemon (PID {pid})...")
        os.kill(pid, signal.SIGTERM)

        # Wait for process to stop
        import time
        for _ in range(10):  # Wait up to 10 seconds
            try:
                os.kill(pid, 0)
                time.sleep(1)
            except OSError:
                break
        else:
            # Force kill if still running
            with suppress(OSError):
                os.kill(pid, signal.SIGKILL)

        with suppress(OSError):
            pid_file.unlink()

        print("Daemon stopped.")

    except (OSError, ValueError) as e:
        print(f"Error stopping daemon: {e}")
        with suppress(OSError):
            pid_file.unlink()
        sys.exit(1)


def show_status() -> None:
    """Show daemon status."""
    if is_daemon_running():
        pid_file = get_pid_file()
        with open(pid_file) as f:
            pid = int(f.read().strip())
        print(f"Daemon is running (PID {pid})")

        # Show log tail if available
        log_file = Path.home() / ".wyndle" / "loader.log"
        if log_file.exists():
            print("\nRecent log entries:")
            with open(log_file) as f:
                lines = f.readlines()
                for line in lines[-5:]:  # Show last 5 lines
                    print(f"  {line.rstrip()}")
    else:
        print("Daemon is not running.")


def cli_main() -> None:
    """CLI entry point for the continuous loader."""
    parser = argparse.ArgumentParser(description="Wyndle background loader daemon")
    parser.add_argument(
        "action",
        choices=["start", "stop", "restart", "status"],
        help="Daemon action"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help="Number of worker processes (default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging (shows all debug messages)"
    )

    args = parser.parse_args()

    if args.action == "start":
        verbosity = " (verbose)" if args.verbose else ""
        print(f"Starting daemon with {args.workers} workers{verbosity}...")
        start_daemon(args.workers, args.verbose)
        print("Daemon started in background.")

    elif args.action == "stop":
        stop_daemon()

    elif args.action == "restart":
        if is_daemon_running():
            stop_daemon()
        verbosity = " (verbose)" if args.verbose else ""
        print(f"Starting daemon with {args.workers} workers{verbosity}...")
        start_daemon(args.workers, args.verbose)
        print("Daemon restarted.")

    elif args.action == "status":
        show_status()


# Global variable to store worker count from CLI
NUM_WORKERS = 3


async def main_daemon() -> None:
    """Main continuous loader function."""
    logger.info("Starting Slack PA continuous loader")

    # Test Slack auth
    if not test_slack_auth():
        logger.error("Slack authentication failed")
        return

    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        logger.error(f"Could not load config: {e}")
        return

    # Bootstrap schedule if needed (using minimal approach to avoid rate limits)
    await bootstrap_schedule_minimal(config)

    # Create token bucket for rate limiting
    bucket = TokenBucket(BUCKET_SIZE, TOKEN_REFILL_INTERVAL)

    # Start worker tasks
    num_workers = NUM_WORKERS
    logger.info(f"Starting {num_workers} workers")

    try:
        # Use gather instead of TaskGroup for compatibility
        tasks = [worker(bucket, i) for i in range(num_workers)]
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Unexpected error in main loop: {e}")


if __name__ == "__main__":
    cli_main()
