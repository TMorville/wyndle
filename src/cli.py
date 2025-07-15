import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from data.duckdb_database import get_duckdb
from ingest.loader import run_loader

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()

# Constants - use absolute paths relative to project root
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def main() -> None:
    """Main function to run the data processing pipeline."""
    parser = argparse.ArgumentParser(description="Slack PA data processing pipeline")
    parser.add_argument(
        "--dataloader",
        action="store_true",
        help=(
            "Fetch messages from Slack and load into database "
            "(automatically syncs user/channel mappings first)"
        ),
    )
    parser.add_argument(
        "--reset-sync-state",
        action="store_true",
        help="Delete sync_state table to force full refresh",
    )
    parser.add_argument(
        "--wipe-database",
        action="store_true",
        help="Completely wipe the database and start fresh (WARNING: Deletes all data)",
    )
    parser.add_argument(
        "--database-stats",
        action="store_true",
        help="Show database statistics and overview",
    )

    args = parser.parse_args()

    if not any([args.dataloader, args.wipe_database, args.database_stats]):
        parser.print_help()
        return

    try:
        if args.dataloader:
            logger.info("Starting dataloader (incremental)...")
            run_loader(
                str(CONFIG_PATH),
                reset_sync_state=args.reset_sync_state,
            )
            logger.info("Dataloader completed")

        if args.wipe_database:
            logger.info("Wiping database...")
            wipe_database()
            logger.info("Database wiped successfully!")

        if args.database_stats:
            logger.info("Getting database statistics...")
            show_database_stats()

        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


def wipe_database() -> None:
    """Completely wipe the database and all related data files."""
    import shutil

    # Get database path
    db = get_duckdb()
    db_path = db.db_path

    # Close any existing connections
    del db

    # Remove database file if it exists
    if db_path.exists():
        db_path.unlink()
        logger.info(f"Removed database file: {db_path}")

    # Remove raw data directory if it exists
    raw_dir = PROJECT_ROOT / "data" / "raw"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
        logger.info(f"Removed raw data directory: {raw_dir}")

    # Remove any other data files in data directory except for the database we just deleted
    data_dir = PROJECT_ROOT / "data"
    if data_dir.exists():
        for item in data_dir.iterdir():
            if (
                item.is_file()
                and item.name != "conversations.duckdb"
                and not item.name.startswith(".")
            ):
                item.unlink()
                logger.info(f"Removed data file: {item}")

    # Clear meta.json sync state
    meta_path = PROJECT_ROOT / "meta.json"
    if meta_path.exists():
        meta_path.unlink()
        logger.info(f"Removed meta file: {meta_path}")

    logger.info("Database and all related data files have been wiped")


def show_database_stats() -> None:
    """Show database statistics and overview."""
    db = get_duckdb()
    stats = db.get_stats()

    print("\n" + "=" * 50)  # noqa: T201
    print("DATABASE OVERVIEW")  # noqa: T201
    print("=" * 50)  # noqa: T201

    print(f"Total Messages: {stats.get('total_messages', 0):,}")  # noqa: T201
    print(f"Total Conversations: {stats.get('total_conversations', 0):,}")  # noqa: T201
    print(f"  - Channels: {stats.get('channels', 0):,}")  # noqa: T201
    print(f"  - DMs: {stats.get('dms', 0):,}")  # noqa: T201
    print(f"Total Users: {stats.get('total_users', 0):,}")  # noqa: T201

    if stats.get("date_range"):
        date_range = stats["date_range"]
        start_date = date_range.get("start_date", "Unknown")
        end_date = date_range.get("end_date", "Unknown")
        print(f"Date Range: {start_date} to {end_date}")  # noqa: T201

    if stats.get("database_size_mb"):
        print(f"Database Size: {stats['database_size_mb']:.1f} MB")  # noqa: T201

    # Show top channels by activity if available
    if stats.get("top_channels"):
        print("\nTop 5 Most Active Channels:")  # noqa: T201
        for i, channel in enumerate(stats["top_channels"][:5], 1):
            name = channel.get("name", "Unknown")
            count = channel.get("message_count", 0)
            print(f"  {i}. {name} ({count:,} messages)")  # noqa: T201

    print("\n" + "=" * 50)  # noqa: T201


if __name__ == "__main__":
    main()
