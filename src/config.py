"""Configuration management for Slack PA."""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = get_project_root() / "config" / "config.yaml"

    try:
        with open(config_path) as f:
            config_data = yaml.safe_load(f)
            return config_data if isinstance(config_data, dict) else {}
    except FileNotFoundError:
        logger.warning(f"Config file not found at {config_path}")
        return {}
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config: {e}")
        return {}


def get_data_paths() -> dict[str, Path]:
    """Get standardized data directory paths."""
    root = get_project_root()
    return {
        "data": root / "data",
        "config": root / "config",
    }
