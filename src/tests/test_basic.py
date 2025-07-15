"""Minimal tests for CI/CD - only basic imports and functionality."""


def test_core_imports() -> None:
    """Test that core modules can be imported without errors."""
    import config

    assert config is not None


def test_model_creation() -> None:
    """Test basic model creation without external dependencies."""
    from models import SlackMessage

    message = SlackMessage(
        message_id="1234567890.000100",
        conversation_id="C123456",
        timestamp=1234567890.0,
        text="test message",
    )
    assert message.timestamp == 1234567890.0
    assert message.conversation_id == "C123456"
    assert message.text == "test message"


def test_project_root() -> None:
    """Test project root detection."""
    from config import get_project_root  # type: ignore[attr-defined]

    root = get_project_root()
    assert root.exists()
    assert root.is_dir()


def test_basic_config() -> None:
    """Test basic config model creation."""
    from ingest.loader import LoaderConfig

    config = LoaderConfig(channels=["test"])
    assert config.channels == ["test"]
    assert isinstance(config.sync_days, int)
