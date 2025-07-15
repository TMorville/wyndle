# Contributing to Wyndle

Thank you for your interest in contributing! This guide will help you get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/wyndle.git
   cd wyndle
   ```

2. **Install uv** (if not already installed)
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Install dependencies**
   ```bash
   uv sync --all-extras --dev
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Slack token
   ```

5. **Set up configuration**
   ```bash
   cp config/config.example.yaml config/config.yaml
   # Edit config.yaml with your channels
   ```

## Development Workflow

### Running Tests

```bash
# Run all tests
./scripts/test.sh

# Run specific test file
uv run pytest tests/test_mcp/test_server.py -v

# Run with coverage
uv run pytest tests/ --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
uv run ruff format .

# Lint code
uv run ruff check .

# Type check
uv run mypy .

# Security check
uv run bandit -r src/
uv run pip-audit
```

### Running the Application

```bash
# One-time data load
uv run wyndle-pipeline --dataloader

# Start MCP server
uv run wyndle-server

# Continuous loader
uv run wyndle-loader
```

## Code Style

- **Line length**: 88 characters (Black/Ruff default)
- **Type hints**: Required for all functions and methods
- **Docstrings**: Use Google-style docstrings for all public functions
- **Import order**: Follow isort conventions (automatic with Ruff)

## Testing Guidelines

- Write tests for all new functionality
- Use descriptive test names that explain what is being tested
- Mock external dependencies (Slack API, database, etc.)
- Use pytest fixtures for common test data
- Aim for high test coverage (>90%)

### Test Structure

```python
class TestFeatureName:
    """Test cases for FeatureName functionality."""
    
    def test_method_name_condition(self, fixture_name):
        """Test that method_name does X when Y condition."""
        # Arrange
        input_data = "test_input"
        expected = "expected_output"
        
        # Act
        result = function_under_test(input_data)
        
        # Assert
        assert result == expected
```

## Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write code following the style guidelines
   - Add tests for new functionality
   - Update documentation if needed

3. **Test your changes**
   ```bash
   ./scripts/test.sh
   ```

4. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

### Commit Message Format

We follow conventional commits:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `test:` for adding tests
- `refactor:` for code refactoring
- `perf:` for performance improvements
- `chore:` for maintenance tasks

## Architecture Overview

### Core Components

- **`src/cli.py`**: Command-line interface
- **`src/config.py`**: Core configuration management
- **`src/data/`**: Data processing and storage
- **`src/ingest/`**: Data ingestion pipeline
- **`src/server.py`**: MCP server implementation
- **`src/slack_client/`**: Slack API integration
- **`src/analysis/`**: Writing style and organizational analysis
- **`src/repository.py`**: Database operations
- **`src/models.py`**: Data models and types

### Key Patterns

- **Dependency Injection**: Use factories and dependency injection for testability
- **Async/Await**: Use async patterns for I/O operations
- **Type Safety**: Extensive use of type hints and mypy
- **Configuration**: YAML-based configuration with validation
- **Error Handling**: Graceful error handling with logging

## Database Schema

The project uses DuckDB with the following main tables:

- `messages`: Core message storage with full-text search
- `conversations`: Conversation metadata
- `user_mappings`: User ID to name mappings
- `channel_mappings`: Channel ID to name mappings

## Performance Considerations

- Use DuckDB's columnar storage for fast queries
- Implement rate limiting for Slack API calls
- Cache user/channel name resolutions
- Use batch operations for database writes

## Security Guidelines

- Never commit API tokens or sensitive data
- Use environment variables for secrets
- Validate all user inputs
- Follow principle of least privilege
- Regular security audits with bandit and pip-audit

## Getting Help

- **Documentation**: Check the README and code comments
- **Issues**: Search existing issues or create a new one
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Don't hesitate to ask for feedback in PRs

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.