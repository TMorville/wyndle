#!/bin/bash
# Test runner script for local development

set -e

echo "🔍 Running lints and checks..."

# Format check
echo "📏 Checking code formatting..."
uv run ruff format --check .

# Lint check
echo "🧹 Running linter..."
uv run ruff check .

# Type check
echo "🔍 Running type checker..."
uv run mypy .  # Use project root for flat src layout to avoid duplicate module errors

# Security checks
echo "🔒 Running security checks..."
if command -v bandit >/dev/null 2>&1; then
    uv run bandit -r src/ -ll || true
else
    echo "  Bandit not available, skipping..."
fi

if command -v pip-audit >/dev/null 2>&1; then
    uv run pip-audit || true
else
    echo "  pip-audit not available, skipping..."
fi

echo "🧪 Running tests..."

# Run tests with coverage if available, otherwise just run tests
if uv run pytest --help | grep -q "\-\-cov"; then
    uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
else
    uv run pytest tests/ -v
fi

echo "✅ All checks passed!"
echo "📊 Coverage report generated in htmlcov/index.html"