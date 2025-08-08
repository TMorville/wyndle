# 🌿 Wyndle - Your Slack Conversation Assistant

> *"I must be helpful! It is very important that I be helpful!"*

Transform your Slack data into an intelligent personal assistant using AI and high-performance DuckDB storage.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## ✨ Features

### 🧠 **AI-Powered Intelligence**
- **Smart Summaries**: "Nothing urgent from Emil! He wishes you good holiday" instead of raw message dumps
- **Priority Management**: Automatically identifies what needs your attention with urgency levels
- **Relationship Analysis**: Understand conversation dynamics and outcomes across all contexts

### ⚡ **High-Performance Data Pipeline** 
- **DuckDB Storage**: Columnar database for sub-millisecond query performance
- **Continuous Sync**: Background loader with intelligent rate limiting (respects Slack's 45 calls/minute)
- **Human-Readable**: All data uses real names instead of cryptic IDs

### 🔗 **MCP Integration**
- **Model Context Protocol**: Works seamlessly with Raycast, OpenAI, and other AI tools
- **Natural Queries**: Ask "What needs my attention?" and get actionable insights
- **Configurable**: Smart bot filtering and customizable response styles

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- UV package manager (recommended) or pip
- Slack workspace with bot token

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/wyndle.git
cd wyndle

# Install dependencies
uv sync

# Configure your setup
cp config/config.example.yaml config/config.yaml
cp .env.example .env

# Edit your configuration files
# - Add your Slack API tokens to .env
# - Configure channels to monitor in config/config.yaml
```

> **⚠️ Important**: Make sure to set your `SLACK_USER_KEY` and `OPENAI_API_KEY` in the `.env` file before running the data loader. The database schema will be created automatically on first run.

### Configuration

1. **Slack Bot Setup**: Create a Slack app with these scopes:
   ```
   channels:history, groups:history, im:history, mpim:history
   users:read, channels:read
   ```

2. **Environment Variables**: Add to `.env`:
   ```bash
   SLACK_USER_KEY=xoxp-your-user-token-here
   ```

3. **Channel Selection**: Edit `config/config.yaml`:
   ```yaml
   slack:
     channels:
       - general
       - engineering
       - data-team
     ignored_bots:
       - slackbot
       - github
       - jira
   ```

### Usage

```bash
# One-time data load
uv run wyndle-pipeline --dataloader

# Start continuous background sync
uv run wyndle-loader

# Launch MCP server for AI integration
uv run wyndle-server

# View database statistics
uv run wyndle-pipeline --database-stats
```

## 🎯 Raycast Integration

### Demo

![Wyndle Raycast Integration Demo](wyndle_demo.gif)

*Watch how easy it is to add Wyndle to Raycast and start using it as your AI-powered Slack assistant.*

### Setup Instructions

1. **Install MCP Extension in Raycast**
   - Open Raycast
   - Go to Extensions → Store
   - Search for "MCP" and install the MCP extension

2. **Install MCP Server** 
   - Complete the installation steps above first
   - Make sure you have your `.env` file configured with API keys

3. **Add Wyndle to Raycast**
   - Open Raycast MCP settings
   - Click "Add MCP Server"
   - **Command**: `<your-path-to-repo>/scripts/run-wyndle.sh`
   - **Name**: `Wyndle`
   - Save the configuration

### Usage

Once configured, you can interact with Wyndle directly in Raycast:

```
@Wyndle please look at my squad channel, make a list of action points for me and add them as todo's with @todoist

@Wyndle anything to follow up on today?

@Wyndle summarize my latest interaction with Emil B

@Wyndle show me what's happening in the data team channel this week
```

Wyndle will provide intelligent, context-aware responses and can integrate with other Raycast extensions like Todoist for task management.

## 💬 AI Integration Examples

### With Raycast
```
"What needs my attention?"
→ 🔥 **Urgent**: Sarah needs feedback on design mockups
→ 📅 **This Week**: Confirm availability for bi-weekly data meetings

"Summarize my interaction with John"
→ ✅ **Project complete** - John thanked you for the pipeline work and signed off
```

### With OpenAI/Claude
The MCP server provides these intelligent tools:
- `content_get_user_interactions()` - Relationship summaries
- `productivity_list_followups()` - Action items and priorities  
- `content_get_channel_activity()` - Team/project intelligence

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Slack API     │───▶│   DuckDB        │───▶│   MCP Server    │
│                 │    │   (Columnar)    │    │   (FastMCP)     │
│ • Rate Limited  │    │ • Human Names   │    │ • AI Assistant  │
│ • Continuous    │    │ • Sub-ms Query  │    │ • Smart Filter  │
│ • Smart Sync    │    │ • 70% Smaller   │    │ • Raycast Ready │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Key Components
- **`src/ingest/`** - Continuous data pipeline with rate limiting
- **`src/data/`** - DuckDB storage layer with name resolution
- **`src/server.py`** - Personal assistant MCP server
- **`src/cli.py`** - Command-line interface
- **`src/slack_client/`** - Slack API integration
- **`src/analysis/`** - Writing style and organizational analysis

## 🧪 Development

### Running Tests
```bash
# Run minimal test suite
uv run pytest tests/

# Type checking
uv run mypy src/

# Code quality
uv run ruff check .
uv run ruff format .
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`uv run pytest`)
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [FastMCP](https://github.com/jlowin/fastmcp) for Model Context Protocol integration
- Powered by [DuckDB](https://duckdb.org/) for high-performance analytics
- Inspired by the need for intelligent Slack data management

## 📊 Performance

- **Query Speed**: Sub-millisecond response times with DuckDB columnar storage
- **Memory Usage**: ~300MB constant footprint with efficient compression  
- **Storage**: 70% smaller than equivalent SQLite databases
- **Rate Limiting**: Respects Slack's API limits with intelligent backoff

---

<div align="center">
  <strong>🌿 "I must be helpful!" - Transform your Slack chaos into AI-powered insights</strong><br>
  Made with ❤️ for productive teams everywhere
</div>