-- Optimized DuckDB schema for Slack PA
-- Uses flat, human-readable structure for maximum performance and searchability

-- Conversations table - stores channels and DMs with human-readable names
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL, -- 'channel' or 'dm'
    participants TEXT, -- JSON array of participant names
    created_at TEXT,
    updated_at TEXT,
    message_count INTEGER DEFAULT 0,
    latest_message_ts REAL
);

-- Messages table - flat structure with resolved names for optimal search
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_name TEXT NOT NULL,
    conversation_type TEXT NOT NULL,
    timestamp REAL NOT NULL,
    datetime TEXT NOT NULL,
    user_name TEXT,
    author_name TEXT, -- alias for user_name for consistency
    text TEXT,
    clean_text TEXT, -- processed text with resolved mentions
    message_type TEXT,
    subtype TEXT,
    thread_ts REAL,
    is_thread_parent BOOL DEFAULT FALSE,
    reply_count INTEGER DEFAULT 0,
    has_attachments BOOL DEFAULT FALSE,
    has_reactions BOOL DEFAULT FALSE,
    is_edited BOOL DEFAULT FALSE,
    edited_ts REAL,
    mentions_users TEXT, -- JSON array of mentioned user names
    contains_links BOOL DEFAULT FALSE,
    word_count INTEGER DEFAULT 0,
    search_text TEXT -- optimized search field combining content and metadata
);

-- Users table - for name resolution and user mapping
CREATE TABLE IF NOT EXISTS users (
    slack_id TEXT PRIMARY KEY,
    name TEXT,
    display_name TEXT,
    real_name TEXT,
    email TEXT,
    is_bot BOOL DEFAULT FALSE,
    created_at TEXT,
    updated_at TEXT
);

-- Channels table - for name resolution and channel mapping  
CREATE TABLE IF NOT EXISTS channels (
    slack_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    display_name TEXT,
    purpose TEXT,
    topic TEXT,
    is_private BOOL DEFAULT FALSE,
    member_count INTEGER DEFAULT 0,
    created_at TEXT,
    updated_at TEXT
);

-- Attachments table - file shares and media
CREATE TABLE IF NOT EXISTS attachments (
    id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    file_name TEXT,
    file_type TEXT,
    file_size INTEGER,
    url_private TEXT,
    title TEXT,
    is_image BOOL DEFAULT FALSE,
    is_video BOOL DEFAULT FALSE,
    is_document BOOL DEFAULT FALSE
);

-- Reactions table - emoji reactions on messages
CREATE TABLE IF NOT EXISTS reactions (
    message_id TEXT NOT NULL,
    emoji TEXT NOT NULL,
    emoji_name TEXT,
    user_name TEXT NOT NULL,
    timestamp REAL,
    PRIMARY KEY (message_id, emoji, user_name)
);

-- Thread relationships table - for optimized thread queries
CREATE TABLE IF NOT EXISTS thread_relationships (
    parent_message_id TEXT NOT NULL,
    child_message_id TEXT NOT NULL,
    conversation_name TEXT NOT NULL,
    thread_depth INTEGER DEFAULT 1,
    PRIMARY KEY (parent_message_id, child_message_id)
);

-- Search chunks table - for text search optimization  
CREATE TABLE IF NOT EXISTS search_chunks (
    id TEXT PRIMARY KEY,
    conversation_name TEXT NOT NULL,
    chunk_text TEXT NOT NULL,
    message_ids TEXT, -- JSON array of message IDs in this chunk
    created_at TEXT
);

-- Create indexes for optimal performance
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_name);
CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_name);
CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_ts);
CREATE INDEX IF NOT EXISTS idx_messages_search ON messages(search_text);
CREATE INDEX IF NOT EXISTS idx_conversations_name ON conversations(name);
CREATE INDEX IF NOT EXISTS idx_conversations_type ON conversations(type);
CREATE INDEX IF NOT EXISTS idx_users_name ON users(name, real_name);
CREATE INDEX IF NOT EXISTS idx_channels_name ON channels(name);