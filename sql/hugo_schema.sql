-- Table for storing metadata about source files
CREATE TABLE IF NOT EXISTS file_metadata (
    source_file TEXT PRIMARY KEY,
    last_modified REAL,
    summary TEXT,
    tags TEXT
);

-- Table for storing text chunks and their embeddings
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file TEXT NOT NULL,
    chunk_index INTEGER,
    chunk_text TEXT,
    embedding BLOB,
    tags TEXT,
    FOREIGN KEY (source_file) REFERENCES file_metadata(source_file) ON DELETE CASCADE,
    UNIQUE (source_file, chunk_index)
);

-- Table for storing conversations
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    conversation TEXT, -- JSON string of conversation history
    conversation_summary TEXT,
    start_datetime TEXT, -- ISO format
    last_modified_datetime TEXT, -- ISO format
    deleted BOOLEAN DEFAULT 0 NOT NULL CHECK (deleted IN (0, 1)) -- Soft delete flag
);