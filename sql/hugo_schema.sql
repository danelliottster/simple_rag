CREATE TABLE IF NOT EXISTS file_metadata (
    source_file TEXT PRIMARY KEY,
    last_modified REAL,
    summary TEXT,
    tags TEXT
);

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