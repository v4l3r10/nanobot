PRAGMA foreign_keys = ON;

CREATE TABLE messages (
  id            TEXT PRIMARY KEY,
  thread_id     TEXT NOT NULL,
  in_reply_to   TEXT NULL REFERENCES messages(id),
  from_agent    TEXT NOT NULL,
  to_agent      TEXT NOT NULL,
  type          TEXT NOT NULL CHECK (type IN ('notification','request','response','broadcast')),
  subject       TEXT NOT NULL,
  body          TEXT NOT NULL,
  priority      TEXT NOT NULL DEFAULT 'normal' CHECK (priority IN ('low','normal','high')),
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
  read_at       TEXT NULL
);
CREATE INDEX idx_messages_inbox  ON messages(to_agent, read_at, created_at DESC);
CREATE INDEX idx_messages_thread ON messages(thread_id, created_at);

CREATE TABLE attachments (
  id            INTEGER PRIMARY KEY AUTOINCREMENT,
  message_id    TEXT NULL REFERENCES messages(id) ON DELETE CASCADE,
  owner_agent   TEXT NOT NULL,
  filename      TEXT NOT NULL,
  mime          TEXT NOT NULL,
  size_bytes    INTEGER NOT NULL,
  storage       TEXT NOT NULL CHECK (storage IN ('inline','blob')),
  inline_b64    TEXT NULL,
  blob_path     TEXT NULL,
  created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
CREATE INDEX idx_attachments_message ON attachments(message_id);
CREATE INDEX idx_attachments_orphan  ON attachments(message_id, created_at) WHERE message_id IS NULL;

CREATE TABLE rate_limits (
  key           TEXT PRIMARY KEY,
  count         INTEGER NOT NULL DEFAULT 0,
  window_start  TEXT NOT NULL
);
