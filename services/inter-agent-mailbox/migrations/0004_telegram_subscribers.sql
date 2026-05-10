-- Telegram observer subscribers. The mailbox optionally runs an embedded
-- Telegram bot that streams forwarded peer frames in real time so an
-- operator can watch the inter-agent traffic. Subscriptions persist across
-- restarts: a chat that called /start once stays subscribed (modulo /stop)
-- so the operator does not have to re-subscribe every deploy.
--
-- chat_id is the Telegram chat where messages are posted (DM or group),
-- user_id is the Telegram user that subscribed (used to enforce the
-- allow-list on subsequent commands), muted=1 silences the dump for that
-- chat without removing it from the subscriber set.
CREATE TABLE IF NOT EXISTS tg_subscribers (
    chat_id     INTEGER PRIMARY KEY,
    user_id     INTEGER NOT NULL,
    muted       INTEGER NOT NULL DEFAULT 0,
    started_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
);
