-- Track real-time WS delivery distinct from MCP mailbox_read consumption.
-- read_at  = recipient consumed via legacy MCP mailbox_read (kept for back-compat)
-- delivered_at = frame forwarded to recipient over a live peer WS connection
ALTER TABLE messages ADD COLUMN delivered_at TEXT NULL;
CREATE INDEX idx_messages_pending_delivery
  ON messages(to_agent, delivered_at, created_at)
  WHERE delivered_at IS NULL;
