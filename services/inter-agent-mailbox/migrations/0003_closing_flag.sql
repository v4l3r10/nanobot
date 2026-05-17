-- closing=1 marks a "thread close" message: the sender declares this is the
-- last message in the exchange. Receiver-side, the channel uses this flag to
-- skip waking the agent loop, breaking pleasantry/echo loops by design rather
-- than via heuristic suppression. The message body is still persisted and
-- visible via /messages, so peer_thread_show shows it normally.
ALTER TABLE messages ADD COLUMN closing INTEGER NOT NULL DEFAULT 0;
