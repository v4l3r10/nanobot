# nanobot-mailbox

Inter-agent messaging router for nanobot multi-instance deployments. Each
nanobot gateway opens a long-lived WebSocket to this service and stays
connected; frames are routed between connected peers in real time and
persisted in SQLite (WAL) for offline-buffered delivery.

## Architecture

A single multiplexer with two transport surfaces:

1. **Peer plane (real-time)** — `WS /peer` accepts one persistent connection
   per registered agent. Each frame from agent A targeted at agent B is
   forwarded to B's live socket; if B is offline the frame is queued in the
   DB and delivered the next time B connects. Presence (the online roster)
   is broadcast to all live peers on every connect/disconnect.
2. **Blob plane** — `POST /files/` and `GET /files/<id>` provide streaming
   upload/download of message attachments (256 KB – 50 MB). Attachments get
   linked to the carrying message at send-time so the recipient can
   download them with the same bearer used for the WebSocket.

There is no MCP plane: the previous FastMCP tools (`mailbox_send`,
`mailbox_read`, `mailbox_list_agents`, `mailbox_attachment_get`) have been
removed. Agents now talk to each other through their own native chat channel
(`peer:` chat_ids) and discover the roster via presence pushes.

## Endpoints

| Method | Path                       | Auth                                | Purpose                          |
|--------|----------------------------|-------------------------------------|----------------------------------|
| `WS`   | `/peer`                    | `Authorization: Bearer <token>` *or* `?token=` | Persistent peer-to-peer hub      |
| `POST` | `/files/`                  | bearer + multipart                  | Upload attachment blob           |
| `GET`  | `/files/<attachment_id>`   | bearer                              | Stream-download attachment blob  |
| `GET`  | `/healthz`                 | none                                | Liveness + version + online roster |

Bearer tokens are mapped server-side to agent identities via
`MAILBOX_AGENT_TOKENS`; the `X-Agent-Id` header is never trusted.

## Configuration

| Variable                  | Required | Description                                                                 |
|---------------------------|----------|-----------------------------------------------------------------------------|
| `MAILBOX_AGENT_TOKENS`    | yes      | JSON `{ "<agent>": "<bearer-token>", ... }`. One token per agent.           |
| `MAILBOX_LOG_LEVEL`       | no       | Defaults to `INFO`. JSON-line output to stdout.                             |
| `MAILBOX_DATA_DIR`        | no       | Defaults to `/data`. Holds `mailbox.db` and `blobs/`.                       |

## Wire-level frame (v1)

One JSON object per WebSocket frame:

```jsonc
{
  "v": 1,
  "type": "msg" | "presence" | "ping",
  "from": "<agent_id>",                 // server-overridden on outbound
  "to":   "<agent_id>",
  "text": "<plain text body>",
  "thread_id": "thr_<ULID> | null",
  "in_reply_to": "msg_<ULID> | null",
  "attachments": [                      // inbound only (server-populated)
    {"id": <int>, "name": "<str>", "mime": "<str>", "size_bytes": <int>}
  ],
  "attachment_ids": [<int>, ...],       // outbound only (sender uploads first)
  "id": "msg_<ULID>",                   // set by server on forward
  "ts": "<ISO 8601 UTC>"
}
```

A `presence` frame carries only `{"v":1, "type":"presence", "online": [...]}`.

## Runbook

> **First deploy of v2 (peer-only) on a host that ran the legacy MCP mailbox**
> Wipe the existing `mailbox.db` before bringing the new service up. The v2
> schema adds `delivered_at`; legacy rows would be re-delivered on the first
> peer reconnect, flooding agents with stale traffic.
>
> ```bash
> rm -f /home/nanobot/nanobot_workspace/_shared/inter-agent-mailbox/mailbox.db
> # blob/ subdirs can stay if you want; the FK from messages is gone after the
> # wipe so you can also `rm -rf .../inter-agent-mailbox/blobs/` to free disk.
> ```

```bash
# Build and start (from /home/nanobot/nanobot)
docker compose build nanobot-mailbox
docker compose up -d nanobot-mailbox

# Health
docker run --rm --network nanobot_default curlimages/curl \
  http://nanobot-mailbox:8765/healthz

# Inspect DB
docker exec nanobot-mailbox sqlite3 /data/mailbox.db \
  "SELECT id, from_agent, to_agent, delivered_at FROM messages ORDER BY created_at DESC LIMIT 20;"

# Live logs (JSON lines)
docker logs -f nanobot-mailbox
```

## Troubleshooting

- **`MAILBOX_AGENT_TOKENS not set`** at startup: the env var is missing or empty.
- **WS close 4401**: bearer not in registry or token rotated.
- **WS close 4409**: a new connection for the same agent superseded this one
  (single-connection-per-agent invariant).
- **Disk fill in `/data`**: blobs are stored under `/data/blobs/<YYYY>/<MM>/`.
  Run a one-shot cleanup:
  ```sql
  DELETE FROM messages
   WHERE delivered_at IS NOT NULL
     AND created_at < datetime('now', '-30 days');
  ```
