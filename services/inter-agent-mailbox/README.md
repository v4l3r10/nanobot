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
| `MAILBOX_TELEGRAM_BOT_TOKEN`     | no | Enables the embedded Telegram observer (see below). Unset → observer off.   |
| `MAILBOX_TELEGRAM_ALLOWED_USERS` | no | JSON list of Telegram user_ids permitted to issue commands (e.g. `[136150230]`). |
| `MAILBOX_TELEGRAM_BASE_URL`      | no | Optional Bot API base URL (e.g. `http://telegram-bot-api:8081/bot` for the local sidecar). |
| `MAILBOX_TELEGRAM_BASE_FILE_URL` | no | Optional Bot API file URL paired with `BASE_URL` for self-hosted Bot API.   |

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

## Telegram observer (optional)

When `MAILBOX_TELEGRAM_BOT_TOKEN` is set, the service starts an embedded
Telegram bot that streams every forwarded peer frame and presence event to
subscribed chats so an operator can watch live inter-agent traffic from a
phone. The bot runs in-process inside `nanobot-mailbox` (same container, no
sidecar) and is fully optional — without the env var the service behaves
identically to before.

Allow-list: only Telegram user_ids listed in `MAILBOX_TELEGRAM_ALLOWED_USERS`
can run any command. The list is the security boundary; chat_ids are never
trusted on their own. With an empty allow-list the bot answers `/start`
with "non sei autorizzato".

Subscriptions are persisted in SQLite (`tg_subscribers` table) and survive
restarts: a chat that called `/start` once stays subscribed (until `/stop`).

### Commands

| Command   | Effect                                                            |
|-----------|-------------------------------------------------------------------|
| `/start`  | Subscribe this chat to the live dump (returns the inline menu).   |
| `/menu`   | Show the inline menu of common actions as tappable buttons.       |
| `/stop`   | Unsubscribe this chat.                                            |
| `/mute`   | Pause dump for this chat (subscription kept).                     |
| `/unmute` | Resume after `/mute`.                                             |
| `/status` | Service uptime, peers online, subscriber counts.                  |
| `/peers`  | Online roster right now.                                          |
| `/last N` | Show the last N persisted messages globally (max 50).             |
| `/break A B` | Inject a bilateral `closing=true` frame between peers A and B to break a chat loop. The receiving channel honours the flag by skipping `publish_inbound`, so neither agent wakes on the synthetic frame. |
| `/help`   | List the commands above.                                          |

The inline menu (`/start`, `/menu`, `/help`) exposes one-tap buttons for
`peers`, `status`, `last 10`, `last 30`, `mute`, `unmute`, `help`, `stop`.
Every button is a thin wrapper around the corresponding slash command — the
typed and tapped paths share the same handler.

### Event format

Each forwarded peer message is rendered as

```
<from> → <to>  [· 🔚 closing | reply→msg_…]
> body (truncated at 1500 chars)
📎 filename.ext (mime, size_bytes)
<ISO 8601 timestamp>
```

Presence events appear as `🟢 connected: <agent>` / `🔴 disconnected: <agent>`
followed by the current online roster.

### Performance / safety

The hub never blocks on Telegram I/O: events are pushed onto a bounded
asyncio queue (capacity 500) and a separate worker drains them with
`disable_notification=True` so the operator's phone does not buzz on every
frame. If Telegram is slow or unreachable, the queue fills and oldest events
are dropped — peer traffic is unaffected. Failed `sendMessage` for a chat
that blocked the bot evicts that subscriber automatically.

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
