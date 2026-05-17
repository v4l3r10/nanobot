# Peer-pair conversation loop mitigation — analysis & open work

Working notes from the deploy of the peer-chat architecture (branch
`personal/bronzo-peer`, May 2026). The system shipped is functional; this
document collects what we tried, what didn't fully solve the loop problem,
and the proposed next step we deferred.

## Symptom observed

Two bots (e.g. `bronzo` ↔ `manuzio`) entered courtesy reply loops after
finishing a real task. Pattern: 12+ messages of the form
"Alla prossima!" / "👋" / "Buona serata!" with no information content,
each waking the other's agent loop and consuming tokens.

After the first round of mitigations the loop was shorter (~8 msgs) but
mutated into a **meta-narrative loop**: bots writing italicised asides
about their own silence — `*tace*`, `*il gatto tace*`,
`*nessuna risposta — fine vera*` — recursively responding to each
other's asides.

## Root cause

LLM bias toward politeness reciprocity. The model sees an inbound
message and feels obliged to reply, even when the substantive task is
over. Skill instructions saying "you decide whether to reply" do not
override this prior reliably.

## What we shipped

### `closing=true` protocol (system-level break, model-driven decision)

A boolean flag on the wire frame and as a `peer_say` parameter. Semantics:

- Sender declares "this is my last message — don't expect a reply"
- Router persists `closing=1` in `messages.closing` (migration 0003)
- Receiver's `PeerChannel._on_frame` sees `closing=true` and **skips
  `publish_inbound`** — the agent loop is NOT awakened. Zero LLM call,
  zero tokens, no reply possible.
- Message is still in DB; `peer_thread_show` shows it.

This breaks the loop *deterministically* whenever the sender uses the
flag. It is the strongest property of the design: receiver cannot reply
to a closing message because its agent never sees the turn.

### Skill (peer-chat/SKILL.md, v5)

Distributed to all 5 workspaces. Hard rules:

1. **HARD RULE 1**: end every exchange with `closing=true`. List of
   forbidden pleasantries-without-closing. Required substitutes. Two
   NEVER clauses:
   - never on a question
   - never another message after closing — including narrative asides
2. **HARD RULE 2**: never promise the user "ti giro la risposta appena
   arriva". Use `peer_thread_show` for on-demand recap.

## Residual problem

Even with skill v5 the model occasionally:

- Sends substantive follow-up *after* its own closing (defeats the
  protocol — the post-closing message is non-closing and wakes the peer)
- Adds ironic micro-asides (`*Già.*`, `*Stop.*`) interpreting the rule as
  "no serious messages after closing" rather than "no messages at all"

The loop is shorter and lower-cost than v0, but not zero.

## Open work — option B (deferred)

System-level lock on outbound after a self-issued closing. Sketch:

- `PeerChannel` keeps `_closed_threads_outbound: dict[str, float]`
  mapping `peer_id → timestamp_of_last_closing_sent`
- In `send()`, before serializing the frame, check: if peer in map AND
  now − ts < N seconds AND new message lacks `closing=true`, raise
  `RuntimeError("you closed thread with <peer> N seconds ago — wait or
  open a new thread by omitting thread_id")`
- The tool surfaces the error to the agent: it learns from the failed
  call rather than from skill prose
- Window: 30 s seems right (covers narrative urge, short enough not to
  block legitimate reopens)

This honours the model's own declaration (`closing=true`) and prevents
it from immediately changing its mind. It is **not heuristic** — only
acts on the explicit, model-set flag — so the criticism we levelled
against echo-suppression heuristics doesn't apply.

Estimated cost: ~20 lines in `nanobot/channels/peer.py`, 2 unit tests,
no router/db changes.

## Other options we considered and rejected

- **Echo suppression heuristic** (whitelist of pleasantry tokens, or
  trigram similarity between inbound and last outbound): rejected as
  language-bound, fragile, and opaque to the model.
- **Rate-limit per peer-pair**: useful as a safety net but blunt — could
  block legitimate bursty traffic (e.g. 14 chunks of file transfer).
- **`peer_close_thread(peer)` tool**: redundant with `closing=true` on
  the last `peer_say`; one less primitive to teach.
- **System prompt injection ("you closed N s ago, are you sure?")**:
  considered for option B but felt softer than a hard reject.

## Patologia "duplicated final_content" — fix shipped 2026-05-08 20:35

After fixing the `_progress` dispatch (below), a second mechanism for the
same symptom showed up in the bronzo↔naldo log at 20:04:

```
20:04:30  naldo → bronzo  cls=1  "Ciao Bronzo! Oggi con Arnaldo..."  (the real reply)
20:04:33  naldo → bronzo  cls=0  "Fatto. Messaggio inviato e chiuso. Ora silenzio."
```

The second message is the **agent loop's `final_content`** —
`nanobot/agent/loop.py:1112` always emits an OutboundMessage to
`msg.channel/msg.chat_id` after the turn completes, carrying the model's
narrative summary. When the channel is `peer:*` and the model has
already addressed the peer through `peer_say(closing=true)`, the
narrative becomes a SECOND message without the closing flag, waking the
receiver and re-igniting the loop.

`MessageTool` already had a guard for this exact pattern
(`mt._sent_in_turn` check at loop.py:1097) — `PeerSayTool` did not.

**Fix shipped:**
- `PeerSayTool` now mirrors `MessageTool`: `ContextVar` flag
  `_sent_in_turn`, `start_turn()` reset, set to True on successful
  `execute()`.
- `loop.py` calls `peer_say_tool.start_turn()` alongside the existing
  `MessageTool.start_turn()` at the top of every dispatch.
- After the turn, if `msg.channel == "peer"` AND the peer_say tool's
  `_sent_in_turn` is True, the narrative final_content is dropped
  (parallel to MessageTool's existing suppression).

Tests added in `tests/tools/test_peer_tool.py`:
- `test_peer_say_marks_sent_in_turn_after_success`
- `test_peer_say_does_not_mark_sent_on_validation_error`
- `test_peer_say_does_not_mark_sent_on_send_failure`

Combined with the `_progress` filter, this should eliminate every
non-LLM-controlled source of duplicated peer messages. Anything left
over is the model's deliberate choice (which the skill v5 NEVER rules
plus optional Option B post-close lock are meant to address).

## Patologia "ghost twin" — RESOLVED 2026-05-08 20:30

**Root cause identified and fixed.** The empty/echo twin messages observed
on the peer plane were NOT generated by the LLM completion text-parts
(initial hypothesis) — they were `_progress` / `_tool_hint` events
emitted by the agent loop's progress callback (`on_progress` /
`_bus_progress` in `nanobot/agent/loop.py:1028`) and dispatched to every
channel with `send_progress=True` (default in `ChannelsConfig`).

For human-facing channels (Telegram) those events surface as "tool hint"
status updates. For the peer plane they were forwarded as full peer_say
frames with empty content (or with the tool name as content), no
`closing=true` flag, waking the receiver's agent loop and re-igniting
the loop alongside legitimate peer_say outputs.

**Fix** (commit pending): `PeerChannel.send()` now drops any
OutboundMessage carrying any of `_progress`, `_tool_hint`, `_retry_wait`,
`_stream_delta`, `_stream_end`, `_streamed` — these are bookkeeping
events meant for human channels, not for inter-agent traffic. Also drops
any message whose `content` is empty/whitespace-only as defence in depth.

Tests added in `tests/channels/test_peer_channel.py`:
- `test_send_filters_progress_noise_outbound` (6 noise flag variants)
- `test_send_filters_empty_content` (empty/whitespace-only)

This eliminates the structural source of the loop. Option B
(post-close outbound lock) becomes optional polish rather than a
must-have.

## Patologia "ghost twin" osservata 2026-05-08 19:50-19:53 (bronzo↔naldo) — initial hypothesis (now superseded)

NEW root-cause discovered, more important than option B:

**Symptom**: every meaningful `peer_say` from the agent loop produces TWO
DB rows in `messages` at the same created_at timestamp:
- Row A: `closing=1` with the real body (the tool_use args)
- Row B: `closing=0` with body length 0 (empty), or with body = literal
  string `"peer_say"`, `"peer_thread_show"`, `"```json\n{}\n```"`, etc.

Sample (DB extract):

```
19:51:43  naldo→bronzo  cls=1  len=441  "Bronzo, mi sa che c'è un bug..."
19:51:43  naldo→bronzo  cls=0  len=0    (empty)
```

The agent loop logged **only one** `Tool call: peer_say(...)` in that
window. So the second row is being emitted by something other than the
agent's tool call itself.

**Hypothesis**: the agent loop is materialising the LLM completion's
non-tool-use text parts as a second OutboundMessage. When the model
returns `[tool_use, text=""]` or `[text="peer_say", tool_use, text=""]`,
each text fragment becomes a separate OutboundMessage to the peer chat
session. The empty/echo messages are not closing, so they wake the
receiver despite the legitimate closing=true on the sibling tool_use.

**Effect**: closing=true is technically working (5+ skip events logged
at the receiver), but the empty twin always wakes them anyway, so the
loop never actually ends. Worse, the receiver sees `peer:bronzo` chat
turns containing literal strings `peer_say` / `{}` which then enter the
context history as if Bronzo had said them — corrupting future turns.

**Where to look next**:

- `nanobot/agent/loop.py`: how does it iterate the LLM's content blocks
  and turn them into OutboundMessage? Does it filter empty text parts?
  Does it skip text that is literally a tool name?
- `nanobot/channels/peer.py:send()`: maybe add a guard `if not
  msg.content.strip(): return` as a defence-in-depth (silently drop
  empty outbound). But the real fix is upstream.
- Cross-check with Telegram channel: does Telegram filter empty replies?
  If yes, we need the same guard on peer.

**Why this matters more than option B (post-close lock)**: option B
addresses the model "changing its mind" after closing. The ghost twin
is structural — even a perfectly disciplined model gets bitten by it
because the empty twin is generated by the harness, not by the model's
choice. Fix the twin first, then re-evaluate whether option B is still
needed.

## Next-time checklist

When picking this back up:

1. Read this note
2. **First priority**: investigate ghost-twin. Reproduce by sending one
   peer_say from a Bronzo session and counting DB rows. Trace through
   `loop.py` content-block handling.
3. Add `if not msg.content.strip(): return` guard in
   `PeerChannel.send()` as a quick mitigation while the upstream fix is
   designed
4. THEN implement option B (post-close outbound lock with 30 s window),
   tests, skill v6 update
5. Re-run the manuzio + naldo exchanges from this session as regression
6. Monitor for the "tool name echoed as body" case (`peer_say` as
   literal body content) — likely fixed by the same upstream change
