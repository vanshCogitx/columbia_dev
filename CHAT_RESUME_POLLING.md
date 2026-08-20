# Resuming a chat after disconnect (frontend integration guide)

## The problem this solves

`POST /api/chats/message` streams the AI's reply over SSE. If the client
disconnects mid-stream — tab switch, backgrounding, network blip — two
things used to break:

1. **Data loss.** The in-flight response generation was cancelled along
   with the SSE connection, so the AI's reply was never saved.
2. **No way to tell what's happening on reopen.** Even after fixing (1),
   reopening the chat while the reply was still generating looked
   identical to a chat that simply had no reply yet.

Both are now fixed on the backend:

- Response generation runs as an independent background task, decoupled
  from the client's HTTP connection. A disconnect no longer aborts it —
  the reply is still generated and saved.
- `GET /api/chats/history` now reports whether a reply is still being
  generated for the requested chat, plus a live-updating status phrase,
  so the frontend can show the same "Searching the catalog…" style
  indicator it shows during an active SSE stream — even when there's no
  SSE stream open at all.

**No frontend changes are needed for the data-loss fix** — it's purely
backend. The rest of this doc is about the second part: showing progress
when you reopen a chat mid-generation.

## API contract

`GET /api/chats/history?email=...&session_id=...` response now includes
two extra fields (both `null`/absent unless `session_id` is passed and
resolves to a chat you own):

```json
{
  "email": "user@example.com",
  "messages": [ ... ],
  "pending": true,
  "pending_status": "Searching the catalog..."
}
```

| Field            | Type            | Meaning                                                                 |
|------------------|-----------------|--------------------------------------------------------------------------|
| `pending`        | `bool \| null`  | `true` while a reply is being generated for this chat. `null` if `session_id` wasn't passed (can't represent "pending" across multiple chats at once). |
| `pending_status` | `string \| null`| Current rotating status phrase (same pool of ~38 phrases used during live SSE streaming, e.g. `"Checking availability..."`). Only set while `pending` is `true`. |

`pending_status` updates roughly every 3-5 seconds for as long as
`pending` stays `true`, then both fields reset to `null`/`false` the
moment the reply is saved — at which point `messages` in that same
response already includes the finished reply.

This reuses the endpoint you already call to load chat history — there's
no separate endpoint to hit.

## What to add to the frontend

### 1a. Framework-agnostic version

Use this if you're not on React, or want the plain logic to adapt to
whatever state layer you're using:

```js
async function loadChatAndResume(email, sessionId, { onMessages, onPendingChange, onStatusChange, signal }) {
  const POLL_MS = 3000; // matches the backend ticker's own cadence

  while (!signal.aborted) {
    const res = await fetch(
      `/api/chats/history?email=${encodeURIComponent(email)}&session_id=${sessionId}`,
      { signal }
    );
    const data = await res.json();

    onMessages(data.messages);
    onPendingChange(!!data.pending);
    onStatusChange(data.pending_status ?? null);

    if (!data.pending) break; // done — stop polling
    await new Promise(r => setTimeout(r, POLL_MS));
  }
}
```

Call it with an `AbortController` so you can cancel the loop when the
chat is closed/unmounted:

```js
const controller = new AbortController();
loadChatAndResume(email, sessionId, {
  onMessages: (msgs) => { /* update your message list state */ },
  onPendingChange: (p) => { /* show/hide the pending indicator */ },
  onStatusChange: (s) => { /* update the status phrase text */ },
  signal: controller.signal,
});
// later, on close/unmount:
controller.abort();
```

### 1b. React hook version (if that's your stack)

```jsx
function useChatHistory(email, sessionId) {
  const [messages, setMessages] = useState([]);
  const [pending, setPending] = useState(false);
  const [pendingStatus, setPendingStatus] = useState(null);

  useEffect(() => {
    if (!sessionId) return;
    const controller = new AbortController();

    async function poll() {
      try {
        const res = await fetch(
          `/api/chats/history?email=${encodeURIComponent(email)}&session_id=${sessionId}`,
          { signal: controller.signal }
        );
        const data = await res.json();
        setMessages(data.messages);
        setPending(!!data.pending);
        setPendingStatus(data.pending_status ?? null);
        if (data.pending) setTimeout(poll, 3000); // matches the backend ticker's own cadence
      } catch (e) {
        if (e.name !== "AbortError") console.error(e);
      }
    }
    poll();

    return () => controller.abort();
  }, [email, sessionId]);

  return { messages, pending, pendingStatus };
}
```

### 2. Wire it up wherever you already load a chat's history

Trigger this on:
- Initial chat open/mount.
- Switching back to a chat tab that was left mid-generation.

No new call needed beyond what you already do to load history — just
read the two extra fields off the response you're already getting.

### 3. Render `pendingStatus` like a live status event

While `pending` is `true`, show `pendingStatus` exactly where you'd
render the `status` SSE event during an active stream (e.g. a "Searching
the catalog…" line above the message list). When `pending` flips to
`false`, `messages` already contains the finished reply — render it
normally, no extra fetch required.

### 4. Decide what happens if the live SSE stream itself disconnects

Right now, nothing auto-falls-back to polling while a stream is actively
being watched — the poll only kicks in when the chat is (re)opened. If
you want the UI to seamlessly switch from "live SSE" to "poll-based" the
moment a stream drops (rather than only on next open), start the poll
loop from the SSE's `onerror`/close handler too.

## Notes / edge cases

- Both a live SSE viewer and a polling client read from the same
  Redis-backed rotation, so the phrases you see match between the two —
  they're not two independently-random sequences.
- Polling every 3s can occasionally show the same phrase twice in a row
  if the backend hasn't rotated yet (its own cadence is also ~3-5s) —
  this is expected and matches what a live viewer would perceive too.
- `pending`/`pending_status` are backed by a Redis key with a 300s
  safety-net TTL, in case the backend process dies mid-generation without
  running its cleanup — so worst case a stuck `pending: true` self-clears
  after 5 minutes even if something crashes.
