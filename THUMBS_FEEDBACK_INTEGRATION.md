# Thumbs up/down feedback (frontend integration guide)

## What this is

`POST /api/chats/feedback` lets a user rate a specific AI response with a
thumbs up/down and an optional reason. It's the entry point to the backend's
evals system (Layer 1: human feedback capture; Layer 2: an LLM judge scores
the response and, for bad ones, attributes the likely cause — none of that
concerns the frontend, it's all async and invisible to the user).

The endpoint returns immediately after saving the rating — the judging
happens in the background. **Nothing in this integration is
request/response-blocking**; you never need to wait on it.

## API contract

```
POST /api/chats/feedback
Authorization: Bearer <jwt>
Content-Type: application/json

{
  "message_id": 909,
  "rating": "up",
  "reason": "Optional free-text reason, mainly useful for 'down' ratings"
}
```

| Field        | Type              | Notes                                                                 |
|--------------|-------------------|------------------------------------------------------------------------|
| `message_id` | `int`, required   | The AI message being rated (see below for where this comes from).      |
| `rating`     | `string`, required| `"up"` or `"down"` — anything else returns `422`.                      |
| `reason`     | `string`, optional| Free text. Send it for `"down"` ratings especially — it feeds directly into the judge's scoring. |

Response (`200`):
```json
{ "id": 12, "message_id": 909, "rating": "up" }
```

Errors:
- `404` — message doesn't exist, isn't an AI message, or doesn't belong to the logged-in user (ownership is checked server-side via the JWT).
- `422` — `rating` isn't `"up"`/`"down"`.

## Where `message_id` comes from

`GET /api/chats/history` already returns an `id` field on every message
(including AI ones) — that's the same id this endpoint expects. You don't
need a new call to get it; it's already in the history response you're
presumably already rendering:

```json
{
  "messages": [
    { "id": 909, "role": "assistant", "content": "...", "created_at": "..." }
  ]
}
```

Only messages with `role: "assistant"` can be rated — the backend rejects
anything else (user messages, or a message from a different chat/user).

## What to add to the frontend

### 1. A feedback API helper

```js
async function submitFeedback(messageId, rating, reason) {
  const res = await fetch('/api/chats/feedback', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ message_id: messageId, rating, reason }),
  });
  if (!res.ok) throw new Error(`feedback failed: ${res.status}`);
  return res.json();
}
```

### 2. Thumbs buttons on each AI message

Render them only on `role === "assistant"` messages (every message object
from `/api/chats/history` already carries the `id` you need):

```jsx
function MessageFeedback({ messageId }) {
  const [rating, setRating] = useState(null); // null | 'up' | 'down'
  const [showReasonInput, setShowReasonInput] = useState(false);

  async function rate(newRating, reason) {
    setRating(newRating); // optimistic — the call is fire-and-forget
    try {
      await submitFeedback(messageId, newRating, reason);
    } catch (e) {
      console.error(e);
      setRating(null); // revert on failure
    }
  }

  if (rating) {
    return <span className="feedback-done">{rating === 'up' ? '👍' : '👎'} Thanks for the feedback!</span>;
  }

  return (
    <div className="feedback-buttons">
      <button onClick={() => rate('up')}>👍</button>
      <button onClick={() => setShowReasonInput(true)}>👎</button>
      {showReasonInput && (
        <ReasonPrompt
          onSubmit={(reason) => rate('down', reason)}
          onSkip={() => rate('down')}
        />
      )}
    </div>
  );
}
```

### 3. UX notes

- **Thumbs up**: fire immediately on click, no reason needed. A reason is
  accepted for `up` ratings too if you want to let users say *why* they
  liked it, but it's optional and mostly matters for `down`.
- **Thumbs down**: worth prompting for a short reason before submitting
  (a text input, or a couple of canned options like "wrong products" /
  "didn't understand my request" / "other") — the more specific the
  reason, the more useful the judge's scoring is. Not required though;
  submitting with `reason: null` is fine.
- **One rating per message**: the backend doesn't currently enforce
  uniqueness (nothing stops a second `POST` for the same `message_id`),
  so once a user has rated a message, disable/hide the buttons client-side
  (as in the example above) rather than relying on the backend to reject
  a duplicate.
- **Don't block the UI on this call** — it's a background-trigger, not a
  data fetch the rest of the page depends on. Fire it, show an optimistic
  "thanks" state, and move on.
