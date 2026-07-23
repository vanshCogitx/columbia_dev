# Columbia Backend — Frontend Integration Guide

Base URL: `https://columbia-backend-copy-v1.onrender.com`
(local dev: `http://localhost:8000`)

All `/api/*` endpoints below (except `/api/auth/register` and `/api/auth/login`) require:
```
Authorization: Bearer <token>
```
Token comes from `/api/auth/login`.

---

## 1. Auth

### `POST /api/auth/register`
**Request**
```json
{
  "username": "vansh",
  "email": "vansh@example.com",
  "password": "password123",
  "name": "Vansh"
}
```
**Response** `200`
```json
{
  "id": 1,
  "username": "vansh",
  "email": "vansh@example.com",
  "name": "Vansh"
}
```
**Errors**: `400` if username/email already registered.

---

### `POST /api/auth/login`
**Request**
```json
{
  "username": "vansh",
  "password": "password123"
}
```
**Response** `200`
```json
{
  "token": "16d9769b-b3dc-4f0f-bb03-c889fc9d715a",
  "user": {
    "id": 1,
    "username": "vansh",
    "email": "vansh@example.com",
    "name": "Vansh"
  }
}
```
**Errors**: `401` invalid username/password.

Store `token` (e.g. localStorage) and attach as `Authorization: Bearer <token>` on every request below.

---

### `GET /api/auth/me`
No body. Returns the logged-in user's profile.
**Response** `200`
```json
{
  "id": 1,
  "username": "vansh",
  "email": "vansh@example.com",
  "name": "Vansh"
}
```

---

## 2. Chat — send a message (SSE stream)

### `POST /api/chats/message`

This is the **only** endpoint needed to both start a new chat and continue an existing one. There is no separate "create chat" call anymore.

**Request — new chat** (omit `chat_id`, or send `null`)
```json
{
  "text": "suggest a waterproof hiking jacket"
}
```

**Request — continue existing chat** (send the `chat_id` you got back from a previous call)
```json
{
  "chat_id": 12,
  "text": "what about something cheaper"
}
```

**Response**: `Content-Type: text/event-stream`. A sequence of SSE events, always in this order:

```
event: chat
data: {"v": {"id": 12, "session_id": "32667075-9c07-4d08-9e6a-3e7d8acbe5d0", "title": "New Chat"}}

event: status
data: {"v": "Thinking..."}

event: status
data: {"v": "Searching the catalog..."}

event: status
data: {"v": "Fetching product details..."}

event: message
data: {"v": "For your next adventure, here are some top-notch waterproof hiking jackets designed to keep you dry and comfortable.\n\nLet me know if you'd like more details or need help picking the perfect one!"}

event: product_discovery
data: {"v": [{"opening": "...", "status": {"result": "success", "message": "Found waterproof hiking jackets suitable for your needs."}, "products": [{"family_id": "F001829", "title": "Switchback III Waterproof Hiking Jacket", "description": "...", "category_level_1": "Apparel", "category_level_2": "Men's", "product_type": "Outerwear", "sport": "Hiking", "price_min": 37.99, "price_max": 44.99, "primary_product_id": "9895865188631", "thumbnail_image": "https://cdn.shopify.com/.../WK0127-010_1.jpg", "variants": [{"product_id": "9895865188631", "name": "Columbia Women's Black Switchback III Waterproof Hiking Jacket", "price": 37.99, "size": "OS", "color": "Black", "stock_quantity": 53, "availability": "in_stock", "url": "https://...", "image_url": "https://...", "handle": "...", "tags": "..."}], "score": 0.6106}], "tech_highlight": {"requested_technology": "Waterproof", "highlighted_benefit": "..."}, "closing": "Let me know if you'd like more details or need help picking the perfect jacket for your needs!", "contentType": "product_discovery"}]}

event: conversation_id
data: {"v": "32667075-9c07-4d08-9e6a-3e7d8acbe5d0"}

event: end
data: {"v": {}}
```

**Event reference**

| event | fires | `v` shape | what to do with it |
|---|---|---|---|
| `chat` | always, first | `{id, session_id, title}` | store `id` — send back as `chat_id` on the next message in this conversation |
| `status` | 0 or more times | string, e.g. `"Thinking..."` | show as a typing/progress indicator; number of ticks varies per request |
| `message` | always, once | string | the AI's prose reply — render as the chat bubble |
| `product_discovery` | only if products were found | array with **one** object: `{opening, status, products[], activity_context/tech_highlight, closing, contentType}` | `v[0].products` is the array to render as product cards |
| `conversation_id` | always, once | string (Cartesian session uuid) | informational only, not needed for your own state — `chat.id` is what you send back |
| `end` | always, last | `{}` | stream is done, stop reading |
| `error` | only on failure, replaces remaining events | `{detail: "..."}` | show an error state |

**Note on `product_discovery`**: the object's exact extra fields (`activity_context`, `tech_highlight`, `tech_summary`, etc.) can vary call to call — Cartesian doesn't always return the same secondary fields. Only `opening`, `status`, `products`, `closing`, `contentType` are guaranteed present when this event fires. Design the UI to read `products` defensively and ignore unknown extra keys.

**Timing**: this can take anywhere from ~5s to ~60s depending on how long Cartesian takes to answer. Don't set a client-side timeout under 60s.

**Auth**: this is a `POST` with a custom `Authorization` header — you **cannot** use the native `EventSource` API (GET-only, no custom headers). Use `fetch()` + manual `ReadableStream` parsing. Reference implementation: `frontend_integration_example.js` in this repo (`sendChatMessage()` function — copy-paste ready, includes a commented React usage example).

---

## 3. Chat — list, history, delete

### `GET /api/chats`
No body. Lists all chats for the logged-in user, most recent first.
**Response** `200`
```json
[
  {
    "id": 12,
    "session_id": "32667075-9c07-4d08-9e6a-3e7d8acbe5d0",
    "title": "suggest a waterproof hiking jacket",
    "created_at": "2026-07-17 13:43:43"
  }
]
```

### `GET /api/chats/{chat_id}/messages`
No body. Full message history for one chat, oldest first.
**Response** `200`
```json
[
  {
    "id": 53,
    "chat_id": 12,
    "sender": "user",
    "text": "suggest a waterproof hiking jacket",
    "created_at": "2026-07-17 13:43:43"
  },
  {
    "id": 54,
    "chat_id": 12,
    "sender": "ai",
    "text": "{\"opening\": \"...\", \"products\": [...], \"closing\": \"...\"}",
    "created_at": "2026-07-17 13:44:41"
  }
]
```
**Note**: for `sender: "ai"` messages, `text` is the **raw** Cartesian response — may be a plain string, or a JSON string (possibly fenced in ` ```json `), matching whatever `/api/chats/message` streamed live. If you need it pre-parsed the same way the live stream does, ask backend to add a parsed variant — currently history returns raw text only.

### `DELETE /api/chats/{chat_id}`
No body. Deletes the chat and all its messages.
**Response** `200`
```json
{ "detail": "Chat deleted successfully" }
```
**Errors**: `404` if chat doesn't exist or isn't owned by the logged-in user.

---

## 4. Error shape (all endpoints)

```json
{ "detail": "human-readable error message" }
```
Common codes: `401` (bad/missing auth), `403` (bad API key on tool endpoints), `404` (not found/not yours), `422` (bad request body — FastAPI validation), `500`/`502`/`503`/`504` (server/upstream errors).
