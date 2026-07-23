// Core streaming client — works in any JS frontend (React, plain JS, etc.)

const API_BASE = "http://127.0.0.1:8123"; // change to your deployed backend URL

/**
 * Sends a message and streams the response.
 * @param {Object} opts
 * @param {string} opts.token - bearer token from /api/auth/login
 * @param {number|null} opts.chatId - pass null/omit for a new chat, existing id to continue one
 * @param {string} opts.text - the user's message
 * @param {Object} handlers - callbacks per event type
 */
async function sendChatMessage(
  { token, chatId, text },
  { onChat, onStatus, onMessage, onProductDiscovery, onConversationId, onEnd, onError }
) {
  const res = await fetch(`${API_BASE}/api/chats/message`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(chatId ? { chat_id: chatId, text } : { text }),
  });

  if (!res.ok || !res.body) {
    onError?.({ detail: `HTTP ${res.status}` });
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });

    let idx;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const rawEvent = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      if (!rawEvent.trim()) continue;

      const lines = rawEvent.split("\n");
      const eventLine = lines.find((l) => l.startsWith("event:"));
      const dataLine = lines.find((l) => l.startsWith("data:"));
      if (!eventLine || !dataLine) continue;

      const eventName = eventLine.replace("event:", "").trim();
      const payload = JSON.parse(dataLine.replace("data:", "").trim());
      const v = payload.v;

      switch (eventName) {
        case "chat":
          onChat?.(v); // { id, session_id, title }
          break;
        case "status":
          onStatus?.(v); // "Thinking..." etc
          break;
        case "message":
          onMessage?.(v); // final prose string
          break;
        case "product_discovery":
          onProductDiscovery?.(v); // [ { opening, status, products, activity_context, closing, contentType } ]
          break;
        case "conversation_id":
          onConversationId?.(v); // cartesian session_id string
          break;
        case "end":
          onEnd?.();
          break;
        case "error":
          onError?.(payload); // { detail }
          break;
      }
    }
  }
}

// ── Example usage (React component) ──────────────────────────────────────
//
// import { useState } from "react";
//
// function ChatBox({ token }) {
//   const [chatId, setChatId] = useState(null);
//   const [status, setStatus] = useState("");
//   const [reply, setReply] = useState("");
//   const [products, setProducts] = useState([]);
//
//   async function handleSend(text) {
//     setStatus("");
//     setReply("");
//     setProducts([]);
//     await sendChatMessage(
//       { token, chatId, text },
//       {
//         onChat: (chat) => setChatId(chat.id),          // store for next message
//         onStatus: (s) => setStatus(s),                  // show as typing indicator
//         onMessage: (msg) => { setReply(msg); setStatus(""); },
//         onProductDiscovery: (arr) => setProducts(arr[0]?.products ?? []),
//         onEnd: () => console.log("stream done"),
//         onError: (e) => console.error(e.detail),
//       }
//     );
//   }
//
//   return (/* render status, reply, products.map(...) */);
// }

export { sendChatMessage };
