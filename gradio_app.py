"""
gradio_app.py
=============
AtlasCare Customer-Facing Gradio UI.

Features:
- Conversation history passed to backend via enriched message context
- Session timeout: warns after inactivity, auto-terminates session
- Graceful exit on "exit" keyword
- Separate from admin dashboard (runs on port 7860)
"""

import os
import time
import threading
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

PORT             = os.getenv("PORT", "8000")
API_HOST         = "127.0.0.1"
API_URL          = f"http://{API_HOST}:{PORT}/query"
GRADIO_HOST      = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT      = int(os.getenv("GRADIO_PORT", "7860"))
DEFAULT_SESSION  = os.getenv("DEFAULT_SESSION_ID", "sess-cust001")

# Inactivity settings (seconds)
WARN_AFTER_S     = int(os.getenv("SESSION_WARN_AFTER_S",  "120"))   # 2 min
TIMEOUT_AFTER_S  = int(os.getenv("SESSION_TIMEOUT_AFTER_S", "180")) # 3 min

# ---------------------------------------------------------------------------
# Per-session state (keyed by session_id)
# ---------------------------------------------------------------------------
_sessions: dict[str, dict] = {}
_lock = threading.Lock()


def _get_session(session_id: str) -> dict:
    with _lock:
        if session_id not in _sessions:
            _sessions[session_id] = {
                "last_activity": time.time(),
                "warned":        False,
                "terminated":    False,
                "history":       [],   # list of (user_msg, bot_msg)
            }
        return _sessions[session_id]


def _touch(session_id: str):
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["last_activity"] = time.time()
            _sessions[session_id]["warned"]        = False


def _reset_session(session_id: str):
    with _lock:
        _sessions[session_id] = {
            "last_activity": time.time(),
            "warned":        False,
            "terminated":    False,
            "history":       [],
        }


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def chat(message: str, history: list, session_id: str):
    """
    Main chat handler.
    Returns (updated_history, cleared_input, status_text).
    """
    session_id = session_id.strip() or DEFAULT_SESSION
    sess       = _get_session(session_id)

    # ── Terminated session ────────────────────────────────────────────
    if sess["terminated"]:
        history = history + [
            (message,
             "Your session has ended. Please refresh the page or enter a "
             "new Session ID to start a new conversation.")
        ]
        return history, "", "🔴 Session ended"

    # ── Exit keyword ──────────────────────────────────────────────────
    if message.strip().lower() in ("exit", "quit", "bye", "goodbye"):
        sess["terminated"] = True
        history = history + [
            (message,
             "Thank you for reaching out to AtlasCare! Your session has been "
             "closed. We hope we were able to assist you today. "
             "Have a great day! 😊")
        ]
        return history, "", "🔴 Session closed"

    # ── Empty message guard ───────────────────────────────────────────
    if not message.strip():
        return history, "", _status_text(session_id)

    # ── Build payload — send conversation context to backend ──────────
    # We include the last 3 exchanges as context prefix so the backend
    # planner has memory of recent turns (order IDs, etc.)
    context_prefix = _build_context(sess["history"])
    full_message   = f"{context_prefix}{message}" if context_prefix else message

    payload = {"message": full_message, "session_id": session_id}

    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        bot_reply = response.json()["response"]
    except requests.exceptions.Timeout:
        bot_reply = (
            "I'm sorry, the request took too long to process. "
            "Please try again in a moment."
        )
    except requests.exceptions.ConnectionError:
        bot_reply = (
            "I'm unable to connect to the support service right now. "
            "Please ensure the server is running and try again."
        )
    except Exception as exc:
        bot_reply = (
            f"Something went wrong on our end. Please try again. "
            f"(Error: {str(exc)[:80]})"
        )

    # ── Update state ──────────────────────────────────────────────────
    _touch(session_id)
    sess["history"].append((message, bot_reply))
    # Keep only last 6 turns in memory
    if len(sess["history"]) > 6:
        sess["history"] = sess["history"][-6:]

    # ── Detect task completion and prompt for exit ────────────────────
    completion_keywords = [
        "anything else", "further assistance", "help you with",
        "is there anything", "let us know", "have a great"
    ]
    task_complete = any(k in bot_reply.lower() for k in completion_keywords)

    history = history + [(message, bot_reply)]

    if task_complete:
        history = history + [
            (None,
             "✅ It looks like your request has been resolved! "
             "If you have any further questions, feel free to ask. "
             "Otherwise, type **exit** to close the session.")
        ]

    return history, "", _status_text(session_id)


def _build_context(history: list) -> str:
    """Build a compact context prefix from recent conversation turns."""
    if not history:
        return ""
    recent = history[-3:]  # last 3 turns
    lines  = ["[Recent conversation for context:]"]
    for user_msg, bot_msg in recent:
        lines.append(f"Customer: {user_msg[:150]}")
        lines.append(f"Agent: {bot_msg[:150]}")
    lines.append("[Current message:]")
    return "\n".join(lines) + "\n"


def _status_text(session_id: str) -> str:
    sess = _get_session(session_id)
    if sess["terminated"]:
        return "🔴 Session ended"
    idle = int(time.time() - sess["last_activity"])
    remaining = max(0, TIMEOUT_AFTER_S - idle)
    if idle < WARN_AFTER_S:
        return f"🟢 Session active"
    return f"🟡 Idle {idle}s — session closes in {remaining}s"


# ---------------------------------------------------------------------------
# Inactivity checker — runs every 15 seconds in background
# ---------------------------------------------------------------------------
def _inactivity_checker():
    """Background thread that pushes inactivity warnings."""
    while True:
        time.sleep(15)
        now = time.time()
        with _lock:
            for sid, sess in _sessions.items():
                if sess["terminated"]:
                    continue
                idle = now - sess["last_activity"]
                if idle >= TIMEOUT_AFTER_S:
                    sess["terminated"] = True
                elif idle >= WARN_AFTER_S and not sess["warned"]:
                    sess["warned"] = True
                    # Append warning to history — Gradio will pick it up on next render
                    warn_mins = int((TIMEOUT_AFTER_S - idle) / 60) + 1
                    sess["history"].append((
                        None,
                        f"⚠️ **Inactivity detected.** Your session will be "
                        f"automatically closed in approximately **{warn_mins} minute(s)** "
                        f"due to inactivity. Please send a message to stay connected, "
                        f"or type **exit** if you're done."
                    ))


threading.Thread(target=_inactivity_checker, daemon=True).start()


# ---------------------------------------------------------------------------
# Inactivity poll — called by Gradio timer to show warnings
# ---------------------------------------------------------------------------
def poll_inactivity(history: list, session_id: str):
    """
    Called by Gradio every 10s to surface any inactivity warnings
    that the background thread has queued.
    """
    session_id = session_id.strip() or DEFAULT_SESSION
    sess       = _get_session(session_id)

    # Drain any queued system messages
    new_messages = [
        turn for turn in sess["history"]
        if turn[0] is None and turn not in history
    ]
    if new_messages:
        history = history + new_messages

    # Auto-termination message
    if sess["terminated"] and not any(
        "session has been closed" in (t[1] or "") or "automatically closed" in (t[1] or "")
        for t in history
    ):
        idle = int(time.time() - sess["last_activity"])
        history = history + [(
            None,
            f"🔴 **Your session has been automatically closed** after "
            f"{idle // 60} minute(s) of inactivity. "
            f"Please refresh the page or enter a new Session ID to start again. "
            f"Thank you for using AtlasCare!"
        )]

    return history, _status_text(session_id)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AtlasCare Customer Support", theme=gr.themes.Soft()) as demo:

    gr.Markdown("# 🛒 AtlasCare Customer Support")
    gr.Markdown(
        "Welcome! I'm your AtlasCare support assistant. "
        "Ask me about your orders, refunds, returns, or anything else."
    )

    with gr.Row():
        session_input = gr.Textbox(
            label="Session ID",
            value=DEFAULT_SESSION,
            scale=3,
            placeholder="e.g. sess-cust001",
        )
        status_box = gr.Textbox(
            label="Session Status",
            value="🟢 Session active",
            interactive=False,
            scale=2,
        )

    chatbot = gr.Chatbot(
        height=480,
        show_label=False,
        bubble_full_width=False,
    )

    with gr.Row():
        message_box = gr.Textbox(
            placeholder="Type your message here... (type 'exit' to end session)",
            show_label=False,
            scale=8,
        )
        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("🗑 Clear Chat", scale=1)
        new_session_btn = gr.Button("🔄 New Session", scale=1)

    gr.Markdown(
        "_Session auto-closes after 3 minutes of inactivity. "
        "You'll receive a warning at 2 minutes._",
    )

    # ── Event handlers ────────────────────────────────────────────────
    send_btn.click(
        fn=chat,
        inputs=[message_box, chatbot, session_input],
        outputs=[chatbot, message_box, status_box],
    )
    message_box.submit(
        fn=chat,
        inputs=[message_box, chatbot, session_input],
        outputs=[chatbot, message_box, status_box],
    )

    def clear_chat():
        return [], ""

    clear_btn.click(fn=clear_chat, outputs=[chatbot, message_box])

    def new_session(session_id):
        _reset_session(session_id.strip() or DEFAULT_SESSION)
        return [], "", "🟢 Session active"

    new_session_btn.click(
        fn=new_session,
        inputs=[session_input],
        outputs=[chatbot, message_box, status_box],
    )

    # ── Inactivity poll every 10 seconds ──────────────────────────────
    demo.load(
        fn=poll_inactivity,
        inputs=[chatbot, session_input],
        outputs=[chatbot, status_box],
        every=10,
    )


if __name__ == "__main__":
    print(f"AtlasCare Customer UI → http://{GRADIO_HOST}:{GRADIO_PORT}")
    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=False,
    )