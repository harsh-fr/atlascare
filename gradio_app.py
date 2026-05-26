"""
gradio_app.py
=============
AtlasCare Customer-Facing Gradio UI.

Features
--------
- Authentication: login / register / forgot-password (OTP = 9999 in demo)
- Conversation history passed to backend via enriched message context
- Session timeout: warns after inactivity, auto-terminates session
- Server-side session history cleared on session end (Fix 1)
- Greetings routed through LLM for natural responses (Fix 2)
- "Request resolved" banner uses backend task_complete signal (Fix 3)
- Graceful exit on "exit" keyword
"""

import os
import secrets
import time
import threading
import requests
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

PORT         = os.getenv("PORT", "8000")
API_HOST     = "127.0.0.1"
API_URL      = f"http://{API_HOST}:{PORT}/query"
AUTH_URL     = f"http://{API_HOST}:{PORT}/auth"
DELETE_URL   = f"http://{API_HOST}:{PORT}/session"
GRADIO_HOST  = os.getenv("GRADIO_HOST", "127.0.0.1")
GRADIO_PORT  = int(os.getenv("GRADIO_PORT", "7860"))

# Inactivity settings (seconds)
WARN_AFTER_S    = int(os.getenv("SESSION_WARN_AFTER_S",  "120"))   # 2 min
TIMEOUT_AFTER_S = int(os.getenv("SESSION_TIMEOUT_AFTER_S", "180")) # 3 min

# ---------------------------------------------------------------------------
# Per-session Gradio-side state (keyed by session_id)
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
                "history":       [],
            }
        return _sessions[session_id]


def _touch(session_id: str):
    with _lock:
        if session_id in _sessions:
            _sessions[session_id]["last_activity"] = time.time()
            _sessions[session_id]["warned"]        = False


def _reset_session(session_id: str):
    """Clear Gradio-side session state AND backend memory."""
    _clear_backend_session(session_id)
    with _lock:
        _sessions[session_id] = {
            "last_activity": time.time(),
            "warned":        False,
            "terminated":    False,
            "history":       [],
        }


def _clear_backend_session(session_id: str) -> None:
    """Notify backend to free server-side conversation history (best-effort)."""
    if not session_id:
        return
    try:
        requests.delete(f"{DELETE_URL}/{session_id}", timeout=5)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------
def _do_login(username: str, password: str):
    try:
        resp = requests.post(f"{AUTH_URL}/login",
                             json={"username": username, "password": password},
                             timeout=10)
        return resp.json()
    except Exception as exc:
        return {"success": False, "error": f"Connection error: {str(exc)[:60]}"}


def _do_register(username: str, password: str, email: str, customer_id: str):
    try:
        resp = requests.post(f"{AUTH_URL}/register",
                             json={"username": username, "password": password,
                                   "email": email, "customer_id": customer_id},
                             timeout=10)
        return resp.json()
    except Exception as exc:
        return {"success": False, "error": f"Connection error: {str(exc)[:60]}"}


def _do_request_otp(username: str):
    try:
        resp = requests.post(f"{AUTH_URL}/request-otp",
                             json={"username": username}, timeout=10)
        return resp.json()
    except Exception as exc:
        return {"success": False, "message": f"Connection error: {str(exc)[:60]}"}


def _do_reset_password(username: str, otp: str, new_password: str):
    try:
        resp = requests.post(f"{AUTH_URL}/reset-password",
                             json={"username": username, "otp": otp,
                                   "new_password": new_password},
                             timeout=10)
        return resp.json()
    except Exception as exc:
        return {"success": False, "error": f"Connection error: {str(exc)[:60]}"}


# ---------------------------------------------------------------------------
# Core chat function
# ---------------------------------------------------------------------------
def chat(message: str, history: list, session_id: str):
    """Main chat handler. Returns (updated_history, cleared_input, status_text)."""
    if not session_id:
        return (history + [(message, "Please log in to start a conversation.")],
                "", "❌ Not authenticated")

    sess = _get_session(session_id)

    # ── Terminated session ─────────────────────────────────────────────
    if sess["terminated"]:
        return (
            history + [(message,
                        "Your session has ended. Click **New Session** "
                        "to start a fresh conversation.")],
            "", "🔴 Session ended",
        )

    # ── Exit keyword ───────────────────────────────────────────────────
    if message.strip().lower() in ("exit", "quit", "bye", "goodbye"):
        sess["terminated"] = True
        _clear_backend_session(session_id)
        return (
            history + [(message,
                        "Thank you for reaching out to AtlasCare! "
                        "Your session has been closed. Have a great day! 😊")],
            "", "🔴 Session closed",
        )

    # ── Empty message guard ────────────────────────────────────────────
    if not message.strip():
        return history, "", _status_text(session_id)

    # ── Build enriched payload with recent context ─────────────────────
    context_prefix = _build_context(sess["history"])
    full_message   = f"{context_prefix}{message}" if context_prefix else message
    payload        = {"message": full_message, "session_id": session_id}

    try:
        response      = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data          = response.json()
        bot_reply     = data["response"]
        task_complete = data.get("task_complete", False)
    except requests.exceptions.Timeout:
        bot_reply     = ("I'm sorry, the request took too long. "
                         "Please try again in a moment.")
        task_complete = False
    except requests.exceptions.ConnectionError:
        bot_reply     = ("I'm unable to connect to the support service. "
                         "Please ensure the server is running and try again.")
        task_complete = False
    except Exception as exc:
        bot_reply     = (f"Something went wrong. Please try again. "
                         f"(Error: {str(exc)[:80]})")
        task_complete = False

    _touch(session_id)
    sess["history"].append((message, bot_reply))
    if len(sess["history"]) > 6:
        sess["history"] = sess["history"][-6:]

    history = history + [(message, bot_reply)]

    # Show resolved banner only when the backend confirms the task is done
    if task_complete:
        history = history + [(
            None,
            "✅ Your request has been resolved! "
            "If you have more questions, feel free to ask. "
            "Otherwise, type **exit** to close the session.",
        )]

    return history, "", _status_text(session_id)


def _build_context(history: list) -> str:
    if not history:
        return ""
    recent = history[-3:]
    lines  = ["[Recent conversation for context:]"]
    for user_msg, bot_msg in recent:
        lines.append(f"Customer: {user_msg[:150]}")
        lines.append(f"Agent: {bot_msg[:150]}")
    lines.append("[Current message:]")
    return "\n".join(lines) + "\n"


def _status_text(session_id: str) -> str:
    if not session_id:
        return "❌ Not authenticated"
    sess = _get_session(session_id)
    if sess["terminated"]:
        return "🔴 Session ended"
    idle      = int(time.time() - sess["last_activity"])
    remaining = max(0, TIMEOUT_AFTER_S - idle)
    if idle < WARN_AFTER_S:
        return "🟢 Session active"
    return f"🟡 Idle {idle}s — session closes in {remaining}s"


# ---------------------------------------------------------------------------
# Inactivity checker — background thread
# ---------------------------------------------------------------------------
def _inactivity_checker():
    while True:
        time.sleep(15)
        now = time.time()
        with _lock:
            for sid, sess in list(_sessions.items()):
                if sess["terminated"]:
                    continue
                idle = now - sess["last_activity"]
                if idle >= TIMEOUT_AFTER_S:
                    sess["terminated"] = True
                    # Clear backend memory when session times out
                    threading.Thread(
                        target=_clear_backend_session, args=(sid,), daemon=True
                    ).start()
                elif idle >= WARN_AFTER_S and not sess["warned"]:
                    sess["warned"]  = True
                    warn_mins = int((TIMEOUT_AFTER_S - idle) / 60) + 1
                    sess["history"].append((
                        None,
                        f"⚠️ **Inactivity detected.** Your session will be "
                        f"automatically closed in approximately **{warn_mins} minute(s)**. "
                        f"Send a message to stay connected, or type **exit** if you're done.",
                    ))


threading.Thread(target=_inactivity_checker, daemon=True).start()


# ---------------------------------------------------------------------------
# Inactivity poll — Gradio timer
# ---------------------------------------------------------------------------
def poll_inactivity(history: list, session_id: str):
    if not session_id:
        return history, "❌ Not authenticated"

    sess = _get_session(session_id)

    new_messages = [
        turn for turn in sess["history"]
        if turn[0] is None and turn not in history
    ]
    if new_messages:
        history = history + new_messages

    if sess["terminated"] and not any(
        "session has been closed" in (t[1] or "") or "automatically closed" in (t[1] or "")
        for t in history
    ):
        idle = int(time.time() - sess["last_activity"])
        history = history + [(
            None,
            f"🔴 **Your session has been automatically closed** after "
            f"{idle // 60} minute(s) of inactivity. "
            f"Click **New Session** to start again. Thank you for using AtlasCare!",
        )]

    return history, _status_text(session_id)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="AtlasCare Customer Support", theme=gr.themes.Soft()) as demo:

    # Persistent states
    session_id_state   = gr.State("")
    customer_id_state  = gr.State("")

    # ── AUTH SECTION ─────────────────────────────────────────────────────
    with gr.Column(visible=True) as auth_col:
        gr.Markdown("# 🛒 AtlasCare Customer Support")
        gr.Markdown("Sign in to get help with your orders, refunds, and more.")

        with gr.Tabs():

            # ── Login ────────────────────────────────────────────────────
            with gr.Tab("🔐 Login"):
                login_username = gr.Textbox(label="Username", placeholder="e.g. priya")
                login_password = gr.Textbox(label="Password", type="password")
                login_btn      = gr.Button("Login", variant="primary")
                login_msg      = gr.Markdown("")

            # ── Register ─────────────────────────────────────────────────
            with gr.Tab("📝 Register"):
                reg_username    = gr.Textbox(label="Username")
                reg_email       = gr.Textbox(label="Email")
                reg_customer_id = gr.Textbox(label="Customer ID",
                                             placeholder="e.g. CUST-004")
                reg_password    = gr.Textbox(label="Password", type="password")
                reg_btn         = gr.Button("Register", variant="primary")
                reg_msg         = gr.Markdown("")

            # ── Forgot Password ──────────────────────────────────────────
            with gr.Tab("🔑 Forgot Password"):
                fp_username    = gr.Textbox(label="Username")
                fp_request_btn = gr.Button("Request OTP")
                fp_otp_msg     = gr.Markdown("")
                with gr.Column(visible=False) as fp_reset_col:
                    fp_otp         = gr.Textbox(label="Enter OTP")
                    fp_new_pass    = gr.Textbox(label="New Password", type="password")
                    fp_confirm_btn = gr.Button("Reset Password", variant="primary")
                    fp_reset_msg   = gr.Markdown("")

    # ── CHAT SECTION ──────────────────────────────────────────────────────
    with gr.Column(visible=False) as chat_col:
        with gr.Row():
            gr.Markdown("# 🛒 AtlasCare Customer Support")
            logout_btn = gr.Button("🚪 Logout", scale=0, variant="secondary")

        with gr.Row():
            logged_in_as = gr.Textbox(
                label="Logged in as", interactive=False, scale=3,
            )
            status_box = gr.Textbox(
                label="Session Status", value="🟢 Session active",
                interactive=False, scale=2,
            )

        chatbot = gr.Chatbot(height=480, show_label=False, bubble_full_width=False)

        with gr.Row():
            message_box = gr.Textbox(
                placeholder="Type your message here... (type 'exit' to end session)",
                show_label=False, scale=8,
            )
            send_btn = gr.Button("Send ➤", variant="primary", scale=1)

        with gr.Row():
            clear_btn       = gr.Button("🗑 Clear Chat", scale=1)
            new_session_btn = gr.Button("🔄 New Session", scale=1)

        gr.Markdown(
            "_Session auto-closes after 3 minutes of inactivity. "
            "You'll receive a warning at 2 minutes._",
        )

    # ── EVENT HANDLERS ────────────────────────────────────────────────────

    def do_login(username, password):
        data = _do_login(username.strip(), password)
        if data.get("success"):
            sid = data["session_id"]
            cid = data["customer_id"]
            _get_session(sid)
            return (
                gr.update(visible=False),   # hide auth
                gr.update(visible=True),    # show chat
                sid,                        # session_id_state
                cid,                        # customer_id_state
                username.strip(),           # logged_in_as
                "🟢 Session active",        # status_box
                [],                         # clear chatbot
                "",                         # clear login_msg
            )
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            "", "",
            "",
            "❌ Not authenticated",
            [],
            f"❌ {data.get('error', 'Login failed')}",
        )

    login_btn.click(
        fn=do_login,
        inputs=[login_username, login_password],
        outputs=[auth_col, chat_col, session_id_state, customer_id_state,
                 logged_in_as, status_box, chatbot, login_msg],
    )
    login_password.submit(
        fn=do_login,
        inputs=[login_username, login_password],
        outputs=[auth_col, chat_col, session_id_state, customer_id_state,
                 logged_in_as, status_box, chatbot, login_msg],
    )

    def do_register(username, email, customer_id, password):
        data = _do_register(username.strip(), password,
                            email.strip(), customer_id.strip())
        if data.get("success"):
            return f"✅ {data.get('message', 'Account created. Please log in.')}"
        return f"❌ {data.get('error', 'Registration failed')}"

    reg_btn.click(
        fn=do_register,
        inputs=[reg_username, reg_email, reg_customer_id, reg_password],
        outputs=[reg_msg],
    )

    def do_request_otp(username):
        data = _do_request_otp(username.strip())
        msg  = data.get("message", "OTP requested.")
        return gr.update(value=msg), gr.update(visible=True)

    fp_request_btn.click(
        fn=do_request_otp,
        inputs=[fp_username],
        outputs=[fp_otp_msg, fp_reset_col],
    )

    def do_reset_password(username, otp, new_password):
        data = _do_reset_password(username.strip(), otp.strip(), new_password)
        if data.get("success"):
            return f"✅ {data.get('message', 'Password reset. Please log in.')}"
        return f"❌ {data.get('error', 'Reset failed')}"

    fp_confirm_btn.click(
        fn=do_reset_password,
        inputs=[fp_username, fp_otp, fp_new_pass],
        outputs=[fp_reset_msg],
    )

    def do_logout(session_id):
        if session_id:
            _clear_backend_session(session_id)
            with _lock:
                _sessions.pop(session_id, None)
        return (
            gr.update(visible=True),   # show auth
            gr.update(visible=False),  # hide chat
            "", "",                    # clear states
            "", "",                    # clear logged_in_as, login_msg
            [], "",                    # clear chatbot, message_box
            "🟢 Session active",       # reset status_box
        )

    logout_btn.click(
        fn=do_logout,
        inputs=[session_id_state],
        outputs=[auth_col, chat_col, session_id_state, customer_id_state,
                 logged_in_as, login_msg, chatbot, message_box, status_box],
    )

    send_btn.click(
        fn=chat,
        inputs=[message_box, chatbot, session_id_state],
        outputs=[chatbot, message_box, status_box],
    )
    message_box.submit(
        fn=chat,
        inputs=[message_box, chatbot, session_id_state],
        outputs=[chatbot, message_box, status_box],
    )

    def clear_chat():
        return [], ""

    clear_btn.click(fn=clear_chat, outputs=[chatbot, message_box])

    def new_session(customer_id, old_session_id):
        if not customer_id:
            return [], "", "🟢 Session active", old_session_id
        _clear_backend_session(old_session_id)
        with _lock:
            _sessions.pop(old_session_id, None)
        number  = customer_id.split("-")[-1]
        new_sid = f"sess-CUST{number}-{secrets.token_hex(4)}"
        _get_session(new_sid)
        return [], "", "🟢 Session active", new_sid

    new_session_btn.click(
        fn=new_session,
        inputs=[customer_id_state, session_id_state],
        outputs=[chatbot, message_box, status_box, session_id_state],
    )

    demo.load(
        fn=poll_inactivity,
        inputs=[chatbot, session_id_state],
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
