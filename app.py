"""
HR Policy Chatbot — Streamlit App (merged frontend + backend)
Gradient dark-blue UI with rich chat bubbles and session management.
Runs fully self-contained — no separate FastAPI server needed.
"""

import os
import uuid
import time

import streamlit as st

# ---- Inject Groq secrets into env before pipeline imports ----------------
try:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["GROQ_MODEL"]   = st.secrets["GROQ_MODEL"]
except Exception:
    pass  # fall back to .env / environment variables

from rag_pipeline import RAGPipeline       # noqa: E402
from chat_history import ChatHistoryManager  # noqa: E402

# ---- Configuration -------------------------------------------------------
PAGE_TITLE = "HR Policy Assistant"
PAGE_ICON  = "👥"

# ---- Page Config (must be first Streamlit call) -------------------------
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="centered",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS ----------------------------------------------------------
st.markdown("""
<style>
/* ===================== Global Background ===================== */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
    min-height: 100vh;
}
[data-testid="stHeader"] {
    background: transparent;
}

/* ===================== Sidebar ===================== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #0d0d2b 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] * {
    color: #e0e0ff !important;
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem !important;
    padding-bottom: 1rem !important;
}
[data-testid="stSidebar"] .block-container {
    padding-top: 0.5rem !important;
}
[data-testid="stSidebar"] hr {
    margin: 0.4rem 0 !important;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stMarkdown {
    margin-bottom: 0.3rem !important;
}
[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(90deg, #6a3de8, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    width: 100%;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(106,61,232,0.4) !important;
}

/* ===================== Main Content ===================== */
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.main-header h1 {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.main-header p {
    color: rgba(200,200,255,0.65);
    font-size: 1rem;
}

/* ===================== Chat Container ===================== */
.chat-container {
    max-height: 520px;
    overflow-y: auto;
    padding: 1rem 0.5rem;
    scrollbar-width: thin;
    scrollbar-color: rgba(106,61,232,0.4) transparent;
}
.chat-container::-webkit-scrollbar { width: 5px; }
.chat-container::-webkit-scrollbar-track { background: transparent; }
.chat-container::-webkit-scrollbar-thumb {
    background: rgba(106,61,232,0.5);
    border-radius: 10px;
}

/* ===================== Message Bubbles ===================== */
.msg-wrapper {
    display: flex;
    margin: 0.6rem 0;
    align-items: flex-end;
    gap: 0.6rem;
}
.msg-wrapper.user  { flex-direction: row-reverse; }
.msg-wrapper.bot   { flex-direction: row; }

.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    flex-shrink: 0;
}
.avatar.user { background: linear-gradient(135deg, #6a3de8, #a855f7); }
.avatar.bot  { background: linear-gradient(135deg, #0ea5e9, #38bdf8); }

.bubble {
    max-width: 78%;
    padding: 0.75rem 1.1rem;
    border-radius: 18px;
    font-size: 0.93rem;
    line-height: 1.55;
    word-wrap: break-word;
}
.bubble.user {
    background: linear-gradient(135deg, #6a3de8dd, #a855f7dd);
    color: #fff;
    border-bottom-right-radius: 4px;
    box-shadow: 0 4px 15px rgba(106,61,232,0.3);
}
.bubble.bot {
    background: rgba(255,255,255,0.08);
    color: #e8e8ff;
    border: 1px solid rgba(255,255,255,0.1);
    border-bottom-left-radius: 4px;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 15px rgba(0,0,0,0.2);
}

/* ===================== Sources Badge ===================== */
.sources-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin: 0.3rem 0 0 2.8rem;
}
.source-badge {
    background: rgba(52,211,153,0.15);
    border: 1px solid rgba(52,211,153,0.35);
    color: #6ee7b7;
    font-size: 0.73rem;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-weight: 500;
}

/* ===================== Timestamp ===================== */
.msg-time {
    font-size: 0.68rem;
    color: rgba(200,200,255,0.35);
    margin: 0 0.4rem;
    align-self: flex-end;
}

/* ===================== Input Area ===================== */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #000000 !important;
    padding: 0.75rem 1rem !important;
    font-size: 0.95rem !important;
    caret-color: #a78bfa !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(167,139,250,0.6) !important;
    box-shadow: 0 0 0 3px rgba(167,139,250,0.15) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #000000 !important;
}

/* ===================== Send Button ===================== */
div[data-testid="column"] .stButton button {
    background: linear-gradient(90deg, #6a3de8, #a855f7) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    height: 46px !important;
    font-size: 1.2rem !important;
    font-weight: 700 !important;
    width: 100% !important;
    margin-top: 0 !important;
    transition: all 0.2s;
    box-shadow: 0 4px 15px rgba(106,61,232,0.35) !important;
}
div[data-testid="column"] .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(106,61,232,0.5) !important;
}

/* ===================== Divider ===================== */
hr { border-color: rgba(255,255,255,0.08) !important; }

/* ===================== Metrics / Info Boxes ===================== */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 0.75rem 1rem;
    border: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stMetric"] label { color: rgba(200,200,255,0.6) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #a78bfa !important;
    font-size: 1.4rem !important;
}

/* ===================== Typing Indicator ===================== */
.typing-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    background: rgba(255,255,255,0.06);
    border-radius: 18px;
    border-bottom-left-radius: 4px;
    width: fit-content;
    border: 1px solid rgba(255,255,255,0.1);
    margin: 0.6rem 0;
}
.dot {
    width: 7px; height: 7px;
    background: #60a5fa;
    border-radius: 50%;
    animation: bounce 1.2s infinite;
}
.dot:nth-child(2) { animation-delay: 0.2s; }
.dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.6; }
    30%            { transform: translateY(-6px); opacity: 1; }
}

/* ===================== Suggested Questions ===================== */
.suggest-btn button {
    background: rgba(96,165,250,0.12) !important;
    border: 1px solid rgba(96,165,250,0.3) !important;
    color: #93c5fd !important;
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.9rem !important;
    margin: 0.2rem !important;
    transition: all 0.2s;
}
.suggest-btn button:hover {
    background: rgba(96,165,250,0.25) !important;
}

/* ===================== Welcome Card ===================== */
.welcome-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    text-align: center;
    color: rgba(200,200,255,0.75);
    font-size: 0.92rem;
    line-height: 1.7;
}
.welcome-card .emoji { font-size: 2.5rem; margin-bottom: 0.5rem; }

/* ===================== Status Dot ===================== */
.status-online {
    display: inline-block;
    width: 8px; height: 8px;
    background: #34d399;
    border-radius: 50%;
    margin-right: 6px;
    box-shadow: 0 0 6px #34d399;
}
</style>
""", unsafe_allow_html=True)


# ---- Cached singletons ---------------------------------------------------

@st.cache_resource(show_spinner="Loading HR Policy knowledge base…")
def get_pipeline():
    return RAGPipeline()

@st.cache_resource
def get_history_manager():
    return ChatHistoryManager()


# ---- Session State Defaults ----------------------------------------------
if "session_id"  not in st.session_state:
    st.session_state.session_id  = str(uuid.uuid4())[:8]
if "messages"    not in st.session_state:
    st.session_state.messages    = []
if "thinking"    not in st.session_state:
    st.session_state.thinking    = False
if "input_value" not in st.session_state:
    st.session_state.input_value = ""
if "input_key"   not in st.session_state:
    st.session_state.input_key   = 0
if "input_placeholder" not in st.session_state:
    st.session_state.input_placeholder = "Ask an HR policy question…"


# ---- Helper Functions ----------------------------------------------------

def ask_question(query: str) -> dict:
    pipeline = get_pipeline()
    history  = get_history_manager()
    session_history = history.get_history(st.session_state.session_id)
    result = pipeline.chat(query, session_history)
    history.add_message(st.session_state.session_id, query, result["answer"])
    return result


def clear_session():
    history = get_history_manager()
    history.clear_history(st.session_state.session_id)


def render_message(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    sources = msg.get("sources", [])
    ts      = msg.get("ts", "")
    avatar  = "👤" if role == "user" else "🤖"

    st.markdown(f"""
    <div class="msg-wrapper {role}">
        <div class="avatar {role}">{avatar}</div>
        <div class="bubble {role}">{content}</div>
        <span class="msg-time">{ts}</span>
    </div>
    """, unsafe_allow_html=True)

    if sources and role == "bot":
        badges = "".join(
            f'<span class="source-badge">📄 {s}</span>' for s in sources
        )
        st.markdown(
            f'<div class="sources-row">{badges}</div>',
            unsafe_allow_html=True,
        )


def submit_query(query: str):
    if not query.strip():
        return
    ts = time.strftime("%H:%M")
    st.session_state.messages.append(
        {"role": "user", "content": query.strip(), "ts": ts}
    )
    st.session_state.thinking = True


# ---- Sidebar -------------------------------------------------------------
with st.sidebar:
    st.markdown("## 👥 HR Assistant")
    st.markdown("---")

    # Recent chat history (last 5 user questions)
    st.markdown("**Recent Chats**")
    user_msgs = [m for m in st.session_state.messages if m["role"] == "user"]
    recent = user_msgs[-5:] if len(user_msgs) >= 5 else user_msgs
    if recent:
        for msg in reversed(recent):
            label = msg["content"][:40] + "…" if len(msg["content"]) > 40 else msg["content"]
            st.markdown(
                f'<div style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);'
                f'border-radius:8px;padding:0.4rem 0.7rem;margin-bottom:0.35rem;'
                f'font-size:0.8rem;color:rgba(200,200,255,0.75);cursor:default;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" '
                f'title="{msg["content"]}">💬 {label}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<p style="font-size:0.78rem;color:rgba(200,200,255,0.35);">No history yet.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button("🗑️  Clear Conversation"):
        clear_session()
        st.session_state.messages           = []
        st.session_state.session_id         = str(uuid.uuid4())[:8]
        st.session_state.input_key         += 1
        st.session_state.input_placeholder  = "Ask an HR policy question…"
        st.rerun()

    st.markdown("---")

    # Suggested topics
    st.markdown("**Quick Topics**")
    suggestions = [
        "Vacation leave policy",
        "Remote work rules",
        "Health insurance",
        "Sick leave days",
        "401(k) matching",
        "Performance reviews",
        "Expense reimbursement",
        "Maternity leave",
    ]
    for s in suggestions:
        if st.button(s, key=f"sug_{s}"):
            submit_query(s)
            st.rerun()


# ---- Main Area -----------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>👥 HR Policy Assistant</h1>
    <p>Ask me anything about Apex Technologies HR policies</p>
</div>
""", unsafe_allow_html=True)

# Welcome card (shown only when no messages)
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-card">
        <div class="emoji">👋</div>
        <strong>Welcome!</strong> I'm your HR Policy Assistant.<br>
        Ask me about <em>leave policies, benefits, remote work, expenses, performance,
        conduct guidelines</em>, and more.<br><br>
        <em>Try: "How many vacation days do I get after 3 years?"</em>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
chat_placeholder = st.container()
with chat_placeholder:
    for msg in st.session_state.messages:
        render_message(msg)

    # Typing indicator + actual pipeline call
    if st.session_state.thinking:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:0.5rem;margin:0.5rem 0;">
            <div class="avatar bot" style="width:36px;height:36px;border-radius:50%;
                background:linear-gradient(135deg,#0ea5e9,#38bdf8);
                display:flex;align-items:center;justify-content:center;font-size:1.1rem;">🤖</div>
            <div class="typing-indicator">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        last_user = next(
            (m["content"] for m in reversed(st.session_state.messages)
             if m["role"] == "user"),
            None,
        )
        if last_user:
            try:
                result = ask_question(last_user)
                st.session_state.messages.append({
                    "role":    "bot",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "ts":      time.strftime("%H:%M"),
                })
            except Exception as e:
                st.session_state.messages.append({
                    "role":    "bot",
                    "content": f"⚠️ Error: {str(e)}",
                    "sources": [],
                    "ts":      time.strftime("%H:%M"),
                })
        st.session_state.thinking          = False
        st.session_state.input_key        += 1
        st.session_state.input_placeholder = "Ask a follow-up question…"
        st.rerun()

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
st.markdown("---")

# ---- Input Row -----------------------------------------------------------
col_input, col_btn = st.columns([9, 1])
with col_input:
    user_input = st.text_input(
        label            = "input",
        label_visibility = "collapsed",
        placeholder      = st.session_state.input_placeholder,
        key              = f"chat_input_{st.session_state.input_key}",
    )
with col_btn:
    send_clicked = st.button("➤", key="send_btn")

# Handle send
if (send_clicked or (user_input and st.session_state.get("_last_input") != user_input)) \
        and user_input.strip():
    st.session_state["_last_input"] = user_input
    submit_query(user_input)
    st.rerun()

# Handle Enter key
if user_input and not send_clicked:
    if st.session_state.get("_prev_input") != user_input:
        st.session_state["_prev_input"] = user_input

# ---- Footer --------------------------------------------------------------
st.markdown(
    "<p style='text-align:center;color:rgba(200,200,255,0.25);font-size:0.75rem;"
    "margin-top:1.5rem;'>Apex Technologies · HR Policy Chatbot v1.0 · "
    "For urgent matters contact HR Ext. 1001</p>",
    unsafe_allow_html=True,
)
