import streamlit as st
import time
from memory import add_message, get_conversation
from prompting import build_prompt
from llm import stream_llm

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Local LLM Chatbot",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Custom styling (Streamlit-stable)
# -----------------------------
st.markdown(
    """
    <style>
        .chat-shell {
            max-width: 700px;
            margin-left: auto;
            margin-right: auto;
            display: flex;
            flex-direction: column;
        }

        /* Full-width row for alignment */
        .row {
            width: 100%;
            display: flex;
        }
        .row.user { justify-content: flex-end; }
        .row.assistant { justify-content: flex-start; }

        /* Message bubbles */
        .bubble {
            display: inline-block;
            width: fit-content;
            max-width: 75%;
            padding: 0.6rem 0.9rem;
            border-radius: 14px;
            line-height: 1.45;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            font-size: 1rem;
        }

        .bubble.user {
            background: #6B7280;
            color: #FFFFFF;
        }

        .bubble.assistant {
            background: #FFFFFF;
            color: #000000;
            border: 1px solid #E5E7EB;
        }

        /* Spacer layers (background-matching) */
        .spacer-sm {
            height: 8px;
            width: 100%;
        }

        .spacer-lg {
            height: 16px;
            width: 100%;
        }

        .latency-row {
            display: flex;
            justify-content: space-between;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Session state init
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "latencies" not in st.session_state:
    st.session_state.latencies = []  # TTFT in ms

if "pending_user" not in st.session_state:
    st.session_state.pending_user = None

# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("Local LLM running via **Ollama**")

    st.divider()
    st.markdown("**Model**")
    st.code("phi", language="text")

    last_latency = (
        f"{st.session_state.latencies[-1]:.0f} ms"
        if st.session_state.latencies
        else "—"
    )
    avg_latency = (
        f"{sum(st.session_state.latencies) / len(st.session_state.latencies):.0f} ms"
        if st.session_state.latencies
        else "—"
    )

    st.markdown("**Latency (ms)**")
    st.markdown(
        f"""
        <div class="latency-row">
            <span>Last (TTFT)</span>
            <span>{last_latency}</span>
        </div>
        <div class="latency-row">
            <span>Avg</span>
            <span>{avg_latency}</span>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    if st.button("🧹 Clear conversation"):
        st.session_state.messages = []
        st.session_state.latencies = []
        st.session_state.pending_user = None
        st.rerun()

# -----------------------------
# Main title
# -----------------------------
st.title("💬 Local LLM Chatbot")
st.caption("GPU-accelerated · Streaming · Offline")

# -----------------------------
# Render chat history (double-spaced)
# -----------------------------
st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    st.markdown(
        f"""
        <div class="row {role}">
            <div class="bubble {role}">{content}</div>
        </div>
        <div class="spacer-sm"></div>
        <div class="spacer-lg"></div>
        """,
        unsafe_allow_html=True
    )

# Dedicated placeholder for streaming assistant
stream_row = st.empty()

st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Input
# -----------------------------
user_input = st.chat_input("Type your message…")

if user_input and st.session_state.pending_user is None:
    st.session_state.messages.append({"role": "user", "content": user_input})
    add_message("user", user_input)
    st.session_state.pending_user = user_input
    st.rerun()

# -----------------------------
# Generate assistant response (TTFT latency)
# -----------------------------
if st.session_state.pending_user is not None:
    prompt = build_prompt(get_conversation())

    full_response = ""
    start_time = time.perf_counter()
    first_token_time = None

    for token in stream_llm(prompt):
        if first_token_time is None:
            first_token_time = time.perf_counter()
            ttft_ms = (first_token_time - start_time) * 1000
            st.session_state.latencies.append(ttft_ms)

        full_response += token
        stream_row.markdown(
            f"""
            <div class="row assistant">
                <div class="bubble assistant">{full_response}</div>
            </div>
            <div class="spacer-sm"></div>
            <div class="spacer-lg"></div>
            """,
            unsafe_allow_html=True
        )

    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )
    add_message("assistant", full_response)

    st.session_state.pending_user = None
    st.rerun()
