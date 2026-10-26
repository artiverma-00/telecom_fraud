"""Streamlit support desk for the project_2 telecom RAG assistant."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import streamlit as st
from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
load_dotenv(PROJECT_DIR.parent / ".env")

st.set_page_config(
    page_title="Telecom Care | Support Desk",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

SAMPLE_QUESTIONS = {
    "Connectivity": [
        "Why is my mobile internet so slow?",
        "My calls keep dropping. What should I do?",
        "How do I enable Wi-Fi calling?",
    ],
    "Billing": [
        "Why is my bill higher than usual this month?",
        "I was charged for roaming but had a bundle active.",
    ],
    "SIM & device": [
        "My phone shows SIM not detected after a restart.",
        "How do I unlock my phone for another network?",
        "How do I activate international roaming?",
    ],
}

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root {
      color-scheme:dark;
      --ink:#eff7f4; --muted:#9bb0a9; --teal:#48d7bd; --line:#263f3b;
      --panel:#142320; --panel-raised:#192c28; --background:#0b1514;
    }
    html, body, [data-testid="stApp"], .stApp,
    [data-testid="stAppViewContainer"], [data-testid="stMain"],
    [data-testid="stMainBlockContainer"], [data-testid="stBottomBlockContainer"] {
      background:var(--background) !important; color:var(--ink); font-family:'DM Sans',sans-serif;
    }
    [data-testid="stHeader"], [data-testid="stToolbar"] { background:var(--background) !important; }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] { background:#101d1b !important; }
    [data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"] { background:transparent; }
    h1,h2,h3,h4 { color:var(--ink) !important; font-family:'Manrope',sans-serif; letter-spacing:-.035em; }
    p, label, li, [data-testid="stMarkdownContainer"] { color:#dce9e5; }
    [data-testid="stCaptionContainer"], .stCaption { color:var(--muted) !important; }
    [data-testid="stSidebar"] { border-right:1px solid var(--line); }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label { color:#dce9e5 !important; }
    .brand { color:var(--teal); font:800 1.18rem Manrope,sans-serif; letter-spacing:-.04em; }
    .eyebrow { color:var(--teal); font-size:.72rem; font-weight:700; letter-spacing:.16em; text-transform:uppercase; }
    .hero { padding:1.2rem 0 .7rem; }
    .hero h1 { font-size:clamp(2rem,4vw,3.15rem); margin:.45rem 0 .35rem; }
    .hero-copy { color:#a9bdb6; font-size:1.02rem; max-width:700px; }
    .status { display:inline-flex; align-items:center; gap:.5rem; color:#a9c7bd; font-size:.82rem; }
    .status-dot { width:.52rem; height:.52rem; display:inline-block; background:#4bddaa; border-radius:50%; box-shadow:0 0 12px #4bddaa66; }
    .feature { background:linear-gradient(145deg,#162a26,#11201e); border:1px solid var(--line); border-radius:16px; padding:.85rem 1rem; min-height:84px; }
    .feature-title { color:#eff7f4; font-weight:700; font-size:.92rem; }
    .feature-copy { color:#95aaa2; font-size:.79rem; margin-top:.2rem; }
    .stButton > button { border-radius:12px; color:#e9f4ef; background:#182a26; border:1px solid #2b4841; transition:all .15s ease; }
    .stButton > button:hover { background:#203a34; color:#fff; border-color:#48d7bd; }
    .stButton > button[kind="primary"] { background:#40cdb4; color:#08201b; border-color:#40cdb4; font-weight:700; }
    .stButton > button[kind="primary"]:hover { background:#70e2cd; color:#08201b; border-color:#70e2cd; }
    div[data-testid="stChatMessage"] { background:var(--panel) !important; border:1px solid var(--line); border-radius:18px; padding:.65rem 1rem; }
    div[data-testid="stChatMessage"] p, div[data-testid="stChatMessage"] li { color:#e3efeb !important; }
    div[data-testid="stChatInput"] { background:var(--panel) !important; border:1px solid #31524a; }
    div[data-testid="stChatInput"] textarea { color:var(--ink) !important; }
    div[data-testid="stChatInput"] textarea::placeholder { color:#91a69e !important; opacity:1; }
    [data-baseweb="input"] input, [data-baseweb="textarea"] textarea { color:var(--ink) !important; background:var(--panel) !important; }
    [data-testid="stAlert"] { background:#142923 !important; border-color:#285849 !important; color:#e5f4eb !important; }
    [data-testid="stAlert"] p, [data-testid="stAlert"] div { color:#e5f4eb !important; }
    hr { border-color:var(--line); }
    a, a:visited { color:#70dfcb !important; }
    ::selection { background:#1c7464; color:#fff; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_chain():
    """Build and cache the RAG chain; embedding weights load on first question."""
    from rag_chain import build_chain

    return build_chain()


def ask_support(question: str) -> None:
    """Append the turn and stream an answer from the RAG chain."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📡"):
        try:
            with st.spinner("Searching support articles and resolved cases…"):
                chain = get_chain()
                recent_turns = st.session_state.messages[:-1][-4:]
                if recent_turns:
                    prior_context = "\n".join(
                        f"{turn['role'].title()}: {turn['content']}" for turn in recent_turns
                    )
                    chain_question = (
                        f"Recent conversation:\n{prior_context}\n\n"
                        f"Customer's current question: {question}"
                    )
                else:
                    chain_question = question
                answer = st.write_stream(chain.stream(chain_question))
            if not isinstance(answer, str):
                answer = str(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            detail = str(exc).lower()
            if "groq_api_key" in detail or "api_key" in detail or "authentication" in detail:
                message = (
                    "I can’t connect to Groq. Check that `GROQ_API_KEY` is set in the repository "
                    "`.env` file, then restart Streamlit."
                )
            elif "model" in detail and ("not found" in detail or "404" in detail):
                message = "The configured Groq model is unavailable. Check the model ID in `project_2/rag_chain.py`."
            else:
                message = "I couldn’t complete that support search. Please try again in a moment."
            st.error(message)
            st.session_state.messages.append({"role": "assistant", "content": message})


if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


with st.sidebar:
    st.markdown('<div class="brand">📡 TELECOM CARE</div>', unsafe_allow_html=True)
    st.caption("Support, made simpler.")
    st.divider()
    st.markdown("**Browse common issues**")
    st.caption("Choose a question to start a support search.")

    for category, questions in SAMPLE_QUESTIONS.items():
        with st.expander(category, expanded=(category == "Connectivity")):
            for index, sample in enumerate(questions):
                if st.button(sample, key=f"{category}_{index}", use_container_width=True):
                    st.session_state.pending_question = sample
                    st.rerun()

    st.divider()
    st.markdown("**Need a person?**")
    st.caption("For urgent account help, call **611** or use the MyTelecom app.")
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.caption("Answers use support FAQs, help guides, and resolved cases.")


st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">CUSTOMER CARE · ALWAYS HERE TO HELP</div>
      <h1>Let’s get you reconnected.</h1>
      <p class="hero-copy">Tell us what’s going on with your service. We’ll search support guides and resolved cases for practical next steps.</p>
      <div class="status"><span class="status-dot"></span> Support assistant ready</div>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_columns = st.columns(3)
features = [
    ("◉  Connectivity", "Slow data, dropped calls, Wi-Fi calling"),
    ("▤  Billing & plans", "Charges, roaming bundles, plan questions"),
    ("▣  SIM & device", "Activation, SIM issues, phone unlocking"),
]
for column, (title, description) in zip(feature_columns, features):
    with column:
        st.markdown(
            f'<div class="feature"><div class="feature-title">{title}</div>'
            f'<div class="feature-copy">{description}</div></div>',
            unsafe_allow_html=True,
        )

if not st.session_state.messages:
    st.markdown("#### What can we help with?")
    st.caption("Pick a common question or describe the issue in your own words.")
    quick_questions = [
        "Why is my mobile internet so slow?",
        "Why is my bill higher than usual?",
        "My phone says SIM not detected.",
    ]
    quick_columns = st.columns(3)
    for index, (column, sample) in enumerate(zip(quick_columns, quick_questions)):
        with column:
            if st.button(sample, key=f"quick_{index}", use_container_width=True):
                st.session_state.pending_question = sample
                st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "📡"):
        st.markdown(message["content"])

typed_question = st.chat_input("Describe your mobile service issue…")
question = st.session_state.pending_question or typed_question
st.session_state.pending_question = None

if question:
    ask_support(question)
