import os

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
SUGGESTED_PROMPTS = [
    "Pod stuck in Terminating",
    "Liveness probe returning 500 error",
    "ImagePullBackOff troubleshooting",
    "CrashLoopBackOff next steps",
    "OpenStack instance stuck in ERROR",
    "Delete all available OpenStack volumes with a safe script",
]

st.set_page_config(page_title="GlueBot", page_icon=":robot_face:", layout="centered")

if "messages" not in st.session_state:
    st.session_state.messages = []


def call_chat_api(message: str) -> dict[str, str]:
    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={"message": message},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "reply": data.get("reply", "No reply received."),
        "source": data.get("source", "unknown"),
    }


def send_user_message(message: str) -> None:
    msg = message.strip()
    if not msg:
        return

    st.session_state.messages.append({"role": "user", "content": msg})
    try:
        with st.spinner("GlueBot is thinking..."):
            result = call_chat_api(msg)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["reply"],
                "source": result["source"],
            }
        )
    except requests.RequestException as exc:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Chat request failed: {exc}",
                "source": "error",
            }
        )


st.title("GlueBot")
st.caption("Kubernetes + OpenStack assistant")

with st.sidebar:
    st.subheader("System")
    st.caption(f"LLM configured: {'Yes' if (OPENAI_API_KEY or OPENROUTER_API_KEY) else 'No'}")
    if st.button("Check API health", use_container_width=True):
        try:
            response = requests.get(f"{BACKEND_URL}/", timeout=5)
            response.raise_for_status()
            st.success("API is reachable")
        except requests.RequestException as exc:
            st.error(f"API unreachable: {exc}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Quick prompts")
    for idx, prompt in enumerate(SUGGESTED_PROMPTS):
        if st.button(prompt, key=f"prompt_{idx}", use_container_width=True):
            send_user_message(prompt)
            st.rerun()


if not st.session_state.messages:
    st.info("Start with a question or use a quick prompt from the sidebar.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            st.caption(f"Source: {message.get('source', 'unknown')}")


user_input = st.chat_input("Ask GlueBot about your issue...")
if user_input:
    send_user_message(user_input)
    st.rerun()
