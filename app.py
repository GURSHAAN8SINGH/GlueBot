# -*- coding: utf-8 -*-
import os
import json
import re
import hashlib
from io import BytesIO
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import requests
import streamlit as st
from dotenv import load_dotenv

try:
    from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency
    PdfReader = None

load_dotenv(override=True)

BASE = Path(__file__).parent
TESTER_UPLOAD_DIR = BASE / "tester_uploads"
TESTER_UPLOAD_DIR.mkdir(exist_ok=True)

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()
MAX_UPLOAD_SIZE_MB = 5
SUGGESTED_PROMPTS = [
    "pod stuck in terminating",
    "CrashLoopBackOff next steps",
    "delete all openstack volumes in available state",
    "nova hard reboot command for an instance",
    "openstack command to set a server state as active",
    "openstack command to restart a server",
    "ImagePullBackOff troubleshooting",
    "check nova computes health",
    "vepg issue after deployment",
]

st.set_page_config(page_title="GlueBot", page_icon=":robot_face:", layout="centered")

for key, value in {
    "messages": [],
    "findings_cache": [],
    "buckets_cache": [],
    "memory_selected_topic": "Manual entry",
    "memory_question": "",
    "memory_answer": "",
    "operating_mode": "standard",
    "test_channel": "",
    "test_name": "",
    "test_details": "",
    "tester_document_records": [],
    "tester_uploaded_hashes": [],
    "tester_doc_uploader_nonce": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def safe_button(label: str, key: str = None, use_container_width: bool = False) -> bool:
    try:
        return st.button(label, key=key, use_container_width=use_container_width)
    except TypeError:
        return st.button(label, key=key)


def safe_toggle(label: str, value: bool = False, key: str = None) -> bool:
    if hasattr(st, "toggle"):
        return st.toggle(label, value=value, key=key)
    return st.checkbox(label, value=value, key=key)


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return slug or "document"


def _document_hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _document_meta_path(raw_path: Path) -> Path:
    return raw_path.with_suffix(raw_path.suffix + ".meta.json")


def _load_persisted_tester_documents() -> List[Dict[str, str]]:
    if not TESTER_UPLOAD_DIR.exists():
        return []
    records: List[Dict[str, str]] = []
    for meta_path in sorted(TESTER_UPLOAD_DIR.glob("*.meta.json")):
        try:
            record = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(record, dict) and record.get("name"):
            records.append(
                {
                    "name": str(record.get("name", "uploaded_document")),
                    "content": str(record.get("content", "")),
                    "error": str(record.get("error", "")),
                    "hash": str(record.get("hash", "")),
                    "saved_at": str(record.get("saved_at", "")),
                }
            )
    return records


def _persist_tester_document(uploaded_file) -> Dict[str, str]:
    filename = getattr(uploaded_file, "name", "uploaded_document")
    suffix = Path(filename).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    # File size limit check
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        return {
            "name": filename,
            "content": "",
            "error": f"File too large ({size_mb:.1f}MB). Max allowed: {MAX_UPLOAD_SIZE_MB}MB.",
            "hash": "",
            "saved_at": "",
        }

    file_hash = _document_hash(raw_bytes)
    safe_name = _safe_slug(Path(filename).stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    raw_path = TESTER_UPLOAD_DIR / f"{timestamp}_{safe_name}{suffix or '.bin'}"

    extracted = _read_uploaded_document(uploaded_file)
    record = {
        "name": extracted.get("name", filename),
        "content": extracted.get("content", ""),
        "error": extracted.get("error", ""),
        "hash": file_hash,
        "saved_at": timestamp,
        "raw_file": raw_path.name,
    }
    try:
        raw_path.write_bytes(raw_bytes)
        _document_meta_path(raw_path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError as exc:
        record["error"] = record.get("error", "") or f"Failed to persist upload: {exc}"
    return record


def _clear_persisted_tester_documents() -> None:
    if not TESTER_UPLOAD_DIR.exists():
        return
    for path in TESTER_UPLOAD_DIR.iterdir():
        try:
            if path.is_file():
                path.unlink()
        except OSError:
            continue


def _read_uploaded_document(uploaded_file) -> Dict[str, str]:
    filename = getattr(uploaded_file, "name", "uploaded_document")
    suffix = Path(filename).suffix.lower()
    raw_bytes = uploaded_file.getvalue()

    if suffix == ".pdf":
        if PdfReader is None:
            return {
                "name": filename,
                "content": "",
                "error": "PDF support is unavailable. Install the optional `pypdf` dependency to read PDFs.",
            }
        reader = PdfReader(BytesIO(raw_bytes))
        pages = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        content = "\n".join(pages)
    else:
        content = raw_bytes.decode("utf-8", errors="replace")

    content = " ".join(content.split())
    return {
        "name": filename,
        "content": content[:12000],
        "error": "",
    }


def _format_tester_documents(documents: List[Dict[str, str]]) -> str:
    if not documents:
        return ""
    parts = []
    for doc in documents[:5]:
        name = doc.get("name", "uploaded_document")
        content = doc.get("content", "").strip()
        error = doc.get("error", "").strip()
        if error:
            parts.append(f"File: {name}\nNote: {error}")
            continue
        if not content:
            parts.append(f"File: {name}\nNote: No readable text extracted.")
            continue
        parts.append(f"File: {name}\nContent: {content}")
    joined = "\n\n".join(parts)
    return joined[:16000]


if not st.session_state.tester_document_records:
    st.session_state.tester_document_records = _load_persisted_tester_documents()
    st.session_state.tester_uploaded_hashes = [str(item.get("hash", "")) for item in st.session_state.tester_document_records if str(item.get("hash", ""))]


def _confidence_badge(confidence: float) -> str:
    if confidence >= 0.85:
        return "🟢 High"
    if confidence >= 0.6:
        return "🟡 Medium"
    return "🔴 Low"


def render_chat_message(message: Dict[str, str], idx: int) -> None:
    role = message.get("role", "assistant")
    content = message.get("content", "")
    source = message.get("source", "unknown")
    operating_mode = message.get("operating_mode", "")
    confidence = float(message.get("confidence", 0))
    related = message.get("related_questions", [])
    intent = message.get("intent", "")
    severity = message.get("severity", "")
    detected_patterns = message.get("detected_patterns", [])

    caption = f"Source: {source}"
    if operating_mode:
        caption += f" | Mode: {operating_mode}"
    if confidence > 0:
        caption += f" | {_confidence_badge(confidence)} ({confidence:.0%})"
    if intent and intent != "general":
        caption += f" | Intent: {intent}"
    if severity:
        sev_icons = {"P1": "[P1 CRITICAL]", "P2": "[P2 HIGH]", "P3": "[P3 LOW]"}
        caption += f" | {sev_icons.get(severity, severity)}"
    if detected_patterns:
        caption += f" | Patterns: {', '.join(detected_patterns[:3])}"

    feedback_key = f"fb_{idx}"
    if feedback_key not in st.session_state:
        st.session_state[feedback_key] = None

    if hasattr(st, "chat_message"):
        with st.chat_message(role):
            st.markdown(content)
            if role == "assistant":
                st.caption(caption)
        if role == "assistant":
            feedback_given = st.session_state[feedback_key]
            if feedback_given is None:
                col1, col2, col3 = st.columns([1, 1, 10])
                with col1:
                    if st.button("[+]", key=f"fb_up_{idx}"):
                        _send_feedback(message, helpful=True)
                        st.session_state[feedback_key] = "up"
                        rerun_app()
                with col2:
                    if st.button("[-]", key=f"fb_dn_{idx}"):
                        _send_feedback(message, helpful=False)
                        st.session_state[feedback_key] = "dn"
                        rerun_app()
            elif feedback_given == "up":
                st.caption("Helpful - thanks!")
            else:
                st.caption("Feedback recorded - we will improve this answer")
            if related:
                st.markdown("**Try also:** " + " · ".join(f"`{q}`" for q in related[:3]))
        return
    st.markdown(f"**You:** {content}" if role == "user" else f"**GlueBot:** {content}")
    if role == "assistant":
        st.caption(caption)


def _send_feedback(message: Dict, helpful: bool) -> None:
    try:
        requests.post(
            f"{BACKEND_URL}/feedback",
            json={
                "message": message.get("_user_message", ""),
                "reply": message.get("content", ""),
                "source": message.get("source", ""),
                "helpful": helpful,
                "comment": "",
            },
            timeout=5,
        )
    except requests.RequestException:
        pass


def read_user_input() -> str:
    if hasattr(st, "chat_input"):
        return st.chat_input("Ask GlueBot about your issue...") or ""
    text = st.text_input("Ask GlueBot about your issue...", key="legacy_user_input")
    return text if safe_button("Send", key="legacy_send", use_container_width=True) else ""


def call_chat_api(message: str) -> Dict[str, str]:
    # Build last 5 turns of history for multi-turn context
    history = []
    for msg in st.session_state.messages[-10:]:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content.strip():
            history.append({"role": role, "content": content})
    response = requests.post(
        f"{BACKEND_URL}/chat",
        json={
            "message": message,
            "operating_mode": st.session_state.operating_mode,
            "test_channel": st.session_state.test_channel,
            "test_name": st.session_state.test_name,
            "test_details": st.session_state.test_details,
            "test_documents": _format_tester_documents(st.session_state.tester_document_records),
            "history": history,
        },
        timeout=30,
    )
    if response.status_code == 429:
        return {
            "reply": "Rate limit reached. Please wait a moment before sending more messages.",
            "source": "rate_limited",
            "operating_mode": "standard",
            "confidence": 0,
            "related_questions": [],
        }
    response.raise_for_status()
    data = response.json()
    return {
        "reply": data.get("reply", "No reply received."),
        "source": data.get("source", "unknown"),
        "operating_mode": data.get("operating_mode", "standard"),
        "confidence": data.get("confidence", 0),
        "related_questions": data.get("related_questions", []),
        "intent": data.get("intent", ""),
        "severity": data.get("severity", ""),
        "detected_patterns": data.get("detected_patterns", []),
    }


def fetch_findings() -> List[Dict]:
    response = requests.get(f"{BACKEND_URL}/findings", timeout=10)
    response.raise_for_status()
    return response.json().get("items", [])


def fetch_buckets() -> List[Dict]:
    response = requests.get(f"{BACKEND_URL}/buckets", timeout=10)
    response.raise_for_status()
    return response.json().get("items", [])


def smart_search(query: str) -> List[Dict]:
    """Call /search endpoint for live KB search."""
    response = requests.post(f"{BACKEND_URL}/search", json={"query": query, "limit": 5}, timeout=10)
    response.raise_for_status()
    return response.json().get("results", [])


def export_knowledge() -> List[Dict]:
    response = requests.get(f"{BACKEND_URL}/knowledge/export", timeout=10)
    response.raise_for_status()
    return response.json().get("items", [])


def update_memory(question: str, answer: str, source_topic: str = "") -> Dict:
    response = requests.post(
        f"{BACKEND_URL}/memory/upsert",
        json={
            "question": question,
            "answer": answer,
            "source_topic": source_topic,
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("item", {})


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
                "operating_mode": result["operating_mode"],
                "confidence": result.get("confidence", 0),
                "related_questions": result.get("related_questions", []),
                "intent": result.get("intent", ""),
                "severity": result.get("severity", ""),
                "detected_patterns": result.get("detected_patterns", []),
                "_user_message": msg,
            }
        )
    except requests.RequestException as exc:
        st.session_state.messages.append({"role": "assistant", "content": f"Chat request failed: {exc}", "source": "error"})


st.title("GlueBot")
st.caption("Team Glue operations assistant")

with st.sidebar:
    st.subheader("Session")
    st.session_state.operating_mode = safe_toggle(
        "Tester mode",
        value=st.session_state.operating_mode == "tester",
        key="tester_mode_toggle",
    ) and "tester" or "standard"
    if st.session_state.operating_mode == "tester":
        st.session_state.test_channel = st.text_input("Test channel", value=st.session_state.test_channel)
        st.session_state.test_name = st.text_input("Test name", value=st.session_state.test_name)
        st.session_state.test_details = st.text_area("Test details", value=st.session_state.test_details, height=100)
        uploader_key = f"tester_doc_uploader_{st.session_state.tester_doc_uploader_nonce}"
        uploaded_docs = st.file_uploader(
            f"Upload test logs (max {MAX_UPLOAD_SIZE_MB}MB each)",
            type=["txt", "log", "md", "csv", "json", "yaml", "yml", "xml", "pdf"],
            accept_multiple_files=True,
            key=uploader_key,
        )
        if uploaded_docs:
            known_hashes = set(st.session_state.tester_uploaded_hashes)
            new_records = []
            for doc in uploaded_docs:
                file_hash = _document_hash(doc.getvalue())
                if file_hash in known_hashes:
                    continue
                record = _persist_tester_document(doc)
                if record.get("error"):
                    st.warning(f"{record.get('name', 'file')}: {record['error']}")
                if record.get("hash"):
                    known_hashes.add(record["hash"])
                new_records.append(record)
            if new_records:
                st.session_state.tester_document_records = _load_persisted_tester_documents()
                st.session_state.tester_uploaded_hashes = [str(item.get("hash", "")) for item in st.session_state.tester_document_records if str(item.get("hash", ""))]
        if st.session_state.tester_document_records:
            st.caption(f"{len(st.session_state.tester_document_records)} uploaded document(s) attached to tester context")
            for doc in st.session_state.tester_document_records[:3]:
                label = doc.get("name", "uploaded_document")
                if doc.get("error"):
                    st.warning(f"{label}: {doc['error']}")
                else:
                    st.caption(f"{label}: {len(doc.get('content', ''))} characters indexed")
            if safe_button("Clear uploaded docs", key="clear_tester_docs", use_container_width=True):
                _clear_persisted_tester_documents()
                st.session_state.tester_document_records = []
                st.session_state.tester_uploaded_hashes = []
                st.session_state.tester_doc_uploader_nonce += 1
                rerun_app()
    else:
        st.session_state.test_channel = ""
        st.session_state.test_name = ""
        st.session_state.test_details = ""

    st.subheader("System")
    st.caption(f"LLM configured: {'Yes' if (OPENAI_API_KEY or OPENROUTER_API_KEY) else 'No'}")
    st.caption(f"Mode: {st.session_state.operating_mode}")
    if safe_button("Check API health", use_container_width=True):
        try:
            requests.get(f"{BACKEND_URL}/", timeout=5).raise_for_status()
            st.success("API is reachable")
        except requests.RequestException as exc:
            st.error(f"API unreachable: {exc}")
    if safe_button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        rerun_app()

    st.subheader("Knowledge base")
    if safe_button("Export knowledge.json", key="export_kb", use_container_width=True):
        try:
            kb_data = export_knowledge()
            kb_json = json.dumps(kb_data, indent=2)
            st.download_button(
                label="Download",
                data=kb_json,
                file_name="knowledge.json",
                mime="application/json",
                key="download_kb",
            )
        except requests.RequestException as exc:
            st.error(f"Export failed: {exc}")

    st.subheader("Team Glue findings")
    if safe_button("Refresh findings", use_container_width=True):
        try:
            st.session_state.findings_cache = fetch_findings()
            st.session_state.buckets_cache = fetch_buckets()
            if not st.session_state.findings_cache:
                st.info("No findings recorded yet.")
            for finding in st.session_state.findings_cache[:5]:
                st.caption(f"{str(finding.get('status', 'observed')).upper()} | Seen {finding.get('count', 0)} times")
                st.write(str(finding.get("topic", "Unknown trend")))
        except requests.RequestException as exc:
            st.error(f"Findings unavailable: {exc}")

    st.subheader("Issue buckets")
    if st.session_state.buckets_cache:
        for bucket in st.session_state.buckets_cache[:5]:
            st.caption(
                "{0} | {1} | Seen {2}".format(
                    str(bucket.get("domain", "general")).upper(),
                    str(bucket.get("symptom", "generic-issue")),
                    bucket.get("count", 0),
                )
            )
            st.write(", ".join(bucket.get("topics", [])[:2]))
    else:
        st.caption("Refresh findings to view grouped issue buckets.")

    st.subheader("Update bot memory")
    findings_options = ["Manual entry"] + [str(item.get("topic", "")) for item in st.session_state.findings_cache if str(item.get("topic", "")).strip()]
    selected_topic = st.selectbox("Source finding", findings_options, key="memory_source")
    if selected_topic != st.session_state.memory_selected_topic:
        st.session_state.memory_selected_topic = selected_topic
        st.session_state.memory_question = "" if selected_topic == "Manual entry" else selected_topic
    memory_question = st.text_input("Question to remember", key="memory_question")
    memory_answer = st.text_area("Answer to save", key="memory_answer", height=140)
    if safe_button("Save to bot memory", key="save_memory", use_container_width=True):
        try:
            item = update_memory(memory_question, memory_answer, "" if selected_topic == "Manual entry" else selected_topic)
            st.success(f"Memory {item.get('action', 'updated')}: {item.get('question', memory_question)}")
            try:
                st.session_state.findings_cache = fetch_findings()
                st.session_state.buckets_cache = fetch_buckets()
            except requests.RequestException:
                pass
        except requests.RequestException as exc:
            st.error(f"Memory update failed: {exc}")

    st.subheader("🔍 Smart Search")
    search_query = st.text_input("Search knowledge base", key="smart_search_input", placeholder="Type to search...")
    if search_query and len(search_query) >= 2:
        try:
            results = smart_search(search_query)
            if results:
                for r in results[:5]:
                    score_pct = int(r.get("score", 0) * 100)
                    st.markdown(f"**{r['question']}** ({score_pct}% match)")
                    st.caption(r["answer"][:150] + ("..." if len(r["answer"]) > 150 else ""))
                    st.markdown("---")
            else:
                st.caption("No matches found.")
        except requests.RequestException:
            st.caption("Search unavailable.")

    st.subheader("Quick prompts")
    for p_idx, prompt in enumerate(SUGGESTED_PROMPTS):
        if safe_button(prompt, key=f"prompt_{p_idx}", use_container_width=True):
            send_user_message(prompt)
            rerun_app()


if not st.session_state.messages:
    st.info("Start with a question, switch to tester mode, or use a quick prompt from the sidebar.")

for msg_idx, message in enumerate(st.session_state.messages):
    render_chat_message(message, msg_idx)

user_input = read_user_input()
if user_input:
    send_user_message(user_input)
    rerun_app()
