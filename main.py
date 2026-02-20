import json
import os
import re
from pathlib import Path

import requests
from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel

app = FastAPI(title="gluebot1")
KNOWLEDGE_PATH = Path(__file__).with_name("knowledge.json")
DOTENV_PATH = Path(__file__).with_name(".env")
load_dotenv(dotenv_path=DOTENV_PATH, override=True)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str
    source: str


def _smalltalk_reply(message: str) -> str | None:
    text = message.strip().lower()
    if not text:
        return "Please type a message so I can help."

    greetings = {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}
    thanks = {"thanks", "thank you", "thx"}

    if text in greetings:
        return (
            "Hi, I can help with Kubernetes and OpenStack operations. "
            "Try: `pod stuck in termination`, `CrashLoopBackOff`, or "
            "`delete all available openstack volumes`."
        )
    if text in thanks:
        return "You are welcome. Share the next issue when ready."
    if text in {"help", "what can you do", "what can you help with"}:
        return (
            "I can troubleshoot Kubernetes and OpenStack issues, and generate safe scripts. "
            "Include exact errors and I will provide step-by-step commands."
        )
    return None


def _is_incident_like(message: str) -> bool:
    tokens = set(re.findall(r"[a-z0-9]+", message.lower()))
    incident_keywords = {
        "pod",
        "kubernetes",
        "k8s",
        "probe",
        "liveness",
        "readiness",
        "crashloopbackoff",
        "imagepullbackoff",
        "error",
        "failed",
        "timeout",
        "terminating",
        "restart",
        "openstack",
        "nova",
        "neutron",
        "cinder",
        "glance",
        "keystone",
        "volume",
        "instance",
        "server",
    }
    return bool(tokens & incident_keywords)


def _openstack_script_reply(message: str) -> str | None:
    text = message.lower()
    if (
        "openstack" in text
        and "volume" in text
        and ("delete" in text or "remove" in text)
        and "available" in text
    ):
        return (
            "Use this safe script (dry-run by default) to delete all OpenStack volumes in `available` state:\n\n"
            "```bash\n"
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n\n"
            "# Usage:\n"
            "#   ./delete_available_volumes.sh            # dry-run\n"
            "#   DRY_RUN=false ./delete_available_volumes.sh\n\n"
            "DRY_RUN=${DRY_RUN:-true}\n\n"
            "mapfile -t volumes < <(openstack volume list -f value -c ID -c Status | awk '$2==\"available\" {print $1}')\n\n"
            "if [ ${#volumes[@]} -eq 0 ]; then\n"
            "  echo \"No available volumes found.\"\n"
            "  exit 0\n"
            "fi\n\n"
            "echo \"Found ${#volumes[@]} available volumes\"\n"
            "for vol in \"${volumes[@]}\"; do\n"
            "  if [ \"$DRY_RUN\" = \"true\" ]; then\n"
            "    echo \"[DRY-RUN] openstack volume delete $vol\"\n"
            "  else\n"
            "    echo \"Deleting volume: $vol\"\n"
            "    openstack volume delete \"$vol\"\n"
            "  fi\n"
            "done\n"
            "```\n\n"
            "Before deleting, confirm tenant/project scope and snapshots/backups."
        )
    return None


def _load_knowledge() -> list[dict]:
    if not KNOWLEDGE_PATH.exists() or KNOWLEDGE_PATH.stat().st_size == 0:
        return []
    try:
        data = json.loads(KNOWLEDGE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _find_reply(message: str, knowledge: list[dict]) -> str | None:
    text = message.strip().lower()
    if not text:
        return "Please type a message so I can help."

    msg_tokens = set(re.findall(r"[a-z0-9]+", text))
    for item in knowledge:
        question = str(item.get("question", item.get("q", ""))).strip().lower()
        answer = str(item.get("answer", item.get("a", ""))).strip()
        if not (question and answer):
            continue

        if question in text or text in question:
            return answer

        q_tokens = set(re.findall(r"[a-z0-9]+", question))
        if not q_tokens:
            continue
        overlap = len(msg_tokens & q_tokens) / len(q_tokens)
        if overlap >= 0.6:
            return answer

    return None


def _save_knowledge(knowledge: list[dict]) -> None:
    KNOWLEDGE_PATH.write_text(json.dumps(knowledge, indent=2), encoding="utf-8")


def _track_unknown_issue(message: str, knowledge: list[dict]) -> None:
    question = message.strip()
    if not question:
        return

    existing_questions = {
        str(item.get("question", item.get("q", ""))).strip().lower()
        for item in knowledge
        if isinstance(item, dict)
    }
    if question.lower() in existing_questions:
        return

    knowledge.append(
        {
            "question": question,
            "answer": "",
            "status": "unresolved",
            "note": "Captured from unknown user issue. Fill answer later.",
        }
    )
    _save_knowledge(knowledge)


def _llm_reply(message: str, knowledge: list[dict]) -> str | None:
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    llm_api_key = openai_api_key or openrouter_api_key
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()
    llm_api_base = os.getenv("LLM_API_BASE", "").strip()
    openrouter_site_url = os.getenv("OPENROUTER_SITE_URL", "").strip()
    openrouter_app_name = os.getenv("OPENROUTER_APP_NAME", "GlueBot").strip()

    if not llm_api_key:
        return None

    topic_list = []
    for item in knowledge[:30]:
        question = str(item.get("question", item.get("q", ""))).strip()
        answer = str(item.get("answer", item.get("a", ""))).strip()
        if question and answer:
            topic_list.append(f"- {question}: {answer}")
    kb_context = "\n".join(topic_list) if topic_list else "- No curated KB answers yet."

    prompt = (
        "You are GlueBot, a Kubernetes + OpenStack SRE assistant. "
        "Provide concise, actionable troubleshooting steps. "
        "For Kubernetes, include kubectl commands. For OpenStack, include openstack CLI commands. "
        "If user asks for scripts/automation, return a safe script with dry-run default and a short warning.\n\n"
        f"Known KB:\n{kb_context}\n\n"
        f"User issue: {message}"
    )

    base = llm_api_base.lower()
    is_openrouter = "openrouter" in base or bool(openrouter_api_key)

    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "Content-Type": "application/json",
    }
    if is_openrouter:
        if openrouter_site_url:
            headers["HTTP-Referer"] = openrouter_site_url
        if openrouter_app_name:
            headers["X-Title"] = openrouter_app_name

    try:
        if is_openrouter:
            url = llm_api_base or "https://openrouter.ai/api/v1/chat/completions"
            response = requests.post(
                url,
                headers=headers,
                json={
                    "model": llm_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are GlueBot, a Kubernetes + OpenStack SRE assistant. "
                                "Give concise troubleshooting steps and safe automation scripts."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=20,
            )
            response.raise_for_status()
            data = response.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                or None
            )

        url = llm_api_base or "https://api.openai.com/v1/responses"
        response = requests.post(
            url,
            headers=headers,
            json={
                "model": llm_model,
                "input": prompt,
            },
            timeout=20,
        )
        response.raise_for_status()
        data = response.json()
        text = (data.get("output_text") or "").strip()
        return text or None
    except requests.RequestException:
        return None


def _related_topics(message: str, knowledge: list[dict], limit: int = 3) -> list[str]:
    msg_tokens = set(re.findall(r"[a-z0-9]+", message.lower()))
    scored: list[tuple[float, str]] = []
    for item in knowledge:
        question = str(item.get("question", item.get("q", ""))).strip()
        answer = str(item.get("answer", item.get("a", ""))).strip()
        if not question:
            continue
        if not answer:
            continue
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower()))
        if not q_tokens:
            continue
        score = len(msg_tokens & q_tokens) / len(q_tokens)
        if score > 0:
            scored.append((score, question))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [topic for _, topic in scored[:limit]]


def _fallback_reply(message: str, knowledge: list[dict]) -> str:
    text = message.lower()
    related = _related_topics(message, knowledge)
    related_text = ""
    if related:
        related_text = "\nRelated topics you can ask: " + ", ".join(related)

    if "liveness" in text and "500" in text:
        return (
            "Try this checklist for liveness probe 500 errors:\n"
            "1) Confirm the probe path/port matches your app endpoint.\n"
            "2) Check app logs around probe failures (`kubectl logs <pod> -c <container>`).\n"
            "3) Increase `initialDelaySeconds` and `timeoutSeconds` if startup is slow.\n"
            "4) Verify dependencies (DB/cache/API) required by the health endpoint.\n"
            "5) Use `kubectl describe pod <pod>` to confirm probe failure events."
            f"{related_text}"
        )

    if "readiness" in text or "liveness" in text or "probe" in text:
        return (
            "Probe issue detected. Verify probe path, port, and timing fields "
            "(`initialDelaySeconds`, `timeoutSeconds`, `periodSeconds`, `failureThreshold`), "
            "then inspect events with `kubectl describe pod <pod>` and container logs."
            f"{related_text}"
        )

    if "openstack" in text or "nova" in text or "neutron" in text or "cinder" in text:
        return (
            "OpenStack issue detected. Start with: "
            "`openstack token issue`, `openstack server list`, `openstack volume list`, "
            "`openstack network agent list`, then check service logs and project quotas."
            f"{related_text}"
        )

    return (
        "I do not have an exact match yet, but start with: "
        "`kubectl describe pod <pod>`, `kubectl logs <pod> -c <container> --previous`, "
        "and check recent events in the namespace."
        f"{related_text}"
    )


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    llm_api_key = openai_api_key or openrouter_api_key
    llm_model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip()

    knowledge = _load_knowledge()

    smalltalk = _smalltalk_reply(payload.message)
    if smalltalk:
        return ChatResponse(reply=smalltalk, source="intent")

    matched = _find_reply(payload.message, knowledge)
    if matched:
        return ChatResponse(reply=matched, source="knowledge.json")

    scripted = _openstack_script_reply(payload.message)
    if scripted:
        return ChatResponse(reply=scripted, source="template:openstack_script")

    if _is_incident_like(payload.message):
        _track_unknown_issue(payload.message, knowledge)

    llm = _llm_reply(payload.message, knowledge)
    if llm:
        return ChatResponse(reply=llm, source=f"llm:{llm_model}")

    if not llm_api_key:
        return ChatResponse(
            reply=_fallback_reply(payload.message, knowledge),
            source="fallback:no_llm_api_key",
        )

    return ChatResponse(reply=_fallback_reply(payload.message, knowledge), source="fallback:llm_unavailable")
