import json
import logging
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

# --- Logging ---
LOG_PATH = Path(__file__).parent / "gluebot.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("gluebot")

app = FastAPI(title="gluebot1")
BASE = Path(__file__).parent
KNOWLEDGE_PATH = BASE / "knowledge.json"
UNRESOLVED_PATH = BASE / "unresolved_issues.json"
FINDINGS_PATH = BASE / "findings.json"
DOTENV_PATH = BASE / ".env"
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
TREND_THRESHOLD = 2
AUTO_PROMOTE_THRESHOLD = 3
SEMANTIC_THRESHOLD = 0.45
FILE_LOCK = threading.Lock()

# --- Rate Limiting ---
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 20
_request_timestamps = deque()
_rate_lock = threading.Lock()

# --- Intent Classification ---
INTENT_PATTERNS = {
    "give_command": re.compile(r"\b(command|cli|one.?liner|how to run|give me|show me)\b", re.IGNORECASE),
    "troubleshoot": re.compile(r"\b(issue|error|fail|stuck|crash|not working|broken|down|timeout|debug|troubleshoot|fix|outage)\b", re.IGNORECASE),
    "explain_concept": re.compile(r"\b(what is|what are|explain|meaning of|define|difference between|why does|how does)\b", re.IGNORECASE),
    "status_check": re.compile(r"\b(check|status|health|verify|inspect|monitor|list|show)\b", re.IGNORECASE),
    "how_to": re.compile(r"\b(how to|how do i|how can i|steps to|guide|procedure)\b", re.IGNORECASE),
    "script_request": re.compile(r"\b(script|bash|shell|automate|loop|for each|bulk)\b", re.IGNORECASE),
}

# --- Conversational Fillers (no LLM needed) ---
FILLER_RESPONSES = {
    "okay": "Got it. What's the next issue?",
    "ok": "Got it. What's the next issue?",
    "sure": "Ready when you are. What do you need help with?",
    "yes": "Sure, which pattern do you want to dig into — oom_killed, connection_refused, or crashloop?",
    "no": "No problem. Let me know if something else comes up.",
    "cool": "👍 What else can I help with?",
    "got it": "Great. Next question whenever you're ready.",
    "alright": "Standing by. What's next?",
    "fine": "Good. Share your next issue when ready.",
    "understood": "Perfect. What else do you need?",
    "i see": "Let me know if you need more details or have another question.",
    "never mind": "No worries. I'm here when you need me.",
    "nvm": "No worries. I'm here when you need me.",
}

# --- Severity Classification ---
SEVERITY_KEYWORDS = {
    "P1": {"production down", "all pods", "cluster down", "complete outage", "service unavailable", "critical", "p1", "sev1", "severity 1", "emergency"},
    "P2": {"multiple pods", "degraded", "intermittent", "partial outage", "high impact", "p2", "sev2", "severity 2"},
    "P3": {"single pod", "one instance", "minor", "low impact", "non-critical", "p3", "sev3", "severity 3", "cosmetic"},
}

# --- Log Pattern Detection ---
LOG_PATTERNS = {
    "oom_killed": re.compile(r"(OOMKilled|out of memory|memory limit|cannot allocate)", re.IGNORECASE),
    "connection_refused": re.compile(r"(connection refused|ECONNREFUSED|dial tcp.*refused)", re.IGNORECASE),
    "timeout": re.compile(r"(timed? ?out|deadline exceeded|context deadline|request timeout)", re.IGNORECASE),
    "permission_denied": re.compile(r"(permission denied|forbidden|403|RBAC|unauthorized)", re.IGNORECASE),
    "image_pull": re.compile(r"(ImagePullBackOff|ErrImagePull|pull access denied|manifest unknown)", re.IGNORECASE),
    "crashloop": re.compile(r"(CrashLoopBackOff|back-off restarting|restart count)", re.IGNORECASE),
    "disk_pressure": re.compile(r"(DiskPressure|no space left|disk full|volume full)", re.IGNORECASE),
    "dns_failure": re.compile(r"(could not resolve|DNS|NXDOMAIN|name resolution)", re.IGNORECASE),
    "certificate_error": re.compile(r"(certificate|x509|TLS|SSL|cert expired|untrusted)", re.IGNORECASE),
    "resource_quota": re.compile(r"(quota|exceeded|resource limit|LimitRange|insufficient)", re.IGNORECASE),
    "node_not_ready": re.compile(r"(NotReady|node.*not ready|SchedulingDisabled)", re.IGNORECASE),
    "prometheus_error": re.compile(r"(PrometheusException|unknown host.*prometheus|metrics.*unavailable)", re.IGNORECASE),
    "guice_injection": re.compile(r"(ProvisionException|Guice|Unable to provision|ErrorInCustomProvider)", re.IGNORECASE),
    "null_response": re.compile(r"(response is null|NullPointerException|null reference|exit status code)", re.IGNORECASE),
}

LOG_PATTERN_ADVICE = {
    "oom_killed": "Memory issue detected. Check container memory limits with `kubectl describe pod`. Consider increasing `resources.limits.memory` or investigating memory leaks.",
    "connection_refused": "Connection refused. Check if the target service is running, port is correct, NetworkPolicies, and service endpoints: `kubectl get ep <service>`.",
    "timeout": "Timeout detected. Check network connectivity, DNS resolution, target service health, and consider increasing timeout values if the service is slow to start.",
    "permission_denied": "Permission/RBAC issue. Check ServiceAccount, Role/ClusterRole bindings: `kubectl auth can-i --list --as=system:serviceaccount:<ns>:<sa>`.",
    "image_pull": "Image pull failure. Verify image name/tag exists, registry credentials (imagePullSecrets), and network access to the registry.",
    "crashloop": "CrashLoop detected. Check logs: `kubectl logs <pod> --previous`, describe pod for exit codes, verify probes, env vars, and mounted configs.",
    "disk_pressure": "Disk pressure. Check node disk usage, clean up unused images (`crictl rmi --prune`), PVCs, and consider expanding volumes.",
    "dns_failure": "DNS resolution failure. Check CoreDNS pods, DNS policy in pod spec, and network connectivity: `kubectl exec <pod> -- nslookup <service>`.",
    "certificate_error": "Certificate/TLS error. Check cert expiry, CA bundle, cert-manager status, and ensure the correct secret is mounted.",
    "resource_quota": "Resource quota exceeded. Check quotas: `kubectl describe quota -n <ns>`. Either request more quota or reduce resource requests on pods.",
    "node_not_ready": "Node not ready. Check kubelet status on the node, node conditions: `kubectl describe node <node>`, and look for disk/memory/PID pressure.",
    "prometheus_error": "Prometheus connectivity issue. Verify Prometheus hostname/DNS is reachable from the pod, check service endpoints, and confirm Prometheus is running.",
    "guice_injection": "Dependency injection failure (Guice). A required service dependency could not be provisioned. Check if the dependent service (e.g., Prometheus, DB) is reachable and properly configured.",
    "null_response": "Null/empty response from remote command execution. The target host may be unresponsive, SSH session may have timed out, or the service on the host is not running.",
}

# --- Intent-Specific Clarifying Questions ---
INTENT_QUESTIONS = {
    "give_command": [
        "What platform are you targeting? (Kubernetes, OpenStack, Linux, etc.)",
        "Any specific version or environment constraints?",
    ],
    "troubleshoot": [
        "What error message or symptom are you seeing?",
        "When did this start and has anything changed recently?",
    ],
    "explain_concept": [
        "What's your current understanding so I can pitch the explanation right?",
    ],
    "status_check": [
        "Which namespace or environment should I focus on?",
        "What component are you checking — pod, node, volume, service?",
    ],
    "how_to": [
        "What's your target environment? (Kubernetes, OpenStack, bare metal)",
        "Any prerequisites already in place?",
    ],
    "script_request": [
        "What should the script do — what's the input and expected output?",
        "Should it have a dry-run mode?",
    ],
}


def _get_intent_question(intent: str, message: str) -> Optional[str]:
    """Return a clarifying question if the message is too vague for the detected intent."""
    questions = INTENT_QUESTIONS.get(intent, [])
    if not questions:
        return None
    text = message.strip().lower()
    # Never ask clarifying questions for urgent/P1 situations
    if _classify_severity(text) == "P1":
        return None
    # Only ask if message is short and lacks enough detail
    if len(text.split()) > 10 or len(text) > 80:
        return None
    return questions[0]


# --- Multi-Step Troubleshooting Flows ---
TROUBLESHOOTING_FLOWS = {
    "pod_debug": {
        "trigger": re.compile(r"\b(pod|container).*(fail|error|crash|not working|issue|problem|debug)\b", re.IGNORECASE),
        "questions": [
            "What namespace is the pod in?",
            "What's the pod name (or deployment name)?",
            "When did the issue start (approximate time)?",
        ],
        "summary_prompt": "Based on the context: namespace={0}, pod/deployment={1}, timeframe={2}. Provide targeted debugging commands and likely root cause.",
    },
    "openstack_instance": {
        "trigger": re.compile(r"\b(instance|server|vm).*(stuck|error|fail|not working|issue|down)\b", re.IGNORECASE),
        "questions": [
            "What's the instance name or ID?",
            "What state is it currently in (ERROR, SHUTOFF, etc.)?",
            "Is this affecting a single instance or multiple?",
        ],
        "summary_prompt": "OpenStack instance issue: name/ID={0}, state={1}, scope={2}. Provide targeted recovery commands and root cause investigation steps.",
    },
}

# --- Synonym expansion ---
SYNONYMS = {
    "reboot": ["restart", "reset", "power cycle"],
    "restart": ["reboot", "reset"],
    "delete": ["remove", "destroy", "purge"],
    "instance": ["server", "vm", "virtual machine"],
    "server": ["instance", "vm", "virtual machine"],
    "pod": ["container", "workload"],
    "stuck": ["hanging", "frozen", "unresponsive"],
    "error": ["failure", "failed", "fault", "issue"],
    "volume": ["disk", "storage", "block storage"],
    "command": ["cli", "one-liner"],
    "check": ["verify", "inspect", "show", "list"],
    "start": ["boot", "power on", "launch"],
    "stop": ["shut off", "power off", "halt"],
    "terminate": ["kill", "force delete", "remove"],
}

# TF-IDF vectorizer
_tfidf_vectorizer: Optional[TfidfVectorizer] = None
_tfidf_matrix = None


# === Utility Functions ===

def _expand_with_synonyms(text: str) -> str:
    tokens = TOKEN_PATTERN.findall(text.lower())
    expanded = list(tokens)
    for token in tokens:
        if token in SYNONYMS:
            expanded.extend(SYNONYMS[token])
    return " ".join(expanded)


def _check_rate_limit() -> bool:
    with _rate_lock:
        now = time.time()
        while _request_timestamps and now - _request_timestamps[0] > RATE_LIMIT_WINDOW:
            _request_timestamps.popleft()
        if len(_request_timestamps) >= RATE_LIMIT_MAX:
            return False
        _request_timestamps.append(now)
        return True


def _classify_intent(message: str) -> str:
    """Classify user intent from message text."""
    for intent, pattern in INTENT_PATTERNS.items():
        if pattern.search(message):
            return intent
    return "general"


def _classify_severity(message: str) -> str:
    """Classify issue severity from message keywords."""
    text = message.lower()
    for severity, keywords in SEVERITY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return severity
    return "P3"


def _detect_log_patterns(message: str) -> List[Tuple[str, str]]:
    """Detect known error patterns in pasted logs/text."""
    detected = []
    for pattern_name, regex in LOG_PATTERNS.items():
        if regex.search(message):
            advice = LOG_PATTERN_ADVICE.get(pattern_name, "")
            detected.append((pattern_name, advice))
    return detected


def _is_filler(message: str) -> Optional[str]:
    """Check if message is conversational filler."""
    text = message.strip().lower().rstrip("!.?")
    return FILLER_RESPONSES.get(text)


def _get_intent_system_modifier(intent: str) -> str:
    """Return LLM system prompt modifier based on intent."""
    modifiers = {
        "give_command": "Respond with the exact command(s) needed. Be concise. Show the command first, then a brief explanation.",
        "troubleshoot": "Provide a step-by-step troubleshooting approach. Start with the most likely cause and include diagnostic commands.",
        "explain_concept": "Explain clearly and concisely. Use analogies if helpful. Keep it practical and relevant to ops work.",
        "status_check": "Provide the relevant commands to check status/health. Show expected vs problematic output.",
        "how_to": "Provide a clear step-by-step guide. Number the steps. Include prerequisites if any.",
        "script_request": "Provide a complete, working script with comments. Include a dry-run option where applicable.",
    }
    return modifiers.get(intent, "")


# === Data Classes ===

@dataclass
class Settings:
    openai_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model: str = "gpt-4.1-mini"
    llm_api_base: str = ""
    openrouter_site_url: str = ""
    openrouter_app_name: str = "GlueBot"

    @property
    def llm_api_key(self) -> str:
        return self.openai_api_key or self.openrouter_api_key

    @property
    def is_openrouter(self) -> bool:
        return "openrouter" in self.llm_api_base.lower() or bool(self.openrouter_api_key)


@dataclass
class KnowledgeEntry:
    question: str
    answer: str
    normalized_question: str
    tokens: Set[str]
    tfidf_index: int = -1


@dataclass
class BotState:
    settings: Settings = field(default_factory=Settings)
    settings_mtime: Optional[float] = None
    curated_entries: List[KnowledgeEntry] = field(default_factory=list)
    curated_mtime: Optional[float] = None
    unresolved_questions: Set[str] = field(default_factory=set)
    unresolved_mtime: Optional[float] = None
    findings: List[Dict] = field(default_factory=list)
    findings_mtime: Optional[float] = None
    buckets: List[Dict] = field(default_factory=list)


STATE = BotState()


# === Pydantic Models ===

class ChatRequest(BaseModel):
    message: str
    operating_mode: str = "standard"
    test_channel: str = ""
    test_name: str = ""
    test_details: str = ""
    test_documents: str = ""
    history: List[Dict[str, str]] = []


class ChatResponse(BaseModel):
    reply: str
    source: str
    operating_mode: str
    confidence: float = 0.0
    related_questions: List[str] = []
    intent: str = "general"
    severity: str = ""
    detected_patterns: List[str] = []


class FeedbackRequest(BaseModel):
    message: str
    reply: str
    source: str
    helpful: bool
    comment: str = ""


class FindingRecordRequest(BaseModel):
    topic: str
    details: str = ""


class MemoryUpdateRequest(BaseModel):
    question: str
    answer: str
    source_topic: str = ""


class SearchRequest(BaseModel):
    query: str
    limit: int = 5


# === Core Helpers ===

def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _file_mtime(path: Path) -> Optional[float]:
    return path.stat().st_mtime if path.exists() else None


def _normalize_text(value: str) -> str:
    return " ".join(TOKEN_PATTERN.findall(value.lower()))


def _tokenize(value: str) -> Set[str]:
    return set(TOKEN_PATTERN.findall(value.lower()))


def _load_json_list(path: Path) -> List[Dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def _save_json_list(path: Path, items: List[Dict]) -> None:
    with FILE_LOCK:
        path.write_text(json.dumps(items, indent=2), encoding="utf-8")


def _load_settings(force: bool = False) -> Settings:
    mtime = _file_mtime(DOTENV_PATH)
    if not force and STATE.settings_mtime == mtime:
        return STATE.settings
    load_dotenv(dotenv_path=DOTENV_PATH, override=True)
    STATE.settings = Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        llm_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini").strip(),
        llm_api_base=os.getenv("LLM_API_BASE", "").strip(),
        openrouter_site_url=os.getenv("OPENROUTER_SITE_URL", "").strip(),
        openrouter_app_name=os.getenv("OPENROUTER_APP_NAME", "GlueBot").strip(),
    )
    STATE.settings_mtime = mtime
    return STATE.settings


def _merge_unresolved_items(items: List[Dict]) -> None:
    existing = _load_json_list(UNRESOLVED_PATH)
    lookup = {_normalize_text(str(item.get("question", "")).strip()) for item in existing if str(item.get("question", "")).strip()}
    updated = list(existing)
    for item in items:
        question = str(item.get("question", "")).strip()
        normalized = _normalize_text(question)
        if question and normalized not in lookup:
            updated.append(item)
            lookup.add(normalized)
    _save_json_list(UNRESOLVED_PATH, updated)
    STATE.unresolved_questions = lookup
    STATE.unresolved_mtime = _file_mtime(UNRESOLVED_PATH)


def _rebuild_tfidf(questions: List[str]) -> None:
    global _tfidf_vectorizer, _tfidf_matrix
    if not questions:
        _tfidf_vectorizer = None
        _tfidf_matrix = None
        return
    expanded = [_expand_with_synonyms(q) for q in questions]
    _tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    _tfidf_matrix = _tfidf_vectorizer.fit_transform(expanded)


def _tfidf_similarity(message: str) -> List[float]:
    if _tfidf_vectorizer is None or _tfidf_matrix is None:
        return []
    expanded_message = _expand_with_synonyms(message)
    message_vec = _tfidf_vectorizer.transform([expanded_message])
    scores = sklearn_cosine(message_vec, _tfidf_matrix)[0]
    return scores.tolist()


def _load_curated_knowledge(force: bool = False) -> List[KnowledgeEntry]:
    mtime = _file_mtime(KNOWLEDGE_PATH)
    if not force and STATE.curated_mtime == mtime:
        return STATE.curated_entries
    raw_items = _load_json_list(KNOWLEDGE_PATH)
    curated_rows = []
    migrated = []
    for item in raw_items:
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            curated_rows.append({"question": question, "answer": answer})
        elif question:
            migrated.append({"question": question, "answer": "", "status": "unresolved", "note": "Captured from unknown user issue. Fill answer later."})
    if migrated:
        _merge_unresolved_items(migrated)
    if raw_items != curated_rows:
        _save_json_list(KNOWLEDGE_PATH, curated_rows)
        mtime = _file_mtime(KNOWLEDGE_PATH)
    questions = [item["question"] for item in curated_rows]
    _rebuild_tfidf(questions)
    STATE.curated_entries = [
        KnowledgeEntry(
            question=item["question"],
            answer=item["answer"],
            normalized_question=_normalize_text(item["question"]),
            tokens=_tokenize(item["question"]),
            tfidf_index=i,
        )
        for i, item in enumerate(curated_rows)
    ]
    STATE.curated_mtime = mtime
    return STATE.curated_entries


def _load_unresolved_questions(force: bool = False) -> Set[str]:
    mtime = _file_mtime(UNRESOLVED_PATH)
    if not force and STATE.unresolved_mtime == mtime:
        return STATE.unresolved_questions
    items = _load_json_list(UNRESOLVED_PATH)
    STATE.unresolved_questions = {_normalize_text(str(item.get("question", "")).strip()) for item in items if str(item.get("question", "")).strip()}
    STATE.unresolved_mtime = mtime
    return STATE.unresolved_questions


def _load_findings(force: bool = False) -> List[Dict]:
    mtime = _file_mtime(FINDINGS_PATH)
    if not force and STATE.findings_mtime == mtime:
        return STATE.findings
    STATE.findings = _load_json_list(FINDINGS_PATH)
    STATE.findings_mtime = mtime
    return STATE.findings


def _save_findings(findings: List[Dict]) -> None:
    _save_json_list(FINDINGS_PATH, findings)
    STATE.findings = findings
    STATE.findings_mtime = _file_mtime(FINDINGS_PATH)


def _save_curated_knowledge(rows: List[Dict]) -> None:
    _save_json_list(KNOWLEDGE_PATH, rows)
    STATE.curated_mtime = _file_mtime(KNOWLEDGE_PATH)
    _load_curated_knowledge(force=True)


def _remove_unresolved_question(question: str) -> None:
    normalized = _normalize_text(question)
    if not normalized:
        return
    existing = _load_json_list(UNRESOLVED_PATH)
    filtered = [item for item in existing if _normalize_text(str(item.get("question", "")).strip()) != normalized]
    if len(filtered) != len(existing):
        _save_json_list(UNRESOLVED_PATH, filtered)
        STATE.unresolved_questions = {_normalize_text(str(item.get("question", "")).strip()) for item in filtered if str(item.get("question", "")).strip()}
        STATE.unresolved_mtime = _file_mtime(UNRESOLVED_PATH)


def _refresh_state() -> None:
    _load_settings()
    _load_curated_knowledge()
    _load_unresolved_questions()
    _load_findings()
    STATE.buckets = _build_buckets(STATE.findings)


# === Domain / Symptom Classification ===

def _prefers_command_response(message: str) -> bool:
    text = message.lower()
    return any(p in text for p in {"command", "one liner", "one-liner", "for loop", "single command", "cli only"}) and not any(p in text for p in {"script", "bash", "shell script"})


def _smalltalk_reply(message: str) -> Optional[str]:
    text = message.strip().lower()
    if not text:
        return "Please type a message so I can help."
    if text in {"hi", "hello", "hey", "yo", "good morning", "good afternoon", "good evening"}:
        return "Hi, I can help with Kubernetes and OpenStack issues, plus Team Glue topics like VEPG, EPG, and SSR."
    if text in {"thanks", "thank you", "thx"}:
        return "You are welcome. Share the next issue when ready."
    if text in {"help", "what can you do", "what can you help with"}:
        return "I can troubleshoot platform issues, suggest pod fixes, assist in tester mode, detect log patterns, and document findings."
    return None


def _is_incident_like(message: str) -> bool:
    return bool(_tokenize(message) & {"pod", "kubernetes", "k8s", "probe", "liveness", "readiness", "crashloopbackoff", "imagepullbackoff", "error", "failed", "timeout", "terminating", "restart", "openstack", "nova", "neutron", "cinder", "glance", "keystone", "volume", "instance", "server", "glue", "vepg", "epg", "ssr", "test", "channel"})


def _classify_domain(text: str) -> str:
    normalized = text.lower()
    if any(t in normalized for t in {"vepg", "epg", "ssr"}):
        return "openstack-domain"
    if any(t in normalized for t in {"openstack", "nova", "neutron", "cinder", "glance", "keystone", "instance", "volume", "server"}):
        return "openstack"
    if any(t in normalized for t in {"kubernetes", "k8s", "pod", "probe", "namespace", "deployment"}):
        return "kubernetes"
    if any(t in normalized for t in {"test", "channel"}):
        return "testing"
    return "general"


def _classify_symptom(text: str) -> str:
    normalized = text.lower()
    for token, label in {"crashloopbackoff": "crashloopbackoff", "imagepullbackoff": "imagepullbackoff", "terminating": "terminating", "probe": "probe-failure", "timeout": "timeout", "attach": "attach-failure", "quota": "quota", "error": "error", "restart": "restart", "reboot": "reboot", "volume": "volume-operation", "instance": "instance-state"}.items():
        if token in normalized:
            return label
    return "generic-issue"


def _bucket_key(topic: str) -> str:
    return f"{_classify_domain(topic)}::{_classify_symptom(topic)}"


def _build_buckets(findings: List[Dict]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for finding in findings:
        topic = str(finding.get("topic", "")).strip()
        if not topic:
            continue
        key = _bucket_key(topic)
        bucket = grouped.setdefault(key, {"bucket_key": key, "domain": _classify_domain(topic), "symptom": _classify_symptom(topic), "count": 0, "topics": [], "status": "observed", "last_seen": ""})
        bucket["count"] += int(finding.get("count", 1))
        bucket["topics"].append(topic)
        bucket["topics"] = sorted(set(bucket["topics"]))[:5]
        bucket["last_seen"] = max(str(bucket.get("last_seen", "")), str(finding.get("last_seen", "")))
        if str(finding.get("status", "")).lower() in {"trending", "memorized"}:
            bucket["status"] = str(finding.get("status", "")).lower()
    buckets = list(grouped.values())
    buckets.sort(key=lambda item: (int(item.get("count", 0)), str(item.get("last_seen", ""))), reverse=True)
    return buckets


# === Heuristic Replies ===

def _is_pod_error(message: str) -> bool:
    text = message.lower()
    return "pod" in text and any(k in text for k in {"error", "failed", "crashloopbackoff", "imagepullbackoff", "terminating", "restart", "pending", "evicted"})


def _pod_fix_reply(message: str) -> Optional[str]:
    text = message.lower()
    if not _is_pod_error(text):
        return None
    if "crashloopbackoff" in text:
        return "For `CrashLoopBackOff`, check `kubectl describe pod <pod> -n <ns>`, `kubectl logs <pod> -c <container> --previous -n <ns>`, probe timing, env/config changes, and OOM/resource pressure."
    if "imagepullbackoff" in text:
        return "For `ImagePullBackOff`, verify image name/tag, registry access, and imagePullSecrets, then use `kubectl describe pod <pod> -n <ns>` for the exact pull failure."
    return "Pod issue detected. Start with `kubectl describe pod <pod> -n <ns>`, `kubectl logs <pod> -c <container> --previous -n <ns>`, and `kubectl get events -n <ns> --sort-by=.lastTimestamp`."


def _tester_context_summary(payload: ChatRequest) -> str:
    if payload.operating_mode != "tester":
        return ""
    parts = ["Operating mode: tester"]
    if payload.test_channel.strip():
        parts.append(f"Test channel: {payload.test_channel.strip()}")
    if payload.test_name.strip():
        parts.append(f"Test name: {payload.test_name.strip()}")
    if payload.test_details.strip():
        parts.append(f"Test details: {payload.test_details.strip()}")
    return "\n".join(parts)


def _tester_documents_summary(payload: ChatRequest) -> str:
    if payload.operating_mode != "tester":
        return ""
    documents = payload.test_documents.strip()
    if not documents:
        return ""
    return f"Test documents:\n{documents[:16000]}"


def _openstack_script_reply(message: str) -> Optional[str]:
    text = message.lower()
    if "openstack" in text and "volume" in text and ("delete" in text or "remove" in text) and "available" in text:
        if _prefers_command_response(text):
            return "Use this dry-run loop:\n\n```bash\nfor vol in $(openstack volume list -f value -c ID -c Status | awk '$2==\"available\" {print $1}'); do echo openstack volume delete \"$vol\"; done\n```\n\nRemove `echo` to actually delete."
        return "Use this safe script template with dry-run default for deleting `available` volumes."
    return None


# === Matching ===

def _match_score(message: str, message_tokens: Set[str], entry: KnowledgeEntry, tfidf_scores: Optional[List[float]] = None) -> float:
    if not message or not entry.normalized_question:
        return 0.0
    if message == entry.normalized_question:
        return 1.0
    if entry.normalized_question in message or message in entry.normalized_question:
        return 0.95
    tfidf_score = 0.0
    if tfidf_scores and 0 <= entry.tfidf_index < len(tfidf_scores):
        tfidf_score = tfidf_scores[entry.tfidf_index]
    overlap_ratio = len(message_tokens & entry.tokens) / max(len(entry.tokens), 1)
    coverage_ratio = len(message_tokens & entry.tokens) / max(len(message_tokens), 1)
    token_score = overlap_ratio * 0.7 + coverage_ratio * 0.3
    if tfidf_scores is not None:
        return tfidf_score * 0.6 + token_score * 0.4
    fuzzy_ratio = SequenceMatcher(None, message, entry.normalized_question).ratio()
    return max(overlap_ratio * 0.7 + coverage_ratio * 0.2 + fuzzy_ratio * 0.1, fuzzy_ratio)


def _top_matches(message: str, knowledge: List[KnowledgeEntry], limit: int = 5) -> List[tuple]:
    normalized_message = _normalize_text(message)
    message_tokens = _tokenize(normalized_message)
    tfidf_scores = _tfidf_similarity(message)
    scored = []
    for entry in knowledge:
        score = _match_score(normalized_message, message_tokens, entry, tfidf_scores if tfidf_scores else None)
        if score >= SEMANTIC_THRESHOLD:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored[:limit]


def _find_reply(message: str, knowledge: List[KnowledgeEntry]) -> tuple:
    if not message.strip():
        return ("Please type a message so I can help.", 1.0)
    matches = _top_matches(message, knowledge, limit=1)
    if not matches:
        return (None, 0.0)
    score, entry = matches[0]
    if score >= 0.55:
        return (entry.answer, round(score, 2))
    return (None, 0.0)


def _related_questions(message: str, knowledge: List[KnowledgeEntry], limit: int = 3) -> List[str]:
    matches = _top_matches(message, knowledge, limit=limit + 1)
    return [entry.question for _, entry in matches if entry.answer][:limit]


# === Findings & Memory ===

def _upsert_finding(topic: str, details: str, finding_type: str) -> Dict:
    findings = _load_findings()
    normalized_topic = _normalize_text(topic)
    if not normalized_topic:
        return {}
    for finding in findings:
        if _normalize_text(str(finding.get("topic", ""))) == normalized_topic:
            finding["count"] = int(finding.get("count", 1)) + 1
            finding["last_seen"] = _now_iso()
            if details:
                finding["latest_details"] = details
            if int(finding.get("count", 1)) >= TREND_THRESHOLD:
                finding["status"] = "trending"
            _save_findings(findings)
            return finding
    item = {"topic": topic.strip(), "type": finding_type, "count": 1, "status": "observed", "first_seen": _now_iso(), "last_seen": _now_iso(), "latest_details": details.strip()}
    findings.append(item)
    _save_findings(findings)
    return item


def _tag_finding_as_memorized(topic: str) -> None:
    normalized = _normalize_text(topic)
    findings = _load_findings()
    for item in findings:
        if _normalize_text(str(item.get("topic", ""))) == normalized:
            item["status"] = "memorized"
            item["last_seen"] = _now_iso()
            _save_findings(findings)
            return


def _auto_promote_finding(topic: str, llm_answer: str) -> bool:
    findings = _load_findings()
    normalized = _normalize_text(topic)
    for finding in findings:
        if _normalize_text(str(finding.get("topic", ""))) == normalized:
            if int(finding.get("count", 0)) >= AUTO_PROMOTE_THRESHOLD and finding.get("status") == "trending" and llm_answer.strip():
                for entry in STATE.curated_entries:
                    if entry.normalized_question == normalized:
                        return False
                rows = _load_json_list(KNOWLEDGE_PATH)
                rows.append({"question": topic.strip(), "answer": llm_answer.strip()})
                _save_curated_knowledge(rows)
                _remove_unresolved_question(topic)
                finding["status"] = "memorized"
                finding["last_seen"] = _now_iso()
                _save_findings(findings)
                logger.info("Auto-promoted to KB: %s", topic.strip())
                return True
            break
    return False


def _auto_learn_from_feedback(message: str, reply: str) -> bool:
    """Auto-save a Q&A pair to KB when user gives positive feedback."""
    normalized = _normalize_text(message)
    if not normalized or not reply.strip() or len(reply) < 20:
        return False
    # Don't save if already in KB
    for entry in STATE.curated_entries:
        if entry.normalized_question == normalized:
            return False
    # Don't save filler messages
    if _is_filler(message):
        return False
    # Only save if it looks like a real question
    if not _is_incident_like(message) and _classify_intent(message) == "general":
        return False
    rows = _load_json_list(KNOWLEDGE_PATH)
    rows.append({"question": message.strip(), "answer": reply.strip()})
    _save_curated_knowledge(rows)
    _remove_unresolved_question(message)
    logger.info("Auto-learned from positive feedback: %s", message[:80])
    return True


def _upsert_memory(question: str, answer: str, source_topic: str = "") -> Dict:
    normalized = _normalize_text(question)
    if not normalized or not answer.strip():
        return {}
    rows = _load_json_list(KNOWLEDGE_PATH)
    updated = False
    for item in rows:
        if _normalize_text(str(item.get("question", "")).strip()) == normalized:
            item["question"] = question.strip()
            item["answer"] = answer.strip()
            updated = True
            break
    if not updated:
        rows.append({"question": question.strip(), "answer": answer.strip()})
    _save_curated_knowledge(rows)
    _remove_unresolved_question(question)
    if source_topic.strip():
        _remove_unresolved_question(source_topic)
        _tag_finding_as_memorized(source_topic)
    return {"question": question.strip(), "answer": answer.strip(), "source_topic": source_topic.strip(), "action": "updated" if updated else "created"}


def _track_unknown_issue(message: str) -> None:
    normalized = _normalize_text(message)
    if not normalized:
        return
    if normalized not in {entry.normalized_question for entry in STATE.curated_entries} and normalized not in _load_unresolved_questions():
        _merge_unresolved_items([{"question": message.strip(), "answer": "", "status": "unresolved", "note": "Captured from unknown user issue. Fill answer later."}])
    _upsert_finding(topic=message.strip(), details="Repeated unresolved incident or new Team Glue signal.", finding_type="issue_pattern")


# === LLM ===

def _build_conversation_messages(payload: ChatRequest, knowledge: List[KnowledgeEntry], settings: Settings, intent: str, detected_patterns: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    matches = _top_matches(payload.message, knowledge, limit=5)
    related_entries = [entry for _, entry in matches]
    trending = [item["topic"] for item in _load_findings() if str(item.get("status", "")).lower() == "trending"][:5]

    intent_modifier = _get_intent_system_modifier(intent)
    pattern_context = ""
    if detected_patterns:
        pattern_context = "\nDetected error patterns in user message:\n" + "\n".join(f"- {name}: {advice}" for name, advice in detected_patterns[:3])

    system_prompt = (
        "You are GlueBot, an intelligent Team Glue operations assistant.\n"
        "Kubernetes guidance for K8s/pod issues. VEPG/EPG/SSR are OpenStack-oriented unless user says otherwise.\n"
        f"{intent_modifier}\n\n"
        f"{_tester_context_summary(payload) or 'Operating mode: standard'}\n"
        f"{_tester_documents_summary(payload)}\n"
        f"{pattern_context}\n"
        f"Known KB:\n{chr(10).join(f'- {e.question}: {e.answer}' for e in related_entries) or '- None'}\n\n"
        f"Trends:\n{chr(10).join(f'- {t}' for t in trending) or '- None'}"
    )
    messages = [{"role": "system", "content": system_prompt}]
    history = payload.history[-10:] if payload.history else []
    for msg in history:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role in {"user", "assistant"} and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": payload.message})
    return messages


def _llm_reply(payload: ChatRequest, knowledge: List[KnowledgeEntry], settings: Settings, intent: str, detected_patterns: List[Tuple[str, str]]) -> Optional[str]:
    if not settings.llm_api_key:
        return None
    allowed_hosts = {"openrouter.ai", "api.openai.com"}
    api_base = settings.llm_api_base.lower()
    if api_base and not any(host in api_base for host in allowed_hosts):
        return None
    messages = _build_conversation_messages(payload, knowledge, settings, intent, detected_patterns)
    headers = {"Authorization": f"Bearer {settings.llm_api_key}", "Content-Type": "application/json"}
    if settings.is_openrouter:
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        if settings.openrouter_app_name:
            headers["X-Title"] = settings.openrouter_app_name
    try:
        if settings.is_openrouter:
            response = requests.post(
                settings.llm_api_base or "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json={"model": settings.llm_model, "messages": messages},
                timeout=25,
            )
            response.raise_for_status()
            choices = response.json().get("choices", [])
            if not choices or not isinstance(choices[0], dict):
                return None
            return choices[0].get("message", {}).get("content", "").strip() or None
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        response = requests.post(settings.llm_api_base or "https://api.openai.com/v1/responses", headers=headers, json={"model": settings.llm_model, "input": prompt}, timeout=25)
        response.raise_for_status()
        return (response.json().get("output_text") or "").strip() or None
    except requests.RequestException as exc:
        logger.error("LLM failed: %s", exc)
        return None


def _fallback_reply(payload: ChatRequest, knowledge: List[KnowledgeEntry]) -> str:
    text = payload.message.lower()
    related = [e.question for _, e in _top_matches(payload.message, knowledge, limit=3)]
    related_text = f"\nRelated topics: {', '.join(related)}" if related else ""
    tester_text = f"\nTester mode: `{payload.test_channel or '?'}` / `{payload.test_name or '?'}`." if payload.operating_mode == "tester" else ""
    pod_fix = _pod_fix_reply(payload.message)
    if pod_fix:
        return pod_fix + tester_text + related_text
    if any(k in text for k in {"vepg", "epg", "ssr"}):
        return "VEPG/EPG/SSR issue. Check instance health, service status, volumes, networking, and recent config changes." + tester_text + related_text
    return "No exact match yet. Start with service health checks, recent changes, and component logs." + tester_text + related_text


# === API Endpoints ===

@app.on_event("startup")
def startup_event() -> None:
    _refresh_state()
    logger.info("GlueBot started. KB: %d entries, Findings: %d", len(STATE.curated_entries), len(STATE.findings))


@app.get("/")
def read_root() -> Dict[str, str]:
    _refresh_state()
    return {"status": "ok"}


@app.post("/search")
def api_search(payload: SearchRequest) -> Dict[str, List[Dict]]:
    _refresh_state()
    normalized_message = _normalize_text(payload.query)
    message_tokens = _tokenize(normalized_message)
    tfidf_scores = _tfidf_similarity(payload.query)
    scored = []
    for entry in STATE.curated_entries:
        score = _match_score(normalized_message, message_tokens, entry, tfidf_scores if tfidf_scores else None)
        if score >= 0.2:
            scored.append((score, entry))
    scored.sort(key=lambda item: item[0], reverse=True)
    return {"results": [{"question": e.question, "answer": e.answer, "score": round(s, 3)} for s, e in scored[:payload.limit]]}


@app.get("/findings")
def list_findings() -> Dict[str, List[Dict]]:
    _refresh_state()
    return {"items": sorted(STATE.findings, key=lambda i: (int(i.get("count", 0)), str(i.get("last_seen", ""))), reverse=True)}


@app.get("/buckets")
def list_buckets() -> Dict[str, List[Dict]]:
    _refresh_state()
    return {"items": STATE.buckets}


@app.get("/knowledge/export")
def export_knowledge() -> Dict[str, List[Dict]]:
    _refresh_state()
    return {"items": _load_json_list(KNOWLEDGE_PATH)}


@app.post("/knowledge/import")
def import_knowledge(items: List[Dict]) -> Dict[str, str]:
    _refresh_state()
    existing = _load_json_list(KNOWLEDGE_PATH)
    existing_normalized = {_normalize_text(str(i.get("question", ""))) for i in existing}
    added = 0
    for item in items:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if q and a and _normalize_text(q) not in existing_normalized:
            existing.append({"question": q, "answer": a})
            existing_normalized.add(_normalize_text(q))
            added += 1
    if added:
        _save_curated_knowledge(existing)
    return {"status": "ok", "added": str(added)}


@app.post("/findings/record")
def record_finding(payload: FindingRecordRequest) -> Dict[str, Dict]:
    _refresh_state()
    return {"item": _upsert_finding(payload.topic, payload.details, "manual_finding")}


@app.post("/memory/upsert")
def upsert_memory(payload: MemoryUpdateRequest) -> Dict[str, Dict]:
    _refresh_state()
    item = _upsert_memory(payload.question, payload.answer, payload.source_topic)
    return {"item": item or {}}


@app.post("/feedback")
def submit_feedback(payload: FeedbackRequest) -> Dict[str, str]:
    logger.info("FEEDBACK | helpful=%s | source=%s | msg=%s", payload.helpful, payload.source, payload.message[:80])
    if payload.helpful and payload.source.startswith("llm:"):
        _auto_learn_from_feedback(payload.message, payload.reply)
    elif not payload.helpful and _is_incident_like(payload.message):
        _track_unknown_issue(payload.message)
    return {"status": "recorded"}


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    if not _check_rate_limit():
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")

    _refresh_state()

    intent = _classify_intent(payload.message)
    severity = ""
    detected_patterns = []

    logger.info("CHAT | intent=%s | mode=%s | msg=%s", intent, payload.operating_mode, payload.message[:80])

    # 1. Filler detection (save LLM calls)
    filler = _is_filler(payload.message)
    if filler:
        return ChatResponse(reply=filler, source="intent:filler", operating_mode=payload.operating_mode, confidence=1.0, intent=intent)

    # 2. Small talk
    smalltalk = _smalltalk_reply(payload.message)
    if smalltalk:
        return ChatResponse(reply=smalltalk, source="intent", operating_mode=payload.operating_mode, confidence=1.0, intent=intent)

    # 3. Log pattern detection
    detected_patterns = _detect_log_patterns(payload.message)
    if detected_patterns and len(payload.message) > 100:
        # Long message with detected patterns — likely a pasted log
        pattern_names = [p[0] for p in detected_patterns]
        combined_advice = "\n\n".join(f"**{name}**: {advice}" for name, advice in detected_patterns[:3])
        reply = f"I detected the following error patterns in your log:\n\n{combined_advice}\n\nWant me to dig deeper into any of these?"
        severity = _classify_severity(payload.message)
        return ChatResponse(
            reply=reply, source="pattern_detection", operating_mode=payload.operating_mode,
            confidence=0.85, intent=intent, severity=severity, detected_patterns=pattern_names,
            related_questions=_related_questions(payload.message, STATE.curated_entries, limit=3),
        )

    # 4. KB match
    matched_answer, confidence = _find_reply(payload.message, STATE.curated_entries)
    related = _related_questions(payload.message, STATE.curated_entries, limit=3)
    if matched_answer:
        severity = _classify_severity(payload.message) if _is_incident_like(payload.message) else ""
        return ChatResponse(reply=matched_answer, source="knowledge.json", operating_mode=payload.operating_mode, confidence=confidence, related_questions=related, intent=intent, severity=severity)

    # 4b. Intent clarifying question for vague messages (never for P1/incident-like)
    severity_check = _classify_severity(payload.message)
    intent_question = _get_intent_question(intent, payload.message)
    if intent_question and not matched_answer and not _is_incident_like(payload.message) and severity_check != "P1":
        return ChatResponse(reply=intent_question, source="intent:clarify", operating_mode=payload.operating_mode, confidence=1.0, intent=intent)

    # 5. Pod heuristic
    pod_fix = _pod_fix_reply(payload.message)
    if pod_fix:
        severity = _classify_severity(payload.message)
        return ChatResponse(reply=pod_fix, source="heuristic:pod_fix", operating_mode=payload.operating_mode, confidence=0.85, related_questions=related, intent=intent, severity=severity)

    # 6. OpenStack script
    scripted = _openstack_script_reply(payload.message)
    if scripted:
        return ChatResponse(reply=scripted, source="template:openstack_script", operating_mode=payload.operating_mode, confidence=0.9, related_questions=related, intent=intent)

    # 7. Track unknown
    if _is_incident_like(payload.message):
        _track_unknown_issue(payload.message)
        severity = _classify_severity(payload.message)

    # 8. LLM with intent-aware prompt
    llm = _llm_reply(payload, STATE.curated_entries, STATE.settings, intent, detected_patterns)
    if llm:
        _auto_promote_finding(payload.message, llm)
        pattern_names = [p[0] for p in detected_patterns]
        return ChatResponse(reply=llm, source=f"llm:{STATE.settings.llm_model}", operating_mode=payload.operating_mode, confidence=0.7, related_questions=related, intent=intent, severity=severity, detected_patterns=pattern_names)

    # 9. Fallback
    fallback_source = "fallback:no_llm_api_key" if not STATE.settings.llm_api_key else "fallback:llm_unavailable"
    return ChatResponse(reply=_fallback_reply(payload, STATE.curated_entries), source=fallback_source, operating_mode=payload.operating_mode, confidence=0.3, related_questions=related, intent=intent, severity=severity)
