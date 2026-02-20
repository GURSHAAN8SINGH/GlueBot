# GlueBot

GlueBot is a local chatbot for SRE/ops support focused on:
- Kubernetes troubleshooting
- OpenStack troubleshooting
- Safe script generation (example: bulk delete available OpenStack volumes with dry-run)

It uses:
- `FastAPI` backend (`main.py`)
- `Streamlit` frontend (`app.py`)
- Local knowledge base (`knowledge.json`)
- Optional LLM fallback (OpenRouter or OpenAI) for unknown issues

## Project Structure

- `main.py`: API server and bot logic (`/` health + `/chat`)
- `app.py`: Streamlit chat UI
- `knowledge.json`: curated Q/A + auto-captured unresolved questions
- `.env`: runtime configuration and API keys
- `requirements.txt`: Python dependencies

## How We Built This Bot

1. Created a FastAPI backend and Streamlit UI skeleton.
2. Added health check and chat endpoint integration.
3. Added local KB matching from `knowledge.json`.
4. Added fallback logic for unknown issues.
5. Added interactive UI (chat history, quick prompts, clear chat).
6. Added OpenRouter/OpenAI LLM fallback for unknown issues.
7. Added auto-capture of unresolved issues into `knowledge.json`.
8. Expanded domain support from Kubernetes-only to Kubernetes + OpenStack.
9. Added OpenStack script-template response for safe bulk volume deletion.

## Prerequisites

- Python 3.10+ (works with newer versions too)
- PowerShell (for commands below on Windows)

## Setup

1. Create virtual environment:
```powershell
python -m venv .venv
```

2. Activate it:
```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Configuration (`.env`)

Use OpenRouter (recommended in this project):

```env
OPENAI_API_KEY=
OPENAI_MODEL=openai/gpt-4o-mini
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx
LLM_API_BASE=https://openrouter.ai/api/v1/chat/completions
OPENROUTER_SITE_URL=http://localhost:8501
OPENROUTER_APP_NAME=GlueBot
BACKEND_URL=http://127.0.0.1:8000
```

Notes:
- Keep `OPENAI_API_KEY` empty if using OpenRouter.
- `OPENAI_MODEL` here is used as the model field for both providers.

## Run the Bot

Open 2 terminals (both with `.venv` activated):

1. Start backend:
```powershell
uvicorn main:app --reload
```

2. Start frontend:
```powershell
streamlit run app.py
```

Then open:
- Streamlit UI: `http://localhost:8501`
- API health: `http://127.0.0.1:8000`

## How Chat Routing Works

When you send a message:

1. Small-talk intent check (`hi`, `help`, etc.)
2. Knowledge-base match from `knowledge.json`
3. OpenStack script template check (volume delete case)
4. LLM fallback (OpenRouter/OpenAI) for unknown issues
5. Rule-based fallback if LLM is unavailable

Response `source` in UI explains which path was used:
- `intent`
- `knowledge.json`
- `template:openstack_script`
- `llm:<model>`
- `fallback:no_llm_api_key`
- `fallback:llm_unavailable`

## Knowledge Base Management

`knowledge.json` supports entries like:

```json
[
  {
    "question": "pod stuck in termination",
    "answer": "Check finalizers and preStop hooks ..."
  }
]
```

Unknown incident-like questions are auto-added as unresolved:
- `answer: ""`
- `status: "unresolved"`
- `note: "Captured from unknown user issue. Fill answer later."`

## Example Prompts

- `pod stuck in termination`
- `liveness probe returning 500 error`
- `openstack instance stuck in ERROR`
- `delete all openstack volumes in available state`

## Troubleshooting

- `pip is not recognized`:
  - Use `python -m pip ...` instead of `pip ...`.

- `Error loading ASGI app. Attribute "app" not found`:
  - Ensure `main.py` defines `app = FastAPI(...)`.

- UI works but LLM not used:
  - Check `.env` key values.
  - If source is `fallback:no_llm_api_key`, key is missing/empty.
  - If source is `fallback:llm_unavailable`, key exists but provider/model call failed.

- No response change after `.env` edit:
  - Save file and send a new message.
  - Restart backend/frontend if needed.

## Security Notes

- Do not commit real API keys.
- Use dry-run before destructive OpenStack scripts.
- Confirm project/tenant scope before delete operations.

