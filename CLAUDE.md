# CLAUDE.md — Developer Context for Scientific RAG QA System

This file gives a Claude Code agent (or a new human contributor) everything needed to
understand the project without reading every file from scratch. All facts below were
confirmed by reading actual source files.

---

## 1. Project Overview

This is a **production-ready Retrieval-Augmented Generation (RAG) system** that answers
questions about scientific NLP papers with real-time streaming and multi-turn memory.

### Two-service architecture

| Service | Technology | Port | Entry point |
|---|---|---|---|
| **backend-api** | FastAPI + Uvicorn | 8080 | `backend/app/main.py` |
| **frontend-ui** | Chainlit | 8001 | `frontend/app.py` |

Both services are shipped as frozen Docker images built by GitHub Actions and stored in
GHCR (`ghcr.io/mnguegnang/rag-app-scientific-qa-{backend,frontend}:latest`).
Deployment is a single `docker compose pull && docker compose up -d`.

### Key pipeline stages (backend)

1. **Artifact download** — FAISS indices and BM25 pickle are pulled from Google Drive on
   first boot via `gdown` (two-strategy fallback to `drive.usercontent.google.com`).
2. **Hybrid retrieval** — FAISS HNSW dense index (Sentence-Transformers embeddings) +
   BM25 sparse index; top-50 candidates are fused.
3. **Cross-encoder reranking** — `adapters` cross-encoder re-scores the 50 candidates;
   best 7 are forwarded.
4. **CRAG gate** — if `best_score < pipeline.CRAG_THRESHOLD` the system returns a
   low-confidence warning instead of generating a hallucinated answer.
5. **Streaming generation** — Ollama (CPU, Llama 3) or HuggingFace Transformers (GPU,
   auto-detected). Tokens are pushed via `asyncio.to_thread` / `httpx.AsyncClient` as
   NDJSON to the frontend.
6. **Multi-turn memory** — last 6 conversation pairs (12 entries) are injected into the
   LLM prompt inside a `<Conversation History>` XML block.

---

## 2. Commit History Summary (last 7 commits)

```
c86ace5 Revised Readme
8fd125e Update README.md visual with rag_system_architecture.svg and rag_project_structure.html
c8fc155 Update with the implementation of: Async Streaming Architecture, CI/CD SHA Tagging,
        Health Check Endpoint, and Multi-Turn Memory
1b0ff9f Merge branch 'master' of https://github.com/mnguegnang/RAG-app-scientific-QA
dd22f7a  Update the production server compose.yml replace the build commands with image
94bc879 Improve the RAG chatboard to correctly display the thinking steps and the final answer
        with. Add Github Actions CI/CD pipeline
635ca65 Update project title in README.md
```

---

## 3. Confirmed Findings and Patches Applied

All 6 findings were confirmed by reading actual source files and are fixed. All patches
were verified by a 14-check regression guard (100 % pass).

| # | Severity | File | Finding | Patch applied |
|---|---|---|---|---|
| 1 | **High** | `frontend/app.py` | History save was placed AFTER `msg.update()` (a Chainlit websocket call). A websocket disconnect between stream-end and UI update silently lost the entire exchange from session memory. | Moved `cl.user_session.set("history", ...)` to immediately after `final_answer` is finalized — before any UI operations (`msg.content`, `msg.actions`, `msg.update()`). Now at line 288, `msg.update()` at line 302. |
| 2 | **High** | `frontend/app.py` | Both `except httpx.HTTPError` and `except Exception` handlers called `return` without saving any accumulated `full_response` to session history. Any network error or Ollama timeout during generation silently reset multi-turn context. | Added guarded history save (`if full_response.strip()`) inside both handlers before `return`. Partial LLM output is parsed with `_parse_final_answer()` and stored. |
| 3 | **High** | `README.md` §7 | `ollama serve &` followed immediately by `ollama pull llama3` caused a race condition — the pull could fail or connect to an unready daemon. | Added `until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 2; done` readiness loop between the two commands. |
| 4 | **High** | `README.md` §7 | `network_mode: "host"` is Linux-only. macOS/Windows Docker Desktop users cannot reach host-installed Ollama via `localhost` from inside a container. | Added clearly labelled callout explaining Docker Desktop's Linux VM isolation and providing `OLLAMA_URL=http://host.docker.internal:11434/api/generate docker compose up -d` as the override. |
| 5 | **Medium** | `README.md` §7 | First-boot time was stated as "3–5 minutes". Actual first-boot includes ~230 MB FAISS indices + ~400 MB embedding model + ~100 MB reranker — total well over 700 MB plus pipeline init. | Updated to "10–20 minutes" with an explanation of all downloads. Added `docker compose logs -f backend-api` log-watch command. |
| 6 | **Medium** | `README.md` §7 | No readiness verification step after `docker compose up -d`. Users opened the browser URL before the RAG pipeline had finished initializing, seeing errors. | Added a looping health check: `until curl -sf http://localhost:8080/health \| grep -q '"status":"healthy"'; do sleep 10; done`. |

### Section 7 coherence pass (same session)

In addition to the 6 individual patches, a coherence review of the full "How to Run"
section identified and fixed the following structural problems:

- **Misleading heading**: "Quick Start (One Command)" promised one command but contained
  3 numbered steps. Rewritten as sequential numbered sub-sections (Steps 1–5).
- **Flow inversion**: the URL table appeared *before* the log-watch and health-check
  commands, implying the services were reachable before the pipeline had loaded.
  Reordered so Steps 4 (wait/health-check) and 5 (open URLs) are in the correct sequence.
- **macOS/Windows note placement**: the callout was separated from the `docker compose up -d`
  command it applied to by the URL table. It now sits directly beneath Step 3.
- **One-shot health check**: the original `curl` ran once with no loop. Replaced with an
  `until` polling loop with a human-readable status message.
- **Missing GPU skip note**: added a callout at Step 2 clarifying that Ollama setup is
  for CPU mode only and can be skipped when a CUDA device is present.
- **Sub-section separators**: added `---` horizontal rules between each step so the
  section is scannable without dense continuous prose.

---

## 4. How to Run for Developers (Local, Without Docker)

### Backend

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# The backend imports from the RAG research project.
# Either install it:
pip install git+https://github.com/mnguegnang/RAG-for-Scientific-QA
# Or clone it and symlink/add to PYTHONPATH so `from src.run_rag import ScientificRAGPipeline` resolves.

# Ensure Ollama is running with llama3
ollama serve &
until curl -s http://localhost:11434/api/tags > /dev/null 2>&1; do sleep 2; done
ollama pull llama3

# Artifacts must be present in backend/app/artifacts/ — either pre-download manually
# (see README §5) or let the server download them on first start.

# Start the backend
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

Health check: `curl http://localhost:8080/health` — should return `{"status":"healthy","pipeline":"loaded"}`.

### Frontend

```bash
cd frontend

pip install -r requirements.txt

# Point at the local backend
export BACKEND_API_URL=http://localhost:8080/api/chat

chainlit run app.py --host 0.0.0.0 --port 8001
```

Open `http://localhost:8001` in a browser.

---

## 5. Architecture Notes

### Key files

| File | Role |
|---|---|
| `backend/app/main.py` | FastAPI app: lifespan startup (artifact download + pipeline init), `/health`, `/api/chat` (sync), `/api/chat/stream` (async NDJSON) |
| `frontend/app.py` | Chainlit UI: NDJSON stream consumer, tag-scanning state machine (`pre → reasoning → between → answer → done`), session history management, references renderer |
| `compose.yml` | Two-service Compose config; both services use `network_mode: "host"` (Linux only) so the frontend can reach `localhost:8080` and the backend can reach `localhost:11434` (Ollama) |
| `backend/Dockerfile` | `python:3.10-slim` + build-essential + git; runs `uvicorn app.main:app` |
| `frontend/Dockerfile` | `python:3.10-slim`; runs `chainlit run app.py` |
| `backend/requirements.txt` | Pinned: `fastapi==0.133.0`, `faiss-cpu==1.13.2`, `adapters==1.2.0`, `rank-bm25==0.2.2`; unpinned: `sentence-transformers`, `pydantic`, `httpx` |

### Non-obvious design decisions

- **`network_mode: "host"`** is deliberately chosen so the containerised backend can
  reach a host-installed Ollama without any port remapping. This is Linux-only and breaks
  on Docker Desktop (macOS/Windows). See README §7 for the `host.docker.internal` workaround.

- **Tag-scanning state machine in `frontend/app.py`**: the frontend parses raw LLM token
  streams looking for `<Reasoning>…</Reasoning>` and `<Final Answer>…</Final Answer>` XML
  tags. A `_TAG_LOOKAHEAD = 20` byte buffer prevents tags from being split across chunk
  boundaries. The state machine has five states: `pre`, `reasoning`, `between`, `answer`,
  `done`. The `_parse_final_answer()` regex is a second-pass safety net applied to the
  complete `full_response` after streaming ends.

- **Dual-strategy Google Drive download** (`_gdrive_download` in `backend/app/main.py`):
  strategy 1 uses `gdown` with fuzzy URL matching; strategy 2 falls back to the
  `drive.usercontent.google.com` direct endpoint via `requests`. This handles Google's
  virus-scan confirmation pages transparently.

- **`sparse.pkl` uses Python `pickle`** (`backend/app/artifacts/sparse.pkl`): the BM25
  index is loaded via `pickle.load()`. This is safe only because the artifact is produced
  by the controlled companion research project, not user input. Never load arbitrary
  user-supplied pickle files.

- **CRAG gate**: `pipeline.CRAG_THRESHOLD` is defined in the upstream RAG research project
  (not in this repo). The gate fires when `max(rerank_score for doc in top_docs) < threshold`.
  The rerank score is an uncalibrated cross-encoder logit, so the threshold value matters
  significantly for recall vs. precision of the gate.

- **History window**: the frontend keeps the last 12 entries (6 pairs) in
  `cl.user_session`. The backend further truncates to `history[-12:]` before injecting
  into the prompt, and each entry is hard-capped at 500 characters to bound prompt growth.

---

## 6. Open Issue: CRAG Gate Blocks Multi-Turn Memory for Follow-Up Questions

**Status:** Confirmed bug — not yet fixed. Requires a design decision before implementation.  
**Confirmed by:** Live end-to-end test, Apr 21 2026 (see test evidence below).

### The problem

Multi-turn history injection only fires when the **CRAG gate does not trigger**. Inside the streaming generator in `backend/app/main.py`, the execution order is:

```
Stage 2: reranking (line 228–231)
  → CRAG gate check  (lines 249–255)  ← returns early if best_score < CRAG_THRESHOLD
  → build full_prompt (line 261)
  → history injection (lines 263–277) ← NEVER reached when CRAG fires
  → Ollama streaming  (line 281+)
```

Follow-up questions that depend on conversational context — *"tell me more"*, *"give me an example"*, *"what did you mean by X"*, *"elaborate on that"* — do not independently match scientific NLP paper abstracts. Their cross-encoder rerank scores are very low, the CRAG gate triggers, and the LLM returns a fixed low-confidence warning **without ever seeing the conversation history**.

### Confirmed test evidence (Apr 21 2026)

Two back-to-back live requests sent to `POST /api/chat/stream` with identical history payload (`2 entries`):

| Prompt | CRAG best logit | Gate | Stage 3 | `HISTORY` log line |
|---|---|---|---|---|
| `"Can you give me a concrete example of what you described?"` | `0.0823` | ✗ triggered | ✗ blocked | ✗ absent |
| `"How does self-attention in BERT differ from the original Transformer?"` | `0.9689` | ✓ passed | ✓ reached | ✓ `HISTORY: Injecting 2 entries into prompt.` |

Backend log confirmation for the passing case:
```
2026-04-21 03:26:54,854 - INFO - CRAG check — best logit: 0.9689 (threshold: 0.5170)
2026-04-21 03:26:54,854 - INFO - Stage 3: Streaming generation...
2026-04-21 03:26:54,855 - INFO - HISTORY: Injecting 2 entries into prompt.
```

### Impact

Multi-turn memory is functionally broken for the most natural follow-up patterns. It only works when the user asks a new, self-contained research question that independently retrieves well from the corpus. The `<Conversation History>` block never reaches the LLM for context-dependent follow-ups.

### Three fix options (not yet implemented)

**Option A — Move history injection before the CRAG gate** *(smallest change)*  
Relocate the history injection block (lines 263–277) to immediately after `full_prompt = gen._build_prompt(...)` on line 261, before the CRAG gate. The LLM then always has conversation context, even when it returns the low-confidence warning. Change is ~5 lines in `backend/app/main.py`.

**Option B — History-augmented retrieval** *(most architecturally correct)*  
Prepend the last assistant turn to the retrieval query so dense/BM25 search operates over `<prior_answer> + <follow_up>` rather than the bare follow-up. Improves recall for contextual queries without touching the CRAG gate logic. Requires modifying the `pipeline.retriever.search()` call at line 217.

**Option C — Context-query bypass** *(riskiest)*  
Detect context-only follow-ups (pronouns referencing prior turns: *"you said"*, *"that"*, *"above"*, *"your previous"*) and lower or skip the CRAG threshold for those queries. Fragile — hard to detect reliably.

**Recommendation:** Ship Option A immediately (unblocks memory for CRAG-triggered responses), then implement Option B as the long-term fix.
