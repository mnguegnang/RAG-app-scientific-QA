# Issue Resolution Log — RAG App Scientific QA

**Date:** 2026-04-22  
**Project:** `RAG-app-scientific-QA`  
**Reported symptom:** `docker compose up -d` failed with the backend container exiting immediately (exit code 1 / 3).

---

## Issue 1 — Backend image pulled from GHCR was broken

### Symptom
```
✘ Container rag-app-scientific-qa-backend-api-1  Error
dependency failed to start: container ... exited (1)
```

### Root Cause
`compose.yml` was configured to pull a pre-built image from the GitHub Container Registry:
```yaml
image: ghcr.io/mnguegnang/rag-app-scientific-qa-backend:latest
```
That published image was missing dependencies and could not be updated without a new push to GHCR.

### Fix — `compose.yml`
Replaced the remote image reference with a local build directive so the project's own `Dockerfile` and `requirements.txt` are used:

```yaml
# Before
image: ghcr.io/mnguegnang/rag-app-scientific-qa-backend:latest

# After
build: ./backend
```

---

## Issue 2 — `ModuleNotFoundError: No module named 'ragatouille'`

### Symptom
```
File ".../src/retrieval/reranker.py", line 27, in <module>
    from ragatouille import RAGPretrainedModel
ModuleNotFoundError: No module named 'ragatouille'
```

### Root Cause
`ragatouille` was not listed in `backend/requirements.txt`.

### Fix — `backend/requirements.txt`
Added the package under the Retrieval AI section:
```
ragatouille==0.0.9.post2
```

---

## Issue 3 — `ModuleNotFoundError: No module named 'psutil'`

### Symptom
```
File ".../fast_pytorch_kmeans/util.py", line 4, in <module>
    import psutil
ModuleNotFoundError: No module named 'psutil'
```

### Root Cause
`fast_pytorch_kmeans` (a transitive dependency of `ragatouille`) depends on `psutil` but does not declare it, so pip does not install it automatically.

### Fix — `backend/requirements.txt`
Added `psutil` explicitly:
```
psutil
```

---

## Issue 4 — `ModuleNotFoundError: No module named 'langchain.retrievers'`

### Symptom
```
ModuleNotFoundError: No module named 'langchain.retrievers'
```

### Root Cause
In LangChain 1.x, `langchain.retrievers` was moved to the `langchain-community` package. The backend had `langchain-core` and `langchain-ollama` but not `langchain-community`.

### Fix — `backend/requirements.txt`
Added the missing package:
```
langchain-community
```

---

## Issue 5 — Path traversal guard blocking FAISS artifact loading (exit code 3)

### Symptom
```
ValueError: Path traversal blocked: '/app/artifacts/dense.index.meta' resolves to
'/app/artifacts/dense.index.meta' which is outside the project root
'/usr/local/lib/python3.10/site-packages'.
```

### Root Cause
`src/retrieval/hybrid_retriever.py` contains a security guard (`_load_pickle_verified`) that checks loaded file paths are within `_PROJECT_ROOT`, computed as:
```python
_PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
```
Because `requirements.txt` contained `git+https://github.com/mnguegnang/RAG-for-Scientific-QA`, pip installed the `src` package into `site-packages/src/`. As a result, `__file__` resolved to a path inside `site-packages`, making `_PROJECT_ROOT = /usr/local/lib/python3.10/site-packages` — which does not contain `/app/artifacts/`, causing the guard to reject every artifact load.

### Fix — `backend/Dockerfile`
Added a `git clone` step to place `src/` directly at `/app/src/`. Python's import system finds this local copy first (the working directory `/app` is ahead of `site-packages` in `sys.path`), so `__file__` now resolves under `/app` and `_PROJECT_ROOT = /app`.

```dockerfile
# Clone the RAG source so that src/ lives at /app/src/.
# This makes _PROJECT_ROOT resolve to /app instead of site-packages,
# fixing the path-traversal guard when loading artifacts from /app/artifacts/.
RUN git clone --depth=1 https://github.com/mnguegnang/RAG-for-Scientific-QA /tmp/rag-src \
    && mv /tmp/rag-src/src /app/src \
    && rm -rf /tmp/rag-src
```

### Fix — `backend/requirements.txt`
Removed the `git+https://github.com/mnguegnang/RAG-for-Scientific-QA` line so pip no longer installs `src` into `site-packages` (which was the source of the bad `_PROJECT_ROOT`).

---

## Issue 6 — GPU not detected inside the Docker container (`GPUs=0`)

### Symptom
Backend logs showed `GPUs=0` and `generator_backend=auto resolved to: ollama` even on a machine with an NVIDIA L4 GPU.

### Root Cause
Two problems:
1. `compose.yml` had no GPU device reservation, so Docker never exposed the GPU to the container.
2. `requirements.txt` didn't explicitly install `accelerate`, required by HuggingFace `device_map="auto"`.

### Fix — `compose.yml`
Added `deploy.resources.reservations.devices` to pass the GPU through:
```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

### Fix — `backend/requirements.txt`
Added `accelerate` under Generative AI dependencies.

---

## Issue 7 — Chatbot shows empty answers (`full_response len=0`)

### Symptom
Every query returned `Copy answer` and References but no answer text. Frontend logs showed:
```
STATE final section=pre | full_response len=0 | parsed len=0 | answer_tokens len=0
```

### Root Cause
The external RAG project (`RAG-for-Scientific-QA`) was updated between builds. The streaming endpoint in `main.py` called `pipeline.CRAG_THRESHOLD` (a simple float), which no longer exists — replaced by `pipeline.crag_evaluator` (a `CRAGEvaluator` object). The backend threw `AttributeError` before sending a single token, so `full_response` was empty.

Additionally, the Dockerfile cloned with `--depth=1` (always latest HEAD), so upstream changes were silently pulled in on every `docker build`.

### Fix — `backend/app/main.py`
Replaced the old single-threshold CRAG gate:
```python
# Before (broken)
if best_score < pipeline.CRAG_THRESHOLD: ...

# After — uses the new three-way CRAGEvaluator API
crag_action, refined_docs, crag_details = await asyncio.to_thread(
    lambda: pipeline.crag_evaluator.evaluate_and_refine(request.prompt, top_docs)
)
if crag_action == 'Incorrect':
    refined_docs = sorted(top_docs, key=lambda x: x.get('rerank_score', 0.0), reverse=True)[:3]
```

Also updated downstream `top_docs` → `refined_docs` in `_build_prompt` and `generate_answer` calls.

### Fix — `backend/Dockerfile`
Pinned the external RAG project to a specific commit SHA so upstream changes don't silently break the API contract on the next build:
```dockerfile
ARG RAG_COMMIT=f96c2bd36f7d1d8377ebfc88d6ce9d65322b8b1d
RUN git clone https://github.com/mnguegnang/RAG-for-Scientific-QA /tmp/rag-src \
    && cd /tmp/rag-src && git checkout ${RAG_COMMIT} \
    && mv src /app/src && rm -rf /tmp/rag-src
```

### Fix — `backend/app/main.py` (startup assertion)
Added an API contract check at server boot that fails immediately with a clear message if the external project renames or removes an attribute `main.py` depends on:
```python
missing = [attr for attr in ("retriever", "reranker", "crag_evaluator", "generator")
           if not hasattr(pipeline, attr)]
if missing:
    raise AttributeError(f"ScientificRAGPipeline API mismatch — missing: {missing}. ...")
```

---

## Issue 8 — HuggingFace model cache lost on container rebuild

### Symptom
After `docker compose build`, the backend failed with `401 Client Error` trying to re-download `meta-llama/Llama-3.1-8B-Instruct` because the cached weights lived in the old container's ephemeral layer.

### Fix — `compose.yml`
Added a bind mount to Lightning AI's persistent teamspace storage and set `HF_HOME` inside the container to point to it:
```yaml
environment:
  - HF_HOME=/hf_cache
volumes:
  - /teamspace/studios/this_studio/.cache/huggingface:/hf_cache
```
The model now downloads once and survives any number of container rebuilds.

---

## Current State After All Fixes

The full stack runs on GPU with a single command:

```bash
export HF_TOKEN=<your_huggingface_token>   # only needed on first boot to download Llama
cd ~/RAG-app-scientific-QA
docker compose up -d
```

- SPECTER2 encoder → `cuda:0`
- ColBERT v2 reranker → `cuda:0`
- Llama-3.1-8B-Instruct → `device_map=auto` on `cuda:0`
- 887 papers indexed, streaming answers with `<Reasoning>` + `<Final Answer>` tags
- Model weights cached at `/teamspace/.cache/huggingface` (persistent across rebuilds)

---

## Summary of All Files Changed

| File | Change |
|---|---|
| `compose.yml` | `build: ./backend`; GPU device reservation; `HF_TOKEN` + `HF_HOME` env vars; HF cache bind mount |
| `backend/requirements.txt` | Added `ragatouille==0.0.9.post2`, `psutil`, `langchain-community`, `accelerate`; removed `git+https://...` |
| `backend/Dockerfile` | `git clone` → `git checkout ${RAG_COMMIT}` (pinned SHA) |
| `backend/app/main.py` | Fixed CRAG gate to use `pipeline.crag_evaluator.evaluate_and_refine()`; startup API assertion |
