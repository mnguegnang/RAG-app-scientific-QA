import asyncio
import json as json_module
import os
import logging
import sys
import gdown
import httpx
import requests as _req_gdrive
try:
    import torch as _torch
    _CUDA_AVAILABLE = _torch.cuda.is_available()
except ImportError:
    _CUDA_AVAILABLE = False
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Dict, List, Optional

# The source RAG project src/ is bind-mounted to /app/src at runtime (compose.yml)
# Pulled directly from github.com/mnguegnang/RAG-for-Scientific-QA
from src.run_rag import ScientificRAGPipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- GLOBAL ML STATE ---
ml_models = {}

ARTIFACTS_DIR = "/app/artifacts"

def pull_artifacts_from_gdrive():
    """Downloads the FAISS indices from Google Drive to /app/artifacts/, bypassing virus-scan warnings."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # GOOGLE DRIVE IDs of the artifacts (indices). gdown bypasses the virus-scan warning
    # and enables direct downloading of large files.
    drive_files = {
        "dense.index":      "1l40d9f-yleEBzOp3zGPH0LHa28XxIyoq",
        "dense.index.meta": "1Qxf1mKFEDKKOvEwphZJKv7mtp53MndYf",
        "sparse.pkl":       "1F74_vEnlRC2St36twyhH7Ve5gHDU5rA4",
    }

    for filename, file_id in drive_files.items():
        output_path = os.path.join(ARTIFACTS_DIR, filename)

        # Skip download if file already exists (e.g. provided via Docker volume mount)
        if os.path.exists(output_path):
            logging.info(f"{filename} already exists at {output_path}. Skipping download.")
            continue

        logging.info(f"Downloading {filename} from Google Drive...")
        _gdrive_download(file_id, output_path, filename)  # raises on failure
        logging.info(f"Successfully downloaded {filename}.")


def _gdrive_download(file_id: str, output_path: str, filename: str) -> None:
    """
    Robustly download a file from Google Drive using two strategies:
      1. gdown with the standard shareable-link URL + fuzzy mode
         (handles virus-scan confirmation pages automatically).
      2. drive.usercontent.google.com direct endpoint via requests
         (bypasses the confirmation page entirely — works when gdown fails).
    """
    # Strategy 1: gdown with shareable-link URL
    share_url = f"https://drive.google.com/file/d/{file_id}/view"
    try:
        gdown.download(url=share_url, output=output_path, quiet=False, fuzzy=True)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return
    except Exception as e1:
        logging.warning("  GDrive strategy 1 (gdown) failed for %s: %s", filename, e1)
        if os.path.exists(output_path):  # remove any partial file
            os.remove(output_path)

    # Strategy 2: direct usercontent endpoint (no confirmation required)
    logging.info("  Trying direct usercontent endpoint for %s...", filename)
    direct_url = (
        f"https://drive.usercontent.google.com/download"
        f"?id={file_id}&export=download&authuser=0&confirm=t"
    )
    try:
        with _req_gdrive.get(direct_url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            with open(output_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
        if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
            return
    except Exception as e2:
        logging.warning("  GDrive strategy 2 (requests) failed for %s: %s", filename, e2)
        if os.path.exists(output_path):
            os.remove(output_path)

    raise RuntimeError(
        f"All download strategies failed for '{filename}'. "
        "Verify the file is shared as 'Anyone with the link' on Google Drive."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Executes exactly ONCE when the server starts."""
    gpu_info = f"CUDA available — {_torch.cuda.get_device_name(0)}" if _CUDA_AVAILABLE else "No GPU detected"
    logging.info("SERVER BOOT: Initializing Scientific RAG Pipeline... (%s)", gpu_info)
    logging.info("Generator backend will be: %s", "transformers (GPU)" if _CUDA_AVAILABLE else "ollama (CPU)")
    try:
        # 1. Download Artifacts from Google Drive
        pull_artifacts_from_gdrive()

        # 2. Initialize the Pipeline using GitHub Code + Google Drive Artifacts
        logging.info("Loading models into memory...")
        #Load all 3 required artifacts securely from the read-only Docker Volume
        # Load the heavy FAISS models and Ollama connections into global RAM
        ml_models["rag_pipeline"] = ScientificRAGPipeline(
            dense_index_path=os.path.join(ARTIFACTS_DIR, "dense.index"),
            dense_meta_path=os.path.join(ARTIFACTS_DIR, "dense.index.meta"),
            sparse_index_path=os.path.join(ARTIFACTS_DIR, "sparse.pkl"),
            generator_backend="auto",
            ollama_model="llama3",
        )

        # Build paper_id → title lookup from dense metadata (used by streaming endpoint)
        title_map = {}
        for entry in ml_models["rag_pipeline"].retriever.dense_meta:
            pid = entry.get("paper_id", "")
            if pid and pid not in title_map:
                title_map[pid] = entry.get("title", pid)
        ml_models["title_map"] = title_map
        logging.info("Built title lookup for %d papers.", len(title_map))

        logging.info("SERVER BOOT: RAG Pipeline is LIVE")
    except Exception as e:
        logging.error(f"CRITICAL: Failed to load FAISS artifacts. Check Docker Volumes: {e}")
        raise e
    
    yield # Server accepts requests while paused here
    
    logging.info("SERVER SHUTDOWN: Clearing ML memory...")
    ml_models.clear()

# Initialize FastAPI
app = FastAPI(title="Scientific RAG Backend", lifespan=lifespan)

# --- PYDANTIC DATA CONTRACTS ---
class ChatRequest(BaseModel):
    prompt: str
    history: Optional[List[Dict[str, str]]] = None  # [{"role": "user/assistant", "content": "..."}]

class ChatResponse(BaseModel):
    answer: str
    contexts: List[str] # We return the raw text of the 5 retrieved documents 

# --- THE INFERENCE ENDPOINT ---
@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Receives a prompt, queries the local ML pipeline (RAG), and returns the answer."""
    try:
        pipeline = ml_models.get("rag_pipeline")
        if not pipeline:
            raise HTTPException(status_code=500, detail="Model not loaded in memory.")
        
        # Run the RAG inference
        # the run_rag() method returns {"answer": str, "retrieved_docs": top_10_docs}
        result = pipeline.ask(request.prompt)
        
        # Ensure contexts are pure strings for JSON serialization
        context_strings = [
            doc["text"] if isinstance(doc, dict) else str(doc) 
            for doc in result["retrieved_docs"]
        ]
        
        return ChatResponse(
            answer=result["answer"],
            contexts=context_strings
        )
    except Exception as e:
        logging.error(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- HEALTH CHECK ENDPOINT ---
@app.get("/health")
async def health_check():
    """Returns 200 if the RAG pipeline is loaded and ready to serve requests."""
    pipeline = ml_models.get("rag_pipeline")
    if pipeline is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "RAG pipeline not loaded."},
        )
    return {"status": "healthy", "pipeline": "loaded"}


# --- STREAMING ENDPOINT (native async — no threads) ---
@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streams tokens word-by-word via NDJSON for real-time UI updates.

    Architecture: pure async generator — blocking ML calls (retrieval,
    reranking) are dispatched via ``asyncio.to_thread`` and Ollama
    streaming uses ``httpx.AsyncClient`` so the event-loop is never
    blocked and there are no thread-safety concerns.
    """
    pipeline = ml_models.get("rag_pipeline")
    if not pipeline:
        raise HTTPException(status_code=500, detail="Model not loaded in memory.")

    async def event_stream():
        def _ndjson(obj):
            return json_module.dumps(obj) + "\n"

        try:
            # ── Stage 1: Hybrid retrieval (blocking → asyncio.to_thread) ──
            logging.info("Stage 1: Hybrid Search...")
            yield _ndjson({"type": "progress", "stage": "retrieve_start"})
            broad_results = await asyncio.to_thread(
                lambda: pipeline.retriever.search(request.prompt, k=50)
            )
            yield _ndjson({"type": "progress", "stage": "retrieve_done"})

            if not broad_results:
                yield _ndjson({"type": "error", "data": "No documents found in the database."})
                return

            # ── Stage 2: Rerank (blocking → asyncio.to_thread) ───────────
            logging.info("Stage 2: Reranking...")
            yield _ndjson({"type": "progress", "stage": "rerank_start"})
            top_docs = await asyncio.to_thread(
                lambda: pipeline.reranker.rerank(request.prompt, broad_results, top_k=7)
            )
            yield _ndjson({"type": "progress", "stage": "rerank_done"})

            # Send enriched context metadata immediately (id + title + preview + text)
            title_map = ml_models.get("title_map", {})
            context_metas = []
            for i, doc in enumerate(top_docs):
                if isinstance(doc, dict):
                    doc_id = doc.get("doc_id", doc.get("id", f"doc_{i + 1}"))
                    text   = doc.get("text", "")
                else:
                    doc_id = f"doc_{i + 1}"
                    text   = str(doc)
                title = title_map.get(doc_id, doc_id)
                first_line = text.split(".")[0].split("\n")[0].strip()
                preview = (first_line[:120] + "\u2026") if len(first_line) > 120 else first_line
                context_metas.append({"id": doc_id, "title": title, "text": text, "preview": preview})
            yield _ndjson({"type": "contexts", "data": context_metas})

            # ── CRAG gate ────────────────────────────────────────────────
            best_score = max(d.get("rerank_score", 0.0) for d in top_docs)
            logging.info("CRAG check — best logit: %.4f (threshold: %.4f)", best_score, pipeline.CRAG_THRESHOLD)
            if best_score < pipeline.CRAG_THRESHOLD:
                logging.warning("CRAG gate triggered.")
                yield _ndjson({"type": "token", "data": "The retrieved documents do not contain enough information to answer this question reliably."})
                return

            # ── Stage 3: Stream from Ollama / GPU ────────────────────────
            logging.info("Stage 3: Streaming generation...")
            yield _ndjson({"type": "progress", "stage": "generate_start"})
            gen = pipeline.generator
            full_prompt = gen._build_prompt(request.prompt, top_docs)

            # ── Inject conversation history before the user query ────────
            if request.history:
                logging.info("HISTORY: Injecting %d entries into prompt.", len(request.history))
                history_lines = [
                    f"[{t.get('role','user').capitalize()}]: {t.get('content','')[:500]}"
                    for t in request.history[-12:]
                ]
                history_section = (
                    "<Conversation History>\n"
                    + "\n\n".join(history_lines)
                    + "\n</Conversation History>\n\n"
                )
                full_prompt = full_prompt.replace(
                    "<User Query>", history_section + "<User Query>", 1
                )
            else:
                logging.info("HISTORY: No conversation history received.")

            if gen.backend == "ollama":
                payload = {
                    "model": gen.model_name,
                    "prompt": full_prompt,
                    "stream": True,
                    "options": {"temperature": 0.1},
                }
                ollama_timeout = httpx.Timeout(
                    connect=30.0, read=900.0, write=30.0, pool=30.0
                )
                print("\n" + "=" * 40 + " LLM OUTPUT (Streaming) " + "=" * 40 + "\n")
                async with httpx.AsyncClient(timeout=ollama_timeout) as client:
                    async with client.stream("POST", gen.api_url, json=payload) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if line:
                                body = json_module.loads(line)
                                token = body.get("response", "")
                                if token:
                                    sys.stdout.write(token)
                                    sys.stdout.flush()
                                    yield _ndjson({"type": "token", "data": token})
                                if body.get("done"):
                                    print("\n\n" + "=" * 92 + "\n")
                                    break
            else:
                # Transformers (GPU) backend: full generation then word-by-word
                answer = await asyncio.to_thread(
                    gen.generate_answer, request.prompt, top_docs
                )
                for word in answer.split(" "):
                    yield _ndjson({"type": "token", "data": word + " "})

        except Exception as e:
            logging.error(f"Streaming pipeline error: {e}")
            yield _ndjson({"type": "error", "data": str(e)})

    return StreamingResponse(event_stream(), media_type="text/plain")