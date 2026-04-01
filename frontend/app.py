import chainlit as cl
import httpx
import json
import logging
import os
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Use environment variable for Docker compatibility, default to localhost for local testing
BACKEND_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080/api/chat")
STREAM_URL = BACKEND_URL.rstrip("/").replace("/api/chat", "/api/chat/stream")

# Look-ahead buffer for tag detection (must be >= longest tag: "</Final Answer>" = 15)
_TAG_LOOKAHEAD = 20


def _parse_final_answer(full_text: str) -> str:
    """Extracts <Final Answer> block; falls back to full text when tags are absent."""
    match = re.search(
        r"<Final Answer>(.*?)(?:</Final Answer>|$)", full_text, re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else full_text.strip()


# ── Session init: seed conversation history ───────────────────────────────────
@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(
        content=(
            "Welcome to the **Scientific RAG Chatboard Assistant**!\n\n"
            "Ask me anything about NLP research. "
        )
    ).send()


# ── Copy-answer action callback ───────────────────────────────────────────────
@cl.action_callback("copy_answer")
async def on_copy_answer(action: cl.Action):
    text = (
        action.payload.get("text", "")
        if isinstance(action.payload, dict)
        else str(action.payload)
    )
    await cl.Message(content=f"```\n{text}\n```", author="Copied answer").send()


@cl.on_message
async def main(message: cl.Message):
    """
    Rendering order
    ───────────────
    1. 🔎 Retrieving step   (progress, collapses when done)
    2. ⚖️  Reranking step    (progress, collapses when done)
    3. 🔍 Thinking step     (streams live token-by-token, collapses at </Reasoning>,
                              click to re-expand full reasoning)
    4. Final answer message (created lazily so it always appears AFTER the above steps)
    5. References message (one chip per source document, title visible, content hidden)

    Multi-turn memory
    ─────────────────
    The last 6 message pairs (12 entries) are sent to the backend with every request.
    The LLM prompt carries a <Conversation History> block so follow-up questions work.
    """
    # ── Feature 1: load session history and include in request ────────────────
    history: list = cl.user_session.get("history") or []
    logging.info("HISTORY SEND: %d entries to backend", len(history))

    full_response = ""
    contexts: list = []
    answer_tokens = ""  # tracks text streamed into the answer message

    # Progress step handles
    retrieve_step: cl.Step | None = None
    rerank_step:   cl.Step | None = None

    # Reasoning step (opened on <Reasoning> tag, closed on </Reasoning>)
    reasoning_step: cl.Step | None = None

    # Answer message — created lazily so it appears AFTER the thinking steps
    msg: cl.Message | None = None

    # ── Helper: create answer message on first need ───────────────────────────
    async def ensure_msg() -> cl.Message:
        nonlocal msg
        if msg is None:
            logging.info("MSG CREATED in section=%s", section)
            msg = cl.Message(content="")
            await msg.send()
        return msg

    # ── Tag-scanning state machine ─────────────────────────────────────────────
    # States:  pre → reasoning → between → answer → done
    section = "pre"
    buf = ""

    # ── Stream from backend ───────────────────────────────────────────────────
    # Use separate read timeout (None = no limit) so long reranking/generation
    # doesn't kill the connection.  aiter_bytes + manual line split avoids
    # internal buffering issues that aiter_lines() can have.
    timeout = httpx.Timeout(connect=30.0, read=None, write=30.0, pool=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            async with client.stream(
                "POST",
                STREAM_URL,
                json={"prompt": message.content, "history": history},
            ) as response:
                response.raise_for_status()
                logging.info("STREAM connected, status=%s", response.status_code)

                line_buf = ""
                async for chunk in response.aiter_bytes():
                    line_buf += chunk.decode("utf-8", errors="replace")
                    while "\n" in line_buf:
                        line, line_buf = line_buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except json.JSONDecodeError:
                            logging.warning("Skipping malformed JSON line: %s", line[:100])
                            continue

                        # ── Feature 2: pipeline progress steps ───────────
                        if data["type"] == "progress":
                            stage = data["stage"]

                            if stage == "retrieve_start":
                                retrieve_step = cl.Step(
                                    name="Retrieving 50 candidates…",
                                    type="tool",
                                    show_input=False,
                                )
                                await retrieve_step.send()

                            elif stage == "retrieve_done" and retrieve_step:
                                retrieve_step.output = "Done"
                                await retrieve_step.update()

                            elif stage == "rerank_start":
                                rerank_step = cl.Step(
                                    name="⚖️ Reranking to top 7…",
                                    type="tool",
                                    show_input=False,
                                )
                                await rerank_step.send()

                            elif stage == "rerank_done" and rerank_step:
                                rerank_step.output = "Done"
                                await rerank_step.update()

                        # ── Contexts metadata ─────────────────────────────
                        elif data["type"] == "contexts":
                            contexts = data["data"]

                        # ── Raw LLM token ─────────────────────────────────
                        elif data["type"] == "token":
                            token = data["data"]
                            full_response += token
                            buf += token

                            # ── PRE: scan for <Reasoning> ─────────────────
                            if section == "pre":
                                lower = buf.lower()
                                if "<reasoning>" in lower:
                                    idx = lower.index("<reasoning>") + len("<reasoning>")
                                    buf = buf[idx:]
                                    section = "reasoning"
                                    reasoning_step = cl.Step(
                                        name="Thinking…",
                                        type="tool",
                                        show_input=False,
                                    )
                                    await reasoning_step.send()
                                elif len(buf) > 200:
                                    section = "answer"

                            # ── REASONING: stream live, close on </Reasoning>
                            elif section == "reasoning":
                                lower = buf.lower()
                                if "</reasoning>" in lower:
                                    idx = lower.index("</reasoning>")
                                    if buf[:idx]:
                                        await reasoning_step.stream_token(buf[:idx])
                                    buf = buf[idx + len("</reasoning>"):]
                                    await reasoning_step.update()
                                    reasoning_step = None
                                    section = "between"
                                else:
                                    safe = max(0, len(buf) - _TAG_LOOKAHEAD)
                                    if safe:
                                        await reasoning_step.stream_token(buf[:safe])
                                        buf = buf[safe:]

                            # ── BETWEEN: watch for <Final Answer> ─────────
                            elif section == "between":
                                lower = buf.lower()
                                if "<final answer>" in lower:
                                    idx = lower.index("<final answer>") + len("<final answer>")
                                    buf = buf[idx:]
                                    section = "answer"

                            # ── ANSWER: stream into answer message ────────
                            elif section == "answer":
                                lower = buf.lower()
                                if "</final answer>" in lower:
                                    idx = lower.index("</final answer>")
                                    if buf[:idx]:
                                        m = await ensure_msg()
                                        await m.stream_token(buf[:idx])
                                        answer_tokens += buf[:idx]
                                    buf = ""
                                    section = "done"
                                else:
                                    safe = max(0, len(buf) - _TAG_LOOKAHEAD)
                                    if safe:
                                        m = await ensure_msg()
                                        await m.stream_token(buf[:safe])
                                        answer_tokens += buf[:safe]
                                        buf = buf[safe:]

                        # ── Backend error ─────────────────────────────────
                        elif data["type"] == "error":
                            m = await ensure_msg()
                            await m.stream_token(f"\n\n {data['data']}")

        except httpx.HTTPError as e:
            logging.error("HTTP error during streaming: %s", e)
            await cl.Message(content=f"**Backend Connection Error:** {e}").send()
            return
        except Exception as e:
            logging.error("Unexpected error during streaming: %s", e, exc_info=True)
            await cl.Message(content=f"**Error:** {e}").send()
            return

    # ── Post-stream cleanup ───────────────────────────────────────────────────
    if buf.strip():
        if section == "reasoning" and reasoning_step:
            await reasoning_step.stream_token(buf)
            await reasoning_step.update()
        else:
            m = await ensure_msg()
            await m.stream_token(buf)
            if section == "answer":
                answer_tokens += buf

    # Close any steps that were never explicitly finalised
    for step in (reasoning_step, retrieve_step, rerank_step):
        if step:
            await step.update()

    # Ensure the answer message exists even for flat (no-tag) CRAG replies
    await ensure_msg()

    # ── Parse clean final answer ──────────────────────────────────────────────
    final_answer = _parse_final_answer(full_response)
    logging.info("STATE final section=%s | full_response len=%d | parsed len=%d | answer_tokens len=%d",
                 section, len(full_response), len(final_answer), len(answer_tokens))

    # Fallback: if parsing produced empty text, use the streamed answer tokens
    if not final_answer.strip() and answer_tokens.strip():
        logging.warning("Parsed final_answer was empty — falling back to streamed answer_tokens")
        final_answer = answer_tokens.strip()
    # Second fallback: use everything after reasoning tags
    if not final_answer.strip() and full_response.strip():
        logging.warning("Still empty — falling back to full_response (stripped of tags)")
        cleaned = re.sub(r"</?(?:Reasoning|Final Answer)>", "", full_response, flags=re.IGNORECASE)
        final_answer = cleaned.strip()

    # ── Copy-answer action (Feature 5) ────────────────────────────────────────
    copy_action = cl.Action(
        name="copy_answer",
        label="Copy answer",
        payload={"text": final_answer},
        description="Copy the answer text",
    )

    # Replace streamed content with cleanly parsed answer + copy button
    msg.content = final_answer
    msg.actions = [copy_action]
    await msg.update()

    # ── Feature 1: save this exchange to session history (BEFORE references) ──
    # Save early so that even if the references section fails, multi-turn
    # memory is preserved for the next message.
    history.append({"role": "user",      "content": message.content})
    history.append({"role": "assistant", "content": final_answer})
    # Keep last 6 pairs (12 entries) to avoid bloating the prompt
    cl.user_session.set("history", history[-12:])
    logging.info("HISTORY SAVE: now %d entries stored", len(history[-12:]))

    # ── References section (Feature 4) ────────────────────────────────────────
    # Only show documents actually cited in the answer — parse [Doc N] references.
    # Each title is the cl.Text element name so Chainlit auto-links it as clickable.
    if contexts:
        # Extract cited doc numbers from the LLM answer (1-based: [Doc 1], [Doc 2], …)
        cited_nums = sorted(set(
            int(m) for m in re.findall(r"\[Doc\s*(\d+)\]", final_answer)
        ))

        # Fall back to all docs if no [Doc N] citations found (e.g. flat CRAG reply)
        if not cited_nums:
            cited_nums = list(range(1, len(contexts) + 1))

        ref_elements = []
        ref_lines = []
        seen_titles: dict[str, int] = {}  # deduplicate titles across chunks

        ref_idx = 0
        for doc_num in cited_nums:
            i = doc_num - 1  # 0-based index into contexts list
            if i < 0 or i >= len(contexts):
                continue
            ctx_item = contexts[i]

            if isinstance(ctx_item, dict):
                doc_id = ctx_item.get("id", f"doc_{i + 1}")
                title  = ctx_item.get("title", doc_id)
                text   = ctx_item.get("text", "")
            else:
                doc_id = f"doc_{i + 1}"
                title  = doc_id
                text   = str(ctx_item)

            # Skip duplicate titles (same paper, different chunks)
            if title in seen_titles:
                continue
            seen_titles[title] = ref_idx
            ref_idx += 1

            # In Chainlit, cl.Text(display="side") becomes clickable when the
            # element's *name* appears literally in the message content.
            # Use the title as name so users can click it.
            ref_elements.append(
                cl.Text(
                    name=title,
                    content=(
                        f"### {title}\n\n"
                        f"**Paper ID:** {doc_id}\n\n"
                        f"---\n\n"
                        f"{text}"
                    ),
                    display="side",
                )
            )
            ref_lines.append(f"**[{ref_idx}]** {title}")

        if ref_lines:
            ref_body = "\n\n".join(ref_lines)
            await cl.Message(
                content=(
                    "**References** — click a title to read the supporting content"
                    f"\n\n{ref_body}"
                ),
                elements=ref_elements,
            ).send()
