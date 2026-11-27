"""Runs the agent with hard limits and a citation guard. Shared by the UI and smoke test.

Hard limits (enforced in code, not only in the prompt):
- MAX_LLM_CALLS per user turn via ADK RunConfig(max_llm_calls=...).
  ADK raises LlmCallsLimitExceededError when exceeded.
- MAX_TOOL_CALLS per user turn: counted here; the run is aborted before the
  extra tool executes.
Both fail closed: the user gets a fixed refusal, never a partial answer.

Search gate: if a turn yields a substantive answer (not a refusal) without any
search_knowledge_base call, or citing chunk_ids no search in this turn returned,
the draft is discarded and the agent is retried once with a reminder. The retry
gets only the turn's remaining LLM/tool budget. If it still fails, the answer is
withheld ("no_search", or "invalid_citation" via the citation guard).

Citation guard: a final answer must either be a refusal or cite at least one
chunk_id, and every cited chunk_id must have been returned by the tool in this
same turn (so follow-ups must search again instead of reusing old answers).
Otherwise the answer is replaced by a refusal (fail closed). The guard checks
provenance, not whether a chunk semantically supports the sentence.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk.agents import RunConfig
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from .agent import REFUSAL_PREFIX, root_agent

MAX_LLM_CALLS = 8
MAX_TOOL_CALLS = 10
TURN_TIMEOUT_S = 120

APP_NAME = "kb_research_agent"
CITATION_RE = re.compile(r"\[([^\[\]\s]+::chunk_\d+)\]")

LIMIT_REFUSAL = (
    f"{REFUSAL_PREFIX}: the agent hit its step limit before finding sufficient evidence. "
    "Try a narrower question."
)
GUARD_REFUSAL = (
    f"{REFUSAL_PREFIX}: the drafted answer was not backed by valid citations, so it was withheld."
)
ERROR_REFUSAL = f"{REFUSAL_PREFIX}: the agent failed with an internal error."
NO_SEARCH_REFUSAL = (
    f"{REFUSAL_PREFIX}: the agent answered without searching the knowledge base, "
    "so the answer was withheld."
)

# Sent as a follow-up message in the same session when a turn produced a
# substantive answer without any tool call.
RETRY_NUDGE = (
    "[Runtime check] Your previous reply was rejected because you did not call "
    "search_knowledge_base in this turn. You must call search_knowledge_base first. "
    "Rewrite the user's last question into a standalone search query (include the "
    "context from the conversation), search, and answer citing ONLY chunk_ids returned "
    "by searches made now. Never cite chunk_ids from earlier turns.\n"
    "User's last question: {question}"
)
STALE_CITATION_NUDGE = (
    "[Runtime check] Your previous reply was rejected because it cited chunk_ids that "
    "no search in this turn returned: {ids}. Search again now with refined terms "
    "(different wording, synonyms, section or policy terms) so that the supporting "
    "chunk is actually retrieved, and cite only chunk_ids returned by searches made "
    "now. If the evidence is still not retrieved, say that part is not covered.\n"
    "User's last question: {question}"
)

_SECRET_VARS = ("GOOGLE_API_KEY", "OPENAI_API_KEY", "PINECONE_API_KEY", "INGEST_DEMO_TOKEN")


def _redact(text: str) -> str:
    for name in _SECRET_VARS:
        value = (os.getenv(name) or "").strip()
        if value:
            text = text.replace(value, f"<{name}>")
    return text


@dataclass
class AgentResult:
    answer: str
    status: str  # ok | refused | limit_llm | limit_tools | uncited | invalid_citation | timeout | error
    trace: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    llm_calls: int = 0
    cited_ids: list[str] = field(default_factory=list)
    detail: str | None = None
    retried: bool = False  # True if the search gate forced a second attempt


class _LoopThread:
    """One persistent event loop in a background thread.

    Streamlit reruns the script synchronously; keeping one loop per conversation
    avoids clients that are bound to a closed loop on the second turn.
    """

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro, timeout: float):
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            raise


class AgentSession:
    """One conversation: keeps ADK session history so follow-up questions work."""

    def __init__(
        self,
        max_llm_calls: int = MAX_LLM_CALLS,
        max_tool_calls: int = MAX_TOOL_CALLS,
    ) -> None:
        self.max_llm_calls = max_llm_calls
        self.max_tool_calls = max_tool_calls
        self.user_id = "local-user"
        self._loop = _LoopThread()
        self._service = InMemorySessionService()
        self._runner = Runner(agent=root_agent, app_name=APP_NAME, session_service=self._service)
        session = self._loop.run(
            self._service.create_session(app_name=APP_NAME, user_id=self.user_id,
                                         session_id=uuid.uuid4().hex),
            timeout=30,
        )
        self.session_id = session.id
        self.turn_chunk_ids: set[str] = set()  # chunks the tool returned in the current turn

    # ------------------------------------------------------------------ public
    def ask(self, question: str, timeout: float = TURN_TIMEOUT_S) -> AgentResult:
        deadline = time.monotonic() + timeout
        try:
            first = self._loop.run(
                self._run_turn(question, self.max_llm_calls, self.max_tool_calls), timeout=timeout
            )
            reason, stale = self._retry_reason(first)
            if reason is None:
                return self._apply_citation_guard(first)

            # Search gate: a draft without a search, or citing chunks not retrieved in
            # this turn, is never shown. Retry once with a nudge; shares the turn's budget.
            if reason == "no_search":
                note = "Answer without search_knowledge_base in this turn"
                nudge = RETRY_NUDGE.format(question=question)
            else:
                note = f"Draft cited chunks not retrieved in this turn ({', '.join(stale)})"
                nudge = STALE_CITATION_NUDGE.format(ids=", ".join(stale), question=question)
            first.trace.append({"kind": "guard",
                                "text": f"{note}; draft discarded, retrying once."})
            llm_left = self.max_llm_calls - first.llm_calls
            tools_left = self.max_tool_calls - first.tool_calls
            if llm_left <= 0:
                return self._merge(first, AgentResult(answer=LIMIT_REFUSAL, status="limit_llm"))
            second = self._loop.run(
                self._run_turn(nudge, llm_left, tools_left),
                timeout=max(1.0, deadline - time.monotonic()),
            )
            result = self._merge(first, second)
            if self._is_unsearched_answer(second):
                result.trace.append({"kind": "guard",
                                     "text": "Still no search after retry; answer withheld."})
                result.status, result.detail, result.answer = "no_search", second.answer, NO_SEARCH_REFUSAL
                return result
            return self._apply_citation_guard(result)
        except concurrent.futures.TimeoutError:
            return AgentResult(answer=LIMIT_REFUSAL, status="timeout",
                               detail=f"Turn exceeded {timeout}s.")

    # ---------------------------------------------------------------- internal
    @staticmethod
    def _is_unsearched_answer(r: AgentResult) -> bool:
        """A substantive answer (not a refusal) produced without any tool call."""

        return r.status == "ok" and r.tool_calls == 0 and not r.answer.startswith(REFUSAL_PREFIX)

    def _retry_reason(self, r: AgentResult) -> tuple[str | None, list[str]]:
        """Why a first draft gets one retry: no search, or citations not retrieved this turn."""

        if self._is_unsearched_answer(r):
            return "no_search", []
        if r.status == "ok" and not r.answer.startswith(REFUSAL_PREFIX):
            stale = [c for c in dict.fromkeys(CITATION_RE.findall(r.answer))
                     if c not in self.turn_chunk_ids]
            if stale:
                return "stale_citation", stale
        return None, []

    @staticmethod
    def _merge(first: AgentResult, second: AgentResult) -> AgentResult:
        """Combine both attempts of one turn: trace and counters add up, outcome is the second's."""

        second.trace = first.trace + second.trace
        second.tool_calls += first.tool_calls
        second.llm_calls += first.llm_calls
        second.retried = True
        return second

    async def _run_turn(self, message: str, max_llm_calls: int, max_tool_calls: int) -> AgentResult:
        self.turn_chunk_ids = set()
        trace: list[dict[str, Any]] = []
        tool_calls = 0
        llm_calls = 0
        final_text = ""
        content = types.Content(role="user", parts=[types.Part(text=message)])
        run_config = RunConfig(max_llm_calls=max_llm_calls)

        events = self._runner.run_async(
            user_id=self.user_id, session_id=self.session_id,
            new_message=content, run_config=run_config,
        )
        try:
            async for event in events:
                if not (event.content and event.content.parts):
                    continue
                if event.content.role == "model":
                    llm_calls += 1
                is_final = event.is_final_response()
                for part in event.content.parts:
                    if part.function_call:
                        tool_calls += 1
                        args = dict(part.function_call.args or {})
                        trace.append({"kind": "act", "tool": part.function_call.name, "args": args})
                        if tool_calls > max_tool_calls:
                            # Abort before the tool runs: closing the generator stops ADK here.
                            trace.append({"kind": "guard",
                                          "text": f"Tool-call limit {self.max_tool_calls} exceeded; run aborted."})
                            return AgentResult(answer=LIMIT_REFUSAL, status="limit_tools", trace=trace,
                                               tool_calls=tool_calls, llm_calls=llm_calls)
                    elif part.function_response:
                        trace.append(self._observe(part.function_response))
                    elif part.text and part.text.strip():
                        if getattr(part, "thought", False):
                            trace.append({"kind": "think", "text": part.text.strip()})
                        elif is_final:
                            final_text += part.text
                        else:
                            # Visible model text before a tool call is its reasoning.
                            trace.append({"kind": "think", "text": part.text.strip()})
        except LlmCallsLimitExceededError as exc:
            trace.append({"kind": "guard", "text": f"LLM-call limit hit: {exc}"})
            return AgentResult(answer=LIMIT_REFUSAL, status="limit_llm", trace=trace,
                               tool_calls=tool_calls, llm_calls=llm_calls)
        except Exception as exc:  # fail closed on anything unexpected
            msg = _redact(f"{type(exc).__name__}: {exc}")[:300]
            trace.append({"kind": "guard", "text": f"Error: {msg}"})
            return AgentResult(answer=ERROR_REFUSAL, status="error", trace=trace,
                               tool_calls=tool_calls, llm_calls=llm_calls, detail=msg)
        finally:
            await events.aclose()

        final_text = final_text.strip()
        trace.append({"kind": "answer", "text": final_text})
        return AgentResult(answer=final_text, status="ok", trace=trace,
                           tool_calls=tool_calls, llm_calls=llm_calls)

    def _observe(self, fr) -> dict[str, Any]:
        resp = fr.response or {}
        results = resp.get("results") or []
        ids = [r.get("chunk_id") for r in results if r.get("chunk_id")]
        self.turn_chunk_ids.update(ids)
        return {
            "kind": "observe",
            "tool": fr.name,
            "status": resp.get("status"),
            "error": resp.get("error"),
            "hits": [{"chunk_id": r.get("chunk_id"), "score": r.get("score"),
                      "preview": (r.get("text") or "")[:160]} for r in results],
        }

    def _apply_citation_guard(self, result: AgentResult) -> AgentResult:
        if result.status != "ok":
            return result
        answer = result.answer
        if not answer:
            result.status, result.answer = "error", ERROR_REFUSAL
            result.trace.append({"kind": "guard", "text": "Empty final answer; withheld."})
            return result
        if answer.startswith(REFUSAL_PREFIX):
            result.status = "refused"
            return result

        cited = list(dict.fromkeys(CITATION_RE.findall(answer)))
        result.cited_ids = cited
        unknown = [c for c in cited if c not in self.turn_chunk_ids]
        if not cited:
            result.status, result.detail = "uncited", answer
            result.answer = GUARD_REFUSAL
            result.trace.append({"kind": "guard", "text": "Answer had no citations; withheld."})
        elif unknown:
            result.status, result.detail = "invalid_citation", answer
            result.answer = GUARD_REFUSAL
            result.trace.append({"kind": "guard",
                                 "text": f"Cited IDs not retrieved in this turn: {', '.join(unknown)}; withheld."})
        return result
