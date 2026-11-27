"""Smoke test for the Session 3 agent (live: calls Gemini, OpenAI embeddings, Pinecone).

Run from the repo root (any working directory works):
    python ai-engineering-bootcamp-v2/week-3/agent/smoke_test.py

Cases:
  0. search gate, offline (stubbed turns, no API calls)
  1. multi-part question              -> cited answer
  2. follow-up in the same session    -> cited answer (needs conversation memory)
     follow-up repeated 5x            -> must search every time
  3. out-of-scope question            -> refusal
  4. hard limit (max_llm_calls=1)     -> fail-closed refusal
Content checks assume the Northwind policy manual is the indexed document.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # week-3/, so `agent` is a package

from agent.runtime import AgentResult, AgentSession  # noqa: E402

CHUNK = "northwind-support-policy-manual::chunk_{}"
FOLLOW_UP_RUNS = 5


def show(label: str, question: str, r: AgentResult) -> None:
    print(f"\n{'=' * 78}\n{label}\nQ: {question}")
    print(f"status={r.status}  tool_calls={r.tool_calls}  model_responses={r.llm_calls}  cited={r.cited_ids}")
    for i, step in enumerate(r.trace, 1):
        kind = step["kind"].upper()
        if step["kind"] == "act":
            print(f"  {i:>2}. ACT      {step['tool']}({step['args']})")
        elif step["kind"] == "observe":
            hits = ", ".join(f"{h['chunk_id'].split('::')[-1]}@{h['score']}" for h in step["hits"][:5])
            print(f"  {i:>2}. OBSERVE  status={step['status']} hits=[{hits}]")
        elif step["kind"] in ("think", "guard"):
            text = step["text"].replace("\n", " ")
            print(f"  {i:>2}. {kind:<8} {text[:160]}{'...' if len(text) > 160 else ''}")
    print(f"A: {r.answer}")
    if r.detail and r.status != "ok":
        print(f"(detail: {r.detail[:300]})")


def main() -> int:
    failures = []

    def check(name: str, ok: bool) -> None:
        print(f"-> {'PASS' if ok else 'FAIL'}: {name}")
        if not ok:
            failures.append(name)

    # 0) Search gate, offline and deterministic: stub the model turns so the retry
    #    path is exercised even when the live model happens to search on its own.
    gate = AgentSession()
    c3 = CHUNK.format(3)

    def stub(script):
        calls = iter(script)

        async def fake_run_turn(message, max_llm_calls, max_tool_calls):
            budgets.append(max_llm_calls)
            kind = next(calls)
            if kind == "unsearched":
                gate.turn_chunk_ids = set()
                return AgentResult(answer=f"Tier 3 handles it [{c3}].", status="ok", llm_calls=1,
                                   trace=[{"kind": "answer", "text": "..."}])
            if kind == "stale":  # searched, but cites a chunk this turn did not retrieve
                gate.turn_chunk_ids = {CHUNK.format(9)}
                return AgentResult(answer=f"Tier 3 handles it [{c3}].", status="ok", tool_calls=1,
                                   llm_calls=2, trace=[{"kind": "act", "tool": "s", "args": {}}])
            gate.turn_chunk_ids = {c3}
            return AgentResult(answer=f"Tier 3 handles it [{c3}].", status="ok", tool_calls=1,
                               llm_calls=2, trace=[{"kind": "act", "tool": "s", "args": {}}])
        gate._run_turn = fake_run_turn

    budgets: list[int] = []
    stub(["unsearched", "searched"])
    g1 = gate.ask("x")
    check("gate: unsearched answer is retried, then accepted",
          g1.retried and g1.status == "ok" and g1.tool_calls == 1)
    check("gate: retry only gets the remaining LLM budget",
          budgets == [gate.max_llm_calls, gate.max_llm_calls - 1])
    stub(["unsearched", "unsearched"])
    g2 = gate.ask("x")
    check("gate: still unsearched after retry -> fail closed", g2.status == "no_search")
    stub(["stale", "searched"])
    g3 = gate.ask("x")
    check("gate: stale citation is retried, then accepted", g3.retried and g3.status == "ok")
    stub(["stale", "stale"])
    g4 = gate.ask("x")
    check("gate: still stale after retry -> fail closed", g4.status == "invalid_citation")

    session = AgentSession()

    q1 = ("Until when can I return a purchase made during the holiday season, "
          "and who has to approve a refund of $600?")
    r1 = session.ask(q1)
    show("1) Multi-part", q1, r1)
    check("multi-part answered with citations", r1.status == "ok" and len(r1.cited_ids) >= 1)
    # Content checks for the Northwind manual: holiday window is in chunk_7,
    # refund approval >$500 (Tier 2) and order value >$999 (Tier 3) are in chunk_3.
    check("multi-part cites the right evidence (chunk_7 + chunk_3)",
          {CHUNK.format(7), CHUNK.format(3)} <= set(r1.cited_ids))

    q2 = "And who handles it if the order value is above $1,000 instead?"
    r2 = session.ask(q2)
    show("2) Follow-up (same session)", q2, r2)
    check("follow-up answered with citations", r2.status == "ok" and len(r2.cited_ids) >= 1)
    check("follow-up cites chunk_3 and names Tier 3",
          CHUNK.format(3) in r2.cited_ids and "Tier 3" in r2.answer)

    # Regression for the UI bug: the follow-up was answered from the previous turn
    # without searching. Ask it 5x in a row in the same session (hardest case: the
    # answer is already in the history every time) and require a search each time.
    q_fu = "And what if the refund is over $999?"
    print(f"\n{'=' * 78}\nFollow-up x{FOLLOW_UP_RUNS} (same session as 1+2)\nQ: {q_fu}")
    rows = []
    for i in range(1, FOLLOW_UP_RUNS + 1):
        r = session.ask(q_fu)
        queries = [s["args"].get("query") for s in r.trace if s["kind"] == "act"]
        searched = r.tool_calls >= 1
        rows.append((i, r, searched))
        print(f"  run {i}: status={r.status:<9} tool_calls={r.tool_calls} retried={r.retried} "
              f"cited={[c.split('::')[-1] for c in r.cited_ids]} queries={queries}")
        print(f"         A: {r.answer[:220]}")
    check(f"follow-up searched in every run ({FOLLOW_UP_RUNS}/{FOLLOW_UP_RUNS})",
          all(s for _, _, s in rows))
    check("follow-up answered with chunk_3 in every run",
          all(r.status == "ok" and CHUNK.format(3) in r.cited_ids for _, r, _ in rows))

    q3 = "What was Northwind's total revenue in 2025?"
    r3 = session.ask(q3)
    show("3) Out of scope", q3, r3)
    check("out-of-scope question refused", r3.status in ("refused", "uncited"))

    limited = AgentSession(max_llm_calls=1)
    r4 = limited.ask(q1)
    show("4) Hard limit (max_llm_calls=1)", q1, r4)
    check("limit fails closed", r4.status in ("limit_llm", "limit_tools"))

    total = 13
    print(f"\n{'=' * 78}\n{total - len(failures)}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
