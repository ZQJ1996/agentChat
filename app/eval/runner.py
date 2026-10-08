"""Evaluation runner: accuracy, refuse rate, ticket completion, recall@k."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.graph import run_agent
from app.observability import new_trace_id
from app.rag import get_knowledge_store


@dataclass
class EvalCaseResult:
    question: str
    intent_expected: str
    intent_pred: str
    intent_ok: bool
    answer: str
    answer_ok: bool
    refuse_ok: bool
    ticket_ok: bool
    human_ok: bool
    recall_at_k: float | None
    route_path: list[str]


@dataclass
class EvalReport:
    total: int
    intent_accuracy: float
    answer_accuracy: float
    refuse_rational_rate: float
    ticket_completion_rate: float
    avg_recall_at_k: float | None
    cases: list[EvalCaseResult]
    created_at: str


def load_golden_set(path: Path | None = None) -> list[dict[str, Any]]:
    settings = get_settings()
    path = path or settings.golden_set_file
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _normalize(text: str) -> str:
    import re

    text = text.lower().replace("–", "-").replace("—", "-")
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"\s+", "", text)
    return text


def _contains_expected(answer: str, expected: str) -> bool:
    if not expected:
        return True
    answer_n = _normalize(answer)
    expected_n = _normalize(expected)
    parts = [p for p in expected_n.replace(",", " ").split() if p]
    # after normalize spaces are gone; split by common separators
    if " " not in expected.strip() and len(expected.split()) <= 1:
        return expected_n in answer_n
    # original expected may be space-separated alternatives/requirements
    raw_parts = [ _normalize(p) for p in expected.replace("–", "-").split() if p]
    if len(raw_parts) <= 1:
        return expected_n in answer_n
    return sum(1 for p in raw_parts if p in answer_n) >= max(1, len(raw_parts) // 2)


def evaluate_case(case: dict[str, Any], idx: int) -> EvalCaseResult:
    question = case["question"]
    result = run_agent(
        query=question,
        thread_id=f"eval-{idx}",
        trace_id=new_trace_id(),
        user_id="eval_user",
        role="agent",
    )
    answer = result.get("final_answer") or result.get("agent_answer") or ""
    intent_pred = result.get("intent") or ""
    intent_expected = case.get("intent") or ""
    intent_ok = intent_pred == intent_expected

    expected_answer = case.get("expected_answer") or ""
    allow_refuse = bool(case.get("allow_refuse"))
    needs_human = bool(result.get("needs_human"))
    refused = needs_human or any(k in answer for k in ("证据不足", "无法确定", "转人工", "无法可靠"))

    if case.get("expect_human"):
        answer_ok = refused or needs_human
        refuse_ok = True
    elif refused and allow_refuse:
        answer_ok = True
        refuse_ok = True
    elif refused and not allow_refuse:
        answer_ok = False
        refuse_ok = False
    else:
        answer_ok = _contains_expected(answer, expected_answer)
        refuse_ok = True

    expect_ticket = bool(case.get("expect_ticket"))
    ticket_ok = (not expect_ticket) or bool(result.get("ticket_id")) or ("工单 #" in answer) or ("工单#" in answer)

    recall = None
    expected_sources = case.get("expected_sources") or []
    if expected_sources:
        evidence = result.get("evidence") or []
        got = {e.get("source") for e in evidence}
        hit = len(got & set(expected_sources))
        recall = hit / len(expected_sources)

    return EvalCaseResult(
        question=question,
        intent_expected=intent_expected,
        intent_pred=intent_pred,
        intent_ok=intent_ok,
        answer=answer,
        answer_ok=answer_ok,
        refuse_ok=refuse_ok,
        ticket_ok=ticket_ok,
        human_ok=(not case.get("expect_human")) or needs_human or refused,
        recall_at_k=recall,
        route_path=list(result.get("route_path") or []),
    )


def run_evaluation(path: Path | None = None) -> EvalReport:
    cases = load_golden_set(path)
    results = [evaluate_case(c, i) for i, c in enumerate(cases)]
    total = len(results) or 1
    intent_acc = sum(1 for r in results if r.intent_ok) / total
    answer_acc = sum(1 for r in results if r.answer_ok) / total
    refuse_cases = [r for r, c in zip(results, cases) if c.get("allow_refuse") or c.get("expect_human")]
    refuse_rate = (
        sum(1 for r in refuse_cases if r.refuse_ok and r.human_ok) / len(refuse_cases)
        if refuse_cases
        else 1.0
    )
    ticket_cases = [r for r, c in zip(results, cases) if c.get("expect_ticket")]
    ticket_rate = (
        sum(1 for r in ticket_cases if r.ticket_ok) / len(ticket_cases) if ticket_cases else 1.0
    )
    recalls = [r.recall_at_k for r in results if r.recall_at_k is not None]
    avg_recall = sum(recalls) / len(recalls) if recalls else None
    return EvalReport(
        total=len(results),
        intent_accuracy=round(intent_acc, 4),
        answer_accuracy=round(answer_acc, 4),
        refuse_rational_rate=round(refuse_rate, 4),
        ticket_completion_rate=round(ticket_rate, 4),
        avg_recall_at_k=round(avg_recall, 4) if avg_recall is not None else None,
        cases=results,
        created_at=datetime.utcnow().isoformat(),
    )


def write_report(report: EvalReport) -> Path:
    settings = get_settings()
    out_dir = settings.eval_report_path
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "eval_report.json"
    md_path = out_dir / "eval_report.md"
    payload = asdict(report)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Evaluation Report",
        "",
        f"- created_at: {report.created_at}",
        f"- total: {report.total}",
        f"- intent_accuracy: {report.intent_accuracy}",
        f"- answer_accuracy: {report.answer_accuracy}",
        f"- refuse_rational_rate: {report.refuse_rational_rate}",
        f"- ticket_completion_rate: {report.ticket_completion_rate}",
        f"- avg_recall_at_k: {report.avg_recall_at_k}",
        "",
        "## Cases",
        "",
    ]
    for c in report.cases:
        lines.append(
            f"- Q: {c.question}\n"
            f"  - intent: {c.intent_pred} (expected {c.intent_expected}) ok={c.intent_ok}\n"
            f"  - answer_ok={c.answer_ok} ticket_ok={c.ticket_ok} path={c.route_path}\n"
            f"  - answer: {c.answer[:180]}"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def evaluate_retrieval_recall(path: Path | None = None, k: int = 3) -> float:
    cases = load_golden_set(path)
    store = get_knowledge_store()
    scores: list[float] = []
    for case in cases:
        sources = case.get("expected_sources") or []
        if not sources:
            continue
        hits = store.hybrid_search(case["question"], top_k=k)
        got = {h.source for h in hits}
        scores.append(len(got & set(sources)) / len(sources))
    return round(sum(scores) / len(scores), 4) if scores else 0.0
