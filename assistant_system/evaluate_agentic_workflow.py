#!/usr/bin/env python3
"""Evaluate agentic protocol workflow on a local query set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistant_system.agentic_workflow import AgentWorkflowEngine
from hybrid_protocol_backend import HybridProtocolAssistant


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                continue
            query = str(row.get("query", "")).strip()
            if not query:
                continue
            rows.append(row)
    return rows


def _title_matches(title: str, keywords: List[str]) -> bool:
    if not keywords:
        return False
    title_low = (title or "").casefold()
    return any(str(kw).casefold() in title_low for kw in keywords)


def _question_hit(question_ids: List[str], expected_ids: List[str]) -> bool:
    if not expected_ids:
        return False
    qset = {str(q).strip().lower() for q in question_ids if str(q).strip()}
    eset = {str(q).strip().lower() for q in expected_ids if str(q).strip()}
    return bool(qset & eset)


def run_eval(
    project_root: Path,
    queries_path: Path,
    workflow_path: Path,
    top_k: int,
    output_json: Path | None,
) -> Dict[str, Any]:
    backend = HybridProtocolAssistant(project_root=project_root)
    backend.ensure_index()
    engine = AgentWorkflowEngine(search_backend=backend, workflow_path=workflow_path)

    rows = _load_jsonl(queries_path)
    cases: List[Dict[str, Any]] = []
    top1_hits = 0
    top3_hits = 0
    question_hits = 0
    clarification_required_count = 0
    per_domain: Dict[str, Dict[str, Any]] = {}

    for idx, row in enumerate(rows, start=1):
        query = str(row.get("query", "")).strip()
        expected_keywords = [str(x) for x in row.get("expected_title_keywords", []) if str(x).strip()]
        expected_question_ids = [str(x) for x in row.get("expected_question_ids", []) if str(x).strip()]
        expected_domain = str(row.get("expected_domain", "")).strip().lower()
        domain_key = expected_domain or "unknown"
        if domain_key not in per_domain:
            per_domain[domain_key] = {
                "cases_total": 0,
                "top1_hits": 0,
                "top3_hits": 0,
                "question_hits": 0,
                "clarification_required_count": 0,
            }

        result = engine.run(query=query, top_k=top_k, include_trace=False)
        ranked = result.get("results") or []
        top1_title = str((ranked[0] if ranked else {}).get("title", ""))
        top3_titles = [str(item.get("title", "")) for item in ranked[:3]]

        top1_hit = _title_matches(top1_title, expected_keywords)
        top3_hit = any(_title_matches(title, expected_keywords) for title in top3_titles)
        if top1_hit:
            top1_hits += 1
            per_domain[domain_key]["top1_hits"] += 1
        if top3_hit:
            top3_hits += 1
            per_domain[domain_key]["top3_hits"] += 1

        clarification = result.get("clarification") or {}
        confidence = result.get("confidence") or {}
        required = bool(clarification.get("required"))
        if required:
            clarification_required_count += 1
            per_domain[domain_key]["clarification_required_count"] += 1
        question_ids = [
            str(q.get("id", "")).strip()
            for q in (clarification.get("questions") or [])
            if str(q.get("id", "")).strip()
        ]
        q_hit = _question_hit(question_ids, expected_question_ids)
        if q_hit:
            question_hits += 1
            per_domain[domain_key]["question_hits"] += 1

        per_domain[domain_key]["cases_total"] += 1

        cases.append(
            {
                "case_id": idx,
                "query": query,
                "expected_domain": expected_domain,
                "expected_title_keywords": expected_keywords,
                "expected_question_ids": expected_question_ids,
                "top1_title": top1_title,
                "top3_titles": top3_titles,
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "clarification_required": required,
                "generated_question_ids": question_ids,
                "question_hit": q_hit,
                "top_confidence_pct": float(confidence.get("top_protocol_confidence_pct", 0.0)),
            }
        )

    total = len(rows) or 1
    summary = {
        "cases_total": len(rows),
        "top1_accuracy_proxy": round(top1_hits / total, 4),
        "top3_accuracy_proxy": round(top3_hits / total, 4),
        "question_quality_proxy": round(question_hits / total, 4),
        "clarification_required_ratio": round(clarification_required_count / total, 4),
        "top1_hits": top1_hits,
        "top3_hits": top3_hits,
        "question_hits": question_hits,
    }

    by_domain: Dict[str, Dict[str, Any]] = {}
    for domain, stats in sorted(per_domain.items()):
        total_domain = int(stats.get("cases_total", 0)) or 1
        by_domain[domain] = {
            "cases_total": int(stats.get("cases_total", 0)),
            "top1_hits": int(stats.get("top1_hits", 0)),
            "top3_hits": int(stats.get("top3_hits", 0)),
            "question_hits": int(stats.get("question_hits", 0)),
            "clarification_required_count": int(stats.get("clarification_required_count", 0)),
            "top1_accuracy_proxy": round(int(stats.get("top1_hits", 0)) / total_domain, 4),
            "top3_accuracy_proxy": round(int(stats.get("top3_hits", 0)) / total_domain, 4),
            "question_quality_proxy": round(int(stats.get("question_hits", 0)) / total_domain, 4),
            "clarification_required_ratio": round(
                int(stats.get("clarification_required_count", 0)) / total_domain,
                4,
            ),
        }

    report = {
        "summary": summary,
        "summary_by_domain": by_domain,
        "cases": cases,
        "queries_file": str(queries_path),
        "workflow_file": str(workflow_path),
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate agentic workflow quality")
    parser.add_argument(
        "--project-root",
        default=".",
        help="Project root path",
    )
    parser.add_argument(
        "--queries",
        default="docs/agentic_eval_queries.jsonl",
        help="Path to eval queries JSONL",
    )
    parser.add_argument(
        "--workflow",
        default="assistant_system/workflows/protocol-assistant.workflow.json",
        help="Workflow config path",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Top-k for workflow run")
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write JSON report",
    )
    parser.add_argument(
        "--show-domains",
        action="store_true",
        help="Print domain-level summary in addition to overall metrics",
    )
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    queries = Path(args.queries).resolve()
    workflow = Path(args.workflow).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else None

    report = run_eval(
        project_root=project_root,
        queries_path=queries,
        workflow_path=workflow,
        top_k=max(1, int(args.top_k)),
        output_json=output_json,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    if args.show_domains:
        print(json.dumps(report.get("summary_by_domain", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
