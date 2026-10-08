#!/usr/bin/env python
"""Bootstrap DB + knowledge ingest, then optionally run eval."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tools.db import init_db
from app.rag import get_knowledge_store
from app.eval import run_evaluation, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true", help="run evaluation after bootstrap")
    args = parser.parse_args()

    init_db()
    store = get_knowledge_store()
    n = store.ingest_directory()
    print(f"ingested chunks: {n}")
    if args.eval:
        report = run_evaluation()
        path = write_report(report)
        print(
            {
                "report": str(path),
                "intent_accuracy": report.intent_accuracy,
                "answer_accuracy": report.answer_accuracy,
                "refuse_rational_rate": report.refuse_rational_rate,
                "ticket_completion_rate": report.ticket_completion_rate,
                "avg_recall_at_k": report.avg_recall_at_k,
            }
        )


if __name__ == "__main__":
    main()
