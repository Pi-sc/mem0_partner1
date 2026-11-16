import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import portalocker

from .data import load_index, select_rows
from .download import PROJECT, REPO, download


def build_parser():
    parser = argparse.ArgumentParser(description="Independent LongMemEval benchmark mode for Mem0 Partner 1")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("download", help="Download/verify pinned official Oracle, S and M datasets")
    setup.add_argument("--dataset", choices=("all", "oracle", "s", "m"), default="all")
    inspect = sub.add_parser("inspect", help="Validate data and summarize its scale (no API calls)")
    inspect.add_argument("--dataset", choices=("oracle", "s", "m"), default="s")
    run = sub.add_parser("run", help="Replay complete selected histories, then answer questions")
    run.add_argument("--dataset", choices=("oracle", "s", "m"), default="s")
    run.add_argument("--mode", choices=("mem0", "oracle", "no-memory"), default="mem0")
    run.add_argument("--limit", type=int, default=1, help="Selected questions, NOT history truncation; 0 = all 500")
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--question-id", action="append", help="Explicit smoke/debug IDs, overrides --limit")
    run.add_argument("--top-k", type=int, default=8)
    run.add_argument("--threshold", type=float, default=0.1)
    run.add_argument("--model", default="deepseek-chat")
    run.add_argument("--embedding-model", default="nomic-embed-text")
    run.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    run.add_argument("--max-answer-tokens", type=int, default=2000)
    run.add_argument("--max-write-calls", type=int, default=1000,
                     help="Reject before running if indexing calls exceed this budget; 0 explicitly disables budget")
    run.add_argument("--output", type=Path)
    run.add_argument("--resume", action="store_true", help="Requires --output and the exact original settings")
    run.add_argument("--dry-run", action="store_true", help="Print selected IDs and full-history workload; no model calls")
    scoring = sub.add_parser("score", help="Use official QA prompts; GPT-4o is the paper's judge")
    scoring.add_argument("--run-dir", type=Path, required=True)
    scoring.add_argument("--judge", choices=("gpt-4o", "gpt-4o-mini", "deepseek"), default="gpt-4o")
    cleanup = sub.add_parser("cleanup", help="Keep the newest ten experiments' test databases; preserve scores/results")
    cleanup.add_argument("--dry-run", action="store_true", help="Preview database cleanup without deleting memory")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == "cleanup":
            from .retention import cleanup_memories
            print(json.dumps(cleanup_memories(dry_run=args.dry_run), ensure_ascii=False, indent=2))
            return 0
        if args.command == "download":
            if not (REPO / "README.md").exists():
                raise ValueError(f"First clone the official repo to {REPO}")
            print(json.dumps(download(("oracle", "s", "m") if args.dataset == "all" else (args.dataset,)), indent=2))
            return 0
        if args.command == "score":
            from .scoring import score
            report = score(args.run_dir, args.judge)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["coverage"] == 1 else 2
        index = load_index(args.dataset)
        if args.command == "inspect":
            rows = index["questions"]
            report = {"dataset": args.dataset, "questions": len(rows), "revision": index["revision"], "sha256": index["sha256"],
                      "by_type": dict(Counter(x["question_type"] for x in rows)),
                      "abstention": sum(x["abstention"] for x in rows),
                      "sessions_range": [min(x["sessions"] for x in rows), max(x["sessions"] for x in rows)],
                      "total_round_write_calls": sum(x["write_calls"] for x in rows),
                      "sessions_after_question": sum(x["sessions_after_question"] for x in rows),
                      "duplicate_session_ids": sum(x["duplicate_session_ids"] for x in rows),
                      "timestamp_note": "Preserve all official sessions even when their timestamp is later than question_date; do not silently filter."}
            print(json.dumps(report, indent=2))
            return 0
        if args.resume and not args.output:
            raise ValueError("--resume requires --output")
        if args.top_k < 1 or not 0 <= args.threshold <= 1 or args.max_answer_tokens < 1 or args.max_write_calls < 0:
            raise ValueError("Invalid top-k, threshold or token/call budget")
        if args.mode == "oracle" and args.dataset != "oracle":
            raise ValueError("oracle reader mode must use --dataset oracle")
        selected = select_rows(index, args.limit, args.seed, args.question_id)
        workload = sum(x["write_calls"] for x in selected) if args.mode == "mem0" else 0
        report = {"dataset": args.dataset, "mode": args.mode, "selected_questions": len(selected),
                  "selected_ids": [x["question_id"] for x in selected], "sessions": sum(x["sessions"] for x in selected),
                  "planned_indexing_llm_calls": workload, "planned_reader_llm_calls": len(selected),
                  "planned_judge_llm_calls": len(selected), "full_history_preserved": True,
                  "budget_note": "LLM requests may retry; call counts are logical operations, not a currency or token cap."}
        print(json.dumps(report, indent=2), flush=True)
        if args.dry_run:
            return 0
        if args.max_write_calls and workload > args.max_write_calls:
            raise ValueError(f"Full history needs {workload} indexing calls, exceeds --max-write-calls {args.max_write_calls}. "
                             "Select fewer questions or explicitly raise the budget. History will not be truncated.")
        from .runner import run
        settings = {"mode": args.mode, "model": args.model, "embedding_model": args.embedding_model,
                    "ollama_url": args.ollama_url, "top_k": args.top_k, "threshold": args.threshold,
                    "max_answer_tokens": args.max_answer_tokens}
        output = args.output or PROJECT / "benchmarks" / "runs" / f"{datetime.now():%Y%m%d_%H%M%S}_{args.dataset}_{args.mode}"
        status = run(index, selected, settings, output, resume=args.resume)
        print(f"Run directory: {output.resolve()}", flush=True)
        print(json.dumps(status, indent=2))
        return 0 if not status["missing_ids"] else 2
    except (ValueError, FileNotFoundError, portalocker.exceptions.LockException) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
