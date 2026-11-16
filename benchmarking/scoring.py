"""Use the upstream QA judge prompt and decision rule; report coverage separately."""
import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path

from .data import atomic_json, iter_questions, load_index
from .download import DATA_DIR, FILES, REPO, sha256_file
from .runner import append_jsonl, read_jsonl

JUDGES = {"gpt-4o": "gpt-4o-2024-08-06", "gpt-4o-mini": "gpt-4o-mini-2024-07-18", "deepseek": "deepseek-chat"}


def official_evaluator():
    path = REPO / "src" / "evaluation" / "evaluate_qa.py"
    spec = importlib.util.spec_from_file_location("longmemeval_official_evaluator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(selected, labels, judge):
    expected = {x["question_id"]: x for x in selected}
    if set(labels) - set(expected):
        raise ValueError("Judge labels contain unknown IDs")
    tasks = defaultdict(list)
    abstention = []
    for qid, value in labels.items():
        tasks[expected[qid]["question_type"]].append(int(value))
        if qid.endswith("_abs"):
            abstention.append(int(value))
    scores = {task: {"accuracy": sum(values) / len(values), "count": len(values)} for task, values in sorted(tasks.items())}
    # Match upstream: abstention stays in its original task and also gets a separate metric.
    return {"judge": JUDGES[judge], "official_paper_judge": judge == "gpt-4o",
            "selected": len(selected), "scored": len(labels), "correct": sum(labels.values()),
            "coverage": len(labels) / len(selected) if selected else 0,
            "overall_accuracy_on_scored": sum(labels.values()) / len(labels) if labels else None,
            "task_averaged_accuracy_on_observed_tasks": sum(x["accuracy"] for x in scores.values()) / len(scores) if scores else None,
            "correct_over_selected": sum(labels.values()) / len(selected) if selected else None,
            "missing_ids": sorted(set(expected) - set(labels)), "by_question_type": scores,
            "abstention_accuracy": sum(abstention) / len(abstention) if abstention else None,
            "abstention_count": len(abstention),
            "warning": "Partial runs/alternate judges are pilot results, not a complete published benchmark score."}


def score(run_dir, judge="gpt-4o"):
    from openai import OpenAI

    run_dir = Path(run_dir).resolve()
    manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
    index = load_index(manifest["identity"]["dataset"])
    if index["sha256"] != manifest["identity"]["dataset_sha256"]:
        raise ValueError("Score data differs from the run's dataset")
    predictions = read_jsonl(run_dir / "predictions.jsonl")
    if not predictions:
        raise ValueError("No completed predictions to score")
    by_id = {x["question_id"]: x for x in predictions}
    if len(by_id) != len(predictions) or set(by_id) - set(manifest["identity"]["selected_ids"]):
        raise ValueError("Duplicate/unexpected predictions")
    api_env = "DEEPSEEK_API_KEY" if judge == "deepseek" else "OPENAI_API_KEY"
    if not os.getenv(api_env):
        raise ValueError(f"{api_env} is not set. DeepSeek is available only via explicit --judge deepseek (alternative judge).")
    base_url = (os.getenv("DEEPSEEK_API_BASE") or "https://api.deepseek.com") if judge == "deepseek" else None
    client = OpenAI(api_key=os.getenv(api_env), base_url=base_url, timeout=90, max_retries=2)
    upstream = official_evaluator()
    source_digest = sha256_file(REPO / "src" / "evaluation" / "evaluate_qa.py")
    judgments_dir = run_dir / "judgments" / judge
    judgments_dir.mkdir(parents=True, exist_ok=True)
    labels, logs = {}, []
    try:
        for reference in iter_questions(DATA_DIR / FILES[manifest["identity"]["dataset"]], by_id):
            qid = reference["question_id"]
            pred = by_id[qid]
            prompt = upstream.get_anscheck_prompt(reference["question_type"], reference["question"], reference["answer"],
                                                  pred["hypothesis"], abstention="_abs" in qid)
            fingerprint = hashlib.sha256((JUDGES[judge] + source_digest + prompt).encode()).hexdigest()
            checkpoint = judgments_dir / (hashlib.sha256(qid.encode()).hexdigest()[:16] + ".json")
            if checkpoint.exists():
                entry = json.loads(checkpoint.read_text("utf-8"))
                if entry["fingerprint"] != fingerprint:
                    raise ValueError("Judged prediction/prompt changed; use a fresh run directory")
            else:
                try:
                    completion = client.chat.completions.create(model=JUDGES[judge], messages=[{"role": "user", "content": prompt}],
                                                              n=1, temperature=0, max_tokens=10)
                    response = completion.choices[0].message.content.strip()
                    entry = {**pred, "autoeval_label": {"model": JUDGES[judge], "label": "yes" in response.lower()},
                             "judge_response": response, "fingerprint": fingerprint,
                             "usage": completion.usage.model_dump() if completion.usage else None}
                    atomic_json(checkpoint, entry)
                except Exception as exc:
                    append_jsonl(run_dir / f"judge_errors_{judge}.jsonl", {"question_id": qid, "error_type": type(exc).__name__, "error": str(exc)})
                    print(f"Judge FAILED {qid}: {type(exc).__name__}: {exc}", flush=True)
                    continue
            labels[qid] = entry["autoeval_label"]["label"]
            logs.append(entry)
            print(f"Judged {qid}: {entry['autoeval_label']['label']}", flush=True)
        # Compatible with upstream print_qa_metrics.py (which asserts the GPT-4o model).
        log_file = run_dir / f"predictions.jsonl.eval-results-{judge}"
        log_file.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in logs), encoding="utf-8")
        report = summarize(manifest["selected"], labels, judge)
        report["official_evaluator_sha256"] = source_digest
        report["prompt_and_decision_rule"] = "Unmodified upstream get_anscheck_prompt; temperature=0, max_tokens=10; 'yes' in response.lower()"
        atomic_json(run_dir / f"metrics_{judge}.json", report)
        return report
    finally:
        client.close()
