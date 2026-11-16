"""Stream data and keep scoring labels out of the memory/reader inputs."""
import json
import random
import re
from datetime import datetime
from pathlib import Path

from .download import DATA_DIR, FILES, REVISION, sha256_file

INDEX_VERSION = 2


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def parse_date(text):
    # Official release: 2023/04/10 (Mon) 23:07. Weekday is redundant.
    clean = re.sub(r"\s*\([^)]*\)", "", text).replace("/", "-")
    return datetime.fromisoformat(clean)


def iter_questions(path, ids=None):
    import ijson

    pending = set(ids) if ids is not None else None
    with Path(path).open("rb") as f:
        for case in ijson.items(f, "item"):
            if pending is None or case["question_id"] in pending:
                yield case
                if pending is not None:
                    pending.remove(case["question_id"])
                    if not pending:
                        return
    if pending:
        raise ValueError(f"Missing question IDs: {sorted(pending)}")


def history_rounds(case):
    """Chronological sessions; each user message with its following assistant reply(s)."""
    sessions = zip(case["haystack_session_ids"], case["haystack_dates"], case["haystack_sessions"])
    for session_id, date, turns in sorted(sessions, key=lambda x: parse_date(x[1])):
        batch, indices = [], []
        for index, turn in enumerate(turns):
            if turn["role"] == "user" and batch:
                yield {"session_id": session_id, "date": date, "turn_indices": indices, "messages": batch}
                batch, indices = [], []
            # Deliberately whitelist fields: never pass has_answer or other annotations.
            batch.append({"role": turn["role"], "content": turn["content"]})
            indices.append(index)
        if batch:
            yield {"session_id": session_id, "date": date, "turn_indices": indices, "messages": batch}


def validate_case(case):
    for field in ("question_id", "question_type", "question", "answer", "question_date", "haystack_session_ids",
                  "haystack_dates", "haystack_sessions", "answer_session_ids"):
        if field not in case:
            raise ValueError(f"Missing {field}")
    ids, dates, sessions = case["haystack_session_ids"], case["haystack_dates"], case["haystack_sessions"]
    if not len(ids) == len(dates) == len(sessions):
        raise ValueError(f"Mismatched session arrays: {case['question_id']}")
    # The official cleaned S release contains repeated session IDs in some histories.
    # Preserve every occurrence as supplied; IDs are annotations, not unique array keys.
    parse_date(case["question_date"])
    for date, turns in zip(dates, sessions):
        parse_date(date)
        for turn in turns:
            if turn.get("role") not in ("user", "assistant") or not isinstance(turn.get("content"), str):
                raise ValueError(f"Invalid history turn: {case['question_id']}")


def load_index(variant):
    path = DATA_DIR / FILES[variant]
    if not path.exists():
        raise FileNotFoundError(f"Data not downloaded: {path}. Run: python longmemeval_benchmark.py download")
    stat = path.stat()
    stamp = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    index_path = path.with_suffix(".index.json")
    if index_path.exists():
        cached = json.loads(index_path.read_text("utf-8"))
        if cached.get("file_stamp") == stamp and cached.get("index_version") == INDEX_VERSION:
            return cached
    print(f"Validating and indexing {path.name} (streaming)...", flush=True)
    digest = sha256_file(path)
    manifest = json.loads((DATA_DIR / "download_manifest.json").read_text("utf-8"))
    if manifest.get("revision") != REVISION or manifest["files"][path.name]["sha256"] != digest:
        raise ValueError("Dataset checksum/version does not match the download manifest")
    rows, seen = [], set()
    for case in iter_questions(path):
        validate_case(case)
        qid = case["question_id"]
        if qid in seen:
            raise ValueError(f"Duplicate question ID: {qid}")
        seen.add(qid)
        rows.append({"question_id": qid, "question_type": case["question_type"], "abstention": qid.endswith("_abs"),
                     "sessions": len(case["haystack_sessions"]), "messages": sum(map(len, case["haystack_sessions"])),
                     "write_calls": sum(1 for _ in history_rounds(case)),
                     "history_chars": sum(len(t["content"]) for s in case["haystack_sessions"] for t in s),
                     "duplicate_session_ids": len(case["haystack_session_ids"]) - len(set(case["haystack_session_ids"])),
                     "sessions_after_question": sum(parse_date(d) > parse_date(case["question_date"]) for d in case["haystack_dates"])})
    index = {"index_version": INDEX_VERSION, "variant": variant, "revision": REVISION, "sha256": digest, "file_stamp": stamp, "questions": rows}
    atomic_json(index_path, index)
    return index


def select_rows(index, limit=1, seed=42, question_ids=None):
    if limit < 0:
        raise ValueError("limit must be >= 0; 0 means the complete dataset")
    rows = index["questions"]
    if question_ids:
        wanted = set(question_ids)
        selected = [x for x in rows if x["question_id"] in wanted]
        if {x["question_id"] for x in selected} != wanted:
            raise ValueError("Unknown question ID(s)")
        return selected
    if limit == 0:
        return rows
    # Round-robin tasks + abstention gives reproducible mixed pilots, not an accuracy-biased sample.
    rng = random.Random(seed)
    buckets = {}
    for row in rows:
        bucket = "abstention" if row["abstention"] else row["question_type"]
        buckets.setdefault(bucket, []).append(row)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for values in buckets.values():
        rng.shuffle(values)
    selected = []
    while len(selected) < min(limit, len(rows)):
        for key in keys:
            if buckets[key]:
                selected.append(buckets[key].pop())
                if len(selected) == min(limit, len(rows)):
                    break
    return selected
