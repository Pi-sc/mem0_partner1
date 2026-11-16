"""Download the official cleaned release, pinned to an immutable revision."""
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
REPO = PROJECT / "benchmarks" / "LongMemEval"
DATA_DIR = REPO / "data"
DATASET = "xiaowu0162/longmemeval-cleaned"
REVISION = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
FILES = {"oracle": "longmemeval_oracle.json", "s": "longmemeval_s_cleaned.json", "m": "longmemeval_m_cleaned.json"}


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(variants=("oracle", "s", "m")):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = f"https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(api, timeout=60) as response:
        release = json.load(response)
    remote = {x["rfilename"]: x for x in release["siblings"]}

    def fetch(variant):
        name = FILES[variant]
        dest = DATA_DIR / name
        info = remote[name]
        size = info.get("size") or info.get("lfs", {}).get("size")
        expected_sha = info.get("lfs", {}).get("sha256")
        if dest.exists() and (not size or dest.stat().st_size == size):
            digest = sha256_file(dest)
            if not expected_sha or digest == expected_sha:
                print(f"Verified existing {name}", flush=True)
                return name, {"bytes": dest.stat().st_size, "sha256": digest, "url": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"}
        partial = dest.with_suffix(dest.suffix + ".part")
        base_url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/{name}"
        if size and size > 64 * 1024**2:
            # Range workers improve throughput on the large M release. Keep completed
            # chunks on interruption, and combine only in byte order before SHA-256.
            chunks_dir = DATA_DIR / (name + ".chunks")
            chunks_dir.mkdir(exist_ok=True)
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > size:
                raise ValueError(f"Partial file exceeds official size: {partial}")
            ranges = [(start, min(start + 16 * 1024**2, size) - 1) for start in range(offset, size, 16 * 1024**2)]

            def fetch_chunk(bounds):
                start, end = bounds
                chunk = chunks_dir / f"{start}-{end}.bin"
                if chunk.exists() and chunk.stat().st_size == end - start + 1:
                    return chunk
                for retry in range(5):
                    req = urllib.request.Request(base_url + f"?download=true&chunk={start}&ts={time.time_ns()}",
                                                 headers={"Range": f"bytes={start}-{end}"})
                    try:
                        with urllib.request.urlopen(req, timeout=90) as response:
                            expected_range = f"bytes {start}-{end}/{size}"
                            if response.status != 206 or response.headers.get("Content-Range") != expected_range:
                                raise RuntimeError("Server did not honor the requested byte range")
                            temp = chunk.with_suffix(".part")
                            with temp.open("wb") as f:
                                while block := response.read(1024 * 1024):
                                    f.write(block)
                        if temp.stat().st_size != end - start + 1:
                            raise RuntimeError("Incomplete range download")
                        temp.replace(chunk)
                        print(f"{name}: chunk {end / 1024**2:.0f} MiB / {size / 1024**2:.0f} MiB ready", flush=True)
                        return chunk
                    except Exception:
                        if retry == 4:
                            raise
                        time.sleep(min(2 ** retry, 10))

            with ThreadPoolExecutor(max_workers=12) as workers:
                with partial.open("ab") as assembled:
                    for chunk in workers.map(fetch_chunk, ranges):
                        with chunk.open("rb") as f:
                            for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                                assembled.write(block)
                        assembled.flush()
                        # Retain chunks until the complete file has passed its checksum.
            digest = sha256_file(partial)
            if partial.stat().st_size != size or (expected_sha and digest != expected_sha):
                raise RuntimeError(f"Size/checksum mismatch for {name}")
            partial.replace(dest)
            for chunk in chunks_dir.glob("*.bin"):
                chunk.unlink()
            for temp in chunks_dir.glob("*.part"):
                temp.unlink()
            chunks_dir.rmdir()
            print(f"Downloaded and verified {name}", flush=True)
            return name, {"bytes": size, "sha256": digest, "url": base_url}
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            # Cache-busting avoids stale signed CDN URLs; Range resumes interrupted downloads.
            request = urllib.request.Request(base_url + f"?download=true&ts={time.time_ns()}")
            if offset:
                request.add_header("Range", f"bytes={offset}-")
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    resumed = response.status == 206
                    if resumed and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                        raise RuntimeError("Invalid range response")
                    with partial.open("ab" if resumed else "wb") as f:
                        last = time.monotonic()
                        while block := response.read(4 * 1024 * 1024):
                            f.write(block)
                            if time.monotonic() - last > 15:
                                print(f"{name}: {f.tell() / 1024**2:.1f} MiB / {(size or 0) / 1024**2:.1f} MiB", flush=True)
                                last = time.monotonic()
                if size and partial.stat().st_size != size:
                    raise RuntimeError(f"Incomplete download: {partial.stat().st_size} / {size}")
                digest = sha256_file(partial)
                if expected_sha and digest != expected_sha:
                    raise RuntimeError(f"Checksum mismatch for {name}; remove only this .part file before retrying")
                os.replace(partial, dest)
                print(f"Downloaded and verified {name}", flush=True)
                return name, {"bytes": dest.stat().st_size, "sha256": digest, "url": base_url}
            except Exception as exc:
                print(f"{name}: retry {attempt + 1}/5: {type(exc).__name__}: {exc}", flush=True)
                if attempt == 4:
                    raise
                time.sleep(min(2 ** attempt, 10))

    manifest_path = DATA_DIR / "download_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    if manifest and manifest.get("revision") != REVISION:
        raise ValueError("Existing manifest uses another data revision")
    manifest.update({"dataset": DATASET, "revision": REVISION})
    manifest.setdefault("files", {})
    with ThreadPoolExecutor(max_workers=3) as pool:
        for name, info in pool.map(fetch, variants):
            manifest["files"][name] = info
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    download()
