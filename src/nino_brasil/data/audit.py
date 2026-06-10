from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import xarray as xr

from nino_brasil.config import project_path


DEFAULT_AUDIT_LOG = project_path("data/audit/ledger.jsonl")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_info(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    info: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
    }
    if include_hash and path.is_file():
        info["sha256"] = sha256_file(path)
    return info


def dataset_summary(path: Path, *, zarr: bool = False) -> dict[str, Any]:
    opener = xr.open_zarr if zarr else xr.open_dataset
    ds = opener(path)
    try:
        return {
            "dims": dict(ds.sizes),
            "variables": list(ds.data_vars),
            "coords": list(ds.coords),
        }
    finally:
        ds.close()


class AuditLog:
    def __init__(self, path: Path = DEFAULT_AUDIT_LOG) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **event: Any) -> None:
        event.setdefault("timestamp_utc", utc_now())
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return rows

    def latest_by_task(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.read():
            task_id = row.get("task_id")
            if task_id:
                latest[str(task_id)] = row
        return latest

    def print_summary(self) -> None:
        rows = self.read()
        print(f"audit log: {self.path}")
        print(f"events: {len(rows)}")
        status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
        for status, count in sorted(status_counts.items()):
            print(f"- {status}: {count}")
