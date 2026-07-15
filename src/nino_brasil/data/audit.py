from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import xarray as xr

from nino_brasil.config import project_path


DEFAULT_AUDIT_LOG = project_path("data/audit/ledger.jsonl")
LEDGER_SCHEMA_VERSION = "1.1"
PROCESS_RUN_ID = os.environ.get(
    "NINO_RUN_ID",
    f"run_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{os.getpid()}_{uuid.uuid4().hex[:8]}",
)
TERMINAL_SUCCESS = frozenset({"ok", "complete", "completed", "skipped", "already_exists"})
TERMINAL_FAILURE = frozenset({"error", "failed", "cancelled", "canceled"})
LIFECYCLE_STATUSES = TERMINAL_SUCCESS | TERMINAL_FAILURE | {"started", "running"}
_THREAD_LOCKS: dict[str, threading.Lock] = {}
_THREAD_LOCKS_GUARD = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def _ledger_lock(path: Path, *, timeout_seconds: float = 30.0) -> Iterator[None]:
    """Serialize JSONL appends across threads and processes.

    A persistent one-byte sidecar is locked by the operating system; it is not
    a stale sentinel and therefore needs no deletion after crashes.
    """
    resolved = str(Path(path).resolve())
    with _THREAD_LOCKS_GUARD:
        thread_lock = _THREAD_LOCKS.setdefault(resolved, threading.Lock())
    lock_path = Path(resolved + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with thread_lock, lock_path.open("a+b") as handle:
        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + timeout_seconds
            while True:
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"timed out locking audit ledger: {path}")
                    time.sleep(0.01)
            try:
                yield
            finally:
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file or a directory tree deterministically.

    Directory hashes include each relative path and file hash.  This is useful
    for Zarr stores without loading their arrays into memory.
    """
    path = Path(path)
    if path.is_file():
        return sha256_file(path, chunk_size=chunk_size)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        child_hash = sha256_file(child, chunk_size=chunk_size).encode("ascii")
        digest.update(child_hash)
    return digest.hexdigest()


def file_info(path: Path, *, include_hash: bool = False) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    info: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size_bytes": path.stat().st_size,
    }
    if include_hash:
        info["sha256"] = sha256_path(path)
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
        event.setdefault("schema_version", LEDGER_SCHEMA_VERSION)
        event.setdefault("run_id", PROCESS_RUN_ID)
        event.setdefault("event_id", uuid.uuid4().hex)
        event.pop("record_sha256", None)
        checksum_payload = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        event["record_sha256"] = hashlib.sha256(checksum_payload).hexdigest()
        payload = (json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
        with _ledger_lock(self.path):
            descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
            try:
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    if written <= 0:
                        raise OSError(f"short append to audit ledger at byte {offset}/{len(payload)}")
                    offset += written
                os.fsync(descriptor)
            finally:
                os.close(descriptor)

    def read(self) -> list[dict[str, Any]]:
        rows, _ = self.read_with_issues()
        return rows

    def read_with_issues(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Read valid objects and report every malformed line non-destructively."""
        rows: list[dict[str, Any]] = []
        issues: list[dict[str, Any]] = []
        for record in iter_ledger(self.path):
            if record.event is not None:
                rows.append(record.event)
            elif record.issue is not None:
                issues.append(record.issue)
        return rows, issues

    def latest_by_task(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for row in self.read():
            task_id = row.get("task_id")
            if task_id:
                latest[str(task_id)] = row
        return latest

    def print_summary(self) -> None:
        rows, issues = self.read_with_issues()
        print(f"audit log: {self.path}")
        print(f"events: {len(rows)}")
        print(f"malformed lines: {len(issues)}")
        status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
        for status, count in sorted(status_counts.items()):
            print(f"- {status}: {count}")


@dataclass(frozen=True)
class LedgerRecord:
    line_number: int
    event: dict[str, Any] | None = None
    issue: dict[str, Any] | None = None
    raw_line: str = ""


def iter_ledger(path: Path) -> Iterator[LedgerRecord]:
    """Yield valid events or structured issues; never mutate ``path``."""
    path = Path(path)
    if not path.exists():
        return
    # Snapshot complete lines under the same inter-process lock used by
    # writers, then release it before yielding/parsing to avoid long blocking.
    with _ledger_lock(path):
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
    for line_number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped:
            yield LedgerRecord(
                line_number=line_number,
                issue={"line_number": line_number, "issue": "blank_line", "bytes": len(raw.encode("utf-8"))},
                raw_line=raw,
            )
            continue
        try:
            value = json.loads(stripped)
        except json.JSONDecodeError as exc:
            yield LedgerRecord(
                line_number=line_number,
                issue={
                    "line_number": line_number,
                    "issue": "malformed_json",
                    "error": str(exc),
                    "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "bytes": len(raw.encode("utf-8")),
                },
                raw_line=raw,
            )
            continue
        if not isinstance(value, dict):
            yield LedgerRecord(
                line_number=line_number,
                issue={
                    "line_number": line_number,
                    "issue": "json_value_is_not_object",
                    "json_type": type(value).__name__,
                    "raw_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                    "bytes": len(raw.encode("utf-8")),
                },
                raw_line=raw,
            )
            continue
        yield LedgerRecord(line_number=line_number, event=value, raw_line=raw)


def _contains_sha256(value: Any) -> bool:
    if isinstance(value, dict):
        return any(
            (str(key).lower() == "sha256" and isinstance(item, str) and len(item) == 64)
            or _contains_sha256(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_sha256(item) for item in value)
    return False


def audit_ledger(path: Path = DEFAULT_AUDIT_LOG, *, detail_limit: int = 100) -> dict[str, Any]:
    """Return an audit report for syntax, task state, IDs, and hash coverage."""
    valid: list[tuple[int, dict[str, Any]]] = []
    issues: list[dict[str, Any]] = []
    for record in iter_ledger(Path(path)):
        if record.event is None:
            if record.issue is not None:
                issues.append(record.issue)
        else:
            valid.append((record.line_number, record.event))

    status_counts = Counter(str(event.get("status", "unknown")) for _, event in valid)
    latest_lifecycle: dict[str, tuple[int, dict[str, Any]]] = {}
    successful_without_hash: list[dict[str, Any]] = []
    missing_run_id = 0
    missing_event_id = 0
    timestamp_issues: list[dict[str, Any]] = []
    checksum_issues: list[dict[str, Any]] = []
    seen_event_ids: Counter[str] = Counter()
    for line_number, event in valid:
        task_id = event.get("task_id")
        status = str(event.get("status", "unknown")).lower()
        if task_id and status in LIFECYCLE_STATUSES:
            latest_lifecycle[str(task_id)] = (line_number, event)
        if status in TERMINAL_SUCCESS and not _contains_sha256(event):
            successful_without_hash.append(
                {"line_number": line_number, "task_id": task_id, "status": status}
            )
        if not event.get("run_id"):
            missing_run_id += 1
        event_id = event.get("event_id")
        if event_id:
            seen_event_ids[str(event_id)] += 1
        else:
            missing_event_id += 1
        timestamp = event.get("timestamp_utc")
        if not timestamp:
            timestamp_issues.append({"line_number": line_number, "issue": "missing_timestamp_utc"})
        else:
            try:
                datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
            except ValueError:
                timestamp_issues.append(
                    {"line_number": line_number, "issue": "invalid_timestamp_utc", "value": str(timestamp)}
                )
        record_checksum = event.get("record_sha256")
        if record_checksum:
            checksum_event = dict(event)
            checksum_event.pop("record_sha256", None)
            checksum_payload = json.dumps(
                checksum_event, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            expected_checksum = hashlib.sha256(checksum_payload).hexdigest()
            if record_checksum != expected_checksum:
                checksum_issues.append(
                    {"line_number": line_number, "issue": "record_sha256_mismatch"}
                )

    incomplete_tasks = [
        {
            "task_id": task_id,
            "line_number": line_number,
            "latest_status": str(event.get("status", "unknown")),
            "timestamp_utc": event.get("timestamp_utc"),
            "run_id": event.get("run_id"),
        }
        for task_id, (line_number, event) in sorted(latest_lifecycle.items())
        if str(event.get("status", "")).lower() in {"started", "running"}
    ]
    duplicate_event_ids = sorted(event_id for event_id, count in seen_event_ids.items() if count > 1)
    return {
        "ledger": str(Path(path).resolve()),
        "exists": Path(path).exists(),
        "audited_at_utc": utc_now(),
        "valid_events": len(valid),
        "malformed_lines": len(issues),
        "status_counts": dict(sorted(status_counts.items())),
        "task_count": len(latest_lifecycle),
        "incomplete_task_count": len(incomplete_tasks),
        "incomplete_tasks": incomplete_tasks,
        "successful_events_without_sha256_count": len(successful_without_hash),
        "successful_events_without_sha256_sample": successful_without_hash[:detail_limit],
        "successful_events_without_sha256_sample_truncated": len(successful_without_hash) > detail_limit,
        "missing_run_id_count": missing_run_id,
        "missing_event_id_count": missing_event_id,
        "timestamp_issue_count": len(timestamp_issues),
        "timestamp_issue_sample": timestamp_issues[:detail_limit],
        "record_checksum_issue_count": len(checksum_issues),
        "record_checksum_issue_sample": checksum_issues[:detail_limit],
        "duplicate_event_ids": duplicate_event_ids,
        "issues": issues,
        "original_preserved": True,
    }


def write_clean_ledger_copy(
    source: Path,
    destination: Path,
    *,
    overwrite: bool = False,
) -> dict[str, int]:
    """Write valid original lines to a *different* file, preserving the source."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("clean ledger copy must not overwrite the original ledger")
    if destination.exists() and not overwrite:
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    kept = rejected = 0
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        for record in iter_ledger(source):
            if record.event is None:
                rejected += 1
                continue
            line = record.raw_line
            handle.write(line if line.endswith("\n") else line + "\n")
            kept += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return {"kept": kept, "rejected": rejected}


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Audit a JSONL ledger without modifying the original.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_AUDIT_LOG)
    parser.add_argument("--report-json", type=Path)
    parser.add_argument("--clean-copy", type=Path, help="optional new path containing valid original lines")
    parser.add_argument("--overwrite-clean-copy", action="store_true")
    parser.add_argument("--detail-limit", type=int, default=100)
    args = parser.parse_args(argv)
    if args.detail_limit < 0:
        parser.error("--detail-limit must be non-negative")
    report = audit_ledger(args.ledger, detail_limit=args.detail_limit)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report_json.with_suffix(args.report_json.suffix + ".tmp")
        temporary.write_text(rendered + "\n", encoding="utf-8")
        os.replace(temporary, args.report_json)
    if args.clean_copy:
        result = write_clean_ledger_copy(args.ledger, args.clean_copy, overwrite=args.overwrite_clean_copy)
        print(f"clean copy: {args.clean_copy} ({result})")
    return 1 if report["malformed_lines"] or report["incomplete_task_count"] else 0


if __name__ == "__main__":
    raise SystemExit(_main())
