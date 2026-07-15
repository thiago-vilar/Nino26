"""Pequenas funções compartilhadas pelos runners completos F2/F5–F8."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print(">>>", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def python(script: str, *arguments: str) -> list[str]:
    return [sys.executable, script, *arguments]


def latest_complete_run(
    phase: int,
    predicate: Callable[[dict[str, object]], bool],
) -> Path:
    root = ROOT / f"data/processed/runs/official/fase{int(phase)}"
    candidates: list[tuple[str, Path]] = []
    for directory in root.glob(f"F{int(phase)}_*") if root.is_dir() else []:
        manifest_path = directory / "run_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("phase") != int(phase)
            or manifest.get("mode") != "official"
            or manifest.get("status") != "complete"
            or not predicate(manifest)
        ):
            continue
        candidates.append((str(manifest.get("finished_at", "")), directory))
    if not candidates:
        raise RuntimeError(f"nenhum ArtifactRun F{phase} completo corresponde ao braço executado")
    return max(candidates, key=lambda item: item[0])[1]


def execute_viewer(code: str, *, kernel: str, timeout: int) -> None:
    run(
        python(
            "scripts/run_all_notebooks.py",
            "--only",
            code,
            "--mode",
            "official",
            "--require-artifacts",
            "--kernel",
            kernel,
            "--timeout",
            str(timeout),
        )
    )
