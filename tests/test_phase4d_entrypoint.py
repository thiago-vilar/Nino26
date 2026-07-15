from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_fase4d_targets.py"


@pytest.mark.parametrize(
    ("command", "cwd"),
    [
        ([sys.executable, str(RUNNER), "--help"], None),
        ([sys.executable, "-m", "scripts.run_fase4d_targets", "--help"], ROOT),
    ],
    ids=("script-from-arbitrary-cwd", "module-from-project-root"),
)
def test_phase4d_entrypoint_supports_both_invocation_styles(
    command: list[str], cwd: Path | None, tmp_path: Path
) -> None:
    """The F4D runner must import cleanly before touching scientific outputs."""

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    completed = subprocess.run(
        command,
        cwd=cwd or tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Fase 4D" in completed.stdout
