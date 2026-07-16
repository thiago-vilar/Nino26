from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.credentials import load_local_env


def _python(*parts: str) -> list[str]:
    return [sys.executable, *parts]


def _render(command: list[str]) -> str:
    return " ".join(f'"{item}"' if " " in item else item for item in command)


def _copernicus_marine_preflight() -> tuple[bool, str]:
    """Validate the separate Marine Data Store login before long downloads."""
    executable_name = "copernicusmarine.exe" if sys.platform == "win32" else "copernicusmarine"
    executable = Path(sys.executable).with_name(executable_name)
    command = [str(executable) if executable.exists() else executable_name, "login", "--check-credentials-valid"]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=90,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"Nao foi possivel executar {_render(command)}: {exc}"
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    lowered = output.casefold()
    rejected_markers = (
        "no credentials found",
        "credentials are invalid",
        "invalid credentials",
        "authentication failed",
    )
    ready = result.returncode == 0 and not any(marker in lowered for marker in rejected_markers)
    return ready, output


def commands(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    operational_end = args.operational_end or (pd.Timestamp.now().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    tasks: list[tuple[str, list[str]]] = [
        (
            "NOAA UFS daily core 1981-1992",
            _python(
                "scripts/ocean_daily_pipeline.py",
                "ingest-ufs",
                "--start-year",
                "1981",
                "--end-year",
                "1992",
                "--build-features",
                "--execute",
            ),
        ),
    ]
    if not args.skip_overlap:
        tasks.append(
            (
                "NOAA UFS overlap 1993-1995",
                _python(
                    "scripts/ocean_daily_pipeline.py",
                    "ingest-ufs",
                    "--start-year",
                    "1993",
                    "--end-year",
                    "1995",
                    "--build-features",
                    "--execute",
                ),
            )
        )
    tasks.extend(
        [
            (
                "GLORYS12 multiyear daily",
                _python(
                    "scripts/ocean_daily_pipeline.py",
                    "ingest-glorys",
                    "--start-year",
                    "1993",
                    "--end-year",
                    str(pd.Timestamp(args.glorys_my_end).year),
                    "--end-date",
                    args.glorys_my_end,
                    "--delete-source-after-zarr",
                    "--execute",
                ),
            ),
            (
                "GLO12 operational analysis tail",
                _python(
                    "scripts/ocean_daily_pipeline.py",
                    "ingest-glorys-operational",
                    "--start-date",
                    (pd.Timestamp(args.glorys_my_end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    "--end-date",
                    operational_end,
                    "--delete-source-after-zarr",
                    "--execute",
                ),
            ),
            (
                "Daily ocean Phase 2 audit",
                _python(
                    "scripts/audit_ocean_phase2.py",
                    "--glorys-my-end",
                    args.glorys_my_end,
                    "--operational-end",
                    operational_end,
                    *([] if args.skip_overlap else ["--overlap-year", "1993", "--overlap-year", "1994", "--overlap-year", "1995"]),
                ),
            ),
            (
                "Executive panel",
                _python("scripts/update_painel_executivo.py"),
            ),
        ]
    )
    return tasks


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Resumable orchestrator for daily ocean inputs aggregated to complete W-SUN weeks.")
    root.add_argument("--glorys-my-end", default="2026-05-26")
    root.add_argument("--operational-end")
    root.add_argument("--skip-overlap", action="store_true")
    root.add_argument("--execute", action="store_true")
    root.add_argument("--continue-on-error", action="store_true")
    return root


def main() -> int:
    load_local_env()
    args = parser().parse_args()
    if args.execute:
        marine_ready, marine_output = _copernicus_marine_preflight()
        print("[Copernicus Marine credential preflight]")
        if marine_output:
            print(marine_output)
        if not marine_ready:
            print(
                "ABORTADO antes de qualquer download: crie a conta em "
                "https://data.marine.copernicus.eu/register e execute "
                ".venv\\Scripts\\copernicusmarine.exe login."
            )
            return 2
    tasks = commands(args)
    final_exit_code = 0
    for label, command in tasks:
        print(f"[{label}]\n{_render(command)}")
        if not args.execute:
            continue
        environment = os.environ.copy()
        environment.setdefault("NINO_CDS_RETRY_MAX", "5")
        environment.setdefault("NINO_CDS_SLEEP_MAX", "60")
        environment.setdefault("NINO_CDS_TIMEOUT", "120")
        result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if result.returncode:
            final_exit_code = final_exit_code or int(result.returncode)
            if not args.continue_on_error:
                return int(result.returncode)
    return final_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
