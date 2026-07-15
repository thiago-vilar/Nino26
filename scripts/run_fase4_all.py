#!/usr/bin/env python3
"""Run one isolated F4 signal: numerical core C -> D -> clean viewers."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
ENSO_TYPES = ("el_nino", "la_nina")
TARGET = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"


def notebook_contract(enso_type: str) -> tuple[Path, dict[str, str]]:
    if enso_type == "el_nino":
        return ROOT / "notebooks/fase4_nino", {
            "4C": "F4NinoC_sinal_pixel_lags.ipynb",
            "4D": "F4NinoD_clusters_alvo.ipynb",
        }
    return ROOT / "notebooks/fase4_nina", {
        "4C": "F4NinaC_sinal_pixel_lags.ipynb",
        "4D": "F4NinaD_clusters_alvo.ipynb",
    }


def execute_notebook(
    source: Path,
    destination: Path,
    *,
    kernel: str,
    timeout: int,
    environment: dict[str, str],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output-dir",
            str(destination.parent),
            "--output",
            destination.name,
            f"--ExecutePreprocessor.timeout={timeout}",
            f"--ExecutePreprocessor.kernel_name={kernel}",
            str(source),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )


def _ensure_target(*, allow_build: bool) -> None:
    if TARGET.exists():
        return
    if not allow_build:
        raise FileNotFoundError(
            "CHIRPS nativo ausente. A F4 não reconstrói o alvo sem "
            "--allow-build-chirps."
        )
    subprocess.run(
        [sys.executable, "scripts/build_phase4_chirps_targets.py"],
        cwd=ROOT,
        check=True,
    )


def _run_numeric_core(enso_type: str, keys: list[str]) -> None:
    if "4C" in keys:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_fase4c_regional.py",
                "--enso-type",
                enso_type,
                "--field-permutations",
                "199",
                "--replace-existing",
            ],
            cwd=ROOT,
            check=True,
        )
    if "4D" in keys:
        subprocess.run(
            [
                sys.executable,
                "scripts/run_fase4d_targets.py",
                "--enso-type",
                enso_type,
            ],
            cwd=ROOT,
            check=True,
        )


def _validate_scoped_outputs(enso_type: str, keys: list[str]) -> None:
    from scripts.notebook_run_viewer import audit_phase4_outputs
    from scripts.run_fase4c_regional import scoped_artifact_path
    from scripts.run_fase4d_targets import f4c_artifact_paths

    groups: list[tuple[str, list[Path], bool]] = []
    if "4C" in keys:
        groups.append(("F4C", list(f4c_artifact_paths(enso_type).values()), True))
    if "4D" in keys:
        stats = ROOT / "data/processed/parquet/statistics"
        groups.append(
            (
                "F4D",
                [
                    scoped_artifact_path(stats / "phase4D_native_clusters_pixels.parquet", enso_type),
                    scoped_artifact_path(stats / "phase4D_native_cluster_profiles.csv", enso_type),
                    scoped_artifact_path(stats / "phase4D_native_cluster_ranking.csv", enso_type),
                    scoped_artifact_path(stats / "phase4D_native_gate_event_jackknife.csv", enso_type),
                    scoped_artifact_path(stats / "phase4D_native_hypothesis_summary.csv", enso_type),
                    scoped_artifact_path(stats / "phase4D_native_target_coverage.csv", enso_type),
                ],
                False,
            )
        )
    for stage, paths, canonical_f4c in groups:
        audit, run_id = audit_phase4_outputs(
            paths,
            canonical_f4c=canonical_f4c,
            expected_stage=stage,
            expected_enso_type=enso_type,
        )
        if run_id is None:
            raise RuntimeError(
                f"validação scoped {stage}/{enso_type} falhou:\n{audit.to_string(index=False)}"
            )
        print(f"[ok] {stage}/{enso_type}: {run_id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--enso-type", required=True, choices=ENSO_TYPES)
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help="executa o núcleo numérico antes dos notebooks viewers",
    )
    parser.add_argument("--allow-build-chirps", action="store_true")
    parser.add_argument("--only", nargs="+", choices=("4C", "4D"))
    parser.add_argument("--kernel", default="nino-brasil")
    parser.add_argument("--timeout", type=int, default=43200)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout deve ser positivo")

    keys = [key for key in ("4C", "4D") if args.only is None or key in args.only]
    nbdir, inventory = notebook_contract(args.enso_type)
    sources = [nbdir / inventory[key] for key in keys]
    missing_notebooks = [path for path in sources if not path.is_file()]
    if missing_notebooks:
        raise FileNotFoundError(f"notebooks F4 ausentes: {missing_notebooks}")

    _ensure_target(allow_build=args.allow_build_chirps)
    if args.run_pipeline:
        _run_numeric_core(args.enso_type, keys)
    _validate_scoped_outputs(args.enso_type, keys)

    environment = os.environ.copy()
    environment.update(
        {
            "NINO26_NOTEBOOK_MODE": "official",
            "NINO26_RUN_PIPELINE": "0",
            "NINO26_ENSO_TYPE": args.enso_type,
        }
    )
    local_jupyter = ROOT / ".venv/share/jupyter"
    if local_jupyter.exists():
        current = environment.get("JUPYTER_PATH", "")
        environment["JUPYTER_PATH"] = os.pathsep.join(
            value for value in (str(local_jupyter), current) if value
        )
    runtime = ROOT / "data/interim/notebook-runtime"
    (runtime / "ipython").mkdir(parents=True, exist_ok=True)
    (runtime / "jupyter").mkdir(parents=True, exist_ok=True)
    environment["IPYTHONDIR"] = str(runtime / "ipython")
    environment["JUPYTER_RUNTIME_DIR"] = str(runtime / "jupyter")

    for key, source in zip(keys, sources, strict=True):
        destination = nbdir / "executed" / source.name
        print(f">>> {key}: {source.name}", flush=True)
        execute_notebook(
            source,
            destination,
            kernel=args.kernel,
            timeout=args.timeout,
            environment=environment,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
