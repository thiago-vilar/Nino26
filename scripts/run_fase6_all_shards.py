#!/usr/bin/env python3
"""Run/resume every native-pixel F6 shard and merge an explicit run list.

The orchestrator never guesses coverage from a single 0:500 example.  It reads
the canonical CHIRPS pixel inventory, reuses only complete runs whose manifest
matches the requested model/target/lags/conditions, and passes the exact shard
directories to the merger so prior merged runs cannot contaminate discovery.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nino_brasil.artifacts import (  # noqa: E402
    scientific_git_state,
    scientific_input_record,
    validate_artifact_run,
)

RUNS = ROOT / "data/processed/runs/official/fase6"
TARGET = ROOT / "data/processed/zarr/features/chirps_native_weekly_targets.zarr"
MASTER = ROOT / "data/processed/parquet/features/nino34_master_weekly.csv"
PHASE = ROOT / "data/processed/parquet/modeling/f3_bridge/fases_semanais_en_ln.csv"
RUNNER_CONTRACT = "f6-pixel-native-v3-causal-persistence-ridge"


def _csv_tuple(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _matching_run(
    *,
    model: str,
    target_variable: str,
    target_transform: str,
    pixel_start: int,
    pixel_stop: int,
    lags: tuple[str, ...],
    conditions: tuple[str, ...],
    n_estimators: int,
    seed: int,
    grid_sha256: str,
    input_identity: tuple[dict[str, object], ...],
    git_identity: dict[str, object],
) -> Path | None:
    for directory in sorted(RUNS.glob("F6_*"), reverse=True) if RUNS.exists() else []:
        manifest_path = directory / "run_manifest.json"
        coverage_path = directory / "tables/shard_coverage_contract.csv"
        predictions_path = directory / "tables/pixel_oos_predictions.csv"
        if (
            not manifest_path.exists()
            or not coverage_path.exists()
            or not predictions_path.exists()
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        parameters = manifest.get("parameters", {})
        if manifest.get("status") != "complete":
            continue
        matches = (
            str(parameters.get("model")) == model
            and str(parameters.get("target_variable")) == target_variable
            and str(parameters.get("target_transform")) == target_transform
            and int(parameters.get("pixel_start", -1)) == pixel_start
            and int(parameters.get("pixel_stop", -1)) == pixel_stop
            and tuple(map(str, parameters.get("lags", ()))) == lags
            and tuple(map(str, parameters.get("conditions", ()))) == conditions
            and int(parameters.get("n_estimators", -1)) == n_estimators
            and int(manifest.get("seed", -1)) == seed
            and str(parameters.get("grid_sha256", "")) == grid_sha256
            and str(parameters.get("runner_contract", "")) == RUNNER_CONTRACT
            and bool(parameters.get("store_predictions", False))
            and not bool(parameters.get("research_override_gate", False))
        )
        manifest_inputs = tuple(
            {
                key: record.get(key)
                for key in ("path", "exists", "is_directory", "sha256", "tree_sha256")
            }
            for record in manifest.get("inputs", [])
        )
        recorded_git = manifest.get("git", {})
        git_matches = all(
            recorded_git.get(key) == git_identity.get(key)
            for key in ("commit", "status_sha256")
        )
        if matches and manifest_inputs == input_identity and git_matches:
            if not validate_artifact_run(directory).empty:
                continue
            coverage = pd.read_csv(coverage_path)
            if coverage.empty or not coverage["has_oos_metric"].astype(bool).all():
                continue
            return directory
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("rf", "xgb"), required=True)
    parser.add_argument("--target", type=Path, default=TARGET)
    parser.add_argument("--master", type=Path, default=MASTER)
    parser.add_argument("--phase-table", type=Path, default=PHASE)
    parser.add_argument("--target-variable", default="precip_weekly_mm")
    parser.add_argument(
        "--target-transform",
        choices=("none", "train_anomaly_mm", "train_robust_z"),
        default="train_robust_z",
    )
    parser.add_argument("--lags", type=_csv_tuple, default=("4", "8", "12", "16", "20", "24"))
    parser.add_argument(
        "--conditions",
        type=_csv_tuple,
        default=(
            "el_nino_genese",
            "el_nino_crescimento",
            "el_nino_pico",
            "el_nino_decaimento",
            "la_nina_genese",
            "la_nina_crescimento",
            "la_nina_pico",
            "la_nina_decaimento",
        ),
    )
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.shard_size < 1:
        parser.error("--shard-size deve ser positivo")
    for required in (args.target, args.master, args.phase_table):
        if not required.exists():
            raise FileNotFoundError(required)
    dataset = xr.open_zarr(args.target, consolidated=None)
    if "brazil_fraction" not in dataset:
        raise ValueError("Alvo CHIRPS sem brazil_fraction.")
    n_pixels = int((dataset["brazil_fraction"].values.reshape(-1) > 0).sum())
    grid_sha256 = str(dataset.attrs.get("grid_hash_sha256", ""))
    dataset.close()
    current_inputs = tuple(
        scientific_input_record(path)
        for path in (args.target, args.master, args.phase_table)
    )
    input_identity = tuple(
        {
            key: record.get(key)
            for key in ("path", "exists", "is_directory", "sha256", "tree_sha256")
        }
        for record in current_inputs
    )
    git_identity = scientific_git_state()
    selected_runs: list[Path] = []
    for start in range(0, n_pixels, args.shard_size):
        stop = min(start + args.shard_size, n_pixels)
        existing = _matching_run(
            model=args.model,
            target_variable=args.target_variable,
            target_transform=args.target_transform,
            pixel_start=start,
            pixel_stop=stop,
            lags=args.lags,
            conditions=args.conditions,
            n_estimators=args.n_estimators,
            seed=args.seed,
            grid_sha256=grid_sha256,
            input_identity=input_identity,
            git_identity=git_identity,
        )
        if existing is not None:
            print(f"[reuse] {start}:{stop} -> {existing.name}")
            selected_runs.append(existing)
            continue
        command = [
            sys.executable,
            "scripts/run_fase6_brazil_ml.py",
            "--mode",
            "official",
            "--model",
            args.model,
            "--target",
            str(args.target),
            "--target-variable",
            args.target_variable,
            "--target-transform",
            args.target_transform,
            "--master",
            str(args.master),
            "--phase-table",
            str(args.phase_table),
            "--lags",
            ",".join(args.lags),
            "--conditions",
            ",".join(args.conditions),
            "--pixel-start",
            str(start),
            "--pixel-stop",
            str(stop),
            "--n-estimators",
            str(args.n_estimators),
            "--seed",
            str(args.seed),
        ]
        print("[run]", " ".join(command))
        if args.dry_run:
            continue
        before = set(RUNS.glob("F6_*")) if RUNS.exists() else set()
        subprocess.run(command, cwd=ROOT, check=True)
        created = sorted(set(RUNS.glob("F6_*")) - before)
        if len(created) != 1:
            raise RuntimeError(f"Nao foi possivel identificar o run do shard {start}:{stop}.")
        selected_runs.append(created[0])
    if args.dry_run:
        print(f"[dry-run] pixels={n_pixels} shards={(n_pixels + args.shard_size - 1)//args.shard_size}")
        return 0
    if len(selected_runs) != (n_pixels + args.shard_size - 1) // args.shard_size:
        raise RuntimeError("Lista de shards incompleta; merge recusado.")
    merge = [
        sys.executable,
        "scripts/merge_fase6_shards.py",
        "--model",
        args.model,
        "--target",
        str(args.target),
        "--runs",
        *map(str, selected_runs),
    ]
    subprocess.run(merge, cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
