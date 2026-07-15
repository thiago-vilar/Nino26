#!/usr/bin/env python3
"""Constrói a ponte F3 isolada -> F5/F6/F7/F8, sem inferência conjunta.

Os testes e compostos permanecem separados em F3Nino e F3Nina. Este script
somente reúne o cadastro de eventos e recompõe a linha semanal de nove estados
necessária aos folds preditivos posteriores.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.events.enso import EnsoLifecycleConfig, build_enso_lifecycle


STATS = ROOT / "data/processed/parquet/statistics"
MASTER = ROOT / "data/processed/parquet/features/nino34_master_weekly.csv"
OUTPUT = ROOT / "data/processed/parquet/modeling/f3_bridge"
INPUTS = {
    "nino_events": STATS / "F3Nino/TabF3NinoB1_eventos.csv",
    "nino_lifecycle": STATS / "F3Nino/TabF3NinoB2_fases_semanais.csv",
    "nina_events": STATS / "F3Nina/TabF3NinaB1_eventos.csv",
    "nina_lifecycle": STATS / "F3Nina/TabF3NinaB2_fases_semanais.csv",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> dict[str, object]:
    sidecar = Path(f"{path}.manifest.json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"manifesto semântico ausente: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if not str(payload.get("run_id", "")).strip():
        raise ValueError(f"run_id ausente: {sidecar}")
    return payload


def _atomic_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", delete=False,
        dir=destination.parent,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    temporary.replace(destination)


def build(*, output_dir: Path = OUTPUT) -> dict[str, object]:
    for path in (*INPUTS.values(), MASTER):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifests = {name: _manifest(path) for name, path in INPUTS.items()}
    nino_events = pd.read_csv(INPUTS["nino_events"])
    nina_events = pd.read_csv(INPUTS["nina_events"])
    if set(nino_events["tipo"].dropna().astype(str)) != {"el_nino"}:
        raise ValueError("F3Nino contém evento fora do escopo El Niño")
    if set(nina_events["tipo"].dropna().astype(str)) != {"la_nina"}:
        raise ValueError("F3Nina contém evento fora do escopo La Niña")
    events = pd.concat([nino_events, nina_events], ignore_index=True)
    if events["event_id"].duplicated().any():
        raise ValueError("event_id duplicado entre os escopos F3")
    events = events.sort_values("onset").reset_index(drop=True)

    master = pd.read_csv(MASTER, usecols=["week_ending_sunday"])
    weeks = pd.DatetimeIndex(pd.to_datetime(master["week_ending_sunday"])).sort_values()
    lifecycle = build_enso_lifecycle(
        events,
        weeks,
        config=EnsoLifecycleConfig(),
    ).reset_index(names="week_ending_sunday")

    # Confere que a ponte não alterou nenhum rótulo ativo emitido pelos dois
    # núcleos isolados. Linhas neutras não carregam inferência.
    indexed = lifecycle.set_index("week_ending_sunday")
    for signal, path in (
        ("el_nino", INPUTS["nino_lifecycle"]),
        ("la_nina", INPUTS["nina_lifecycle"]),
    ):
        scoped = pd.read_csv(path, parse_dates=["week_ending_sunday"])
        scoped = scoped.loc[scoped["tipo"].eq(signal)].set_index("week_ending_sunday")
        observed = indexed.reindex(scoped.index)
        mismatch = (
            observed[["tipo", "fase", "event_id", "estado_enso"]].astype(str)
            != scoped[["tipo", "fase", "event_id", "estado_enso"]].astype(str)
        ).any(axis=1)
        if mismatch.any():
            raise ValueError(
                f"ponte alteraria {int(mismatch.sum())} rótulos ativos de {signal}"
            )

    events_path = output_dir / "events_en_ln.csv"
    lifecycle_path = output_dir / "fases_semanais_en_ln.csv"
    _atomic_csv(events, events_path)
    _atomic_csv(lifecycle, lifecycle_path)
    payload: dict[str, object] = {
        "contract": "f3-isolated-to-predictive-bridge-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "purpose": "rótulos para folds preditivos; nenhuma inferência estatística conjunta",
        "parent_run_ids": {
            name: str(manifest["run_id"])
            for name, manifest in manifests.items()
        },
        "inputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): _sha256(path)
            for path in (*INPUTS.values(), MASTER)
        },
        "outputs": {
            str(path.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": _sha256(path),
                "rows": int(len(frame)),
            }
            for path, frame in ((events_path, events), (lifecycle_path, lifecycle))
        },
        "n_events": {
            "el_nino": int(events["tipo"].eq("el_nino").sum()),
            "la_nina": int(events["tipo"].eq("la_nina").sum()),
        },
    }
    manifest_path = output_dir / "bridge_manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    payload = build(output_dir=output)
    print(
        "[ok] ponte F3: "
        f"El Niño={payload['n_events']['el_nino']} | "
        f"La Niña={payload['n_events']['la_nina']} | {output.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
