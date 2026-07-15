from __future__ import annotations

import csv
from pathlib import Path

from scripts import quarantine_stale_derived_outputs as quarantine


def _write_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["codigo", "arquivo", "run_id"])
        writer.writeheader()
        writer.writerows(
            [
                {"codigo": "Fig_4A01", "arquivo": "fase4/Fig_4A01.png", "run_id": ""},
                {"codigo": "Fig_3A01", "arquivo": "fase3/Fig_3A01.png", "run_id": "F3"},
            ]
        )


def _configure(tmp_path: Path, monkeypatch, *, active_exists: bool) -> Path:
    relative = "data/processed/figures/fase4/Fig_4A01.png"
    quarantine_root = tmp_path / "data/quarantine/stale_derived"
    quarantined = quarantine_root / "20260713T000000Z" / relative
    quarantined.parent.mkdir(parents=True)
    quarantined.write_bytes(b"historical")
    if active_exists:
        active = tmp_path / relative
        active.parent.mkdir(parents=True)
        active.write_bytes(b"still-active")
    registry = tmp_path / "data/processed/figuras_manifesto.csv"
    _write_registry(registry)
    monkeypatch.setattr(quarantine, "ROOT", tmp_path)
    monkeypatch.setattr(quarantine, "QUARANTINE", quarantine_root)
    monkeypatch.setattr(quarantine, "FIGURE_REGISTRY", registry)
    monkeypatch.setattr(
        quarantine,
        "KNOWN_STALE",
        (
            quarantine.Candidate(
                relative,
                "historical_figure",
                "test",
            ),
        ),
    )
    return registry


def test_registry_reconciliation_archives_and_removes_quarantined_row(
    tmp_path, monkeypatch
):
    registry = _configure(tmp_path, monkeypatch, active_exists=False)
    destination = quarantine.QUARANTINE / "reconcile"

    result = quarantine._reconcile_quarantined_registry(destination)

    assert result["removed_codes"] == ["Fig_4A01"]
    assert (destination / "figuras_manifesto_before.csv").is_file()
    assert (destination / "figuras_manifesto_quarantined_rows.csv").is_file()
    with registry.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["codigo"] for row in rows] == ["Fig_3A01"]


def test_registry_reconciliation_keeps_row_while_active_figure_exists(
    tmp_path, monkeypatch
):
    registry = _configure(tmp_path, monkeypatch, active_exists=True)

    result = quarantine._reconcile_quarantined_registry(
        quarantine.QUARANTINE / "reconcile"
    )

    assert result["removed_codes"] == []
    with registry.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert {row["codigo"] for row in rows} == {"Fig_4A01", "Fig_3A01"}
