from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nino_brasil.data.audit import AuditLog, audit_ledger, write_clean_ledger_copy
from nino_brasil.data.phase2_master import (
    ATMOSPHERIC_COLUMNS,
    OCEAN_COLUMNS,
    PHYSICAL_COLUMNS,
    normalize_era5_daily_units,
    seam_audit,
    source_aware_ocean_adjustment,
    validate_master,
    variable_contract_frame,
    vector_wind_stress_x,
)


def test_phase2_contract_publishes_every_contracted_pressure_level() -> None:
    assert len(PHYSICAL_COLUMNS) == 44
    assert len(OCEAN_COLUMNS) == 17
    assert len(ATMOSPHERIC_COLUMNS) == 27
    for prefix in ("u", "v", "q", "z", "omega", "div"):
        for level in (200, 500, 850):
            assert f"{prefix}{level}_anom" in ATMOSPHERIC_COLUMNS
    assert "ocean_source_code" not in PHYSICAL_COLUMNS
    contract = variable_contract_frame()
    physical = contract.loc[contract["is_physical_predictor"], "name"].tolist()
    assert physical == list(PHYSICAL_COLUMNS)
    assert not bool(contract.loc[contract["name"].eq("ocean_source_code"), "is_physical_predictor"].iloc[0])


def test_vector_wind_stress_uses_both_wind_components() -> None:
    u = pd.Series([3.0, -3.0])
    v = pd.Series([4.0, 4.0])
    result = vector_wind_stress_x(u, v, rho_air=1.0, drag_coefficient=1.0)
    np.testing.assert_allclose(result, [15.0, -15.0])


def test_era5_accumulated_flux_unit_and_sign_contract() -> None:
    frame = pd.DataFrame(
        {
            "slhf": [-360_000.0],
            "sshf": [-36_000.0],
            "ssr": [720_000.0],
            "str": [-180_000.0],
            "u10": [2.0],
        }
    )
    converted = normalize_era5_daily_units(frame)
    assert converted.loc[0, "slhf"] == pytest.approx(100.0)
    assert converted.loc[0, "sshf"] == pytest.approx(10.0)
    assert converted.loc[0, "ssr"] == pytest.approx(200.0)
    assert converted.loc[0, "str"] == pytest.approx(-50.0)
    assert converted.loc[0, "u10"] == 2.0


def test_source_aware_adjustment_does_not_pool_ufs_and_glorys() -> None:
    index = pd.date_range("1988-01-01", "1997-12-31", freq="D")
    source = pd.Series(np.where(index.year <= 1992, 1, 2), index=index)
    seasonal = 8.0 * np.sin(2.0 * np.pi * index.dayofyear.to_numpy() / 365.25)
    offset = np.where(source.to_numpy() == 1, 100.0, 250.0)
    trend = np.where(source.to_numpy() == 1, 0.002, -0.003) * np.arange(len(index))
    ssta = np.cos(np.arange(len(index)) / 30.0)
    ocean = pd.DataFrame({"d20_m": offset + seasonal + trend, "nino34_ssta": ssta}, index=index)
    adjusted = source_aware_ocean_adjustment(ocean, source)
    np.testing.assert_allclose(adjusted["nino34_ssta"], ssta)
    for code in (1, 2):
        group_mean = float(adjusted.loc[source.eq(code), "d20_m"].mean())
        assert abs(group_mean) < 0.5
    assert adjusted["d20_m"].std() < ocean["d20_m"].std() / 5.0


def test_seam_audit_keeps_raw_and_adjusted_diagnostics() -> None:
    index = pd.date_range("1992-09-06", periods=40, freq="W-SUN")
    source = pd.Series([1.0] * 20 + [2.0] * 20, index=index)
    raw = pd.DataFrame({name: np.arange(40, dtype=float) for name in OCEAN_COLUMNS}, index=index)
    adjusted = raw - raw.groupby(source).transform("mean")
    table = seam_audit(raw, adjusted, source, window_weeks=10)
    assert set(table["representation"]) == {"raw", "source_adjusted_v1"}
    assert set(table["variable"]) == set(OCEAN_COLUMNS)
    assert table["transition_week"].nunique() == 1


def test_master_validator_enforces_schema_and_metadata() -> None:
    index = pd.date_range("1981-01-04", "2026-01-04", freq="W-SUN")
    frame = pd.DataFrame(0.0, index=index, columns=PHYSICAL_COLUMNS)
    frame["ocean_source_code"] = np.where(index.year < 1993, 1.0, 2.0)
    result = validate_master(frame)
    assert result["passou"].all()
    broken = frame.rename(columns={"d20_m": "wrong_name"})
    broken_result = validate_master(broken)
    assert not bool(
        broken_result.loc[broken_result["checagem"].eq("contrato_variaveis_fisicas"), "passou"].iloc[0]
    )


def test_ledger_audit_reports_malformed_incomplete_and_missing_hash(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.jsonl"
    original = (
        json.dumps({"task_id": "a", "status": "started", "timestamp_utc": "2026-01-01T00:00:00Z"})
        + "\n{not json}\n"
        + json.dumps({"task_id": "b", "status": "ok", "timestamp_utc": "2026-01-01T01:00:00Z"})
        + "\n"
    )
    ledger.write_text(original, encoding="utf-8")
    report = audit_ledger(ledger)
    assert report["valid_events"] == 2
    assert report["malformed_lines"] == 1
    assert report["incomplete_task_count"] == 1
    assert report["successful_events_without_sha256_count"] == 1
    assert report["missing_run_id_count"] == 2
    assert ledger.read_text(encoding="utf-8") == original

    clean = tmp_path / "ledger.clean.jsonl"
    counts = write_clean_ledger_copy(ledger, clean)
    assert counts == {"kept": 2, "rejected": 1}
    assert ledger.read_text(encoding="utf-8") == original
    assert len(clean.read_text(encoding="utf-8").splitlines()) == 2
    with pytest.raises(ValueError):
        write_clean_ledger_copy(ledger, ledger, overwrite=True)


def test_audit_log_append_adds_run_event_and_schema_ids(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    AuditLog(path).record(task_id="unit", status="ok")
    event = json.loads(path.read_text(encoding="utf-8"))
    assert event["run_id"].startswith("run_") or event["run_id"]
    assert len(event["event_id"]) == 32
    assert event["schema_version"] == "1.1"
    assert len(event["record_sha256"]) == 64
    assert audit_ledger(path)["record_checksum_issue_count"] == 0


def test_audit_log_concurrent_appends_remain_valid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    log = AuditLog(path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: log.record(task_id=f"task-{value}", status="ok"), range(32)))
    report = audit_ledger(path)
    assert report["valid_events"] == 32
    assert report["malformed_lines"] == 0
    assert report["record_checksum_issue_count"] == 0
