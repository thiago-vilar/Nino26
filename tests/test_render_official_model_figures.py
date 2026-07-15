from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd
import pytest

from nino_brasil.artifacts import sha256_file
from scripts import render_official_model_figures as renderer


def _make_run(
    root: Path,
    *,
    phase: int,
    suffix: str,
    finished_at: str,
    model: str = "",
    mode: str = "official",
    status: str = "complete",
    role: str = "",
    tables: dict[str, pd.DataFrame] | None = None,
) -> Path:
    run_id = f"F{phase}_{suffix}"
    directory = root / "data" / "processed" / "runs" / "official" / f"fase{phase}" / run_id
    table_dir = directory / "tables"
    table_dir.mkdir(parents=True)
    table_rows = []
    for name, frame in (tables or {}).items():
        path = table_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        table_rows.append(
            {
                "table": path.name,
                "path": f"tables/{path.name}",
                "rows": len(frame),
                "columns": frame.shape[1],
                "sha256": sha256_file(path),
            }
        )
    pd.DataFrame(
        table_rows,
        columns=("table", "path", "rows", "columns", "sha256"),
    ).to_csv(directory / "tables_manifest.csv", index=False)
    parameters = {"model": model} if model else {}
    if role:
        parameters["role"] = role
    (directory / "run_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "nino26-run-v1",
                "run_id": run_id,
                "phase": phase,
                "mode": mode,
                "status": status,
                "finished_at": finished_at,
                "parameters": parameters,
            }
        ),
        encoding="utf-8",
    )
    return directory


def _no_problems(_directory: str | Path) -> pd.DataFrame:
    return pd.DataFrame()


def test_discovery_is_fail_closed_and_latest_uses_finished_timestamp(tmp_path: Path) -> None:
    older = _make_run(
        tmp_path,
        phase=5,
        suffix="zzz_older",
        finished_at="2026-07-12T12:00:00+00:00",
        model="rf",
    )
    newer = _make_run(
        tmp_path,
        phase=5,
        suffix="aaa_newer",
        finished_at="2026-07-13T12:00:00+00:00",
        model="rf",
    )
    _make_run(
        tmp_path,
        phase=5,
        suffix="smoke_in_wrong_tree",
        finished_at="2026-07-14T12:00:00+00:00",
        model="rf",
        mode="smoke",
    )
    invalid = _make_run(
        tmp_path,
        phase=5,
        suffix="invalid_hash",
        finished_at="2026-07-15T12:00:00+00:00",
        model="rf",
    )

    def validate(directory: str | Path) -> pd.DataFrame:
        if Path(directory) == invalid:
            return pd.DataFrame([{"type": "hash_mismatch", "item": "table.csv"}])
        return pd.DataFrame()

    runs = renderer.discover_valid_official_runs(tmp_path, 5, validator=validate)
    assert {run.directory for run in runs} == {older.resolve(), newer.resolve()}
    selected = renderer.select_latest_official_runs(runs, 5)
    assert selected["rf"].directory == newer.resolve()


def test_f6_discovery_requires_merged_field_gate_role(tmp_path: Path) -> None:
    _make_run(
        tmp_path,
        phase=6,
        suffix="shard",
        finished_at="2026-07-13T12:00:00+00:00",
        model="rf",
    )
    merged = _make_run(
        tmp_path,
        phase=6,
        suffix="merged",
        finished_at="2026-07-13T13:00:00+00:00",
        model="rf",
        role="merge_pixel_shards_and_field_gate",
    )
    runs = renderer.discover_valid_official_runs(tmp_path, 6, validator=_no_problems)
    assert [run.directory for run in runs] == [merged.resolve()]


def test_sidecar_is_deterministic_and_binds_artifact_to_run(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=5,
        suffix="sidecar",
        finished_at="2026-07-13T12:00:00+00:00",
        model="rf",
        tables={"scientific_gate": pd.DataFrame({"component": ["h04"], "value": [0.1]})},
    )
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    run = renderer.OfficialRun(
        directory=directory.resolve(),
        manifest=manifest,
        finished_at=datetime(2026, 7, 13, 12, tzinfo=timezone.utc),
        model="rf",
    )
    source = renderer._declared_table(run, "scientific_gate", required_columns=("value",))
    sidecar = renderer.write_table_sidecar(run, source)
    first_bytes = sidecar.read_bytes()
    renderer.write_table_sidecar(run, source)
    assert sidecar.read_bytes() == first_bytes
    payload = json.loads(first_bytes)
    assert payload["run_id"] == directory.name
    assert Path(payload["artifact"]["path"]) == source.path
    assert payload["artifact"]["sha256"] == sha256_file(source.path)
    assert payload["artifact"]["rows"] == 1
    assert payload["artifact"]["columns"] == 2


def _tables_for_phase(phase: int) -> dict[str, pd.DataFrame]:
    if phase == 5:
        return {
            "state_importance_oos": pd.DataFrame(
                {
                    "state": ["el_nino_genese", "la_nina_pico"],
                    "predictor": ["nino34_ssta", "wwv"],
                    "delta_brier_permutation_oos": [0.04, -0.01],
                }
            ),
            "scientific_gate": pd.DataFrame(
                {
                    "component": ["classification_h04", "event_dimension_peak"],
                    "value": [-0.1, 0.2],
                    "gate_pass": [False, True],
                }
            ),
        }
    if phase == 6:
        return {
            "field_gate": pd.DataFrame(
                {
                    "condition": ["el_nino_genese", "la_nina_pico"],
                    "lag_weeks": [4, 8],
                    "area_weighted_mean_minimum_pixel_skill": [0.1, -0.2],
                    "fraction_pixels_positive_skill": [0.7, 0.4],
                    "gate_pass": [True, False],
                }
            )
        }
    if phase == 7:
        return {
            "scientific_gate": pd.DataFrame(
                {
                    "mean_skill_f1_vs_best_persistence_or_seasonal": [0.05],
                    "skill_f1_f7_minus_best_f5_paired": [-0.02],
                    "scientific_gate_pass": [False],
                }
            ),
            "scalar_variable_importance_oos": pd.DataFrame(
                {"variable": ["wwv"], "delta_brier_occlusion_oos": [0.03]}
            ),
            "spatial_channel_importance_oos": pd.DataFrame(
                {"channel": ["sst"], "delta_brier_occlusion_oos": [0.02]}
            ),
        }
    if phase == 8:
        return {
            "confirmatory_gate_by_condition": pd.DataFrame(
                {
                    "condition": ["el_nino_genese", "la_nina_pico"],
                    "skill_vs_persistence": [0.1, -0.1],
                    "skill_vs_f6": [0.02, -0.03],
                    "gate_pass": [True, False],
                }
            ),
            "pixel_metrics": pd.DataFrame(
                {
                    "fold": ["fold_1", "fold_1"],
                    "lat": [-10.0, -9.95],
                    "lon": [-50.0, -49.95],
                    "is_brazil": [True, True],
                    "skill_rmse_vs_baseline": [0.1, -0.2],
                }
            ),
            "input_importance_oos": pd.DataFrame(
                {"variable": ["wwv", "sst"], "delta_rmse_occlusion_oos": [0.04, -0.01]}
            ),
        }
    raise AssertionError(phase)


@pytest.mark.parametrize(
    ("phase", "expected_codes"),
    [
        (5, {"Fig_5A01_rf", "Fig_5A02_rf"}),
        (6, {"Fig_6A01_rf"}),
        (7, {"Fig_7A01", "Fig_7A02"}),
        (8, {"Fig_8A01", "Fig_8A02"}),
    ],
)
def test_renderers_use_only_path_sources_with_adjacent_sidecars(
    tmp_path: Path,
    phase: int,
    expected_codes: set[str],
) -> None:
    model = "rf" if phase in (5, 6) else ""
    role = "merge_pixel_shards_and_field_gate" if phase == 6 else ""
    directory = _make_run(
        tmp_path,
        phase=phase,
        suffix="official",
        finished_at="2026-07-13T12:00:00+00:00",
        model=model,
        role=role,
        tables=_tables_for_phase(phase),
    )
    calls: list[tuple[str, dict[str, object]]] = []

    def spy(_figure, code: str, **kwargs: object) -> Path:
        calls.append((code, kwargs))
        return tmp_path / f"{code}.png"

    report = renderer.render_phase(
        tmp_path,
        phase,
        validator=_no_problems,
        registrar=spy,
    )
    assert set(report.rendered) == expected_codes
    assert {code for code, _ in calls} == expected_codes
    for _code, kwargs in calls:
        assert kwargs["run_id"] == directory.name
        assert kwargs["notebook"] == renderer.NOTEBOOKS[phase]
        fontes = kwargs["fontes"]
        assert isinstance(fontes, dict)
        assert fontes
        for source in fontes.values():
            assert isinstance(source, Path)
            sidecar = source.with_suffix(source.suffix + ".manifest.json")
            assert sidecar.is_file()
            assert json.loads(sidecar.read_text(encoding="utf-8"))["run_id"] == directory.name


def test_tampered_declared_table_is_skipped_and_strict_mode_fails(tmp_path: Path) -> None:
    directory = _make_run(
        tmp_path,
        phase=5,
        suffix="tampered",
        finished_at="2026-07-13T12:00:00+00:00",
        model="rf",
        tables=_tables_for_phase(5),
    )
    table = directory / "tables" / "scientific_gate.csv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = renderer.render_phase(
        tmp_path,
        5,
        validator=_no_problems,
        registrar=lambda *_args, **_kwargs: tmp_path / "unused.png",
    )
    assert "Fig_5A01_rf" in report.rendered
    assert any(item.startswith("Fig_5A02_rf: hash divergente") for item in report.skipped)

    with pytest.raises(renderer.EvidenceError, match="Figuras obrigatórias"):
        renderer.render_phase(
            tmp_path,
            5,
            strict=True,
            validator=_no_problems,
            registrar=lambda *_args, **_kwargs: tmp_path / "unused.png",
        )
