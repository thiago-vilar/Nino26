from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import run_fase4c_regional as phase4c
from scripts import run_fase4d_targets as phase4d


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(phase4c, "ROOT", tmp_path)
    input_path = tmp_path / "inputs" / "source.csv"
    output_path = tmp_path / "outputs" / "result.csv"
    input_path.parent.mkdir(parents=True)
    output_path.parent.mkdir(parents=True)
    input_path.write_text("x\n1\n", encoding="utf-8")
    frame = pd.DataFrame({"condition": ["el_nino_genese"], "value": [1.0]})
    frame.to_csv(output_path, index=False)
    phase4c.write_phase4_csv_manifests(
        [(output_path, frame)],
        run_id="F4C_TEST",
        stage="F4C",
        contract={"target_build_id": "target-test"},
        inputs=[input_path],
    )
    return input_path, output_path


def test_phase4_sidecar_binds_artifact_and_inputs(tmp_path, monkeypatch):
    _, output_path = _fixture(tmp_path, monkeypatch)

    phase4c.verify_phase4_output_manifest(
        output_path, expected_run_id="F4C_TEST"
    )
    sidecar = json.loads(
        Path(f"{output_path}.manifest.json").read_text(encoding="utf-8")
    )
    assert sidecar["contract"]["phase"] == "F4"
    assert sidecar["contract"]["stage"] == "F4C"
    assert sidecar["artifact"]["rows"] == 1
    assert len(sidecar["artifact"]["sha256"]) == 64
    assert len(sidecar["inputs"][0]["sha256"]) == 64


def test_phase4_sidecar_rejects_artifact_drift(tmp_path, monkeypatch):
    _, output_path = _fixture(tmp_path, monkeypatch)
    output_path.write_text("condition,value\nel_nino_genese,999\n", encoding="utf-8")

    with pytest.raises(ValueError, match="artifact hash drift"):
        phase4c.verify_phase4_output_manifest(
            output_path, expected_run_id="F4C_TEST"
        )


def test_phase4_sidecar_rejects_input_drift(tmp_path, monkeypatch):
    input_path, output_path = _fixture(tmp_path, monkeypatch)
    input_path.write_text("x\n2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="input hash drift"):
        phase4c.verify_phase4_output_manifest(
            output_path, expected_run_id="F4C_TEST"
        )


def test_ibge_bundle_contract_changes_when_any_component_changes(tmp_path, monkeypatch):
    region = tmp_path / "regions" / "regions.shp"
    biome = tmp_path / "biomes" / "biomes.shp"
    for base in (region, biome):
        base.parent.mkdir(parents=True)
        for suffix in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
            base.with_suffix(suffix).write_text(
                f"{base.stem}:{suffix}", encoding="utf-8"
            )
    monkeypatch.setattr(phase4c, "REG_SHP", region)
    monkeypatch.setattr(phase4c, "BIO_SHP", biome)

    before = phase4c.ibge_geometry_contract()
    region.with_suffix(".dbf").write_text("changed", encoding="utf-8")
    after = phase4c.ibge_geometry_contract()

    assert before["ibge_biomes_bundle_sha256"] == after["ibge_biomes_bundle_sha256"]
    assert before["ibge_regions_bundle_sha256"] != after["ibge_regions_bundle_sha256"]
    assert before["ibge_geometry_bundle_sha256"] != after["ibge_geometry_bundle_sha256"]
    assert len(phase4c.ibge_geometry_input_paths()) == 10


def test_membership_cache_requires_current_geometry_contract(tmp_path, monkeypatch):
    membership_path = tmp_path / "membership.parquet"
    geometry = {
        "ibge_regions_bundle_sha256": "r" * 64,
        "ibge_biomes_bundle_sha256": "b" * 64,
        "ibge_geometry_bundle_sha256": "g" * 64,
    }
    pd.DataFrame(
        {
            "grid_hash": ["grid"],
            **{column: [value] for column, value in geometry.items()},
        }
    ).to_parquet(membership_path, index=False)
    monkeypatch.setattr(phase4c, "MEMBERSHIP_EXACT", membership_path)

    loaded = phase4c.load_membership(
        None,
        pd.DataFrame(),
        centroid_quick=False,
        grid_hash="grid",
        geometry_contract=geometry,
    )
    assert len(loaded) == 1

    stale = dict(geometry)
    stale["ibge_geometry_bundle_sha256"] = "x" * 64
    with pytest.raises(ValueError, match="current grid/IBGE geometry"):
        phase4c.load_membership(
            None,
            pd.DataFrame(),
            centroid_quick=False,
            grid_hash="grid",
            geometry_contract=stale,
        )


def test_phase4_directory_sidecar_rejects_tree_drift(tmp_path, monkeypatch):
    monkeypatch.setattr(phase4c, "ROOT", tmp_path)
    input_path = tmp_path / "input.csv"
    output_path = tmp_path / "atlas"
    input_path.write_text("x\n1\n", encoding="utf-8")
    output_path.mkdir()
    (output_path / "chunk").write_text("one", encoding="utf-8")
    phase4c.write_phase4_directory_manifest(
        output_path,
        run_id="F4C_TEST",
        stage="F4C",
        contract={"selection_contract": "canonical_all_31_physical_variables"},
        inputs=[input_path],
    )
    phase4c.verify_phase4_output_manifest(
        output_path, expected_run_id="F4C_TEST"
    )

    (output_path / "chunk").write_text("two", encoding="utf-8")
    with pytest.raises(ValueError, match="tree hash drift"):
        phase4c.verify_phase4_output_manifest(
            output_path, expected_run_id="F4C_TEST"
        )


def test_f4d_refuses_quick_or_underpowered_f4c_manifest(tmp_path, monkeypatch):
    _, output_path = _fixture(tmp_path, monkeypatch)
    sidecar_path = Path(f"{output_path}.manifest.json")
    manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))
    manifest["contract"].update(
        {
            "stage": "F4C_QUICK",
            "selection_contract": "quick:key-predictor",
            "predictor_count": 1,
            "predictor_names": ["nino34_ssta"],
            "predictor_catalog_sha256": phase4c.predictor_catalog_sha256(
                ["nino34_ssta"]
            ),
            "field_permutations": 19,
        }
    )
    sidecar_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(phase4d, "ROOT", tmp_path)

    with pytest.raises(ValueError, match="non-canonical F4C stage"):
        phase4d.verify_canonical_f4c_output(
            output_path, expected_run_id="F4C_TEST"
        )

    manifest["contract"].update(
        {
            "stage": "F4C",
            "selection_contract": "canonical_all_31_physical_variables",
            "predictor_count": 31,
            "predictor_names": list(phase4c.PACIFIC_VARS),
            "predictor_catalog_sha256": phase4c.predictor_catalog_sha256(
                phase4c.PACIFIC_VARS
            ),
            "field_permutations": 19,
        }
    )
    sidecar_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="at least 199"):
        phase4d.verify_canonical_f4c_output(
            output_path, expected_run_id="F4C_TEST"
        )
