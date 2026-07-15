from __future__ import annotations

from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS, specs_for_phase


def test_catalog_has_unique_codes_and_paths():
    codes = [spec.code for spec in CANONICAL_NOTEBOOKS]
    paths = [spec.relative_path for spec in CANONICAL_NOTEBOOKS]
    assert len(codes) == len(set(codes)) == 32
    assert len(paths) == len(set(paths)) == 32


def test_every_f3_signal_has_all_scientific_blocks():
    expected = set("ABCDEFGHIKL")
    for enso_type, marker in (("el_nino", "F3Nino"), ("la_nina", "F3Nina")):
        specs = specs_for_phase(3, enso_type)
        assert {spec.code.removeprefix(marker) for spec in specs} == expected


def test_f3_nino_has_non_generic_scientific_context():
    for spec in specs_for_phase(3, "el_nino"):
        assert len(spec.context) >= 150
        assert len(spec.hypothesis_statement) >= 100
        assert len(spec.method_rationale) >= 150
        assert len(spec.expected_outputs) >= 3
        assert len(spec.references) >= 2
        assert spec.hypothesis_statement != "HIP0"


def test_phase4_is_only_the_canonical_c_d_chain_per_signal():
    assert [spec.code for spec in specs_for_phase(4, "el_nino")] == [
        "F4NinoC",
        "F4NinoD",
    ]
    assert [spec.code for spec in specs_for_phase(4, "la_nina")] == [
        "F4NinaC",
        "F4NinaD",
    ]
