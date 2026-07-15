from __future__ import annotations

import pytest

from nino_brasil.artifact_codes import (
    assert_artifact_pair,
    figure_code,
    notebook_code_for,
    parse_artifact_code,
    parse_notebook_code,
    table_code,
)


@pytest.mark.parametrize(
    "value,namespace,enso_type",
    [
        ("F2Z", "F2", None),
        ("F3NinoA", "F3Nino", "el_nino"),
        ("F3NinaL", "F3Nina", "la_nina"),
        ("F4NinoC", "F4Nino", "el_nino"),
        ("F8A", "F8", None),
    ],
)
def test_notebook_code_contract(value, namespace, enso_type):
    parsed = parse_notebook_code(value)
    assert parsed.namespace == namespace
    assert parsed.enso_type == enso_type


def test_artifact_pair_uses_the_same_notebook_precode():
    figure = figure_code("F3NinoA", 1)
    table = table_code("F3NinoA", 1)
    assert figure == "FigF3NinoA1"
    assert table == "TabF3NinoA1"
    assert_artifact_pair(figure, table)


def test_descriptive_slug_does_not_change_pair_key():
    figure = figure_code("F3NinaA", 2, slug="SSTA média")
    table = table_code("F3NinaA", 2, slug="resumo numérico")
    assert figure == "FigF3NinaA2_ssta_media"
    assert table == "TabF3NinaA2_resumo_numerico"
    assert parse_artifact_code(figure).paired_precode == "TabF3NinaA2"
    assert_artifact_pair(figure, table)


def test_phase_signal_rules_are_strict():
    with pytest.raises(ValueError):
        parse_notebook_code("F3A")
    with pytest.raises(ValueError):
        parse_notebook_code("F5NinoA")
    assert notebook_code_for(4, "C", "la_nina") == "F4NinaC"


def test_mismatched_pair_is_rejected():
    with pytest.raises(ValueError):
        assert_artifact_pair("FigF3NinoA1", "TabF3NinoA2")
