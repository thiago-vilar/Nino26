from __future__ import annotations

import pandas as pd
import pytest

from scripts.run_fase8_brazil_convlstm import (
    F8_INTERVAL_AGGREGATION_UNIT,
    _f8_confirmatory_gate_pass,
    _f8_interval_calibration_audit,
    _validate_interval90_contract,
    build_parser,
)


CONDITION = "el_nino_genese"


def _components(*rows: tuple[str, str, float, float, int]) -> pd.DataFrame:
    """Build fold, event, coverage, availability, pixel-week-count fixtures."""

    return pd.DataFrame(
        [
            {
                "fold": fold,
                "condition": CONDITION,
                "event_id": event_id,
                "interval90_coverage_event": interval_coverage,
                "interval90_value_coverage_fraction": value_coverage,
                "n_native_pixel_weeks": n_pixel_weeks,
            }
            for (
                fold,
                event_id,
                interval_coverage,
                value_coverage,
                n_pixel_weeks,
            ) in rows
        ]
    )


def test_phase8_defaults_are_central_ic90_with_predeclared_tolerance() -> None:
    args = build_parser().parse_args([])

    assert args.quantiles == (0.05, 0.5, 0.95)
    assert args.max_interval90_absolute_calibration_error == pytest.approx(0.15)
    _validate_interval90_contract(args)

    args.quantiles = (0.1, 0.5, 0.9)
    with pytest.raises(ValueError, match="IC90 central"):
        _validate_interval90_contract(args)

    args.quantiles = (0.05, 0.5, 0.95)
    args.max_interval90_absolute_calibration_error = 0.151
    with pytest.raises(ValueError, match="nao aceita tolerancia mais frouxa"):
        _validate_interval90_contract(args)


def test_phase8_calibration_is_event_equal_not_pixel_week_weighted() -> None:
    # The first event contributes 100,000 native pixel-weeks and the second
    # contributes one.  Both must still receive exactly one event vote.
    components = _components(
        ("fold_1", "event_large", 0.75, 1.0, 100_000),
        ("fold_1", "event_small", 0.95, 1.0, 1),
    )

    events, folds, conditions = _f8_interval_calibration_audit(
        components, conditions=(CONDITION,)
    )

    assert len(events) == 2
    assert folds.loc[0, "interval90_coverage_event_equal"] == pytest.approx(0.85)
    assert conditions.loc[0, "interval90_coverage_event_equal"] == pytest.approx(
        0.85
    )
    assert bool(conditions.loc[0, "interval_calibration_gate_pass"])
    assert conditions.loc[0, "interval_aggregation_unit"] == (
        F8_INTERVAL_AGGREGATION_UNIT
    )


def test_phase8_gate_requires_every_fold_even_when_condition_pool_is_calibrated() -> None:
    # The condition-wide event-equal coverage is exactly 0.90, but fold_1 is
    # outside tolerance.  Pooling cannot hide a miscalibrated OOS fold.
    components = _components(
        ("fold_1", "event_1", 0.60, 1.0, 10),
        ("fold_2", "event_2", 1.00, 1.0, 10),
        ("fold_2", "event_3", 1.00, 1.0, 10),
        ("fold_2", "event_4", 1.00, 1.0, 10),
    )

    _, folds, conditions = _f8_interval_calibration_audit(
        components, conditions=(CONDITION,)
    )

    assert conditions.loc[0, "interval90_coverage_event_equal"] == pytest.approx(
        0.90
    )
    assert conditions.loc[
        0, "maximum_fold_interval90_absolute_calibration_error"
    ] == pytest.approx(0.30)
    assert not bool(
        folds.loc[folds["fold"].eq("fold_1"), "interval_calibration_gate_pass"].iloc[0]
    )
    assert not bool(conditions.loc[0, "interval_calibration_gate_pass"])


@pytest.mark.parametrize(
    ("coverage", "expected_pass"),
    [(0.75, True), (0.749, False)],
)
def test_phase8_calibration_tolerance_is_inclusive(
    coverage: float, expected_pass: bool
) -> None:
    components = _components(("fold_1", "event_1", coverage, 1.0, 10))

    _, _, conditions = _f8_interval_calibration_audit(
        components, conditions=(CONDITION,)
    )

    assert bool(conditions.loc[0, "interval_calibration_gate_pass"]) is expected_pass


def test_phase8_gate_rejects_incomplete_native_interval_values() -> None:
    components = _components(("fold_1", "event_1", 0.90, 0.99, 10))

    _, folds, conditions = _f8_interval_calibration_audit(
        components, conditions=(CONDITION,)
    )

    assert not bool(folds.loc[0, "complete_native_interval_values"])
    assert not bool(conditions.loc[0, "interval_calibration_gate_pass"])


def test_phase8_gate_rejects_event_repeated_across_folds() -> None:
    components = _components(
        ("fold_1", "event_1", 0.90, 1.0, 10),
        ("fold_2", "event_1", 0.90, 1.0, 10),
    )

    with pytest.raises(ValueError, match="whole-event OOS violado"):
        _f8_interval_calibration_audit(components, conditions=(CONDITION,))


def test_phase8_confirmatory_gate_cannot_pass_without_calibration() -> None:
    common = {
        "minimum_event_support_gate_pass": True,
        "persistence_gate_pass": True,
        "f6_comparison_gate_pass": True,
    }

    assert not _f8_confirmatory_gate_pass(
        **common, interval_calibration_gate_pass=False
    )
    assert _f8_confirmatory_gate_pass(
        **common, interval_calibration_gate_pass=True
    )
