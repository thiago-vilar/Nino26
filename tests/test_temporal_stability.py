import numpy as np
import pandas as pd
import pytest

from nino_brasil.stats.temporal_stability import (
    leave_one_event_out_correlation,
    moving_block_bootstrap_correlation,
    summarize_correlation_stability,
)


def test_moving_block_bootstrap_is_reproducible_and_contains_strong_signal():
    rng = np.random.default_rng(11)
    x = np.cumsum(rng.normal(size=240))
    y = 0.8 * x + rng.normal(scale=0.8, size=240)

    first = moving_block_bootstrap_correlation(x, y, block_length=12, n_boot=250, random_state=7)
    second = moving_block_bootstrap_correlation(x, y, block_length=12, n_boot=250, random_state=7)

    assert np.allclose(first.bootstrap_r, second.bootstrap_r, equal_nan=True)
    assert first.ci_low <= first.observed_r <= first.ci_high
    assert first.sign_agreement == 1.0
    assert first.n_samples == 240


def test_moving_block_bootstrap_validates_inputs():
    with pytest.raises(ValueError, match="block_length"):
        moving_block_bootstrap_correlation([1, 2, 3], [1, 2, 3], block_length=1)
    with pytest.raises(ValueError, match="constant"):
        moving_block_bootstrap_correlation([1, 1, 1, 1], [1, 2, 3, 4], block_length=2)


@pytest.mark.parametrize("gap_kind", ["nan", "missing_week"])
def test_moving_blocks_never_cross_a_temporal_gap(gap_kind):
    index = pd.date_range("2000-01-02", periods=10, freq="W-SUN")
    x = pd.Series(np.arange(10, dtype=float), index=index)
    y = x.copy()
    if gap_kind == "nan":
        x.iloc[5] = np.nan
    else:
        x = x.drop(index[5])
        y = y.drop(index[5])

    # The two contiguous runs are shorter than six weeks. The old compressed
    # implementation incorrectly joined them and admitted a six-week block.
    with pytest.raises(ValueError, match="cannot cross gaps"):
        moving_block_bootstrap_correlation(x, y, block_length=6, n_boot=10, random_state=1)


def test_leave_one_event_out_keeps_unlabelled_dates_and_reports_influence():
    index = pd.date_range("2000-01-02", periods=12, freq="W-SUN")
    x = pd.Series(np.arange(12, dtype=float), index=index)
    y = x.copy()
    y.iloc[4:6] *= -3
    labels = pd.Series(pd.NA, index=index, dtype="object")
    labels.iloc[1:3] = "event_a"
    labels.iloc[4:6] = "event_b"

    result = leave_one_event_out_correlation(x, y, labels, min_samples=3).set_index("evento_removido")

    assert set(result.index) == {"event_a", "event_b"}
    assert result.loc["event_a", "n_removido"] == 2
    assert result.loc["event_a", "n_restante"] == 10
    assert result.loc["event_b", "r_sem_evento"] > result.loc["event_a", "r_sem_evento"]


def test_summary_has_no_structural_stability_gate():
    index = pd.date_range("2000-01-02", periods=120, freq="W-SUN")
    x = pd.Series(np.linspace(-1, 1, len(index)), index=index)
    y = x + 0.05 * np.sin(np.arange(len(index)))
    labels = pd.Series(pd.NA, index=index, dtype="object")
    labels.iloc[20:30] = "event_a"
    bootstrap = moving_block_bootstrap_correlation(x, y, block_length=8, n_boot=100, random_state=3)
    loo = leave_one_event_out_correlation(x, y, labels)

    summary = summarize_correlation_stability(bootstrap, loo)

    assert "estavel" not in summary
    assert "breakpoint" not in summary
    assert summary["loo_eventos_n"] == 1
    assert summary["bootstrap_fracao_mesmo_sinal"] == 1.0
