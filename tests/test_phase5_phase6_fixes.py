"""Testes das correcoes do parecer 2026-07-10.

Cobrem: (1) Y_duracao_sem por diferenca de datas (bug x13 corrigido);
(2) anomalizacao harmonica + detrend das variaveis oceanicas cruas;
(3) baselines semana-do-ano/persistencia no classificador de fases (Fase 5);
(4) baselines de persistencia/climatologia sazonal na Fase 6;
(5) regressao por evento com LOO; (6) escrita atomica de CSV.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.io_utils import write_csv_atomic
from nino_brasil.models.phase5_cycle_ml import (
    WEEKS_PER_MONTH,
    build_event_targets,
    build_lagged_features,
    fit_event_regressions,
    fit_phase_classifier,
    precursor_features_at_onset,
    prepare_pacific_predictors,
)
from nino_brasil.models.phase6_brazil_ml import fit_unit_teleconnection
from nino_brasil.stats.climatology import harmonic_anomaly_matrix


def synthetic_weekly_master(n_years: int = 30, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.date_range("1991-01-06", periods=52 * n_years, freq="W-SUN")
    t = np.arange(len(index), dtype=float)
    annual = np.sin(2 * np.pi * index.dayofyear.to_numpy() / 365.2425)
    enso = np.sin(2 * np.pi * t / (52 * 4.0))  # pseudo-ciclo interanual
    frame = pd.DataFrame(index=index)
    frame["nino34_ssta"] = enso + 0.1 * rng.normal(size=len(index))
    frame["d20_m"] = 128.0 + 12.0 * annual + 0.02 * t + 8.0 * enso + rng.normal(size=len(index))
    frame["tau_x_anom"] = 0.01 * enso + 0.002 * rng.normal(size=len(index))
    return frame


class TestEventTargets(unittest.TestCase):
    def events_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_id": ["el_nino_1997_1998", "la_nina_1998_2001"],
                "tipo": ["el_nino", "la_nina"],
                "onset": [pd.Timestamp("1997-06-01"), pd.Timestamp("1998-06-01")],
                "pico": [pd.Timestamp("1997-12-01"), pd.Timestamp("1999-12-01")],
                "fim": [pd.Timestamp("1998-04-30"), pd.Timestamp("2001-03-31")],
                "duracao_estacoes": [11, 34],
                "oni_pico_c": [2.146, -1.704],
            }
        )

    def test_duration_from_dates_not_13x(self):
        events = self.events_frame()
        targets = build_event_targets(events)
        dur = targets.set_index("event_id")["Y_duracao_sem"]
        expected = (
            (pd.to_datetime(events["fim"]) - pd.to_datetime(events["onset"])).dt.days / 7.0
        )
        expected.index = events["event_id"]
        # EN 97/98 ~47.6 sem (nao 143!); LN 98-01 ~147.7 sem (nao 442!)
        self.assertAlmostEqual(dur["el_nino_1997_1998"], expected["el_nino_1997_1998"], places=6)
        self.assertAlmostEqual(dur["la_nina_1998_2001"], expected["la_nina_1998_2001"], places=6)
        self.assertLess(dur["el_nino_1997_1998"], 60)
        self.assertTrue((dur < 200).all(), "nenhum evento pode durar 8 anos")

    def test_fallback_without_fim_uses_weeks_per_month(self):
        events = self.events_frame().drop(columns=["fim"])
        targets = build_event_targets(events)
        dur = targets.set_index("event_id")["Y_duracao_sem"]
        self.assertAlmostEqual(dur["el_nino_1997_1998"], 11 * WEEKS_PER_MONTH, places=3)
        # coerente (mesma ordem) com a duracao por datas: ~47.8 vs 47.6
        self.assertLess(abs(dur["el_nino_1997_1998"] - 333 / 7.0), 2.0)

    def test_tempo_para_pico_unchanged(self):
        targets = build_event_targets(self.events_frame())
        row = targets.set_index("event_id").loc["el_nino_1997_1998"]
        self.assertAlmostEqual(row["Y_tempo_para_pico_sem"], 183 / 7.0, places=2)


class TestHarmonicAnomaly(unittest.TestCase):
    def test_removes_seasonal_cycle_and_trend(self):
        master = synthetic_weekly_master()
        out = harmonic_anomaly_matrix(master, ["d20_m"], base=("1995-01-01", "2014-12-31"))
        anom = out["d20_m"]
        # ciclo anual removido: amplitude da climatologia semanal cai >80%
        def seasonal_amplitude(s: pd.Series) -> float:
            clim = s.groupby(s.index.isocalendar().week).mean()
            return float(clim.max() - clim.min())

        self.assertLess(seasonal_amplitude(anom), 0.2 * seasonal_amplitude(master["d20_m"]))
        # tendencia removida: inclinacao OLS ~0 (era 0.02/semana)
        t = np.arange(len(anom), dtype=float)
        slope = np.polyfit(t, anom.to_numpy(), 1)[0]
        self.assertLess(abs(slope), 0.004)
        # sinal interanual preservado: correlacao alta com o pseudo-ENSO
        self.assertGreater(anom.corr(master["nino34_ssta"]), 0.6)
        # colunas fora do alvo ficam intactas
        pd.testing.assert_series_equal(out["nino34_ssta"], master["nino34_ssta"])

    def test_prepare_pacific_predictors_only_touches_raw_ocean(self):
        master = synthetic_weekly_master()
        prepared = prepare_pacific_predictors(master, ["nino34_ssta", "d20_m", "tau_x_anom"])
        pd.testing.assert_series_equal(prepared["tau_x_anom"], master["tau_x_anom"])
        pd.testing.assert_series_equal(prepared["nino34_ssta"], master["nino34_ssta"])
        self.assertLess(abs(prepared["d20_m"].mean()), 2.0)  # centrada apos anomalia
        self.assertFalse(np.allclose(prepared["d20_m"], master["d20_m"]))


class TestPhaseClassifierBaselines(unittest.TestCase):
    def test_cv_reports_seasonal_and_persistence_baselines(self):
        rng = np.random.default_rng(3)
        index = pd.date_range("2000-01-02", periods=520, freq="W-SUN")
        month = index.month
        labels = pd.Series(
            np.select(
                [np.isin(month, [3, 4, 5]), np.isin(month, [6, 7, 8]),
                 np.isin(month, [12, 1, 2])],
                ["genese", "crescimento", "pico"], default="decaimento",
            ),
            index=index,
        )
        X = pd.DataFrame(
            {"a": rng.normal(size=len(index)), "b": rng.normal(size=len(index))},
            index=index,
        )
        result = fit_phase_classifier(X, labels, model="rf", n_splits=3)
        cv = result.cv_scores
        for column in ("f1_macro", "f1_baseline_semana_do_ano", "f1_baseline_persistencia"):
            self.assertIn(column, cv.columns)
            self.assertTrue(np.isfinite(cv[column]).all())
        # rotulos puramente sazonais: o baseline semana-do-ano deve ser quase
        # perfeito e superar o ML de features aleatorias.
        self.assertGreater(cv["f1_baseline_semana_do_ano"].mean(), 0.9)
        self.assertGreater(
            cv["f1_baseline_semana_do_ano"].mean(), cv["f1_macro"].mean()
        )


class TestEventRegressionLOO(unittest.TestCase):
    def test_loo_metrics_and_predictions(self):
        master = synthetic_weekly_master(n_years=34)
        predictors = ["nino34_ssta", "d20_m"]
        prepared = prepare_pacific_predictors(master, predictors)
        lagged = build_lagged_features(prepared, predictors, lags=(4, 8, 12))
        onsets = pd.date_range("1993-06-06", periods=10, freq="26W-SUN")
        rng = np.random.default_rng(5)
        events = pd.DataFrame(
            {
                "event_id": [f"ev_{i}" for i in range(10)],
                "tipo": ["el_nino", "la_nina"] * 5,
                "onset": onsets,
                "pico": onsets + pd.Timedelta(weeks=20),
                "fim": onsets + pd.Timedelta(weeks=45),
                "duracao_estacoes": 10,
                "oni_pico_c": rng.uniform(0.6, 2.4, size=10) * np.where(np.arange(10) % 2, -1, 1),
            }
        )
        targets = build_event_targets(events)
        onset_feats = precursor_features_at_onset(lagged, events)
        result = fit_event_regressions(onset_feats, targets, model="rf", jitter_copies=1)
        self.assertFalse(result.metrics.empty)
        self.assertEqual(
            set(result.metrics["alvo"]),
            {"Y_pico", "Y_tempo_para_pico_sem", "Y_duracao_sem"},
        )
        self.assertTrue((result.metrics["n_eventos"] == 10).all())
        preds = result.predictions
        self.assertEqual(len(preds), 30)  # 10 eventos x 3 alvos
        self.assertTrue(np.isfinite(preds["previsto_loo"]).all())


class TestPhase6Baselines(unittest.TestCase):
    def test_skill_table_has_strong_baselines(self):
        rng = np.random.default_rng(9)
        index = pd.date_range("1995-01-01", periods=600, freq="W-SUN")
        x = pd.Series(np.sin(2 * np.pi * np.arange(600) / 200.0), index=index)
        lagged = pd.DataFrame({"x_lag4": x.shift(4), "x_lag8": x.shift(8)}, index=index)
        # alvo com sinal defasado + persistencia (autocorrelacao)
        y = 0.7 * x.shift(4) + 0.3 * rng.normal(size=600)
        unit = pd.DataFrame({"regiao_teste": y}, index=index)
        phase_table = pd.DataFrame(
            {"fase": "neutro", "tipo": "neutro", "event_id": ""}, index=index
        )
        result = fit_unit_teleconnection(
            lagged, unit, phase_table, model="rf", conditions=("todas",),
            n_splits=3, min_obs=100,
        )
        self.assertEqual(len(result.skill), 1)
        row = result.skill.iloc[0]
        for column in (
            "rmse_baseline", "rmse_persistencia", "rmse_clim_semana_do_ano",
            "melhor_baseline", "rmse_melhor_baseline",
            "skill_rmse_vs_baseline", "skill_rmse_vs_melhor_baseline",
        ):
            self.assertIn(column, result.skill.columns)
        self.assertTrue(np.isfinite(row["rmse_persistencia"]))
        self.assertTrue(np.isfinite(row["rmse_melhor_baseline"]))
        self.assertLessEqual(row["rmse_melhor_baseline"], row["rmse_baseline"] + 1e-12)
        # com sinal real defasado, o ML deve ter r fora-da-amostra positivo
        self.assertGreater(row["r_ml"], 0.3)


class TestAtomicWrite(unittest.TestCase):
    def test_write_csv_atomic_roundtrip(self):
        import tempfile

        frame = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "tabela.csv"
            write_csv_atomic(frame, path)
            back = pd.read_csv(path)
            pd.testing.assert_frame_equal(back, frame)
            leftovers = [p for p in path.parent.iterdir() if p.suffix == ".tmp"]
            self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
