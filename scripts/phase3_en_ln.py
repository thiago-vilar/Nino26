#!/usr/bin/env python3
"""phase3_en_ln.py - Fecha as lacunas da Fase 3 (El Nino E La Nina).

Gera, a partir da matriz-mestre semanal (Fase 2) e do ONI local mensal (so para
delimitar eventos - calibracao), os artefatos que a diretriz exige:

  phase3_events_en_ln.csv            eventos EN e LN 1981-2026, classe, datas
  phase3_event_lifecycle_en_ln.csv   4 periodos (genese/crescimento/pico/decaimento) por evento
  phase3_duracao_por_tipo_classe.csv duracao media por tipo/classe e por fase
  phase3_fase_stats_variaveis.csv    nivel, volatilidade semanal e poder discriminante por (tipo,fase,variavel)
  phase3_discriminantes_por_periodo.csv  ranking das variaveis que delimitam cada periodo
  phase3_pca_por_fase.csv            variancia PCA por (tipo,fase)
  phase3_pca_loadings_por_fase.csv   loadings PCA por (tipo,fase)

Toda estatistica avancada roda no eixo SEMANAL; o mensal so define os eventos.
Sem ML, sem redes neurais.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import kruskal
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "data/processed/parquet/features"
STATS = ROOT / "data/processed/parquet/statistics"
STATS.mkdir(parents=True, exist_ok=True)

LIMIAR = 0.5
MIN_EST = 5
GENESE_SEMANAS = 26
PICO_FRAC = 0.90
FASES = ["genese", "crescimento", "pico", "decaimento"]
EN_CLASSES = [(2.0, "muito_forte"), (1.5, "forte"), (1.0, "moderado"), (0.5, "fraco")]
LN_CLASSES = [(2.0, "muito_forte"), (1.5, "forte"), (1.0, "moderada"), (0.5, "fraca")]


def load_master() -> pd.DataFrame:
    m = pd.read_csv(FEAT / "nino34_master_weekly.csv", parse_dates=["week_ending_sunday"]).set_index("week_ending_sunday")
    return m.drop(columns=[c for c in ["ocean_source_code"] if c in m.columns])


def load_oni() -> pd.Series:
    o = pd.read_csv(FEAT / "nino34_monthly_oisst.csv", parse_dates=["time"])
    o = o[o.get("month_complete", True).astype(bool)]
    return o.set_index("time")["oni_local_c"].astype(float)


def detect_events(oni: pd.Series) -> pd.DataFrame:
    rows = []
    for tipo, sign in (("el_nino", 1.0), ("la_nina", -1.0)):
        cond = (sign * oni) >= LIMIAR
        grp = (cond != cond.shift()).cumsum()
        for _, b in oni.groupby(grp):
            if not cond.loc[b.index[0]] or len(b) < MIN_EST:
                continue
            mag = sign * b
            pico_t = mag.idxmax()
            pico_v = float(oni.loc[pico_t])
            classes = EN_CLASSES if sign > 0 else LN_CLASSES
            classe = next(lbl for thr, lbl in classes if abs(pico_v) >= thr)
            rows.append({"event_id": f"{tipo}_{b.index[0].year}_{b.index[-1].year}",
                         "tipo": tipo, "classe": classe,
                         "onset": b.index[0], "pico": pico_t,
                         "fim": b.index[-1] + pd.offsets.MonthEnd(0),
                         "duracao_meses": int(len(b)), "oni_pico_c": round(pico_v, 3)})
    return pd.DataFrame(rows).sort_values("onset").reset_index(drop=True)


def lifecycle(oni: pd.Series, ev: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    fase = pd.Series("neutro", index=weeks)
    tipo = pd.Series("neutro", index=weeks)
    eid = pd.Series("", index=weeks)

    def mark(t0, t1, f, tp, e, only_neutral=False):
        m = (weeks >= pd.to_datetime(t0)) & (weeks <= pd.to_datetime(t1))
        if only_neutral:
            m = m & (fase.values == "neutro")
        idx = np.where(m)[0]
        fase.iloc[idx] = f; tipo.iloc[idx] = tp; eid.iloc[idx] = e

    for _, e in ev.iterrows():
        sign = 1.0 if e.tipo == "el_nino" else -1.0
        bloco = oni.loc[e.onset:e.fim]; mag = sign * bloco
        plateau = mag[mag >= PICO_FRAC * mag.max()]
        p0, p1 = plateau.index.min(), plateau.index.max() + pd.offsets.MonthEnd(0)
        mark(e.onset, p0 - pd.Timedelta(days=1), "crescimento", e.tipo, e.event_id)
        mark(p0, p1, "pico", e.tipo, e.event_id)
        mark(p1 + pd.Timedelta(days=1), e.fim, "decaimento", e.tipo, e.event_id)
    for _, e in ev.iterrows():
        g0 = pd.to_datetime(e.onset) - pd.Timedelta(weeks=GENESE_SEMANAS)
        mark(g0, pd.to_datetime(e.onset) - pd.Timedelta(days=1), "genese", e.tipo, e.event_id, only_neutral=True)
    out = pd.concat([fase.rename("fase"), tipo.rename("tipo"), eid.rename("event_id")], axis=1)
    out.index.name = "week_ending_sunday"
    return out


def eps2(groups):
    groups = [g.dropna() for g in groups if g.notna().sum() >= 8]
    if len(groups) < 3:
        return np.nan
    H, _ = kruskal(*groups)
    n = sum(len(g) for g in groups); k = len(groups)
    return max((H - k + 1) / (n - k), 0.0)


def main() -> int:
    m = load_master()
    oni = load_oni()
    vars = list(m.columns)
    z = (m - m.mean()) / m.std()

    ev = detect_events(oni)
    ev.to_csv(STATS / "phase3_events_en_ln.csv", index=False)
    print(f"[eventos] EN={int((ev.tipo=='el_nino').sum())} LN={int((ev.tipo=='la_nina').sum())}")

    lc = lifecycle(oni, ev, m.index)
    lc.to_csv(STATS / "phase3_fases_semanais_en_ln.csv")

    # lifecycle por evento (datas e duracao de cada fase)
    rows = []
    for _, e in ev.iterrows():
        sub = lc[lc.event_id == e.event_id]
        for f in FASES:
            wk = sub[sub.fase == f].index
            if len(wk):
                rows.append({"event_id": e.event_id, "tipo": e.tipo, "classe": e.classe, "fase": f,
                             "inicio": wk.min().date(), "fim": wk.max().date(), "duracao_semanas": len(wk)})
    lce = pd.DataFrame(rows)
    lce.to_csv(STATS / "phase3_event_lifecycle_en_ln.csv", index=False)

    # duracao media por tipo/classe e por fase
    dur = (lce.groupby(["tipo", "classe", "fase"])["duracao_semanas"].mean().round(1)
           .reset_index().rename(columns={"duracao_semanas": "duracao_media_semanas"}))
    dur.to_csv(STATS / "phase3_duracao_por_tipo_classe.csv", index=False)

    # estatistica por (tipo, fase, variavel): nivel z, volatilidade semanal, discriminancia
    rows = []
    for tipo in ["el_nino", "la_nina"]:
        for v in vars:
            for f in FASES:
                mask = (lc.tipo == tipo) & (lc.fase == f)
                s = z.loc[mask, v].dropna()
                if len(s) < 8:
                    continue
                vol = m.loc[mask, v].diff().abs().mean()   # volatilidade semanal (|delta| medio)
                rows.append({"tipo": tipo, "fase": f, "variavel": v,
                             "nivel_z_medio": round(float(s.mean()), 3),
                             "volatilidade_sem": round(float(vol), 4) if np.isfinite(vol) else np.nan,
                             "n_semanas": int(len(s))})
    fst = pd.DataFrame(rows)
    fst.to_csv(STATS / "phase3_fase_stats_variaveis.csv", index=False)

    # poder discriminante: quanto cada variavel separa as 4 fases (eps2 de Kruskal), por tipo
    rows = []
    for tipo in ["el_nino", "la_nina"]:
        for v in vars:
            grupos = [z.loc[(lc.tipo == tipo) & (lc.fase == f), v] for f in FASES]
            e2 = eps2(grupos)
            rows.append({"tipo": tipo, "variavel": v, "epsilon2_entre_fases": round(e2, 3) if np.isfinite(e2) else np.nan})
    disc = pd.DataFrame(rows).sort_values(["tipo", "epsilon2_entre_fases"], ascending=[True, False])
    disc.to_csv(STATS / "phase3_discriminantes_por_periodo.csv", index=False)

    # PCA/EOF por fase e por tipo
    var_rows, load_rows = [], []
    for tipo in ["el_nino", "la_nina"]:
        for f in FASES:
            X = z.loc[(lc.tipo == tipo) & (lc.fase == f), vars].dropna()
            if len(X) < max(10, len(vars) // 2):
                continue
            p = PCA().fit(StandardScaler().fit_transform(X.values))
            evr = p.explained_variance_ratio_
            for i in range(min(4, len(evr))):
                var_rows.append({"tipo": tipo, "fase": f, "componente": f"PC{i+1}",
                                 "var_explicada": round(float(evr[i]), 3),
                                 "var_acumulada": round(float(evr[:i+1].sum()), 3), "n_semanas": len(X)})
            for vi, vname in enumerate(vars):
                load_rows.append({"tipo": tipo, "fase": f, "variavel": vname,
                                  "PC1": round(float(p.components_[0][vi]), 3),
                                  "PC2": round(float(p.components_[1][vi]), 3) if len(evr) > 1 else np.nan})
    pd.DataFrame(var_rows).to_csv(STATS / "phase3_pca_por_fase.csv", index=False)
    pd.DataFrame(load_rows).to_csv(STATS / "phase3_pca_loadings_por_fase.csv", index=False)

    print("=== duracao media por tipo/classe/fase (semanas) ===")
    print(dur.to_string(index=False))
    print("\n=== top variaveis discriminantes das 4 fases ===")
    for tipo in ["el_nino", "la_nina"]:
        top = disc[disc.tipo == tipo].head(5)
        print(f"  {tipo}:", ", ".join(f"{r.variavel}({r.epsilon2_entre_fases})" for _, r in top.iterrows()))
    print("\n[ok] 7 tabelas gravadas em data/processed/parquet/statistics/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
