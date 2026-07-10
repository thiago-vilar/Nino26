#!/usr/bin/env python3
"""Gera o relatorio interpretativo final da Fase 3."""
from __future__ import annotations

from datetime import datetime
from math import ceil
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
NBDIR = ROOT / "notebooks" / "fase3"
STATS = ROOT / "data" / "processed" / "parquet" / "statistics"
FIGS = ROOT / "data" / "processed" / "figures" / "fase3"


FIGURE_CATALOG = [
    ("3A", "3A1_series_semanais.png", "Series semanais", "Cobertura, unidades e comparacao visual das variaveis fisicas.", "phase3A_cobertura_variaveis.csv"),
    ("3A", "3A2_hovmoller_ssta.png", "Hovmoller SSTA", "Mostra propagacao longitudinal da anomalia de SST na faixa equatorial.", "phase3A_fontes_variaveis.csv"),
    ("3A", "3A3_hovmoller_sla_taux.png", "Hovmoller SLA + vento", "Conecta SLA local e setas de tau_x para leitura qualitativa de Kelvin/acoplamento.", "phase3A_fontes_variaveis.csv"),
    ("3B", "3B1_trajetorias_compostas.png", "Trajetorias por classe", "Compara a evolucao media da SSTA por classe NOAA/ONI.", "phase3B_trajetorias_compostas.csv"),
    ("3B", "3B2_autocorrelacao.png", "Autocorrelacao", "Mede memoria/persistencia da SSTA que qualquer previsao deve superar.", "phase3B_autocorrelacao.csv"),
    ("3B", "3B3_mapa_composto_pico.png", "Mapa composto do pico", "Mostra a assinatura espacial media no pico dos eventos.", "phase3B_mapa_composto_resumo.csv"),
    ("3C", "3C1_heatmap_lags.png", "Heatmap de lags", "Triagem bruta de quais variaveis antecedem a SSTA e em que lag.", "phase3C_ranking_lags.csv"),
    ("3C", "3C2_mapa_lon_lag.png", "Longitude x lag", "Mostra onde no Pacifico equatorial o sinal antecedente aparece por longitude.", "phase3C_lag_correlacoes.csv"),
    ("3D", "3D1_forest_ic95.png", "Forest IC95", "Aplica N_eff, FDR e IC95 para reduzir falsos positivos.", "phase3D_ranking_significativo.csv"),
    ("3D", "3D2_mapa_lon_lag_fdr.png", "Mapa FDR", "Mostra regioes longitude-lag que sobrevivem ao controle estatistico.", "phase3D_testes_completos.csv"),
    ("3E", "Fig_3E1_sensibilidade_bootstrap_loo.png", "Sensibilidade sem breakpoint", "Compara r completo, envelope IC95 do bootstrap movel e faixa leave-one-event-out; nao define exclusao.", "phase3E_sensibilidade_resumo.csv"),
    ("3E", "Fig_3E2_influencia_eventos_loo.png", "Influencia de eventos", "Mostra quanto r muda ao retirar cada evento EN/LN, sem dividir a serie por ano.", "phase3E_leave_one_event_out.csv"),
    ("3F", "3F1_hovmoller_sla_kelvin.png", "Kelvin por SLA", "Diagnostico visual de propagacao oeste-leste por SLA/SSH em eventos fortes.", "phase3F_kelvin_eventos_resumo.csv"),
    ("3F", "3F2_taux_sla_eventos.png", "Vento e SLA", "Resume tau_x_anom na Nino 3.4 junto ao sinal de SLA por evento.", "phase3F_kelvin_eventos_resumo.csv"),
    ("3G", "3G1_composto_ssta_noaa.png", "SSTA por classe", "Compara a evolucao termica media por classe NOAA/ONI.", "phase3G_composto_ssta_classes_noaa.csv"),
    ("3G", "3G2_escalonamento_ssta.png", "Escalonamento termico", "Relaciona pico ONI local, duracao e taxas de crescimento/decaimento.", "phase3G_escalonamento_ssta.csv"),
    ("3G", "3G3_mapa_ssta_lon.png", "SSTA longitude", "Compara fortes/super historicos com a formacao atual 2025/26 por longitude.", "phase3G_mapa_ssta_lon_eventos_forte_super.csv"),
    ("3H", "3H1_compostos_onset.png", "Onset por classe", "Mostra quais variaveis se separam na genese dos eventos.", "phase3H_estado_precursor_por_classe.csv"),
    ("3H", "3H2_ciclo_vida.png", "Ciclo de vida", "Resume genese, crescimento, pico e decaimento com variaveis em z-score.", "phase3H_ciclo_vida_media.csv"),
    ("3I", "3I1_sintese_parecer.png", "Sintese do parecer", "Organiza evidencias do 3D por bloco fisico, lag e metricas continuas de sensibilidade do 3E.", "phase3I_conclusoes_decisao.csv"),
    ("3I", "3I2_antecipacao_pico.png", "Antecipacao", "Mostra variaveis candidatas para antecipar o aquecimento maximo.", "phase3I_conjunto_antecipacao_pico.csv"),
    ("3I", "3I3_previsao_condicional_nested.png", "Nested LOO", "Avalia selecao+ajuste por nested LOO e gera projecao condicional.", "phase3I_nested_loo_metricas.csv"),
    ("3K", "3K1_skill_loo_nested.png", "Skill PCA", "Testa se PCA reduz redundancia sem perder skill preditivo.", "phase3K_previsao_pico_nested_loo_metricas.csv"),
    ("3K", "3K2_scree.png", "Scree PCA", "Mostra quantos componentes explicam a variancia de crescimento.", "phase3K_pca_variancia.csv"),
    ("3K", "3K3_biplot.png", "Biplot PCA", "Mostra agrupamentos fisicos e colinearidade entre variaveis.", "phase3K_pca_loadings.csv"),
    ("3L", "phase3L_ciclo_vida_en_ln.png", "Ciclo de vida EN/LN", "Compara genese, crescimento, pico e decaimento de El Nino e La Nina.", "phase3_event_lifecycle_en_ln.csv"),
    ("3L", "phase3L_duracao_fases_en_ln.png", "Duracao por fase EN/LN", "Resume a duracao media por tipo, classe e fase do ciclo de vida.", "phase3_duracao_por_tipo_classe.csv"),
    ("3L", "phase3L_discriminantes_heatmap.png", "Discriminantes por periodo", "Mostra quais variaveis delimitam melhor as quatro fases por sinal ENSO.", "phase3_discriminantes_por_periodo.csv"),
    ("3L", "phase3L_pca_por_fase.png", "PCA por fase", "Resume a estrutura multivariada por genese, crescimento, pico e decaimento.", "phase3_pca_por_fase.csv"),
]


NOTEBOOK_SUMMARY = [
    ("3A", "Materializa matriz semanal, cobertura e Hovmollers.", "Pergunta: quais variaveis existem, em que forma e janela real?", "Base fisica pronta; nem tudo e anomalia, por isso z-score e usado em comparacoes."),
    ("3B", "Define eventos NOAA/ONI locais, classes, ciclo e memoria.", "Pergunta: como eventos nascem, crescem, picam e decaem?", "Autocorrelacao vira baseline de persistencia."),
    ("3C", "Faz triagem bruta de lags preditivos.", "Pergunta: quem antecede a SSTA e com quantas semanas?", "Ranking bruto guia, mas nao basta sem rigor."),
    ("3D", "Aplica N_eff, FDR e IC95.", "Pergunta: o que sobrevive ao controle estatistico?", "Reduz falsos positivos e define evidencias robustas."),
    ("3E", "Quantifica sensibilidade sem breakpoint.", "Pergunta: a relacao depende da autocorrelacao ou de um evento ENSO isolado?", "Bootstrap movel + LOO por evento informam incerteza; nao criam gate."),
    ("3F", "Avalia leitura qualitativa de ondas de Kelvin por SLA/SSH e vento.", "Pergunta: ha propagacao dinamica compativel com Kelvin?", "Kelvin e diagnostico visual/dinamico, nao detector automatico."),
    ("3G", "Compara SSTA por classe NOAA/ONI e com 2025/26.", "Pergunta: como intensidade, duracao e propagacao longitudinal se organizam?", "SSTA por classe mostra escala termica sem metricas acumuladas auxiliares."),
    ("3H", "Mostra genese e ciclo de vida fisico.", "Pergunta: o estado pre-onset separa classes?", "Recarga cresce antes do pico e descarrega depois."),
    ("3K", "Reduz variaveis por PCA e testa skill.", "Pergunta: quais variaveis sao redundantes?", "PC1/OHC0-300 representa eixo de recarga com parcimonia."),
    ("3I", "Integra parecer e nested LOO.", "Pergunta: quais variaveis predizem o pico e como ler 2025/26?", "Entrega projecao condicional exploratoria, nao operacional."),
    ("3L", "Caracteriza El Nino e La Nina por evento/fase.", "Pergunta: o protocolo completo EN/LN fecha a diretriz da Fase 3?", "Eventos, fases, duracoes, discriminantes e PCA por fase ficam materializados."),
]

PHASE3_COMPLETION_ROWS = [
    {
        "item_pedido": "Separar eventos El Nino e La Nina 1981-2026",
        "estado_atual": "feito",
        "evidencia": "phase3_events_en_ln.csv",
        "produto": "12 El Nino + 11 La Nina pela regra ONI local simetrica (+/-0.5 C)",
    },
    {
        "item_pedido": "Duracao media por tipo, incluindo fortes/super, para El Nino e La Nina",
        "estado_atual": "feito",
        "evidencia": "phase3_duracao_por_tipo_classe.csv",
        "produto": "duracao media por tipo, classe e fase do ciclo de vida",
    },
    {
        "item_pedido": "Classificar genese, crescimento, pico e decaimento de cada evento",
        "estado_atual": "feito",
        "evidencia": "phase3_event_lifecycle_en_ln.csv e phase3_fases_semanais_en_ln.csv",
        "produto": "quatro periodos por evento, para El Nino e La Nina",
    },
    {
        "item_pedido": "Hovmoller, Bjerknes, Kelvin, mapas e ciclos de vida",
        "estado_atual": "feito",
        "evidencia": "3A/3F/3G/3H + 3L",
        "produto": "Hovmoller/Kelvin/mapas do pacote fisico e figuras EN/LN de ciclo e duracao",
    },
    {
        "item_pedido": "PCA e EOF por ciclo de vida",
        "estado_atual": "feito",
        "evidencia": "phase3_pca_por_fase.csv e phase3_pca_loadings_por_fase.csv",
        "produto": "PCA por fase e por sinal; EOF espacial fica como extensao, nao bloqueio da Fase 3",
    },
    {
        "item_pedido": "Quais variaveis delimitam os quatro periodos",
        "estado_atual": "feito",
        "evidencia": "phase3_fase_stats_variaveis.csv e phase3_discriminantes_por_periodo.csv",
        "produto": "nivel, volatilidade semanal e poder discriminante por periodo",
    },
]


def read_csv(name: str) -> pd.DataFrame:
    path = STATS / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def md_table(df: pd.DataFrame, cols: list[str] | None = None, max_rows: int = 20) -> str:
    if df.empty:
        return "_Tabela ainda nao encontrada._"
    out = df.copy()
    if cols:
        out = out[[c for c in cols if c in out.columns]]
    out = out.head(max_rows)
    out = out.fillna("")
    headers = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in out.iterrows():
        vals = []
        for val in row.tolist():
            text = str(val).replace("\n", " ").replace("|", "/")
            vals.append(text)
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def build_catalog() -> pd.DataFrame:
    rows = []
    for nb, fig, title, reading, table in FIGURE_CATALOG:
        fpath = FIGS / fig
        tpath = STATS / table
        rows.append({
            "notebook": nb,
            "figura": fig,
            "titulo": title,
            "interpreta": reading,
            "tabela_referencia": table,
            "figura_existe": fpath.exists(),
            "tabela_existe": tpath.exists(),
            "figura_path": str(fpath.relative_to(ROOT)),
            "tabela_path": str(tpath.relative_to(ROOT)),
        })
    cat = pd.DataFrame(rows)
    STATS.mkdir(parents=True, exist_ok=True)
    cat.to_csv(STATS / "phase3_figuras_catalogo.csv", index=False)
    return cat


def build_gallery(cat: pd.DataFrame) -> Path | None:
    rows = cat[cat["figura_existe"]].reset_index(drop=True)
    if rows.empty:
        return None
    ncols = 5
    nrows = ceil(len(rows) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.2 * nrows))
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, rows.iterrows()):
        img = mpimg.imread(FIGS / row["figura"])
        ax.imshow(img)
        ax.set_title(f"{row['notebook']} - {row['figura']}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Fase 3 - galeria padronizada de figuras", fontsize=18, y=0.995)
    out = FIGS / "3R1_galeria_figuras_fase3.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def best_by_variable() -> pd.DataFrame:
    skill = read_csv("phase3I_skill_por_variavel.csv")
    if skill.empty:
        return skill
    return (
        skill.sort_values("skill_vs_climatologia", ascending=False)
        .groupby(["variavel", "rotulo"], as_index=False)
        .head(1)
        .sort_values("skill_vs_climatologia", ascending=False)
        .reset_index(drop=True)
    )


def build_report(cat: pd.DataFrame, gallery: Path | None) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    nested = read_csv("phase3I_nested_loo_metricas.csv")
    proj = read_csv("phase3I_projecao_pico_2026.csv")
    pca_nested = read_csv("phase3K_previsao_pico_nested_loo_metricas.csv")
    best_var = best_by_variable()
    est = read_csv("phase3E_sensibilidade_resumo.csv")
    eventos = read_csv("phase3I_medias_classes_noaa.csv")
    audit = read_csv("phase3_temporal_integrity_audit.csv")

    nested_txt = md_table(nested)
    proj_txt = md_table(proj, [
        "pico_projetado_c", "ic95_baixo_c", "ic95_alto_c", "modelo",
        "variaveis", "horizonte_sem", "r_loo", "mae_loo_c",
        "skill_vs_climatologia", "leitura",
    ])
    pca_txt = md_table(pca_nested)
    best_txt = md_table(best_var, [
        "variavel", "rotulo", "lead_semanas", "r_loo",
        "mae_loo_c", "skill_vs_climatologia",
    ])
    est_txt = md_table(est, [
        "variavel", "lag_semanas", "r_full",
        "bootstrap_ic95_inf_envelope", "bootstrap_ic95_sup_envelope",
        "bootstrap_min_fracao_mesmo_sinal", "loo_eventos_n",
        "loo_eventos_r_min", "loo_eventos_r_max",
        "loo_evento_maior_influencia", "loo_max_delta_r", "papel_3E",
    ])
    eventos_txt = md_table(eventos)
    audit_txt = md_table(audit, [
        "artifact", "scope", "expected_freq", "start", "end", "rows",
        "freshness_days", "max_key_null_pct", "status", "notes",
    ], max_rows=20)
    if audit.empty or "status" not in audit.columns:
        audit_summary = "auditoria temporal ainda nao encontrada"
    else:
        counts = audit["status"].value_counts().sort_index()
        audit_summary = ", ".join(f"{status}={count}" for status, count in counts.items())

    nb_table = pd.DataFrame(NOTEBOOK_SUMMARY, columns=["notebook", "faz", "pergunta", "leitura"])
    cat_view = cat[["notebook", "figura", "titulo", "interpreta", "tabela_referencia", "figura_existe", "tabela_existe"]]
    completion_table = pd.DataFrame(PHASE3_COMPLETION_ROWS)

    gallery_md = ""
    if gallery is not None:
        rel = Path("../../data/processed/figures/fase3") / gallery.name
        gallery_md = f"\n![Galeria Fase 3]({rel.as_posix()})\n"

    return f"""# Relatorio final interpretativo - Fase 3 NINO26

Gerado em: {now}

## Veredito executivo

A Fase 3 esta **concluida como diagnostico fisico semanal do Nino 3.4** pela
diretriz canonica atual. O pipeline agora separa **El Nino e La Nina**, delimita
genese, crescimento, pico e decaimento por evento, calcula duracoes por tipo e
classe, materializa discriminantes por periodo e entrega PCA por fase.

O conjunto de variaveis que melhor antecipa o aquecimento maximo do El Nino, no
estado atual do pipeline, e o bloco de **recarga/subsuperficie**:

- `ohc_0_300`: melhor preditor individual no hindcast; representa calor armazenado nos 0-300 m.
- `ssh_m`: proxy dinamico de expansao/recarga da coluna d'agua.
- `tau_x_anom_nino34_pa`: acoplamento vento-superficie; anomalias de oeste favorecem downwelling Kelvin e aquecimento.
- `ohc_0_700`, `tilt_m` e `d20_m`: confirmam profundidade/inclinacao da termoclina e memoria subsuperficial.
- `wwv`: volume acima da isoterma de 20 C integrado em area no Pacifico equatorial; candidato do bloco de recarga, derivado de D20 e nao representante obrigatorio.

## Fechamento contra a diretriz atual da Fase 3

{md_table(completion_table, max_rows=20)}

## Integridade temporal dos dados

Resumo da auditoria: **{audit_summary}**. Alertas `warning` indicam defasagem ou
cobertura a acompanhar; `error` indicaria quebra de integridade regular.

{audit_txt}

## Metodologia preditiva adotada

O 3I/3K usa **nested leave-one-event-out**. O loop interno escolhe o candidato
apenas nos eventos de treino; o loop externo preve o evento retido. Isso reduz o
vies otimista do LOO simples quando ele tambem escolhe o melhor modelo.

Referencias metodologicas: Jin (1997), Meinen & McPhaden (2000), WMO SVSLRF,
Barnston et al. (2012), Ambroise & McLachlan (2002), Cawley & Talbot (2010).

### Resultado nested LOO do 3I

{nested_txt}

### Projecao condicional 2025/26

{proj_txt}

Leitura: a projecao estima amplitude condicional dado o estado recente. Ela ainda
nao e previsao operacional de timing; isso fica para a Fase 5 com walk-forward,
embargo temporal, barreira de primavera e baseline de persistencia amortecida.

### Resultado nested LOO do 3K/PCA

{pca_txt}

## Melhores preditores por variavel (triagem flat LOO)

{best_txt}

## Sensibilidade temporal sem breakpoint

{est_txt}

## Classes El Nino NOAA/ONI locais

{eventos_txt}

## O que cada notebook responde

{md_table(nb_table)}

## Catalogo de figuras e tabelas

{md_table(cat_view, max_rows=40)}

## Galeria padronizada
{gallery_md}

## Interpretacao para cientistas

O aquecimento maximo do El Nino nao e explicado apenas pela SSTA superficial.
A SSTA e a resposta observavel final; antes dela, o sistema precisa acumular
calor e alterar a estrutura vertical/oceanica. OHC0-300, SSH, D20 e tilt medem
essa recarga e a geometria da termoclina. O `tau_x_anom` representa o elo de
acoplamento com a atmosfera: anomalias de oeste reduzem/alteram os alisios,
favorecem ondas Kelvin de downwelling e aprofundam a termoclina no centro-leste
do Pacifico. O WWV e um indicador classico do oscilador de recarga, mas e
derivado de D20 e compartilha informacao com D20/OHC/SSH/tilt; por isso fica
como candidato, sem privilegio a priori. O 3E usa bootstrap em blocos e
leave-one-event-out apenas como sensibilidade, sem corte cronologico. A Fase 3 nao usa metricas acumuladas artificiais como preditoras:
o foco interpretativo fica em SSTA, recarga/subsuperficie e acoplamento do vento.

## Interpretacao para pessoas comuns

Pense no El Nino como uma panela grande. A temperatura da superficie e o que se
ve por cima, mas o pico depende do calor ja guardado embaixo e de como o vento
empurra esse calor pelo Pacifico. As melhores pistas sao: quanto calor ha nos
primeiros 300 m, se o nivel do mar/coluna d'agua indica recarga, se a termoclina
esta mais profunda/inclinada e se o vento esta ajudando o calor a ir para leste.
Quando essas pistas aparecem juntas, a chance de um pico maior aumenta.
"""


def write_index(cat: pd.DataFrame) -> None:
    lines = [
        "# Indice de figuras - Fase 3",
        "",
        "Convencao: `3A1` = Fase 3, notebook A, figura 1.",
        "",
        md_table(cat[["notebook", "figura", "titulo", "interpreta", "tabela_referencia"]], max_rows=40),
        "",
    ]
    (NBDIR / "INDICE_FIGURAS_FASE3.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    cat = build_catalog()
    gallery = build_gallery(cat)
    report = build_report(cat, gallery)
    (NBDIR / "RELATORIO_FINAL_FASE3.md").write_text(report, encoding="utf-8")
    write_index(cat)
    print(f"[texto] {(NBDIR / 'RELATORIO_FINAL_FASE3.md').relative_to(ROOT)}")
    print(f"[texto] {(NBDIR / 'INDICE_FIGURAS_FASE3.md').relative_to(ROOT)}")
    print(f"[tabela] {(STATS / 'phase3_figuras_catalogo.csv').relative_to(ROOT)}")
    if gallery is not None:
        print(f"[figura] {gallery.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
