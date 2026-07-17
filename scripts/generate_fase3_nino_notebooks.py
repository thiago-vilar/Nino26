#!/usr/bin/env python3
"""Gera do zero os notebooks da fase3_nino (Dinâmica e Evolução do El Niño).

Sete notebooks, um por etapa construtiva do prompt base (4.1–4.7). Cada
notebook abre com as seções obrigatórias TÍTULO, CONTEXTO, DESAFIO,
METODOLOGIA, RESULTADOS ESPERADOS e REFERÊNCIAS BIBLIOGRÁFICAS, exporta
figuras a 600 dpi (png+pdf) em data/processed/figures/fase3 e as tabelas
numéricas associadas em data/processed/numeric-tables/fase3.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "notebooks/fase3_nino"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text)


def code(text: str) -> nbf.NotebookNode:
    cell = nbf.v4.new_code_cell(text)
    cell["execution_count"] = None
    cell["outputs"] = []
    return cell


def header(
    *,
    titulo: str,
    contexto: str,
    desafio: str,
    metodologia: str,
    resultados: str,
    referencias: list[str],
    make_cell: bool = False,
) -> list[nbf.NotebookNode]:
    cells: list[nbf.NotebookNode] = []
    if make_cell:
        cells.append(md("**COMANDO WSL2 — EXECUTAR FASE COMPLETA**\n\n```bash\nmake fase3\n```"))
    refs = "\n".join(f"{i}. {r}" for i, r in enumerate(referencias, start=1))
    cells.append(
        md(
            f"**TÍTULO**\n\n{titulo}\n\n"
            f"**CONTEXTO**\n\n{contexto}\n\n"
            f"**DESAFIO**\n\n{desafio}\n\n"
            f"**METODOLOGIA**\n\n{metodologia}\n\n"
            f"**RESULTADOS ESPERADOS**\n\n{resultados}\n\n"
            f"**REFERÊNCIAS BIBLIOGRÁFICAS**\n\n{refs}"
        )
    )
    return cells


SETUP = (
    "from pathlib import Path\n"
    "import sys\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n\n"
    "RAIZ = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())\n"
    "if str(RAIZ / 'src') not in sys.path:\n"
    "    sys.path.insert(0, str(RAIZ / 'src'))\n"
    "from nino_brasil import fase3_nino as f3\n\n"
    "f3.ensure_dirs()\n"
    "master = f3.load_master_weekly()\n"
    "fisicas = [c for c in master.columns if c != 'ocean_source_code']\n"
    "print(f'Master semanal F2: {master.shape[0]} semanas x {len(fisicas)} variaveis fisicas'\n"
    "      f\" | simulado={master.attrs.get('dado_simulado', False)}\")\n"
    "print(f'Periodo: {master.index.min().date()} a {master.index.max().date()}')"
)


REF_HUANG = (
    "HUANG, B. et al. Improvements of the Daily Optimum Interpolation Sea Surface "
    "Temperature (DOISST) Version 2.1. Journal of Climate, v. 34, p. 2923-2939, 2021. "
    "DOI: https://doi.org/10.1175/JCLI-D-20-0166.1"
)
REF_HERSBACH = (
    "HERSBACH, H. et al. The ERA5 global reanalysis. Quarterly Journal of the Royal "
    "Meteorological Society, v. 146, p. 1999-2049, 2020. DOI: https://doi.org/10.1002/qj.3803"
)
REF_TRENBERTH = (
    "TRENBERTH, K. E. The Definition of El Niño. Bulletin of the American Meteorological "
    "Society, v. 78, p. 2771-2777, 1997. DOI: https://doi.org/10.1175/1520-0442(1997)010<2759:TDOENO>2.0.CO;2"
)
REF_JIN = (
    "JIN, F.-F. An Equatorial Ocean Recharge Paradigm for ENSO. Part I: Conceptual Model. "
    "Journal of the Atmospheric Sciences, v. 54, p. 811-829, 1997. "
    "DOI: https://doi.org/10.1175/1520-0469(1997)054<0811:AEORPF>2.0.CO;2"
)
REF_MEINEN = (
    "MEINEN, C. S.; McPHADEN, M. J. Observations of Warm Water Volume Changes in the "
    "Equatorial Pacific and Their Relationship to El Niño and La Niña. Journal of Climate, "
    "v. 13, p. 3551-3559, 2000. DOI: https://doi.org/10.1175/1520-0442(2000)013<3551:OOWWVC>2.0.CO;2"
)
REF_WILKS = (
    "WILKS, D. S. Statistical Methods in the Atmospheric Sciences. 3. ed. Academic Press, 2011. "
    "DOI: https://doi.org/10.1016/C2010-0-66249-4"
)
REF_BRETHERTON = (
    "BRETHERTON, C. S. et al. The Effective Number of Spatial Degrees of Freedom of a "
    "Time-Varying Field. Journal of Climate, v. 12, p. 1990-2009, 1999."
)
REF_BJERKNES = (
    "BJERKNES, J. Atmospheric Teleconnections from the Equatorial Pacific. Monthly Weather "
    "Review, v. 97, p. 163-172, 1969. DOI: https://doi.org/10.1175/1520-0493(1969)097<0163:ATFTEP>2.3.CO;2"
)
REF_CUI = (
    "CUI, Y. et al. Oceanic Kelvin waves and their role in ENSO evolution. Journal of "
    "Geophysical Research: Oceans, 2025. DOI: https://doi.org/10.1029/2025JC023275"
)
REF_WHEELER = (
    "WHEELER, M.; KILADIS, G. N. Convectively Coupled Equatorial Waves: Analysis of Clouds "
    "and Temperature in the Wavenumber-Frequency Domain. Journal of the Atmospheric Sciences, "
    "v. 56, p. 374-399, 1999. DOI: https://doi.org/10.1175/1520-0469(1999)056<0374:CCEWAO>2.0.CO;2"
)


def notebook_f3n1() -> nbf.NotebookNode:
    cells = header(
        make_cell=True,
        titulo="F3N1 — Organização e normalização da série histórica semanal (1981–2026)",
        contexto=(
            "A fase3_nino investiga a evolução do El Niño estritamente no Pacífico (região "
            "Niño 3.4), da gênese ao decaimento. A base de entrada é a matriz semanal W-SUN "
            "publicada pela Fase 2 a partir de NOAA OISST v2.1 (SSTA), ERA5 (atmosfera) e "
            "UFS+GLORYS (oceano subsuperficial). Comparações entre variáveis de unidades e "
            "escalas distintas exigem uma base comum de anomalias normalizadas."
        ),
        desafio=(
            "Hipótese: após remoção da sazonalidade e padronização Z-score, as anomalias das "
            "variáveis oceânicas e atmosféricas tornam-se diretamente comparáveis, revelando a "
            "coevolução do sistema acoplado. Objetivos: (i) estruturar as séries semanais "
            "1981–2026; (ii) remover o ciclo sazonal de todas as variáveis; (iii) padronizar "
            "via Z-score com período climatológico de referência 1991–2020."
        ),
        metodologia=(
            "Para cada variável x(t): anomalia semanal x'(t) = x(t) − c(w(t)), onde c é a "
            "climatologia por semana-do-ano (1..53) ajustada em 1991–2020 e suavizada "
            "circularmente (janela de 5 semanas); em seguida z(t) = (x'(t) − μ)/σ com μ e σ "
            "do período de referência. Colunas que já são anomalias na F2 (sufixo _anom e "
            "nino34_ssta) recebem apenas o Z-score. Não há preenchimento de lacunas: ausências "
            "permanecem explícitas."
        ),
        resultados=(
            "TabF3N1_zscores_semanais.csv (matriz completa normalizada), "
            "TabF3N1_resumo_estatistico.csv e FigF3N1_series_normalizadas (png+pdf, 600 dpi) "
            "com painéis oceânico e atmosférico das anomalias padronizadas."
        ),
        referencias=[REF_HUANG, REF_HERSBACH, REF_WILKS],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "zscores = f3.deseason_and_zscore(master[fisicas])\n"
            "f3.save_table(zscores, 'TabF3N1_zscores_semanais')\n\n"
            "resumo = pd.DataFrame({\n"
            "    'media': zscores.mean(), 'desvio': zscores.std(ddof=1),\n"
            "    'minimo': zscores.min(), 'maximo': zscores.max(),\n"
            "    'semanas_validas': zscores.notna().sum(),\n"
            "    'cobertura_pct': (100.0 * zscores.notna().mean()).round(2),\n"
            "}).round(3)\n"
            "f3.save_table(resumo, 'TabF3N1_resumo_estatistico')\n"
            "resumo.head(12)"
        )
    )
    cells.append(
        code(
            "oceanicas = [c for c in ['nino34_ssta','d20_m','ohc_0_300','wwv','ssh_m','t100m'] if c in zscores]\n"
            "atmosfericas = [c for c in ['tau_x_anom','u10_anom','u850_anom','mslp_anom','tcwv_anom','ssr_anom'] if c in zscores]\n"
            "fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)\n"
            "for coluna in oceanicas:\n"
            "    axes[0].plot(zscores.index, zscores[coluna], lw=0.7, label=coluna)\n"
            "for coluna in atmosfericas:\n"
            "    axes[1].plot(zscores.index, zscores[coluna], lw=0.7, label=coluna)\n"
            "axes[0].set_title('Anomalias normalizadas (Z-score) — bloco oceanico')\n"
            "axes[1].set_title('Anomalias normalizadas (Z-score) — bloco atmosferico')\n"
            "for ax in axes:\n"
            "    ax.axhline(0, color='k', lw=0.5)\n"
            "    ax.legend(loc='upper left', ncol=3, fontsize=7)\n"
            "    ax.set_ylabel('z')\n"
            "fig.tight_layout()\n"
            "f3.save_table(zscores[oceanicas + atmosfericas], 'FigF3N1_series_normalizadas_dados')\n"
            "f3.save_figure(fig, 'FigF3N1_series_normalizadas')\n"
            "plt.show()"
        )
    )
    return build_notebook("F3N1", cells)


def notebook_f3n2() -> nbf.NotebookNode:
    cells = header(
        titulo="F3N2 — Classificação dos períodos de El Niño na região Niño 3.4",
        contexto=(
            "A definição operacional de El Niño (ONI/NOAA) usa médias trimestrais mensais de "
            "SSTA em Niño 3.4 com limiar de +0,5 °C por cinco estações sobrepostas. Esta fase "
            "trabalha em resolução semanal; o critério é adaptado preservando as escalas "
            "temporais originais, conforme a discussão de definição de Trenberth (1997)."
        ),
        desafio=(
            "Hipótese: eventos de El Niño identificados na série semanal reproduzem o catálogo "
            "histórico (1982/83, 1997/98, 2015/16, 2023/24 entre os fortes) com datação mais "
            "fina de início, pico e término. Objetivos: (i) detectar eventos semanais; "
            "(ii) medir duração e intensidade; (iii) traçar os perfis temporais da SSTA média."
        ),
        metodologia=(
            "SSTA de Niño 3.4 suavizada por média móvel centrada de 13 semanas (~3 meses, "
            "análogo semanal da média trimestral do ONI). Evento: SSTA suavizada ≥ +0,5 °C por "
            "pelo menos 22 semanas consecutivas (~5 meses, equivalente às 5 estações "
            "sobrepostas). Intensidade no pico define a classe: fraco (0,5–1,0), moderado "
            "(1,0–1,5), forte (1,5–2,0), muito forte (≥ 2,0 °C). Perfis alinhados pela semana "
            "do pico (lag 0) em janela de ±52 semanas."
        ),
        resultados=(
            "TabF3N2_catalogo_eventos.csv (início, fim, duração, pico, classe), "
            "FigF3N2_serie_eventos (série com eventos sombreados) e FigF3N2_perfis_eventos "
            "(evolução alinhada pelo pico), com tabelas numéricas associadas."
        ),
        referencias=[REF_TRENBERTH, REF_HUANG],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "ssta = pd.to_numeric(master['nino34_ssta'], errors='coerce')\n"
            "suave = f3.smooth_ssta(ssta, 13)\n"
            "catalogo = f3.detect_el_nino_events(ssta, smooth_weeks=13, threshold_c=0.5, min_duration_weeks=22)\n"
            "f3.save_table(catalogo, 'TabF3N2_catalogo_eventos', index=False)\n"
            "catalogo"
        )
    )
    cells.append(
        code(
            "fig, ax = plt.subplots(figsize=(14, 4.5))\n"
            "ax.plot(ssta.index, ssta, color='0.6', lw=0.5, label='SSTA semanal')\n"
            "ax.plot(suave.index, suave, color='crimson', lw=1.2, label='media movel 13 semanas')\n"
            "ax.axhline(0.5, color='k', ls='--', lw=0.8, label='limiar +0,5 C')\n"
            "for _, ev in catalogo.iterrows():\n"
            "    ax.axvspan(pd.Timestamp(ev['inicio']), pd.Timestamp(ev['fim']), color='orange', alpha=0.25)\n"
            "ax.set_ylabel('SSTA Nino 3.4 (C)')\n"
            "ax.set_title('Eventos El Nino semanais (criterio ONI adaptado, Trenberth 1997)')\n"
            "ax.legend(loc='upper left', fontsize=8)\n"
            "fig.tight_layout()\n"
            "f3.save_table(pd.DataFrame({'ssta': ssta, 'ssta_suavizada': suave}), 'FigF3N2_serie_eventos_dados')\n"
            "f3.save_figure(fig, 'FigF3N2_serie_eventos')\n"
            "plt.show()"
        )
    )
    cells.append(
        code(
            "janela = 52\n"
            "perfis = {}\n"
            "for _, ev in catalogo.iterrows():\n"
            "    pico = pd.Timestamp(ev['semana_pico'])\n"
            "    trecho = suave.loc[pico - pd.Timedelta(weeks=janela):pico + pd.Timedelta(weeks=janela)]\n"
            "    rel = ((trecho.index - pico).days / 7).astype(int)\n"
            "    perfis[ev['evento']] = pd.Series(trecho.to_numpy(), index=rel)\n"
            "matriz_perfis = pd.DataFrame(perfis).reindex(range(-janela, janela + 1))\n"
            "matriz_perfis.index.name = 'semanas_relativas_ao_pico'\n"
            "f3.save_table(matriz_perfis, 'FigF3N2_perfis_eventos_dados')\n\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "for evento in matriz_perfis.columns:\n"
            "    ax.plot(matriz_perfis.index, matriz_perfis[evento], lw=0.8, alpha=0.6, label=evento)\n"
            "ax.plot(matriz_perfis.index, matriz_perfis.mean(axis=1), color='k', lw=2.2, label='composto medio')\n"
            "ax.axvline(0, color='k', lw=0.6, ls=':')\n"
            "ax.axhline(0.5, color='r', lw=0.6, ls='--')\n"
            "ax.set_xlabel('Semanas relativas ao pico')\n"
            "ax.set_ylabel('SSTA suavizada (C)')\n"
            "ax.set_title('Evolucao temporal dos eventos alinhados pelo pico')\n"
            "ax.legend(fontsize=7, ncol=2)\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N2_perfis_eventos')\n"
            "plt.show()"
        )
    )
    return build_notebook("F3N2", cells)


def notebook_f3n3() -> nbf.NotebookNode:
    cells = header(
        titulo="F3N3 — Classificação das fases do ciclo de vida: gênese, crescimento, faixa de pico e decaimento",
        contexto=(
            "No paradigma de recarga-descarga (Jin, 1997), o El Niño nasce de um estado "
            "recarregado de calor subsuperficial (gênese), amplifica-se pelo acoplamento com o "
            "vento (crescimento), atinge um patamar quase estacionário (faixa de pico) e decai "
            "quando a descarga de calor para fora do equador domina (decaimento)."
        ),
        desafio=(
            "Hipótese: as quatro fases têm assinaturas temporais distintas e duração "
            "sistematicamente assimétrica (crescimento mais longo que decaimento em eventos "
            "fortes). Objetivos: segmentar cada evento do catálogo F3N2 nas quatro fases, "
            "justificar fisicamente as janelas e quantificar durações por classe."
        ),
        metodologia=(
            "Faixa de pico: patamar contínuo com SSTA suavizada ≥ 90% do máximo do evento "
            "(documentação do projeto). Crescimento: do cruzamento de +0,5 °C até o início do "
            "patamar. Decaimento: do fim do patamar até a saída do limiar. Gênese: até 26 "
            "semanas antes do cruzamento, iniciando no último mínimo local da SSTA suavizada — "
            "janela coerente com o tempo de recarga equatorial de Jin (1997) e Meinen & "
            "McPhaden (2000)."
        ),
        resultados=(
            "TabF3N3_fases_semanais.csv (fase de cada semana), TabF3N3_duracao_fases.csv "
            "(duração por evento e fase), FigF3N3_linha_do_tempo (série colorida por fase) e "
            "FigF3N3_duracao_por_classe (duração das fases por classe de evento)."
        ),
        referencias=[REF_JIN, REF_MEINEN, REF_TRENBERTH],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "ssta = pd.to_numeric(master['nino34_ssta'], errors='coerce')\n"
            "suave = f3.smooth_ssta(ssta, 13)\n"
            "catalogo = f3.detect_el_nino_events(ssta)\n"
            "fases = f3.classify_life_cycle(ssta, catalogo, peak_fraction=0.90)\n"
            "tabela_fases = pd.DataFrame({'ssta_suavizada': suave, 'fase': fases})\n"
            "f3.save_table(tabela_fases, 'TabF3N3_fases_semanais')\n"
            "tabela_fases['fase'].value_counts()"
        )
    )
    cells.append(
        code(
            "linhas = []\n"
            "for _, ev in catalogo.iterrows():\n"
            "    bloco = fases.loc[pd.Timestamp(ev['inicio']) - pd.Timedelta(weeks=26):pd.Timestamp(ev['fim'])]\n"
            "    contagem = bloco.value_counts()\n"
            "    linhas.append({'evento': ev['evento'], 'classe': ev['classe'],\n"
            "                   **{f'{fase}_semanas': int(contagem.get(fase, 0))\n"
            "                      for fase in ('genese', 'crescimento', 'faixa_pico', 'decaimento')}})\n"
            "duracoes = pd.DataFrame(linhas)\n"
            "f3.save_table(duracoes, 'TabF3N3_duracao_fases', index=False)\n"
            "duracoes"
        )
    )
    cells.append(
        code(
            "cores = {'neutro': '0.85', 'genese': '#7fbf7f', 'crescimento': '#f4a742',\n"
            "         'faixa_pico': '#d62728', 'decaimento': '#1f77b4'}\n"
            "fig, ax = plt.subplots(figsize=(14, 4.5))\n"
            "ax.plot(suave.index, suave, color='k', lw=0.8)\n"
            "for fase, cor in cores.items():\n"
            "    if fase == 'neutro':\n"
            "        continue\n"
            "    selecao = fases == fase\n"
            "    ax.fill_between(suave.index, 0, suave.where(selecao), color=cor, alpha=0.7, label=fase)\n"
            "ax.axhline(0.5, color='k', ls='--', lw=0.6)\n"
            "ax.set_ylabel('SSTA suavizada (C)')\n"
            "ax.set_title('Ciclo de vida do El Nino: genese, crescimento, faixa de pico (>=90% do maximo) e decaimento')\n"
            "ax.legend(loc='upper left', fontsize=8, ncol=4)\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N3_linha_do_tempo')\n"
            "plt.show()\n\n"
            "colunas_fase = ['genese_semanas', 'crescimento_semanas', 'faixa_pico_semanas', 'decaimento_semanas']\n"
            "media_classe = duracoes.groupby('classe')[colunas_fase].mean().round(1)\n"
            "f3.save_table(media_classe, 'FigF3N3_duracao_por_classe_dados')\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "media_classe.plot(kind='bar', ax=ax)\n"
            "ax.set_ylabel('Duracao media (semanas)')\n"
            "ax.set_title('Duracao media das fases por classe de evento')\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N3_duracao_por_classe')\n"
            "plt.show()"
        )
    )
    return build_notebook("F3N3", cells)


def notebook_f3n4() -> nbf.NotebookNode:
    cells = header(
        titulo="F3N4 — Triagem de variáveis por correlação defasada e redução de dimensionalidade (PCA/EOF)",
        contexto=(
            "O sistema ENSO é altamente colinear: D20, OHC, WWV, SSH e temperaturas de "
            "subsuperfície compartilham a memória de recarga (Meinen & McPhaden, 2000), e os "
            "campos atmosféricos respondem coletivamente ao mesmo gradiente de SST. Um baseline "
            "estatístico linear rigoroso exige triagem por significância e compressão da "
            "redundância antes de qualquer modelagem."
        ),
        desafio=(
            "Hipótese: um subconjunto de variáveis lidera a SSTA de Niño 3.4 em lags de "
            "semanas a meses, e poucos componentes principais concentram ≥ 80% da variância do "
            "bloco preditor. Objetivos: (i) correlação cruzada defasada preditor(t−lag) × "
            "SSTA(t); (ii) descartar variáveis sem significância (p < 0,05) em nenhuma janela; "
            "(iii) PCA no bloco filtrado retendo ≥ 80% da variância; (iv) interpretar sinais "
            "dos coeficientes."
        ),
        metodologia=(
            "Anomalias normalizadas da F3N1 (sazonalidade removida de TODAS as variáveis antes "
            "da modelagem). Lags de 1 a 52 semanas. Pearson e Spearman com N efetivo de "
            "Bretherton et al. (1999) para autocorrelação serial; teste t bicaudal com "
            "graus de liberdade N_eff − 2. Triagem: mantém-se a variável se p < 0,05 em pelo "
            "menos um lag (em qualquer dos dois coeficientes). PCA por SVD (pacote eofs quando "
            "disponível) na matriz filtrada; retêm-se os primeiros componentes até acumular "
            "80% da variância (Wilks, 2011)."
        ),
        resultados=(
            "TabF3N4_correlacoes_defasadas.csv, TabF3N4_triagem.csv (mantidas/descartadas), "
            "TabF3N4_variancia_pca.csv, TabF3N4_loadings.csv; FigF3N4_mapa_lag_correlacao "
            "(heatmap r × lag), FigF3N4_scree e FigF3N4_loadings_pc1_pc2. Coeficientes "
            "positivos: forçantes diretamente proporcionais à SSTA; negativos: forçantes que "
            "freiam ou invertem o fenômeno."
        ),
        referencias=[REF_MEINEN, REF_WILKS, REF_BRETHERTON],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "tabela_z = f3.table_dir() / 'TabF3N1_zscores_semanais.csv'\n"
            "if tabela_z.exists():\n"
            "    zscores = pd.read_csv(tabela_z, index_col=0, parse_dates=True)\n"
            "else:\n"
            "    zscores = f3.deseason_and_zscore(master[fisicas])\n"
            "alvo = zscores['nino34_ssta']\n"
            "preditores = zscores.drop(columns=['nino34_ssta'])\n"
            "lags = [1, 2, 3, 4, 6, 8, 13, 17, 22, 26, 33, 39, 45, 52]\n"
            "pearson = f3.lagged_correlations(preditores, alvo, lags_weeks=lags, method='pearson')\n"
            "spearman = f3.lagged_correlations(preditores, alvo, lags_weeks=lags, method='spearman')\n"
            "correlacoes = pearson.merge(spearman, on=['variavel', 'lag_semanas'], suffixes=('_pearson', '_spearman'))\n"
            "f3.save_table(correlacoes, 'TabF3N4_correlacoes_defasadas', index=False)\n"
            "correlacoes.head()"
        )
    )
    cells.append(
        code(
            "significativa = correlacoes.assign(\n"
            "    sig=lambda d: (d['p_valor_pearson'] < 0.05) | (d['p_valor_spearman'] < 0.05)\n"
            ").groupby('variavel')['sig'].any()\n"
            "mantidas = sorted(significativa[significativa].index)\n"
            "descartadas = sorted(significativa[~significativa].index)\n"
            "triagem = pd.DataFrame({'variavel': list(significativa.index),\n"
            "                        'mantida_p_menor_005': significativa.values})\n"
            "f3.save_table(triagem.sort_values('variavel'), 'TabF3N4_triagem', index=False)\n"
            "print(f'mantidas: {len(mantidas)} | descartadas: {len(descartadas)}')\n"
            "print('descartadas:', descartadas if descartadas else 'nenhuma')"
        )
    )
    cells.append(
        code(
            "mapa = correlacoes.pivot(index='variavel', columns='lag_semanas', values='r_pearson').loc[mantidas]\n"
            "f3.save_table(mapa, 'FigF3N4_mapa_lag_correlacao_dados')\n"
            "fig, ax = plt.subplots(figsize=(10, max(6, 0.28 * len(mapa))))\n"
            "im = ax.imshow(mapa.to_numpy(), aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1)\n"
            "ax.set_xticks(range(len(mapa.columns)), mapa.columns)\n"
            "ax.set_yticks(range(len(mapa.index)), mapa.index, fontsize=7)\n"
            "ax.set_xlabel('Lag (semanas): preditor(t-lag) x SSTA(t)')\n"
            "ax.set_title('Correlacao defasada com SSTA Nino 3.4 (variaveis significativas)')\n"
            "fig.colorbar(im, ax=ax, label='r de Pearson')\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N4_mapa_lag_correlacao')\n"
            "plt.show()"
        )
    )
    cells.append(
        code(
            "scores, loadings, variancia, n_pcs = f3.pca_from_zscores(preditores[mantidas], variance_target=0.80)\n"
            "f3.save_table(variancia.to_frame(), 'TabF3N4_variancia_pca')\n"
            "f3.save_table(loadings, 'TabF3N4_loadings')\n"
            "f3.save_table(scores, 'TabF3N4_scores_pca')\n"
            "print(f'{n_pcs} componentes retem {100 * variancia.sum():.1f}% da variancia')\n\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "acumulada = variancia.cumsum()\n"
            "ax.bar(range(1, n_pcs + 1), 100 * variancia.values, label='individual')\n"
            "ax.plot(range(1, n_pcs + 1), 100 * acumulada.values, 'ko-', label='acumulada')\n"
            "ax.axhline(80, color='r', ls='--', lw=0.8, label='alvo 80%')\n"
            "ax.set_xlabel('Componente principal')\n"
            "ax.set_ylabel('Variancia explicada (%)')\n"
            "ax.set_title('Scree plot do bloco preditor filtrado')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N4_scree')\n"
            "plt.show()\n\n"
            "fig, ax = plt.subplots(figsize=(8, 8))\n"
            "ax.axhline(0, color='0.7', lw=0.6); ax.axvline(0, color='0.7', lw=0.6)\n"
            "ax.scatter(loadings['PC1'], loadings['PC2'], s=12)\n"
            "for nome, linha in loadings.iterrows():\n"
            "    ax.annotate(nome, (linha['PC1'], linha['PC2']), fontsize=6)\n"
            "ax.set_xlabel('Loading PC1'); ax.set_ylabel('Loading PC2')\n"
            "ax.set_title('Loadings PC1 x PC2 — sinal positivo refor\\u00e7a, negativo freia a SSTA')\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N4_loadings_pc1_pc2')\n"
            "plt.show()\n\n"
            "orientacao = loadings['PC1'].sort_values()\n"
            "print('Interpretacao PC1 — extremos negativos (freiam o fenomeno):')\n"
            "print(orientacao.head(5).round(3).to_string())\n"
            "print('Interpretacao PC1 — extremos positivos (reforcam o fenomeno):')\n"
            "print(orientacao.tail(5).round(3).to_string())"
        )
    )
    return build_notebook("F3N4", cells)


def notebook_f3n5() -> nbf.NotebookNode:
    cells = header(
        titulo="F3N5 — Ventos zonais e o feedback de Bjerknes",
        contexto=(
            "Bjerknes (1969) descreveu o laço de retroalimentação positiva do ENSO: o "
            "enfraquecimento dos alísios reduz o gradiente zonal de SST, que enfraquece a "
            "circulação de Walker e os próprios alísios. Quantificar esse acoplamento é "
            "central para explicar crescimento e pico dos eventos."
        ),
        desafio=(
            "Hipótese: anomalias de vento zonal (tau_x, u850) covariam com o gradiente zonal "
            "de SSTA com defasagem de poucas semanas, e a estrutura espacial da correlação "
            "vento × SSTA(longitude) apresenta o dipolo característico do acoplamento. "
            "Objetivos: correlações e regressões defasadas série × série e série × campo."
        ),
        metodologia=(
            "Séries semanais normalizadas de tau_x_anom e u850_anom (caixa Niño 3.4/Niño 4, "
            "F2). Gradiente zonal de SSTA a partir do campo OISST equatorial (média 2°S–2°N): "
            "G(t) = SSTA_oeste(170°W–150°W) − SSTA_leste(90°W–80°W). Correlações defasadas "
            "(±26 semanas) com N efetivo de Bretherton; regressão espacial "
            "SSTA(lon, t) = a(lon)·tau_x(t−lag) + b com significância por t-teste. O domínio "
            "OISST local cobre 170°W–30°W; longitudes a oeste, quando exigidas, são declaradas "
            "como indisponíveis (sem preenchimento)."
        ),
        resultados=(
            "TabF3N5_correlacao_defasada_vento_gradiente.csv, TabF3N5_regressao_espacial.csv; "
            "FigF3N5_lagcorr_vento_gradiente, FigF3N5_regressao_espacial (a(lon) com faixa de "
            "significância) e FigF3N5_dispersao_acoplamento (tau_x × gradiente, com ajuste)."
        ),
        referencias=[REF_BJERKNES, REF_HERSBACH, REF_BRETHERTON],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "sst_lon = f3.load_oisst_equatorial_weekly()\n"
            "ssta_lon = f3.lon_anomaly_matrix(sst_lon)\n"
            "lons = np.array([float(c) for c in ssta_lon.columns])\n"
            "oeste = ssta_lon.loc[:, (lons >= -170) & (lons <= -150)].mean(axis=1)\n"
            "leste = ssta_lon.loc[:, (lons >= -90) & (lons <= -80)].mean(axis=1)\n"
            "gradiente = (oeste - leste).rename('gradiente_zonal_ssta_c')\n"
            "tau = f3.zscore(pd.to_numeric(master['tau_x_anom'], errors='coerce'))\n"
            "u850 = f3.zscore(pd.to_numeric(master['u850_anom'], errors='coerce'))\n"
            "print(f'campo SSTA: {ssta_lon.shape}; simulado={sst_lon.attrs.get(\"dado_simulado\", False)}')"
        )
    )
    cells.append(
        code(
            "linhas = []\n"
            "for nome, serie in (('tau_x_anom', tau), ('u850_anom', u850)):\n"
            "    for lag in range(-26, 27):\n"
            "        par = pd.concat([serie.shift(lag), gradiente], axis=1).dropna()\n"
            "        if len(par) < 52:\n"
            "            continue\n"
            "        r = float(par.iloc[:, 0].corr(par.iloc[:, 1]))\n"
            "        n_eff = f3.effective_sample_size(par.iloc[:, 0], par.iloc[:, 1])\n"
            "        linhas.append({'vento': nome, 'lag_semanas': lag, 'r': r,\n"
            "                       'n_eff': round(n_eff, 1), 'p_valor': f3.correlation_p_value(r, n_eff)})\n"
            "lagcorr = pd.DataFrame(linhas)\n"
            "f3.save_table(lagcorr, 'TabF3N5_correlacao_defasada_vento_gradiente', index=False)\n\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "for nome, grupo in lagcorr.groupby('vento'):\n"
            "    ax.plot(grupo['lag_semanas'], grupo['r'], marker='.', label=nome)\n"
            "    sig = grupo[grupo['p_valor'] < 0.05]\n"
            "    ax.scatter(sig['lag_semanas'], sig['r'], s=18, zorder=5)\n"
            "ax.axvline(0, color='k', lw=0.6); ax.axhline(0, color='k', lw=0.6)\n"
            "ax.set_xlabel('Lag (semanas; negativo = vento lidera o gradiente)')\n"
            "ax.set_ylabel('r')\n"
            "ax.set_title('Acoplamento vento zonal x gradiente zonal de SSTA (pontos: p<0,05)')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N5_lagcorr_vento_gradiente')\n"
            "plt.show()"
        )
    )
    cells.append(
        code(
            "melhor = lagcorr.loc[lagcorr.groupby('vento')['r'].apply(lambda s: s.abs().idxmax())]\n"
            "lag_otimo = int(melhor.loc[melhor['vento'] == 'tau_x_anom', 'lag_semanas'].iloc[0])\n"
            "linhas = []\n"
            "for coluna in ssta_lon.columns:\n"
            "    par = pd.concat([tau.shift(lag_otimo), ssta_lon[coluna]], axis=1).dropna()\n"
            "    if len(par) < 52:\n"
            "        continue\n"
            "    x = par.iloc[:, 0].to_numpy(); y = par.iloc[:, 1].to_numpy()\n"
            "    a, b = np.polyfit(x, y, 1)\n"
            "    r = float(np.corrcoef(x, y)[0, 1])\n"
            "    n_eff = f3.effective_sample_size(par.iloc[:, 0], par.iloc[:, 1])\n"
            "    linhas.append({'lon': float(coluna), 'coef_regressao_c_por_sigma_tau': a,\n"
            "                   'r': r, 'p_valor': f3.correlation_p_value(r, n_eff)})\n"
            "regressao = pd.DataFrame(linhas)\n"
            "f3.save_table(regressao, 'TabF3N5_regressao_espacial', index=False)\n\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(regressao['lon'], regressao['coef_regressao_c_por_sigma_tau'], lw=1.2)\n"
            "sig = regressao[regressao['p_valor'] < 0.05]\n"
            "ax.scatter(sig['lon'], sig['coef_regressao_c_por_sigma_tau'], s=10, color='crimson', label='p<0,05')\n"
            "ax.axhline(0, color='k', lw=0.6)\n"
            "ax.set_xlabel('Longitude (graus; dominio OISST local 170W-30W)')\n"
            "ax.set_ylabel('dSSTA/dtau_x (C por sigma)')\n"
            f"ax.set_title('Regressao espacial SSTA(lon) sobre tau_x (feedback de Bjerknes)')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N5_regressao_espacial')\n"
            "plt.show()\n\n"
            "par = pd.concat([tau.shift(lag_otimo).rename('tau_x_z'), gradiente], axis=1).dropna()\n"
            "a, b = np.polyfit(par['tau_x_z'], par['gradiente_zonal_ssta_c'], 1)\n"
            "fig, ax = plt.subplots(figsize=(6.5, 6))\n"
            "ax.scatter(par['tau_x_z'], par['gradiente_zonal_ssta_c'], s=6, alpha=0.4)\n"
            "xx = np.linspace(par['tau_x_z'].min(), par['tau_x_z'].max(), 50)\n"
            "ax.plot(xx, a * xx + b, color='crimson', label=f'ajuste: {a:.3f} C/sigma (lag {lag_otimo} sem)')\n"
            "ax.set_xlabel('tau_x_anom (z)'); ax.set_ylabel('gradiente zonal de SSTA (C)')\n"
            "ax.set_title('Dispersao do acoplamento oceano-atmosfera')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "f3.save_table(par, 'FigF3N5_dispersao_acoplamento_dados')\n"
            "f3.save_figure(fig, 'FigF3N5_dispersao_acoplamento')\n"
            "plt.show()"
        )
    )
    return build_notebook("F3N5", cells)


def notebook_f3n6() -> nbf.NotebookNode:
    cells = header(
        titulo="F3N6 — Ondas de Kelvin oceânicas: extração, rastreamento e velocidade de propagação",
        contexto=(
            "Ondas de Kelvin equatoriais descendentes transportam o sinal de recarga do "
            "Pacífico oeste para leste, precedendo o aquecimento de superfície em Niño 3.4 "
            "(Cui et al., 2025). Sua assinatura aparece como anomalias positivas de altura da "
            "superfície do mar (SSHA) e aprofundamento da termoclina propagando para leste a "
            "~2–3 m/s."
        ),
        desafio=(
            "Hipótese: pulsos de SSHA filtrada propagam-se para leste com velocidade "
            "compatível com o primeiro modo baroclínico (~2,7 m/s) e antecedem os eventos do "
            "catálogo F3N2. Objetivos: (i) extrair o sinal de Kelvin da SSHA equatorial "
            "UFS+GLORYS (120°E–280°E); (ii) rastrear pulsos ao longo da bacia; (iii) estimar a "
            "velocidade de fase por correlação defasada entre longitudes."
        ),
        metodologia=(
            "SSH semanal médio 2°S–2°N por longitude; anomalia por climatologia semanal por "
            "longitude; filtro espacial: remoção da média zonal instantânea e suavização em "
            "longitude (~10°) para reter escalas de onda longas. Pulsos: excedências de +1,5σ "
            "na longitude de referência 160°E. Velocidade: lag de correlação máxima entre a "
            "referência e cada longitude a leste; regressão distância × lag fornece a "
            "velocidade média (m/s), com a série d20 do master como verificação da resposta "
            "da termoclina."
        ),
        resultados=(
            "TabF3N6_ssha_filtrada.csv (matriz lon × tempo), TabF3N6_pulsos_kelvin.csv, "
            "TabF3N6_velocidade_fase.csv; FigF3N6_hovmoller_ssha_filtrada (janela recente), "
            "FigF3N6_lag_distancia (ajuste da velocidade) e FigF3N6_pulsos_vs_d20."
        ),
        referencias=[REF_CUI, REF_MEINEN],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "ssh = f3.load_ssh_equatorial_weekly()\n"
            "ssha = f3.lon_anomaly_matrix(ssh)\n"
            "kelvin = f3.kelvin_bandpass(ssha, smooth_deg=10.0)\n"
            "f3.save_table(kelvin, 'TabF3N6_ssha_filtrada')\n"
            "print(f'SSHA filtrada: {kelvin.shape}; simulado={ssh.attrs.get(\"dado_simulado\", False)}')"
        )
    )
    cells.append(
        code(
            "referencia_lon = 160.0\n"
            "lons = np.array([float(c) for c in kelvin.columns])\n"
            "col_ref = kelvin.columns[int(np.argmin(np.abs(lons - referencia_lon)))]\n"
            "serie_ref = kelvin[col_ref]\n"
            "sigma = serie_ref.std(ddof=1)\n"
            "pulsos = serie_ref[serie_ref > 1.5 * sigma]\n"
            "grupos = (pulsos.index.to_series().diff() > pd.Timedelta(weeks=4)).cumsum()\n"
            "catalogo_pulsos = pulsos.groupby(grupos).agg(['idxmax', 'max', 'count'])\n"
            "catalogo_pulsos.columns = ['semana_pico_pulso', 'ssha_pico_m', 'duracao_semanas']\n"
            "f3.save_table(catalogo_pulsos.reset_index(drop=True), 'TabF3N6_pulsos_kelvin', index=False)\n"
            "print(f'{len(catalogo_pulsos)} pulsos de Kelvin > 1,5 sigma em {referencia_lon}E')"
        )
    )
    cells.append(
        code(
            "lags = f3.phase_speed_from_lags(kelvin, reference_lon=referencia_lon, max_lag_weeks=10)\n"
            "ajuste = f3.fit_phase_speed(lags)\n"
            "f3.save_table(lags.assign(**ajuste), 'TabF3N6_velocidade_fase', index=False)\n"
            "print(f\"velocidade de fase media: {ajuste['velocidade_m_s']:.2f} m/s (r2={ajuste['r2']:.2f});\"\n"
            "      ' referencia teorica c1 ~ 2,7 m/s')\n\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "validos = lags[lags['r_max'] >= 0.3]\n"
            "ax.scatter(validos['lag_otimo_semanas'], validos['distancia_km'], s=14)\n"
            "if np.isfinite(ajuste['velocidade_m_s']):\n"
            "    xx = np.linspace(0, validos['lag_otimo_semanas'].max() + 1, 20)\n"
            "    ax.plot(xx, ajuste['velocidade_m_s'] * xx * 7 * 86400 / 1000, color='crimson',\n"
            "            label=f\"{ajuste['velocidade_m_s']:.2f} m/s\")\n"
            "ax.set_xlabel('Lag otimo (semanas) vs 160E')\n"
            "ax.set_ylabel('Distancia para leste (km)')\n"
            "ax.set_title('Propagacao para leste do sinal de Kelvin (SSHA filtrada)')\n"
            "ax.legend()\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N6_lag_distancia')\n"
            "plt.show()"
        )
    )
    cells.append(
        code(
            "janela = kelvin.loc[kelvin.index.max() - pd.Timedelta(weeks=200):]\n"
            "fig, ax = plt.subplots(figsize=(9, 10))\n"
            "malha = ax.pcolormesh(np.array([float(c) for c in janela.columns]), janela.index,\n"
            "                      janela.to_numpy(), cmap='RdBu_r',\n"
            "                      vmin=-np.nanmax(np.abs(janela.to_numpy())),\n"
            "                      vmax=np.nanmax(np.abs(janela.to_numpy())))\n"
            "ax.set_xlabel('Longitude (E)')\n"
            "ax.set_title('Hovmoller da SSHA filtrada (assinatura de Kelvin) — ultimas ~200 semanas')\n"
            "fig.colorbar(malha, ax=ax, label='SSHA filtrada (m)')\n"
            "fig.tight_layout()\n"
            "f3.save_figure(fig, 'FigF3N6_hovmoller_ssha_filtrada')\n"
            "plt.show()\n\n"
            "if 'd20_m' in master:\n"
            "    d20 = f3.zscore(f3.weekly_anomaly(pd.to_numeric(master['d20_m'], errors='coerce')))\n"
            "    fig, ax = plt.subplots(figsize=(12, 4.5))\n"
            "    ax.plot(serie_ref.index, serie_ref / sigma, lw=0.8, label=f'SSHA filtrada {referencia_lon}E (sigma)')\n"
            "    ax.plot(d20.index, d20, lw=0.8, label='anomalia D20 Nino 3.4 (z)')\n"
            "    for semana in catalogo_pulsos['semana_pico_pulso']:\n"
            "        ax.axvline(pd.Timestamp(semana), color='orange', lw=0.5, alpha=0.6)\n"
            "    ax.legend(); ax.set_title('Pulsos de Kelvin e resposta da termoclina (D20)')\n"
            "    fig.tight_layout()\n"
            "    f3.save_table(pd.DataFrame({'ssha_ref_sigma': serie_ref / sigma, 'd20_z': d20}),\n"
            "                  'FigF3N6_pulsos_vs_d20_dados')\n"
            "    f3.save_figure(fig, 'FigF3N6_pulsos_vs_d20')\n"
            "    plt.show()"
        )
    )
    return build_notebook("F3N6", cells)


def notebook_f3n7() -> nbf.NotebookNode:
    cells = header(
        titulo="F3N7 — Diagramas de Hovmöller do Pacífico equatorial: SSTA, SSHA e vento",
        contexto=(
            "O diagrama de Hovmöller (longitude × tempo) é a ferramenta clássica para "
            "visualizar propagação zonal de sinais equatoriais e distinguir ondas de Kelvin "
            "(para leste, rápidas) de Rossby (para oeste, lentas), como no arcabouço de "
            "Wheeler & Kiladis (1999)."
        ),
        desafio=(
            "Hipótese: nos eventos fortes, o aquecimento de superfície em Niño 3.4 é "
            "precedido por anomalias de SSH propagando de oeste para leste, com inclinação "
            "consistente com a velocidade estimada em F3N6. Objetivos: construir Hovmöllers de "
            "SSTA com contornos de SSHA sobrepostos, janelas dos eventos fortes, e anotar a "
            "velocidade de fase."
        ),
        metodologia=(
            "SSTA: OISST equatorial (média 2°S–2°N) por longitude, anomalia semanal; domínio "
            "local 170°W–30°W (leste da linha de data). SSHA filtrada de F3N6 no domínio "
            "120°E–280°E; sobreposição no trecho comum 170°W–80°W (190°E–280°E). Vento: "
            "tau_x_anom (média da caixa) como painel lateral sincronizado — o ERA5 local "
            "cobre apenas as caixas do projeto, e isso é declarado no gráfico. Janelas: "
            "eventos de classe forte/muito forte do catálogo F3N2 (pico ± 40 semanas)."
        ),
        resultados=(
            "Por evento forte: FigF3N7_hovmoller_<evento> (SSTA sombreada + contornos de SSHA "
            "+ painel de tau_x, reta da velocidade de fase) e tabelas "
            "TabF3N7_hovmoller_ssta_<evento>.csv / TabF3N7_hovmoller_ssha_<evento>.csv."
        ),
        referencias=[REF_WHEELER, REF_CUI, REF_HUANG],
    )
    cells.append(code(SETUP))
    cells.append(
        code(
            "sst_lon = f3.load_oisst_equatorial_weekly()\n"
            "ssta_lon = f3.lon_anomaly_matrix(sst_lon)\n"
            "ssta_lon.columns = [float(c) % 360 for c in ssta_lon.columns]  # 190E..330E\n"
            "ssh = f3.load_ssh_equatorial_weekly()\n"
            "kelvin = f3.kelvin_bandpass(f3.lon_anomaly_matrix(ssh))\n"
            "tau = pd.to_numeric(master['tau_x_anom'], errors='coerce')\n"
            "ssta_nino34 = pd.to_numeric(master['nino34_ssta'], errors='coerce')\n"
            "catalogo = f3.detect_el_nino_events(ssta_nino34)\n"
            "fortes = catalogo[catalogo['classe'].isin(['forte', 'muito_forte'])]\n"
            "if fortes.empty:\n"
            "    fortes = catalogo.tail(2)\n"
            "velocidade = f3.fit_phase_speed(f3.phase_speed_from_lags(kelvin, reference_lon=160.0))\n"
            "print(fortes[['evento', 'classe', 'semana_pico']].to_string(index=False))"
        )
    )
    cells.append(
        code(
            "for _, ev in fortes.iterrows():\n"
            "    pico = pd.Timestamp(ev['semana_pico'])\n"
            "    inicio, fim = pico - pd.Timedelta(weeks=40), pico + pd.Timedelta(weeks=40)\n"
            "    bloco_ssta = ssta_lon.loc[inicio:fim]\n"
            "    bloco_ssh = kelvin.loc[inicio:fim]\n"
            "    bloco_tau = tau.loc[inicio:fim]\n"
            "    if bloco_ssta.empty:\n"
            "        print(f\"{ev['evento']}: fora do dominio de dados; ignorado\")\n"
            "        continue\n"
            "    f3.save_table(bloco_ssta, f\"TabF3N7_hovmoller_ssta_{ev['evento']}\")\n"
            "    f3.save_table(bloco_ssh, f\"TabF3N7_hovmoller_ssha_{ev['evento']}\")\n\n"
            "    fig, (ax, ax_tau) = plt.subplots(1, 2, figsize=(11, 9), sharey=True,\n"
            "                                     gridspec_kw={'width_ratios': [4, 1]})\n"
            "    lons_sst = np.array([float(c) for c in bloco_ssta.columns])\n"
            "    limite = np.nanmax(np.abs(bloco_ssta.to_numpy()))\n"
            "    malha = ax.pcolormesh(lons_sst, bloco_ssta.index, bloco_ssta.to_numpy(),\n"
            "                          cmap='RdBu_r', vmin=-limite, vmax=limite)\n"
            "    lons_ssh = np.array([float(c) for c in bloco_ssh.columns])\n"
            "    comum = (lons_ssh >= lons_sst.min()) & (lons_ssh <= lons_sst.max())\n"
            "    if comum.sum() > 4 and not bloco_ssh.empty:\n"
            "        ax.contour(lons_ssh[comum], bloco_ssh.index,\n"
            "                   bloco_ssh.loc[:, bloco_ssh.columns[comum]].to_numpy(),\n"
            "                   levels=6, colors='k', linewidths=0.5)\n"
            "    if np.isfinite(velocidade['velocidade_m_s']) and velocidade['velocidade_m_s'] > 0:\n"
            "        semanas = np.arange(0, 30)\n"
            "        graus = velocidade['velocidade_m_s'] * semanas * 7 * 86400 / f3.EARTH_DEG_M\n"
            "        origem = lons_sst.min()\n"
            "        dentro = origem + graus <= lons_sst.max()\n"
            "        ax.plot(origem + graus[dentro],\n"
            "                inicio + pd.to_timedelta(semanas[dentro] * 7, unit='D'),\n"
            "                color='lime', lw=1.5, label=f\"{velocidade['velocidade_m_s']:.1f} m/s (F3N6)\")\n"
            "        ax.legend(loc='upper right', fontsize=7)\n"
            "    ax.set_xlabel('Longitude (graus E)')\n"
            "    ax.set_title(f\"Hovmoller {ev['evento']} ({ev['classe']}): SSTA sombreada + contornos SSHA\")\n"
            "    fig.colorbar(malha, ax=ax, label='SSTA (C)')\n"
            "    ax_tau.plot(bloco_tau.to_numpy(), bloco_tau.index, lw=0.8)\n"
            "    ax_tau.axvline(0, color='k', lw=0.5)\n"
            "    ax_tau.set_xlabel('tau_x_anom (Pa)\\n(media da caixa; ERA5 local)')\n"
            "    fig.tight_layout()\n"
            "    f3.save_figure(fig, f\"FigF3N7_hovmoller_{ev['evento']}\")\n"
            "    plt.show()"
        )
    )
    return build_notebook("F3N7", cells)


def build_notebook(code_name: str, cells: list[nbf.NotebookNode]) -> nbf.NotebookNode:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3.12 (.venv NINO26)",
            "language": "python",
            "name": "nino-brasil",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "nino26": {"canonical": True, "notebook_code": code_name, "phase": "fase3_nino"},
    }
    notebook["cells"] = cells
    return notebook


BUILDERS = {
    "F3N1_series_semanais_normalizacao.ipynb": notebook_f3n1,
    "F3N2_eventos_el_nino.ipynb": notebook_f3n2,
    "F3N3_ciclo_de_vida.ipynb": notebook_f3n3,
    "F3N4_triagem_pca.ipynb": notebook_f3n4,
    "F3N5_bjerknes_feedback.ipynb": notebook_f3n5,
    "F3N6_ondas_kelvin.ipynb": notebook_f3n6,
    "F3N7_hovmoller.ipynb": notebook_f3n7,
}


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, builder in BUILDERS.items():
        path = OUTPUT_DIR / filename
        nbf.write(builder(), path)
        print(f"[gravado] {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
