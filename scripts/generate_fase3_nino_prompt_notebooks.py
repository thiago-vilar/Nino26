"""Gera os notebooks científicos canônicos da FASE3-NINO."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = ROOT / "notebooks" / "fase3_nino" / "cientifica"

REFERENCES = """**REFERÊNCIAS BIBLIOGRÁFICAS**

1. NOAA Climate Prediction Center. *Oceanic Niño Index (ONI)*. Critério histórico: anomalia de SST em Niño-3.4 em média móvel de três meses, persistente por pelo menos cinco estações sobrepostas. https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
2. Huang, B. et al. (2021). *Improvements of the Daily Optimum Interpolation Sea Surface Temperature (DOISST) Version 2.1*. Journal of Climate, 34, 2923–2939. https://doi.org/10.1175/JCLI-D-20-0166.1
3. Jin, F.-F. (1997). *An Equatorial Ocean Recharge Paradigm for ENSO*. Journal of the Atmospheric Sciences, 54, 811–829. https://doi.org/10.1175/1520-0469(1997)054%3C0811:AEORPF%3E2.0.CO;2
4. Meinen, C. S.; McPhaden, M. J. (2000). *Observations of Warm Water Volume Changes in the Equatorial Pacific and Their Relationship to El Niño and La Niña*. Journal of Climate, 13, 3551–3559. https://doi.org/10.1175/1520-0442(2000)013%3C3551:OOWWVC%3E2.0.CO;2
5. Kessler, W. S.; McPhaden, M. J.; Weickmann, K. M. (1995). *Forcing of intraseasonal Kelvin waves in the equatorial Pacific*. Journal of Geophysical Research: Oceans, 100, 10613–10631. https://doi.org/10.1029/95JC00382
6. Timmermann, A. et al. (2018). *El Niño–Southern Oscillation complexity*. Nature, 559, 535–545. https://doi.org/10.1038/s41586-018-0252-6
"""

NOTEBOOKS = [
    ("F3NINO_01", "F3NINO_01_series_historicas_31_variaveis.ipynb", "Série histórica semanal e preparação das 31 variáveis", "Organiza a matriz semanal de 1981–presente. As variáveis oceânicas originalmente absolutas são convertidas em anomalias sazonais contra 1991–2020; as anomalias independentes de OISST e ERA5 são preservadas. Cada gráfico compara a SSTA Niño-3.4 com uma variável em sua própria linha, eliminando painéis comprimidos.", "A remoção do ciclo sazonal e a padronização por uma referência comum tornam comparáveis a amplitude e o momento das respostas sem apagar as unidades originais nas tabelas.", "Inspecionar cobertura, unidade, transformação e série histórica é necessário antes de qualquer composto ou classificação; lacunas são registradas, não interpoladas.", "Auditoria de 31 variáveis, anomalia sazonal para variáveis absolutas, z-score de referência, e gráficos temporais com SSTA e uma variável por linha.", "Tabela completa de contrato/cobertura e série z; figuras históricas legíveis. O início da cobertura OISST (setembro de 1981) é declarado."),
    ("F3NINO_02", "F3NINO_02_eventos_oni_p90_e_fases.ipynb", "Eventos El Niño pelo ONI, intensidade P90 e quatro fases", "Define os eventos pelo critério ONI compatível na série OISST local: média móvel centrada de três meses em Niño-3.4 >= +0,5 °C por no mínimo cinco estações móveis consecutivas. P90 é mantido, mas exclusivamente como estratificação de semanas muito quentes.", "O ONI produzirá episódios contínuos coerentes com a literatura; P90 identificará o núcleo de maior intensidade sem fragmentar artificialmente um evento.", "P90 descreve extremos da distribuição, mas não tem persistência térmica nem a interpretação operacional do ONI. Por isso ele não é usado como critério primário de ocorrência.", "Detecção ONI, catálogo de eventos, faixa de pico como 90% do ONI máximo do evento, gênese de 26 semanas antes do onset e rótulos retrospectivos. P90 é sobreposto à série e exportado em janelas próprias.", "Catálogo ONI, janelas P90 e rótulos semanais das quatro fases; duas figuras separam visualmente ocorrência e intensidade."),
    ("F3NINO_03", "F3NINO_03_compostos_31_variaveis.ipynb", "Compostos por evento ONI das 31 variáveis", "Resume a evolução das 31 variáveis nos eventos ONI, com o evento como unidade independente. Além do ciclo completo, compara as semanas P90 internas a cada evento, sem usar P90 para definir sua ocorrência.", "A média evento–fase revelará assinaturas recorrentes de recarga, acoplamento de vento, aquecimento e descarga; o núcleo P90 mostrará como a intensidade extrema se diferencia do evento completo.", "O composto permite distinguir comportamento típico de uma trajetória individual e mantém a exigência de analisar a média de El Niños de alta intensidade sem fragmentar a série pelo percentil.", "Média de cada variável para cada evento e fase; composto evento–fase; composto paralelo evento ONI completo versus núcleo P90, sempre com peso igual por evento; gráficos em linhas verticais, uma variável por linha.", "Tabelas completas de médias evento–fase e de intensidade P90; figuras legíveis para todas as 31 variáveis, sem mosaicos pequenos."),
    ("F3NINO_04", "F3NINO_04_diagnostico_quatro_fases.ipynb", "Diagnóstico físico das quatro fases do El Niño", "Examina gênese, crescimento, faixa de pico e decaimento com marcadores de recarga/termoclina, conteúdo de calor, SSTA, vento e atmosfera. A interpretação descreve evidência observada e não converte associação em causalidade.", "Antes do onset, WWV/D20/OHC devem expressar pré-condicionamento; no crescimento, vento zonal e aquecimento se amplificam; no pico, SSTA é máxima; no decaimento, os indicadores de recarga tendem a enfraquecer ou inverter.", "A estrutura recarga–descarga e o acoplamento oceano–atmosfera são mecanismos físicos testáveis por assinaturas de fase, não por um limiar de percentil.", "Médias por evento, IC bootstrap, Friedman para diferenças entre fases, tamanho de efeito Kendall W e correção FDR para as 31 variáveis.", "Assinaturas por fase, teste múltiplo e quatro gráficos grandes, um para cada fase, com intervalos de incerteza."),
    ("F3NINO_05", "F3NINO_05_reducao_variaveis_ciclo_vida.ipynb", "Redução transparente de variáveis e ciclo de vida", "Reduz as 30 variáveis explicativas, excluindo SSTA Niño-3.4 por ser o alvo térmico. Variáveis colineares são agrupadas por |rho de Spearman| >= 0,85; o representante é escolhido por sinal entre fases e tamanho de efeito, com motivo registrado.", "Poucos representantes não redundantes preservam a leitura física do ciclo e melhoram a interpretação frente a uma lista de 31 variáveis altamente correlacionadas.", "A redução deve ser rastreável e não pode chamar uma proxy do alvo de explicadora independente; por isso o alvo é excluído e o critério é explícito.", "Matriz de médias evento–fase, agrupamento por correlação, ranking descritivo e seleção de um representante por cluster.", "Tabela de redundância, tabela de representantes e ciclo de vida legível, com uma variável por linha. 'Indispensável' significa representante descritivo, não causalidade necessária."),
    ("F3NINO_06", "F3NINO_06_pca_comparativa_eof_mapas.ipynb", "PCA comparativa por fase e EOF espacial de SSTA", "Executa PCA separada para cada fase sobre as 30 explicadoras padronizadas e, em paralelo, EOF1 de mapas de SSTA por evento. Assim, a comparação por fases não é escondida em uma única PCA misturada.", "A importância relativa dos modos multivariados e o padrão espacial dominante de SSTA variam entre fases do ciclo.", "PCA é exploratória: resume covariação, não identifica causalidade. A separação por fase evita que a fase de pico domine artificialmente toda a estrutura.", "Imputação mediana declarada, padronização, PCA por fase, variância, loadings, scores e EOF1 espacial com sinal orientado positivo em Niño-3.4.", "Tabelas completas de PCA/EOF e scree, loadings grandes por fase e mapas EOF1 para comparação espacial."),
    ("F3NINO_07", "F3NINO_07_compostos_ssta_mapas.ipynb", "Mapas compostos de SSTA por fase", "Cria mapas de SSTA OISST em 20°S–20°N e 120°E–280°E, usando climatologia mensal 1991–2020 e composto de semanas de cada fase ONI. A caixa Niño-3.4 é desenhada no mapa, mas não substitui a bacia equatorial.", "O aquecimento composto se deslocará e se intensificará de maneira distinta nas quatro fases; a estrutura espacial permitirá avaliar a representatividade do índice Niño-3.4.", "Mapas são indispensáveis para não confundir um índice regional com aquecimento simultâneo da bacia inteira.", "Extração de OISST 0,25° agregada a 1°, remoção de climatologia mensal 1991–2020, composto por fase e contagem de semanas/eventos.", "Tabela de todos os pixels compostos e um mapa amplo e legível por fase, salvo tanto no notebook quanto no diretório de figuras."),
    ("F3NINO_08", "F3NINO_08_kelvin_e_ventos.ipynb", "Kelvin, ventos e acoplamento oceano–atmosfera", "Relaciona tensão/vento zonal em Niño-3.4 às fases ONI e verifica a coerência temporal oeste–leste do SSH nos anos com cobertura longitudinal (1997–98, 2015–16 e 2023–presente).", "Anomalias de vento de oeste e pulsos positivos de SSH com defasagem oeste→leste serão compatíveis com a ponte dinâmica de ondas de Kelvin, mas não constituem prova de causalidade isolada.", "O diagnóstico de Kelvin exige estrutura longitude–tempo; a tabela separa essa evidência do simples vento médio da caixa Niño-3.4.", "Média evento–fase de tau_x/u10/v10 com IC bootstrap; SSH relativo aos 84 dias pré-onset e busca de máxima correlação defasada entre bandas oeste e leste.", "Tabela de vento, tabela de coerência SSH e gráfico de vento por fase. Cobertura espacial e limitações são explícitas."),
    ("F3NINO_09", "F3NINO_09_hovmoller_ssta.ipynb", "Hovmöller equatorial de SSTA", "Constrói um Hovmöller composto de SSTA em 2°S–2°N e 120°E–280°E, alinhado ao onset ONI. As fases são marcadas por suas posições medianas em semanas relativas.", "O padrão longitude–tempo permite distinguir propagação/organização zonal de uma resposta apenas local em Niño-3.4.", "O Hovmöller é a leitura correta para tempo–longitude; ele não é apresentado como um mapa latitude–longitude nem como prova automática de uma onda de Kelvin.", "Composto equatorial semanal, alinhamento por onset ONI, guias de fase e caixa Niño-3.4 demarcada.", "Tabela longitude–tempo completa, guias de fase e uma figura Hovmöller ampla e diretamente interpretável."),
    ("F3NINO_10", "F3NINO_10_sensibilidade_e_incerteza.ipynb", "Sensibilidade das fronteiras das fases", "Avalia a robustez das quatro fases a três janelas de gênese (13, 26 e 39 semanas) e três frações de faixa de pico (80%, 90% e 95% do ONI máximo do evento).", "A estrutura geral do ciclo deverá persistir, embora as durações exatas das fases dependam da escolha operacional.", "Fronteiras de fase são diagnósticas e retrospectivas; a análise de sensibilidade impede que uma escolha única seja confundida com verdade física.", "Grade 3×3, redetecção das fronteiras, duração por evento/fase e comparação de medianas. A configuração 26 semanas/90% é apenas a canônica, não a única exibida.", "Tabela completa de robustez e gráfico de duração por fase, permitindo identificar decisões estáveis e frágeis."),
    ("F3NINO_11", "F3NINO_11_sintese_e_referencias.ipynb", "Síntese científica e referências", "Fecha a fase com critérios, cobertura, variáveis representantes e referências. A síntese preserva a distinção entre evento ONI, intensidade P90 e diagnóstico retrospectivo das fases.", "Uma narrativa que combina estatística por evento, redução transparente e mapas espaciais é mais informativa e auditável que um único limiar ou um gráfico comprimido.", "A síntese é necessária para deixar explícito o que foi observado, o que é inferência física e o que permanece limitado pela cobertura de dados.", "Releitura das tabelas canônicas e ciclo de vida reduzido; registro de referências e dos contratos de interpretação.", "Tabela metodológica, representantes, bibliografia e figura final do ciclo de vida. Não adiciona testes novos nem reinterpreta resultados sem tabela de suporte."),
]


def intro(code: str, title: str, context: str, hypothesis: str, motivation: str, methodology: str, expected: str) -> str:
    command = ""
    if code == "F3NINO_01":
        command = (
            "**COMANDO WSL2 — EXECUTAR FASE COMPLETA**\n\n"
            "```bash\nmake fase3\n```\n\n"
        )
    return f"""{command}**TÍTULO**

{code} — {title}

**Projeto:** NINO-BRASIL · **Fase:** FASE3-NINO · **domínio:** Pacífico equatorial, com foco em Niño-3.4  
**Período:** 1981–presente em semanas · **variáveis:** 31 · **resolução de figuras:** {700} ppi (mínimo exigido: 600 ppi)

**CONTEXTO**

{context}

**MOTIVAÇÃO**

*Hipótese específica*

{hypothesis}

*Função dos testes*

{motivation}

**METODOLOGIA**

{methodology}

**RESULTADOS ESPERADOS**

{expected}

*Regra científica transversal*

- **Ocorrência de El Niño:** ONI local compatível, não P90.
- **P90:** estratificação de intensidade da SSTA semanal; nunca é usado para fragmentar ou definir eventos.
- **Fases:** rótulos retrospectivos; não são disponíveis na origem do evento.
- **Unidade independente:** evento, não semana individual.
- **Saídas completas:** tabelas em `data/processed/numeric-tables/fase3_nino_cientifica/` e figuras em `data/processed/figures/fase3_nino_cientifica/`. O notebook mostra apenas o início de cada tabela e todas as figuras correspondentes.
"""


def code_cell(code: str) -> str:
    return f"""from pathlib import Path
import sys
import pandas as pd
from IPython.display import Image, Markdown, display

ROOT = next(path for path in [Path.cwd(), *Path.cwd().parents] if (path / 'pyproject.toml').exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.fase3_nino_cientifica import run_notebook

RESULTADO = run_notebook('{code}')
display(Markdown(
    f\"**Eventos ONI:** {{RESULTADO['n_eventos_oni']}} · **P90 de intensidade:** {{RESULTADO['limiar_p90_c']:.3f}} °C · **Figuras:** {{RESULTADO['dpi']}} ppi\"
))
for table_path in RESULTADO['tables']:
    table = pd.read_csv(table_path)
    display(Markdown(f\"**INÍCIO DA TABELA — {{Path(table_path).name}}**\"))
    display(table.head(12))
    display(Markdown(f\"Tabela completa persistida em `{{table_path}}` ({{len(table)}} linhas).\"))
for figure_path in RESULTADO['figures']:
    display(Markdown(f\"**FIGURA — {{Path(figure_path).name}}**\"))
    display(Image(filename=str(figure_path)))
"""


def notebook(item: tuple[str, str, str, str, str, str, str, str]) -> dict:
    code, filename, title, context, hypothesis, motivation, methodology, expected = item
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": intro(code, title, context, hypothesis, motivation, methodology, expected).splitlines(keepends=True)},
        {"cell_type": "markdown", "metadata": {}, "source": ["**SALVAGUARDAS DE INTERPRETAÇÃO**\n\n", "- Não se infere causalidade de correlação, PCA ou composto.\n", "- O ONI local é compatível com a regra NOAA, mas usa OISST local; não é apresentado como a série oficial ERSSTv5.\n", "- Campos espaciais de vento disponíveis são limitados à cobertura Niño-3.4 quando essa limitação for indicada no resultado.\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_cell(code).splitlines(keepends=True)},
        {"cell_type": "markdown", "metadata": {}, "source": REFERENCES.splitlines(keepends=True)},
    ]
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}, "fase3_nino": {"specification": "oni-scientific-rebuild", "notebook_code": code, "figure_dpi": 700}}, "nbformat": 4, "nbformat_minor": 5}


def main() -> int:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    for item in NOTEBOOKS:
        (_, filename, *_rest) = item
        (NOTEBOOK_DIR / filename).write_text(json.dumps(notebook(item), ensure_ascii=False, indent=2), encoding="utf-8")
    (NOTEBOOK_DIR / "README.md").write_text(
        "# FASE3-NINO — reconstrução científica\n\n"
        "Série de 11 notebooks baseada em ONI local compatível para ocorrência, P90 somente para intensidade, quatro fases retrospectivas, estatística por evento, redução de redundância, PCA por fase, mapas OISST, Kelvin/vento, Hovmöller e sensibilidade.\n\n"
        "Todas as tabelas completas são salvas em `data/processed/numeric-tables/fase3_nino_cientifica/` e todas as figuras em `data/processed/figures/fase3_nino_cientifica/`, com 700 ppi. Os notebooks mostram somente as primeiras 12 linhas das tabelas.\n",
        encoding="utf-8",
    )
    print(f"[ok] {len(NOTEBOOKS)} notebooks científicos gerados em {NOTEBOOK_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
