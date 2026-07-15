#!/usr/bin/env python3
"""Padroniza a premissa de TODOS os notebooks do NINO-BRASIL (idempotente).

Injeta, preservando todas as células de código:
  - Cabeçalho com Descritivo, Pergunta, Desafio (hipótese) e Metodologia (+refs),
    o CÓDIGO da fase/letra e o Contrato de saídas (código predecessor único
    Fig_<F><B><NN> que nomeia figura E numeric-table, com sobreposição).
  - Rodapé com as Referências Bibliográficas.

Reexecutar sobrescreve o cabeçalho/rodapé (marcados por sentinela), nunca o código.

Uso:
    python scripts/padronizar_notebooks.py --in-place      # grava no próprio .ipynb
    python scripts/padronizar_notebooks.py --out-dir DIR   # grava espelho (revisão)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NBROOT = ROOT / "notebooks"
SENT_H = "<!-- NINO26-CABECALHO v1 -->"
SENT_R = "<!-- NINO26-REFERENCIAS v1 -->"

BIB = {
    "bjerknes1969": "Bjerknes, J. (1969). Atmospheric Teleconnections from the Equatorial Pacific. *Monthly Weather Review*, 97(3), 163-172. https://doi.org/10.1175/1520-0493(1969)097<0163:ATFTEP>2.3.CO;2",
    "jin1997": "Jin, F.-F. (1997). An Equatorial Ocean Recharge Paradigm for ENSO. Part I. *J. Atmos. Sci.*, 54, 811-829. https://doi.org/10.1175/1520-0469(1997)054<0811:AEORPF>2.0.CO;2",
    "meinen2000": "Meinen, C. S., & McPhaden, M. J. (2000). Observations of Warm Water Volume Changes... *J. Climate*, 13, 3551-3559. https://doi.org/10.1175/1520-0442(2000)013<3551:OOWWVC>2.0.CO;2",
    "trenberth1997": "Trenberth, K. E. (1997). The Definition of El Niño. *BAMS*, 78, 2771-2777. https://doi.org/10.1175/1520-0477(1997)078<2771:TDOENO>2.0.CO;2",
    "timmermann2018": "Timmermann, A., et al. (2018). El Niño-Southern Oscillation complexity. *Nature*, 559, 535-545. https://doi.org/10.1038/s41586-018-0252-6",
    "huang2021": "Huang, B., et al. (2021). Improvements of the DOISST Version 2.1. *J. Climate*, 34, 2923-2939. https://doi.org/10.1175/JCLI-D-20-0166.1",
    "hersbach2020": "Hersbach, H., et al. (2020). The ERA5 global reanalysis. *QJRMS*, 146, 1999-2049. https://doi.org/10.1002/qj.3803",
    "grimm2009": "Grimm, A. M., & Tedeschi, R. G. (2009). ENSO and Extreme Rainfall Events in South America. *J. Climate*, 22, 1589-1609. https://doi.org/10.1175/2008JCLI2429.1",
    "grimm1998": "Grimm, A. M., Ferraz, S. E. T., & Gomes, J. (1998). Precipitation Anomalies in Southern Brazil Associated with El Niño and La Niña. *J. Climate*, 11, 2863-2880. https://doi.org/10.1175/1520-0442(1998)011<2863:PAISBA>2.0.CO;2",
    "cai2020": "Cai, W., et al. (2020). Climate impacts of ENSO on South America. *Nat. Rev. Earth Environ.*, 1, 215-231. https://doi.org/10.1038/s43017-020-0040-3",
    "funk2015": "Funk, C., et al. (2015). CHIRPS. *Scientific Data*, 2, 150066. https://doi.org/10.1038/sdata.2015.66",
    "vicente2010": "Vicente-Serrano, S. M., et al. (2010). SPEI. *J. Climate*, 23, 1696-1718. https://doi.org/10.1175/2009JCLI2909.1",
    "benjamini1995": "Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B*, 57, 289-300. https://doi.org/10.1111/j.2517-6161.1995.tb02031.x",
    "wilks2016": "Wilks, D. S. (2016). 'The Stippling Shows Statistically Significant Grid Points'. *BAMS*, 97, 2263-2273. https://doi.org/10.1175/BAMS-D-15-00267.1",
    "bretherton1999": "Bretherton, C. S., et al. (1999). The Effective Number of Spatial Degrees of Freedom. *J. Climate*, 12, 1990-2009. https://doi.org/10.1175/1520-0442(1999)012<1990:TENOSD>2.0.CO;2",
    "roberts2017": "Roberts, D. R., et al. (2017). Cross-validation strategies for data with temporal/spatial/hierarchical structure. *Ecography*, 40, 913-929. https://doi.org/10.1111/ecog.02881",
    "cawley2010": "Cawley, G. C., & Talbot, N. L. C. (2010). On Over-fitting in Model Selection... *JMLR*, 11, 2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html",
    "shi2015": "Shi, X., et al. (2015). Convolutional LSTM Network. *NeurIPS 28*. https://arxiv.org/abs/1506.04214",
    "ham2019": "Ham, Y.-G., Kim, J.-H., & Luo, J.-J. (2019). Deep learning for multi-year ENSO forecasts. *Nature*, 573, 568-572. https://doi.org/10.1038/s41586-019-1559-7",
    "chattopadhyay2020": "Chattopadhyay, A., et al. (2020). Predicting clustered weather patterns (CNN espaço-temporal). *Scientific Reports*, 10, 1317. https://doi.org/10.1038/s41598-020-57897-9",
    "taylor2022": "Taylor, J., & Feng, M. (2022). A deep learning model for forecasting global monthly mean SST anomalies. *Front. Climate*, 4, 932932. https://doi.org/10.3389/fclim.2022.932932",
    "mir2024": "Mir, A. A., et al. (2024). ENSO dataset & comparison of deep learning models. *Earth Sci. Inform.* https://doi.org/10.1007/s12145-024-01295-6",
    "bachelery2025": "Bachèlery, M.-L., et al. (2025). Predicting Atlantic and Benguela Niño events with deep learning. *Science Advances*, 11, eads5185. https://doi.org/10.1126/sciadv.ads5185",
    "bracco2024": "Bracco, A., et al. (2024). Machine learning for the physics of climate. *Nat. Rev. Phys.*, 7, 6-20. https://doi.org/10.1038/s42254-024-00776-3",
    "tedeschi2025": "Tedeschi, R. G., et al. (2025). Multivariable modelling (stats+ML) for monthly precipitation forecasting. *Front. Earth Sci.*, 13, 1576377. https://doi.org/10.3389/feart.2025.1576377",
    "dantas2020": "Dantas, L. G., et al. (2020). Rainfall Prediction in Paraíba (NE Brazil) using GAMs. *Water*, 12, 2478. https://doi.org/10.3390/w12092478",
    "cui2025": "Cui, J., et al. (2025). Mixed Layer and Oceanic Kelvin Wave Response to Equatorial Pacific WWEs. *JGR-Oceans*. https://doi.org/10.1029/2025JC023275",
}


def figs(*pairs):
    return [{"codigo": c, "desc": d} for c, d in pairs]


# nb_rel -> metadados. `figs` = contrato de saídas (código único -> descrição).
NB: dict[str, dict] = {
    "fase2/2Z_sanidade_variaveis.ipynb": dict(
        codigo="2Z", fase=2, bloco="Z", hip="— (pré-condição das Fases 3-8)",
        titulo="Sanidade sensorial de todas as variáveis (fecho da Fase 2)",
        descritivo="A matriz-mestre semanal (31 variáveis físicas: 17 oceânicas + 14 ERA5) é a base de todas as fases seguintes. Antes de qualquer inferência é obrigatório inspecionar cada série para flagrar unidade errada, salto na emenda de fontes, lacuna não declarada e artefato sazonal.",
        pergunta="Todas as variáveis do master semanal 1981-2026 têm unidade, cobertura e comportamento sazonal plausíveis, sem descontinuidade espúria na emenda UFS->GLORYS?",
        desafio="Nenhuma variável avança para a Fase 3 com unidade inconsistente, salto de fonte ou lacuna silenciosa; anomalia e valor físico cru ficam explicitamente distinguidos.",
        metodologia="Plotagem padronizada de cada série com sombreamento El Niño (vermelho)/La Niña (azul); conferência de escala/amplitude sazonal e de unidades (fluxos ERA5 acumulados; OISST/GLORYS) — Huang et al. (2021); Hersbach et al. (2020); Funk et al. (2015).",
        refs=["huang2021", "hersbach2020", "funk2015"],
        figs=figs(("Fig_2Z01", "sanidade oceano superfície (SSTA)"), ("Fig_2Z02", "sanidade oceano recarga (D20/OHC/WWV/SSH)"),
                  ("Fig_2Z03", "sanidade perfil de temperatura T(z)"), ("Fig_2Z04", "sanidade atmosfera Bjerknes (ERA5)"),
                  ("Fig_2Z05", "painel z-score de todas as variáveis")),
    ),
    "fase3/3A_indices_fisicos_semanais.ipynb": dict(
        codigo="3A", fase=3, bloco="A", hip="HIP0 (mecanismo do ciclo)",
        titulo="Índices físicos semanais do Niño 3.4",
        descritivo="Materializa a matriz semanal canônica da Fase 3 e mapeia, variável a variável, a proveniência (anomalia vs valor físico original) e a cobertura temporal real — a base física para diagnosticar gênese, crescimento, pico e decaimento.",
        pergunta="Quais variáveis descrevem o estado do sistema Niño 3.4, quais são anomalias, quais são valores físicos crus e qual é a cobertura real de cada série?",
        desafio="Distinguir anomalia de valor cru é pré-requisito: uma variável sazonal crua (ex.: D20) não pode ser lida como sinal interanual sem anomalização.",
        metodologia="Séries semanais e Hovmöller de SSTA/SLA por longitude, sobre a caixa Niño 3.4 (definição de El Niño de Trenberth, 1997; acoplamento de Bjerknes, 1969).",
        refs=["trenberth1997", "bjerknes1969"],
        figs=figs(("Fig_3A01", "séries físicas semanais"), ("Fig_3A02", "Hovmöller de SSTA por longitude"),
                  ("Fig_3A03", "Hovmöller de SLA e tensão zonal")),
    ),
    "fase3/3B_alvo_eventos_ciclo_vida.ipynb": dict(
        codigo="3B", fase=3, bloco="B", hip="HIP0",
        titulo="Alvo, eventos e ciclo de vida (NOAA/ONI local)",
        descritivo="Deriva o catálogo local de eventos El Niño/La Niña pela regra NOAA/ONI (sem rótulo externo) e delimita a FAIXA de pico — o alvo de todas as análises de fase da coluna do mecanismo.",
        pergunta="Como os eventos locais nascem, crescem, atingem o pico e decaem por classe (fraco/moderado/forte/muito forte) e onde começa/termina a faixa de pico de cada evento?",
        desafio="O critério local (ONI>=±0,5 °C por >=5 estações móveis) precisa reproduzir a cronologia NOAA e definir o pico como faixa reprodutível, não como um único mês.",
        metodologia="Média móvel trimestral do SSTA, limiar ONI e classificação por pico; faixa de pico por fração do |ONI| máximo (Trenberth, 1997; Timmermann et al., 2018).",
        refs=["trenberth1997", "timmermann2018"],
        figs=figs(("Fig_3B01", "trajetórias compostas por classe"), ("Fig_3B02", "autocorrelação/persistência"),
                  ("Fig_3B03", "composto de SSTA no pico"), ("Fig_3B04", "faixa de pico do ONI")),
    ),
    "fase3/3C_precursores_lags.ipynb": dict(
        codigo="3C", fase=3, bloco="C", hip="HIP0",
        titulo="Precursores: o que antecede o pico do Niño 3.4?",
        descritivo="Constrói o ranking preliminar de precursores do aquecimento máximo — o teste direto do paradigma de recarga (WWV/OHC/D20 liderando o SSTA).",
        pergunta="Quais variáveis físicas lideram a SSTA do Niño 3.4 e em que defasagem semanal?",
        desafio="Separar precursor verdadeiro (lidera com antecedência) de estado contemporâneo; o sinal de recarga deve anteceder o pico por semanas.",
        metodologia="Correlação defasada preditor(t-lag)->SSTA(t), excluindo o alvo e seus aliases do catalogo auditavel, e ranking por lag antes dos filtros de rigor do 3D/3E (paradigma de recarga de Jin, 1997; Meinen & McPhaden, 2000).",
        refs=["jin1997", "meinen2000"],
        figs=figs(("Fig_3C01", "heatmap de correlação por lag"), ("Fig_3C02", "mapa longitude x lag")),
    ),
    "fase3/3D_rigor_estatistico.ipynb": dict(
        codigo="3D", fase=3, bloco="D", hip="HIP0",
        titulo="Rigor: o que sobrevive estatisticamente?",
        descritivo="Converte a triagem do 3C em um conjunto defensável, aplicando graus de liberdade efetivos, FDR e IC95 — o filtro que evita significância inflada por autocorrelação.",
        pergunta="Quais relações defasadas do 3C sobrevivem a N_eff, FDR e IC95 que exclua zero?",
        desafio="Em séries semanais fortemente autocorrelacionadas, n bruto superestima a significância; só relações que passam N_eff+FDR entram no parecer.",
        metodologia="N_eff de Bretherton por par, teste t com graus efetivos, controle FDR de Benjamini-Hochberg sobre todos os precursores candidatos x lags, alvo/aliases excluidos, e IC95 (Bretherton et al., 1999; Benjamini & Hochberg, 1995; Wilks, 2016).",
        refs=["bretherton1999", "benjamini1995", "wilks2016"],
        figs=figs(("Fig_3D01", "forest plot de IC95 significativos"), ("Fig_3D02", "mapa longitude x lag pós-FDR")),
    ),
    "fase3/3E_sensibilidade_temporal.ipynb": dict(
        codigo="3E", fase=3, bloco="E", hip="HIP0",
        titulo="Sensibilidade temporal sem breakpoint pré-fixado",
        descritivo="Quantifica quanto os achados dependem de eventos individuais e da autocorrelação, sem impor regimes artificiais — a defesa contra conclusões dominadas por poucos eventos.",
        pergunta="Quanto as correlações que sobreviveram ao 3D dependem de eventos ENSO individuais e da estrutura autocorrelacionada?",
        desafio="A unidade de incerteza deve ser o evento, não a semana; nenhum corte temporal fixo (ex.: 1993/2010) pode entrar como filtro.",
        metodologia="Bootstrap em blocos e leave-one-event-out, com envelope de IC e fração de mesmo sinal (validação para dados dependentes — Roberts et al., 2017; Wilks, 2016).",
        refs=["roberts2017", "wilks2016"],
        figs=figs(("Fig_3E01", "sensibilidade bootstrap/LOO"), ("Fig_3E02", "influência de cada evento (LOO)")),
    ),
    "fase3/3F_kelvin_sla.ipynb": dict(
        codigo="3F", fase=3, bloco="F", hip="HIP0",
        titulo="Ondas de Kelvin por SLA/SSH e vento",
        descritivo="Documenta a propagação equatorial oeste-leste compatível com ondas de Kelvin de downwelling — a ponte dinâmica entre rajadas de oeste e o aquecimento do leste.",
        pergunta="A dinâmica equatorial mostra propagação compatível com ondas de Kelvin nos eventos fortes e na formação 2025/26?",
        desafio="A leitura de propagação exige a tensão do vento adequada (não só |u|u zonal); o sinal de Kelvin deve preceder o aquecimento.",
        metodologia="Hovmöller de SLA/SSH com setas de propagação e sobreposição da tensão zonal por evento (Bjerknes, 1969; vieses de resposta a WWEs em Cui et al., 2025).",
        refs=["bjerknes1969", "cui2025"],
        figs=figs(("Fig_3F01", "Hovmöller SLA com setas de Kelvin"), ("Fig_3F02", "tensão zonal x SLA por evento")),
    ),
    "fase3/3G_compostos_ssta.ipynb": dict(
        codigo="3G", fase=3, bloco="G", hip="HIP0",
        titulo="Compostos de SSTA por classe NOAA e comparação 2025/26",
        descritivo="Compara a evolução espacial da SSTA por classe de intensidade e situa a formação 2025/26 frente aos eventos fortes — a leitura de epicentro e escalonamento do aquecimento.",
        pergunta="Como a SSTA evolui por classe NOAA e como 2025/26 se compara aos eventos fortes/muito fortes por longitude? Onde fica o epicentro?",
        desafio="Distinguir intensidade de localização (canônico vs Modoki) exige comparar padrões espaciais, não só o índice escalar.",
        metodologia="Compostos de SSTA alinhados por classe e por longitude, com estado atual sobreposto (diversidade de eventos — Timmermann et al., 2018; Trenberth, 1997).",
        refs=["timmermann2018", "trenberth1997"],
        figs=figs(("Fig_3G01", "composto de SSTA por classe NOAA"), ("Fig_3G02", "escalonamento de SSTA"),
                  ("Fig_3G03", "mapa de SSTA por longitude")),
    ),
    "fase3/3H_genese_precursores_classe.ipynb": dict(
        codigo="3H", fase=3, bloco="H", hip="HIP0",
        titulo="Gênese: o que separa as classes NOAA de El Niño?",
        descritivo="Caracteriza o estado precursor (pré-onset) e testa se ele já separa as classes de intensidade — o pré-condicionamento de recarga como discriminante da magnitude futura.",
        pergunta="Que valores as variáveis assumem antes do onset e o estado precursor separa fraco/moderado/forte/muito forte?",
        desafio="Rótulos de gênese não podem usar informação futura; o discriminante deve ser causal (só dados até o pré-onset).",
        metodologia="Compostos alinhados ao onset e ciclo de vida médio de sub/superfície+atmosfera por classe (recarga de Jin, 1997; Meinen & McPhaden, 2000).",
        refs=["jin1997", "meinen2000"],
        figs=figs(("Fig_3H01", "compostos alinhados ao onset"), ("Fig_3H02", "ciclo de vida médio"),
                  ("Fig_3H03", "ciclo de vida subsuperfície+atmosfera")),
    ),
    "fase3/3I_interpretacao_integrada.ipynb": dict(
        codigo="3I", fase=3, bloco="I", hip="HIP0",
        titulo="Interpretação integrada da Fase 3",
        descritivo="Sintetiza 3A-3H/3K num conjunto de precursores com antecedência e numa leitura cautelosa para 2025/26 — a evidência que entra no parecer do mecanismo.",
        pergunta="Qual conjunto de variáveis explica o aquecimento máximo com antecedência e qual é a leitura prospectiva cautelosa para 2025/26?",
        desafio="A projeção condicional deve ser rotulada como exploratória, com IC largo declarado e sem vazamento de informação futura.",
        metodologia="Consolidação dos precursores, antecipação do pico e projeção condicional com validação nested LOO (Cawley & Talbot, 2010; complexidade de ENSO em Timmermann et al., 2018; Cai et al., 2020).",
        refs=["timmermann2018", "cai2020", "cawley2010"],
        figs=figs(("Fig_3I01", "síntese para o parecer"), ("Fig_3I02", "antecipação do pico"),
                  ("Fig_3I03", "projeção condicional nested"), ("Fig_3I04", "galeria de figuras da Fase 3")),
    ),
    "fase3/3K_pca_crescimento.ipynb": dict(
        codigo="3K", fase=3, bloco="K", hip="HIP0",
        titulo="PCA do bloco de crescimento/acoplamento",
        descritivo="Reduz o bloco de crescimento ao conjunto mínimo de dimensões físicas independentes — evita tratar variáveis colineares como evidências separadas.",
        pergunta="Qual é o conjunto mínimo indispensável de dimensões físicas independentes que descrevem o crescimento/acoplamento?",
        desafio="PCA de índices escalares não é EOF espacial; a nomenclatura e o número de graus de liberdade efetivos devem ser explícitos.",
        metodologia="PCA com scree/biplot e skill LOO nested; graus de liberdade efetivos (Bretherton et al., 1999; Cawley & Talbot, 2010).",
        refs=["bretherton1999", "cawley2010"],
        figs=figs(("Fig_3K01", "skill LOO nested"), ("Fig_3K02", "scree plot"), ("Fig_3K03", "biplot")),
    ),
    "fase3/3L_en_ln_caracterizacao.ipynb": dict(
        codigo="3L", fase=3, bloco="L", hip="HIP0",
        titulo="Caracterização das 4 fases: El Niño x La Niña",
        descritivo="Camada visual que percorre os quatro períodos para EN e LN, com duração por classe e discriminantes por fase — o fecho descritivo do mecanismo.",
        pergunta="Como El Niño e La Niña percorrem gênese, crescimento, pico e decaimento, quanto dura cada fase por classe e quais variáveis melhor delimitam cada período?",
        desafio="A assimetria EN/LN precisa aparecer; a caracterização é diagnóstica (rótulos post hoc), não previsão.",
        metodologia="Ciclo de vida EN/LN com faixa de pico canônica de 90% (sensibilidade 80/90/95%), Friedman pareado por evento + Kendall W, BH-FDR sobre todas as variaveis separadamente por tipo ENSO e PCA por fase; diagnóstico retrospectivo separado de rolling-origin (Bjerknes, 1969; Jin, 1997).",
        refs=["bjerknes1969", "jin1997"],
        figs=figs(("Fig_3L01", "ciclo de vida EN x LN"), ("Fig_3L02", "heatmap de discriminantes"),
                  ("Fig_3L03", "duração das fases EN x LN"), ("Fig_3L04", "PCA por fase")),
    ),
    "fase4/4_0_fase4_abertura.ipynb": dict(
        codigo="40", fase=4, bloco="0", hip="HIP1 (teleconexão Brasil)",
        titulo="Abertura da Fase 4 e redução dimensional do vetor do Pacífico",
        descritivo="Abre a teleconexão Pacífico->chuva do Brasil e reduz as ~31 variáveis do Pacífico a um conjunto não redundante, evitando testar dezenas de proxies colineares da mesma dimensão física.",
        pergunta="Qual é o subconjunto mínimo e não redundante de preditores do Pacífico que representa as dimensões físicas independentes para os testes da Fase 4?",
        desafio="Reduzir sem descartar sinal: o filtro deve ser estável no tempo e ajustado só no treino (sem vazamento).",
        metodologia="Filtro de redundância + PCA(90%) + estabilidade temporal sobre anomalias harmônicas ajustadas no treino (graus de liberdade efetivos — Bretherton et al., 1999; Wilks, 2016).",
        refs=["bretherton1999", "wilks2016"],
        figs=figs(("Fig_4001", "seleção de variáveis por PCA"), ("Fig_4002", "cobertura de dados")),
    ),
    "fase4/4A_ciclo_enso_fases.ipynb": dict(
        codigo="4A", fase=4, bloco="A", hip="HIP1",
        titulo="Ciclo ENSO em 4 fases: rótulo semanal e cronologia",
        descritivo="Transforma o ONI local mensal numa tabela semanal auditável (tipo/fase/evento por domingo) — o rótulo que condiciona toda a teleconexão da Fase 4.",
        pergunta="Em que fase do ciclo ENSO está cada semana do projeto, para El Niño e La Niña?",
        desafio="O rótulo semanal precisa ser consistente com o catálogo da Fase 3 e não sobrescrever eventos na janela de gênese.",
        metodologia="Rotulagem semanal por evento (gênese/crescimento/pico/decaimento) e cronologia por classe (Trenberth, 1997; teleconexões de Grimm & Tedeschi, 2009).",
        refs=["trenberth1997", "grimm2009"],
        figs=figs(("Fig_4A01", "ciclo ENSO em 4 fases"), ("Fig_4A02", "distribuição de duração por fase"),
                  ("Fig_4A03", "plano cronológico ONI/tendência")),
    ),
    "fase4/4B_variaveis_determinantes_fases.ipynb": dict(
        codigo="4B", fase=4, bloco="B", hip="HIP1",
        titulo="Variáveis determinantes de cada fase + MLR",
        descritivo="Mede quais variáveis do Pacífico determinam cada fase, por efeito não-paramétrico e por regressão linear multivariada — a leitura estrutural antes do sinal espacial.",
        pergunta="Quais variáveis do Pacífico mais determinam gênese, crescimento, pico e decaimento?",
        desafio="Coeficientes precisam de incerteza por evento (HAC/Newey-West); barras cruzando zero não são distinguíveis de nulo.",
        metodologia="Friedman pareado por evento/Kendall W e MLR padronizada com incerteza por evento; controle de multiplicidade e significância de campo (Wilks, 2016; Bretherton et al., 1999).",
        refs=["wilks2016", "bretherton1999"],
        figs=figs(("Fig_4B01", "coeficientes MLR por fase (El Niño)"), ("Fig_4B02", "heatmap de determinantes"),
                  ("Fig_4B03", "ranking de discriminância"), ("Fig_4B04", "relações entre pares")),
    ),
    "fase4/4C_sinal_pixel_lags.ipynb": dict(
        codigo="4C", fase=4, bloco="C", hip="HIP1",
        titulo="Sinal pixel-a-pixel, por região IBGE e por bioma",
        descritivo="Distribui o sinal Pacífico->chuva por pixel, região e bioma, por fase do ciclo — o coração espacial da teleconexão e o que sustenta ou refuta a hipótese NEB seco/Sul úmido.",
        pergunta="Onde, quando e com que sinal (seco/úmido) e em que lag o Pacífico altera a chuva do Brasil, por pixel/região/bioma?",
        desafio="O alvo precisa manter os pixels CHIRPS nativos, semanas completas e escalas robustas; a significância exige N_eff, FDR e teste de campo por família.",
        metodologia="Pacífico(t-lag)->CHIRPS(t) nos pixels originais, condicionado à fase em t-lag, para 31 variáveis e EN/LN x 4 fases; N_eff, BH-FDR e significância de campo, com região/bioma apenas como agregações reversíveis (Grimm & Tedeschi, 2009; Bretherton et al., 1999; Benjamini & Hochberg, 1995; Funk et al., 2015).",
        refs=["grimm2009", "bretherton1999", "benjamini1995", "funk2015", "cai2020"],
        figs=figs(
            ("Fig_4C01_lags_regiao_bioma_el_nino", "lags por região/bioma (El Niño)"),
            ("Fig_4C02_lags_regiao_bioma_la_nina", "lags por região/bioma (La Niña)"),
            ("Fig_4C03_mapa_pixel_el_nino_pico", "mapa nativo FDR no pico de El Niño"),
            ("Fig_4C04_mapa_pixel_la_nina_pico", "mapa nativo FDR no pico de La Niña"),
        ),
    ),
    "fase4/4D_clusters_alvo.ipynb": dict(
        codigo="4D", fase=4, bloco="D", hip="HIP1",
        titulo="Alvos clusterizados e gate estatístico da hipótese",
        descritivo="Agrupa áreas com resposta parecida e testa formalmente a hipótese NEB seco/Sul úmido em El Niño — o gate da teleconexão (HIP1).",
        pergunta="Quais áreas respondem de modo parecido, em que lag, com que sentido seco/úmido e com que estabilidade? A evidência sustenta a hipótese?",
        desafio="O gate precisa de FDR e da máscara EN/LN correta, sem corte temporal fixo e sem confundir média de anomalia com seca/extremo — usar índices de extremos.",
        metodologia="Clusterização de perfis, lag por sinal e gate com estabilidade temporal; extremos por SPI/SPEI/ETCCDI em vez da média (Grimm & Tedeschi, 2009; Cai et al., 2020; Vicente-Serrano et al., 2010).",
        refs=["grimm2009", "cai2020", "vicente2010"],
        figs=figs(
            ("Fig_4D01_mapa_clusters_pixels_nativos", "mapa de clusters descritivos"),
            ("Fig_4D02_perfis_clusters_fdr", "perfis FDR dos clusters"),
            ("Fig_4D03_gate_multialvo_eventos", "síntese do gate multialvo por evento"),
        ),
    ),
    "fase5/5_ciclo_ml.ipynb": dict(
        codigo="5A", fase=5, bloco="A", hip="HIP2 (ML do ciclo)",
        titulo="Ciclo ENSO com Machine Learning (RF/XGBoost) + XAI",
        descritivo="Testa RF/XGBoost nos nove estados (neutro + EN/LN x quatro fases) e em alvos futuros rolling-origin, usando todas as 31 variáveis F2.",
        pergunta="Quais variáveis mudam de importância entre EN/LN e gênese/crescimento/pico/decaimento, e o ML supera estatística, climatologia e persistência fora de eventos inteiros?",
        desafio="Semanas adicionais não são eventos novos: janelas do mesmo evento ficam no mesmo fold, com embargo e preprocessing no treino; augmentation conserva original_event_id.",
        metodologia="RF/XGBoost com 31 variáveis, lags 4-52 e nove estados; folds rolling-origin por evento, embargo, transformações dentro do fold, pesos evento/tipo/fase e augmentation conservador apenas no treino; importância por estado no teste (Cawley & Talbot, 2010; Roberts et al., 2017; Ham et al., 2019).",
        refs=["cawley2010", "roberts2017", "ham2019", "bracco2024"],
        figs=figs(("Fig_5A01_rf", "importância OOS por variável/estado (RF)"), ("Fig_5A01_xgb", "importância OOS por variável/estado (XGB)"),
                  ("Fig_5A02_rf", "skill por horizonte e dimensão do evento (RF)"), ("Fig_5A02_xgb", "skill por horizonte e dimensão do evento (XGB)")),
    ),
    "fase6/6_brasil_ml.ipynb": dict(
        codigo="6A", fase=6, bloco="A", hip="HIP3 (ML Brasil)",
        titulo="Distribuição no Brasil com Machine Learning (RF/XGBoost) + XAI",
        descritivo="Testa RF/XGBoost separadamente em cada pixel CHIRPS original, usando 31 variáveis e EN/LN x quatro fases no tempo fonte t-lag.",
        pergunta="O ML prevê cada pixel nativo melhor que persistência/climatologia e a Fase 4, e quais variáveis dominam por fase e lag?",
        desafio="Nenhum regrid/agregado pode substituir o alvo: shards precisam cobrir pixels nativos sem sobreposição, e a climatologia da chuva é ajustada só no treino.",
        metodologia="RF/XGBoost pixel-a-pixel com 31 variáveis, fase em t-lag, folds inteiros por evento e target preprocessing no treino; região/bioma são resumos posteriores e o gate exige skill espacial area-ponderado (Roberts et al., 2017; Cawley & Talbot, 2010; Grimm & Tedeschi, 2009).",
        refs=["roberts2017", "cawley2010", "grimm2009", "tedeschi2025", "dantas2020"],
        figs=figs(("Fig_6A01_rf", "skill do ML por unidade (RF)"), ("Fig_6A01_xgb", "skill do ML por unidade (XGB)")),
    ),
    "fase7/7_ciclo_convlstm.ipynb": dict(
        codigo="7A", fase=7, bloco="A", hip="HIP4 (ConvLSTM do ciclo)",
        titulo="Ciclo ENSO com redes neurais ConvLSTM",
        descritivo="Testa se campos espaciais do Pacífico (ConvLSTM) acrescentam skill sobre índices escalares e ML tabular — ENSO é propagação espaço-temporal, não só uma série do Niño 3.4.",
        pergunta="Campos espaciais do Pacífico acrescentam skill sobre índices escalares e ML tabular na caracterização/previsão do ciclo?",
        desafio="Amostra observacional pequena: todas as janelas de um evento ficam juntas; augmentation/masking não aumenta o N de eventos e preserva provenance.",
        metodologia="PyTorch ConvLSTM em campos GLORYS reais fundido a uma sequência nomeada das 31 variáveis; nove estados + intensidade/tempo/duração, folds por evento com embargo, masking/noise no treino e pré-treino auto-supervisionado/CMIP-NMME opcional (Shi et al., 2015; Ham et al., 2019).",
        refs=["shi2015", "ham2019", "chattopadhyay2020", "taylor2022", "mir2024"],
        figs=figs(("Fig_7A01", "skill por lead time (a treinar cientificamente)"),
                  ("Fig_7A02", "importância por oclusão de canais (a treinar)")),
    ),
    "fase8/8_brasil_convlstm.ipynb": dict(
        codigo="8A", fase=8, bloco="A", hip="HIP5 (ConvLSTM Brasil)",
        titulo="Distribuição no Brasil com redes neurais ConvLSTM (encoder-decoder)",
        descritivo="Testa se uma rede espaço-temporal mapeia a sequência do Pacífico no campo futuro de chuva do Brasil com skill superior às Fases 4 e 6 — teleconexão como mapeamento espaço->espaço condicionado ao tempo.",
        pergunta="A rede aprende a distribuição espaço-temporal da chuva do Brasil com skill superior às Fases 4 e 6, por pixel/região/bioma?",
        desafio="Alinhamento por timestamp/offset da sequência, target preprocessing no treino, decoder sem Dense gigante e preservação exata do grid/pixel_id CHIRPS.",
        metodologia="PyTorch ConvLSTM + fusão das 31 variáveis e decoder convolucional probabilístico para a forma nativa CHIRPS; máscara/fração Brasil e pesos de área atuam apenas na loss, com métricas e campos OOS por pixel original (Shi et al., 2015; Ham et al., 2019; Chattopadhyay et al., 2020).",
        refs=["shi2015", "ham2019", "bachelery2025", "chattopadhyay2020"],
        figs=figs(("Fig_8A01", "campo previsto de anomalia de chuva (a treinar)"),
                  ("Fig_8A02", "skill por pixel/região (a treinar)")),
    ),
}


def _stable_cell_id(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def md_cell(text: str, *, cell_id: str) -> dict:
    src = text.splitlines(keepends=True)
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": src}


def build_header(m: dict, nb_rel: str = "") -> str:
    linhas = [
        SENT_H,
        f"# {m['codigo']} — {m['titulo']}",
        "",
        "**Projeto NINO-BRASIL — Oceanografia Física UFPE — Thiago Vilar**  ",
        f"**Código da fase/letra:** `{m['codigo']}`  ·  **Hipótese:** {m['hip']}",
        "",
        "## Descritivo (por que este notebook existe)",
        m["descritivo"],
        "",
        "## Pergunta",
        m["pergunta"],
        "",
        "## Desafio (hipótese a testar)",
        m["desafio"],
        "",
        "## Metodologia (com referências)",
        m["metodologia"],
        "",
        "## Contrato de saídas — código predecessor único",
        "Cada figura nasce do **mesmo** `registrar_figura(...)` que congela sua "
        "numeric-table sob o **mesmo código**, reescrevendo por **sobreposição** a cada execução. "
        "A fonte deve ser uma tabela persistida com sidecar e hash do mesmo `run_id`:",
        "",
        "```python",
        "from nino_brasil.viz import registrar_figura",
        f"registrar_figura(fig, \"{m['figs'][0]['codigo']}\", fase={m['fase']}, bloco=\"{m['bloco']}\",",
        f"                 titulo=..., descricao=..., hipotese=\"{m['hip'].split()[0]}\",",
        f"                 notebook=\"notebooks/{nb_rel}\",",
        "                 run_id=run.run_id,",
        f"                 fontes={{\"<tabela>\": tabela_path}})   # Path + .manifest.json -> figures/fase{m['fase']}/<codigo>.png + numeric-tables/fase{m['fase']}/<codigo>/",
        "```",
        "",
        f"| Código | Figura (`figures/fase{m['fase']}/<código>.png`) | Numeric-table (`numeric-tables/fase{m['fase']}/<código>/`) | Descrição |",
        "|---|---|---|---|",
    ]
    for f in m["figs"]:
        linhas.append(f"| `{f['codigo']}` | `{f['codigo']}.png` | `{f['codigo']}/` | {f['desc']} |")
    linhas += [
        "",
        "> Padrão em `docs/PADRAO_NOTEBOOKS.md`; compatibilidade por "
        "`python scripts/validar_figuras.py --strict --allow-render-extraction`; "
        "promoção por `python scripts/validar_figuras.py --strict`.",
    ]
    return "\n".join(linhas)


def build_refs(m: dict) -> str:
    linhas = [SENT_R, "## Referências Bibliográficas", ""]
    for i, k in enumerate(m["refs"], 1):
        linhas.append(f"{i}. {BIB[k]}")
    linhas += ["", "Relação completa em `Artigos_Referências/Referências_Bibliográficas.xls`."]
    return "\n".join(linhas)


def is_sentinel(cell: dict, sent: str) -> bool:
    return cell.get("cell_type") == "markdown" and "".join(cell.get("source", [])).lstrip().startswith(sent)


def padronizar(nb_rel: str, m: dict, out_dir: Path | None) -> str:
    src_path = NBROOT / nb_rel
    d = json.loads(src_path.read_text(encoding="utf-8"))
    cells = d["cells"]
    # remove sentinelas antigas (idempotência) e o cabeçalho markdown legado (1a célula '# ...')
    cells = [c for c in cells if not is_sentinel(c, SENT_H) and not is_sentinel(c, SENT_R)]
    while cells and cells[0].get("cell_type") == "markdown" and "".join(cells[0].get("source", [])).lstrip().startswith("# "):
        cells.pop(0)
    for position, cell in enumerate(cells):
        if not cell.get("id"):
            cell["id"] = _stable_cell_id(
                nb_rel,
                position,
                cell.get("cell_type", ""),
                "".join(cell.get("source", [])),
            )
    cells.insert(
        0,
        md_cell(
            build_header(m, nb_rel),
            cell_id=_stable_cell_id(nb_rel, "nino26-header-v1"),
        ),
    )
    cells.append(
        md_cell(
            build_refs(m),
            cell_id=_stable_cell_id(nb_rel, "nino26-references-v1"),
        )
    )
    d["cells"] = cells
    metadata = d.setdefault("metadata", {})
    metadata["kernelspec"] = {
        "display_name": "Python 3.12 (.venv NINO26)",
        "language": "python",
        "name": "nino-brasil",
    }
    metadata.setdefault("language_info", {})["name"] = "python"
    metadata["language_info"]["version"] = "3.12"
    dest = (out_dir / nb_rel) if out_dir else src_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    return f"{m['codigo']:>3}  {nb_rel:44}  figs={len(m['figs'])} refs={len(m['refs'])}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--in-place", action="store_true", help="grava no próprio notebook")
    g.add_argument("--out-dir", type=Path, help="grava espelho para revisão")
    args = ap.parse_args()
    out = None if args.in_place else args.out_dir
    for nb_rel, m in NB.items():
        print(padronizar(nb_rel, m, out))
    print(f"\n{len(NB)} notebooks padronizados"
          + (f" em {out}" if out else " (in-place)")
          + ".  Valide compatibilidade e depois o gate semantico conforme docs/PADRAO_NOTEBOOKS.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
