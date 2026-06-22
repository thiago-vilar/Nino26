# Painel descritivo - Fisica informada, Newton e acoplamento oceano-atmosfera

Data da revisao: 2026-06-14

## Veredito executivo

Sim, faz sentido considerar a cadeia fisica proposta no projeto NINO-BRASIL:

`F = m.a -> Navier-Stokes -> equacoes primitivas -> balancos geostrofico/termodinamico/continuidade -> Walker, Hadley, Bjerknes, Kelvin/Rossby -> ENSO -> teleconexoes -> NEB e Sul`

O ponto cientifico principal e que a Segunda Lei de Newton nao precisa entrar como uma equacao isolada no modelo, mas como a raiz mecanica das equacoes de movimento usadas em dinamica atmosferica e oceanica. Para o projeto, a decisao mais forte e pragmatica e tratar essa fisica como informacao de primeira classe: variaveis, diagnosticos, restricoes, perdas auxiliares, testes de consistencia e ablations. Resolver diretamente Navier-Stokes/equacoes primitivas completas nao e necessario nesta etapa e provavelmente deslocaria a tese para a construcao de um modelo dinamico numerico completo.

## Implicacao para o NINO-BRASIL

O projeto deve explicitar a fisica de acoplamento oceano-atmosfera, porque previsao de chuvas extremas e secas no Brasil depende de mecanismos dinamicos e termodinamicos, nao apenas de correlacao estatistica. ENSO atua por anomalias de SST, termoclina, conteudo de calor oceanico, ventos alisios, pressao ao nivel do mar, conveccao tropical, celulas de Walker/Hadley e teleconexoes. Portanto, o ganho esperado de acuracia deve ser testado por comparacao quantitativa:

1. modelo estatistico apenas com indices climaticos;
2. modelo com campos fisicos oceanicos;
3. modelo com campos fisicos atmosfericos;
4. modelo acoplado oceano-atmosfera;
5. modelo acoplado com restricoes/regularizacao fisica.

## O que implementar primeiro

1. Manter o principio ja adotado nos pareceres: toda saida visual deve nascer de um produto numerico.
2. Fortalecer a tabela mensal de eventos ENOS com variaveis fisicas: SST/SSTA, OHC, D20/termoclina, gradiente leste-oeste, vento zonal, vento meridional, SLP, TCWV e fluxos de superficie.
3. Criar diagnosticos de acoplamento: correlacoes defasadas oceano-atmosfera, vento -> SST, SST -> pressao/conveccao, OHC -> pico ENOS, e teleconexoes ENOS -> precipitacao NEB/Sul.
4. Usar ablation studies para responder numericamente: quanto a fisica melhora RMSE, MAE, Brier score, F1 de eventos extremos e skill por lead time?
5. Deixar PINN/Neural ODE como fase posterior, depois que os baselines classicos comprovarem que as variaveis fisicas aumentam skill fora da amostra.

## Como traduzir a fisica em features e restricoes

| Camada fisica | Produto numerico recomendado | Uso no projeto |
|---|---|---|
| Momentum/rotacao | u/v wind, curl, divergencia, gradiente de pressao, indices de Walker | Diagnosticar transporte, convergencia e teleconexao |
| Termodinamica | SST, SSTA, OHC, fluxos de calor, TCWV | Explicar energia disponivel para conveccao e persistencia de ENOS |
| Oceano subsuperficial | D20, termoclina, slope longitudinal, conteudo de calor | Antecedencia fisica do pico de El Nino/La Nina |
| Continuidade/conservacao | checks de massa/energia e suavidade temporal | Regularizacao e controle de qualidade |
| Acoplamento | defasagens vento-SST-SLP-precipitacao | Medir causalidade operacional e teleconexao |
| Extremos regionais | chuva extrema/seca por NEB e Sul | Alvo final de impacto climatico |

## Leitura da literatura consultada

- SSTODE propoe Neural ODEs informadas por fisica para SST, com termos associados a transporte, difusao/adveccao e troca de energia: https://arxiv.org/abs/2511.05629
- ClimODE formula previsao meteorologica como dinamica continua informada por uma PDE de adveccao e conservacao: https://arxiv.org/abs/2404.10024
- Ola separa e acopla dinamicas de oceano e atmosfera em um modelo de ML de sistema terrestre, aprendendo ondas tropicais e ENSO: https://arxiv.org/abs/2406.08632
- PTSTnet reforca que previsao ENOS alem de 24 meses exige interacoes multiescala oceano-atmosfera e telecorrelacoes: https://arxiv.org/abs/2503.21211
- Zhou et al. mostram que previsoes ENOS hibridas, combinando modelos dinamicos e deep learning, superam abordagens isoladas: https://www.nature.com/articles/s41467-025-59173-8
- Yuan et al. tratam modos climaticos globais como dinamicas acopladas entre bacias oceanicas, relevante para uma tese que nao isola ENOS de teleconexoes: https://www.nature.com/articles/s42256-026-01245-5
- Kashinath et al. revisam physics-informed ML em clima e tempo, com ganhos de consistencia fisica, eficiencia de dados e generalizacao: https://royalsocietypublishing.org/rsta/article/379/2194/20200093/41210/Physics-informed-machine-learning-case-studies-for
- Ouarda/Pinheiro oferecem baseline regional para previsao sazonal de precipitacao no Nordeste do Brasil: https://www.nature.com/articles/s41598-023-47841-y
- Lieber et al. reforcam a relevancia de teleconexoes ENOS assimetricas para extremos historicos e futuros: https://journals.ametsoc.org/view/journals/clim/37/22/JCLI-D-23-0619.1.xml
- Chen et al. aplicam PINN/PILSTM a variaveis regionais-chave de ENOS, usando conhecimento fisico para interpretabilidade: https://www.researchsquare.com/article/rs-2624566/v1
- Tziperman et al. ajudam a enquadrar ENOS como sistema dinamico com componentes deterministicas, caoticas e estocasticas: https://arxiv.org/abs/1206.5657

## Status dos artigos em `papers`

| Status | Arquivo |
|---|---|
| Baixado | `JIANG_2025_SSTODE-Ocean-Atmosphere-Physics-Informed-Neural-ODEs-for-SST-Prediction.pdf` |
| Baixado | `VERMA_2024_ClimODE-Climate-and-Weather-Forecasting-with-Physics-Informed-Neural-ODEs.pdf` |
| Baixado | `WANG_2024_Ola-Coupled-Ocean-Atmosphere-Dynamics-in-a-Machine-Learning-Earth-System-Model.pdf` |
| Baixado | `HAO_2025_PTSTnet-Interpretable-Cross-Sphere-Multiscale-Deep-Learning-Predicts-ENSO.pdf` |
| Baixado | `ZHOU_2025_Combined-Dynamical-Deep-Learning-ENSO-Forecasts.pdf` |
| Suplementar baixado | `YUAN_2026_UniCM-Learning-Coupled-Dynamics-of-Global-Climate-Modes-Supplementary.pdf`; artigo principal Nature MI tem acesso pago |
| Bloqueado | `Kashinath_2021_Physics-Informed-ML-Case-Studies-for-Weather-and-Climate-Modelling.pdf`; PDF automatico retornou 403 na Royal Society/JSTOR |
| Baixado | `OUARDA_2023_Short-Lead-Seasonal-Precipitation-Forecast-NEB.pdf` |
| Baixado | `LIEBER_2024_Historical-Future-ENSO-Teleconnections-with-Extremes.pdf` |
| Baixado via Research Square | `CHEN_2023_Physics-Informed-Neural-Networks-for-ENSO-Key-Regional-Variables.pdf`; ResearchGate direto retornou 403 |
| Baixado, adicional | `TZIPERMAN_2012_ENSO-Dynamics-Low-Dimensional-Chaotic-or-Stochastic.pdf` |
| Baixado, adicional | `BEHROUZ_2026_Memory-Caching-RNNs-with-Growing-Memory.pdf`; artigo de ML geral encontrado previamente na pasta |

## Recomendacao final

Adotar uma trilha "fisica informada progressiva":

1. agora: diagnosticos fisicos numericos e ablation studies;
2. depois: modelos classicos com features de acoplamento oceano-atmosfera;
3. em seguida: redes profundas com perdas auxiliares de conservacao/adveccao;
4. apenas se necessario: Neural ODE/PINN como capitulo metodologico especifico.

Essa escolha preserva rigor fisico, aumenta interpretabilidade e evita transformar o projeto em um simulador climatico completo antes de provar, com dados, onde a fisica melhora a previsao de chuvas extremas e secas.
