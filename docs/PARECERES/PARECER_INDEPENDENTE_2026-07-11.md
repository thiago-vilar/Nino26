# Parecer científico independente — NINO-BRASIL

**Autor do parecer:** revisão técnica independente
**Data:** 2026-07-11
**Solicitante:** Thiago Vilar (UFPE — Oceanografia Física)
**Escopo:** releitura integral do repositório `C:\DEV\NINO26` (docs, `src/`, `scripts/`, `notebooks/`, artefatos numéricos em disco), com juízo sobre padrões, métricas e coerência científica em quatro eixos — oceanografia, estatística, machine learning e redes neurais — e confronto com a análise anterior fornecida pelo pesquisador.

**Método:** não me limitei a reler a análise anterior. Reproduzi numericamente, no próprio ambiente, as alegações centrais (estatísticas do alvo CHIRPS, F1 dos classificadores, alvo de duração, skill regional, contagem de eventos, integridade do construtor da Fase 2, estado das Fases 7–8). Onde os números batem, digo que batem; onde há nuance entre "código atual" e "artefato em disco", explicito a diferença.

---

## 1. Veredito executivo

O NINO-BRASIL é uma infraestrutura de pesquisa **forte e honesta**: física de recarga bem escolhida (Jin 1997; Meinen & McPhaden 2000), base de dados auditável, cultura de gates e reporte de resultados negativos sem maquiagem. Isso é acima do padrão acadêmico usual.

Ao mesmo tempo, **confirmo integralmente os problemas determinantes** da análise anterior. Reproduzi os números: eles são reais, não retóricos. O estado atual do repositório mistura três coisas que precisam ser separadas antes de qualquer conclusão científica sobre o Brasil:

1. **Artefatos produzidos por versões diferentes do código.** As correções recentes (`fase4_utils.py`, `phase5/6`, `climatology.py`, `io_utils.py`) existem na árvore de trabalho, mas **não foram propagadas para os arquivos em disco**. Os parquets/CSVs auditados ainda são da geração antiga.
2. **Alvo CHIRPS numericamente corrompido** (confirmado abaixo com os mesmos números da análise anterior).
3. **Fases 5–8 sem ganho científico demonstrado**: ML perde para persistência; redes ainda sintéticas.

A decisão cientificamente correta é a que a análise anterior recomenda e que subscrevo: **reabrir formalmente as Fases 2 e 4, congelar os gates 1–5, reconstruir o alvo de precipitação e a harmonização oceanográfica, e só então rejulgar as demais fases em condições comparáveis.** A Fase 3 pode, após exportar as tabelas de auditoria, ser promovida de diagnóstico exploratório a caracterização robusta.

**Nota importante de justiça:** parte das correções P0/P1 já foi **codificada** (ver `docs/PARECERES/CORRECOES_APLICADAS_2026-07-10.md`). O que falta é **reexecutar** e **regenerar os artefatos** — e, em pontos específicos abaixo, ainda há falhas de raiz não resolvidas mesmo no código corrigido.

---

## 2. Verificação independente das alegações (o que eu mesmo reproduzi)

| # | Alegação da análise anterior | Meu resultado independente | Veredito |
|---|---|---|---|
| 1 | Alvo CHIRPS z corrompido: min −1.710.469,75; max 27.115.382; média 1.666; σ 84.688; 1.623 pixels com σ>100k; 35,7% com \|z\|max>10 | Reproduzido **exatamente**: min −1.710.469,75; max 27.115.382,0; média 1.666,41; σ 84.688,2; 1.623 pixels σ>100k; 35,74% \|z\|max>10; matriz 2366×19573 | **Confirmado** |
| 2 | Faltantes convertidos em zero | Fração de NaN no parquet = **0,0** (impossível para anomalia real de chuva zero-inflada) | **Confirmado** |
| 3 | `build_master_weekly.py` termina abruptamente com `i` isolado em `ctd_validation`, sem `main` | Arquivo tem 297 linhas, termina literalmente em `        i`; há `def build()` (l. 221) mas **nenhum** `if __name__=="__main__"` nem `def main` | **Confirmado** |
| 4 | `data/processed/numeric-tables/` vazio; 67 figuras | Diretório com **0 arquivos**; **67** PNGs | **Confirmado** |
| 5 | RF/XGBoost perdem para persistência (persistência 0,944) | Em disco: F1 RF **0,477**, XGB **0,512**, **sem** coluna de baseline. A reavaliação corrigida (com folds por evento e baselines) rebaixa para RF 0,392 / XGB 0,464 e mede persistência 0,944 | **Confirmado** (com nuance §5) |
| 6 | Alvo de duração inflado ~3× (×13) | `phase5_alvos_por_evento.csv` grava `Y_duracao_sem=143` para el_nino_1982/83 (11 estações ×13); correto ≈ 47,7 sem. O `Y_tempo_para_pico=26,3` (por datas) prova a inconsistência interna | **Confirmado** (bug no artefato; corrigido só no código) |
| 7 | Fase 6: 0/13 unidades superam baseline; herda alvo inválido | `phase6_skill_rf.csv`: **13** unidades, todas com `skill_rmse_vs_baseline<0`; RMSE em dezenas de milhares (herda o alvo corrompido) | **Confirmado** |
| 8 | Fases 7–8 apenas com arrays aleatórios; TF não declarado | `7_ciclo_convlstm.ipynb`: `np.random.rand(120,8,24,3)`; `8_brasil_convlstm.ipynb`: `np.random.rand(80,24,8,24,3)` e `(120,40,40)`; nenhum import de TF/torch/lightning | **Confirmado** |
| 9 | Proxy de tensão zonal usa \|u\|u e ignora v | `build_master_weekly.py` l. 238: `RHO_AIR*CD_NEUTRAL*atmo["u10"].abs()*atmo["u10"]`; usa ERA5 real só se `tau_x_anom_nino34_pa` existir | **Confirmado** (com fallback) |
| 10 | 2 testes reprovam (nome legado 4D; corte 1993-2009/2010+) | O notebook 4D ainda contém os tokens `1993-2009` e `2010+`, que o `test_phase4_figure_contract` proíbe | **Confirmado** |
| 11 | ~300 features, ~23 eventos, mesmos eventos cruzam treino/teste | 23 eventos em `phase5_alvos_por_evento.csv`; master 2375 semanas × 33 colunas | **Confirmado** (nº de eventos exato) |

**Conclusão da seção:** a análise anterior é factualmente correta nos pontos que verifiquei. Não encontrei nenhuma alegação central refutada pelos dados.

---

## 3. Coerência oceanográfica

**O que está certo.** A espinha física é a correta para ENSO: o par SST↔recarga (D20/WWV/OHC/SSH) é exatamente o oscilador de recarga-descarga de Jin (1997) e a memória de volume de água quente de Meinen & McPhaden (2000), em que o WWV/OHC lidera o SSTA por 2–3 estações. Os lags diagnosticados na Fase 3 (D20 ~15 semanas, WWV ~20 semanas antecedendo o pico) são fisicamente plausíveis e consistentes com essa literatura. O critério de evento local (ONI local ≥ ±0,5 °C por ≥5 estações móveis) reproduz a definição operacional NOAA/ONI e a cronologia de Trenberth (1997), e coincide com os eventos canônicos (82/83, 97/98, 15/16, 23/24). A emenda observacional UFS(1981–92)→GLORYS12(1993+)→GLO12 é defensável **desde que** a janela real por fonte seja sempre declarada.

**O que precisa de correção (oceanografia).**

- **Fluxos ERA5 acumulados (slhf, sshf, ssr, str).** Em J m⁻², precisam ser tratados como acumulação horária e convertidos ao período em segundos para virar W m⁻² (documentação ECMWF). Amostrar 00/06/12/18 UTC e tirar média **não** representa energia diária nem fluxo médio. Como o construtor da Fase 2 está truncado, não pude reconstituir a agregação exata em disco — mas o risco de unidade permanece e deve ser auditado explicitamente.
- **Tensão do vento.** O proxy ρCd|u|u (confirmado no código) ignora a componente meridional e é insensível a rajadas de oeste (WWB) fora do eixo zonal. Para estudos de Kelvin/WWB o correto é a tensão fornecida pelo ERA5 (iews/inss) ou, no mínimo, ρCd√(u²+v²)·u **ponto a ponto** e, de preferência, no Pacífico oeste/Niño-4 — não depois da média espacial em Niño-3.4.
- **Variáveis de subsuperfície cruas.** Na escala semanal, a variância dominante de D20/T(z)/OHC é o **ciclo anual**, não o ENSO (a própria correção mede: amplitude sazonal de D20 ~25 m vs σ total ~15 m). O novo `climatology.harmonic_anomaly_matrix` (anomalia harmônica 1991–2020 + detrend, ajuste só na base, sem vazamento) resolve isso **corretamente** — mas ainda não regenerou os artefatos 4C/4D em disco.
- **Validação in situ.** CTD/WOD, TAO/TRITON e Argo devem validar D20/OHC colocalizados por perfil, data e pixel — hoje as lacunas estão preservadas (bom), mas a validação colocalizada ponto-a-ponto ainda é uma pendência.

---

## 4. Coerência estatística

**O que está acima do padrão.** A Fase 3D/4C tem a melhor máquina estatística do projeto: N_eff de Bretherton et al. (1999) por par, FDR de Benjamini-Hochberg (1995) global, AR(1) só em semanas consecutivas do mesmo evento, IC95 por unidade, LOO por evento e bootstrap em blocos. Isso é a prática que Wilks (2016) exige para mapas com milhares de testes (significância local ≠ significância de campo).

**Onde a inferência ainda não fecha.**

- **Seleção pós-hoc.** Escolher o melhor lag e depois calcular IC/bootstrap **nos mesmos dados** com o lag vencedor fixo infla a confiança. A seleção de lag/variável deve ser repetida **dentro** de cada reamostragem/fold.
- **Unidade de incerteza errada.** Semanas autocorrelacionadas são tratadas como réplicas em Kruskal/ε²/PCA; eventos longos pesam mais. A unidade primária de incerteza deve ser o **evento ENSO** — bootstrap por evento, blocos móveis dentro do evento, permutação de rótulos por evento/estação.
- **Confundidor de calendário no gate.** O resultado mais surpreendente (NEB "úmido" em El Niño, contrariando a literatura de seca) aparece justamente em lag longo (34 sem) com preditor cru sazonal — é o cenário clássico em que o phase-locking do ENSO e a estação chuvosa produzem correlação de calendário disfarçada de teleconexão. **Antes de aceitar qualquer conclusão que contrarie Grimm & Tedeschi (2009) / Cai et al. (2020), é obrigatório refazer 4C/4D com anomalia + detrend** (já codificado) e estratificar por estação chuvosa (FMAM no NEB).
- **PCA de índices escalares** não é EOF espacial; a nomenclatura deve ser corrigida.
- **Corte fixo 1993-2009/2010+** ainda ativo no 4D — é breakpoint arbitrário e viola a própria diretriz nº 7; uma ruptura estrutural exigiria teste de ponto de mudança pré-especificado.

---

## 5. Coerência de machine learning

**Estado real das métricas.** Aqui está a nuance mais importante do parecer, e ela **reforça** a análise anterior:

- **Em disco** (geração antiga, sem baseline): F1-macro RF 0,477, XGB 0,512 — parecem "acima do acaso" (0,25), mas **sem comparador não significam nada**.
- **Reavaliação corrigida** (folds por evento, pré-processamento dentro do fold, baselines): RF 0,392, XGB 0,464, baseline semana-do-ano 0,476, **persistência de 1 semana 0,944**.

A persistência dominar com 0,944 **não é acaso e é fisicamente esperado**: a fase do ENSO quase não muda de uma semana para a outra; "repita a fase da semana passada" é quase perfeito. Ou seja, **a tarefa de classificar 4 fases, como está posta, é trivial para persistência e o ML não agrega** — **G2 reprova inequivocamente**.

**Problemas de desenho (todos confirmados ou plausíveis):**

- **Vazamento hierárquico:** ~300 features, ~1.620 semanas rotuladas, **apenas 23 eventos**; os mesmos eventos cruzam treino/teste nos folds semanais. Com dependência temporal e hierárquica, a validação tem de **bloquear por evento** (Roberts et al. 2017).
- **Rótulos com informação futura:** gênese e pico são definidos post hoc (o plateau e as 26 semanas pré-onset só são conhecidos depois do evento). Logo o F1 mede **caracterização**, não previsão — a linguagem das figuras deve dizer isso.
- **Classificador só de fase:** exclui semanas neutras e não distingue El Niño de La Niña — classifica a fase **dado** que já se sabe estar em evento.
- **Vazamento de pré-processamento:** climatologia oceânica ajustada antes do `TimeSeriesSplit`; RFECV não aninhado. Seleção e hiperparâmetros têm de ocorrer **integralmente dentro** do loop de treino (Cawley & Talbot 2010).
- **Data augmentation:** o jitter atual replica dados, **não** cria eventos independentes — não resolve o gargalo de 23 eventos (ver resposta à Fase 5 em §7).
- **SHAP calculado mas não salvo; PDP multiclasse perde a classe** e não demonstra limiares de Bjerknes.

**Fase 6.** Parser de `el_nino_pico` quebra `el`/`nino_pico` e elimina silenciosamente EN/LN; 0/13 unidades vencem o baseline; RMSE em dezenas de milhares porque **herda o alvo CHIRPS inválido**. O CSV, embora agora íntegro (escrita atômica aplicada), ainda traz só `baseline=media expansiva` — a geração em disco é **anterior** aos baselines de persistência/climatologia prometidos.

---

## 6. Coerência de redes neurais

**Estado real:** as Fases 7 e 8 **não foram treinadas**. Os notebooks geram cubos `np.random.rand` "para demonstrar a API de sequências" e imprimem shapes. Não há TensorFlow/PyTorch/Lightning importado. **G4 e G5 não são avaliáveis.**

**Problemas de arquitetura já identificáveis (subscrevo):**

- Fase 7: longitudes padrão 120…290 enquanto o Zarr real usa −170…−30 (retornaria zero longitudes); treino/validação compartilhariam semanas na fronteira das sequências; 4 classes não cobrem EN/LN × 4 fases + neutro; "oclusão por acurácia" não recebe rótulos.
- Fase 8: alinhamento temporal ignora `seq_len−1` (chuva anterior pareada com sequência posterior do Pacífico); o "encoder-decoder" é ConvLSTM→Flatten→Dense→reshape, que na grade real explodiria para ~20 bilhões de parâmetros (~76 GiB em FP32) — **inviável**; sem máscara Brasil, pesos de área, loss probabilística ou tratamento de extremos.

ConvLSTM (Shi et al. 2015) é apropriada para o problema espaço-temporal, mas pressupõe amostra suficiente e validação coerente. Em ENSO, a saída para a amostra observacional pequena é o pré-treino em simulações climáticas (CMIP/NMME) e transferência para reanálises, como em Ham et al. (2019) — caminho que o projeto deveria adotar antes de qualquer rede "do zero".

---

## 7. Respostas às perguntas específicas

### Fase 3 — Há sequência de preditores por fase do ciclo? O que caracteriza gênese, crescimento, pico e decaimento? Que combinações lógicas/estatísticas determinam cada período?

**Sim, há uma sequência física e ela é a base do valor do projeto.** No arcabouço de recarga-descarga, a ordem de precedência esperada (e parcialmente diagnosticada na Fase 3) é:

- **Gênese (pré-condicionamento, ~26 sem antes do onset):** recarga de calor equatorial — **WWV/OHC(0–300) e D20 acima do normal** com o SSTA ainda neutro. É a assinatura precursora: o oceano "carrega" antes de a superfície responder. Estatisticamente, é onde WWV/OHC lideram o SSTA com o maior lag (WWV ~20 sem, D20 ~15 sem).
- **Crescimento/acoplamento:** ativação de Bjerknes — **anomalia de tensão zonal (WWB), aprofundamento do D20 no leste, u850/omega** acompanham o SSTA em fase (lag curto, ~0–8 sem). Aqui preditor e estado ficam quase contemporâneos.
- **Pico (faixa):** SSTA máximo com **WWV já em queda** (descarga começa antes do pico de SST) — a divergência de sinal entre OHC↓ e SSTA↑ é o marcador estatístico do topo.
- **Decaimento:** descarga concluída — **WWV/OHC abaixo do normal**, D20 rasa no oeste, frequentemente antecipando a transição para La Niña.

**Combinação lógica recomendada para rotular cada período** (a ser validada prospectivamente, não post hoc): usar o **sinal e a defasagem relativa entre OHC/WWV e SSTA** como discriminante — recarga com SSTA neutro = gênese; recarga + WWB + SSTA subindo = crescimento; SSTA no plateau com OHC caindo = pico; OHC negativo com SSTA caindo = decaimento. Formalmente: um classificador **causal** (só dados até t) sobre {SSTA, dSSTA/dt, OHC0-300, dOHC/dt, WWV, tau_x} tende a separar as fases — mas hoje os rótulos usam informação futura, então a Fase 3 **caracteriza** bem a sequência sem ainda **prevê-la**. A promoção a robusta depende de: (i) uma única definição de pico; (ii) chamar a gênese de pré-condicionamento; (iii) evento como unidade; (iv) walk-forward prospectivo com barreira de primavera.

### Fase 4 — Por que as conclusões sobre chuva/seca/extremos não são válidas? Por que o alvo CHIRPS está corrompido e o gate 4D tem falhas inferenciais?

**Por que o alvo está corrompido (reproduzido por mim):** o parquet tem min −1,7 milhão e max 27 milhões, média 1.666 e σ 84.688 — impossível para um z-score interpretável. A causa é dupla: (1) faltantes viram zero (0% de NaN na saída), o que injeta zeros de chuva onde não há dado; (2) o denominador de padronização pode colapsar — mesmo no código refeito, σ é modelado por harmônicos do resíduo absoluto e **ainda tem piso rígido de 1e-6 mm**, absurdamente pequeno; em pixel árido/estação seca (Caatinga/sertão), E|x|→0, o piso 1e-6 assume e o z explode. Some-se a reamostragem **por store** com `min_count=4` (aceita semanas de 4 dias na virada de ano) e a média de duplicatas — a agregação viola a natureza positiva, assimétrica e zero-inflada do CHIRPS (Funk et al. 2015).

**Por que as conclusões não valem:** o gate 4D (i) clusteriza 19.573 pixels da caixa, dos quais só ~11.375 são Brasil — usa ~8.198 pixels externos; (ii) não aplica máscara EN/LN na correlação final; (iii) usa p<0,10 **sem** FDR na etapa do gate; (iv) mantém o corte arbitrário 1993-2009/2010+ (confirmado nos tokens do notebook); (v) avalia várias variáveis no lag ótimo de **outra** variável (teste de palha); (vi) interpreta a **média** da anomalia semanal como "seca" ou "extremo". Uma anomalia média negativa **não é** seca; positiva **não é** chuva extrema. O sinal ENSO costuma ser mais nítido na **frequência de extremos** do que no total médio (Grimm & Tedeschi 2009). A hipótese tem de ser testada com SPI/SPEI (1/3/6 meses; Vicente-Serrano et al. 2010), CDD/CWD, R95p/R99p, Rx1day/Rx5day e contagem de spells — não com a média da anomalia z.

### Fase 5 — Por que RF/XGBoost perdem para persistência e G2 não passa? Data augmentation permitiria aferir a evolução com algum percentual de certeza?

**Por que perdem:** porque a tarefa, como posta, é dominada pela persistência (0,944) — a fase do ENSO é altamente persistente semana a semana. Um modelo que "repete a semana anterior" quase não erra; qualquer classificador que ignore a persistência parte de uma barra baixíssima e ainda assim não a alcança (RF 0,392 / XGB 0,464 < 0,944). Somado ao vazamento por evento e à seleção pós-hoc, **G2 reprova**.

**Sobre data augmentation:** parcialmente sim, mas **jitter/replicação não resolve** — replicar semanas não cria eventos independentes, e o gargalo real são os **23 eventos**, não as 1.620 semanas. As augmentations que *de fato* ajudam ENSO são de outra natureza:

1. **Pré-treino em simulações climáticas (CMIP6/NMME) e transfer learning para reanálises** — a estratégia de Ham et al. (2019), que contorna a amostra observacional curta com milhares de eventos simulados.
2. **Ensembles de reanálise/condições iniciais** (ORAS5, GLORYS, SODA) como realizações quase independentes.
3. **Aumento físico coerente** (perturbação de fase/amplitude preservando a dinâmica de recarga), não ruído gaussiano.

Com isso, **sim, é plausível** aferir a evolução EN/LN com incerteza quantificada — mas a métrica de "certeza" tem de ser probabilística (Brier/RPS/CRPS) e comparada a persistência e climatologia como baselines obrigatórios (WMO), e a validação tem de bloquear por evento. Antes disso, qualquer "% de certeza" é ilusório.

### Fase 6 — Por que nenhum dos 13 modelos regionais supera o baseline e G3 não passa?

Porque (i) **herdam o alvo CHIRPS inválido** — RMSE em dezenas de milhares, sem sentido físico; (ii) o parser elimina as condições EN/LN, então o "sinal" testado é degradado; (iii) se a própria triagem estatística da Fase 4 mal encontra r≈0,16 no agregado anual, não é surpresa que o ML não ache sinal onde ele foi diluído. **G3 reprova** — mas o veredito só será científico depois de refazer o alvo (SPI/SPEI/ETCCDI), aplicar máscara oficial do Brasil (interseção por área), estratificar por estação chuvosa e usar persistência/climatologia como baselines na **mesma** amostra.

### Fases 7–8 — protótipos com dados aleatórios

Confirmado: são esqueletos com `np.random`. **Não houve treinamento neural científico; G4 e G5 não foram avaliados.** Recomendação: só abrir a Fase 7 depois de o ML clássico (Fase 5) vencer baselines, e a Fase 8 depois de uma Fase 6 **válida e positiva**. A Fase 8 precisa de decoder convolucional/EOF de baixa dimensão (não Dense gigante), alinhamento por timestamps, máscara e pesos de área, loss probabilística e métricas por pixel/região/bioma.

---

## 8. Confronto com a análise anterior

**Concordância:** subscrevo o diagnóstico e o veredito. Verifiquei e reproduzi as alegações centrais (§2); nenhuma foi refutada. O plano P0–P3 é o correto.

**Nuances/complementos que acrescento:**

1. **"Código atual" vs "artefato em disco".** Vários números da análise (F1 0,392/0,464) vêm da **reavaliação em memória do código já corrigido**; os artefatos em disco ainda mostram a geração antiga (0,477/0,512, sem baseline; duração ×13; CHIRPS corrompido). Ambos são verdadeiros — de versões diferentes. O ponto operacional é que **as correções não foram propagadas**: falta reexecutar e regenerar `numeric-tables/`, parquets e CSVs.
2. **A correção do alvo CHIRPS ainda é incompleta.** Mesmo o `harmonic_standardized_anomaly` refeito (E|x|·√(π/2), mais robusto que variância harmônica) mantém o **piso 1e-6** e o `nan_to_num` no ajuste — a raiz da explosão (denominador→0 em pixel seco) **persiste estruturalmente**. A solução não é só refatorar o z: é adotar transformação adequada à chuva (gama/SPI, MAD/quantis empíricos com piso físico, cobertura de 7 dias reais).
3. **"Lieber et al. 2024"** não foi localizado nas bases; para o ponto de assimetria das teleconexões e diversidade de eventos, uso **Cai et al. (2020, Nat. Rev. Earth Environ.)** e **Timmermann et al. (2018, Nature)**, que são autoridade estabelecida.
4. **Emenda UFS–GLORYS:** mantenho e justifico — é defensável como ponte histórica, **desde que** harmonizada por fonte no overlap (1993–1995) e com a janela real sempre declarada; não deve ser vendida como cobertura homogênea desde 1981.

---

## 9. Matriz fase a fase

| Fase | Metodologia — o que faz | Perguntas que responde (e por quê) | Status real de execução | O que melhorar |
|---|---|---|---|---|
| **1 — Ingestão** | Baixa e cataloga OISST, CHIRPS, ERA5, UFS, GLORYS/GLO12, ORAS5, CTD/WOD/TAO/Argo, IBGE, com ledger de auditoria. | Há dados suficientes, rastreáveis e com cobertura conhecida? Sem procedência não há inferência auditável. | **Concluída com ressalvas.** Base local ampla; lacunas preservadas (não imputadas). | Auditar unidades/acumulados ERA5; validar checksums; completar TAO/Argo; declarar incerteza do sistema observacional. |
| **2 — Padronização/master** | Zarr, grade 0,25°, matriz semanal W-SUN 1981–2026 (2375×33): 17 oceânicas + 14 ERA5 + `ocean_source_code`. | É possível alinhar atmosfera, superfície e subsuperfície no mesmo eixo temporal? Pré-condição de lag/correlação/modelo. | **Artefatos existem, mas a fase não é reproduzível:** `build_master_weekly.py` truncado (sem `main`); emenda/unidades não corrigidas; `numeric-tables/` vazio. | Restaurar o entrypoint; teste end-to-end real do comando; harmonização por fonte; fluxos ERA5 em W m⁻²; tensão vetorial; manifesto/hashes; reconstruir `numeric-tables`. |
| **3 — Estatística do ciclo** | Catálogo EN/LN, 4 fases, Hovmöller, D20/OHC/WWV/SSH, lags, N_eff, FDR, bootstrap, LOO, PCA. | Como o ENSO nasce, cresce, atinge o pico e decai? Que variáveis antecedem o pico? Testa recarga e separa precursor de estado. | **Substancial, porém exploratória.** D20~15 sem, WWV~20 sem plausíveis; nested LOO acima do padrão. Auditável só após exportar `numeric-tables`. | Uma definição de pico; gênese = pré-condicionamento; anomalia por fonte; evento como unidade; repetir seleção de lag no bootstrap; EOF espacial real; walk-forward prospectivo. |
| **4 — Estatística Brasil** | Anomalia CHIRPS por pixel, agregação por região/bioma, lags, clusterização, gate de hipótese NEB-seco/Sul-úmido. | Onde, quando e com que sinal o Pacífico altera chuva/seca/extremos? Teleconexão é sazonal, regional e heterogênea. | **Inválida no estado atual. G1 não passa.** Alvo z corrompido (verificado); pixels externos; sem FDR/máscara no gate; corte fixo 1993/2010 ativo. | Refazer CHIRPS (concatenar diário, 7 dias válidos, transformação robusta/SPI-SPEI); máscara oficial; ETCCDI; fases na fonte t−lag; FDR/field significance; bootstrap por evento; estações chuvosas. |
| **5 — ML do ciclo** | RF/XGBoost em lags 4–52; classifica 4 fases; pretende prever pico/tempo-ao-pico/duração; PDP/SHAP. | O ML acrescenta skill fora-da-amostra além de estatística, climatologia e persistência? Só se justifica com ganho real. | **Piloto executado; G2 reprova.** RF 0,392 / XGB 0,464 vs persistência 0,944; bug de duração ×13 no artefato; correções não regeneraram saídas. | Alvo prospectivo (inclui neutro e EN/LN); folds purgados por evento; preprocessing dentro do fold; baselines obrigatórios; IC por evento; salvar SHAP; não ler PDP como causal; augmentation via CMIP/NMME (Ham 2019). |
| **6 — ML Brasil** | RF/XGBoost por região/bioma/fase com preditores defasados do Pacífico. | O ML prevê chuva regional melhor que climatologia, persistência e a Fase 4? Mede valor da não linearidade. | **G3 reprova.** Parser elimina EN/LN; 0/13 unidades vencem baseline; alvo CHIRPS inválido herdado; RMSE em dezenas de milhares. | Corrigir parser e baseline na mesma amostra; alvo reconstruído; validação sazonal/evento; 4 fases; pixel e região; probabilidades/extremos; SHAP fora-da-amostra. |
| **7 — ConvLSTM do ciclo** | Sequências espaciais multicanais do Pacífico para tipo/fase e evolução. | Campos espaciais acrescentam skill sobre índices escalares e ML tabular? ENSO é propagação espaço-temporal. | **Não iniciada cientificamente.** Só arrays `np.random`; sem TF/torch; G4 não avaliado. | Corrigir longitudes/dependências; data loader lazy; normalização no treino; alvo multitarefa; embargo temporal; métricas por lead; pré-treino CMIP/NMME + fine-tuning. |
| **8 — ConvLSTM Brasil** | Mapear sequência do Pacífico em campo futuro de chuva do Brasil. | A rede aprende a distribuição espaço-temporal com skill superior às Fases 4/6? Teleconexão é mapeamento espaço-espaço condicionado ao tempo. | **Não iniciada; arquitetura inviável.** Alinhamento temporal errado; Dense gigante (~20 bi params); G5 não avaliado. | Decoder convolucional/EOF de baixa dimensão; alinhamento por timestamps; máscara e pesos de área; loss probabilística/extremos; ensemble; métricas por pixel/região/bioma. |
| **WEB** | Painel, publicação, recalibração operacional. | Como comunicar resultados e manter recalibração auditável? Evita perda de procedência. | **Esqueleto; documentação divergente** (README/mermaid ainda usam o mapa antigo de fases). | Painel do manifesto; status derivado automaticamente; publicar só gates formalmente aprovados. |

---

## 10. Plano de correção recomendado

**P0 — bloquear conclusões inválidas.** Marcar Fases 4–8 como exploratórias/inválidas; restaurar o construtor completo da Fase 2 e adicionar teste real do comando de build; unificar documentação, código e status; validar mapeamentos antes de remover `numeric-tables`; regenerar `numeric-tables`.

**P1 — reconstruir o dado científico.** Concatenar CHIRPS **diário** entre anos antes da soma semanal; exigir 7 dias válidos (ou registrar cobertura); transformação robusta (climatologia sazonal + MAD/quantis, ou SPI/SPEI conforme o alvo) com **piso físico**, não 1e-6; limitar pixels à geometria oficial do Brasil (interseção por área); harmonizar UFS/GLORYS por fonte e overlap; corrigir unidades ERA5 e tensão do vento; validação CTD–D20 colocalizada.

**P2 — refazer a inferência.** Pré-registrar estimandos, famílias FDR e horizontes; separar estado contemporâneo de precursor verdadeiro; usar eventos como unidades; repetir seleção de lag/variável dentro de cada bootstrap/fold; estratificar NEB por FMAM e Sul por sua estação de maior teleconexão; controlar sensibilidade a tipos de ENSO e tendência (se mantiver só o Pacífico, declarar que estima **associação marginal**, não causalidade).

**P3 — só então retomar ML e redes.** Definir origem de previsão t, lead h e dados permitidos até t; comparar todos os métodos nos mesmos eventos/folds/alvos/horizontes; climatologia e persistência como baselines obrigatórios (WMO); abrir a Fase 7 só se o ML clássico vencer baselines; abrir a Fase 8 só após uma Fase 6 válida e positiva.

---

## 11. Referências

A relação completa (autor, ano, título, revista, indexador e link DOI) está em `Artigos_Referências/Referências_Bibliográficas.xls`. Núcleo:

- **ENSO/oceanografia:** Jin (1997, *J. Atmos. Sci.*); Meinen & McPhaden (2000, *J. Climate*); Trenberth (1997, *BAMS*); Timmermann et al. (2018, *Nature*); Huang et al. (2021, OISST v2.1, *J. Climate*); Hersbach et al. (2020, ERA5, *QJRMS*).
- **Brasil/precipitação/extremos:** Grimm & Tedeschi (2009, *J. Climate*); Cai et al. (2020, *Nat. Rev. Earth Environ.*); Funk et al. (2015, CHIRPS, *Scientific Data*); Vicente-Serrano et al. (2010, SPEI, *J. Climate*); Pinheiro & Ouarda (2023, *Scientific Reports*).
- **Estatística/ML/redes:** Benjamini & Hochberg (1995, *JRSS-B*); Wilks (2016, *BAMS*); Bretherton et al. (1999, *J. Climate*); Roberts et al. (2017, *Ecography*); Cawley & Talbot (2010, *JMLR*); Shi et al. (2015, ConvLSTM, *NeurIPS*); Ham et al. (2019, *Nature*).

---

*Parecer emitido em 2026-07-11 com verificação numérica independente no repositório HEAD. Onde afirmo "confirmado", o número foi reproduzido no ambiente; onde afirmo "código vs artefato", a diferença é entre a árvore de trabalho corrigida e os arquivos ainda não regenerados em disco.*
