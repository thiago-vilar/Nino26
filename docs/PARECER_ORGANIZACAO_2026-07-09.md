# Parecer de organizacao - NINO-BRASIL

**Data:** 2026-07-09  
**Objetivo:** desembaralhar a estrutura do projeto, fixar a diretriz ativa e
separar dado, metodo, fase e produto final.

## 1. Parecer executivo

O projeto esta tecnicamente recuperavel e ja tem uma base semanal forte para
seguir. O principal problema nao era falta de dado: era conflito de narrativa
entre documentos antigos e documentos novos. Alguns arquivos diziam que o
projeto parava na Fase 3 e que a Fase 4 estava pausada; outros ja traziam Fase
4 em execucao, Fase 5/6 e FaseWEB. Isso criava uma leitura confusa do estado
real.

A organizacao correta agora e:

| Bloco | Estado | Evidencia |
|---|---:|---|
| Fase 1 | concluida | base local com CHIRPS, OISST, ERA5, oceano UFS/GLORYS/GLO12, ORAS5 e validacao in situ |
| Fase 2 | concluida | `nino34_master_weekly.csv`, `phase2_master_audit.csv`, `phase2_master_validation.csv`, `phase2_ctd_validation.csv` |
| Fase 3 | concluida | notebooks 3A-3L, eventos EN/LN, quatro periodos por evento, duracoes, discriminantes, PCA por fase, Kelvin/Bjerknes/mapas e relatorio final |
| Fase 4 | em execucao | CHIRPS semanal, P90, anomalias de chuva, lags semanais, N_eff/FDR, sem ML/RN |
| Fase 5 | nao iniciada | Random Forest/XGBoost + XAI, apenas apos gate G1 |
| Fase 6 | nao iniciada | redes neurais nativas + XAI, apenas apos gate G2 |
| FaseWEB | esqueleto | publicacao/painel/operacao recorrente |

## 2. Ajustes feitos nesta organizacao

| Arquivo | Ajuste |
|---|---|
| `README.md` | passou a apontar `docs/DIRETRIZES_FASES.md` como fonte canonica, incluiu Fase 4 ativa, Fase 5/6 e FaseWEB, removeu a narrativa antiga de pausa e escopo obsoleto |
| `docs/ARQUITETURA.md` | reescrito para o fluxo ativo: Fase 1 -> Fase 2 -> Fase 3/4 -> gates -> Fase 5/6 -> FaseWEB |
| `notebooks/README.md` | reorganizado em Fase 2, Fase 3 e Fase 4, com estado e papel de cada pasta |
| `src/nino_brasil/project_phases.py` | lista estrutural de fases atualizada para Fases 1-6 + FaseWEB |
| `scripts/update_painel_executivo.py` | painel automatico passa a reportar Fase 2 pela validacao da matriz semanal e Fase 4 como execucao estatistica |
| `scripts/build_master_weekly.py` | cabecalho documenta tambem `phase2_master_validation.csv` e o modo `--ocean-only` |
| `docs/index.html` | fluxo visual simplificado e FaseWEB incluida |
| `docs/CRONOGRAMA.md` e `docs/DIRETRIZES_FASES.md` | esclarecido que sao 31 variaveis fisicas + `ocean_source_code` como metadado, nao 32 variaveis fisicas |

## 3. Conteudo numerico da matriz semanal

Arquivo: `data/processed/parquet/features/nino34_master_weekly.csv`  
Eixo temporal: semanal W-SUN.  
Linhas observadas no ultimo build: 2372 semanas.  
Colunas de conteudo: 31 variaveis fisicas + 1 metadado de fonte oceanica.

| Fonte | Variavel por escrito (abreviacao) | Serie temporal valida | Intervalo de coleta |
|---|---|---:|---|
| UFS/GLORYS/GLO12 + OISST | Anomalia de SST Nino 3.4 (`nino34_ssta`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Profundidade D20 (`d20_m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Inclinacao da termoclina (`tilt_m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Slope da inclinacao da termoclina (`tilt_slope`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Conteudo de calor oceanico 0-100 m (`ohc_0_100`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Conteudo de calor oceanico 0-300 m (`ohc_0_300`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Conteudo de calor oceanico 0-700 m (`ohc_0_700`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Conteudo de calor oceanico 300-700 m (`ohc_300_700`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Altura da superficie do mar (`ssh_m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Warm water volume (`wwv`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 50 m (`t50m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 100 m (`t100m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 150 m (`t150m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 200 m (`t200m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 300 m (`t300m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 500 m (`t500m`) | 1981-09-06 a 2026-06-14 | semanal |
| UFS/GLORYS/GLO12 | Temperatura a 700 m (`t700m`) | 1981-09-06 a 2026-06-14 | semanal |
| ERA5 | Anomalia de estresse zonal do vento (`tau_x_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do vento zonal a 10 m (`u10_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do vento meridional a 10 m (`v10_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia da pressao ao nivel medio do mar (`mslp_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do vapor d'agua total da coluna (`tcwv_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do fluxo de calor latente de superficie (`slhf_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do fluxo de calor sensivel de superficie (`sshf_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia da radiacao solar liquida de superficie (`ssr_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia da radiacao termica liquida de superficie (`str_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do vento zonal em 850 hPa (`u850_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia do vento zonal em 200 hPa (`u200_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia da velocidade vertical em 850 hPa (`omega850_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia da velocidade vertical em 500 hPa (`omega500_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| ERA5 | Anomalia da divergencia em 850 hPa (`div850_anom`) | 1981-01-04 a 2026-01-04 | semanal |
| Metadado | Codigo de fonte oceanica (`ocean_source_code`: 1=UFS, 2=GLORYS, 3=GLO12) | acompanha oceano | semanal |

## 4. Validacao e lacunas

| Checagem | Resultado |
|---|---:|
| indice monotonico crescente | True |
| semanas duplicadas | nenhuma |
| grade semanal W-SUN regular | True |
| eixo 1981 a 2026 | True |
| variavel totalmente vazia | nenhuma |
| transicoes de fonte oceanica ordenadas | True |

| Grupo | Cobertura | Maior lacuna |
|---|---:|---:|
| 17 variaveis oceanicas | 2337 semanas validas, 98.5% | 35 semanas |
| 14 variaveis ERA5 | 2349 semanas validas, 99.0% | 23 semanas |

Os dados oceanograficos foram validados com CTD/WOD, mas nao corrigidos por
CTD. A validacao e uma comparacao anual entre a termoclina dos perfis in situ e
o D20 da reanalise na caixa Nino 3.4. Exemplo dos primeiros anos:

| Ano | Perfis CTD | Termoclina CTD media (m) | D20 reanalise medio (m) | Diferenca (m) |
|---:|---:|---:|---:|---:|
| 1981 | 6 | 83.3 | 139.0 | -55.6 |
| 1982 | 18 | 145.3 | 141.3 | 3.9 |
| 1983 | 31 | 82.4 | 106.4 | -24.0 |
| 1984 | 24 | 82.5 | 111.6 | -29.1 |
| 1985 | 48 | 106.1 | 129.3 | -23.1 |
| 1986 | 43 | 141.5 | 133.9 | 7.6 |

Leitura: CTD e controle de plausibilidade e vies/ordem de grandeza; nao e fonte
continua suficiente para substituir UFS/GLORYS/GLO12.

## 5. Decisoes metodologicas importantes

| Pergunta | Parecer |
|---|---|
| Por que GLORYS nao foi simplesmente interpolado para 1981-presente? | Porque GLORYS12 nao cobre 1981-1992. A solucao honesta e declarar a emenda UFS -> GLORYS -> GLO12 e auditar transicoes. Interpolar com outra fonte esconderia diferenca de vies, resolucao e produto. |
| ORAS5 mensal pode completar a serie semanal? | Nao. ORAS5 mensal serve para comparacao/calibracao de memoria oceanica; nao deve virar observacao semanal/diaria por interpolacao. |
| O que e anomalia sigma? | Em oceanografia, costuma ser anomalia de densidade potencial (`sigma-theta`/`sigma0`) relativa a uma referencia. Nao faz parte da matriz semanal ativa atual. |
| O que e anomalia theta? | Em geral, anomalia de temperatura potencial (`theta`) em relacao a uma climatologia. A matriz ativa usa temperaturas por profundidade e suas estatisticas derivadas, nao uma coluna `theta_anom` separada. |
| Quais variaveis nao sao variaveis fisicas da matriz? | `ocean_source_code` e metadado de procedencia; nao deve entrar como preditor fisico. |
| Dados mensais importam? | Importam para comparacao/calibracao e coerencia fisica. Nao sustentam a estatistica principal, que e semanal/diaria. |

## 6. Riscos restantes

| Risco | Severidade | Acao recomendada |
|---|---:|---|
| Worktree Git tem muitas alteracoes preexistentes e arquivos gerados | media | `git add .` e outros comandos praticos sao permitidos; revisar o diff antes do commit continua recomendado |
| Relatorios historicos ainda podem conter linguagem antiga | media | manter `DIRETRIZES_FASES.md` e este parecer como fonte atual; nao apagar legado sem decisao |
| Fase 3 precisa congelar artefatos oficiais | media | usar `docs/FASE3_PENDENCIAS.md` como checklist de fechamento e evitar misturar saidas antigas |
| CTD tem cobertura irregular | media | usar como validacao in situ, sempre com `n_perfis_ctd_nino34` e diferenca anual |
| Fase 4 precisa controlar autocorrelacao espacial/temporal | alta | aplicar N_eff, FDR e lags semanais antes de qualquer narrativa regional |

## 7. Proximo passo recomendado

1. Rodar `scripts/update_painel_executivo.py` para regenerar o painel local com
   a nova organizacao.
2. Revisar `git diff` por bloco: documentacao, scripts de Fase 2/painel,
   notebooks gerados.
3. Fazer commits pequenos, evitando misturar dados/notebooks pesados com codigo.
4. Fechar a Fase 3 em um relatorio unico e entao congelar os insumos que a Fase
   4 deve consumir.
