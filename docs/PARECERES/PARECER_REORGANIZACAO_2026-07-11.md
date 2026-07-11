# Parecer de reorganização minimalista — NINO-BRASIL

**Autor do parecer:** revisão técnica independente
**Data:** 2026-07-11
**Solicitante:** Thiago Vilar (UFPE — Oceanografia Física)
**Escopo:** (1) triagem do acervo `C:\DEV\PDFs_artigos` para a bibliografia; (2) diagnóstico e correção do descompasso entre `numeric-tables` e `figures`; (3) plano minimalista para reorganizar/enxugar o código cumprindo todos os objetivos metodológicos; (4) renomeação dos gates G1…G5 como hipóteses HIP1…HIP5; (5) tarefas pós-correção e cronograma.

---

## 1. Bibliografia — o que entrou do acervo

Triei os 73 arquivos de `C:\DEV\PDFs_artigos` e acrescentei **26 referências** relevantes à planilha `Artigos_Referências/Referências_Bibliográficas.xls` (agora com **44** entradas; coluna `Acervo` distingue "Núcleo do parecer" de "Acervo PDFs_artigos"). Todos os DOIs foram extraídos do PDF ou confirmados. Destaques por pilar do projeto:

- **Teleconexão ENSO→Brasil (Fase 4):** Grimm, Ferraz & Gomes (1998, *J. Climate* — Sul do Brasil), Reboita et al. (2021, *Ann. NYAS*), Cabral Júnior et al. (2026), Sparacino et al. (2021), Costa et al. (2020, ETCCDI — índices de extremos, exatamente o que a Fase 4 deveria usar).
- **Confundidor do Atlântico (que o parecer anterior mandou controlar):** Kouadio et al. (2012), Mao et al. (2022), Geirinhas et al. (2023), Koseki et al. (2025), Nobre et al. (2012), Pontes da Silva et al. (2023). Esses papers **justificam cientificamente** a ressalva de que o sinal do NEB não é exclusivamente do Pacífico.
- **Oceanografia física / Kelvin-WWB (Fase 3):** Cui et al. (2025, *JGR-Oceans* — vieses de onda de Kelvin a eventos de vento de oeste), Stewart (2008, livro-texto).
- **Redes neurais para ENSO/Niño (Fases 7–8):** Bachèlery et al. (2025, *Science Advances* — CNN prevê Niños do Atlântico, análogo direto), Ham et al. (2019), Chattopadhyay et al. (2020, CNN espaço-temporal), Taylor & Feng (2022, DL para SSTA), Mir et al. (2024).
- **ML para chuva no Brasil (Fases 5–6):** Tedeschi et al. (2025), Domingos et al. (2025), Dantas et al. (2020), Junior et al. (2021), Araújo et al. (2022), Pinheiro & Ouarda (2025, XAI).
- **Revisão metodológica de ML/física do clima:** Bracco et al. (2024, *Nat. Rev. Physics*).

**Não incluí** os artigos de modelos globais de PNT por IA (GraphCast, Pangu-Weather, FourCastNet, GenCast, Schultz 2021, Fang 2021, etc.) para manter a bibliografia focada no escopo ENSO→Brasil; estão no acervo e podem ser adicionados se quiser embasar decisões de arquitetura das redes. Também deixei de fora arquivos sem DOI atribuível com segurança (ex.: `CERRADO_2019`, `DASILVA_2025`) para não gravar metadado incerto.

---

## 2. Diagnóstico: por que `numeric-tables` está vazio e as figuras não têm mapeamento

O problema **não é aleatório** — é um descompasso estrutural de *registro duplicado*. Encontrei quatro causas concretas.

### 2.1 Causa-raiz: o exportador tem um segundo registro que ficou defasado

`scripts/export_numeric_tables_for_figures.py` carrega um dicionário **fixo no código** (`FIGURE_SOURCES`) que mapeia cada figura → tabelas numéricas. Esse dicionário **deixou de casar com os nomes reais das figuras** produzidas pelos notebooks. Exemplos verificados:

| Chave no exportador (legada) | Figura que existe hoje em disco |
|---|---|
| `fase4/phase4C_recorte_NEB.png` | `Fig_4C1_lags_regiao_bioma_el_nino.png` |
| `fase4/phase4D_mapa_clusters.png` | `Fig_4C3_mapa_pixel_r_melhor_lag.png` |
| `fase4/phase4D_sintese_gate.png` | `Fig_401_selecao_variaveis_pca.png` / `phase40_cobertura_dados.png` |

Como o exportador roda em `--strict` e exige que **toda** figura de `figures/` tenha entrada no dicionário, ao encontrar `Fig_4C1_*` (que não está no dict) ele **falha e aborta antes de escrever qualquer pasta** — por isso `numeric-tables/` fica **vazio** e sobram "12 figuras sem mapeamento".

### 2.2 Três convenções de nome convivendo

- **Fase 2:** `phase2_sanidade_*.png` — **sem código** identificador.
- **Fase 3:** `3A1_*.png`, `3D2_*.png` — código **sem** o prefixo `Fig_`.
- **Fases 4/5/6:** `Fig_4A1_*.png` — código **com** prefixo `Fig_`, mas ainda misturado com legados `phase40_*` e `Fig_401_*`.

O teste de contrato (`tests/test_phase4_figure_contract.py`) só aceita `^Fig_4(0|A|B|C|D)\d+_…` — ou seja, as Fases 2 e 3 **já nascem fora do contrato**, e não há contrato análogo para as Fases 5–8.

### 2.3 Códigos não únicos e poluição na pasta de figuras

- `Fig_5A1_importancia_fases_rf.png` e `Fig_5A1_importancia_fases_xgb.png` **compartilham** o código `5A1` — o código deixa de ser chave única.
- Há lixo em `figures/fase4/`: `_tmp_unit_heatmap_el_nino.png`, `_tmp_unit_heatmap_la_nina.png` (temporários que viraram permanentes).
- Os descritivos vivem em `faseX_legendas_figuras.csv` **dentro de** `figures/` — um **terceiro** registro, desconectado do exportador.

### 2.4 Consequência

Três fontes de verdade desacopladas (nome no notebook · dicionário do exportador · CSV de legendas) **divergem inevitavelmente**. O resultado é o que você vê: tabelas numéricas ausentes, figuras órfãs, teste de contrato reprovando e nenhuma fase visual auditável — violando o Princípio nº 2 do próprio projeto ("nenhuma figura sem tabela/número anterior rastreável").

---

## 3. Correção: um único "código predecessor" que nasce com a figura e a tabela juntas

A solução minimalista é **co-geração**: eliminar os registros paralelos e fazer com que **figura + tabela numérica + descrição sejam criadas pelo mesmo chamado, com o mesmo código**. Nada de exportador post-hoc que redescobre nomes.

### 3.1 Convenção única de código (predecessor)

```
Fig_<F><B><NN>[_<variante>]
 F  = fase           2..8
 B  = bloco          0, A, B, C, D...
 NN = sequência      01, 02, ...
 variante (opcional) = rf, xgb, en, ln...  (mantém o código GLOBALMENTE único)
```

Exemplos: `Fig_2A01`, `Fig_3A01`, `Fig_4C03`, `Fig_5A01_rf`, `Fig_5A02_xgb`. Regra de ouro: **o mesmo código nomeia a figura e a pasta da tabela**:

```
data/processed/figures/fase5/Fig_5A01_rf.png
data/processed/numeric-tables/fase5/Fig_5A01_rf/
    ├── <tabela_1>.csv         (dados congelados usados na figura)
    ├── manifest.csv           (codigo, fase, notebook, fontes, hash, dims, timestamp)
    └── README.md              (título + descrição interpretativa)
```

### 3.2 Um único ponto de escrita (helper compartilhado)

Criar `src/nino_brasil/viz.py::registrar_figura(...)`, chamado por **todos** os notebooks no lugar do `save_fig` disperso:

```python
registrar_figura(
    fig,
    codigo="Fig_4C01",
    fase=4, bloco="C",
    titulo="Lag de resposta por região e bioma (El Niño)",
    descricao="Lag da chuva ao SSTA em EN por unidade IBGE/bioma; célula cheia passa FDR-BH q<0,10.",
    hipotese="HIP1",
    fontes=[tab("phase4C_lag_resposta_neb_sul.csv"),
            tab("phase4C_janelas_lag_variavel.csv")],
)
```

Ele faz, atomicamente: (i) salva o PNG em `figures/fase<F>/<codigo>.png`; (ii) congela cada fonte em `numeric-tables/fase<F>/<codigo>/` + `manifest.csv` (com hash SHA-256 e dimensões) + `README.md`; (iii) faz *upsert* de **uma** linha no manifesto global `data/processed/figuras_manifesto.csv` (colunas: `codigo, fase, bloco, arquivo, notebook, titulo, descricao, hipotese, fontes, atualizado_em`).

### 3.3 O exportador vira validador (não gerador)

`export_numeric_tables_for_figures.py` **perde o dicionário fixo** (≈ 260 linhas → some) e passa a apenas **validar** a coerência tripla:

1. toda figura em `figures/` tem pasta homônima em `numeric-tables/` e linha no manifesto;
2. todo código bate o padrão `Fig_<F><B><NN>` e é **único**;
3. nenhuma pasta de tabela órfã, nenhum `_tmp_*`, nenhum `.csv` de legenda solto em `figures/`.

`--strict` falha **só** se algo divergir — e, por construção, a co-geração impede a divergência.

### 3.4 O que isso elimina (enxugar)

| Hoje (3 registros que divergem) | Depois (1 fonte de verdade) |
|---|---|
| `save_fig` repetido nos notebooks | `registrar_figura` único em `viz.py` |
| `FIGURE_SOURCES` (dict fixo, ~260 linhas) | validador de manifesto (~80 linhas) |
| `faseX_legendas_figuras.csv` (4 arquivos) | coluna `descricao` no manifesto + `README.md` |
| 3 convenções de nome (`phase2_`, `3A1`, `Fig_4A1`) | 1 convenção `Fig_<F><B><NN>` |
| `_tmp_*` e órfãos em `figures/` | bloqueados pelo validador/pre-commit |

---

## 4. Reorganização minimalista do código (cumprindo todos os objetivos)

O objetivo é **código coeso e mínimo**, sem perder nenhum objetivo metodológico. Recomendo a estrutura:

- **`src/nino_brasil/` é a única biblioteca.** Notebooks e `scripts/` só orquestram; nenhuma regra científica dentro de notebook. Hoje `notebooks/fase4/fase4_utils.py` concentra lógica pesada — mover para `src/nino_brasil/phase4/` e deixar o notebook chamando funções.
- **Um módulo por responsabilidade, nomes estáveis:** `io_utils` (escrita atômica), `viz` (registro figura+tabela), `stats/` (N_eff, FDR, bootstrap por evento), `targets/` (alvos: CHIRPS/SPI-SPEI, fases do ciclo), `models/` (RF/XGB, ConvLSTM), `validation/` (folds por evento, baselines).
- **Config central única** (`configs/project.yaml`) já existe — fazer *todos* os caminhos/domínios/limiares saírem dela; remover constantes duplicadas espalhadas (ex.: `CLIM_BASE`, caixas regionais aparecem em vários arquivos).
- **Um comando por fase**, idempotente: `run_fase3_all.py … run_fase8_all.py`, cada um regenerando figuras+tabelas+manifesto e chamando o validador ao fim. Um alvo agregador `scripts/build_all.py --validate`.
- **Remover código morto**: `lagged_corr_pixel` (substituído por `lagged_corr_pixel_matrix`), o dict do exportador, os `faseX_legendas`, notebooks `_broken/` já arquivados.
- **Restaurar o entrypoint da Fase 2** (`build_master_weekly.py` está truncado, sem `main`) — pré-condição de reprodutibilidade.

**Regra de minimalismo:** cada figura responde a **uma** pergunta e cita **uma** hipótese (HIP) e sua(s) tabela(s) predecessora(s). Se uma figura não mapeia para uma HIP, ela sai.

---

## 5. Hipóteses do projeto — substituição de G1…G5 por HIP1…HIP5

Os "gates" eram critérios de governança pass/fail. Reescritos como **hipóteses científicas testáveis**, ficam assim (mantendo o critério de aprovação original):

| Antigo | Nova | Fase | Hipótese (o que se testa) | Critério de sustentação |
|---|---|---|---|---|
| — | **HIP0** | 3 | A sequência de precursores recarga→acoplamento→pico→descarga caracteriza as fases do ciclo e **antecede** o pico (WWV/OHC/D20 lideram o SSTA). | Lags precursores estáveis (LOO por evento), sinal físico coerente, skill vs climatologia. |
| G1 | **HIP1** | 4 | O ENSO do Pacífico **modula a chuva no Brasil** com padrão espacial coerente (NEB seco / Sul úmido em El Niño). | Distribuição pixel-a-pixel com N_eff/FDR, efeito interpretável, estabilidade temporal e lags defensáveis. |
| G2 | **HIP2** | 5 | RF/XGBoost + XAI **caracterizam o ciclo** melhor que a Fase 3 e que os baselines. | Skill fora-da-amostra (folds por evento) > persistência e climatologia; SHAP/PDP coerentes. |
| G3 | **HIP3** | 6 | RF/XGBoost + XAI **preveem a chuva regional** melhor que a triagem estatística da Fase 4 e que os baselines. | `skill_rmse_vs_melhor_baseline > 0` na mesma amostra, por estação chuvosa/evento. |
| G4 | **HIP4** | 7 | A ConvLSTM (campos do Pacífico) **acrescenta skill** sobre índices escalares/ML tabular no mecanismo do ciclo. | Supera climatologia, persistência, Fase 3 e Fase 5 por lead time, com embargo temporal. |
| G5 | **HIP5** | 8 | A rede espaço-temporal **mapeia o Pacífico no campo futuro de chuva do Brasil** com skill superior às Fases 4 e 6. | Supera climatologia, persistência, Fase 4 e Fase 6 por pixel/região/bioma. |

Recomendo trocar o vocabulário em toda a documentação, nos títulos/legendas das figuras (campo `hipotese` no manifesto) e no `DIRETRIZES_FASES.md`, mantendo a tabela de critérios acima como âncora. Cada figura declara a HIP que ajuda a testar; cada HIP tem um conjunto rastreável de figuras+tabelas.

---

## 6. Tarefas pós-correção — manter o código limpo, coeso e com resultado claro

1. **Teste de contrato universal figura↔tabela↔manifesto** (todas as fases, não só a 4), rodando em `pre-commit` e CI. Bloqueia código duplicado, `_tmp_*`, e códigos repetidos.
2. **`build_all.py --validate`**: um comando regenera tudo e valida; se o validador passar, a fase é "auditável" por definição.
3. **Manifesto único versionado** (`figuras_manifesto.csv`) — pequeno, textual, diff-ável; as figuras/tabelas grandes seguem ignoradas pelo Git (já estão).
4. **Docstrings curtas + remoção de código morto** a cada PR; teto de tamanho por módulo.
5. **Linguagem honesta nas legendas**: enquanto os rótulos forem post-hoc, a figura diz "caracterização", não "previsão" (vale para Fases 3/5).
6. **Atualizar README/mermaid** para a matriz 2×3 (o fluxo antigo ainda mostra Fase 5=RF e Fase 6=RN, divergindo do `DIRETRIZES`).
7. **Painel derivado do manifesto** (FaseWEB): status por HIP e por figura sai automático; publica só HIP aprovada.

---

## 7. Cronograma — próximos passos para fechar o projeto

| # | Etapa | Entregável | Hipótese/gate | Critério de conclusão | Depende de |
|---|---|---|---|---|---|
| C0 | **Infra de auditoria** | `viz.registrar_figura`, manifesto único, validador, contrato universal | — | `build_all.py --validate` verde; `numeric-tables` populado; 0 órfãs | — |
| C1 | **Reabrir Fase 2** | `build_master_weekly.py` com `main` + teste end-to-end; unidades ERA5 (W m⁻²) e tensão do vento corrigidas | — | master reproduzível do zero; hashes no manifesto | C0 |
| C2 | **Refazer alvo Fase 4** | CHIRPS diário concatenado → semanal (7 dias válidos) → transformação robusta (SPI/SPEI, MAD/quantis, piso físico); máscara oficial do Brasil | HIP1 | alvo com \|z\| interpretável (sem 27 M); só pixels do Brasil | C1 |
| C3 | **Reinferência Fase 3–4** | Anomalização+detrend a montante; evento como unidade; FDR/field significance; estratificar NEB (FMAM)/Sul | HIP0, HIP1 | HIP1 decidida com número confiável (documento formal) | C2 |
| C4 | **Fase 5 honesta** | Alvo prospectivo (inclui neutro e EN/LN); folds por evento; baselines obrigatórios; salvar SHAP | HIP2 | `skill > persistência` ou HIP2 formalmente reprovada | C3 |
| C5 | **Fase 6 honesta** | Corrigir parser EN/LN; baseline na mesma amostra; validação sazonal/evento | HIP3 | `skill_rmse_vs_baseline > 0` ou reprovação documentada | C2, C4 |
| C6 | **Fase 7 (só se HIP2 passar)** | Data loader lazy; longitudes reais; embargo; pré-treino CMIP/NMME + fine-tuning | HIP4 | supera baselines por lead | C4 |
| C7 | **Fase 8 (só se HIP3 passar)** | Decoder convolucional/EOF baixo-rank; alinhamento por timestamps; máscara/pesos de área; loss probabilística | HIP5 | supera Fases 4 e 6 por pixel/região | C5, C6 |
| C8 | **FaseWEB** | Painel derivado do manifesto; publica só HIP aprovada | — | status automático por HIP e figura | C0–C7 |

**Ordem de ataque imediata:** C0 → C1 → C2. Só depois de C2 (alvo válido) faz sentido reavaliar HIP1 e reabrir ML/redes. Congelar HIP1…HIP5 como "não decididas" até C2 concluída evita gravar conclusões sobre um alvo corrompido.

---

*Parecer emitido em 2026-07-11. O diagnóstico da Seção 2 foi verificado por inspeção direta do exportador, dos nomes de figura em disco e do teste de contrato. A bibliografia da Seção 1 tem DOIs extraídos dos PDFs do acervo ou confirmados; a relação completa está em `Artigos_Referências/Referências_Bibliográficas.xls`.*
