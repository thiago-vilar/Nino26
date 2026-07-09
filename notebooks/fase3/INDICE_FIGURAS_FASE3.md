# Indice de figuras - Fase 3

Convencao: `3A1` = Fase 3, notebook A, figura 1. Toda figura tem tabela numerica de referencia (regra de ouro: nenhuma saida visual sem saida numerica anterior).

| notebook | figura | titulo | interpreta | tabela_referencia |
| --- | --- | --- | --- | --- |
| 3A | 3A1_series_semanais.png | Series semanais | Cobertura, unidades e comparacao visual das variaveis fisicas. | phase3A_cobertura_variaveis.csv |
| 3A | 3A2_hovmoller_ssta.png | Hovmoller SSTA + picos | Propagacao longitudinal da SSTA equatorial com estrela no epicentro de cada pico (mes/ano do pico ONI local) e triangulo no maximo 2025/26 em curso. | phase3A_picos_epicentros.csv |
| 3A | 3A3_hovmoller_sla_taux.png | Hovmoller SLA + vento | Conecta SLA local e setas de tau_x (agora ate 2026) para leitura qualitativa de Kelvin/acoplamento. | phase3A_taux_quinzenal_janelas.csv |
| 3B | 3B1_trajetorias_compostas.png | Trajetorias por classe | Compara a evolucao media da SSTA por classe NOAA/ONI. | phase3B_trajetorias_compostas.csv |
| 3B | 3B2_autocorrelacao.png | Autocorrelacao | Memoria/persistencia da SSTA (~27 sem ~ 6 meses ~ duracao do crescimento onset->pico); baseline que a Fase 5 deve superar. | phase3B_autocorrelacao.csv, phase3B_memoria_persistencia.csv |
| 3B | 3B3_mapa_composto_pico.png | Mapa composto do pico | Mostra a assinatura espacial media no pico dos eventos. | phase3B_mapa_composto_resumo.csv |
| 3B | 3B4_faixa_pico_oni.png | Faixa de pico por evento | Delimita o pico como JANELA (meses com ONI >= pico-0.1 C); fortes tem faixa estreita, fracos tem faixa larga. Fundamentacao: docs/PICO_FAIXA_BIBLIOGRAFIA.md. | phase3B_faixa_pico_eventos.csv |
| 3C | 3C1_heatmap_lags.png | Heatmap de lags | Triagem bruta de quais variaveis antecedem a SSTA e em que lag. | phase3C_ranking_lags.csv |
| 3C | 3C2_mapa_lon_lag.png | Longitude x lag | Mostra onde no Pacifico equatorial o sinal antecedente aparece por longitude. | phase3C_lag_correlacoes.csv |
| 3D | 3D1_forest_ic95.png | Forest IC95 | Aplica N_eff, FDR e IC95 para reduzir falsos positivos. | phase3D_ranking_significativo.csv |
| 3D | 3D2_mapa_lon_lag_fdr.png | Mapa FDR | Mostra regioes longitude-lag que sobrevivem ao controle estatistico. | phase3D_testes_completos.csv |
| 3E | 3E1_scatter_estabilidade.png | Estabilidade | Compara correlacoes 1993-2009 vs 2010-presente. | phase3E_estabilidade.csv |
| 3E | 3E2_mapa_lon_lag_subperiodos.png | Subperiodos | Testa se o padrao longitude-antecedencia e o mesmo nos dois regimes (justifica extrapolar 1993+ para 2025/26); contorno r=0.5 delimita sinal util. | phase3E_lonlag_subperiodos_resumo.csv |
| 3F | 3F1_hovmoller_sla_kelvin.png | Kelvin por SLA | Propagacao oeste-leste por SLA/SSH com SETAS grandes de onda de Kelvin ancoradas nos pulsos do Pacifico oeste (~2.4 m/s). | phase3F_kelvin_setas.csv |
| 3F | 3F2_taux_sla_eventos.png | Vento e SLA | Resume tau_x_anom na Nino 3.4 junto ao sinal de SLA por evento. | phase3F_kelvin_eventos_resumo.csv |
| 3G | 3G1_composto_ssta_noaa.png | SSTA por classe | Compara a evolucao termica media por classe NOAA/ONI. | phase3G_composto_ssta_classes_noaa.csv |
| 3G | 3G2_escalonamento_ssta.png | Escalonamento termico | Relaciona pico ONI local, duracao e taxas de crescimento/decaimento. | phase3G_escalonamento_ssta.csv |
| 3G | 3G3_mapa_ssta_lon.png | SSTA longitude + epicentros | Compara fortes/super historicos com 2025/26 por longitude; estrela = epicentro (mes/ano, longitude, C) e legenda interna de leitura. | phase3G_mapa_ssta_lon_eventos_forte_super.csv |
| 3H | 3H1_compostos_onset.png | Onset por classe | Mostra quais variaveis se separam na genese dos eventos. | phase3H_estado_precursor_por_classe.csv |
| 3H | 3H2_ciclo_vida.png | Ciclo de vida (oceano) | Genese, crescimento, pico (como FAIXA dourada max-0.1 C) e decaimento com variaveis oceanicas em z-score. | phase3H_ciclo_vida_media.csv, phase3H_fases_ciclo_vida.csv |
| 3H | 3H3_ciclo_vida_subsuperficie_atmosfera.png | Ciclo de vida (subsuperficie + atmosfera) | Completa o 3H2 com T50-T300, OHC 0-100/300-700, tilt_slope e o bloco atmosferico de Bjerknes (u10/u850/u200, MSLP, TCWV, SSR, omega500, SLHF). | phase3H_ciclo_vida_media_subsuperficie_atmosfera.csv |
| 3I | 3I1_sintese_parecer.png | Sintese do parecer | Organiza quais evidencias entram, entram com ressalva ou ficam fora. | phase3I_conclusoes_decisao.csv |
| 3I | 3I2_antecipacao_pico.png | Antecipacao (redesenhado) | Barras de antecedencia: quem avisa mais cedo e com que forca (|r|); lag 0 listado na legenda lateral. | phase3I_conjunto_antecipacao_pico.csv |
| 3I | 3I3_previsao_condicional_nested.png | Nested LOO | Triagem flat LOO ampliada para 20 variaveis + nested LOO (selecao+ajuste) + projecao condicional 2025/26. | phase3I_skill_por_variavel.csv, phase3I_nested_loo_metricas.csv, phase3I_projecao_pico_2026.csv |
| 3K | 3K1_skill_loo_nested.png | Skill PCA | Responde se o conjunto reduzido pelo PCA preve o pico tao bem quanto todas as variaveis (rotulos em linguagem direta). | phase3K_previsao_pico_nested_loo_metricas.csv |
| 3K | 3K2_scree.png | Scree PCA | Mostra quantos componentes explicam a variancia de crescimento. | phase3K_pca_variancia.csv |
| 3K | 3K3_biplot.png | Biplot PCA | Mostra agrupamentos fisicos e colinearidade entre variaveis. | phase3K_pca_loadings.csv |
| 3L | phase3L_ciclo_vida_en_ln.png | Ciclo de vida EN/LN | Compara as 4 fases (I. genese, II. crescimento, III. pico, IV. decaimento) de El Nino e La Nina, com recarga espelhada explicada. | phase3_event_lifecycle_en_ln.csv |
| 3L | phase3L_duracao_fases_en_ln.png | Duracao por fase EN/LN | Duracao media por tipo, classe e fase (valores em semanas nas barras). | phase3_duracao_por_tipo_classe.csv |
| 3L | phase3L_discriminantes_heatmap.png | Discriminantes por periodo | Quais variaveis delimitam melhor as quatro fases por sinal ENSO, com nota de leitura do epsilon^2. | phase3_discriminantes_por_periodo.csv |
| 3L | phase3L_pca_por_fase.png | PCA por fase | Complexidade de cada fase: PC1 alto = fase dominada por uma dimensao fisica. | phase3_pca_por_fase.csv |

Nota metodologica: o pico dos eventos e tratado como FAIXA (janela), nao como ponto; ver `docs/PICO_FAIXA_BIBLIOGRAFIA.md` e `phase3B_faixa_pico_eventos.csv`.
