# Indice de figuras - Fase 3

Convencao: `3A1` = Fase 3, notebook A, figura 1.

| notebook | figura | titulo | interpreta | tabela_referencia |
| --- | --- | --- | --- | --- |
| 3A | 3A1_series_semanais.png | Series semanais | Cobertura, unidades e comparacao visual das variaveis fisicas. | phase3A_cobertura_variaveis.csv |
| 3A | 3A2_hovmoller_ssta.png | Hovmoller SSTA | Mostra propagacao longitudinal da anomalia de SST na faixa equatorial. | phase3A_fontes_variaveis.csv |
| 3A | 3A3_hovmoller_sla_taux.png | Hovmoller SLA + vento | Conecta SLA local e setas de tau_x para leitura qualitativa de Kelvin/acoplamento. | phase3A_fontes_variaveis.csv |
| 3B | 3B1_trajetorias_compostas.png | Trajetorias por classe | Compara a evolucao media da SSTA por classe NOAA/ONI. | phase3B_trajetorias_compostas.csv |
| 3B | 3B2_autocorrelacao.png | Autocorrelacao | Mede memoria/persistencia da SSTA que qualquer previsao deve superar. | phase3B_autocorrelacao.csv |
| 3B | 3B3_mapa_composto_pico.png | Mapa composto do pico | Mostra a assinatura espacial media no pico dos eventos. | phase3B_mapa_composto_resumo.csv |
| 3C | 3C1_heatmap_lags.png | Heatmap de lags | Triagem bruta de quais variaveis antecedem a SSTA e em que lag. | phase3C_ranking_lags.csv |
| 3C | 3C2_mapa_lon_lag.png | Longitude x lag | Mostra onde no Pacifico equatorial o sinal antecedente aparece por longitude. | phase3C_lag_correlacoes.csv |
| 3D | 3D1_forest_ic95.png | Forest IC95 | Aplica N_eff, FDR e IC95 para reduzir falsos positivos. | phase3D_ranking_significativo.csv |
| 3D | 3D2_mapa_lon_lag_fdr.png | Mapa FDR | Mostra regioes longitude-lag que sobrevivem ao controle estatistico. | phase3D_testes_completos.csv |
| 3E | 3E1_scatter_estabilidade.png | Estabilidade | Compara correlacoes 1993-2009 vs 2010-presente. | phase3E_estabilidade.csv |
| 3E | 3E2_mapa_lon_lag_subperiodos.png | Subperiodos | Testa se o padrao longitudinal se repete em regimes diferentes. | phase3E_estabilidade.csv |
| 3F | 3F1_hovmoller_sla_kelvin.png | Kelvin por SLA | Diagnostico visual de propagacao oeste-leste por SLA/SSH em eventos fortes. | phase3F_kelvin_eventos_resumo.csv |
| 3F | 3F2_taux_sla_eventos.png | Vento e SLA | Resume tau_x_anom na Nino 3.4 junto ao sinal de SLA por evento. | phase3F_kelvin_eventos_resumo.csv |
| 3G | 3G1_composto_ssta_noaa.png | SSTA por classe | Compara a evolucao termica media por classe NOAA/ONI. | phase3G_composto_ssta_classes_noaa.csv |
| 3G | 3G2_escalonamento_ssta.png | Escalonamento termico | Relaciona pico ONI local, duracao e taxas de crescimento/decaimento. | phase3G_escalonamento_ssta.csv |
| 3G | 3G3_mapa_ssta_lon.png | SSTA longitude | Compara fortes/super historicos com a formacao atual 2025/26 por longitude. | phase3G_mapa_ssta_lon_eventos_forte_super.csv |
| 3H | 3H1_compostos_onset.png | Onset por classe | Mostra quais variaveis se separam na genese dos eventos. | phase3H_estado_precursor_por_classe.csv |
| 3H | 3H2_ciclo_vida.png | Ciclo de vida | Resume genese, crescimento, pico e decaimento com variaveis em z-score. | phase3H_ciclo_vida_media.csv |
| 3I | 3I1_sintese_parecer.png | Sintese do parecer | Organiza quais evidencias entram, entram com ressalva ou ficam fora. | phase3I_conclusoes_decisao.csv |
| 3I | 3I2_antecipacao_pico.png | Antecipacao | Mostra variaveis candidatas para antecipar o aquecimento maximo. | phase3I_conjunto_antecipacao_pico.csv |
| 3I | 3I3_previsao_condicional_nested.png | Nested LOO | Avalia selecao+ajuste por nested LOO e gera projecao condicional. | phase3I_nested_loo_metricas.csv |
| 3K | 3K1_skill_loo_nested.png | Skill PCA | Testa se PCA reduz redundancia sem perder skill preditivo. | phase3K_previsao_pico_nested_loo_metricas.csv |
| 3K | 3K2_scree.png | Scree PCA | Mostra quantos componentes explicam a variancia de crescimento. | phase3K_pca_variancia.csv |
| 3K | 3K3_biplot.png | Biplot PCA | Mostra agrupamentos fisicos e colinearidade entre variaveis. | phase3K_pca_loadings.csv |
| 3L | phase3L_ciclo_vida_en_ln.png | Ciclo de vida EN/LN | Compara genese, crescimento, pico e decaimento de El Nino e La Nina. | phase3_event_lifecycle_en_ln.csv |
| 3L | phase3L_duracao_fases_en_ln.png | Duracao por fase EN/LN | Resume a duracao media por tipo, classe e fase do ciclo de vida. | phase3_duracao_por_tipo_classe.csv |
| 3L | phase3L_discriminantes_heatmap.png | Discriminantes por periodo | Mostra quais variaveis delimitam melhor as quatro fases por sinal ENSO. | phase3_discriminantes_por_periodo.csv |
| 3L | phase3L_pca_por_fase.png | PCA por fase | Resume a estrutura multivariada por genese, crescimento, pico e decaimento. | phase3_pca_por_fase.csv |
