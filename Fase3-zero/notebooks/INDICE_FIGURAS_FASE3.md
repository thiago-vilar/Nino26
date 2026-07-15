# Indice de figuras - Fase 3

Convencao: `3A1` = Fase 3, notebook A, figura 1.

| notebook | figura | titulo | interpreta | tabela_referencia |
| --- | --- | --- | --- | --- |
| 3A | Fig_3A01.png | Series semanais | Cobertura, unidades e comparacao visual das variaveis fisicas. | phase3A_cobertura_variaveis.csv |
| 3A | Fig_3A02.png | Hovmoller SSTA | Mostra propagacao longitudinal da anomalia de SST na faixa equatorial. | phase3A_fontes_variaveis.csv |
| 3A | Fig_3A03.png | Hovmoller SLA + vento | Conecta SLA local e setas de tau_x para leitura qualitativa de Kelvin/acoplamento. | phase3A_fontes_variaveis.csv |
| 3B | Fig_3B01.png | Trajetorias por classe | Compara a evolucao media da SSTA por classe NOAA/ONI. | phase3B_trajetorias_compostas.csv |
| 3B | Fig_3B02.png | Autocorrelacao | Mede memoria/persistencia da SSTA que qualquer previsao deve superar. | phase3B_autocorrelacao.csv |
| 3B | Fig_3B03.png | Mapa composto do pico | Mostra a assinatura espacial media no pico dos eventos. | phase3B_mapa_composto_resumo.csv |
| 3B | Fig_3B04.png | Sensibilidade da faixa de pico | Compara a duração da faixa sob 80%, 90% e 95% do extremo. | phase3_peak_band_sensitivity.csv |
| 3C | Fig_3C01.png | Heatmap de lags | Triagem bruta de quais variaveis antecedem a SSTA e em que lag. | phase3C_ranking_lags.csv |
| 3C | Fig_3C02.png | Longitude x lag | Mostra onde no Pacifico equatorial o sinal antecedente aparece por longitude. | phase3C_lag_correlacoes.csv |
| 3D | Fig_3D01.png | Forest IC95 | Aplica N_eff, FDR e IC95 para reduzir falsos positivos. | phase3D_ranking_significativo.csv |
| 3D | Fig_3D02.png | Mapa FDR | Mostra regioes longitude-lag que sobrevivem ao controle estatistico. | phase3D_testes_completos.csv |
| 3E | Fig_3E01.png | Sensibilidade sem breakpoint | Compara r completo, envelope IC95 do bootstrap movel e faixa leave-one-event-out; nao define exclusao. | phase3E_sensibilidade_resumo.csv |
| 3E | Fig_3E02.png | Influencia de eventos | Mostra quanto r muda ao retirar cada evento EN/LN, sem dividir a serie por ano. | phase3E_leave_one_event_out.csv |
| 3F | Fig_3F01.png | Kelvin por SLA | Diagnostico visual de propagacao oeste-leste por SLA/SSH em eventos fortes. | phase3F_kelvin_eventos_resumo.csv |
| 3F | Fig_3F02.png | Vento e SLA | Resume tau_x_anom na Nino 3.4 junto ao sinal de SLA por evento. | phase3F_kelvin_eventos_resumo.csv |
| 3G | Fig_3G01.png | SSTA por classe | Compara a evolucao termica media por classe NOAA/ONI. | phase3G_composto_ssta_classes_noaa.csv |
| 3G | Fig_3G02.png | Escalonamento termico | Relaciona pico ONI local, duracao e taxas de crescimento/decaimento. | phase3G_escalonamento_ssta.csv |
| 3G | Fig_3G03.png | SSTA longitude | Compara fortes/super historicos com a formacao atual 2025/26 por longitude. | phase3G_mapa_ssta_lon_eventos_forte_super.csv |
| 3H | Fig_3H01.png | Onset por classe | Mostra quais variaveis se separam na genese dos eventos. | phase3H_estado_precursor_por_classe.csv |
| 3H | Fig_3H02.png | Ciclo de vida | Resume genese, crescimento, pico e decaimento com variaveis em z-score. | phase3H_ciclo_vida_media.csv |
| 3H | Fig_3H03.png | Ciclo físico ampliado | Resume subsuperfície e atmosfera nas quatro fases. | phase3H_ciclo_vida_media_subsuperficie_atmosfera.csv |
| 3I | Fig_3I01.png | Sintese do parecer | Organiza evidencias do 3D por bloco fisico, lag e metricas continuas de sensibilidade do 3E. | phase3I_conclusoes_decisao.csv |
| 3I | Fig_3I02.png | Antecipacao | Mostra variaveis candidatas para antecipar o aquecimento maximo. | phase3I_conjunto_antecipacao_pico.csv |
| 3I | Fig_3I03.png | Nested LOO | Avalia selecao+ajuste por nested LOO e gera projecao condicional. | phase3I_nested_loo_metricas.csv |
| 3K | Fig_3K01.png | Skill PCA | Testa se PCA reduz redundancia sem perder skill preditivo. | phase3K_previsao_pico_nested_loo_metricas.csv |
| 3K | Fig_3K02.png | Scree PCA | Mostra quantos componentes explicam a variancia de crescimento. | phase3K_pca_variancia.csv |
| 3K | Fig_3K03.png | Biplot PCA | Mostra agrupamentos fisicos e colinearidade entre variaveis. | phase3K_pca_loadings.csv |
| 3L | Fig_3L01.png | Ciclo de vida EN/LN | Compara genese, crescimento, pico e decaimento de El Nino e La Nina. | phase3_event_lifecycle_en_ln.csv |
| 3L | Fig_3L03.png | Duracao por fase EN/LN | Resume a duracao media por tipo, classe e fase do ciclo de vida. | phase3_duracao_por_tipo_classe.csv |
| 3L | Fig_3L02.png | Friedman pareado e Kendall W | Mostra Kendall W para todas as variaveis e marca como confirmadas somente as com qBH<=0,05, em familia separada por tipo ENSO. | phase3_discriminantes_por_periodo.csv |
| 3L | Fig_3L04.png | PCA por fase | Resume a estrutura multivariada por genese, crescimento, pico e decaimento. | phase3_pca_por_fase.csv |
