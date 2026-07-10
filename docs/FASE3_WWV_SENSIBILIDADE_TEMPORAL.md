# WWV e sensibilidade temporal na Fase 3

## Resposta direta

`WWV` significa *Warm Water Volume*: o volume de agua situado acima da
isoterma de 20 °C no Pacifico equatorial. No produto usado pela Fase 3, ele e
calculado como `D20 x area da celula`, integrado entre 5°S-5°N e 120°E-80°W,
e expresso em m3. O indicador representa o estado de recarga/descarga do
oceano superior no paradigma do oscilador de recarga do ENSO.

WWV nao e uma observacao independente de D20. Ele e derivado da profundidade
da isoterma de 20 °C e compartilha sinal com D20, OHC, SSH e inclinacao da
termoclina. Portanto, neste projeto ele e **um candidato do bloco de recarga**,
nao o eixo obrigatorio da analise, nao o representante fixo do bloco e nao um
criterio de aceitacao.

Ha duas funcoes diferentes no codigo-base que nao devem ser confundidas:

- `warm_water_volume_m3()` produz o WWV integrado em volume e e a funcao usada
  pelas pipelines oceanicas diaria e mensal;
- `warm_water_volume()` produz uma proxy de profundidade media de D20, em
  metros, mantida para usos auxiliares/testes. Ela nao e a coluna
  `wwv_equatorial_pacific_m3` consumida pela matriz semanal da Fase 3.

## De onde veio o corte 1993-2009 / 2010+

O ano de 1993 tem uma justificativa de dados: e o inicio do GLORYS12 usado como
fonte diaria principal e, por isso, o inicio da janela comum de superficie e
subsuperficie. Isso e uma fronteira de cobertura/fonte, nao uma mudanca de
regime climatico.

O ano de 2010 nao tinha uma justificativa equivalente. A configuracao e o
notebook 3E dividiam a janela comum em dois blocos quase equilibrados e
transformavam a perda de `p < 0,05` em um dos blocos em filtro binario. Isso
era legado metodologico, por tres motivos:

1. a referencia citada, McPhaden (2012), comparou 1980-1999 com 2000-2010 e
   descreveu uma mudanca em torno da virada do seculo, nao em 2010;
2. "significativo em um bloco e nao significativo no outro" nao demonstra que
   as duas correlacoes sejam estatisticamente diferentes;
3. a divisao reduzia numero de eventos e potencia estatistica e depois era
   usada indevidamente como corte estrutural para todas as variaveis.

Consequentemente, a divisao 1993-2009 / 2010+ foi removida do protocolo ativo.
A literatura sobre variacao decadal do lead do WWV continua sendo contexto
fisico, mas nao define um breakpoint para estes dados.

## Metodo ativo, sem breakpoint

O 3E agora recebe do 3D o lag ja selecionado e faz apenas analise de
sensibilidade:

- bootstrap movel pareado, com blocos de 26, 52 e 78 semanas, preservando a
  dependencia serial dentro dos blocos;
- leave-one-event-out, retirando um evento El Nino ou La Nina local por vez;
- estimativa do periodo completo como resultado principal;
- IC95, fracao de replicas com o mesmo sinal, faixa de correlacoes LOO e maior
  influencia de um evento como metricas explicitas;
- nenhuma coluna `estavel`, nenhum suposto regime e nenhum descarte automatico.

As rotinas reutilizaveis ficam em
`src/nino_brasil/stats/temporal_stability.py`. O arquivo
`phase3E_estabilidade.csv` e preservado apenas como alias de compatibilidade;
seu conteudo novo e o resumo de sensibilidade sem breakpoint.

## Inventario dos marcos temporais parecidos

| Marco | Papel real | Pode ser chamado de regime climatico? |
|---|---|---|
| 1993 | inicio do GLORYS12 e da janela comum principal | nao; e fronteira de fonte/cobertura |
| 2000+ | sensibilidade de cobertura subsuperficial/Argo e contexto da literatura WWV | nao sem teste formal de mudanca |
| 2010 | antiga divisao equilibrada do 3E | nao; removida por ser arbitraria |
| 2015 | transicao de stream/produto oceanico em configuracao | nao; e emenda de produto |
| 2026 | cauda operacional GLO12 | nao; e atualizacao de fonte |

Emendas de fonte devem ser auditadas como emendas. Uma ruptura fisica so pode
ser declarada por um estudo de ponto de mudanca pre-especificado, com correcao
pela busca do ponto e incerteza apropriada; isso nao e pressuposto na Fase 3.

## Onde WWV continua sendo usado

WWV permanece disponivel na matriz semanal, na triagem de lags, nos compostos
de ciclo de vida, na PCA e nos candidatos de modelos exploratorios. Em todos
esses casos sua leitura correta e "indicador basinwide de recarga". A selecao
de um representante entre WWV, D20, OHC, SSH e tilt deve resultar da pergunta
fisica, da reducao de colinearidade/PCA e da validacao, e nao de preferencia
fixada na configuracao.

## Referencias primarias

- Meinen, C. S.; McPhaden, M. J. (2000), *Observations of Warm Water Volume
  Changes in the Equatorial Pacific and Their Relationship to El Nino and La
  Nina*, Journal of Climate, 13, 3551-3559.
- McPhaden, M. J. (2012), *A 21st century shift in the relationship between
  ENSO SST and warm water volume anomalies*, Geophysical Research Letters,
  39, L09706, doi:10.1029/2012GL051826.
