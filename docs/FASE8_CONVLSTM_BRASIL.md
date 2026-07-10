# Fase 8 - Distribuicao no Brasil com redes neurais ConvLSTM

**Coluna B da matriz metodologica, linha redes neurais.** Mesmo estudo das Fases 4 e
6 com ConvLSTM. Modulo: `src/nino_brasil/models/phase8_convlstm_brazil.py`.

## Desenho
- Arquitetura **encoder-decoder**: o encoder ConvLSTM le sequencias do Pacifico e o
  decoder projeta o campo de anomalia de chuva do Brasil `(lat, lon)` em t+horizonte
  (`build_convlstm_encoder_decoder`).
- Pareamento causal Pacifico(t) -> chuva Brasil(t+horizonte) (`align_pacific_to_brazil`).
- Avaliacao por pixel e agregada por regiao e bioma (reaproveita
  `maps.spatial_support`), comparando com Fases 4 e 6.
- Validacao cronologica (sem shuffle).

## Operacao
Treino exige TensorFlow + GPU na maquina do usuario. Referencias: Ham et al. (2019),
Verma et al. (2024, ClimODE), Zhou et al. (2025, dinamica + deep learning ENSO).

## Gate G5
So avanca se superar climatologia, persistencia, Fase 4 e Fase 6.
