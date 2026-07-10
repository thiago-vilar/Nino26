# Fase 7 - Ciclo ENSO com redes neurais ConvLSTM

**Coluna A da matriz metodologica, linha redes neurais.** Mesmo mecanismo das Fases
3 e 5 com uma rede espaco-temporal. Modulo: `src/nino_brasil/models/phase7_convlstm.py`.

## Desenho
- Entrada: cubo `(time, lat, lon, canal)` de campos regriddados do Pacifico
  equatorial (`load_pacific_cube` a partir de `data/processed/zarr/regridded/`).
- Sequencias deslizantes espaco-temporais (`make_sequences`), alvo = fase na semana
  final da janela (+horizonte).
- Rede: `ConvLSTM2D` empilhada -> pooling -> softmax das 4 fases
  (`build_convlstm_classifier`).
- Validacao: split cronologico (`chronological_split`), nunca aleatorio.
- XAI espacial: oclusao de canais (`channel_occlusion_importance`) ranqueia as
  variaveis mais significativas por etapa.

## Operacao
TensorFlow/Keras e importado sob guarda: a preparacao de dados roda sem TF; o
**treino exige TensorFlow + GPU** na maquina do usuario. Referencias de arquitetura:
Ham et al. (2019, deep learning for ENSO), Chen et al. (2023, PINN ENSO).

## Gate G4
So avanca se superar climatologia, persistencia, Fase 3 e Fase 5.
