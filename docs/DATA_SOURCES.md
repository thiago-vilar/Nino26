# Fontes de dados NINO26

Este inventário descreve papéis científicos, não importância prévia de
variáveis. Caminhos e comandos detalhados permanecem nos runbooks.

| Fonte | Papel |
|---|---|
| NOAA OISST | SST/SSTA local em Niño 3.4 e identificação oceânica ENSO |
| ERA5 | variáveis atmosféricas candidatas |
| CHIRPS | alvos de chuva no Brasil no pixel de tamanho original |
| UFS+GLORYS | base oceânica diária apresentada conjuntamente, com procedência interna por fonte e agregação em semanas completas W-SUN |

Fontes primárias são aceitas quando nativamente diárias, subdiárias ou
semanais. Séries mensais são proibidas no fluxo operacional. A única exceção é
um índice consolidado empregado separadamente para comparação, sem integração,
interpolação ou preenchimento da matriz semanal.
| CTD/WOD | validação in situ de perfis e profundidades |
| TAO/TRITON | validação equatorial por boias |
| Argo | validação por perfis, especialmente a partir dos anos 2000 |
| IBGE | shapefiles oficiais de regiões e biomas |

## ENSO

A definição segue NOAA/CPC: Niño 3.4, média de três meses, limiares `+0,5 °C` e
`−0,5 °C`, persistência por cinco estações móveis sobrepostas e confirmação
atmosférica. P90 não define El Niño ou La Niña.

## UFS+GLORYS

Em textos, gráficos e apresentações, usar sempre **UFS+GLORYS**. Internamente:

- UFS cobre a parte histórica configurada.
- GLORYS cobre o período posterior configurado.
- Indicadores de fonte, cobertura, unidade e resolução são preservados.
- D20, OHC por camadas, WWV, termoclina, SSH/SLA, temperatura e salinidade são
  variáveis candidatas, sem importância definida antes dos testes.

## CHIRPS

Aplicam-se apenas as regras canônicas atuais:

- unidade espacial: pixel no tamanho original CHIRPS;
- alvos: extremo de chuva, chuva forte, chuva normal, estiagem e seca, após
  definição numérica documentada;
- apresentação: pixel a pixel, regiões do país e biomas por região;
- limites: shapefiles oficiais IBGE.

## Validação in situ

CTD/WOD, TAO/TRITON e Argo não são preditores obrigatórios nem substituem
UFS+GLORYS. São camadas independentes de validação. Todo pareamento deve declarar
período, localização, profundidade, tolerância espaço-temporal, número de
observações e métricas de diferença.

## Armazenamento

- bruto: `data/raw/`;
- intermediário: `data/interim/`;
- processado: `data/processed/`;
- imagens: `data/processed/figures/`;
- tabelas de figuras: `data/processed/numeric-tables/`;
- metadados: `data/processed/metadata/` e `data/audit/`.

Consulte [RUNBOOK_DOWNLOADS.md](RUNBOOK_DOWNLOADS.md),
[RUNBOOK_OCEAN_DAILY.md](RUNBOOK_OCEAN_DAILY.md) e
[RUNBOOK_FASE2_OCEANO.md](RUNBOOK_FASE2_OCEANO.md) para comandos operacionais.
