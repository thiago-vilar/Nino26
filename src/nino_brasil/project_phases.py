from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectPhase:
    number: int
    slug: str
    title: str
    focus: str
    milestone: str

    @property
    def label(self) -> str:
        return f"Fase {self.number}"


PHASES: tuple[ProjectPhase, ...] = (
    ProjectPhase(
        1,
        "foundation_ingestion",
        "Base local e ingestao bruta",
        "Fundacao local, Git, catalogo, scripts, credenciais e cache bruto por fonte.",
        "Dados essenciais baixaveis/reprodutiveis e estrutura versionada.",
    ),
    ProjectPhase(
        2,
        "standardization_anomalies_lags_regridding",
        "Padronizacao, anomalias, lags e regrid",
        "QC, climatologia sem vazamento, anomalias, acumulados, percentis, lags e grade comum 0.25 grau.",
        "Cubos Zarr reconciliados em data/processed/zarr/regridded/.",
    ),
    ProjectPhase(
        3,
        "nino34_physical_signal_diagnostics",
        "Diagnostico fisico do sinal Nino 3.4",
        "Alinhamento de anomalias, volume/grau de termoclina, slope, duracao do sinal e comparacao com picos historicos de El Nino.",
        "Feature store fisico com diagnosticos de subsuperficie, superficie e duracao de sinal para Nino 3.4.",
    ),
    ProjectPhase(
        4,
        "exploratory_statistical_preanalysis",
        "Pre-analises estatisticas experimentais",
        "Triagem com regressao multipla, PCA, KNN, correlacoes defasadas e combinacoes de variaveis Nino 3.4.",
        "Ranking experimental das variaveis e combinacoes mais associadas a seca no Nordeste e chuva no Sul.",
    ),
    ProjectPhase(
        5,
        "event_centered_ai_xai",
        "IA em duas etapas: ENSO e teleconexoes",
        "Etapa 5A aprende a progressao diaria OISST/SSTA ate o pico El Nino/Super El Nino; Etapa 5B aprende Nino3.4 -> eventos climaticos por clusters de pixels no Brasil.",
        "Stores Zarr de progressao ENSO, progressao por cluster, metricas, previsoes, importancias e pesos por grupo.",
    ),
    ProjectPhase(
        6,
        "neural_event_progression_xai",
        "Redes neurais 6A/6B/6C e XAI",
        "6A treina um encoder CNN espacial multihorizonte; 6B testa memoria espaco-temporal para progressao ate pico ENSO; 6C aprende teleconexoes neurais Nino3.4 -> clusters/P90 no Brasil.",
        "Stores Zarr de treino neural, previsoes por evento, metricas P90/P10, XAI e comparacao contra Fase 5; CMIP6 fica apenas como fallback de pre-treino/fine-tuning.",
    ),
    ProjectPhase(
        7,
        "publication_operations",
        "Publicacao e operacao",
        "GitHub Pages, relatorios automaticos, comparacao previsao-observado, drift, recalibracao e rotina recorrente.",
        "Produto operacional em docs/ com painel publico e rotina atualizavel.",
    ),
    ProjectPhase(
        8,
        "ham2019_exploratory_benchmark",
        "Exploracao adicional Ham2019",
        "Testes isolados com arquitetura, pesos salvos e dados associados ao estudo Ham2019/reproducao, sem contaminar a trilha principal da Fase 6.",
        "Relatorios e stores separados de compatibilidade, inferencia, skill comparativo e limites de transferencia dos pesos externos.",
    ),
)


PHASES_BY_NUMBER = {phase.number: phase for phase in PHASES}
PHASES_BY_SLUG = {phase.slug: phase for phase in PHASES}


def phase_table_rows() -> list[list[str]]:
    return [[phase.label, phase.title, phase.focus, phase.milestone] for phase in PHASES]
