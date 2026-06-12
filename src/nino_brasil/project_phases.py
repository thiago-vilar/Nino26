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
        "classical_ml_xai",
        "Machine learning classico e XAI",
        "Ridge, Random Forest, XGBoost/LightGBM, walk-forward, permutation importance, SHAP e pesos por grupo.",
        "Stores Zarr de metricas, previsoes, importancias, pesos por grupo e mapas analiticos.",
    ),
    ProjectPhase(
        6,
        "neural_networks_xai",
        "Redes neurais, XAI e memoria experimental",
        "CNN, ConvLSTM, U-Net, Transformer espaco-temporal, XAI neural e experimentos Memory Caching.",
        "Stores Zarr de treino/inferencia neural, explicabilidade neural e comparacao contra fases anteriores.",
    ),
    ProjectPhase(
        7,
        "publication_operations",
        "Publicacao e operacao",
        "GitHub Pages, relatorios automaticos, comparacao previsao-observado, drift, recalibracao e rotina recorrente.",
        "Produto operacional em docs/ com painel publico e rotina atualizavel.",
    ),
)


PHASES_BY_NUMBER = {phase.number: phase for phase in PHASES}
PHASES_BY_SLUG = {phase.slug: phase for phase in PHASES}


def phase_table_rows() -> list[list[str]]:
    return [[phase.label, phase.title, phase.focus, phase.milestone] for phase in PHASES]
