from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProjectPhase:
    number: int | None
    slug: str
    title: str
    focus: str
    milestone: str
    display_label: str | None = None

    @property
    def label(self) -> str:
        if self.display_label:
            return self.display_label
        if self.number is None:
            return self.slug
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
        "Padronizacao, anomalias, matriz semanal, cubos, disponibilizacao e graficos de sanidade no tempo.",
        "Dados disponiveis de forma neutra, sem selecionar variaveis para outras fases.",
    ),
    ProjectPhase(
        3,
        "nino34_physical_signal_diagnostics",
        "Diagnostico fisico do sinal Nino 3.4",
        "Analises puramente estatisticas de El Nino e La Nina, sem importancia previa de variaveis.",
        "Caracterizacao independente de genese, crescimento, faixa de pico e decaimento.",
    ),
    ProjectPhase(
        4,
        "enso_brazil_rainfall_teleconnection",
        "Teleconexao ENSO -> chuvas extremas/secas no Brasil",
        "Afericao puramente estatistica da relacao ENSO-Brasil.",
        "Lags e distribuicao espacial-temporal do sinal por pixel, regiao e bioma.",
    ),
    ProjectPhase(
        5,
        "ml_cycle_rfxgb_xai",
        "Ciclo ENSO com Machine Learning (RF/XGBoost) e XAI",
        "Random Forest e XGBoost para prever antecipadamente a evolucao das fases e a faixa de pico.",
        "Estudo independente; data augmentation somente se necessario e restrito ao treino.",
    ),
    ProjectPhase(
        6,
        "ml_brazil_teleconnection_xai",
        "Distribuicao no Brasil com Machine Learning (RF/XGBoost) e XAI",
        "Random Forest e XGBoost para a relacao ENSO-Brasil por pixel, regiao e bioma.",
        "Lags e distribuicao espacial-temporal; augmentation ainda sem decisao.",
    ),
    ProjectPhase(
        7,
        "convlstm_cycle",
        "Ciclo ENSO com redes neurais ConvLSTM",
        "ConvLSTM para prever antecipadamente a evolucao das fases e a faixa de pico.",
        "Estudo independente; data augmentation somente se necessario e restrito ao treino.",
    ),
    ProjectPhase(
        8,
        "convlstm_brazil_teleconnection",
        "Distribuicao no Brasil com redes neurais ConvLSTM",
        "ConvLSTM para a distribuicao espacial-temporal do sinal no Brasil.",
        "Estudo independente por pixel, regiao e bioma; augmentation ainda sem decisao.",
    ),
    ProjectPhase(
        None,
        "faseweb_publication_operation",
        "Publicacao e operacao",
        "Publicacao, painel e operacao recorrente da previsao antecipada da faixa de pico.",
        "Cada produto declara sua fase de origem; nao ha fusao automatica entre fases.",
        "FaseWEB",
    ),
)


PHASES_BY_NUMBER = {phase.number: phase for phase in PHASES if phase.number is not None}
PHASES_BY_SLUG = {phase.slug: phase for phase in PHASES}


def phase_table_rows() -> list[list[str]]:
    return [[phase.label, phase.title, phase.focus, phase.milestone] for phase in PHASES]
