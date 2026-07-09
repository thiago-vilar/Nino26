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
        "QC, climatologia sem vazamento, anomalias, acumulados, percentis, lags e grade comum 0.25 grau.",
        "Cubos Zarr reconciliados em data/processed/zarr/regridded/.",
    ),
    ProjectPhase(
        3,
        "nino34_physical_signal_diagnostics",
        "Diagnostico fisico do sinal Nino 3.4",
        "Escala semanal; eventos El Nino/La Nina derivados da propria SST OISST baixada; diagnosticos de SSTA, D20, OHC, WWV, termoclina, Bjerknes, Kelvin, mapas e PCA/EOF.",
        "Caracterizacao auditavel de genese, crescimento, pico e decaimento do Nino 3.4, sem rotulo externo e sem ML/RN.",
    ),
    ProjectPhase(
        4,
        "enso_brazil_rainfall_teleconnection",
        "Teleconexao ENSO -> chuvas extremas/secas no Brasil",
        "CHIRPS semanal, anomalias de chuva, metrica P90 do periodo de aquecimento, lags semanais e testes pixel-a-pixel com N_eff/FDR.",
        "Mapas e tabelas estatisticas da teleconexao Pacifico -> Brasil, sem ML/RN.",
    ),
    ProjectPhase(
        5,
        "ml_rfxgb_teleconnection_xai",
        "Mesmo estudo da Fase 4 com ML e XAI",
        "Random Forest e XGBoost com explicabilidade; series semanais e diarias quando possivel; comparacao contra climatologia/persistencia.",
        "Modelos RF/XGBoost so avancam se superarem a triagem estatistica da Fase 4 e os baselines.",
    ),
    ProjectPhase(
        6,
        "native_neural_networks_xai",
        "Redes neurais nativas + XAI",
        "CNN espacial, memoria espaco-temporal e decoder de teleconexao, apenas se Fase 5 justificar a complexidade.",
        "Redes neurais precisam superar climatologia, persistencia, Fase 4 e Fase 5.",
    ),
    ProjectPhase(
        None,
        "faseweb_publication_operation",
        "Publicacao e operacao",
        "Painel/publicacao web e rotina de recalibracao recorrente consumindo saidas numericas das fases disponiveis.",
        "FaseWEB concentra publicacao, painel e operacao recorrente.",
        "FaseWEB",
    ),
)


PHASES_BY_NUMBER = {phase.number: phase for phase in PHASES if phase.number is not None}
PHASES_BY_SLUG = {phase.slug: phase for phase in PHASES}


def phase_table_rows() -> list[list[str]]:
    return [[phase.label, phase.title, phase.focus, phase.milestone] for phase in PHASES]
