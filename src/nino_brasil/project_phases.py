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
        "QC, unidades/sinais, climatologia sem vazamento, anomalias, lags e grades de preditores; alvos CHIRPS preservam pixels nativos.",
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
        "CHIRPS nativo semanal, anomalia/percentil/SPI/extremos, lags por EN/LN x fase e testes pixel-a-pixel com N_eff/FDR/campo.",
        "Mapas e tabelas estatisticas da teleconexao Pacifico -> Brasil, sem ML/RN.",
    ),
    ProjectPhase(
        5,
        "ml_cycle_rfxgb_xai",
        "Ciclo ENSO com Machine Learning (RF/XGBoost) e XAI",
        "Mesmo mecanismo da Fase 3 (genese, crescimento, pico, decaimento) com Random "
        "Forest e XGBoost: identifica ciclos EN/LN, mapeia as 4 fases, seleciona variaveis "
        "por RFECV e projeta pico/tempo-para-pico/duracao com XAI (SHAP, PDP).",
        "Execucao independente sobre dados das Fases 1 e 2; comparacoes historicas sao opcionais.",
    ),
    ProjectPhase(
        6,
        "ml_brazil_teleconnection_xai",
        "Distribuicao no Brasil com Machine Learning (RF/XGBoost) e XAI",
        "Mesmo estudo espaco-temporal da Fase 4 com RF/XGBoost em cada pixel CHIRPS "
        "original, 31 variaveis, fase no tempo fonte e XAI OOS; regioes/biomas sao resumos.",
        "Execucao independente sobre dados das Fases 1 e 2; comparacoes historicas sao opcionais.",
    ),
    ProjectPhase(
        7,
        "convlstm_cycle",
        "Ciclo ENSO com redes neurais ConvLSTM",
        "Mesmo mecanismo das Fases 3 e 5 com PyTorch ConvLSTM: aprende a evolucao espaco-temporal "
        "do Pacifico equatorial, identifica ciclos EN/LN, mapeia as 4 fases e ranqueia "
        "variaveis por etapa com XAI.",
        "Pode ser executada diretamente a partir das Fases 1 e 2, sem gates intermediarios.",
    ),
    ProjectPhase(
        8,
        "convlstm_brazil_teleconnection",
        "Distribuicao no Brasil com redes neurais ConvLSTM",
        "Mesmo estudo das Fases 4 e 6 com PyTorch ConvLSTM probabilistica no grid CHIRPS nativo: projeta a influencia "
        "do El Nino/La Nina sobre a chuva do Brasil no espaco e no tempo, por fase, regiao "
        "e bioma.",
        "Pode ser executada diretamente a partir das Fases 1 e 2, sem gates intermediarios.",
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
