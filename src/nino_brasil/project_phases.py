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
        "ml_cycle_rfxgb_xai",
        "Ciclo ENSO com Machine Learning (RF/XGBoost) e XAI",
        "Mesmo mecanismo da Fase 3 (genese, crescimento, pico, decaimento) com Random "
        "Forest e XGBoost: identifica ciclos EN/LN, mapeia as 4 fases, seleciona variaveis "
        "por RFECV e projeta pico/tempo-para-pico/duracao com XAI (SHAP, PDP).",
        "RF/XGBoost so avancam se superarem a caracterizacao estatistica da Fase 3 e os "
        "baselines de climatologia/persistencia.",
    ),
    ProjectPhase(
        6,
        "ml_brazil_teleconnection_xai",
        "Distribuicao no Brasil com Machine Learning (RF/XGBoost) e XAI",
        "Mesmo estudo espaco-temporal da Fase 4 (Pacifico -> anomalia de chuva no Brasil) "
        "com RF/XGBoost e XAI, por fase do ciclo, regiao IBGE e bioma; series semanais e "
        "diarias quando possivel.",
        "RF/XGBoost so avancam se superarem a triagem estatistica da Fase 4 e os baselines.",
    ),
    ProjectPhase(
        7,
        "convlstm_cycle",
        "Ciclo ENSO com redes neurais ConvLSTM",
        "Mesmo mecanismo das Fases 3 e 5 com ConvLSTM: aprende a evolucao espaco-temporal "
        "do Pacifico equatorial, identifica ciclos EN/LN, mapeia as 4 fases e ranqueia "
        "variaveis por etapa com XAI.",
        "A rede so se justifica se superar climatologia, persistencia, a Fase 3 e a Fase 5.",
    ),
    ProjectPhase(
        8,
        "convlstm_brazil_teleconnection",
        "Distribuicao no Brasil com redes neurais ConvLSTM",
        "Mesmo estudo espaco-temporal das Fases 4 e 6 com ConvLSTM: projeta a influencia "
        "do El Nino/La Nina sobre a chuva do Brasil no espaco e no tempo, por fase, regiao "
        "e bioma.",
        "A rede so se justifica se superar climatologia, persistencia, a Fase 4 e a Fase 6.",
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
