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
        "Escala diaria/semanal; eventos e P90 derivados da propria SST OISST baixada; diagnosticos de SSTA, D20, OHC, WWV, termoclina, DHW e Kelvin.",
        "Parecer fisico auditavel da evolucao do Nino 3.4, sem rotulo externo e sem modelagem/ML.",
    ),
)


PHASES_BY_NUMBER = {phase.number: phase for phase in PHASES}
PHASES_BY_SLUG = {phase.slug: phase for phase in PHASES}


def phase_table_rows() -> list[list[str]]:
    return [[phase.label, phase.title, phase.focus, phase.milestone] for phase in PHASES]
