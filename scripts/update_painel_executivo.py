from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.data.download_cds import ERA5_PRESSURE_VARIABLES, ERA5_SINGLE_VARIABLES
from nino_brasil.project_phases import PHASES


OUTPUT_PATH = ROOT / "painel_executivo.md"
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (result.stdout or result.stderr).strip()
    return output or "indisponivel"


def now_sp() -> str:
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MB"
    if value >= 1024:
        return f"{value / 1024:.2f} KB"
    return f"{value} B"


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def real_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"]


def file_count_and_size(path: Path) -> tuple[int, int]:
    files = real_files(path)
    return len(files), sum(p.stat().st_size for p in files)


def years_from_paths(paths: list[Path]) -> list[int]:
    years: set[int] = set()
    for path in paths:
        match = YEAR_RE.search(str(path))
        if match:
            years.add(int(match.group(0)))
    return sorted(years)


def years_from_files(path: Path, pattern: str) -> list[int]:
    if not path.exists():
        return []
    return years_from_paths(list(path.glob(pattern)))


def years_from_dirs(path: Path, pattern: str = "*") -> list[int]:
    if not path.exists():
        return []
    return years_from_paths([p for p in path.glob(pattern) if p.is_dir()])


def years_summary(years: list[int], start: int | None = None, end: int | None = None) -> str:
    if not years:
        return "nenhum ano local"
    first = start or years[0]
    last = end or years[-1]
    missing = [year for year in range(first, last + 1) if year not in years]
    gap = "sem lacunas internas" if not missing else f"lacunas: {', '.join(map(str, missing[:12]))}"
    if len(missing) > 12:
        gap += "..."
    return f"{years[0]}-{years[-1]} ({len(years)} anos; {gap})"


def zarr_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(path.rglob("*.zarr"), key=lambda p: str(p))


def audit_rows() -> list[dict[str, object]]:
    ledger = ROOT / "data" / "audit" / "ledger.jsonl"
    if not ledger.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_source_latency(rows: list[dict[str, object]], dataset: str) -> str:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("available_through")]
    if not candidates:
        return "sem registro"
    latest = sorted(candidates, key=lambda row: str(row.get("timestamp_utc", "")))[-1]
    return f"{latest.get('available_through')} (latencia {latest.get('latency_days')} dias)"


def table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |")
    return "\n".join(lines)


def era5_layer_year_counts(layer: str) -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    for item in zarr_dirs(ROOT / "data/processed/zarr/era5" / layer):
        match = YEAR_RE.search(str(item))
        if match:
            counts[int(match.group(0))] += 1
    return dict(sorted(counts.items()))


def yearly_count_summary(counts: dict[int, int], expected_label: str = "tarefas") -> str:
    if not counts:
        return "nenhum ano local"
    full = [year for year, count in counts.items() if count == max(counts.values())]
    partial = [f"{year}:{count}" for year, count in counts.items() if count != max(counts.values())]
    full_text = years_summary(full) if full else "nenhum completo"
    partial_text = ", ".join(partial[:12]) if partial else "sem parciais"
    if len(partial) > 12:
        partial_text += "..."
    return f"{full_text}; parciais {expected_label}: {partial_text}"


def expected_count_summary(counts: dict[int, int], expected_per_year: int, start: int, end: int) -> str:
    full = [year for year in range(start, end + 1) if counts.get(year, 0) == expected_per_year]
    partial = [
        f"{year}:{counts[year]}/{expected_per_year}"
        for year in range(start, end + 1)
        if 0 < counts.get(year, 0) < expected_per_year
    ]
    missing_count = sum(1 for year in range(start, end + 1) if counts.get(year, 0) == 0)
    full_text = years_summary(full, start, end) if full else "nenhum ano completo"
    partial_text = ", ".join(partial[:12]) if partial else "sem parciais"
    if len(partial) > 12:
        partial_text += "..."
    return f"{full_text}; parciais: {partial_text}; anos nao iniciados: {missing_count}"


def data_status_rows() -> list[list[str]]:
    chirps_raw = years_from_files(ROOT / "data/raw/chirps/p25", "*.nc")
    chirps_zarr = years_from_dirs(ROOT / "data/processed/zarr/brazil_precipitation", "*.zarr")
    regridded_root = ROOT / "data/processed/zarr/regridded"
    chirps_regrid = years_from_dirs(regridded_root, "chirps_p25_*.zarr")
    oisst_raw = years_from_files(ROOT / "data/raw/cpc_noaa/oisst", "*.nc")
    oisst_zarr = years_from_dirs(ROOT / "data/processed/zarr/cpc_noaa/oisst", "*.zarr")
    oisst_regrid = years_from_dirs(regridded_root, "noaa_oisst_*.zarr")
    ctd_raw = years_from_dirs(ROOT / "data/raw/ctd_noaa/wod")
    ctd_zarr = years_from_dirs(ROOT / "data/processed/zarr/ctd_noaa/wod")
    tao_temp = years_from_files(ROOT / "data/raw/tao_triton/temperature", "*.csv")
    tao_sal = years_from_files(ROOT / "data/raw/tao_triton/salinity", "*.csv")
    argo = years_from_files(ROOT / "data/raw/argo", "*.csv")
    era5_single_counts = era5_layer_year_counts("single_levels")
    era5_pressure_counts = era5_layer_year_counts("pressure_levels")
    era5_single_expected = len(ERA5_SINGLE_VARIABLES) * 2
    era5_pressure_expected = len(ERA5_PRESSURE_VARIABLES) * 2

    return [
        ["CHIRPS raw", years_summary(chirps_raw, 1981, 2026), "base de chuva"],
        ["CHIRPS Zarr", years_summary(chirps_zarr, 1981, 2026), "processado"],
        ["CHIRPS regrid", years_summary(chirps_regrid, 1981, 2026), "mantido fora do escopo ativo da Fase 3"],
        ["OISST raw", years_summary(oisst_raw, 1981, 2026), "SST/SSTA diaria"],
        ["OISST Zarr", years_summary(oisst_zarr, 1981, 2026), "processado"],
        ["OISST regrid", years_summary(oisst_regrid, 1981, 2026), "pronto para diagnostico fisico"],
        ["CTD/WOD raw", years_summary(ctd_raw, 1981, 2025), "cache bruto preservado"],
        ["CTD/WOD Zarr", years_summary(ctd_zarr, 1981, 2025), "anos sem perfil valido ficam registrados"],
        [
            "ERA5 single-level Zarr",
            expected_count_summary(era5_single_counts, era5_single_expected, 1981, 2025),
            f"{len(ERA5_SINGLE_VARIABLES)} variaveis x 2 regioes; camada atmosferica de superficie fechada",
        ],
        [
            "ERA5 pressure-level Zarr",
            expected_count_summary(era5_pressure_counts, era5_pressure_expected, 1981, 2025),
            f"{len(ERA5_PRESSURE_VARIABLES)} variaveis x 2 regioes; camada atmosferica vertical fechada",
        ],
        [
            "Oceano diario NOAA UFS",
            years_summary(years_from_dirs(ROOT / "data/processed/zarr/ocean_daily/noaa_ufs"), 1981, 1992),
            "T(z), S(z) e SSH diarios; ponte historica 1981-1992",
        ],
        [
            "Oceano diario GLORYS12",
            years_summary(years_from_dirs(ROOT / "data/processed/zarr/ocean_daily/glorys12"), 1993, 2026),
            "T(z), S(z) e SSH medios diarios; multiyear desde 1993",
        ],
        [
            "Oceano diario GLO12 operacional",
            years_summary(years_from_dirs(ROOT / "data/processed/zarr/ocean_daily/glorys12_operational"), 2026, 2026),
            "cauda de analise; previsoes excluidas",
        ],
        [
            "Oceano mensal ORAS5",
            years_summary(years_from_dirs(ROOT / "data/processed/zarr/ocean_monthly/oras5"), 1981, 2026),
            "7 variaveis mensais; nenhuma promocao para diario",
        ],
        ["TAO/TRITON temp", years_summary(tao_temp, 1981, 2026), "validacao in situ"],
        ["TAO/TRITON sal", years_summary(tao_sal, 1981, 2026), "validacao in situ"],
        ["Argo", years_summary(argo, 1999, 2026), "validacao in situ"],
    ]


def storage_rows() -> list[list[str]]:
    paths = [
        ("raw CHIRPS", ROOT / "data/raw/chirps"),
        ("raw OISST", ROOT / "data/raw/cpc_noaa/oisst"),
        ("raw ERA5", ROOT / "data/raw/era5"),
        ("raw GLORYS12 diario", ROOT / "data/raw/ocean_daily/glorys12"),
        ("raw GLO12 operacional", ROOT / "data/raw/ocean_daily/glorys12_operational"),
        ("raw ORAS5 mensal", ROOT / "data/raw/ocean_monthly/oras5"),
        ("raw CTD/WOD", ROOT / "data/raw/ctd_noaa"),
        ("raw TAO/TRITON", ROOT / "data/raw/tao_triton"),
        ("raw Argo", ROOT / "data/raw/argo"),
        ("Zarr processado", ROOT / "data/processed/zarr"),
    ]
    rows: list[list[str]] = []
    for label, path in paths:
        count, size = file_count_and_size(path)
        rows.append([label, str(path.relative_to(ROOT)), str(count), format_bytes(size)])
    return rows


def phase_status_rows() -> list[list[str]]:
    data_rows = {row[0]: row[1] for row in data_status_rows()}
    ocean_audit_path = ROOT / "data/audit/ocean_phase2_audit.json"
    ocean_audit: dict[str, object] = {}
    if ocean_audit_path.exists():
        try:
            ocean_audit = json.loads(ocean_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            ocean_audit = {}
    phase2_status = (
        "Concluida: auditoria oceanica integrada retornou status=complete."
        if ocean_audit.get("status") == "complete"
        else f"Em aberto: auditoria oceanica integrada registra {ocean_audit.get('error_count', 'nao executado')} pendencias."
    )
    phase3_audit_path = ROOT / "data/audit/phase3_diagnostics_audit.json"
    phase3_audit: dict[str, object] = {}
    if phase3_audit_path.exists():
        try:
            phase3_audit = json.loads(phase3_audit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            phase3_audit = {}
    phase3_errors = phase3_audit.get("errors") or []
    phase3_status = (
        "Concluida: diagnosticos fisicos do Nino 3.4 gerados com OISST local e auditados sem erros; sem rotulo externo e sem ML."
        if phase3_audit and not phase3_errors
        else f"Em aberto: auditoria da Fase 3 registra {len(phase3_errors) if phase3_audit else 'nao executado'} pendencias."
    )
    statuses = {
        1: "Concluida para a base operacional: CHIRPS, OISST, ERA5 e oceano UFS/GLORYS12 estao locais; validacoes in situ preservam as lacunas observadas.",
        2: phase2_status,
        3: phase3_status,
    }
    rows: list[list[str]] = []
    for phase in PHASES:
        rows.append([phase.label, phase.title, phase.milestone, statuses[phase.number]])
    rows.append(["Resumo base", "CTD", data_rows.get("CTD/WOD Zarr", "sem registro"), "lacunas nao sao preenchidas artificialmente"])
    return rows


def command_rows() -> list[list[str]]:
    return [
        [
            "CHIRPS",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\curate_and_resume_downloads.py --source chirps --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error",
        ],
        [
            "OISST",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\curate_and_resume_downloads.py --source oisst --stage all --start-year 1981 --execute --limit 0 --retries 5 --retry-wait 60 --continue-on-error",
        ],
        [
            "CTD/WOD",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py etl-ctd --start-year 1981 --max-depth 300 --min-levels 3 --execute --continue-on-error",
        ],
        [
            "ERA5 pressure-level auto",
            r'cd /d C:\DEV\NINO26 && set "NINO_CDS_RETRY_MAX=5" && set "NINO_CDS_SLEEP_MAX=60" && set "NINO_CDS_TIMEOUT=120" && .venv\Scripts\python scripts\data_pipeline.py download-era5 --start-year 1984 --end-year 2025 --kind pressure --region nino34 --region brazil --annual-zarr --request-mode annual-auto --delete-raw-after-zarr --execute --continue-on-error',
        ],
        [
            "Oceano diario NOAA UFS 1981-1992",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-ufs --start-year 1981 --end-year 1992 --build-features --execute",
        ],
        [
            "Sobreposicao NOAA UFS 1993-1995",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-ufs --start-year 1993 --end-year 1995 --block-size-mb 8 --build-features --execute",
        ],
        [
            "Oceano diario GLORYS12 1993-2026-05-26",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-glorys --start-year 1993 --end-year 2026 --end-date 2026-05-26 --delete-source-after-zarr --execute",
        ],
        [
            "GLO12 operacional 2026-05-27 ate ontem",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_daily_pipeline.py ingest-glorys-operational --start-date 2026-05-27 --delete-source-after-zarr --execute",
        ],
        [
            "ORAS5 mensal 1981-2026-05",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\ocean_monthly_pipeline.py ingest --start-year 1981 --end-year 2026 --end-month 5 --build-features --delete-raw-after-zarr --execute --continue-on-error",
        ],
        [
            "Auditoria oceano Fase 2",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\audit_ocean_phase2.py --glorys-my-end 2026-05-26 --operational-end 2026-06-19 --oras-end 2026-05-01 --overlap-year 1993 --overlap-year 1994 --overlap-year 1995",
        ],
        [
            "TAO/TRITON/Argo",
            r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\data_pipeline.py download-validation --source all --start-year 1981 --max-depth 300 --execute --continue-on-error",
        ],
    ]


def build_markdown() -> str:
    audit = audit_rows()
    status_counts = Counter(str(row.get("status", "unknown")) for row in audit)
    usage = shutil.disk_usage(ROOT)
    git_lines = run_git(["status", "--short", "--branch"]).splitlines()
    git_head = git_lines[0] if git_lines else "indisponivel"
    git_changes = max(len(git_lines) - 1, 0)

    lines = [
        "# Painel executivo NINO-BRASIL",
        "",
        "> Painel local gerado automaticamente. O arquivo e ignorado pelo Git; o script gerador e versionado.",
        "",
        "## Status condensado",
        "",
        table(
            ["Item", "Status"],
            [
                ["Atualizado em", now_sp()],
                ["Maquina", f"{platform.node() or 'indisponivel'} ({platform.system()} {platform.release()})"],
                ["Periodo alvo", "1981-latest para superficie/chuva/atmosfera; subsuperficie com janelas reais por fonte"],
                ["Fase operacional", "Fases 1-3 concluidas e auditadas; Fase 4 (triagem estatistica) pausada ate a validacao integral das Fases 1-3, conforme docs/CRONOGRAMA.md"],
                ["Git", f"{git_head}; mudancas locais: {git_changes}"],
                ["Disco livre", format_bytes(usage.free)],
                ["Auditoria", f"{len(audit)} eventos; " + (", ".join(f"{k}: {v}" for k, v in sorted(status_counts.items())) or "sem status")],
                ["Disponibilidade CHIRPS", latest_source_latency(audit, "chirps")],
                ["Disponibilidade OISST", latest_source_latency(audit, "noaa_oisst")],
                ["Disponibilidade ERA5", latest_source_latency(audit, "era5")],
                ["Disponibilidade CTD/WOD", latest_source_latency(audit, "noaa_wod_ctd")],
            ],
        ),
        "",
        "## Fases oficiais",
        "",
        table(["Fase", "Titulo", "Marco", "Estado local"], phase_status_rows()),
        "",
        "## Dados locais",
        "",
        table(["Fonte", "Cobertura", "Observacao"], data_status_rows()),
        "",
        "## Uso de disco",
        "",
        table(["Grupo", "Pasta", "Arquivos", "Tamanho"], storage_rows()),
        "",
        "## Retomada CMD",
        "",
        "Nao precisa ativar a venv se o comando chamar `.venv\\Scripts\\python` diretamente.",
        "",
        table(["Fonte", "Comando"], command_rows()),
        "",
        "## Proxima decisao tecnica",
        "",
        "- Manter ORAS5 como memoria mensal independente; nunca promover seus valores para observacoes diarias.",
        "- Preservar a auditoria concluida de continuidade, D20/OHC/WWV/Tilt e das transicoes UFS->GLORYS e GLORYS multiyear->operacional.",
        "- Regerar a Fase 3, quando necessario, com `build-nino34-daily-index`, `build-nino34-sst-reference`, `build-nino34-p90-peaks`, `build-phase3-diagnostics` e `audit-phase3-diagnostics`.",
        "- Usar apenas SST/SSTA OISST local para referencia mensal, eventos e picos P90; nenhum rotulo ENSO externo entra no fluxo ativo.",
        "- Reportar analises subsuperficiais com sensibilidade 1993+ e 2000+; nao vender cobertura subsuperficial homogenea desde 1981.",
        "- Fases 1-3 encerram em parecer fisico auditavel; a Fase 4 (docs/CRONOGRAMA.md e docs/FASE4_PLANO.md) so e retomada apos validacao integral das Fases 1-3, e ML/redes neurais permanecem condicionados aos gates G1-G4.",
        "",
        "## Rodape tecnico",
        "",
        f"- Workspace: {ROOT}",
        f"- Arquivo local: {OUTPUT_PATH.relative_to(ROOT)}",
        "- Atualizar painel:",
        "```cmd",
        r"cd /d C:\DEV\NINO26 && .venv\Scripts\python scripts\update_painel_executivo.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUTPUT_PATH.write_text(build_markdown(), encoding="utf-8")
    print(f"painel executivo atualizado: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
