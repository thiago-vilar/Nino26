from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

REPORT_DIR = ROOT / "data" / "state" / "pipeline_reports"


def ensure_processed_output_roots() -> None:
    """Recria os diretórios contratuais removidos durante uma limpeza local."""
    for relative in ("data/processed/figures", "data/processed/numeric-tables"):
        (ROOT / relative).mkdir(parents=True, exist_ok=True)

from nino_brasil.config import load_config
from nino_brasil.data.availability import iter_available_years, requested_end_date
from nino_brasil.data.download_cds import (
    ATMOSPHERE_AREAS,
    ERA5_PRESSURE_VARIABLES,
    ERA5_SINGLE_VARIABLES,
)


@dataclass
class StepResult:
    name: str
    command: list[str]
    returncode: int
    started_at: str
    finished_at: str


def now_label() -> str:
    try:
        return datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def timestamp_slug() -> str:
    try:
        return datetime.now(ZoneInfo("America/Sao_Paulo")).strftime("%Y%m%d_%H%M%S")
    except Exception:
        return datetime.now().strftime("%Y%m%d_%H%M%S")


def python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, str(ROOT / script), *args]


def month_args(months: list[int] | None) -> list[str]:
    values: list[str] = []
    for month in months or []:
        values.extend(["--month", str(month)])
    return values


def maybe_end_year(end_year: int | None) -> list[str]:
    return ["--end-year", str(end_year)] if end_year is not None else []


def single_year(year: int) -> list[str]:
    return ["--start-year", str(year), "--end-year", str(year)]


def safe_step_name(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_").lower()


def years_for(source: str, start_year: int, end_year: int | None) -> list[int]:
    return list(iter_available_years(start_year, end_year, source, load_config()))


def complete_years_for(source: str, start_year: int, end_year: int | None) -> list[int]:
    cfg = load_config()
    resolved_end = requested_end_date(cfg, source, end_year).date()
    final_year = resolved_end.year if resolved_end >= date(resolved_end.year, 12, 31) else resolved_end.year - 1
    if final_year < start_year:
        return []
    return list(range(start_year, final_year + 1))


def selected_variables(requested: list[str] | None, allowed: list[str]) -> list[str]:
    if not requested:
        return list(allowed)
    return [variable for variable in requested if variable in allowed]


def variable_args(variables: list[str], enabled: bool) -> list[str]:
    args: list[str] = []
    if enabled:
        for variable in variables:
            args.extend(["--variable", variable])
    return args


def validate_variable_filters(args: argparse.Namespace) -> None:
    if args.era5_variable and args.era5_kind == "single" and not selected_variables(args.era5_variable, ERA5_SINGLE_VARIABLES):
        raise ValueError(f"Nenhuma variavel solicitada pertence ao ERA5 single levels: {args.era5_variable}")
    if args.era5_variable and args.era5_kind == "pressure" and not selected_variables(args.era5_variable, ERA5_PRESSURE_VARIABLES):
        raise ValueError(f"Nenhuma variavel solicitada pertence ao ERA5 pressure levels: {args.era5_variable}")


def build_steps(args: argparse.Namespace) -> list[tuple[str, list[str]]]:
    ensure_processed_output_roots()
    validate_variable_filters(args)
    common_years = ["--start-year", str(args.start_year), *maybe_end_year(args.end_year)]
    core_limit = ["--limit", "0"]
    retry = ["--retries", str(args.retries), "--retry-wait", str(args.retry_wait)]
    fast = ["--fast"] if args.fast_check else []
    keep_going = ["--continue-on-error"] if args.continue_on_error else []
    months = args.month or list(range(1, 13))
    regions = args.region or list(ATMOSPHERE_AREAS)
    era5_single_variables = selected_variables(args.era5_variable, ERA5_SINGLE_VARIABLES)
    era5_pressure_variables = selected_variables(args.era5_variable, ERA5_PRESSURE_VARIABLES)

    # CHIRPS/OISST são curados em uma única execução global abaixo. Criar um
    # subprocesso por ano/estágio fazia a retomada validar centenas de vezes os
    # mesmos produtos que já estavam íntegros.
    source_years: dict[str, set[int]] = {"chirps": set(), "noaa_oisst": set()}
    if args.include_cds:
        cds_years = years_for if args.month else complete_years_for
        source_years["era5"] = set(cds_years("era5", args.start_year, args.end_year))
    if args.include_ctd:
        source_years["noaa_wod_ctd"] = set(years_for("noaa_wod_ctd", args.start_year, args.end_year))

    if not args.full_history:
        # Atualização recorrente: fontes remotas pesadas processam somente o
        # último ano elegível. A auditoria global de CHIRPS/OISST continua
        # cobrindo todo o histórico e reconstrói apenas suas lacunas reais.
        for source in ("era5", "noaa_wod_ctd"):
            available = source_years.get(source, set())
            if available:
                source_years[source] = {max(available)}

    steps: list[tuple[str, list[str]]] = [
        (
            "01_curadoria_core_incremental",
            python_cmd(
                "scripts/curate_and_resume_downloads.py",
                "--source",
                "core",
                "--stage",
                "all" if args.include_transforms else "raw",
                *common_years,
                "--chirps-resolution",
                args.chirps_resolution,
                "--execute",
                *core_limit,
                *retry,
                *fast,
                *keep_going,
            ),
        ),
        (
            "02_auditoria_ibge",
            python_cmd("scripts/audit_ibge_boundaries.py"),
        ),
    ]

    for year in sorted(source_years.get("chirps", set())):
        year_args = single_year(year)
        for stage in ["raw", *(["zarr", "regrid"] if args.include_transforms else [])]:
            steps.append(
                (
                    f"fonte_chirps_ano_{year}_{stage}",
                    python_cmd(
                        "scripts/curate_and_resume_downloads.py",
                        "--source",
                        "chirps",
                        "--stage",
                        stage,
                        *year_args,
                        "--chirps-resolution",
                        args.chirps_resolution,
                        "--execute",
                        *core_limit,
                        *retry,
                        *fast,
                        *keep_going,
                    ),
                )
            )

    for year in sorted(source_years.get("noaa_oisst", set())):
        year_args = single_year(year)
        for stage in ["raw", *(["zarr", "regrid"] if args.include_transforms else [])]:
            steps.append(
                (
                    f"fonte_oisst_ano_{year}_{stage}",
                    python_cmd(
                        "scripts/curate_and_resume_downloads.py",
                        "--source",
                        "oisst",
                        "--stage",
                        stage,
                        *year_args,
                        "--execute",
                        *core_limit,
                        *retry,
                        *fast,
                        *keep_going,
                    ),
                )
            )

    if args.include_cds:
        for year in sorted(source_years.get("era5", set())):
            year_args = single_year(year)
            if args.month:
                for month in months:
                    for region in regions:
                        if args.era5_kind in {"single", "both"} and (not args.era5_variable or era5_single_variables):
                            era5_vars = variable_args(era5_single_variables, bool(args.era5_variable))
                            steps.append(
                                (
                                    f"fonte_era5_ano_{year}_mes_{month:02d}_single_{safe_step_name(region)}",
                                    python_cmd(
                                        "scripts/data_pipeline.py",
                                        "download-era5",
                                        *year_args,
                                        "--month",
                                        str(month),
                                        "--kind",
                                        "single",
                                        "--region",
                                        region,
                                        *era5_vars,
                                        "--execute",
                                        *keep_going,
                                    ),
                                )
                            )
                        if args.era5_kind in {"pressure", "both"} and (not args.era5_variable or era5_pressure_variables):
                            era5_vars = variable_args(era5_pressure_variables, bool(args.era5_variable))
                            steps.append(
                                (
                                    f"fonte_era5_ano_{year}_mes_{month:02d}_pressure_{safe_step_name(region)}",
                                    python_cmd(
                                        "scripts/data_pipeline.py",
                                        "download-era5",
                                        *year_args,
                                        "--month",
                                        str(month),
                                        "--kind",
                                        "pressure",
                                        "--region",
                                        region,
                                        *era5_vars,
                                        "--execute",
                                        *keep_going,
                                    ),
                                )
                            )
            else:
                for region in regions:
                    if args.era5_kind in {"single", "both"} and (not args.era5_variable or era5_single_variables):
                        era5_vars = variable_args(era5_single_variables, bool(args.era5_variable))
                        steps.append(
                            (
                                f"fonte_era5_ano_{year}_single_{safe_step_name(region)}_zarr_anual",
                                python_cmd(
                                    "scripts/data_pipeline.py",
                                    "download-era5",
                                    *year_args,
                                    "--kind",
                                    "single",
                                    "--region",
                                    region,
                                    "--annual-zarr",
                                    "--request-mode",
                                    "annual-variable",
                                    "--delete-raw-after-zarr",
                                    *era5_vars,
                                    "--execute",
                                    *keep_going,
                                ),
                            )
                        )
                    if args.era5_kind in {"pressure", "both"} and (not args.era5_variable or era5_pressure_variables):
                        era5_vars = variable_args(era5_pressure_variables, bool(args.era5_variable))
                        steps.append(
                            (
                                f"fonte_era5_ano_{year}_pressure_{safe_step_name(region)}_zarr_anual",
                                python_cmd(
                                    "scripts/data_pipeline.py",
                                    "download-era5",
                                    *year_args,
                                    "--kind",
                                    "pressure",
                                    "--region",
                                    region,
                                    "--annual-zarr",
                                    "--request-mode",
                                    "annual-variable",
                                    "--delete-raw-after-zarr",
                                    *era5_vars,
                                    "--execute",
                                    *keep_going,
                                ),
                            )
                        )

    if args.include_ctd:
        for year in sorted(source_years.get("noaa_wod_ctd", set())):
            year_args = single_year(year)
            steps.append(
                (
                    f"fonte_ctd_wod_ano_{year}",
                    python_cmd(
                        "scripts/data_pipeline.py",
                        "download-ctd",
                        *year_args,
                        "--execute",
                        *keep_going,
                    ),
                )
            )

    steps.append(
        (
            "fonte_oras5_mensal_incremental",
            python_cmd(
                "scripts/ocean_monthly_pipeline.py",
                "ingest",
                "--start-year",
                str(args.start_year),
                "--build-features",
                "--delete-raw-after-zarr",
                "--execute",
                *keep_going,
            ),
        )
    )

    steps.append(
        (
            "auditoria_oras5_mensal",
            python_cmd(
                "scripts/ocean_monthly_pipeline.py",
                "audit",
                "--start-year",
                str(args.start_year),
            ),
        )
    )

    steps.extend(
        [
            (
                "zz_atualizar_painel",
                python_cmd("scripts/update_painel_executivo.py"),
            ),
            (
                "zz_curadoria_final",
                python_cmd(
                "scripts/curate_and_resume_downloads.py",
                "--source",
                    "core",
                    "--stage",
                    "all",
                    *common_years,
                    *fast,
                ),
            ),
        ]
    )
    return steps


def run_step(name: str, command: list[str], log_path: Path) -> StepResult:
    started = now_label()
    print()
    print("=" * 88)
    print(f"INICIANDO {name}")
    print(" ".join(command))
    print("=" * 88)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n===== {name} =====\n")
        log.write(f"started_at: {started}\n")
        log.write("command: " + " ".join(command) + "\n\n")
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        returncode = process.wait()
        finished = now_label()
        log.write(f"\nfinished_at: {finished}\n")
        log.write(f"returncode: {returncode}\n")
    return StepResult(name=name, command=command, returncode=returncode, started_at=started, finished_at=finished)


def write_report(report_path: Path, log_path: Path, results: list[StepResult]) -> None:
    failed = [result for result in results if result.returncode != 0]
    lines = [
        "# Relatorio do pipeline completo de download",
        "",
        f"- Gerado em: {now_label()}",
        f"- Workspace: {ROOT}",
        f"- Log completo: {log_path}",
        f"- Status geral: {'FALHOU' if failed else 'OK'}",
        "",
        "## Resolucao temporal por fonte",
        "",
        "- Ordem de execucao: primeiro por fonte, depois por ano; ERA5 por regiao e tipo.",
        "- Prioridade 1, fontes diarias ou subdiarias convertidas para diario: CHIRPS, NOAA OISST e ERA5.",
        "- Prioridade 2, fonte semanal: nenhuma fonte semanal ativa no catalogo atual.",
        "- Prioridade 3: oceano originalmente diario via scripts/ocean_daily_pipeline.py.",
        "- Fora da serie regular diaria: CTD/WOD, observacional irregular por perfil.",
        "- Politica de cache: bruto em data/raw, depois produto diario em data/processed/zarr; ERA5 consolidado em Zarr anual por regiao/tipo.",
        "",
        "## Etapas",
        "",
        "| Etapa | Codigo | Inicio | Fim |",
        "|---|---:|---|---|",
    ]
    for result in results:
        lines.append(f"| {result.name} | {result.returncode} | {result.started_at} | {result.finished_at} |")
    if failed:
        first = failed[0]
        lines.extend(
            [
                "",
                "## Primeiro bloqueio",
                "",
                f"- Etapa: {first.name}",
                f"- Codigo de saida: {first.returncode}",
                f"- Comando: `{' '.join(first.command)}`",
                "",
                "Abra o log completo para ver a mensagem imediatamente anterior a falha.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Resultado",
                "",
                "Todas as etapas selecionadas terminaram com codigo 0.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the complete NINO-BRASIL download/ETL pipeline in order.")
    parser.add_argument("--start-year", type=int, default=1981)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--chirps-resolution", choices=["p25", "p05"], default="p25")
    parser.set_defaults(include_cds=True, include_ctd=True)
    parser.add_argument("--skip-cds", dest="include_cds", action="store_false", help="Skip ERA5.")
    parser.add_argument("--skip-ctd", dest="include_ctd", action="store_false", help="Skip WOD CTD download+ETL.")
    parser.add_argument("--include-transforms", action="store_true", default=True, help="Convert CHIRPS/OISST to Zarr and regrid after raw downloads.")
    parser.add_argument("--no-transforms", dest="include_transforms", action="store_false")
    parser.add_argument("--month", type=int, action="append", choices=range(1, 13), help="Limit CDS ERA5 to selected months.")
    parser.add_argument("--era5-kind", choices=["single", "pressure", "both"], default="both")
    parser.add_argument("--region", action="append", choices=sorted(ATMOSPHERE_AREAS), help="Limit ERA5 to one region; repeat for many.")
    parser.add_argument(
        "--era5-variable",
        action="append",
        choices=sorted(set(ERA5_SINGLE_VARIABLES + ERA5_PRESSURE_VARIABLES)),
        help="Debug only: limit ERA5 to selected variables. Default: all variables grouped by kind.",
    )
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--retry-wait", type=int, default=60)
    parser.add_argument("--fast-check", action="store_true", help="Use existence checks instead of opening metadata in curation steps.")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Audita/reprocessa todo o histórico remoto de ERA5 e CTD. O padrão é a atualização incremental do último ano elegível.",
    )
    parser.add_argument("--execute", action="store_true", help="Actually run the planned steps. Without this, only prints the plan.")
    parser.add_argument("--report-only", action="store_true", help="Deprecated alias for the default plan-only mode.")
    parser.add_argument("--print-plan-limit", type=int, default=60, help="Maximum planned steps printed to the terminal.")
    parser.set_defaults(continue_on_error=True)
    parser.add_argument("--stop-on-error", dest="continue_on_error", action="store_false", help="Stop at the first failed step.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    steps = build_steps(args)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    slug = timestamp_slug()
    log_path = REPORT_DIR / f"full_download_pipeline_{slug}.log"
    report_path = REPORT_DIR / f"full_download_pipeline_{slug}.md"

    print("Pipeline completo planejado:")
    print(f"- Total de etapas: {len(steps)}")
    print("- Ordem: fonte por fonte; ERA5 por ano/variavel, com Zarr anual.")
    for index, (name, command) in enumerate(steps[: args.print_plan_limit], start=1):
        print(f"{index:02d}. {name}: {' '.join(command)}")
    if len(steps) > args.print_plan_limit:
        print(f"... mais {len(steps) - args.print_plan_limit} etapas omitidas da pre-visualizacao.")

    if args.report_only or not args.execute:
        print()
        print("Modo plano: nada foi executado.")
        print("Para executar de verdade, rode novamente com --execute.")
        return 0

    results: list[StepResult] = []
    for name, command in steps:
        result = run_step(name, command, log_path)
        results.append(result)
        if result.returncode != 0 and not args.continue_on_error:
            print()
            print(f"Parando no primeiro bloqueio: {name} retornou {result.returncode}.")
            break

    write_report(report_path, log_path, results)
    print()
    print(f"Relatorio final: {report_path}")
    print(f"Log completo: {log_path}")
    return 1 if any(result.returncode != 0 for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
