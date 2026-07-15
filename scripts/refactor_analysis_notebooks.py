#!/usr/bin/env python3
"""Mechanically rebuild critical notebooks as thin, auditable run reports.

Scientific logic belongs in tested modules/runners.  These notebooks expose
parameters, optionally execute a runner, and render only semantic run tables.
Re-running this script is idempotent and clears stale outputs/errors.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = "<!-- NINO26-CABECALHO v1 -->"
FOOTER = "<!-- NINO26-REFERENCIAS v1 -->"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "id": None, "metadata": {}, "source": text.splitlines(True)}


def code(text: str, *, parameters: bool = False) -> dict:
    metadata = {"tags": ["parameters"]} if parameters else {}
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": None,
        "metadata": metadata,
        "outputs": [],
        "source": text.splitlines(True),
    }


def source_text(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def rebuild(path: Path, body: list[dict]) -> None:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    header = next((cell for cell in notebook["cells"] if HEADER in source_text(cell)), None)
    footer = next((cell for cell in notebook["cells"] if FOOTER in source_text(cell)), None)
    if header is None or footer is None:
        raise ValueError(f"{path}: sentinelas de cabecalho/rodape ausentes")
    for position, cell in enumerate([header, *body, footer]):
        cell["id"] = f"nino26-{path.stem[:8]}-{position:02d}"
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    notebook["cells"] = [header, *body, footer]
    notebook.setdefault("metadata", {}).setdefault("kernelspec", {}).update(
        {"display_name": "Python 3.12 (NINO26)", "language": "python", "name": "python3"}
    )
    notebook["metadata"].setdefault("language_info", {}).update(
        {"name": "python", "version": "3.12"}
    )
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


COMMON_SETUP = """from pathlib import Path
import json, os, subprocess, sys
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())
PYTHON = sys.executable
MODE = os.environ.get('NINO26_NOTEBOOK_MODE', 'smoke')
RUN_PIPELINE = os.environ.get('NINO26_RUN_PIPELINE', '0') == '1'
assert MODE in {'smoke', 'official'}
print({'root': str(ROOT), 'python': PYTHON, 'mode': MODE, 'run_pipeline': RUN_PIPELINE})
"""


def model_body(phase: int, script: str, inputs: list[str], extra: str = "") -> list[dict]:
    input_literal = repr(inputs)
    return [
        markdown(
            "## Contrato de execução\n\n"
            "O notebook não contém treino duplicado. A implementação testada vive em `src/`; "
            "o runner grava `run_id`, hashes, folds, sementes, tabelas semânticas e modelos. "
            "Por padrão apenas inspeciona; defina `NINO26_RUN_PIPELINE=1` para executar.\n"
        ),
        code(COMMON_SETUP + f"\nPHASE={phase}\nSCRIPT=ROOT/'{script}'\n", parameters=True),
        code(
            f"required = [ROOT/path for path in {input_literal}]\n"
            "preflight = pd.DataFrame({'path':[str(p.relative_to(ROOT)) for p in required], "
            "'exists':[p.exists() for p in required]})\n"
            "display(preflight)\n"
            "if not preflight['exists'].all():\n"
            "    print('Pré-requisitos ausentes: execute as fases anteriores indicadas no runbook.')\n"
        ),
        markdown(
            "## Execução opcional\n\n"
            "`smoke` usa dados reais e orçamento reduzido, em diretório separado. "
            "Resultados científicos só podem vir de `official`; gates podem bloquear fases posteriores.\n"
        ),
        code(
            (extra + "\n" if extra else "")
            + "if RUN_PIPELINE:\n"
            + "    command = [PYTHON, str(SCRIPT), '--mode', MODE]\n"
            + "    print('executando:', ' '.join(command))\n"
            + "    subprocess.run(command, cwd=ROOT, check=True)\n"
            + "else:\n"
            + "    print(f'Para executar: NINO26_RUN_PIPELINE=1 (modo atual={MODE})')\n"
        ),
        markdown("## Último run e tabelas auditáveis\n"),
        code(
            "run_root = ROOT/'data/processed/runs'/MODE/f'fase{PHASE}'\n"
            "runs = sorted(run_root.glob(f'F{PHASE}_*'), reverse=True) if run_root.exists() else []\n"
            "if not runs:\n"
            "    print('Nenhum run nesta modalidade. O notebook não fabrica resultados.')\n"
            "else:\n"
            "    latest = runs[0]\n"
            "    manifest = json.loads((latest/'run_manifest.json').read_text(encoding='utf-8'))\n"
            "    display(pd.DataFrame([manifest]).drop(columns=['inputs','configs','environment','git'], errors='ignore'))\n"
            "    table_manifest = pd.read_csv(latest/'tables_manifest.csv')\n"
            "    display(table_manifest)\n"
            "    for table in ['fold_metrics.csv','field_gate.csv','pixel_metrics.csv','state_importance_oos.csv','input_importance_oos.csv']:\n"
            "        path = latest/'tables'/table\n"
            "        if path.exists():\n"
            "            print(table); display(pd.read_csv(path).head(30))\n"
        ),
        markdown(
            "## Regra de interpretação\n\n"
            "A quantidade de semanas/augmentation melhora a otimização, mas o número independente é o de "
            "eventos no teste. Nunca interpretar amostras sintéticas como novos El Niños/La Niñas.\n"
        ),
    ]


def phase3_body(kind: str) -> list[dict]:
    table_map = {
        "3B": [
            "phase3_events_en_ln.csv",
            "phase3_fases_semanais_en_ln.csv",
            "phase3_event_lifecycle_en_ln.csv",
            "phase3_peak_band_sensitivity.csv",
            "phase3_rolling_origin_targets.csv",
        ],
        "3E": [
            "phase3_lag_event_bootstrap_summary.csv",
            "phase3_lag_event_bootstrap_replicates.csv",
            "phase3_best_lags_fdr.csv",
        ],
        "3I": [
            "phase3_best_lags_fdr.csv",
            "phase3_discriminantes_por_periodo.csv",
            "phase3_rolling_origin_targets.csv",
            "phase3_rolling_origin_folds.csv",
        ],
        "3K": [
            "phase3_pca_por_fase.csv",
            "phase3_pca_loadings_por_fase.csv",
            "phase3_rolling_origin_folds.csv",
        ],
    }
    tables = table_map[kind]
    return [
        markdown(
            "## Contrato científico central\n\n"
            "As tabelas abaixo são geradas pelo executor F3 testado. Rótulos de fase/faixa de pico são "
            "retrospectivos; previsão usa apenas a tabela rolling-origin. O evento é a unidade independente.\n"
        ),
        code(COMMON_SETUP + "\nSTATS=ROOT/'data/processed/parquet/statistics'\n", parameters=True),
        code(
            "if RUN_PIPELINE:\n"
            "    command=[PYTHON, str(ROOT/'scripts/phase3_en_ln.py')]\n"
            "    if MODE == 'smoke': command.append('--quick')\n"
            "    subprocess.run(command, cwd=ROOT, check=True)\n"
        ),
        code(
            f"table_names={tables!r}\n"
            "base = STATS/'pilots/quick' if MODE == 'smoke' else STATS\n"
            "inventory=[]\n"
            "for name in table_names:\n"
            "    path=base/name\n"
            "    inventory.append({'table':name,'exists':path.exists(),'manifest':path.with_suffix('.manifest.json').exists()})\n"
            "display(pd.DataFrame(inventory))\n"
            "for name in table_names:\n"
            "    path=base/name\n"
            "    if path.exists():\n"
            "        print(name); display(pd.read_csv(path).head(30))\n"
        ),
        markdown(
            "## Interpretação permitida\n\n"
            "`diagnostico_retrospectivo` descreve gênese/crescimento/faixa de pico/decaimento após o evento. "
            "`rolling_origin_operacional` mede previsão. Resultados alinhados ao pico conhecido não são hindcasts operacionais.\n"
        ),
    ]


def main() -> int:
    rebuild(ROOT/'notebooks/fase3/3B_alvo_eventos_ciclo_vida.ipynb', phase3_body('3B'))
    rebuild(ROOT/'notebooks/fase3/3E_sensibilidade_temporal.ipynb', phase3_body('3E'))
    rebuild(ROOT/'notebooks/fase3/3I_interpretacao_integrada.ipynb', phase3_body('3I'))
    rebuild(ROOT/'notebooks/fase3/3K_pca_crescimento.ipynb', phase3_body('3K'))
    rebuild(
        ROOT/'notebooks/fase5/5_ciclo_ml.ipynb',
        model_body(
            5,
            'scripts/run_fase5_cycle_ml.py',
            ['data/processed/parquet/features/nino34_master_weekly.csv',
             'data/processed/parquet/statistics/phase3_fases_semanais_en_ln.csv'],
        ),
    )
    rebuild(
        ROOT/'notebooks/fase6/6_brasil_ml.ipynb',
        model_body(
            6,
            'scripts/run_fase6_brazil_ml.py',
            ['data/processed/zarr/features/chirps_native_weekly_targets.zarr',
             'data/processed/parquet/features/nino34_master_weekly.csv'],
        ),
    )
    rebuild(
        ROOT/'notebooks/fase7/7_ciclo_convlstm.ipynb',
        model_body(
            7,
            'scripts/run_fase7_cycle_convlstm.py',
            ['data/processed/zarr/modeling/phase7_pacific_weekly.zarr',
             'data/processed/parquet/features/nino34_master_weekly.csv'],
            extra=(
                "BUILD_SPATIAL_INPUT = os.environ.get('NINO26_BUILD_SPATIAL_INPUT','0') == '1'\n"
                "if RUN_PIPELINE and BUILD_SPATIAL_INPUT and not required[0].exists():\n"
                "    subprocess.run([PYTHON, str(ROOT/'scripts/build_phase7_pacific_cube.py')], cwd=ROOT, check=True)"
            ),
        ),
    )
    rebuild(
        ROOT/'notebooks/fase8/8_brasil_convlstm.ipynb',
        model_body(
            8,
            'scripts/run_fase8_brazil_convlstm.py',
            ['data/processed/zarr/modeling/phase7_pacific_weekly.zarr',
             'data/processed/zarr/features/chirps_native_weekly_targets.zarr'],
        ),
    )
    print('8 notebooks refatorados; outputs antigos removidos.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
