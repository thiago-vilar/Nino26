#!/usr/bin/env python3
"""Install the fail-closed viewer cells used by canonical F4-F8 notebooks."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source(text: str) -> list[str]:
    return (text.strip("\n") + "\n").splitlines(keepends=True)


def _set_code(cell: dict, text: str) -> None:
    if cell.get("cell_type") != "code":
        raise ValueError("Expected a code cell while standardizing a viewer notebook.")
    cell["source"] = _source(text)
    cell["execution_count"] = None
    cell["outputs"] = []


def _set_markdown(cell: dict, text: str) -> None:
    if cell.get("cell_type") != "markdown":
        raise ValueError("Expected a markdown cell while standardizing a viewer notebook.")
    cell["source"] = _source(text)


def _code_cell(text: str, *, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "id": cell_id,
        "execution_count": None,
        "metadata": {"tags": ["nino26-viewer"]},
        "outputs": [],
        "source": _source(text),
    }


def _markdown_cell(text: str, *, cell_id: str) -> dict:
    return {
        "cell_type": "markdown",
        "id": cell_id,
        "metadata": {"tags": ["nino26-viewer"]},
        "source": _source(text),
    }


def _write(path: Path, notebook: dict) -> None:
    path.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )


def standardize_f4c() -> None:
    path = ROOT / "notebooks/fase4/4C_sinal_pixel_lags.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if len(notebook.get("cells", [])) < 9:
        raise ValueError(f"Unexpected 4C notebook layout: {path}")
    _set_code(
        notebook["cells"][1],
        r'''
import os, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())
for candidate in (ROOT, ROOT/'src'):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
from scripts.notebook_run_viewer import audit_phase4_outputs, load_phase4_table

STATS = ROOT/'data/processed/parquet/statistics'
ENSO_TYPE = os.environ.get('NINO26_ENSO_TYPE', '').strip() or None
if ENSO_TYPE not in {None, 'el_nino', 'la_nina'}:
    raise ValueError(f'NINO26_ENSO_TYPE invÃ¡lido: {ENSO_TYPE!r}')
SCOPE_SUFFIX = f'_{ENSO_TYPE}' if ENSO_TYPE else ''
FIGS = ROOT/'data/processed/figures/fase4'/(ENSO_TYPE or '')
ATLAS = ROOT/f'data/processed/zarr/statistics/phase4C_native_pixel_lags{SCOPE_SUFFIX}.zarr'
PREDICTOR_TREATMENT = STATS/f'phase4C_native_predictor_treatment{SCOPE_SUFFIX}.csv'
FIELD_SIGNIFICANCE = STATS/f'phase4C_native_field_significance{SCOPE_SUFFIX}.csv'
BEST_LAG_KEY = STATS/f'phase4C_native_best_lag_pixel_key{SCOPE_SUFFIX}.csv'
F4C_OUTPUTS = [
    STATS/f'phase4C_native_lags_por_unidade{SCOPE_SUFFIX}.csv',
    STATS/f'phase4C_native_cobertura_unidades{SCOPE_SUFFIX}.parquet',
    PREDICTOR_TREATMENT,
    STATS/f'phase4C_native_best_lag_pixel{SCOPE_SUFFIX}.parquet',
    BEST_LAG_KEY,
    FIELD_SIGNIFICANCE,
    ATLAS,
]
''',
    )
    _set_code(
        notebook["cells"][2],
        r'''
# Viewer por padrão: o notebook nunca recalcula silenciosamente.
RUN_PIPELINE = os.environ.get('NINO26_RUN_PIPELINE', '0') == '1'
if RUN_PIPELINE:
    from scripts.run_fase4c_regional import main as run4c
    run_args = ['--field-permutations', '199', '--replace-existing']
    if ENSO_TYPE:
        run_args.extend(['--enso-type', ENSO_TYPE])
    run4c(run_args)
else:
    print('Modo viewer: validando numeric-tables F4C canônicas existentes.')
''',
    )
    _set_markdown(notebook["cells"][3], "## Auditoria, catálogo das 31 variáveis e gates numéricos")
    _set_code(
        notebook["cells"][4],
        r'''
audit_4c, run_id_4c = audit_phase4_outputs(
    F4C_OUTPUTS, canonical_f4c=True, expected_stage='F4C', expected_enso_type=ENSO_TYPE
)
display(audit_4c)
if run_id_4c is None:
    print('F4C canônica ausente ou inválida; nenhuma conclusão é exibida.')
else:
    print({'analysis_run_id': run_id_4c, 'artifacts_validated': len(F4C_OUTPUTS)})
    predictors = load_phase4_table(PREDICTOR_TREATMENT)
    print({'predictor_count': predictors['variavel'].nunique(),
           'selection_contract': sorted(predictors['selection_contract'].astype(str).unique())})
    display(predictors)
    field = load_phase4_table(FIELD_SIGNIFICANCE)
    display(field)
    key = load_phase4_table(BEST_LAG_KEY)
    display(key.head(40))
''',
    )
    _set_markdown(notebook["cells"][5], "## Figuras derivadas das numeric-tables validadas")
    _set_code(
        notebook["cells"][6],
        r'''
if run_id_4c is not None:
    figures = sorted(FIGS.glob('Fig_4C*.png'))
    if not figures:
        print('Nenhuma figura F4C canônica registrada para o run validado.')
    for figure in figures:
        print(figure.name)
        display(Image(filename=str(figure)))
''',
    )
    _write(path, notebook)


def standardize_f4d() -> None:
    path = ROOT / "notebooks/fase4/4D_clusters_alvo.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) < 5:
        raise ValueError(f"Unexpected 4D notebook layout: {path}")
    _set_code(
        cells[1],
        r'''
SEED = 42
MIN_SIGNIFICANT_PROFILES = 2
''',
    )
    _set_code(
        cells[2],
        r'''
import os, sys
from pathlib import Path
import pandas as pd
from IPython.display import Image, display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())
for candidate in (ROOT, ROOT/'src'):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
from scripts.notebook_run_viewer import audit_phase4_outputs, load_phase4_table

STATS = ROOT/'data/processed/parquet/statistics'
ENSO_TYPE = os.environ.get('NINO26_ENSO_TYPE', '').strip() or None
if ENSO_TYPE not in {None, 'el_nino', 'la_nina'}:
    raise ValueError(f'NINO26_ENSO_TYPE invÃ¡lido: {ENSO_TYPE!r}')
SCOPE_SUFFIX = f'_{ENSO_TYPE}' if ENSO_TYPE else ''
FIGS = ROOT/'data/processed/figures/fase4'/(ENSO_TYPE or '')
HYPOTHESIS_SUMMARY = STATS/f'phase4D_native_hypothesis_summary{SCOPE_SUFFIX}.csv'
GATE_EVENT_JACKKNIFE = STATS/f'phase4D_native_gate_event_jackknife{SCOPE_SUFFIX}.csv'
CLUSTER_RANKING = STATS/f'phase4D_native_cluster_ranking{SCOPE_SUFFIX}.csv'
F4D_OUTPUTS = [
    STATS/f'phase4D_native_clusters_pixels{SCOPE_SUFFIX}.parquet',
    STATS/f'phase4D_native_cluster_profiles{SCOPE_SUFFIX}.csv',
    CLUSTER_RANKING,
    GATE_EVENT_JACKKNIFE,
    HYPOTHESIS_SUMMARY,
    STATS/f'phase4D_native_target_coverage{SCOPE_SUFFIX}.csv',
]

RUN_PIPELINE = os.environ.get('NINO26_RUN_PIPELINE', '0') == '1'
if RUN_PIPELINE:
    from scripts.run_fase4d_targets import main as run4d
    run_args = ['--seed', str(SEED), '--min-significant-profiles', str(MIN_SIGNIFICANT_PROFILES)]
    if ENSO_TYPE:
        run_args.extend(['--enso-type', ENSO_TYPE])
    run4d(run_args)
else:
    print('Modo viewer: validando numeric-tables F4D existentes.')
''',
    )
    _set_markdown(cells[3], "## Tabelas, gate multi-alvo e figuras auditáveis")
    references = [
        cell
        for cell in cells
        if "NINO26-REFERENCIAS" in "".join(cell.get("source", []))
    ]
    if len(references) != 1:
        raise ValueError("4D reference cell not found.")
    reference = references[0]
    notebook["cells"] = cells[:4] + [
        _code_cell(
            r'''
audit_4d, run_id_4d = audit_phase4_outputs(
    F4D_OUTPUTS, expected_stage='F4D', expected_enso_type=ENSO_TYPE
)
display(audit_4d)
if run_id_4d is None:
    print('F4D canônica ausente ou inválida; nenhuma conclusão é exibida.')
else:
    print({'analysis_run_id': run_id_4d, 'artifacts_validated': len(F4D_OUTPUTS)})
    summary = load_phase4_table(HYPOTHESIS_SUMMARY)
    gate = load_phase4_table(GATE_EVENT_JACKKNIFE)
    ranking = load_phase4_table(CLUSTER_RANKING)
    display(summary)
    display(gate)
    display(ranking)
''',
            cell_id="f4d-audit-tables",
        ),
        _markdown_cell(
            "Uma área colorida não constitui significância. O suporte exige direção física, "
            "FDR confirmatório, significância de campo e estabilidade ao retirar cada evento inteiro.",
            cell_id="f4d-interpretation",
        ),
        _code_cell(
            r'''
if run_id_4d is not None:
    figures = sorted(FIGS.glob('Fig_4D*.png'))
    if not figures:
        print('Nenhuma figura F4D canônica registrada para o run validado.')
    for figure in figures:
        print(figure.name)
        display(Image(filename=str(figure)))
''',
            cell_id="f4d-audit-figures",
        ),
        reference,
    ]
    _write(path, notebook)


COMMON_SETUP = r'''
from pathlib import Path
import json, os, subprocess, sys
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'pyproject.toml').exists())
for candidate in (ROOT, ROOT/'src'):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
from scripts.notebook_run_viewer import audit_artifact_runs, load_declared_tables

PYTHON = sys.executable
MODE = os.environ.get('NINO26_NOTEBOOK_MODE', 'smoke')
RUN_PIPELINE = os.environ.get('NINO26_RUN_PIPELINE', '0') == '1'
assert MODE in {'smoke', 'official'}
print({'root': str(ROOT), 'python': PYTHON, 'mode': MODE, 'run_pipeline': RUN_PIPELINE})
'''


COMMON_VIEWER = r'''
audit_runs, selected_runs = audit_artifact_runs(ROOT, phase=PHASE, mode=MODE)
if audit_runs.empty:
    print('Nenhum run nesta modalidade. O notebook não fabrica resultados.')
else:
    display(audit_runs.drop(columns=['directory'], errors='ignore'))

REQUESTED = {
    5: ['scientific_gate.csv', 'fold_metrics.csv', 'event_dimension_metrics.csv',
        'independent_support_by_fold.csv', 'state_importance_oos.csv',
        'global_importance.csv'],
    6: ['field_gate.csv', 'fold_metrics.csv', 'pixel_metrics.csv',
        'pixel_variable_importance.csv'],
    7: ['scientific_gate.csv', 'fold_metrics.csv', 'event_probabilistic_metrics.csv',
        'independent_support_by_fold.csv',
        'scalar_variable_importance_oos.csv', 'spatial_channel_importance_oos.csv',
        'paired_f5_comparison.csv'],
    8: ['confirmatory_gate_by_condition.csv', 'fold_metrics.csv', 'pixel_metrics.csv',
        'input_importance_oos.csv'],
}
if not selected_runs:
    print('Nenhum ArtifactRun completo e íntegro foi selecionado.')
for run in selected_runs:
    manifest = run['manifest']
    summary = {
        'run_id': manifest['run_id'], 'phase': manifest['phase'], 'mode': manifest['mode'],
        'status': manifest['status'], 'model': (manifest.get('parameters') or {}).get('model', ''),
        'started_at': manifest.get('started_at'), 'finished_at': manifest.get('finished_at'),
        'notes': manifest.get('notes', ''),
    }
    display(pd.DataFrame([summary]))
    tables = load_declared_tables(run, REQUESTED[PHASE])
    for name, frame in tables.items():
        print(f'{name}: {len(frame)} linhas × {frame.shape[1]} colunas')
        display(frame if 'gate' in name else frame.head(40))
'''


def standardize_model_viewer(relative: str, phase: int, script_name: str) -> None:
    path = ROOT / relative
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = notebook.get("cells", [])
    if len(cells) < 10 or cells[2].get("cell_type") != "code" or cells[7].get("cell_type") != "code":
        raise ValueError(f"Unexpected model viewer layout: {path}")
    setup = COMMON_SETUP + f"\nPHASE = {phase}\nSCRIPT = ROOT/'scripts/{script_name}'\n"
    _set_code(cells[2], setup)
    _set_code(cells[7], COMMON_VIEWER)
    _write(path, notebook)


def main() -> int:
    standardize_f4c()
    standardize_f4d()
    standardize_model_viewer("notebooks/fase5/5_ciclo_ml.ipynb", 5, "run_fase5_cycle_ml.py")
    standardize_model_viewer("notebooks/fase6/6_brasil_ml.ipynb", 6, "run_fase6_brazil_ml.py")
    standardize_model_viewer("notebooks/fase7/7_ciclo_convlstm.ipynb", 7, "run_fase7_cycle_convlstm.py")
    standardize_model_viewer("notebooks/fase8/8_brasil_convlstm.ipynb", 8, "run_fase8_brazil_convlstm.py")
    print("Viewer notebooks F4-F8 standardized with fail-closed auditing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
