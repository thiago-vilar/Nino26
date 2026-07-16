#!/usr/bin/env python3
"""Gera o notebook operacional F1A de atualização das fontes do NINO26."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/fase1/F1A_atualizacao_dados.ipynb"


def build_notebook():
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3.12 (.venv NINO26)",
            "language": "python",
            "name": "nino-brasil",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "nino26": {
            "canonical": True,
            "notebook_code": "F1A",
            "artifact_role": "data-update-operator",
            "execution_requires": "NINO26_UPDATE_DATA=1",
        },
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "**COMANDO WSL2 — EXECUTAR FASE COMPLETA**\n\n"
            "```bash\ncd /mnt/c/DEV/NINO26 && .venv-wsl/bin/python scripts/run_full_download_pipeline.py --execute --stop-on-error\n```\n\n"
            "**TÍTULO**\n\n"
            "F1A — Atualização e ingestão das fontes do NINO26\n\n"
            "**CONTEXTO**\n\n"
            "A Fase 1 é responsável por baixar, atualizar, catalogar e auditar as fontes locais. "
            "Este notebook organiza OISST, ERA5, CHIRPS, UFS+GLORYS, CTD/WOD e as malhas "
            "IBGE de estados, regiões e biomas. Ele não trata nem interpreta os dados; essa função "
            "pertence à Fase 2 e às fases analíticas.\n\n"
            "**MOTIVAÇÃO**\n\n"
            "Hipótese operacional: a base local pode ser atualizada de forma retomável, mantendo "
            "procedência, cobertura real e falhas explícitas. O modo padrão é apenas diagnóstico; "
            "downloads só são iniciados quando `NINO26_UPDATE_DATA=1`.\n\n"
            "**METODOLOGIA**\n\n"
            "O notebook chama o pipeline incremental da Fase 1, que verifica arquivos existentes, "
            "baixa somente pendências, respeita a latência de cada fonte e executa a auditoria das "
            "malhas IBGE. Cada comando registra código de saída e horário.\n\n"
            "**RESULTADOS ESPERADOS**\n\n"
            "- plano de atualização por fonte;\n"
            "- atualização retomável quando autorizada;\n"
            "- auditoria IBGE de 27 UFs, 5 regiões e 6 biomas;\n"
            "- inventário final com arquivos e datas locais;\n"
            "- contrato de saída para a Fase 2, sem executar tratamentos da F2."
        ),
        nbf.v4.new_markdown_cell(
            "**CONFIGURAÇÃO OPERACIONAL**\n\n"
            "Por segurança, a primeira execução apenas mostra o plano. Para atualizar de fato, "
            "defina `NINO26_UPDATE_DATA=1` no ambiente e execute novamente o notebook."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "from datetime import datetime, timezone\n"
            "import os\n"
            "import subprocess\n"
            "import sys\n"
            "import pandas as pd\n"
            "from IPython.display import display\n\n"
            "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())\n"
            "PYTHON = ROOT / '.venv' / 'Scripts' / 'python.exe'\n"
            "if not PYTHON.exists():\n"
            "    PYTHON = Path(sys.executable)\n"
            "EXECUTE_UPDATE = os.environ.get('NINO26_UPDATE_DATA', '0') == '1'\n"
            "print(f'Raiz: {ROOT}')\n"
            "print(f'Atualização autorizada: {EXECUTE_UPDATE}')\n"
            "print(f'Início UTC: {datetime.now(timezone.utc).isoformat()}')"
        ),
        nbf.v4.new_markdown_cell("**PLANO E ATUALIZAÇÃO DAS FONTES**"),
        nbf.v4.new_code_cell(
            "command = [str(PYTHON), 'scripts/run_full_download_pipeline.py']\n"
            "if EXECUTE_UPDATE:\n"
            "    command.extend(['--execute', '--stop-on-error'])\n"
            "print('Comando:', ' '.join(command))\n"
            "completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)\n"
            "print(completed.stdout[-12000:])\n"
            "if completed.stderr:\n"
            "    print(completed.stderr[-6000:])\n"
            "if completed.returncode:\n"
            "    raise RuntimeError(f'Pipeline F1 terminou com código {completed.returncode}')"
        ),
        nbf.v4.new_markdown_cell("**AUDITORIA DAS MALHAS IBGE**"),
        nbf.v4.new_code_cell(
            "audit_command = [str(PYTHON), 'scripts/audit_ibge_boundaries.py']\n"
            "audit = subprocess.run(audit_command, cwd=ROOT, text=True, capture_output=True, check=False)\n"
            "print(audit.stdout)\n"
            "if audit.returncode:\n"
            "    raise RuntimeError('A auditoria IBGE encontrou problemas.')\n"
            "ibge_report = ROOT / 'data/processed/parquet/statistics/phase1_ibge_boundaries_audit.csv'\n"
            "display(pd.read_csv(ibge_report))"
        ),
        nbf.v4.new_markdown_cell("**INVENTÁRIO LOCAL APÓS A ATUALIZAÇÃO**"),
        nbf.v4.new_code_cell(
            "roots = [ROOT/'data/raw', ROOT/'data/interim', ROOT/'data/processed/zarr']\n"
            "rows = []\n"
            "for base in roots:\n"
            "    if not base.exists():\n"
            "        continue\n"
            "    files = [p for p in base.rglob('*') if p.is_file()]\n"
            "    rows.append({'arvore': str(base.relative_to(ROOT)), 'arquivos': len(files), "
            "'ultima_modificacao_utc': max((datetime.fromtimestamp(p.stat().st_mtime, timezone.utc) for p in files), default=pd.NaT)})\n"
            "inventory = pd.DataFrame(rows)\n"
            "display(inventory)"
        ),
        nbf.v4.new_markdown_cell(
            "**REFERÊNCIAS BIBLIOGRÁFICAS**\n\n"
            "1. NOAA Climate Prediction Center. Oceanic Niño Index. "
            "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php\n"
            "2. NOAA OISST v2.1. https://www.ncei.noaa.gov/products/optimum-interpolation-sst\n"
            "3. Climate Hazards Center. CHIRPS. https://www.chc.ucsb.edu/data/chirps\n"
            "4. ECMWF. ERA5. https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5\n"
            "5. IBGE. Geociências e malhas territoriais. https://www.ibge.gov.br/geociencias/"
        ),
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(build_notebook(), OUTPUT)
    print(f"[gravado] {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
