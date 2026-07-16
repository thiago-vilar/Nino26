#!/usr/bin/env python3
"""Generate the clean canonical F2-F8 notebooks from the project catalog.

The generator never executes analysis.  It writes source notebooks with empty
outputs, deterministic metadata and the public Fig/Tab pre-code contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.artifact_codes import figure_code, parse_notebook_code, table_code
from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS, NotebookSpec

FIRST_NOTEBOOK_CODES = {"F2Z", "F3NinoA", "F4NinoC", "F5A", "F6A", "F7A", "F8A"}
WSL_PHASE_COMMANDS = {
    2: "make fase2",
    3: "make fase3",
    4: "make fase4",
    5: "make fase5",
    6: "make fase6",
    7: "make fase7",
    8: "make fase8",
}


def assumptions(spec: NotebookSpec) -> list[str]:
    parsed = parse_notebook_code(spec.code)
    rows = [
        "Toda conclusão deve nascer de uma tabela `Tab...` persistida antes da figura correspondente.",
        "A execução é determinística e usa somente entradas declaradas no inventário mostrado abaixo.",
        "Notebook executado não implica hipótese confirmada; gates científicos são exibidos como resultados.",
    ]
    if parsed.phase == 3:
        rows.extend(
            [
                "El Niño e La Niña são analisados isoladamente; nenhum composto mistura os dois sinais.",
                "Evento é a unidade independente; semanas descrevem a trajetória interna do evento.",
            ]
        )
    if parsed.phase in {4, 6, 8}:
        rows.append(
            "O alvo CHIRPS conserva `pixel_id`, latitude e longitude da grade nativa; não há interpolação do alvo."
        )
    if parsed.phase in {5, 6, 7, 8}:
        rows.append(
            "Pré-processamento, pesos e augmentation são ajustados somente no treino de cada fold."
        )
    return rows


def build_notebook(spec: NotebookSpec):
    parsed = parse_notebook_code(spec.code)
    fig_example = figure_code(spec.code, 1)
    tab_example = table_code(spec.code, 1)
    assumption_markdown = "\n".join(f"- {item}" for item in assumptions(spec))
    context_text = spec.context or (
        f"Este notebook integra a Fase {parsed.phase} do projeto NINO-BRASIL e responde a uma "
        "pergunta delimitada com entradas, método e saídas rastreáveis. O resultado deve ser lido "
        "como evidência da hipótese declarada, respeitando o escopo temporal, espacial e estatístico "
        "registrado no inventário de dados."
    )
    hypothesis_statement = spec.hypothesis_statement or (
        f"A hipótese {spec.hypothesis} é que a pergunta — {spec.question.rstrip('?')} — pode ser "
        "avaliada com as entradas declaradas e o método abaixo; resultados nulos ou insuficientes "
        "permanecem conclusões válidas."
    )
    method_rationale = spec.method_rationale or (
        "O teste foi escolhido para manter a unidade de análise e o encadeamento temporal da fase, "
        "sem transformar observações correlacionadas em réplicas independentes. Os produtos "
        "numéricos são persistidos antes das figuras para permitir auditoria e reprodução."
    )
    expected_outputs = spec.expected_outputs or (
        "tabela numérica auditável que responda à pergunta científica",
        "figura correspondente, com título, unidades e escopo explícitos",
        "conclusões e limitações derivadas apenas das saídas executadas",
    )
    references = spec.references or (
        "NINO-BRASIL. Documentação metodológica, contrato de dados e diretrizes da fase no repositório do projeto.",
    )
    expected_markdown = "\n".join(f"- {item}." for item in expected_outputs)
    references_markdown = "\n".join(
        f"{index}. {reference}" for index, reference in enumerate(references, 1)
    )
    signal = parsed.enso_type or ""
    study_object = {
        "el_nino": "ciclo de vida do El Niño no Pacífico tropical",
        "la_nina": "ciclo de vida da La Niña no Pacífico tropical",
    }.get(signal, "processo físico definido para esta etapa do projeto")
    command_prefix = ""
    if spec.code in FIRST_NOTEBOOK_CODES:
        command_prefix = (
            "**COMANDO WSL2 — EXECUTAR FASE COMPLETA**\n\n"
            "```bash\n" + WSL_PHASE_COMMANDS[parsed.phase] + "\n```\n\n"
        )
    introduction = (
        command_prefix
        + f"**TÍTULO**\n\n{spec.title}\n\n"
        "**Projeto:** NINO-BRASIL — Oceanografia Física — UFPE  \n"
        f"**Código canônico:** `{spec.code}`  \n"
        f"**Objeto de estudo:** {study_object}  \n"
        f"**Família de hipótese:** `{spec.hypothesis}`\n\n"
        "**CONTEXTO**\n\n"
        f"{context_text}\n\n"
        "**PERGUNTA CIENTÍFICA**\n\n"
        f"{spec.question}\n\n"
        "**MOTIVAÇÃO**\n\n*Hipótese específica*\n\n"
        f"{hypothesis_statement}\n\n"
        "A hipótese poderá ser sustentada, parcialmente sustentada ou rejeitada; "
        "a execução do notebook não antecipa o resultado.\n\n"
        "*Função dos testes e unidade de análise*\n\n"
        f"{method_rationale}\n\n"
        "**METODOLOGIA**\n\n"
        f"{spec.method}\n\n"
        "**RESULTADOS ESPERADOS**\n\n"
        "Resultados esperados significam produtos necessários para responder à pergunta, "
        "não valores ou significâncias presumidos:\n\n"
        f"{expected_markdown}\n\n"
        f"- figuras públicas iniciadas por `Fig{spec.code}`;\n"
        f"- tabelas públicas iniciadas por `Tab{spec.code}`;\n"
        f"- primeiro par reservado: `{fig_example}` ↔ `{tab_example}`;\n"
        "- toda interpretação deve apontar para tabela, run_id, unidade, amostra e limitações.\n\n"
        "*Fundamentação científica mínima*\n\n"
        "As referências completas utilizadas estão na última célula do notebook."
    )
    assumptions_cell = (
        "**PREMISSAS DE VALIDADE E LIMITES DE INTERPRETAÇÃO**\n\n"
        f"{assumption_markdown}"
    )
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3.12 (.venv NINO26)",
            "language": "python",
            "name": "nino-brasil",
        },
        "language_info": {"name": "python", "version": "3.12"},
        "nino26": {
            "artifact_role": "canonical-analysis-report",
            "canonical": True,
            "notebook_code": spec.code,
            "enso_type": signal,
            "default_run_pipeline": "0",
            "execution_policy": "numeric-core-first-viewer-publisher",
            "persist_inline_outputs": bool(spec.compact_source),
            "figure_precode": f"Fig{spec.code}",
            "table_precode": f"Tab{spec.code}",
        },
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            introduction
        ),
        nbf.v4.new_markdown_cell(
            assumptions_cell
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import os\n"
            "import sys\n"
            "from IPython.display import display\n\n"
            "ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'pyproject.toml').exists())\n"
            "SRC = ROOT / 'src'\n"
            "if str(SRC) not in sys.path:\n"
            "    sys.path.insert(0, str(SRC))\n"
            "NOTEBOOK_CODE = '" + spec.code + "'\n"
            "MODE = os.environ.get('NINO26_NOTEBOOK_MODE', 'official')\n"
            "RUN_PIPELINE = os.environ.get('NINO26_RUN_PIPELINE', '0') == '1'\n"
            "if RUN_PIPELINE:\n"
            "    raise RuntimeError('O núcleo numérico deve ser executado pelo runner da fase, antes do notebook.')\n"
            "from nino_brasil.notebook_workflows import NotebookWorkflow\n"
            "workflow = NotebookWorkflow(ROOT, NOTEBOOK_CODE, mode=MODE)\n"
            "workflow.describe()"
        ),
        nbf.v4.new_markdown_cell("**DADOS**"),
        nbf.v4.new_code_cell(
            "input_inventory = workflow.input_inventory()\n"
            "display(input_inventory)\n"
            "workflow.require_inputs()"
        ),
        nbf.v4.new_markdown_cell("**RESULTADOS**"),
        nbf.v4.new_code_cell(
            "result = workflow.run()\n"
            "display(result.artifacts)\n"
            "display(result.summary)"
        ),
        nbf.v4.new_markdown_cell("**FIGURAS PARA VERIFICAÇÃO RÁPIDA**"),
        nbf.v4.new_code_cell(
            "from IPython.display import Image, display\n\n"
            "for artifact in result.artifacts.itertuples(index=False):\n"
            "    display(Image(filename=str(artifact.figure_path)))"
        ),
        nbf.v4.new_markdown_cell("**CONCLUSÕES**"),
        nbf.v4.new_code_cell(
            "for item in result.takeaways:\n"
            "    print(f'- {item}')\n"
            "print('\\nLimitações:')\n"
            "for item in result.limitations:\n"
            "    print(f'- {item}')"
        ),
        nbf.v4.new_markdown_cell(
            "**REFERÊNCIAS BIBLIOGRÁFICAS**\n\n" + references_markdown
        ),
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    return notebook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="grava os 31 notebooks canônicos")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="grava somente códigos com este prefixo; repetível (ex.: F3Nino)",
    )
    args = parser.parse_args(argv)
    selected = [
        spec
        for spec in CANONICAL_NOTEBOOKS
        if not args.only or any(spec.code.startswith(prefix) for prefix in args.only)
    ]
    if not selected:
        parser.error("--only não corresponde a nenhum notebook canônico")
    for spec in selected:
        destination = ROOT / spec.relative_path
        print(destination.relative_to(ROOT))
        if not args.write:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        nbf.write(build_notebook(spec), destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
