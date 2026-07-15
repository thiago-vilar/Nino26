#!/usr/bin/env python3
"""Valida, sem executar, os 31 notebooks-fonte canônicos do NINO26."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import nbformat
import pandas as pd
from jsonschema import ValidationError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from nino_brasil.notebook_catalog import CANONICAL_NOTEBOOKS

OPERATIONAL_NOTEBOOKS = {"notebooks/fase1/F1A_atualizacao_dados.ipynb"}


BASE_REQUIRED_SECTIONS = (
    "**DADOS**",
    "**RESULTADOS**",
    "**CONCLUSÕES**",
)

SCIENTIFIC_INTRO_SECTIONS = (
    "**TÍTULO**",
    "**CONTEXTO**",
    "**MOTIVAÇÃO**",
    "**METODOLOGIA**",
    "**RESULTADOS ESPERADOS**",
)
FINAL_REFERENCE_SECTION = "**REFERÊNCIAS BIBLIOGRÁFICAS**"
FIRST_NOTEBOOK_CODES = {"F2Z", "F3NinoA", "F4NinoC", "F5A", "F6A", "F7A", "F8A"}
WSL_COMMAND_HEADING = "**COMANDO WSL2 — EXECUTAR FASE COMPLETA**"


def _text(cell: dict) -> str:
    source = cell.get("source", [])
    return source if isinstance(source, str) else "".join(source)


def _problem(rows: list[dict[str, object]], path: str, kind: str, detail: object) -> None:
    rows.append({"notebook": path, "type": kind, "detail": detail})


def validate() -> pd.DataFrame:
    problems: list[dict[str, object]] = []
    canonical = {
        spec.relative_path.replace("\\", "/"): spec for spec in CANONICAL_NOTEBOOKS
    }
    discovered = {
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in (ROOT / "notebooks").glob("fase*/*.ipynb")
    }
    for extra in sorted(discovered - set(canonical) - OPERATIONAL_NOTEBOOKS):
        _problem(problems, extra, "legacy_notebook_active", "mover para quarentena")
    for missing in sorted(set(canonical) - discovered):
        _problem(problems, missing, "missing_canonical_notebook", "ausente")

    for relative in sorted(OPERATIONAL_NOTEBOOKS):
        path = ROOT / relative
        if not path.is_file():
            _problem(problems, relative, "missing_operational_notebook", "ausente")
            continue
        notebook = json.loads(path.read_text(encoding="utf-8"))
        nbformat.validate(nbformat.from_dict(notebook))
        cells = notebook.get("cells", [])
        intro = _text(cells[0]) if cells else ""
        positions = [intro.find(section) for section in SCIENTIFIC_INTRO_SECTIONS]
        if not cells or cells[0].get("cell_type") != "markdown" or any(p < 0 for p in positions) or positions != sorted(positions):
            _problem(problems, relative, "intro_cell_contract", "primeira célula explicativa inválida")
        if not intro.startswith(WSL_COMMAND_HEADING) or intro.find(WSL_COMMAND_HEADING) > intro.find("**TÍTULO**"):
            _problem(problems, relative, "wsl_phase_command", "comando WSL2 deve anteceder TÍTULO")
        operational_markdown = "\n".join(_text(cell) for cell in cells if cell.get("cell_type") == "markdown")
        if re.search(r"(?m)^#{1,6}\s+", operational_markdown):
            _problem(problems, relative, "large_markdown_heading", "usar maiúsculas/negrito/itálico")
        if not cells or cells[-1].get("cell_type") != "markdown" or not _text(cells[-1]).lstrip().startswith(FINAL_REFERENCE_SECTION):
            _problem(problems, relative, "references_cell_contract", FINAL_REFERENCE_SECTION)

    for relative, spec in canonical.items():
        path = ROOT / relative
        if not path.is_file():
            continue
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
            nbformat.validate(nbformat.from_dict(notebook))
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            _problem(problems, relative, "invalid_notebook", str(exc).splitlines()[0])
            continue

        cells = notebook.get("cells", [])
        if not cells or cells[0].get("cell_type") != "markdown":
            _problem(problems, relative, "intro_cell_position", "primeira célula deve ser Markdown")
        else:
            intro = _text(cells[0])
            positions = [intro.find(section) for section in SCIENTIFIC_INTRO_SECTIONS]
            if any(position < 0 for position in positions) or positions != sorted(positions):
                _problem(problems, relative, "intro_cell_contract", "seções obrigatórias ausentes ou fora de ordem")
        if not cells or cells[-1].get("cell_type") != "markdown":
            _problem(problems, relative, "references_cell_position", "última célula deve ser Markdown")
        elif not _text(cells[-1]).lstrip().startswith(FINAL_REFERENCE_SECTION):
            _problem(problems, relative, "references_cell_contract", FINAL_REFERENCE_SECTION)
        cell_ids = [str(cell.get("id", "")).strip() for cell in cells]
        if any(not value for value in cell_ids):
            _problem(problems, relative, "missing_cell_id", "todas as células")
        duplicate_ids = sorted({value for value in cell_ids if cell_ids.count(value) > 1})
        if duplicate_ids:
            _problem(problems, relative, "duplicate_cell_id", ",".join(duplicate_ids))

        markdown = "\n".join(
            _text(cell) for cell in cells if cell.get("cell_type") == "markdown"
        )
        if re.search(r"(?m)^#{1,6}\s+", markdown):
            _problem(problems, relative, "large_markdown_heading", "usar maiúsculas/negrito/itálico")
        if spec.code in FIRST_NOTEBOOK_CODES:
            intro = _text(cells[0]) if cells else ""
            if not intro.startswith(WSL_COMMAND_HEADING) or intro.find(WSL_COMMAND_HEADING) > intro.find("**TÍTULO**"):
                _problem(problems, relative, "wsl_phase_command", "comando WSL2 deve anteceder TÍTULO")
        source = "\n".join(
            _text(cell) for cell in cells if cell.get("cell_type") == "code"
        )
        # Todo notebook começa com o protocolo científico completo; resumos
        # editoriais abreviados são proibidos pelo contrato do projeto.
        required_sections = list(BASE_REQUIRED_SECTIONS)
        required_sections.extend(SCIENTIFIC_INTRO_SECTIONS)
        required_sections.append(FINAL_REFERENCE_SECTION)
        for section in required_sections:
            if markdown.count(section) != 1:
                _problem(problems, relative, "section_contract", section)

        metadata = notebook.get("metadata", {}).get("nino26", {})
        expected_metadata = {
            "canonical": True,
            "notebook_code": spec.code,
            "figure_precode": f"Fig{spec.code}",
            "table_precode": f"Tab{spec.code}",
            "default_run_pipeline": "0",
        }
        for key, expected in expected_metadata.items():
            if metadata.get(key) != expected:
                _problem(problems, relative, "metadata_contract", f"{key}={metadata.get(key)!r}")

        if f"Fig{spec.code}1" not in markdown or f"Tab{spec.code}1" not in markdown:
            _problem(problems, relative, "artifact_precode", spec.code)
        if "NotebookWorkflow" not in source:
            _problem(problems, relative, "workflow_missing", "NotebookWorkflow")
        if "NINO26_RUN_PIPELINE" not in source or "raise RuntimeError" not in source:
            _problem(problems, relative, "pipeline_guard", "viewer deve recusar núcleo pesado")
        if "np.random" in source or "random.rand" in source:
            _problem(problems, relative, "synthetic_placeholder", "dados aleatórios")
        forbidden_summary = "tl" + ";dr"
        if forbidden_summary in markdown.casefold():
            _problem(problems, relative, "forbidden_editorial_section", forbidden_summary)

        for number, cell in enumerate(cells):
            if cell.get("cell_type") != "code":
                continue
            if (
                not spec.compact_source
                and (cell.get("execution_count") is not None or cell.get("outputs", []))
            ):
                _problem(problems, relative, "source_output_not_clean", f"célula {number}")

        version = str(notebook.get("metadata", {}).get("language_info", {}).get("version", ""))
        if version and not version.startswith("3.12"):
            _problem(problems, relative, "python_version", version)

    return pd.DataFrame(problems, columns=["notebook", "type", "detail"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)
    problems = validate()
    if problems.empty:
        print("[ok] notebooks canônicos e operacionais coerentes com seus contratos.")
        return 0
    print(problems.to_string(index=False))
    return 1 if args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
