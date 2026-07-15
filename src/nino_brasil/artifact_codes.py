"""Canonical notebook, figure and table codes for NINO-BRASIL.

Public analytical artifacts inherit the complete code of the notebook that
publishes them.  Examples::

    F3NinoA       -> FigF3NinoA1 / TabF3NinoA1
    F3NinaA       -> FigF3NinaA1 / TabF3NinaA1
    F4NinoC       -> FigF4NinoC1 / TabF4NinoC1
    F5A           -> FigF5A1 / TabF5A1

Descriptive slugs may follow the code (``TabF3NinoA1_resumo.csv``), but the
pre-code is immutable and is the audit key shared by notebook, figure and
table.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


NOTEBOOK_CODE_RE = re.compile(
    r"^F(?P<phase>[2-8])(?P<signal>Nino|Nina)?(?P<block>[A-Z])$"
)
ARTIFACT_CODE_RE = re.compile(
    r"^(?P<kind>Fig|Tab)"
    r"(?P<notebook>F[2-8](?:Nino|Nina)?[A-Z])"
    r"(?P<ordinal>[1-9][0-9]*)"
    r"(?:_(?P<slug>[a-z0-9]+(?:_[a-z0-9]+)*))?$"
)


@dataclass(frozen=True)
class NotebookCode:
    value: str
    phase: int
    signal: str | None
    block: str

    @property
    def namespace(self) -> str:
        return f"F{self.phase}{self.signal or ''}"

    @property
    def enso_type(self) -> str | None:
        return {"Nino": "el_nino", "Nina": "la_nina"}.get(self.signal)


@dataclass(frozen=True)
class ArtifactCode:
    value: str
    kind: str
    notebook: NotebookCode
    ordinal: int
    slug: str | None

    @property
    def precode(self) -> str:
        return f"{self.kind}{self.notebook.value}{self.ordinal}"

    @property
    def paired_precode(self) -> str:
        paired_kind = "Tab" if self.kind == "Fig" else "Fig"
        return f"{paired_kind}{self.notebook.value}{self.ordinal}"


def parse_notebook_code(value: str) -> NotebookCode:
    match = NOTEBOOK_CODE_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(
            f"codigo de notebook invalido {value!r}; use F2Z, F3NinoA, "
            "F3NinaA, F4NinoC, F5A, ..."
        )
    phase = int(match.group("phase"))
    signal = match.group("signal")
    block = match.group("block")
    if phase in {3, 4} and signal is None:
        raise ValueError(f"{value!r}: F3/F4 exigem Nino ou Nina no codigo")
    if phase not in {3, 4} and signal is not None:
        raise ValueError(f"{value!r}: somente F3/F4 aceitam Nino/Nina no codigo")
    return NotebookCode(value=str(value).strip(), phase=phase, signal=signal, block=block)


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value))
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")


def artifact_code(
    kind: str,
    notebook_code: str,
    ordinal: int,
    *,
    slug: str | None = None,
) -> str:
    if kind not in {"Fig", "Tab"}:
        raise ValueError("kind deve ser Fig ou Tab")
    notebook = parse_notebook_code(notebook_code)
    if int(ordinal) < 1:
        raise ValueError("ordinal deve ser >= 1")
    clean_slug = slugify(slug) if slug else ""
    return (
        f"{kind}{notebook.value}{int(ordinal)}"
        + (f"_{clean_slug}" if clean_slug else "")
    )


def figure_code(notebook_code: str, ordinal: int, *, slug: str | None = None) -> str:
    return artifact_code("Fig", notebook_code, ordinal, slug=slug)


def table_code(notebook_code: str, ordinal: int, *, slug: str | None = None) -> str:
    return artifact_code("Tab", notebook_code, ordinal, slug=slug)


def parse_artifact_code(value: str) -> ArtifactCode:
    match = ARTIFACT_CODE_RE.fullmatch(str(value).strip())
    if not match:
        raise ValueError(
            f"codigo de artefato invalido {value!r}; use FigF3NinoA1 ou "
            "TabF3NinaA1_resumo"
        )
    notebook = parse_notebook_code(match.group("notebook"))
    return ArtifactCode(
        value=str(value).strip(),
        kind=match.group("kind"),
        notebook=notebook,
        ordinal=int(match.group("ordinal")),
        slug=match.group("slug"),
    )


def assert_artifact_pair(figure: str, table: str) -> None:
    parsed_figure = parse_artifact_code(figure)
    parsed_table = parse_artifact_code(table)
    if parsed_figure.kind != "Fig" or parsed_table.kind != "Tab":
        raise ValueError("o par deve ser informado como figura e tabela")
    if (
        parsed_figure.notebook.value != parsed_table.notebook.value
        or parsed_figure.ordinal != parsed_table.ordinal
    ):
        raise ValueError(
            f"pre-codigos divergentes: {parsed_figure.precode} != "
            f"{parsed_table.precode}"
        )


def notebook_code_for(phase: int, block: str, enso_type: str | None = None) -> str:
    signal = {None: "", "el_nino": "Nino", "la_nina": "Nina"}.get(enso_type)
    if signal is None:
        raise ValueError("enso_type deve ser el_nino, la_nina ou None")
    return parse_notebook_code(f"F{int(phase)}{signal}{str(block).upper()}").value
