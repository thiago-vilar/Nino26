"""Nome padronizado de figura (pre-codigo) + legenda interpretativa + registro.

Convencao de nome de arquivo::

    Fig_<FASE><BLOCO><N>_<slug_descritivo>.png
    ex.: Fig_4C1_lags_regiao_bioma.png  |  Fig_5A2_shap_summary_pico.png

- ``FASE``  : numero da fase (3, 4, 5, 6, 7, 8)
- ``BLOCO`` : letra do notebook/bloco (A, B, C, D, 0)
- ``N``     : ordinal da figura dentro do bloco
- ``slug``  : descricao curta em ascii-minusculo separada por ``_``

Toda figura salva por :func:`save_registered_figure` recebe, no rodape, uma
**legenda interpretativa** resumida (o que a figura mostra, lido dos numeros) e
grava uma linha no registro central ``<faseX>_legendas_figuras.csv`` com codigo,
arquivo, titulo, interpretacao, metadados e carimbo de tempo. Isso cumpre a
diretriz de rastreabilidade: nenhuma figura sem legenda auditavel.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

FIGURE_DPI = 300
_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(value: str) -> str:
    import unicodedata

    text = unicodedata.normalize("NFKD", str(value))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _SLUG_RE.sub("_", text.lower()).strip("_")


def figure_code(phase: str | int, block: str, index: int, slug: str) -> str:
    """Return the canonical figure code, e.g. ``Fig_4C1_lags_regiao_bioma``."""

    block = str(block).upper()
    if not re.fullmatch(r"[A-Z0-9]", block):
        raise ValueError("block must be a single letter/digit, e.g. 'C' or '0'.")
    if int(index) < 1:
        raise ValueError("index must be a positive figure ordinal.")
    slug_clean = _slugify(slug)
    if not slug_clean:
        raise ValueError("slug must contain at least one alphanumeric character.")
    return f"Fig_{phase}{block}{int(index)}_{slug_clean}"


def stamp_interpretation(fig, *, metadata: str, interpretation: str) -> None:
    """Stamp two footer lines: technical metadata and a literal interpretation."""

    clean = " ".join(str(interpretation).split())
    if not clean:
        raise ValueError("The interpretive legend cannot be empty.")
    fig.text(
        0.5, 0.045, " ".join(str(metadata).split()),
        ha="center", va="bottom", fontsize=9.5, color="#374151", wrap=True,
    )
    fig.text(
        0.5, 0.012, f"Legenda: {clean}",
        ha="center", va="bottom", fontsize=10.5, color="#111827",
        fontweight="semibold", wrap=True,
    )


def save_registered_figure(
    fig,
    *,
    phase: str | int,
    block: str,
    index: int,
    slug: str,
    interpretation: str,
    metadata: str,
    figures_dir: str | Path,
    registry_dir: str | Path | None = None,
    title: str | None = None,
    dpi: int = FIGURE_DPI,
    reserve_bottom: float = 0.11,
    close: bool = True,
) -> Path:
    """Save a large, high-resolution figure with code name and legend, and log it.

    Returns the PNG path. The registry CSV lives in ``registry_dir`` (defaults to
    ``figures_dir``) and is upserted by figure code, so re-running a notebook
    refreshes the legend instead of duplicating it.
    """

    code = figure_code(phase, block, index, slug)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    output = figures_dir / f"{code}.png"

    if title:
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.99)
    stamp_interpretation(fig, metadata=metadata, interpretation=interpretation)
    # Reserve space so the two footer lines never collide with the axes.
    try:
        fig.subplots_adjust(bottom=max(reserve_bottom, fig.subplotpars.bottom))
    except Exception:  # pragma: no cover - layout is best-effort
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)

    registry_dir = Path(registry_dir) if registry_dir is not None else figures_dir
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"fase{phase}_legendas_figuras.csv"
    row = {
        "codigo": code,
        "arquivo": output.name,
        "fase": str(phase),
        "bloco": str(block).upper(),
        "titulo": title or "",
        "legenda_interpretativa": " ".join(str(interpretation).split()),
        "metadados": " ".join(str(metadata).split()),
        "atualizado_em": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        registry = registry[registry["codigo"] != code]
        registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    else:
        registry = pd.DataFrame([row])
    registry = registry.sort_values("codigo").reset_index(drop=True)
    registry.to_csv(registry_path, index=False)
    return output


def register_only(
    code: str,
    arquivo: str,
    *,
    phase: str | int,
    block: str,
    interpretation: str,
    metadata: str,
    registry_dir: str | Path,
    title: str = "",
) -> Path:
    """Register a legend for a figure saved by another routine (e.g. pixel maps)."""

    import datetime as _dt

    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry_path = registry_dir / f"fase{phase}_legendas_figuras.csv"
    row = {
        "codigo": code,
        "arquivo": arquivo,
        "fase": str(phase),
        "bloco": str(block).upper(),
        "titulo": title,
        "legenda_interpretativa": " ".join(str(interpretation).split()),
        "metadados": " ".join(str(metadata).split()),
        "atualizado_em": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if registry_path.exists():
        registry = pd.read_csv(registry_path)
        registry = registry[registry["codigo"] != code]
        registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    else:
        registry = pd.DataFrame([row])
    registry.sort_values("codigo").reset_index(drop=True).to_csv(registry_path, index=False)
    return registry_path


def write_caption_index(registry_path: str | Path, output_md: str | Path) -> Path:
    """Render the figure/legend registry as a readable Markdown index."""

    registry_path = Path(registry_path)
    registry = pd.read_csv(registry_path)
    lines = ["# Indice de figuras e legendas", ""]
    for _, r in registry.sort_values("codigo").iterrows():
        lines.append(f"### {r['codigo']}")
        if str(r.get("titulo", "")).strip():
            lines.append(f"**{r['titulo']}**")
        lines.append(f"- Arquivo: `{r['arquivo']}`")
        lines.append(f"- Legenda: {r['legenda_interpretativa']}")
        lines.append(f"- Metadados: {r['metadados']}")
        lines.append("")
    output_md = Path(output_md)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return output_md
