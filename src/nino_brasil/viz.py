"""Co-geração figura + numeric-table sob código predecessor único (padrão NINO-BRASIL).

Princípio (Diretrizes, item 2): nenhuma figura analítica sem tabela numérica
rastreável. Aqui a figura e a(s) tabela(s) que a sustentam **nascem do mesmo
chamado**, com o **mesmo código** ``Fig_<F><B><NN>`` (F=fase 2..8, B=bloco
0/A/B/C/..., NN=sequência). A cada execução do notebook os artefatos são
reescritos por **sobreposição** (idempotente) — corrigir/rodar de novo atualiza
figura, tabelas, manifesto por cima, sem lixo acumulado.

Uso típico no notebook::

    from nino_brasil.viz import registrar_figura
    registrar_figura(
        fig, "Fig_3C01",
        fase=3, bloco="C",
        titulo="Ranking de precursores por lag",
        descricao="Correlação defasada preditor->SSTA; barra cheia passa FDR.",
        hipotese="HIP0",
        notebook="notebooks/fase3/3C_precursores_lags.ipynb",
        fontes={"phase3C_ranking_lags": df_ranking, "phase3C_lag_correlacoes": df_corr},
    )

``fontes`` aceita ``{nome: DataFrame}`` ou ``{nome: caminho_csv/parquet}``.
"""
from __future__ import annotations

import hashlib
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "data" / "processed" / "figures"
NUM_ROOT = ROOT / "data" / "processed" / "numeric-tables"
MANIFEST = ROOT / "data" / "processed" / "figuras_manifesto.csv"

CODIGO_RE = re.compile(r"^Fig_[2-8][0A-Z]\d{2}(?:_[a-z0-9]+)?$")
MANIFEST_COLS = [
    "codigo", "fase", "bloco", "arquivo", "notebook", "titulo",
    "hipotese", "descricao", "tabelas", "n_tabelas", "atualizado_em",
]


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()[:16]


def _as_frame(fonte) -> pd.DataFrame:
    if isinstance(fonte, pd.DataFrame):
        return fonte.reset_index() if not isinstance(fonte.index, pd.RangeIndex) else fonte
    p = Path(fonte)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        return df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    return pd.read_csv(p)


def validar_codigo(codigo: str) -> None:
    if not CODIGO_RE.match(codigo):
        raise ValueError(
            f"código inválido '{codigo}': use Fig_<F><B><NN>, ex. Fig_3C01, "
            f"Fig_4001, Fig_5A01_rf (F=2..8, B=0/A/B/C..., NN=2 dígitos)."
        )


def registrar_figura(
    fig,
    codigo: str,
    *,
    fase: int,
    bloco: str,
    titulo: str,
    descricao: str,
    hipotese: str,
    fontes: dict,
    notebook: str = "",
    dpi: int = 150,
) -> Path:
    """Salva a figura e congela suas tabelas numéricas sob o mesmo ``codigo``.

    Sobrescreve (idempotente): remove o conteúdo anterior da pasta da tabela e
    regrava figura + CSVs + ``manifest.csv`` + ``README.md``; faz upsert de uma
    linha em ``data/processed/figuras_manifesto.csv``. Retorna o caminho do PNG.
    """
    validar_codigo(codigo)
    if not fontes:
        raise ValueError(f"{codigo}: 'fontes' vazio — toda figura precisa de tabela numérica.")

    fase_dir = f"fase{fase}"
    fig_dir = FIG_ROOT / fase_dir
    tbl_dir = NUM_ROOT / fase_dir / codigo
    fig_dir.mkdir(parents=True, exist_ok=True)
    if tbl_dir.exists():
        shutil.rmtree(tbl_dir)          # sobreposição limpa (sem CSV órfão)
    tbl_dir.mkdir(parents=True, exist_ok=True)

    # 1) figura (sobrescreve)
    fig_path = fig_dir / f"{codigo}.png"
    if fig is not None:
        fig.savefig(fig_path, dpi=dpi, bbox_inches="tight")

    # 2) tabelas congeladas + manifest da figura
    linhas = []
    for nome, fonte in fontes.items():
        df = _as_frame(fonte)
        df.to_csv(tbl_dir / f"{nome}.csv", index=False)
        linhas.append({
            "tabela": f"{nome}.csv", "linhas": int(df.shape[0]),
            "colunas": int(df.shape[1]), "sha256_16": _sha256_df(df),
        })
    pd.DataFrame(linhas).to_csv(tbl_dir / "manifest.csv", index=False)

    # 3) README humano
    (tbl_dir / "README.md").write_text(
        f"# {codigo} — {titulo}\n\n"
        f"- **Fase/bloco:** {fase}{bloco}\n- **Hipótese:** {hipotese}\n"
        f"- **Notebook:** `{notebook}`\n- **Figura:** `../../figures/{fase_dir}/{codigo}.png`\n"
        f"- **Atualizado:** {_now()}\n\n## Descrição\n{descricao}\n\n"
        f"## Tabelas congeladas\n" + "\n".join(
            f"- `{l['tabela']}` ({l['linhas']}x{l['colunas']}, sha {l['sha256_16']})"
            for l in linhas
        ) + "\n",
        encoding="utf-8",
    )

    # 4) upsert no manifesto global
    row = {
        "codigo": codigo, "fase": fase, "bloco": bloco,
        "arquivo": f"{fase_dir}/{codigo}.png", "notebook": notebook,
        "titulo": titulo, "hipotese": hipotese, "descricao": descricao,
        "tabelas": ";".join(f"{n}.csv" for n in fontes), "n_tabelas": len(fontes),
        "atualizado_em": _now(),
    }
    if MANIFEST.exists():
        man = pd.read_csv(MANIFEST)
        man = man[man["codigo"] != codigo]
        man = pd.concat([man, pd.DataFrame([row])], ignore_index=True)
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        man = pd.DataFrame([row], columns=MANIFEST_COLS)
    man.sort_values("codigo").to_csv(MANIFEST, index=False)
    return fig_path


def validar_saidas(strict: bool = True) -> pd.DataFrame:
    """Valida a coerência tripla figura <-> numeric-table <-> manifesto.

    Regras: (a) todo PNG em figures/ casa o padrão de código e tem pasta de
    tabela homônima + linha no manifesto; (b) todo código é único; (c) sem
    ``_tmp*``/órfãos. Retorna DataFrame de problemas; em ``strict`` levanta erro.
    """
    problemas: list[dict] = []
    man = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=MANIFEST_COLS)
    codigos_man = set(man["codigo"]) if not man.empty else set()

    pngs = sorted(FIG_ROOT.glob("fase*/*.png")) if FIG_ROOT.exists() else []
    vistos: set[str] = set()
    for png in pngs:
        fase_dir, codigo = png.parent.name, png.stem
        if codigo.startswith("_tmp") or codigo.startswith("."):
            problemas.append({"tipo": "lixo", "item": str(png.relative_to(ROOT))}); continue
        if not CODIGO_RE.match(codigo):
            problemas.append({"tipo": "nome_fora_do_padrao", "item": f"{fase_dir}/{codigo}"}); continue
        if codigo in vistos:
            problemas.append({"tipo": "codigo_duplicado", "item": codigo}); continue
        vistos.add(codigo)
        if not (NUM_ROOT / fase_dir / codigo).is_dir():
            problemas.append({"tipo": "sem_numeric_table", "item": f"{fase_dir}/{codigo}"})
        if codigo not in codigos_man:
            problemas.append({"tipo": "fora_do_manifesto", "item": f"{fase_dir}/{codigo}"})

    df = pd.DataFrame(problemas)
    if strict and not df.empty:
        raise AssertionError(
            f"{len(df)} problema(s) de contrato figura/tabela:\n{df.to_string(index=False)}"
        )
    return df
