"""Publicação auditável de figuras e tabelas do NINO-BRASIL.

Novos workflows devem usar :func:`registrar_par_notebook`, cujo contrato
publico vincula ``FigF...`` e ``TabF...`` pelo codigo completo do notebook e
ordinal. :func:`registrar_figura` e os helpers ``Fig_...`` permanecem abaixo
somente para compatibilidade durante a migracao.

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

from dataclasses import dataclass
import hashlib
import json
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from nino_brasil.artifact_codes import (
    assert_artifact_pair,
    figure_code as public_figure_code,
    parse_artifact_code,
    parse_notebook_code,
    table_code as public_table_code,
)

ROOT = Path(__file__).resolve().parents[2]
FIG_ROOT = ROOT / "data" / "processed" / "figures"
NUM_ROOT = ROOT / "data" / "processed" / "numeric-tables"
METADATA_ROOT = ROOT / "data" / "processed" / "metadata" / "figure-tables"
MANIFEST = ROOT / "data" / "processed" / "figuras_manifesto.csv"

CODIGO_RE = re.compile(r"^Fig_[2-8][0A-Z]\d{2}(?:_[a-z0-9]+)*$")
MANIFEST_COLS = [
    "codigo", "fase", "bloco", "arquivo", "notebook", "titulo",
    "hipotese", "descricao", "tabelas", "n_tabelas", "audit_level",
    "run_id", "atualizado_em",
    # Contrato publico FigF... <-> TabF.... As colunas legadas acima sao
    # mantidas para que consumidores existentes continuem funcionando.
    "artifact_format", "pair_key", "figure_code", "table_code",
    "notebook_code", "ordinal", "namespace", "figure_sha256",
    "table_sha256", "pair_manifest", "source_path", "source_sha256",
]


@dataclass(frozen=True)
class NotebookArtifactPair:
    """Resultado imutavel da publicacao de um par Fig/Tab de notebook."""

    notebook_code: str
    ordinal: int
    namespace: str
    figure_code: str
    table_code: str
    figure_path: Path
    table_path: Path
    manifest_path: Path
    figure_sha256: str
    table_sha256: str
    run_id: str


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")


def _sha256_df(df: pd.DataFrame) -> str:
    return hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values.tobytes()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _align_manifest_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in MANIFEST_COLS:
        if column not in out:
            out[column] = ""
    return out.loc[:, MANIFEST_COLS]


def _as_frame(fonte) -> pd.DataFrame:
    if isinstance(fonte, pd.DataFrame):
        return fonte.reset_index() if not isinstance(fonte.index, pd.RangeIndex) else fonte
    p = Path(fonte)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
        return df.reset_index() if not isinstance(df.index, pd.RangeIndex) else df
    return pd.read_csv(p)


def _public_namespace(notebook_code: str) -> str:
    """Diretorio publico de uma fase, sem ``fase3_nino/fase3`` redundante."""

    parsed = parse_notebook_code(notebook_code)
    namespace = f"fase{parsed.phase}"
    if parsed.signal:
        namespace += f"_{parsed.signal.lower()}"
    return namespace


def _same_public_precode(path: Path, precode: str) -> bool:
    stem = path.name
    for suffix in (".manifest.json", ".csv", ".png"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return stem == precode or stem.startswith(f"{precode}_")


def _atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-", suffix=".csv", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-", suffix=".json", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def registrar_par_notebook(
    fig,
    notebook_code: str,
    ordinal: int,
    tabela: pd.DataFrame | Path | str,
    *,
    slug: str | None = None,
    titulo: str = "",
    descricao: str = "",
    hipotese: str = "",
    notebook: str = "",
    run_id: str,
    dpi: int = 150,
) -> NotebookArtifactPair:
    """Publica um par auditavel ``FigF...`` / ``TabF...``.

    A chave do par e ``notebook_code + ordinal``. O ``slug`` e apenas
    descritivo: mudar o slug substitui o par anterior em vez de criar uma nova
    chave. A tabela publica e sempre CSV, inclusive quando a fonte e Parquet.

    Layout publico::

        figures/fase3_nino/FigF3NinoA1_<slug>.png
        numeric-tables/fase3_nino/TabF3NinoA1_<slug>.csv
        metadata/figure-tables/fase3_nino/TabF3NinoA1_<slug>.manifest.json
    """

    parsed_notebook = parse_notebook_code(notebook_code)
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise ValueError("run_id e obrigatorio no contrato publico Fig/Tab")
    if fig is None or not callable(getattr(fig, "savefig", None)):
        raise TypeError("fig deve oferecer o metodo savefig")
    if not isinstance(tabela, (pd.DataFrame, Path, str)):
        raise TypeError("tabela deve ser pandas.DataFrame ou pathlib.Path")

    frame = _as_frame(tabela)
    source_path = ""
    if isinstance(tabela, (Path, str)):
        source = Path(tabela)
        source_path = str(source.resolve())
        source_sha256 = _sha256_file(source)
        source_kind = "path"
    else:
        source_sha256 = _sha256_df(frame)
        source_kind = "dataframe"
    figure_code = public_figure_code(notebook_code, ordinal, slug=slug)
    table_code = public_table_code(notebook_code, ordinal, slug=slug)
    assert_artifact_pair(figure_code, table_code)
    parsed_figure = parse_artifact_code(figure_code)
    pair_key = f"{parsed_notebook.value}:{parsed_figure.ordinal}"
    namespace = _public_namespace(parsed_notebook.value)

    figure_dir = FIG_ROOT / namespace
    table_dir = NUM_ROOT / namespace
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_path = figure_dir / f"{figure_code}.png"
    table_path = table_dir / f"{table_code}.csv"
    pair_manifest_path = METADATA_ROOT / namespace / f"{table_code}.manifest.json"
    pair_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    # O slug nao participa da chave. Remova a representacao anterior do mesmo
    # notebook+ordinal antes de publicar a nova, evitando pares orfaos.
    for candidate in figure_dir.glob(f"{parsed_figure.precode}*.png"):
        if _same_public_precode(candidate, parsed_figure.precode):
            candidate.unlink(missing_ok=True)
    table_precode = parse_artifact_code(table_code).precode
    for candidate in table_dir.glob(f"{table_precode}*"):
        if candidate.is_file() and _same_public_precode(candidate, table_precode):
            candidate.unlink(missing_ok=True)
    metadata_dir = METADATA_ROOT / namespace
    if metadata_dir.exists():
        for candidate in metadata_dir.glob(f"{table_precode}*.manifest.json"):
            if _same_public_precode(candidate, table_precode):
                candidate.unlink(missing_ok=True)

    with tempfile.NamedTemporaryFile(
        prefix=f".{figure_code}-", suffix=".png", dir=figure_dir, delete=False
    ) as stream:
        temporary_figure = Path(stream.name)
    try:
        fig.savefig(temporary_figure, dpi=dpi, bbox_inches="tight")
        temporary_figure.replace(figure_path)
    finally:
        temporary_figure.unlink(missing_ok=True)
    _atomic_write_csv(frame, table_path)

    figure_sha256 = _sha256_file(figure_path)
    table_sha256 = _sha256_file(table_path)
    timestamp = _now()
    pair_manifest = {
        "schema_version": 1,
        "artifact_format": "notebook_public_pair",
        "pair_key": pair_key,
        "figure_code": figure_code,
        "table_code": table_code,
        "notebook_code": parsed_notebook.value,
        "notebook": notebook,
        "ordinal": parsed_figure.ordinal,
        "namespace": namespace,
        "paths": {
            "figure": _relative_to_root(figure_path),
            "table": _relative_to_root(table_path),
        },
        "hashes": {
            "figure_sha256": figure_sha256,
            "table_sha256": table_sha256,
            "source_sha256": source_sha256,
        },
        "source": {"kind": source_kind, "path": source_path},
        "run_id": normalized_run_id,
        "updated_at": timestamp,
    }
    _atomic_write_json(pair_manifest, pair_manifest_path)

    row = {
        # Campos legados preenchidos para leitores ainda nao migrados.
        "codigo": figure_code,
        "fase": parsed_notebook.phase,
        "bloco": parsed_notebook.block,
        "arquivo": f"{namespace}/{figure_path.name}",
        "notebook": notebook,
        "titulo": titulo or figure_code,
        "hipotese": hipotese,
        "descricao": descricao,
        "tabelas": table_path.name,
        "n_tabelas": 1,
        "audit_level": "semantic_source",
        "run_id": normalized_run_id,
        "atualizado_em": timestamp,
        # Contrato publico.
        "artifact_format": "notebook_public_pair",
        "pair_key": pair_key,
        "figure_code": figure_code,
        "table_code": table_code,
        "notebook_code": parsed_notebook.value,
        "ordinal": parsed_figure.ordinal,
        "namespace": namespace,
        "figure_sha256": figure_sha256,
        "table_sha256": table_sha256,
        "pair_manifest": _relative_to_root(pair_manifest_path),
        "source_path": source_path,
        "source_sha256": source_sha256,
    }
    if MANIFEST.exists():
        manifest = _align_manifest_columns(pd.read_csv(MANIFEST))
        manifest = manifest.loc[
            ~manifest["pair_key"].fillna("").astype(str).eq(pair_key)
        ]
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        manifest = pd.DataFrame(columns=MANIFEST_COLS)
    manifest = pd.concat([manifest, pd.DataFrame([row])], ignore_index=True)
    manifest = _align_manifest_columns(manifest).sort_values("codigo")
    _atomic_write_csv(manifest, MANIFEST)

    return NotebookArtifactPair(
        notebook_code=parsed_notebook.value,
        ordinal=parsed_figure.ordinal,
        namespace=namespace,
        figure_code=figure_code,
        table_code=table_code,
        figure_path=figure_path,
        table_path=table_path,
        manifest_path=pair_manifest_path,
        figure_sha256=figure_sha256,
        table_sha256=table_sha256,
        run_id=normalized_run_id,
    )


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
    run_id: str = "",
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
        table_path = tbl_dir / f"{nome}.csv"
        df.to_csv(table_path, index=False)
        source_path = ""
        source_manifest_path = ""
        source_manifest_sha256 = ""
        source_run_id = ""
        if not isinstance(fonte, pd.DataFrame):
            original = Path(fonte).resolve()
            source_path = str(original)
            semantic_manifest = original.with_suffix(original.suffix + ".manifest.json")
            if semantic_manifest.is_file():
                source_manifest_path = str(semantic_manifest.resolve())
                source_manifest_sha256 = _sha256_file(semantic_manifest)
                try:
                    source_run_id = str(
                        json.loads(semantic_manifest.read_text(encoding="utf-8")).get("run_id", "")
                    )
                except (OSError, json.JSONDecodeError):
                    source_run_id = ""
        linhas.append({
            "tabela": f"{nome}.csv", "linhas": int(df.shape[0]),
            "colunas": int(df.shape[1]),
            "sha256": _sha256_file(table_path),
            "dataframe_sha256": _sha256_df(df),
            "schema_json": json.dumps({c: str(t) for c, t in df.dtypes.items()}, sort_keys=True),
            "semantic_source": True,
            "audit_level": "semantic_source",
            "run_id": run_id,
            "source_path": source_path,
            "source_manifest_path": source_manifest_path,
            "source_manifest_sha256": source_manifest_sha256,
            "source_run_id": source_run_id,
        })
    pd.DataFrame(linhas).to_csv(tbl_dir / "manifest.csv", index=False)

    # 3) README humano
    (tbl_dir / "README.md").write_text(
        f"# {codigo} — {titulo}\n\n"
        f"- **Fase/bloco:** {fase}{bloco}\n- **Hipótese:** {hipotese}\n"
        f"- **Notebook:** `{notebook}`\n- **Figura:** `../../figures/{fase_dir}/{codigo}.png`\n"
        f"- **Atualizado:** {_now()}\n\n## Descrição\n{descricao}\n\n"
        f"## Tabelas congeladas\n" + "\n".join(
            f"- `{l['tabela']}` ({l['linhas']}x{l['colunas']}, SHA-256 {l['sha256']})"
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
        "audit_level": "semantic_source", "run_id": run_id,
        "atualizado_em": _now(),
    }
    if MANIFEST.exists():
        man = _align_manifest_columns(pd.read_csv(MANIFEST))
        man = man[man["codigo"] != codigo]
        man = pd.concat([man, pd.DataFrame([row])], ignore_index=True)
    else:
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        man = pd.DataFrame([row], columns=MANIFEST_COLS)
    man.sort_values("codigo").to_csv(MANIFEST, index=False)
    return fig_path


def _manifest_text(value: object) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() == "nan" else text


def _validate_public_pairs(manifest: pd.DataFrame, problemas: list[dict]) -> None:
    """Valida pares publicos sem misturar seu layout plano com o legado."""

    public_mask = (
        manifest["artifact_format"].fillna("").astype(str).eq("notebook_public_pair")
        | manifest["figure_code"].fillna("").astype(str).str.startswith("FigF")
    )
    public = manifest.loc[public_mask].copy()
    if not public.empty and public["pair_key"].fillna("").astype(str).duplicated().any():
        duplicated = public.loc[
            public["pair_key"].fillna("").astype(str).duplicated(keep=False),
            "pair_key",
        ]
        for pair_key in sorted(set(duplicated.astype(str))):
            problemas.append(
                {"tipo": "public_pair_duplicado_manifesto", "item": pair_key}
            )

    manifested_figures: set[str] = set()
    manifested_tables: set[str] = set()
    required = (
        "pair_key",
        "figure_code",
        "table_code",
        "notebook_code",
        "namespace",
        "figure_sha256",
        "table_sha256",
        "pair_manifest",
        "run_id",
    )
    for record in public.to_dict(orient="records"):
        missing = [name for name in required if not _manifest_text(record.get(name))]
        item = _manifest_text(record.get("pair_key")) or _manifest_text(
            record.get("figure_code")
        )
        if missing:
            problemas.append(
                {
                    "tipo": "public_manifest_schema_incompleto",
                    "item": f"{item}: {missing}",
                }
            )
            continue

        figure_code = _manifest_text(record["figure_code"])
        table_code = _manifest_text(record["table_code"])
        notebook_code = _manifest_text(record["notebook_code"])
        namespace = _manifest_text(record["namespace"])
        manifested_figures.add(figure_code)
        manifested_tables.add(table_code)
        try:
            parsed_figure = parse_artifact_code(figure_code)
            parsed_table = parse_artifact_code(table_code)
            assert_artifact_pair(figure_code, table_code)
            parsed_notebook = parse_notebook_code(notebook_code)
        except ValueError as exc:
            problemas.append(
                {"tipo": "public_pair_codigo_invalido", "item": f"{item}: {exc}"}
            )
            continue
        if parsed_figure.kind != "Fig" or parsed_table.kind != "Tab":
            problemas.append(
                {"tipo": "public_pair_tipos_invertidos", "item": item}
            )
        if parsed_figure.notebook.value != parsed_notebook.value:
            problemas.append(
                {
                    "tipo": "public_notebook_code_divergente",
                    "item": f"{item}: {parsed_figure.notebook.value} != {notebook_code}",
                }
            )
        expected_namespace = _public_namespace(notebook_code)
        if namespace != expected_namespace:
            problemas.append(
                {
                    "tipo": "public_namespace_divergente",
                    "item": f"{item}: {namespace} != {expected_namespace}",
                }
            )
        expected_pair_key = f"{notebook_code}:{parsed_figure.ordinal}"
        if _manifest_text(record["pair_key"]) != expected_pair_key:
            problemas.append(
                {
                    "tipo": "public_pair_key_divergente",
                    "item": f"{item}: esperado {expected_pair_key}",
                }
            )

        figure_path = FIG_ROOT / namespace / f"{figure_code}.png"
        table_path = NUM_ROOT / namespace / f"{table_code}.csv"
        expected_figure_entry = f"{namespace}/{figure_code}.png"
        expected_table_entry = f"{table_code}.csv"
        if _manifest_text(record.get("arquivo")) != expected_figure_entry:
            problemas.append(
                {
                    "tipo": "public_figure_path_divergente",
                    "item": f"{item}: esperado {expected_figure_entry}",
                }
            )
        if _manifest_text(record.get("tabelas")) != expected_table_entry:
            problemas.append(
                {
                    "tipo": "public_table_path_divergente",
                    "item": f"{item}: esperado {expected_table_entry}",
                }
            )
        if not figure_path.is_file():
            problemas.append(
                {"tipo": "public_figure_ausente", "item": str(figure_path)}
            )
        elif _sha256_file(figure_path) != _manifest_text(record["figure_sha256"]):
            problemas.append(
                {"tipo": "public_figure_hash_divergente", "item": str(figure_path)}
            )
        if not table_path.is_file():
            problemas.append(
                {"tipo": "public_table_ausente", "item": str(table_path)}
            )
        elif _sha256_file(table_path) != _manifest_text(record["table_sha256"]):
            problemas.append(
                {"tipo": "public_table_hash_divergente", "item": str(table_path)}
            )

        manifest_value = Path(_manifest_text(record["pair_manifest"]))
        pair_manifest_path = (
            manifest_value
            if manifest_value.is_absolute()
            else ROOT / manifest_value
        )
        if not pair_manifest_path.is_file():
            problemas.append(
                {
                    "tipo": "public_pair_manifest_ausente",
                    "item": str(pair_manifest_path),
                }
            )
            continue
        try:
            pair_manifest = json.loads(pair_manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problemas.append(
                {
                    "tipo": "public_pair_manifest_invalido",
                    "item": f"{pair_manifest_path}: {exc}",
                }
            )
            continue
        for field, expected in (
            ("figure_code", figure_code),
            ("table_code", table_code),
            ("notebook_code", notebook_code),
            ("pair_key", expected_pair_key),
            ("namespace", namespace),
            ("run_id", _manifest_text(record["run_id"])),
        ):
            if _manifest_text(pair_manifest.get(field)) != expected:
                problemas.append(
                    {
                        "tipo": "public_pair_manifest_divergente",
                        "item": f"{pair_manifest_path}: {field}",
                    }
                )
        hashes = pair_manifest.get("hashes")
        if not isinstance(hashes, dict):
            problemas.append(
                {
                    "tipo": "public_pair_manifest_sem_hashes",
                    "item": str(pair_manifest_path),
                }
            )
        else:
            for field in ("figure_sha256", "table_sha256"):
                if _manifest_text(hashes.get(field)) != _manifest_text(record[field]):
                    problemas.append(
                        {
                            "tipo": "public_pair_manifest_hash_divergente",
                            "item": f"{pair_manifest_path}: {field}",
                        }
                    )
        paths = pair_manifest.get("paths")
        expected_paths = {
            "figure": _relative_to_root(figure_path),
            "table": _relative_to_root(table_path),
        }
        if not isinstance(paths, dict):
            problemas.append(
                {
                    "tipo": "public_pair_manifest_sem_paths",
                    "item": str(pair_manifest_path),
                }
            )
        else:
            for field, expected in expected_paths.items():
                if _manifest_text(paths.get(field)) != expected:
                    problemas.append(
                        {
                            "tipo": "public_pair_manifest_path_divergente",
                            "item": f"{pair_manifest_path}: {field}",
                        }
                    )
        source_path_value = _manifest_text(record.get("source_path"))
        if source_path_value:
            source_path = Path(source_path_value)
            if not source_path.is_file():
                problemas.append(
                    {"tipo": "public_source_ausente", "item": str(source_path)}
                )
            elif _sha256_file(source_path) != _manifest_text(
                record.get("source_sha256")
            ):
                problemas.append(
                    {
                        "tipo": "public_source_hash_divergente",
                        "item": str(source_path),
                    }
                )

    for figure_path in sorted(FIG_ROOT.glob("fase*/FigF*.png")):
        try:
            parsed = parse_artifact_code(figure_path.stem)
        except ValueError as exc:
            problemas.append(
                {"tipo": "public_figure_codigo_invalido", "item": f"{figure_path}: {exc}"}
            )
            continue
        if parsed.kind != "Fig":
            problemas.append(
                {"tipo": "public_figure_tipo_invalido", "item": str(figure_path)}
            )
        expected_namespace = _public_namespace(parsed.notebook.value)
        if figure_path.parent.name != expected_namespace:
            problemas.append(
                {"tipo": "public_namespace_divergente", "item": str(figure_path)}
            )
        if figure_path.stem not in manifested_figures:
            problemas.append(
                {"tipo": "public_figure_fora_manifesto", "item": str(figure_path)}
            )

    for table_path in sorted(NUM_ROOT.glob("fase*/TabF*.csv")):
        try:
            parsed = parse_artifact_code(table_path.stem)
        except ValueError as exc:
            problemas.append(
                {"tipo": "public_table_codigo_invalido", "item": f"{table_path}: {exc}"}
            )
            continue
        if parsed.kind != "Tab":
            problemas.append(
                {"tipo": "public_table_tipo_invalido", "item": str(table_path)}
            )
        expected_namespace = _public_namespace(parsed.notebook.value)
        if table_path.parent.name != expected_namespace:
            problemas.append(
                {"tipo": "public_namespace_divergente", "item": str(table_path)}
            )
        if table_path.stem not in manifested_tables:
            problemas.append(
                {"tipo": "public_table_fora_manifesto", "item": str(table_path)}
            )

    for nested in sorted(FIG_ROOT.glob("fase*/*/FigF*.png")):
        problemas.append(
            {"tipo": "public_namespace_aninhado_redundante", "item": str(nested)}
        )
    for nested in sorted(NUM_ROOT.glob("fase*/*/TabF*.csv")):
        problemas.append(
            {"tipo": "public_namespace_aninhado_redundante", "item": str(nested)}
        )


def validar_saidas(
    strict: bool = True,
    *,
    require_semantic_lineage: bool = False,
) -> pd.DataFrame:
    """Valida a coerência tripla figura <-> numeric-table <-> manifesto.

    Regras: (a) todo PNG em figures/ casa o padrão de código e tem pasta de
    tabela homônima + linha no manifesto; (b) todo código é único; (c) sem
    ``_tmp*``/órfãos. Retorna DataFrame de problemas; em ``strict`` levanta erro.
    """
    problemas: list[dict] = []
    man = pd.read_csv(MANIFEST) if MANIFEST.exists() else pd.DataFrame(columns=MANIFEST_COLS)
    man = _align_manifest_columns(man)
    if man["codigo"].duplicated().any():
        for code in man.loc[man["codigo"].duplicated(keep=False), "codigo"].astype(str).unique():
            problemas.append({"tipo": "codigo_duplicado_manifesto", "item": code})
    codigos_man = set(man["codigo"].astype(str)) if not man.empty else set()
    if require_semantic_lineage:
        for row in man.itertuples():
            if str(row.audit_level) != "semantic_source":
                problemas.append(
                    {"tipo": "audit_level_nao_semantico", "item": str(row.codigo)}
                )
            if not str(row.run_id).strip() or str(row.run_id).lower() == "nan":
                problemas.append(
                    {"tipo": "run_id_ausente", "item": str(row.codigo)}
                )

    pngs = sorted(FIG_ROOT.glob("fase*/*.png")) if FIG_ROOT.exists() else []
    vistos: set[str] = set()
    for png in pngs:
        fase_dir, codigo = png.parent.name, png.stem
        if codigo.startswith("_tmp") or codigo.startswith("."):
            problemas.append({"tipo": "lixo", "item": str(png.relative_to(ROOT))}); continue
        if not CODIGO_RE.match(codigo):
            try:
                public_code = parse_artifact_code(codigo)
            except ValueError:
                problemas.append({"tipo": "nome_fora_do_padrao", "item": f"{fase_dir}/{codigo}"})
            else:
                if public_code.kind != "Fig":
                    problemas.append({"tipo": "nome_fora_do_padrao", "item": f"{fase_dir}/{codigo}"})
            continue
        if codigo in vistos:
            problemas.append({"tipo": "codigo_duplicado", "item": codigo}); continue
        vistos.add(codigo)
        if not (NUM_ROOT / fase_dir / codigo).is_dir():
            problemas.append({"tipo": "sem_numeric_table", "item": f"{fase_dir}/{codigo}"})
        if codigo not in codigos_man:
            problemas.append({"tipo": "fora_do_manifesto", "item": f"{fase_dir}/{codigo}"})

    # Reverse direction: every numeric-table/global row must resolve to a PNG,
    # and every table hash in the local manifest must still match its content.
    if NUM_ROOT.exists():
        for table_dir in sorted(path for path in NUM_ROOT.glob("fase*/*") if path.is_dir()):
            fase_dir, codigo = table_dir.parent.name, table_dir.name
            png = FIG_ROOT / fase_dir / f"{codigo}.png"
            if not png.exists():
                problemas.append({"tipo": "numeric_table_sem_figura", "item": f"{fase_dir}/{codigo}"})
            if codigo not in codigos_man:
                problemas.append({"tipo": "numeric_table_fora_manifesto", "item": f"{fase_dir}/{codigo}"})
            local_manifest = table_dir / "manifest.csv"
            if not local_manifest.exists():
                problemas.append({"tipo": "sem_manifest_local", "item": f"{fase_dir}/{codigo}"})
                continue
            try:
                local = pd.read_csv(local_manifest)
            except Exception as exc:
                problemas.append({"tipo": "manifest_local_invalido", "item": f"{fase_dir}/{codigo}: {exc}"})
                continue
            if require_semantic_lineage:
                required_columns = {
                    "tabela",
                    "sha256",
                    "schema_json",
                    "semantic_source",
                    "audit_level",
                    "run_id",
                }
                if fase_dir == "fase3":
                    required_columns.update(
                        {
                            "source_path",
                            "source_manifest_path",
                            "source_manifest_sha256",
                            "source_run_id",
                        }
                    )
                missing_columns = sorted(required_columns.difference(local.columns))
                if missing_columns:
                    problemas.append(
                        {
                            "tipo": "manifest_local_schema_incompleto",
                            "item": f"{fase_dir}/{codigo}: {missing_columns}",
                        }
                    )
                else:
                    semantic = local["semantic_source"].astype(str).str.lower().isin(
                        {"true", "1"}
                    )
                    if not bool(semantic.all()):
                        problemas.append(
                            {
                                "tipo": "numeric_table_render_extraction_only",
                                "item": f"{fase_dir}/{codigo}",
                            }
                        )
                    if not bool(local["audit_level"].astype(str).eq("semantic_source").all()):
                        problemas.append(
                            {
                                "tipo": "manifest_local_audit_level_invalido",
                                "item": f"{fase_dir}/{codigo}",
                            }
                        )
                    local_run_ids = local["run_id"].fillna("").astype(str).str.strip()
                    if not bool(local_run_ids.ne("").all()):
                        problemas.append(
                            {
                                "tipo": "manifest_local_run_id_ausente",
                                "item": f"{fase_dir}/{codigo}",
                            }
                        )
                    if fase_dir == "fase3" and required_columns.issubset(local.columns):
                        for source in local.itertuples():
                            source_path = Path(str(source.source_path))
                            source_manifest = Path(str(source.source_manifest_path))
                            if not source_path.is_file() or not source_manifest.is_file():
                                problemas.append(
                                    {
                                        "tipo": "fonte_semantica_ausente",
                                        "item": f"{fase_dir}/{codigo}: {source_path}",
                                    }
                                )
                                continue
                            if _sha256_file(source_manifest) != str(source.source_manifest_sha256):
                                problemas.append(
                                    {
                                        "tipo": "manifesto_fonte_divergente",
                                        "item": f"{fase_dir}/{codigo}: {source_manifest}",
                                    }
                                )
                            if str(source.source_run_id).strip() != str(source.run_id).strip():
                                problemas.append(
                                    {
                                        "tipo": "run_id_fonte_divergente",
                                        "item": f"{fase_dir}/{codigo}: {source.source_run_id} != {source.run_id}",
                                    }
                                )
            for row in local.itertuples():
                table = table_dir / str(row.tabela)
                if not table.exists():
                    problemas.append({"tipo": "tabela_ausente", "item": str(table.relative_to(ROOT))})
                    continue
                expected = getattr(row, "sha256", None)
                if require_semantic_lineage and not (
                    isinstance(expected, str) and expected
                ):
                    problemas.append(
                        {"tipo": "hash_ausente", "item": str(table.relative_to(ROOT))}
                    )
                elif isinstance(expected, str) and expected and _sha256_file(table) != expected:
                    problemas.append({"tipo": "hash_divergente", "item": str(table.relative_to(ROOT))})

    _validate_public_pairs(man, problemas)

    for row in man.itertuples():
        png = FIG_ROOT / str(row.arquivo)
        if not png.exists():
            problemas.append({"tipo": "manifesto_sem_figura", "item": str(row.codigo)})

    df = pd.DataFrame(problemas)
    if strict and not df.empty:
        raise AssertionError(
            f"{len(df)} problema(s) de contrato figura/tabela:\n{df.to_string(index=False)}"
        )
    return df


# ---------------------------------------------------------------------------
# Ligação automática do save_fig legado -> código canônico + numeric-table.
# Permite que os notebooks passem a padronizar nomes e co-gerar a tabela SEM
# reescrever cada célula: basta o save_fig compartilhado chamar cogerar_de_figura.
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(r"^(?:Fig_)?([2-8])([0-9A-Z])0*(\d+)(?:_(.*))?$")

# Nomes legados que não encaixam no regex, mapeados explicitamente.
_ALIAS = {
    "phase2_sanidade_oceano_superficie": "Fig_2Z01_oceano_superficie",
    "phase2_sanidade_oceano_recarga": "Fig_2Z02_oceano_recarga",
    "phase2_sanidade_oceano_temp_perfil": "Fig_2Z03_temp_perfil",
    "phase2_sanidade_atmosfera_bjerknes": "Fig_2Z04_atmosfera_bjerknes",
    "phase2_sanidade_painel_z": "Fig_2Z05_painel_z",
    "phase3L_ciclo_vida_en_ln": "Fig_3L01_ciclo_vida_en_ln",
    "phase3L_discriminantes_heatmap": "Fig_3L02_discriminantes_heatmap",
    "phase3L_duracao_fases_en_ln": "Fig_3L03_duracao_fases_en_ln",
    "phase3L_pca_por_fase": "Fig_3L04_pca_por_fase",
    "phase40_cobertura_dados": "Fig_4002_cobertura_dados",
}


def canonizar(name: str, fase: int | None = None) -> str | None:
    """Converte um nome de figura legado em código canônico Fig_<F><B><NN>.

    Reconhece ``3A1_...``, ``Fig_4C1_...``, ``Fig_401_...`` e variantes de modelo
    (``_rf``/``_xgb``). Devolve ``None`` se o nome não encaixa (ex.: ``phase3L_...``),
    e nesse caso o chamador usa o próprio nome como código de transição.
    """
    stem = name[:-4] if name.lower().endswith(".png") else name
    if stem in _ALIAS:
        return _ALIAS[stem]
    m = _CODE_RE.match(stem)
    if not m:
        return None
    F, B, N, slug = m.group(1), m.group(2), int(m.group(3)), (m.group(4) or "")
    slug = re.sub(r"[^a-z0-9]+", "_", slug.lower()).strip("_")
    # Mantém o descritivo no código: Fig_<F><B><NN>_<descritivo> (uma/duas palavras+).
    return f"Fig_{F}{B}{N:02d}" + (f"_{slug}" if slug else "")


def _slug(s, n: int = 40) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(s)).strip("_").lower()
    return (s or "serie")[:n]


def extrair_dados(fig) -> dict:
    """Extrai os dados efetivamente plotados (linhas, dispersões, imagens) da
    figura para tabelas auditáveis. Best-effort: o que não é vetorial vira meta."""
    tabelas: dict = {}
    for i, ax in enumerate(getattr(fig, "axes", []) or []):
        for j, ln in enumerate(ax.get_lines()):
            x = np.asarray(ln.get_xdata()); y = np.asarray(ln.get_ydata())
            if x.size == 0:
                continue
            lbl = ln.get_label()
            lbl = _slug(lbl) if lbl and not str(lbl).startswith("_") else f"linha{j}"
            try:
                tabelas[f"ax{i}_{lbl}"] = pd.DataFrame({"x": x, "y": y})
            except Exception:
                pass
        for k, im in enumerate(ax.get_images()):
            arr = im.get_array()
            if arr is not None:
                a = np.asarray(arr)
                tabelas[f"ax{i}_imagem{k}"] = pd.DataFrame(
                    a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(-1, 1))
        for k, col in enumerate(getattr(ax, "collections", [])):
            try:
                off = np.asarray(col.get_offsets())
            except Exception:
                off = np.empty((0,))
            if off.ndim == 2 and off.shape[0] > 0:
                d = pd.DataFrame(off[:, :2], columns=["x", "y"])
                try:
                    vals = col.get_array()
                    if vals is not None and len(vals) == len(d):
                        d["valor"] = np.asarray(vals)
                except Exception:
                    pass
                tabelas[f"ax{i}_pontos{k}"] = d
    if not tabelas:
        tabelas["meta"] = pd.DataFrame(
            {"info": ["figura sem dados vetoriais extraiveis (ex.: composicao de imagens)"]})
    return tabelas


def _copiar_retry(tmp: Path, dest: Path, tentativas: int = 5) -> bool:
    """Copia tmp->dest com retries (robusto ao Errno 22 do DrvFs em /mnt/c)."""
    for _ in range(tentativas):
        try:
            shutil.copyfile(tmp, dest)
            if dest.exists():
                return True
        except OSError:
            time.sleep(0.3)
    return False


def cogerar_de_figura(fig, name: str, *, fase: int, bloco: str = "", notebook: str = "") -> Path:
    """Salva a figura sob código canônico e co-gera sua numeric-table.

    Substituto direto do ``save_fig(fig, name)`` legado: padroniza o nome para
    ``Fig_<F><B><NN>`` (quando reconhecível) e grava, por sobreposição, a figura +
    ``numeric-tables/faseN/<codigo>/`` + manifesto. A parte da tabela é best-effort
    (nunca impede a figura). Retorna o caminho do PNG.
    """
    codigo = canonizar(name, fase) or (name[:-4] if name.lower().endswith(".png") else name)
    fase_dir = f"fase{fase}"
    fig_dir = FIG_ROOT / fase_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    png = fig_dir / f"{codigo}.png"

    tmp = Path(tempfile.gettempdir()) / f"{codigo}.png"
    fig.savefig(tmp, dpi=150, bbox_inches="tight")
    if not _copiar_retry(tmp, png):
        raise OSError(f"não consegui gravar a figura {png}")

    try:
        tabelas = extrair_dados(fig)
        tbl_dir = NUM_ROOT / fase_dir / codigo
        if tbl_dir.exists():
            shutil.rmtree(tbl_dir, ignore_errors=True)
        tbl_dir.mkdir(parents=True, exist_ok=True)
        linhas = []
        for nome, df in tabelas.items():
            t = Path(tempfile.gettempdir()) / f"{codigo}__{nome}.csv"
            df.to_csv(t, index=False)
            if _copiar_retry(t, tbl_dir / f"{nome}.csv"):
                table_path = tbl_dir / f"{nome}.csv"
                linhas.append({
                    "tabela": f"{nome}.csv", "linhas": int(df.shape[0]),
                    "colunas": int(df.shape[1]), "sha256": _sha256_file(table_path),
                    "dataframe_sha256": _sha256_df(df),
                    "schema_json": json.dumps({c: str(t) for c, t in df.dtypes.items()}, sort_keys=True),
                    "semantic_source": False,
                    "audit_level": "render_extraction_only",
                    "run_id": "",
                })
        tman = Path(tempfile.gettempdir()) / f"{codigo}__manifest.csv"
        pd.DataFrame(linhas).to_csv(tman, index=False)
        _copiar_retry(tman, tbl_dir / "manifest.csv")
        trd = Path(tempfile.gettempdir()) / f"{codigo}__README.md"
        trd.write_text(
            f"# {codigo}\n\n- **Figura:** `../../figures/{fase_dir}/{codigo}.png`\n"
            f"- **Origem (nome legado):** `{name}`\n- **Notebook:** `{notebook}`\n"
            f"- **Atualizado:** {_now()}\n\n## Tabelas (dados plotados)\n" +
            "\n".join(f"- `{l['tabela']}` ({l['linhas']}x{l['colunas']})" for l in linhas) + "\n",
            encoding="utf-8")
        _copiar_retry(trd, tbl_dir / "README.md")

        row = {"codigo": codigo, "fase": fase, "bloco": bloco or (codigo[4] if len(codigo) > 4 else ""),
               "arquivo": f"{fase_dir}/{codigo}.png", "notebook": notebook, "titulo": name,
               "hipotese": "", "descricao": "auto (dados plotados)",
               "tabelas": ";".join(f"{n}.csv" for n in tabelas), "n_tabelas": len(tabelas),
               "audit_level": "render_extraction_only", "run_id": "",
               "atualizado_em": _now()}
        if MANIFEST.exists():
            man = _align_manifest_columns(pd.read_csv(MANIFEST))
            man = man[man["codigo"] != codigo]
            man = pd.concat([man, pd.DataFrame([row])], ignore_index=True)
        else:
            MANIFEST.parent.mkdir(parents=True, exist_ok=True)
            man = pd.DataFrame([row], columns=MANIFEST_COLS)
        tmn = Path(tempfile.gettempdir()) / "figuras_manifesto.csv"
        man.sort_values("codigo").to_csv(tmn, index=False)
        _copiar_retry(tmn, MANIFEST)
    except Exception as exc:  # nunca quebra a figura por causa da tabela
        print(f"[aviso] numeric-table de {codigo} nao gerada: {exc}")

    print(f"[figura+tabela] {png.relative_to(ROOT)}  <-  {name}")
    return png
