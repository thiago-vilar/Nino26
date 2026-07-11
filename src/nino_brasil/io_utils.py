"""Escrita atomica de artefatos numericos.

Motivacao (parecer 2026-07-10): `phase6_skill_rf.csv` foi encontrado truncado no
meio de uma linha (7 de 10 colunas), invalidando a auditoria da Fase 6. Escrever
em arquivo temporario no MESMO diretorio e promover com `os.replace` garante que
o leitor nunca veja um CSV parcial: ou o arquivo antigo, ou o novo completo.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd


def write_csv_atomic(frame: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Grava ``frame`` em ``path`` de forma atomica (tmp + ``os.replace``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="") as stream:
            frame.to_csv(stream, index=index)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return path
