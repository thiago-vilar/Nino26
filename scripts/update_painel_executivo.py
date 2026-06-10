from __future__ import annotations

import json
import platform
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

OUTPUT_PATH = ROOT / "painel_executivo.md"
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    output = (result.stdout or result.stderr).strip()
    return output or "indisponivel"


def now_sp() -> str:
    try:
        tz = ZoneInfo("America/Sao_Paulo")
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def format_bytes(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.2f} GB"
    if value >= 1024**2:
        return f"{value / 1024**2:.2f} MB"
    if value >= 1024:
        return f"{value / 1024:.2f} KB"
    return f"{value} B"


def read_text_auto(path: Path) -> str:
    data = path.read_bytes()
    if data.startswith(b"\xff\xfe") or data.startswith(b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    if data.startswith(b"\xef\xbb\xbf"):
        return data.decode("utf-8-sig", errors="replace")
    for encoding in ("utf-8", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def real_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return [p for p in path.rglob("*") if p.is_file() and p.name != ".gitkeep"]


def file_count_and_size(path: Path) -> tuple[int, int]:
    files = real_files(path)
    return len(files), sum(p.stat().st_size for p in files)


def years_from_files(path: Path, pattern: str) -> list[int]:
    if not path.exists():
        return []
    years: list[int] = []
    for item in path.glob(pattern):
        match = YEAR_RE.search(item.name)
        if match:
            years.append(int(match.group(0)))
    return sorted(set(years))


def years_summary(years: list[int]) -> str:
    if not years:
        return "nenhum ano local"
    missing = [year for year in range(years[0], years[-1] + 1) if year not in years]
    gap = "sem lacunas internas" if not missing else f"lacunas: {', '.join(map(str, missing[:12]))}"
    if len(missing) > 12:
        gap += "..."
    return f"{years[0]}-{years[-1]} ({len(years)} anos; {gap})"


def zarr_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted([p for p in path.iterdir() if p.is_dir() and p.name.endswith(".zarr")], key=lambda p: p.name)


def audit_rows() -> list[dict[str, object]]:
    ledger = ROOT / "data" / "audit" / "ledger.jsonl"
    if not ledger.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def latest_source_latency(rows: list[dict[str, object]], dataset: str) -> str:
    candidates = [row for row in rows if row.get("dataset") == dataset and row.get("available_through")]
    if not candidates:
        return "sem registro"
    latest = sorted(candidates, key=lambda row: str(row.get("timestamp_utc", "")))[-1]
    return f"{latest.get('available_through')} (latencia {latest.get('latency_days')} dias)"


def latest_download_log() -> dict[str, str]:
    log_dirs = [
        ROOT / "data" / "state" / "pipeline_reports",
        ROOT / "data" / "state" / "download_logs",
    ]
    logs = sorted(
        [log for log_dir in log_dirs if log_dir.exists() for log in log_dir.glob("*.log")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        return {"arquivo": "sem log", "ultimo_download": "sem registro", "erro": "sem registro"}

    latest = logs[0]
    lines = read_text_auto(latest).splitlines()
    last_download = next((line.strip() for line in reversed(lines) if "downloaded:" in line), "sem registro")
    error = "sem erro no trecho final"
    for idx in range(len(lines) - 1, -1, -1):
        line = lines[idx]
        if "HTTPError" in line or "Server Error" in line or "Traceback" in line:
            error = line.strip()
            if error.endswith("url:") and idx + 1 < len(lines):
                error = f"{error} {lines[idx + 1].strip()}"
            break
    return {"arquivo": str(latest.relative_to(ROOT)), "ultimo_download": last_download, "erro": error}


def cds_status() -> str:
    try:
        from nino_brasil.data.credentials import cds_credentials_status

        status = cds_credentials_status()
        return f"ready={status.get('ready')}; CDS_API_URL={status.get('CDS_API_URL')}; CDS_API_KEY={status.get('CDS_API_KEY')}"
    except Exception as exc:
        return f"indisponivel ({type(exc).__name__})"


def table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell).replace("|", "/") for cell in row) + " |")
    return "\n".join(lines)


def git_summary() -> str:
    status = run_git(["status", "--short", "--branch"])
    remote = run_git(["remote", "-v"])
    commits = run_git(["log", "--oneline", "--decorate", "-5"])
    return "\n".join(
        [
            "```text",
            status,
            "",
            remote,
            "",
            commits,
            "```",
        ]
    )


def phase_rows() -> list[list[str]]:
    chirps_years = years_from_files(ROOT / "data/raw/chirps/p25", "*.nc")
    oisst_years = years_from_files(ROOT / "data/raw/cpc_noaa/oisst", "*.nc")
    regridded = zarr_dirs(ROOT / "data/processed/zarr/regridded")
    modeling = zarr_dirs(ROOT / "data/processed/zarr/modeling")
    docs_index = ROOT / "docs" / "index.html"

    return [
        [
            "Fase 1",
            "Fundacao, Git, catalogo, cache bruto, ingestao ano a ano, variavel por variavel quando aplicavel.",
            "Dados essenciais baixaveis/reprodutiveis e estrutura versionada.",
            f"Em andamento. CHIRPS {years_summary(chirps_years)}; OISST {years_summary(oisst_years)}; ERA5/ORAS/CTD ainda sem massa local real.",
        ],
        [
            "Fase 2",
            "Padronizacao, QC, anomalias, climatologia sem vazamento, percentis, lags e regrid 0.25 grau.",
            "Cubos Zarr reconciliados em data/processed/zarr/regridded/ para todos os insumos de modelagem.",
            f"Validada parcialmente com {len(regridded)} stores regridded: {', '.join(p.name for p in regridded) or 'nenhum'}.",
        ],
        [
            "Fase 3",
            "ML + XAI: Ridge, Random Forest, XGBoost, LightGBM, walk-forward, permutation importance, SHAP e pesos por grupo.",
            "Stores Zarr de metricas, previsoes, importancias e pesos por grupo.",
            f"Codigo pronto para execucao; stores de modelagem locais: {', '.join(p.name for p in modeling) or 'nenhum'}",
        ],
        [
            "Fase 4",
            "Redes Neurais + XAI: CNN, ConvLSTM, U-Net, Transformer espaco-temporal, saliency, occlusion e attention maps.",
            "Stores Zarr de treino/inferencia neural, XAI neural e comparacao contra Fase 3.",
            "Planejada; ainda nao executada nesta maquina.",
        ],
        [
            "Fase 5",
            "Teste Memory Caching (Google Research, arXiv:2602.24281): RNNs com memoria crescente via cache de estados, variantes Residual/Gated/Soup/SSC sob o mesmo walk-forward.",
            "Stores Zarr de runs, comparacao de skill contra Fases 3/4 e relatorio skill x eficiencia.",
            "Planejada; aguarda campeoes das Fases 3 e 4.",
        ],
        [
            "Fase 6",
            "Publicacao e operacao: docs/, GitHub Pages, relatorios automaticos, drift, recalibracao e painel publico.",
            "Produto publicado/atualizavel em docs/ com rotina recorrente.",
            "Esqueleto local existe." if docs_index.exists() else "docs/index.html ausente.",
        ],
    ]


def data_rows() -> list[list[str]]:
    paths = [
        ("IBGE bruto", ROOT / "data/raw/ibge"),
        ("IBGE intermediario", ROOT / "data/interim/ibge"),
        ("CHIRPS p25 bruto", ROOT / "data/raw/chirps/p25"),
        ("OISST bruto", ROOT / "data/raw/cpc_noaa/oisst"),
        ("ERA5 bruto", ROOT / "data/raw/era5"),
        ("ORAS bruto", ROOT / "data/raw/oras"),
        ("CTD/WOD bruto", ROOT / "data/raw/ctd_noaa/wod"),
        ("Zarr processado", ROOT / "data/processed/zarr"),
        ("Zarr regridded", ROOT / "data/processed/zarr/regridded"),
        ("Zarr diagnosticos", ROOT / "data/processed/zarr/distributions"),
        ("Zarr modelagem", ROOT / "data/processed/zarr/modeling"),
    ]
    rows: list[list[str]] = []
    for label, path in paths:
        count, size = file_count_and_size(path)
        rows.append([label, str(path.relative_to(ROOT)), str(count), format_bytes(size)])
    return rows


def next_steps() -> list[str]:
    steps = []
    oisst_years = years_from_files(ROOT / "data/raw/cpc_noaa/oisst", "*.nc")
    if oisst_years:
        steps.append("Conferir o plano: python scripts/run_full_download_pipeline.py")
        steps.append("Rodar o pipeline completo automatico ano a ano: python scripts/run_full_download_pipeline.py --execute")
    else:
        steps.append("Conferir o plano: python scripts/run_full_download_pipeline.py")
        steps.append("Iniciar o pipeline completo automatico ano a ano: python scripts/run_full_download_pipeline.py --execute")

    regridded = {p.name for p in zarr_dirs(ROOT / "data/processed/zarr/regridded")}
    if len(regridded) <= 2:
        steps.append("Escalar Fase 2: converter anos brutos CHIRPS/OISST para Zarr e regridar alem de 1981.")
    steps.extend(
        [
            "Ingerir ERA5/ORAS5 por variavel, mantendo cache bruto e Zarr diario por variavel.",
            "Rodar Fase 3 somente depois que os preditores e alvo estiverem reconciliados em Zarr 0.25 grau.",
            "Manter este painel local ignorado pelo Git: atualize com python scripts/update_painel_executivo.py.",
        ]
    )
    return steps


def build_markdown() -> str:
    rows = audit_rows()
    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    download_log = latest_download_log()
    usage = shutil.disk_usage(ROOT)
    chirps_years = years_from_files(ROOT / "data/raw/chirps/p25", "*.nc")
    oisst_years = years_from_files(ROOT / "data/raw/cpc_noaa/oisst", "*.nc")
    regridded = zarr_dirs(ROOT / "data/processed/zarr/regridded")
    diagnostics = zarr_dirs(ROOT / "data/processed/zarr/distributions")
    modeling = zarr_dirs(ROOT / "data/processed/zarr/modeling")
    git_lines = run_git(["status", "--short", "--branch"]).splitlines()
    git_head = git_lines[0] if git_lines else "indisponivel"
    git_changes = max(len(git_lines) - 1, 0)
    next_action = next_steps()[0]

    lines = [
        "# Painel executivo NINO-BRASIL",
        "",
        "> Painel local, gerado automaticamente e ignorado pelo Git. Cada maquina deve ter o seu proprio painel.",
        "",
        "## Onde estamos agora",
        "",
        f"- Atualizado em: {now_sp()}",
        f"- Maquina consultada: {platform.node() or 'indisponivel'} ({platform.system()} {platform.release()})",
        f"- Posicao do projeto: transicao entre Fase 1 e Fase 2.",
        f"- Fase 1 ainda nao fechou porque OISST parou em {max(oisst_years) if oisst_years else 'nenhum ano'} e ERA5/ORAS/CTD ainda nao foram baixados nesta maquina.",
        f"- Fase 2 foi validada tecnicamente, mas so para 1981: {', '.join(p.name for p in regridded) or 'nenhum Zarr regridado'}.",
        "- Fase 3 ainda nao tem execucao historica real; o codigo existe, mas faltam os cubos Zarr completos.",
        f"- Proxima acao objetiva: {next_action}",
        "",
        "## Fase em execucao",
        "",
        "### Execucao principal: Fase 1 - Base local e ingestao bruta",
        "",
        "A fase em execucao nesta maquina ainda e a Fase 1. Existe uma validacao tecnica inicial da Fase 2, mas a Fase 1 nao pode ser considerada completa enquanto a base bruta essencial nao estiver fechada.",
        "",
        "**Ja foi feito**",
        "",
        "- Estrutura local do projeto criada.",
        "- Repositorio Git configurado.",
        "- Catalogo, configuracoes, metodologia e scripts principais existem.",
        "- Credenciais CDS estao configuradas e prontas.",
        "- Politica operacional definida: ano a ano, prioridade diario/subdiario, depois semanal, depois mensal.",
        "- Cache bruto antes da transformacao e Zarr diario por variavel quando a fonte permite.",
        "- IBGE foi baixado e extraido localmente.",
        "- CHIRPS e OISST ja foram testados no fluxo bruto.",
        "- Conversao/regrid piloto de 1981 foi validada para CHIRPS e OISST.",
        "",
        "**Ja foi baixado nesta maquina**",
        "",
        f"- CHIRPS p25: {years_summary(chirps_years)}.",
        f"- NOAA OISST: {years_summary(oisst_years)}.",
        "- IBGE: limites de UF e municipios.",
        "- ERA5: nenhum dado real detectado.",
        "- ORAS5: nenhum dado real detectado.",
        "- CTD/WOD: nenhum dado real detectado.",
        "",
        "**Falta baixar para completar a Fase 1**",
        "",
        f"- NOAA OISST a partir de {max(oisst_years) + 1 if oisst_years else 1981} ate o ultimo dado disponivel registrado ({latest_source_latency(rows, 'noaa_oisst')}).",
        f"- ERA5 single e pressure levels, ano/mes/regiao/variavel desde 1981 ate o ultimo dado disponivel registrado ({latest_source_latency(rows, 'era5')}).",
        "- ORAS5 subsuperficial, ano/mes/variavel desde 1981 ate o ultimo dado disponivel por fonte.",
        f"- CTD/WOD para validacao vertical desde 1981 ate o ultimo dado disponivel registrado ({latest_source_latency(rows, 'noaa_wod_ctd')}).",
        "",
        "**Falta para fechar a Fase 1**",
        "",
        "- Retomar o download do OISST exatamente no ponto de parada.",
        "- Confirmar cobertura anual/local de cada fonte baixada.",
        "- Registrar no ledger os downloads concluidos e eventuais falhas.",
        "- Manter dados grandes fora do Git.",
        "- So depois disso declarar a base bruta pronta para escalar a Fase 2.",
        "",
        "**Ponto de parada atual**",
        "",
        f"- Ultimo arquivo bruto concluido: {download_log['ultimo_download']}",
        f"- Ultimo erro registrado: {download_log['erro']}",
        "",
        "## Dificuldades, aprendizados e decisoes abertas",
        "",
        "Esta secao registra obstaculos reais encontrados nesta maquina. A ideia e transformar erro em criterio de decisao, nao apenas listar falhas.",
        "",
        "### 1. Download OISST interrompido",
        "",
        f"- Sintoma: {download_log['erro']}",
        "- Impacto: a Fase 1 nao fecha, porque OISST e a fonte principal de SST/SSTA diaria.",
        "- Aprendizado: downloads longos devem ser retomados por ano, sem reiniciar tudo que ja foi baixado.",
        f"- Caminho de superacao: retomar em {max(oisst_years) + 1 if oisst_years else 1981} com comando anual ou em blocos menores.",
        "- Decisao aberta: incluir retry/backoff automatico no downloader HTTP para erros 5xx da NOAA.",
        "",
        "### 2. Regrid/Zarr exigiu ajuste de chunks",
        "",
        "- Sintoma: o ledger registrou erro anterior de chunking no regrid do OISST 1981.",
        "- Impacto: sem chunks uniformes, o Zarr nao grava corretamente e a Fase 2 para.",
        "- Aprendizado: todo dado regridado precisa passar por plano de chunks antes de gravar em Zarr.",
        "- Caminho de superacao: manter a funcao de chunk seguro no pipeline e aplicar antes de todo `to_zarr`.",
        "- Decisao aberta: padronizar tamanhos de chunk por tipo de dado: diario, mensal, perfil vertical e tabela.",
        "",
        "### 3. ETL de ERA5/ORAS/CTD ainda nao foi exercitado nesta maquina",
        "",
        "- Sintoma: nao ha massa local real dessas fontes; portanto ainda nao apareceram erros de ETL nelas.",
        "- Impacto: o risco tecnico dessas fontes ainda esta oculto.",
        "- Aprendizado: nao declarar a Fase 1 completa apenas porque CHIRPS/OISST estao parcialmente baixados.",
        "- Caminho de superacao: testar ERA5 e ORAS5 primeiro em janela pequena e variavel unica antes de escalar 1981-presente.",
        "- Decisao aberta: escolher uma janela-piloto curta para validar ETL completo, por exemplo 1981-01 com uma variavel ERA5 e uma ORAS5.",
        "",
        "### 4. Formato de saida foi consolidado em Zarr",
        "",
        "- Sintoma: havia referencias antigas a Parquet para metricas, previsoes e diagnosticos.",
        "- Impacto: misturar formatos confundiria leitura, armazenamento e sincronizacao entre maquinas.",
        "- Aprendizado: o projeto deve usar Zarr como formato padrao tambem para tabelas derivadas.",
        "- Caminho de superacao: `model_pipeline.py` e `data_pipeline.py` agora gravam tabelas como stores `.zarr`.",
        "- Decisao aberta: definir convencao final de nomes para stores de modelagem, diagnostico e XAI neural.",
        "",
        "### 5. Painel deve ser local por maquina",
        "",
        "- Sintoma: casa e UFPE terao estados de dados diferentes.",
        "- Impacto: versionar o painel confundiria o status de uma maquina com o da outra.",
        "- Aprendizado: o painel executivo precisa ser gerado localmente e ignorado pelo Git.",
        "- Caminho de superacao: `painel_executivo.md` esta no `.gitignore`; o gerador versionado e `scripts/update_painel_executivo.py`.",
        "- Decisao aberta: rodar o gerador automaticamente ao fim de cada pipeline ou manter atualizacao manual sob demanda.",
        "",
        "## Fases e grau de execucao",
        "",
        "> Grau estimado e operacional: mede o quanto esta maquina permite avancar na fase, nao e porcentagem cientifica final.",
        "",
        "### Fase 1 - Base local e ingestao bruta",
        "",
        "- Grau estimado: 55%.",
        "- Objetivo: deixar Git, estrutura, catalogo, credenciais e dados brutos essenciais prontos para reproducao local.",
        f"- Concluido: IBGE local; CHIRPS p25 {years_summary(chirps_years)}; OISST {years_summary(oisst_years)}; CDS pronto.",
        "- Falta: completar OISST ate o ultimo disponivel; baixar/ingerir ERA5 e ORAS5 por variavel; baixar/ingerir CTD/WOD.",
        f"- Proximo ano OISST pendente nesta maquina: {max(oisst_years) + 1 if oisst_years else 1981}.",
        f"- Ultimo erro de download registrado: {download_log['erro']}",
        f"- Proxima acao objetiva: {next_action}",
        "",
        "### Fase 2 - Padronizacao, anomalias, lags e regrid",
        "",
        "- Grau estimado: 15%.",
        "- Objetivo: transformar cache bruto em cubos Zarr diarios padronizados, sem vazamento, na grade comum 0.25 grau.",
        f"- Concluido: prova funcional para 1981 com {', '.join(p.name for p in regridded) or 'nenhum store regridado'}.",
        f"- Diagnostico gerado: {', '.join(p.name for p in diagnostics) or 'nenhum diagnostico'}.",
        "- Falta: rodar conversao anual CHIRPS/OISST bruto -> Zarr diario -> regrid para todos os anos baixados; incluir ERA5/ORAS5 diarios por variavel.",
        "- Proximo passo: escalar o processamento alem de 1981 depois de retomar/completar OISST.",
        "",
        "### Fase 3 - ML + XAI",
        "",
        "- Grau estimado: 10%.",
        "- Objetivo: rodar Ridge, Random Forest, XGBoost/LightGBM, walk-forward, permutation importance, SHAP e pesos por grupo.",
        "- Concluido: codigo e testes da modelagem classica existem.",
        f"- Artefatos locais de modelagem: {', '.join(p.name for p in modeling) or 'nenhum'}.",
        "- Falta: executar a modelagem historica real sobre cubos Zarr completos e reconciliados.",
        "- Proximo passo: aguardar Fase 2 completa o suficiente para gerar matriz de modelagem confiavel.",
        "",
        "### Fase 4 - Redes Neurais + XAI",
        "",
        "- Grau estimado: 0%.",
        "- Objetivo: CNN, ConvLSTM, U-Net, Transformer espaco-temporal e XAI neural com saliency, occlusion e attention maps.",
        "- Concluido: fase definida metodologicamente.",
        "- Falta: arquitetura neural, treino, inferencia, XAI neural e comparacao contra Fase 3.",
        "- Proximo passo: so iniciar depois que Fase 3 tiver baseline operacional.",
        "",
        "### Fase 5 - Teste com Memory Caching (Google Research)",
        "",
        "- Grau estimado: 0%.",
        "- Objetivo: avaliar a tecnica de test-time memorization Memory Caching (arXiv:2602.24281) - cache de estados de memoria por segmento, variantes Residual, Gated Residual, Memory Soup e Sparse Selective Caching - sob o mesmo protocolo walk-forward, medindo skill x eficiencia (O(L) a O(L^2)).",
        "- Concluido: fase definida metodologicamente no Plano Diretor.",
        "- Falta: implementacao dos backbones recorrentes com cache, runs comparativos e relatorio de trade-off.",
        "- Proximo passo: so iniciar depois dos campeoes das Fases 3 e 4.",
        "",
        "### Fase 6 - Publicacao e operacao",
        "",
        "- Grau estimado: 5%.",
        "- Objetivo: publicar produto em docs/GitHub Pages, manter rotina recorrente, relatorios, comparacao previsao-observado, drift e recalibracao.",
        "- Concluido: esqueleto local em docs/ e mapa de smoke test.",
        "- Falta: produto cientifico, mapas finais, relatorios e rotina automatizada.",
        "- Proximo passo: esperar saidas reais das Fases 3, 4 e 5.",
        "",
        "## Metodologia que guia todas as fases",
        "",
        "- Frequencia: diaria.",
        "- Prioridade temporal de download/processamento: 1) diario ou subdiario convertido para diario; 2) semanal convertido para diario quando houver fonte ativa; 3) mensal convertido para diario.",
        "- Cache: todo bruto fica em `data/raw/` antes da transformacao.",
        "- Granularidade: ERA5 e ORAS5 sao baixados e transformados variavel por variavel.",
        "- Periodo: 1981-01-01 ate o ultimo dado disponivel por fonte.",
        "- Horizontes: 1 a 24 semanas, ou 7 a 168 dias.",
        "- Validacao: walk-forward em blocos temporais; split aleatorio nao entra.",
        "- Anti-vazamento: climatologia, desvio padrao e percentis P10/P25/P75/P90 sempre estimados no treino e reaplicados em validacao/teste.",
        "- Grade: 0.25 grau nas Fases 1 a 5, longitude 0_360.",
        "- Dados: CHIRPS p25 para precipitacao, OISST para SST/SSTA diaria, ORAS5 para memoria subsuperficial.",
        "- Saida persistida: Zarr. Parquet nao e o padrao deste projeto.",
        "",
        "## Evidencias locais minimas",
        "",
        table(
            ["Item", "Estado nesta maquina"],
            [
                ["CHIRPS p25 bruto", years_summary(chirps_years)],
                ["OISST bruto", years_summary(oisst_years)],
                ["Disponibilidade CHIRPS no ledger", latest_source_latency(rows, "chirps")],
                ["Disponibilidade OISST no ledger", latest_source_latency(rows, "noaa_oisst")],
                ["Zarr regridded", ", ".join(p.name for p in regridded) or "nenhum"],
                ["Zarr diagnosticos", ", ".join(p.name for p in diagnostics) or "nenhum"],
                ["Zarr modelagem", ", ".join(p.name for p in modeling) or "nenhum"],
                ["Auditoria", f"{len(rows)} eventos; " + (", ".join(f"{key}: {value}" for key, value in sorted(status_counts.items())) or "sem status")],
                ["CDS", cds_status()],
            ],
        ),
        "",
        "## Ultima execucao relevante",
        "",
        f"- Log: {download_log['arquivo']}",
        f"- Ultimo download concluido: {download_log['ultimo_download']}",
        f"- Erro que interrompeu a execucao: {download_log['erro']}",
        "",
        "## Proximas acoes",
        "",
        *[f"{idx}. {step}" for idx, step in enumerate(next_steps(), start=1)],
        "",
        "## Rodape tecnico",
        "",
        f"- Workspace: {ROOT}",
        f"- Disco livre: {format_bytes(usage.free)}",
        f"- Git: {git_head}; mudancas locais listadas: {git_changes}",
        f"- Arquivo: {OUTPUT_PATH.relative_to(ROOT)}",
        f"- Atualizar painel:",
        "```powershell",
        ".\\.venv\\Scripts\\python .\\scripts\\update_painel_executivo.py",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUTPUT_PATH.write_text(build_markdown(), encoding="utf-8")
    print(f"painel executivo atualizado: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
