# Ambiente VS Code, kernels e GPU

Diagnostico local em `C:\DEV\NINO26`.

## Kernel para a Fase 4 no VS Code

Use este kernel ao abrir os notebooks em `notebooks/fase4_nino/` ou
`notebooks/fase4_nina/` pelo VS Code no Windows:

```text
Python 3.12 (.venv NINO26)
```

Detalhes:

| Item | Valor |
|---|---|
| Kernelspec | `nino-brasil` |
| Python | `C:\DEV\NINO26\.venv\Scripts\python.exe` |
| Python version | `3.12.3` |
| Uso recomendado | Fase 4, auditoria, tabelas, mapas, Cartopy, estatistica CPU |
| GPU nesse kernel | Nao |

Esse kernel esta configurado em `.vscode/settings.json`.

No seletor de kernel do VS Code Windows, ignore qualquer entrada parecida com:

```text
.venv-wsl (broken)
```

Esse ambiente e Linux/WSL. Ele aparece como quebrado no Windows porque o VS Code
tenta localizar um `python.exe` dentro de um ambiente que foi criado para Linux.
Para a Fase 4 no Windows, escolha a entrada que aponta para:

```text
C:\DEV\NINO26\.venv\Scripts\python.exe
```

## Estado da GPU no Windows nativo

O Windows enxerga a placa NVIDIA:

| Item | Valor |
|---|---|
| GPU | `NVIDIA Quadro T1000 with Max-Q Design` |
| Memoria | `4096 MiB` |
| Driver | `595.95` |
| Driver CUDA runtime | `13.2` |

Mas a `.venv` do Windows esta com PyTorch CPU-only:

| Checagem | Resultado |
|---|---|
| `torch.__version__` | `2.12.0+cpu` |
| `torch.version.cuda` | `None` |
| `torch.cuda.is_available()` | `False` |
| `nvcc` no PATH | Nao encontrado |

Conclusao: a `.venv` Windows e correta para Fase 4, mas nao e o caminho atual
para treino neural com GPU.

## Kernel GPU para proximas fases

Para as fases com treino neural, use VS Code Remote WSL e selecione:

```text
Python 3 (.venv-wsl NINO26 GPU)
```

Detalhes:

| Item | Valor |
|---|---|
| WSL distro | `Ubuntu-22.04` |
| Kernelspec | `nino26-wsl-gpu` |
| Python | `/mnt/c/DEV/NINO26/.venv-wsl/bin/python` |
| Python version | `3.10.12` |
| PyTorch | `2.12.0+cu130` |
| `torch.cuda.is_available()` | `True` |
| GPU via PyTorch | `Quadro T1000 with Max-Q Design` |
| Memoria CUDA reportada | `4294639616 bytes` |

O script oficial confirma:

```text
python scripts/check_gpu.py
torch installed: True
torch version: 2.12.0+cu130
torch cuda available: True
cuda device count: 1
cuda device name: Quadro T1000 with Max-Q Design
```

## Recomendacao pratica

| Fase | Ambiente recomendado | Kernel |
|---|---|---|
| Fases 3–6 (estatística/RF/XGBoost) | VS Code no Windows | `nino-brasil` / `Python 3.12 (.venv NINO26)` |
| Fases 7–8 (ConvLSTM) | VS Code Remote WSL para treino | `nino26-wsl-gpu` |

A Quadro T1000 tem aproximadamente 4 GB de VRAM. Para as fases neurais, usar
batch pequeno, `float16`/mixed precision quando possivel, modelos compactos e
checagem frequente de memoria.
