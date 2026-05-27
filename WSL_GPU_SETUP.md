# Projeto de previsão climática El NINO e seus impactos no Brasil

**Programa de Pós Graduação em Oceanografia UFPE**  
**Área:** Oceanografia Física  
**Responsável:** Thiago Vilar  

## Rodando no VSCode com WSL e GPU

Este projeto pode ser executado no Windows, mas o modo recomendado para treino com placa de vídeo é:

```text
Windows + WSL2 + Ubuntu + VSCode Remote WSL + PyTorch com CUDA
```

## 1. Requisitos

- Windows com WSL2 habilitado.
- Distribuição Linux no WSL, preferencialmente Ubuntu.
- Driver NVIDIA instalado no Windows com suporte a WSL.
- VSCode instalado no Windows.
- Extensão `WSL` instalada no VSCode.
- Python 3.10 ou superior dentro do WSL.
- Git dentro do WSL.

## 2. Verificar GPU no WSL

Abra o terminal do Ubuntu/WSL e rode:

```bash
nvidia-smi
```

Se a GPU aparecer, o WSL está enxergando a placa.

Se `nvidia-smi` não funcionar, o problema está antes do Python: driver NVIDIA, WSL2 ou instalação do Ubuntu.

## 3. Onde deixar o projeto

Para melhor desempenho, mantenha o projeto dentro do sistema de arquivos do Linux:

```text
~/projects/Nino26
```

Evite treinar modelos pesados diretamente em:

```text
/mnt/c/...
```

O caminho `/mnt/c/DEV/NINO26` funciona, mas costuma ser mais lento para muitos arquivos pequenos e grandes volumes de dados.

## 4. Clonar o projeto no WSL

No terminal do Ubuntu/WSL:

```bash
mkdir -p ~/projects
cd ~/projects
git clone git@github.com:thiago-vilar/Nino26.git
cd Nino26
```

Abra no VSCode:

```bash
code .
```

O VSCode deve abrir em modo WSL.

## 5. Criar ambiente Python no WSL

Dentro do terminal WSL do VSCode:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip python3-dev build-essential
sudo apt install -y gdal-bin libgdal-dev libproj-dev proj-data proj-bin libgeos-dev libspatialindex-dev
```

Depois:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Se o clone por SSH ainda não estiver configurado dentro do WSL, use HTTPS:

```bash
git clone https://github.com/thiago-vilar/Nino26.git
```

## 6. Verificar PyTorch e CUDA

Rode:

```bash
python scripts/check_gpu.py
```

Saída desejada:

```text
torch cuda available: True
```

Se aparecer `False`, o projeto ainda roda em CPU, mas o treino dos modelos neurais será mais lento.

Se `nvidia-smi` funcionar e mesmo assim o PyTorch mostrar `False`, reinstale o PyTorch dentro da `.venv` usando o comando oficial do seletor:

```text
https://pytorch.org/get-started/locally/
```

Selecione:

```text
Linux
Pip
Python
CUDA
```

## 7. Rodar o teste inicial

```bash
python scripts/smoke_test.py
```

Esse teste usa dados sintéticos e confirma:

- cálculo de anomalias.
- criação de lags temporais.
- cálculo de variáveis oceânicas sintéticas.
- geração de mapa em `docs/assets/maps/`.

## 8. Organização dos dados no WSL

Os dados brutos devem entrar em:

```text
data/raw/oras/
data/raw/ctd_noaa/
data/raw/era5/
data/raw/cpc_noaa/
data/raw/ibge/
```

Para grandes volumes, mantenha `data/` dentro do WSL ou use um disco rápido montado no Linux.

## 9. Orientação prática

Use CPU para:

- download.
- padronização.
- recorte espacial.
- cálculo de anomalias.
- modelos baseline.

Use GPU para:

- CNN.
- ConvLSTM.
- U-Net.
- Transformer espaço-temporal.
- testes com tensores grandes.

O ganho da GPU aparece principalmente na etapa de Machine Learning, não no download dos dados.
