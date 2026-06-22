# Parecer 3 — Pré-Cálculos Físicos para ML/RN (Ranqueado por Importância)
**NINO-BRASIL — 2ª Lei de Newton aplicada ao pipeline de features**

Responsável: Thiago Vilar — PPGO/UFPE, Oceanografia Física

---

## Argumento central

ERA5 e ORAS5 são soluções numéricas das equações primitivas (Newton aplicado a fluidos). O projeto já consome essas variáveis, mas em *estados instantâneos*. O que falta são os **termos dinâmicos** — derivadas temporais e espaciais que expressam os processos causais do ENSO. Modelos de ML não derivam isso automaticamente a partir de séries brutas. O projeto pode ter as variáveis certas e ainda assim errar o mecanismo.

---

## Rank 1 — CRÍTICOS (lacunas conceituais, não refinamentos)

### 🔴 PC-5 · Warm Water Volume (WWV)

**Por que é crítico:** É o preditor de lead longo mais robusto do ENSO. Mede o volume de água quente acima dos 20°C em toda a bacia equatorial — não apenas num box. O projeto mede D20 pontualmente no Niño 3.4, mas a física do recarregamento é uma integral espacial. WWV lidera SSTA em 2–4 estações. Sem ele, o modelo vai errar sistematicamente a fase de carregamento que precede as secas do NEB.

**Equação:**
```
WWV(t) = média[D20(t)]  em  (5°N–5°S, 120°E–80°W)
```

**Código — adicionar em `ocean_heat.py`:**
```python
def warm_water_volume(
    d20: xr.DataArray,
    lat_bounds: tuple = (-5.0, 5.0),
    lon_bounds: tuple = (120.0, 280.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """WWV — proxy do ciclo de recarregamento do ENSO (Jin 1997)."""
    d20_eq = d20.sel({lat_name: slice(*lat_bounds), lon_name: slice(*lon_bounds)})
    wwv = d20_eq.mean([lat_name, lon_name])
    wwv.attrs.update({"units": "m", "physics": "recharge_oscillator_wwv"})
    return wwv
```

**Custo:** zero. ORAS5 já existe no projeto.

---

### 🔴 PC-6 · Inclinação Zonal da Termoclina (Tilt)

**Por que é crítico:** D20 num único box não distingue El Niño maduro de El Niño em formação de La Niña — três regimes com comportamentos opostos sobre o Brasil. O Tilt (D20_leste − D20_oeste) faz isso com um único escalar. É a assinatura direta do feedback de Bjerknes.

**Equação:**
```
Tilt = D20(5°N–5°S, 90°W–140°W)  −  D20(5°N–5°S, 140°E–180°E)

Positivo → La Niña / estado neutro
Zero     → evento em desenvolvimento
Negativo → El Niño maduro
```

**Código — adicionar em `thermocline.py`:**
```python
def thermocline_tilt(
    d20: xr.DataArray,
    lat_bounds: tuple = (-5.0, 5.0),
    lon_east: tuple = (220.0, 270.0),
    lon_west: tuple = (140.0, 180.0),
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> xr.DataArray:
    """Inclinação zonal da termoclina — fase do ciclo ENSO."""
    sel = {lat_name: slice(*lat_bounds)}
    east = d20.sel({**sel, lon_name: slice(*lon_east)}).mean([lat_name, lon_name])
    west = d20.sel({**sel, lon_name: slice(*lon_west)}).mean([lat_name, lon_name])
    tilt = east - west
    tilt.attrs.update({"units": "m", "physics": "bjerknes_tilt"})
    return tilt
```

**Custo:** zero. ORAS5 já existe no projeto.

---

### 🔴 PC-4 · Rotacional de Tensão de Vento / Bombeamento de Ekman

**Por que é crítico:** Não existe nenhum proxy disto no pipeline atual. É a conexão mecânica direta entre o campo de ventos (Newton na atmosfera) e o deslocamento da termoclina (Newton no oceano). `u850` captura vento médio — este captura a *estrutura espacial* do vento que força o oceano para cima ou para baixo.

**Equação:**
```
curl(τ) = ∂τy/∂x − ∂τx/∂y

Positivo no Pacífico equatorial → suprime upwelling → aprofunda termoclina → El Niño
Negativo → favorece upwelling de água fria → La Niña
```

**Código — novo `src/nino_brasil/features/tendency_features.py`:**
```python
def wind_stress_curl(
    tau_x: xr.DataArray,
    tau_y: xr.DataArray,
    lon_name: str = "lon",
    lat_name: str = "lat",
) -> xr.DataArray:
    """Rotacional de tensão de vento — forçante de Ekman (∂τy/∂x − ∂τx/∂y)."""
    R = 6.371e6
    cos_lat = np.cos(np.deg2rad(tau_x[lat_name]))
    dtau_y_dx = tau_y.differentiate(lon_name) / (np.deg2rad(1.0) * R * cos_lat)
    dtau_x_dy = tau_x.differentiate(lat_name) / (np.deg2rad(1.0) * R)
    curl = dtau_y_dx - dtau_x_dy
    curl.attrs.update({"units": "Pa m-1", "physics": "ekman_pumping"})
    return curl
```

**Custo:** requer ERA5 `tau_x` / `tau_y` — verificar se já está no download.

---

## Rank 2 — IMPORTANTES (ganho real em interpretabilidade física)

Modelos de ML com lags podem aproximar derivadas temporais a partir de dois instantes de tempo consecutivos. O ganho preditivo bruto é moderado. O ganho para o trabalho científico é alto: o SHAP passa a revelar "o modelo responde ao ritmo de afundamento da termoclina, não ao valor absoluto" — transformando um resultado estatístico em afirmação física publicável.

### 🟡 PC-1 · Tendência de D20 (`∂D20/∂t`)

```python
# adicionar em thermocline.py
def d20_tendency(d20, window_days=30, time_name="time"):
    """Velocidade de shoaling/deepening — LHS do oscilador de Jin (1997)."""
    dt = d20[time_name].diff(time_name) / np.timedelta64(1, "D")
    tendency = d20.diff(time_name) / dt
    tendency.attrs.update({"units": "m/day", "physics": "recharge_lhs"})
    return tendency
```

### 🟡 PC-2 · Tendência de OHC (`∂OHC/∂t`)

```python
# adicionar em ocean_heat.py
def ohc_tendency(ohc, time_name="time"):
    """Taxa de acúmulo de calor oceânico — equação de balanço de calor."""
    dt = ohc[time_name].diff(time_name) / np.timedelta64(1, "D")
    tendency = (ohc.diff(time_name) / dt) / 86400.0  # → W/m²
    tendency.attrs.update({"units": "W m-2", "physics": "heat_budget_lhs"})
    return tendency
```

### 🟡 PC-3 · Tendência de SSTA (`∂SSTA/∂t`)

```python
# em tendency_features.py
def ssta_tendency(ssta, window_days=14, time_name="time"):
    """Discrimina aquecimento crescente de aquecimento em decaimento."""
    dt = ssta[time_name].diff(time_name) / np.timedelta64(1, "D")
    tendency = ssta.diff(time_name) / dt
    tendency.attrs.update({"units": "K/day", "physics": "sst_transport"})
    return tendency
```

---

## Rank 3 — ÚTIL (com custo de implementação maior)

### 🟢 PC-8 · SSHA — nova fonte CMEMS/AVISO

**Quando importa:** só no Modelo A operacional. ORAS5 tem latência de 1–2 meses; SSHA tem 7 dias. Sem SSHA, não há sistema operacional. Para a pesquisa (dados históricos), é opcional.

```python
# src/nino_brasil/data/download_ssha.py
import copernicusmarine as cm
cm.subset(
    dataset_id="cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.25deg_P1D",
    variables=["sla"],
    ...
)
```

### 🟢 PC-7 · Stress zonal equatorial médio (`τx_eq`)

`u850` já captura grande parte desta informação. Baixa prioridade.

---

## Integração no pipeline

**Ablação nova** — responde diretamente à pergunta deste parecer:

```python
# ablation.py
PHYSICS_ABLATIONS = {
    "G_with_physics":    {"ocean", "atmosphere", "physics_precalc"},
    "H_without_physics": {"ocean", "atmosphere"},
}
```

`G vs H` mede quanto da capacidade preditiva vem de ter os termos dinâmicos explícitos. Se o ganho for pequeno, os modelos já estavam aprendendo a física dos dados brutos. Se for grande, os pré-cálculos eram necessários.

**Saídas numéricas obrigatórias** (seguindo PARECER_2):

```
outputs/diagnostics/physics_precalc_timeseries.csv
  colunas: date, wwv, tilt, curl_eq, d20_tend, ohc_tend, ssta_tend
```

---

## Resumo executivo

| PC | Feature | Módulo | Custo | Importância |
|----|---------|--------|-------|-------------|
| PC-5 | `warm_water_volume` | `ocean_heat.py` | Zero | 🔴 Crítico |
| PC-6 | `thermocline_tilt` | `thermocline.py` | Zero | 🔴 Crítico |
| PC-4 | `wind_stress_curl` | `tendency_features.py` (novo) | Baixo | 🔴 Crítico |
| PC-1 | `d20_tendency` | `thermocline.py` | Zero | 🟡 Importante |
| PC-2 | `ohc_tendency` | `ocean_heat.py` | Zero | 🟡 Importante |
| PC-3 | `ssta_tendency` | `tendency_features.py` | Zero | 🟡 Importante |
| PC-8 | `ssha` | `data/download_ssha.py` (novo) | Médio | 🟢 Útil (op.) |
| PC-7 | `equatorial_zonal_stress` | `tendency_features.py` | Baixo | 🟢 Útil |

**PC-5, PC-6, PC-1, PC-2, PC-3:** implementáveis esta semana com dados que já existem no projeto — ~200 linhas de código total.

**PC-4:** requer verificar se ERA5 `tau_x`/`tau_y` já está sendo baixado.

**PC-8:** adiar para a fase operacional.
