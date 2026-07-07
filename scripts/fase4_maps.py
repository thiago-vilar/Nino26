"""Contornos do Brasil (IBGE local) para os mapas da Fase 4 - estados + regioes.
Geometria simplificada e cacheada (plot rapido)."""
from pathlib import Path
import geopandas as gpd
_UFb=None; _REGb=None
def _load():
    global _UFb,_REGb
    if _UFb is None:
        root=Path(__file__).resolve().parent.parent
        uf=gpd.read_file(root/'data'/'interim'/'ibge'/'BR_UF_2024'/'BR_UF_2024.shp')
        reg=uf.dissolve(by='SIGLA_RG')
        _UFb=uf.boundary.simplify(0.05)      # ~5 km: leve e rapido
        _REGb=reg.boundary.simplify(0.05)
    return _UFb,_REGb
def add_brazil(ax, states=True, regions=True, box=((-75,-30),(-35,7))):
    """Sobrepoe divisas estaduais (finas) e regionais (grossas) em eixo lon/lat."""
    ufb,regb=_load()
    if states:  ufb.plot(ax=ax, linewidth=0.3, edgecolor='0.45')
    if regions: regb.plot(ax=ax, linewidth=0.9, edgecolor='black')
    (xl,xr),(yb,yt)=box; ax.set_xlim(xl,xr); ax.set_ylim(yb,yt)
    return ax
