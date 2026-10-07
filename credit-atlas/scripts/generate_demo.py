"""Generate fictitious issuers from known asset parameters. No real firm data."""
import json
from pathlib import Path
from creditrisk.model import equity_from_assets
from creditrisk.portfolio import sector_correlation
names=[("ASTR","Aster Systems","Technology"),("NOVA","Nova Networks","Technology"),("LUMA","Luma Software","Technology"),("VRTX","Vertex Devices","Technology"),("ORBT","Orbit Cloud","Technology"),
("ALPN","Alpine Manufacturing","Industrials"),("BRDG","Bridgeway Logistics","Industrials"),("CRST","Crest Machinery","Industrials"),("FRGE","Forge Materials","Industrials"),("HBR","Harbor Transport","Industrials"),
("CEDR","Cedar Retail","Consumer"),("MAPL","Maple Brands","Consumer"),("OAK","Oak Consumer","Consumer"),("PINE","Pine Markets","Consumer"),("REED","Reed Leisure","Consumer"),
("SOLR","Solaris Energy","Energy"),("TIDE","Tide Resources","Energy"),("WIND","Windward Power","Energy"),("GLEN","Glen Utilities","Energy"),("PEAK","Peak Fuels","Energy")]
ratios=[2.3,1.8,2.0,1.45,1.65,1.9,1.35,1.55,1.3,1.5,1.7,1.4,2.1,1.32,1.25,1.65,1.42,1.8,2.2,1.28]
vols=[.22,.28,.24,.32,.3,.2,.27,.26,.3,.24,.23,.29,.2,.28,.33,.28,.31,.25,.18,.35]
firms=[]
for i,(ticker,name,sector) in enumerate(names):
    d=(4+i*1.4)*1e9
    e,v=equity_from_assets(d*ratios[i],d,vols[i],.042,1)
    firms.append(dict(ticker=ticker,name=name,sector=sector,equity=e,equity_volatility=v,debt=d,exposure=500_000,
                      source={"type":"synthetic","construction":"Forward Merton equations from chosen asset parameters; not real market observations"}))
panel=dict(schema_version=1,mode="synthetic",as_of=None,description="20 fictitious issuers. Reproducible educational fixture; no provider data or real-company estimates.",
 rate=.042,rate_source={"series":"illustrative","value":.042},firms=firms,
 correlation=sector_correlation([f["sector"] for f in firms]).tolist(),
 correlation_method="Assumed market factor 0.25 + sector factor 0.15 + idiosyncratic variance",
 debt_definition="Synthetic single zero-coupon debt boundary at the selected horizon")
Path("data/demo_panel.json").write_text(json.dumps(panel,indent=2))
