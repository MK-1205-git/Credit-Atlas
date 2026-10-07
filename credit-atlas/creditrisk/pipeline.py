from pathlib import Path
import json
import numpy as np
from .model import calibrate, default_metrics
from .portfolio import simulate

def analyze(panel, *, simulations=100_000, seed=42, horizon=1.0, confidence=.99,
            asset_shock=0.0, volatility_multiplier=1.0, lgd=.6):
    if not -0.9 <= asset_shock <= .5 or not .1 <= volatility_multiplier <= 5:
        raise ValueError("Stress inputs outside supported range")
    firms=[]
    for firm in panel["firms"]:
        fit=calibrate(firm["equity"],firm["equity_volatility"],firm["debt"],panel["rate"],1.0)
        a=fit.asset_value*(1+asset_shock); v=fit.asset_volatility*volatility_multiplier
        dd,pd=default_metrics(a,firm["debt"],v,panel["rate"],horizon)
        firms.append({**firm,**fit.to_dict(),"stressed_asset_value":a,"stressed_asset_volatility":v,
                      "distance_to_default":dd,"probability_of_default":pd})
    risk=simulate([f["probability_of_default"] for f in firms],[f["exposure"] for f in firms],
                  panel["correlation"],lgd,simulations,confidence,seed)
    losses=risk.pop("losses")
    total=sum(f["exposure"] for f in firms)
    count,edges=np.histogram(losses,bins=np.linspace(0,total*lgd,41))
    for f,observed in zip(firms,risk["empirical_default_rates"]):
        f["simulated_default_rate"]=observed
        f["expected_loss"]=f["exposure"]*lgd*f["probability_of_default"]
    return {"schema_version":1,"panel":panel,"firms":firms,"risk":risk,
            "engine":"Python NumPy/SciPy; numpy.default_rng(seed)",
            "parameters":{"simulations":simulations,"seed":seed,"horizon":horizon,"confidence":confidence,
                          "asset_shock":asset_shock,"volatility_multiplier":volatility_multiplier,"lgd":lgd},
            "total_exposure":total,"histogram":{"edges":edges.tolist(),"counts":count.tolist()},
            "method":"Risk-neutral terminal default; fixed LGD; Gaussian latent dependence; undiscounted default losses"}

def write_report(result, output):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    (out/"results.json").write_text(json.dumps(result,indent=2,allow_nan=False))
    import pandas as pd
    pd.DataFrame([{k:v for k,v in f.items() if k!="source"} for f in result["firms"]]).to_csv(out/"firm_metrics.csv",index=False)
    fig=make_subplots(rows=1,cols=2,subplot_titles=("Portfolio loss distribution","Model-implied default probabilities"))
    h=result["histogram"]
    fig.add_trace(go.Bar(x=[(a+b)/2 for a,b in zip(h["edges"],h["edges"][1:])],y=h["counts"],name="Simulation count"),row=1,col=1)
    fig.add_trace(go.Bar(x=[f["ticker"] for f in result["firms"]],y=[100*f["probability_of_default"] for f in result["firms"]],name="Risk-neutral PD (%)"),row=1,col=2)
    fig.update_layout(title=f"Credit Atlas · {result['panel']['mode'].upper()} inputs · {result['parameters']['simulations']:,} paths",template="plotly_white",showlegend=False)
    fig.update_xaxes(title_text="Undiscounted default loss ($)",row=1,col=1)
    fig.update_yaxes(title_text="Paths",row=1,col=1)
    fig.update_yaxes(title_text="Risk-neutral PD (%)",row=1,col=2)
    fig.write_html(out/"report.html",include_plotlyjs=True)
