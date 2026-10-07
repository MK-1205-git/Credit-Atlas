"""Explicit live adapters. Credentials stay in the environment, never exports.

No synthetic fallback on live errors. Store source payloads for auditability.
Yahoo access uses the unofficial yfinance package; observe data-provider terms.
"""
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import json, os, time
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .portfolio import regularize_correlation

class SourceError(RuntimeError):
    pass

def session():
    s = requests.Session()
    retry = Retry(total=3, backoff_factor=1, status_forcelist=[429,500,502,503,504])
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s

def get_json(client, url, **kwargs):
    try:
        r = client.get(url, timeout=30, **kwargs)
        r.raise_for_status()
        return r.json()
    except (requests.RequestException, ValueError) as exc:
        # requests errors can contain URLs with an API key. Do not echo them.
        raise SourceError(f"Provider request failed ({type(exc).__name__}); check credentials and connectivity") from None

def select_fact(payload, namespace, tag, unit, as_of):
    """Most recent instant fact available by as_of; reject future filings."""
    facts = payload.get("facts", {}).get(namespace, {}).get(tag, {}).get("units", {}).get(unit, [])
    usable = [x for x in facts if x.get("end", "9999") <= as_of
              and x.get("filed", "9999") <= as_of and x.get("form") in ("10-K", "10-Q", "10-K/A", "10-Q/A")
              and "start" not in x and isinstance(x.get("val"), (int,float)) and x["val"] > 0]
    if not usable:
        raise SourceError(f"No positive {namespace}:{tag} ({unit}) available by {as_of}")
    chosen = max(usable, key=lambda x: (x["end"], x["filed"]))
    if (date.fromisoformat(as_of) - date.fromisoformat(chosen["end"])).days > 550:
        raise SourceError(f"Stale {tag}: observation older than 550 days")
    return chosen

def fred_rate(client, api_key, as_of):
    if not api_key:
        raise SourceError("Set FRED_API_KEY to retrieve DGS1")
    payload = get_json(client, "https://api.stlouisfed.org/fred/series/observations", params={
        "series_id":"DGS1", "api_key":api_key, "file_type":"json",
        "observation_end":as_of, "realtime_start":as_of, "realtime_end":as_of,
        "sort_order":"desc", "limit":30})
    valid = [x for x in payload.get("observations", []) if x.get("value") not in (None,".") and x["date"] <= as_of]
    if not valid:
        raise SourceError("No FRED DGS1 observation available")
    x = max(valid, key=lambda x:x["date"])
    if (date.fromisoformat(as_of)-date.fromisoformat(x["date"])).days > 14:
        raise SourceError("FRED observation is stale")
    # DGS1 is a quoted Treasury yield, treated as an annually compounded proxy.
    rate = float(np.log1p(float(x["value"])/100))
    return rate, {"series":"DGS1", "date":x["date"], "yield_percent":float(x["value"]),
                  "conversion":"ln(1 + DGS1/100), approximate continuous-rate proxy"}, payload

def ingest(universe_path, output, as_of=None):
    import yfinance as yf
    as_of = as_of or date.today().isoformat()
    if date.fromisoformat(as_of) > date.today():
        raise ValueError("as_of cannot be in the future")
    ua = os.environ.get("SEC_USER_AGENT", "")
    if "@" not in ua:
        raise SourceError("Set SEC_USER_AGENT to your name and contact email")
    key = os.environ.get("FRED_API_KEY", "")
    out = Path(output); raw = out / "raw"; raw.mkdir(parents=True, exist_ok=True)
    client = session()
    rate, rate_meta, fred_payload = fred_rate(client, key, as_of)
    # FRED response includes no request credentials; persist observations only.
    (raw / "fred.json").write_text(json.dumps({"observations":fred_payload.get("observations", [])}))
    mapping = get_json(client,"https://www.sec.gov/files/company_tickers.json",headers={"User-Agent":ua})
    ciks = {v["ticker"].upper(): v["cik_str"] for v in mapping.values()}
    universe = json.loads(Path(universe_path).read_text())
    start = (date.fromisoformat(as_of)-timedelta(days=550)).isoformat()
    end = (date.fromisoformat(as_of)+timedelta(days=1)).isoformat()
    firms, returns, errors = [], {}, []
    for spec in universe:
        symbol = spec["ticker"]
        try:
            if symbol not in ciks:
                raise SourceError("Ticker not found in SEC mapping")
            time.sleep(.25)  # four requests/sec max, below SEC's 10/sec guideline
            cik = int(ciks[symbol])
            payload = get_json(client, f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers={"User-Agent":ua})
            (raw/f"{symbol}-sec.json").write_text(json.dumps(payload))
            liability = select_fact(payload,"us-gaap","Liabilities","USD",as_of)
            shares = select_fact(payload,"dei","EntityCommonStockSharesOutstanding","shares",as_of)
            prices = yf.Ticker(symbol).history(start=start,end=end,auto_adjust=False,actions=False)
            if prices.empty or "Adj Close" not in prices:
                raise SourceError("Yahoo returned no adjusted close history")
            prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
            prices = prices.loc[~prices.index.duplicated(keep="last")].sort_index()
            prices = prices.loc[prices.index <= pd.Timestamp(as_of)]
            prices.to_csv(raw/f"{symbol}-prices.csv")
            if len(prices)<253 or not np.isfinite(prices["Close"].iloc[-1]):
                raise SourceError("Fewer than 253 daily prices or invalid latest close")
            if (pd.Timestamp(as_of)-prices.index[-1]).days>7:
                raise SourceError("Yahoo latest close is stale")
            adj = prices["Adj Close"].where(prices["Adj Close"]>0)
            log_returns = np.log(adj/adj.shift()).dropna().tail(252)
            if len(log_returns)<200:
                raise SourceError("Insufficient valid returns")
            vol = float(log_returns.std(ddof=1)*np.sqrt(252))
            equity = float(prices["Close"].iloc[-1]*shares["val"])
            firms.append({**spec,"equity":equity,"equity_volatility":vol,"debt":float(liability["val"]),
                          "exposure":500_000,"source":{
                              "yahoo_price_date":str(prices.index[-1].date()),"volatility_observations":len(log_returns),
                              "sec_cik":cik,"liability_tag":"Liabilities","liability_end":liability["end"],
                              "liability_filed":liability["filed"],"liability_accession":liability.get("accn"),
                              "shares_end":shares["end"],"shares_filed":shares["filed"],
                              "warning":"Market cap uses latest disclosed common shares; may differ from current shares. Total liabilities are a simplified default-boundary proxy."}})
            returns[symbol]=log_returns
        except Exception as exc:
            # Deliberately do not publish a partial 20-firm panel or made-up data.
            errors.append({"ticker":symbol,"error":str(exc) if isinstance(exc,SourceError) else type(exc).__name__})
    audit={"as_of":as_of,"requested":len(universe),"retrieved":len(firms),"errors":errors}
    (out/"ingestion-audit.json").write_text(json.dumps(audit,indent=2))
    if errors:
        raise SourceError(f"Live ingestion incomplete ({len(firms)}/{len(universe)}); inspect ingestion-audit.json. No live panel written.")
    aligned=pd.DataFrame(returns).dropna()
    if len(aligned)<126:
        raise SourceError("Less than 126 common daily returns for correlation")
    corr=regularize_correlation(aligned.corr().to_numpy(),.1)
    panel={"schema_version":1,"mode":"live","as_of":as_of,"retrieved_at":datetime.now(timezone.utc).isoformat(),
           "description":"Live provider inputs; educational Merton approximation, not credit ratings",
           "rate":rate,"rate_source":rate_meta,"firms":firms,"correlation":corr.tolist(),
           "correlation_method":f"Aligned equity-return proxy ({len(aligned)} days), 10% shrinkage to identity",
           "debt_definition":"SEC total liabilities as a simplified terminal default boundary; not contractual debt maturity"}
    (out/"panel.json").write_text(json.dumps(panel,indent=2,allow_nan=False))
    return panel
