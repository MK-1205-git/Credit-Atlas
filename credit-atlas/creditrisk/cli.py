import argparse, json
from pathlib import Path
from .pipeline import analyze,write_report

def main():
    parser=argparse.ArgumentParser(description="Credit Atlas — reproducible structural credit risk")
    sub=parser.add_subparsers(dest="command",required=True)
    a=sub.add_parser("analyze")
    a.add_argument("--panel",default="data/demo_panel.json")
    a.add_argument("--output",default="reports/demo")
    a.add_argument("--simulations",type=int,default=100_000)
    a.add_argument("--seed",type=int,default=42)
    a.add_argument("--horizon",type=float,default=1)
    a.add_argument("--confidence",type=float,default=.99)
    a.add_argument("--asset-shock",type=float,default=0)
    a.add_argument("--volatility-multiplier",type=float,default=1)
    a.add_argument("--lgd",type=float,default=.6)
    i=sub.add_parser("ingest")
    i.add_argument("--universe",default="data/universe.json")
    i.add_argument("--output",default="data/live")
    i.add_argument("--as-of",default=None)
    args=parser.parse_args()
    if args.command=="ingest":
        from .ingestion import ingest
        panel=ingest(args.universe,args.output,args.as_of)
        print(f"Wrote {len(panel['firms'])} live firms to {args.output}/panel.json")
    else:
        kw=vars(args).copy(); kw.pop("command"); panel=json.loads(Path(kw.pop("panel")).read_text()); out=kw.pop("output")
        result=analyze(panel,**kw);write_report(result,out)
        print(json.dumps({k:v for k,v in result["risk"].items() if k!="empirical_default_rates"},indent=2))
        print(f"Wrote results.json, firm_metrics.csv, and Plotly report.html to {out}")

if __name__=="__main__":main()
