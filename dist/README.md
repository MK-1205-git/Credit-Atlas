# Credit Atlas — Structural Credit Risk Model

An interactive credit-risk research project by Mushtahir Kabir. Python, NumPy, SciPy, and Plotly power a reproducible Merton model, a 20-issuer panel, and 100,000 correlated Monte Carlo scenarios.

**[Open the interactive demo](https://credit-atlas-mushtahir.mushtahirkabir.chatgpt.site)**

## What this project demonstrates

- Iterative Black–Scholes inversion using SciPy least squares to infer firm asset value and volatility from equity value, equity volatility, and a debt boundary.
- Risk-neutral terminal default probabilities and distance to default.
- Cholesky-factorized Gaussian dependence, seeded Monte Carlo, expected loss, VaR, and expected shortfall with correct treatment of discrete tail ties.
- Stress testing, issuer drilldowns, correlation heatmaps, and CSV/JSON exports.
- Local ingestion adapters for Yahoo Finance, SEC EDGAR company facts, and FRED, with stale-data checks, filing-date filters, retries, provenance, and explicit failures.

**Data honesty:** the hosted demo uses 20 fictional firms with reproducible synthetic inputs. It does not display live market data. Live ingestion requires your FRED key, a descriptive SEC contact User-Agent, and network access. The adapters have not been validated end-to-end against all 20 live issuers in this environment. Missing data never silently becomes synthetic data.

## Quick start

Requires Python 3.10+; Node 20+ is optional for browser-engine tests.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e '.[dev]'
python -m creditrisk.cli analyze --simulations 100000 --seed 42
python -m http.server 8000 --directory dist
```

Open http://localhost:8000. Open `reports/demo/report.html` for the standalone Python Plotly report. The analysis command also writes JSON and CSV.

```bash
python -m pytest -q
node --test tests/test_browser.cjs
```

## Live ingestion

Set credentials in your shell, not in committed code. `.env.example` documents the variables; the program reads environment variables directly and does not automatically load `.env` files.

```bash
export FRED_API_KEY='YOUR_FRED_KEY'
export SEC_USER_AGENT='Mushtahir Kabir your-contact-email@example.com'
python -m creditrisk.cli ingest --universe data/universe.json --output data/live
python -m creditrisk.cli analyze --panel data/live/panel.json --output reports/live --simulations 100000
```

The universe contains 20 real company tickers. Yahoo supplies prices; SEC supplies liabilities and shares; FRED supplies DGS1. The pipeline uses up to 252 daily returns to estimate annualized equity volatility, requiring at least 200. Correlations use overlapping returns (at least 126) and 10% identity shrinkage. Raw responses and provenance are saved locally and excluded from Git. SEC requests are paced; external services can reject or change schemas. Ingestion stops if any issuer fails, preserving an audit log.

For a dated panel add `--as-of YYYY-MM-DD`. This is not a point-in-time backtesting database: adjusted prices can incorporate later corporate actions, and total shares and liabilities have reporting lags.

## Model

For assets A, debt D, asset volatility sigma, continuous rate r, and horizon T:

```
d1 = [ln(A/D) + (r + sigma²/2)T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
E  = A Phi(d1) - D exp(-rT) Phi(d2)
sigma_E = (A/E) Phi(d1) sigma
PD = Phi(-d2)
```

Calibration solves the two equity equations jointly in log parameters with dimensionless residuals. Simulation uses correlated normal draws and issuer-specific default thresholds. Loss = exposure × LGD × terminal-default indicator. VaR is the empirical quantile; expected shortfall averages exactly the worst (1-confidence) fraction of observations, including fractional boundary mass.

**Assumptions:** risk-neutral rather than real-world default probabilities; terminal default only; fixed debt boundary (total liabilities in the live adapter); constant volatility and rates; fixed LGD; no credit migration or mark-to-market valuation; Gaussian dependence with no additional tail-dependence mechanism. Losses are undiscounted. A low calibration residual proves numerical fit, not forecasting accuracy. Educational software, not an investment recommendation.

The browser runs a JavaScript Web Worker stress engine from the Python-calibrated baseline. It uses a different seeded random generator from NumPy, so Monte Carlo values differ within sampling error. The initial dashboard is the Python result. No Python backend or credentials are exposed in the public app.

## Baseline and verification

20 fictional issuers, $10 million total exposure, 60% LGD, 1 year, 99% confidence, seed 42, 100,000 paths:

| Metric | Python baseline |
|---|---:|
| Analytical expected loss | $476,949.04 |
| Monte Carlo mean loss | $472,707.00 |
| Mean standard error | $1,662.47 |
| 99% VaR | $2,100,000.00 |
| 99% expected shortfall | $2,570,700.00 |

17 Python tests and 3 Node tests pass. Coverage includes calibration roundtrips, invalid inputs, stress direction, Cholesky validity, discrete expected-shortfall ties, seeded reproducibility, Monte Carlo sampling bounds, SEC date filtering, pipeline output, and browser/Python analytical agreement. Automated browser UI testing was unavailable in the build environment; JavaScript syntax and numerical tests were run.

## Repository map

- `creditrisk/model.py`: calibration and default probabilities
- `creditrisk/portfolio.py`: correlation, simulation, tail metrics
- `creditrisk/ingestion.py`: three-source data pipeline
- `creditrisk/pipeline.py`: analysis and Plotly reporting
- `creditrisk/cli.py`: command-line interface
- `data/`: fictional demo panel and real ingestion universe
- `dist/`: deployable static frontend, worker, baseline, standalone report
- `tests/`: Python and Node numerical/data checks
- `scripts/`: reproducible demo and release generation

## Publish on your GitHub

Unzip the source archive and upload the contents to a new repository called `credit-atlas`. Set its About website to the interactive demo above. The README already links to the demo. Include `.github/workflows/tests.yml` to run checks on pushes.

Alternatively, from the extracted folder:

```bash
git init
git add .
git commit -m "Build structural credit risk model and interactive demo"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/credit-atlas.git
git push -u origin main
```

For independent hosting, deploy `dist/` to any static host. No build command is needed. Source changes to the Python model require regenerating `dist/baseline.json`; browser stress logic lives separately in `dist/risk-engine.js`. Never publish real source credentials or raw datasets without checking redistribution terms.

## References

- [Merton model equations](https://www.mathworks.com/help/risk/mertonmodel.html)
- [SEC EDGAR APIs](https://www.sec.gov/search-filings/edgar-application-programming-interfaces)
- [FRED series observations](https://fred.stlouisfed.org/docs/api/fred/series_observations.html)
- [yfinance documentation](https://ranaroussi.github.io/yfinance/)

Code is MIT licensed. Plotly.js retains its MIT license notice in the vendored bundle. External data and services retain their respective terms.
