"""Single-point Merton calibration. All monetary inputs share a common unit.

Risk-neutral calibration uses equity as a European call on firm assets.
No cash dividends, constant volatility/rate, one terminal default boundary.
"""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr

@dataclass(frozen=True)
class Calibration:
    asset_value: float
    asset_volatility: float
    distance_to_default: float
    probability_of_default: float
    relative_residual: float
    evaluations: int

    def to_dict(self):
        return asdict(self)

def positive(**values):
    for name, value in values.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

def equity_from_assets(asset_value, debt, asset_volatility, rate=0.042, maturity=1.0):
    positive(asset_value=asset_value, debt=debt, asset_volatility=asset_volatility, maturity=maturity)
    if not np.isfinite(rate):
        raise ValueError("rate must be finite")
    root = asset_volatility * np.sqrt(maturity)
    d1 = (np.log(asset_value / debt) + (rate + .5 * asset_volatility**2) * maturity) / root
    d2 = d1 - root
    equity = asset_value * ndtr(d1) - debt * np.exp(-rate * maturity) * ndtr(d2)
    equity_vol = asset_value * ndtr(d1) * asset_volatility / equity
    return float(equity), float(equity_vol)

def default_metrics(asset_value, debt, asset_volatility, rate=0.042, horizon=1.0):
    """Q-measure PD. This is not an empirically estimated real-world PD."""
    positive(asset_value=asset_value, debt=debt, asset_volatility=asset_volatility, horizon=horizon)
    if not np.isfinite(rate):
        raise ValueError("rate must be finite")
    dd = (np.log(asset_value / debt) + (rate - .5 * asset_volatility**2) * horizon) / (asset_volatility * np.sqrt(horizon))
    return float(dd), float(ndtr(-dd))

def calibrate(equity, equity_volatility, debt, rate=0.042, maturity=1.0):
    """Solve 2 nonlinear equations for A, sigma_A with positive log variables.

    Asset value is scaled by E+D for numerical conditioning; residuals are
    dimensionless relative errors in equity and equity volatility. Failed or
    inaccurate optimization is rejected, never silently published.
    """
    positive(equity=equity, equity_volatility=equity_volatility, debt=debt, maturity=maturity)
    if not np.isfinite(rate):
        raise ValueError("rate must be finite")
    scale = equity + debt
    initial_asset = equity + debt * np.exp(-rate * maturity)
    initial_vol = equity_volatility * equity / initial_asset
    def residual(x):
        a, sigma = np.exp(x) * np.array([scale, 1.0])
        e, v = equity_from_assets(a, debt, sigma, rate, maturity)
        if e <= 0 or not np.isfinite(v):
            return np.array([1e6, 1e6])
        return np.array([(e - equity) / equity, (v - equity_volatility) / equity_volatility])
    fit = least_squares(residual, np.log([initial_asset / scale, initial_vol]),
                        bounds=([-20, -12], [10, 3]), xtol=1e-12, ftol=1e-12,
                        gtol=1e-12, max_nfev=2000)
    error = float(np.max(np.abs(residual(fit.x))))
    if not fit.success or error > 1e-6:
        raise RuntimeError(f"Calibration did not converge: relative residual={error:.3g}")
    a, sigma = np.exp(fit.x) * np.array([scale, 1.0])
    dd, pd = default_metrics(a, debt, sigma, rate, maturity)
    return Calibration(float(a), float(sigma), dd, pd, error, fit.nfev)
