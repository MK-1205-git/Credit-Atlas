"""Gaussian-copula default-only portfolio loss simulation."""
import numpy as np
from scipy.special import ndtri

def validate_correlation(correlation, n):
    c = np.asarray(correlation, dtype=float)
    if c.shape != (n, n) or not np.isfinite(c).all():
        raise ValueError("Correlation shape/values invalid")
    if not np.allclose(c, c.T, atol=1e-10) or not np.allclose(np.diag(c), 1):
        raise ValueError("Correlation must be symmetric with diagonal 1")
    try:
        return np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Correlation must be positive definite") from exc

def sector_correlation(sectors, market=.25, sector=.15):
    if not (0 <= market < 1 and 0 <= sector and market + sector < 1):
        raise ValueError("Require market >= 0, sector >= 0, market + sector < 1")
    n = len(sectors)
    same = np.equal.outer(sectors, sectors).astype(float)
    return market * np.ones((n, n)) + sector * same + (1 - market - sector) * np.eye(n)

def regularize_correlation(correlation, shrinkage=.1):
    """Shrink sample equity-return correlation toward I, used as a proxy.

    No claim that equity correlation equals latent asset correlation.
    """
    c = np.asarray(correlation, dtype=float)
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be in (0,1]")
    c = (1-shrinkage) * c + shrinkage * np.eye(len(c))
    validate_correlation(c, len(c))
    return c

def tail_metrics(losses, confidence=.99):
    """Empirical left quantile VaR; ES is exact worst (1-alpha) mass.

    Fractional inclusion of the boundary observation handles discrete losses;
    taking mean(loss >= VaR) would over-include ties and understate ES.
    """
    x = np.asarray(losses, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError("losses must be finite, nonnegative, nonempty 1D")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0,1)")
    x = np.sort(x)
    var = x[max(0, int(np.ceil(confidence * len(x))) - 1)]
    mass = (1-confidence) * len(x)
    full = int(np.floor(mass + 1e-10))
    fraction = max(0.0, mass - full)
    total = x[-full:].sum() if full else 0.0
    if fraction > 1e-10:
        total += fraction * x[-full-1]
    return float(var), float(total / mass)

def simulate(probabilities, exposures, correlation, lgd=.6, simulations=100_000,
             confidence=.99, seed=42, batch_size=10_000):
    p, e = np.asarray(probabilities, float), np.asarray(exposures, float)
    if p.ndim != 1 or p.shape != e.shape or not len(p):
        raise ValueError("probabilities and exposures must be matching nonempty vectors")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
        raise ValueError("PD must be in [0,1]")
    if not np.isfinite(e).all() or (e <= 0).any():
        raise ValueError("exposures must be finite and positive")
    if not 0 <= lgd <= 1 or not isinstance(simulations, (int, np.integer)) or simulations < 100:
        raise ValueError("LGD must be in [0,1]; simulations must be an integer >=100")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    chol = validate_correlation(correlation, len(p))
    rng, thresholds = np.random.default_rng(seed), ndtri(p)
    loss_chunks, defaults = [], np.zeros(len(p), dtype=int)
    for offset in range(0, simulations, batch_size):
        z = rng.standard_normal((min(batch_size, simulations-offset), len(p))) @ chol.T
        default = z < thresholds
        defaults += default.sum(axis=0)
        loss_chunks.append(default @ (e * lgd))
    losses = np.concatenate(loss_chunks)
    var, es = tail_metrics(losses, confidence)
    return {
        "expected_loss": float(np.dot(p, e) * lgd),
        "simulated_mean_loss": float(losses.mean()),
        "mean_standard_error": float(losses.std(ddof=1)/np.sqrt(simulations)),
        "var": var, "expected_shortfall": es,
        "zero_loss_probability": float(np.mean(losses == 0)),
        "empirical_default_rates": (defaults/simulations).tolist(),
        "losses": losses,
    }
