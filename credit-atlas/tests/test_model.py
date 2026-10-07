import numpy as np
import pytest
from creditrisk.model import calibrate,equity_from_assets,default_metrics
from creditrisk.portfolio import simulate,sector_correlation,tail_metrics,validate_correlation

@pytest.mark.parametrize('a,d,v,r,t',[(150,100,.3,.04,1),(110,100,.5,.01,1),(250,100,.2,.05,2),(8e10,6e10,.27,-.005,1)])
def test_round_trip(a,d,v,r,t):
    e,ev=equity_from_assets(a,d,v,r,t)
    c=calibrate(e,ev,d,r,t)
    assert c.asset_value==pytest.approx(a,rel=1e-7)
    assert c.asset_volatility==pytest.approx(v,rel=1e-7)
    assert c.relative_residual<1e-8

@pytest.mark.parametrize('e,ev,d',[(0,.2,10),(10,-1,10),(10,.2,float('nan'))])
def test_invalid_calibration(e,ev,d):
    with pytest.raises(ValueError):calibrate(e,ev,d)

def test_asset_shock_increases_pd():
    assert default_metrics(120,100,.3)[1]>default_metrics(150,100,.3)[1]

def test_es_handles_ties_and_fractional_boundary():
    # Worst 25% of [0,0,10,20,20,20] = 20; >=VaR would not generally yield exact ES.
    assert tail_metrics([0,0,10,20,20,20],.75)==pytest.approx((20,20))
    # Worst 40% = (20 + 0.6*10) / 1.6, not mean([10,20])
    assert tail_metrics([0,0,10,20],.6)==pytest.approx((10,16.25))

def test_cholesky_reconstruction():
    c=sector_correlation(['A','A','B'],.3,.2)
    l=validate_correlation(c,3)
    assert np.allclose(l@l.T,c)

def test_invalid_correlation_is_rejected():
    with pytest.raises(ValueError):validate_correlation([[1,2],[2,1]],2)

def test_simulation_matches_analytic_and_reproducible():
    p=[.02,.1,.3];e=[100,200,300];c=sector_correlation(['a','b','b'])
    r=simulate(p,e,c,simulations=100_000)
    assert abs(r['simulated_mean_loss']-r['expected_loss'])<5*r['mean_standard_error']
    assert np.array_equal(r['losses'],simulate(p,e,c,simulations=100_000)['losses'])
    for target,actual in zip(p,r['empirical_default_rates']):
        assert abs(target-actual)<5*np.sqrt(target*(1-target)/100_000)
    assert r['expected_shortfall']>=r['var']

def test_pd_endpoints():
    r=simulate([0,1],[100,200],np.eye(2),lgd=.6,simulations=100)
    assert np.all(r['losses']==120)
    assert r['var']==pytest.approx(120)
    assert r['expected_shortfall']==pytest.approx(120)
