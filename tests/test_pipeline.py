import json
from pathlib import Path
from creditrisk.pipeline import analyze

def test_twenty_firm_pipeline():
    panel=json.loads(Path('data/demo_panel.json').read_text())
    result=analyze(panel,simulations=1000)
    assert len(result['firms'])==20
    assert result['total_exposure']==10_000_000
    assert sum(result['histogram']['counts'])==1000
    assert max(x['relative_residual'] for x in result['firms'])<1e-8
    json.dumps(result,allow_nan=False)
