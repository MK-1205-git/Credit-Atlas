import pytest
from creditrisk.ingestion import select_fact,SourceError,fred_rate

def payload(rows):return {'facts':{'us-gaap':{'Liabilities':{'units':{'USD':rows}}}}}

def test_sec_no_future_filing_and_latest_fact():
    base={'form':'10-Q','val':100,'end':'2025-06-30','filed':'2025-08-01'}
    data=payload([base,{**base,'val':500,'filed':'2026-01-01'},{**base,'val':75,'end':'2025-03-31'}])
    assert select_fact(data,'us-gaap','Liabilities','USD','2025-09-01')['val']==100

def test_sec_reject_missing_and_stale():
    with pytest.raises(SourceError):select_fact({},'us-gaap','Liabilities','USD','2025-09-01')
    with pytest.raises(SourceError):select_fact(payload([{'form':'10-K','val':100,'end':'2020-01-01','filed':'2020-02-01'}]),'us-gaap','Liabilities','USD','2025-09-01')

def test_fred_requires_key():
    with pytest.raises(SourceError):fred_rate(None,'','2025-09-01')
