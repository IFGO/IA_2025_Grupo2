import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from data_load import load_crypto_file

def test_load_crypto_file():
    df = load_crypto_file('data/Bitstamp_BTCUSD_d.csv')
    assert not df.empty
    assert all(col in df.columns for col in ['date', 'symbol', 'open', 'high', 'low', 'close', 'Volume USD'])
