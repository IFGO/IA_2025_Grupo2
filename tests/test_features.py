import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from features import engineer_features

def test_engineer_features():
    df = pd.DataFrame({
        'date': pd.date_range(start='2023-01-01', periods=10),
        'close': [10,11,12,13,14,15,16,17,18,19]
    })
    df_feat = engineer_features(df)
    assert 'lag_1' in df_feat.columns
    assert 'rolling_mean_7' in df_feat.columns
    assert not df_feat.isnull().values.any()
