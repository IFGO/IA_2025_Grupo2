import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pandas as pd
from data_load import load_crypto_file
from features import engineer_features
from models import split_features_labels, train_linear

def test_train_linear_executes():
    df = load_crypto_file('data/raw/all/Bitstamp_ETHUSD_d.csv')
    df_feat = engineer_features(df)
    X, y = split_features_labels(df_feat)
    results = train_linear(X, y, k=3)
    assert isinstance(results, dict)
    assert 'mse' in results and 'mae' in results and 'r2' in results
