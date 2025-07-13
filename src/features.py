# src/features.py

import pandas as pd
import logging
from typing import List

logging.basicConfig(level=logging.INFO)

def add_return(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona o retorno percentual diário baseado no fechamento."""
    try:
        df['return'] = df['close'].pct_change()
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular retorno percentual: {e}")
        return df

def add_rolling_features(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """Adiciona média e desvio padrão da série de fechamento com janela deslizante."""
    try:
        df[f'rolling_mean_{window}'] = df['close'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['close'].rolling(window).std()
        return df
    except Exception as e:
        logging.error(f"Erro ao calcular features com janela deslizante: {e}")
        return df

def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
    """Adiciona colunas de defasagem (lag) do preço de fechamento."""
    try:
        for lag in lags:
            df[f'lag_{lag}'] = df['close'].shift(lag)
        return df
    except Exception as e:
        logging.error(f"Erro ao adicionar lags: {e}")
        return df

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai variáveis temporais da coluna 'date'."""
    try:
        df['date'] = pd.to_datetime(df['date'])
        df['dayofweek'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
        df['year'] = df['date'].dt.year
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        return df
    except Exception as e:
        logging.error(f"Erro ao extrair features de data: {e}")
        return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica todas as transformações de engenharia de atributos (features).
    - Features temporais (data)
    - Retorno percentual
    - Médias e desvios móveis
    - Lags do fechamento
    """
    try:
        df = df.copy()
        df = add_date_features(df)
        df = add_return(df)
        df = add_rolling_features(df, window=7)
        df = add_lag_features(df, lags=[1, 2, 3])
        df.dropna(inplace=True)
        return df
    except Exception as e:
        logging.error(f"Erro na engenharia de features: {e}")
        return pd.DataFrame()
