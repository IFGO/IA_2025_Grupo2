# src/models.py

import numpy as np
import pandas as pd
import logging
from typing import Tuple, Dict, Union
from typing import cast
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def split_features_labels(df: pd.DataFrame, target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
    """
    Separa as colunas de features (X) e o alvo (y) para prever o fechamento do próximo dia.
    Args:
        df (pd.DataFrame): DataFrame com todas as features.
        target_col (str): Nome da coluna alvo.
    Returns:
        Tuple[np.ndarray, np.ndarray]: Arrays com X (features) e y (target).
    """
    try:
        selected_features = [
            'high', 'low', 'open',
            'lag_1', 'lag_2', 'lag_3',
            'rolling_mean_7', 'rolling_std_7',
            'Volume USD', 'year'
        ]
        selected_features = [col for col in selected_features if col in df.columns]

        X = df[selected_features]
        y = df[target_col].shift(-1).dropna()
        X = X.iloc[:-1, :]
        return X.to_numpy(), y.to_numpy()

    except Exception as e:
        logging.error(f"Erro ao separar features e target: {e}")
        return np.array([]), np.array([])

def get_model_pipeline(model_name: str, grau: int = 2) -> Pipeline:
    """
    Cria e retorna um pipeline de modelo com escalonamento e regressão.
    Args:
        model_name (str): Nome do modelo ('mlp', 'linear', 'poly').
        grau (int): Grau do polinômio, se aplicável.
    Returns:
        Pipeline: Pipeline treinável com scaler e modelo.
    """
    try:
        if model_name == 'mlp':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('mlp', MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=1000, random_state=42))
            ])
        elif model_name == 'linear':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('regressor', LinearRegression())
            ])
        elif model_name == 'poly':
            return Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(grau)),
                #('regressor', LinearRegression()).
                ('regressor', Ridge(alpha=1.0))
            ])
        else:
            raise ValueError(f"Modelo não reconhecido: {model_name}")
    except Exception as e:
        logging.error(f"Erro ao criar pipeline para modelo '{model_name}': {e}")
        raise

def kfold_cross_validation(model: Pipeline, X: np.ndarray, y: np.ndarray, k: int = 5) -> Dict[str, float]:
    """
    Executa validação cruzada com TimeSeriesSplit e retorna métricas médias.
    Args:
        model (Pipeline): Pipeline com scaler e modelo.
        X (np.ndarray): Features.
        y (np.ndarray): Alvo.
        k (int): Número de divisões.
    Returns:
        Dict[str, float]: Médias de MSE, MAE e R².
    """
    metrics = {'mse': [], 'mae': [], 'r2': []}
    try:
        tscv = TimeSeriesSplit(n_splits=k)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            metrics['mse'].append(mean_squared_error(y_test, preds))
            metrics['mae'].append(mean_absolute_error(y_test, preds))
            metrics['r2'].append(r2_score(y_test, preds))

        return {m: float(np.mean(scores)) for m, scores in metrics.items()}

    except Exception as e:
        logging.error(f"Erro na validação cruzada: {e}")
        return {m: float('nan') for m in metrics}

def train_model(model_name: str, X: np.ndarray, y: np.ndarray, grau: int = 2, k: int = 5) -> Dict[str, float]:
    """
    Treina o modelo especificado com validação cruzada.
    Args:
        model_name (str): Nome do modelo ('mlp', 'linear', 'poly').
        X (np.ndarray): Features.
        y (np.ndarray): Alvo.
        grau (int): Grau do polinômio (se aplicável).
        k (int): Número de folds.
    Returns:
        Dict[str, float]: Métricas médias.
    """
    try:
        model = get_model_pipeline(model_name, grau)
        return kfold_cross_validation(model, X, y, k)
    except Exception as e:
        logging.error(f"Erro ao treinar modelo {model_name}: {e}")
        return {'mse': np.nan, 'mae': np.nan, 'r2': np.nan}

def fit_and_predict_model(model_name: str, X: np.ndarray, y: np.ndarray, grau: int = 2) -> Tuple[np.ndarray, str]:
    """
    Treina o modelo e retorna as previsões e o nome.
    Args:
        model_name (str): Nome do modelo.
        X (np.ndarray): Features.
        y (np.ndarray): Alvo.
        grau (int): Grau polinomial.
    Returns:
        Tuple[np.ndarray, str]: Previsões e nome do modelo.
    """
    try:
        model = get_model_pipeline(model_name, grau)
        model.fit(X, y)
        preds = cast(np.ndarray, model.predict(X))
        return preds, model_name
    except Exception as e:
        logging.error(f"Erro ao ajustar e prever com modelo {model_name}: {e}")
        return np.array([]), model_name