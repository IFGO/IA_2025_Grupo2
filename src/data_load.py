# src/data_load.py

import pandas as pd
import os
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_crypto_file(filepath: str) -> pd.DataFrame:
    """
    Carrega um arquivo CSV de dados históricos de criptomoeda.
    Args:
        filepath (str): Caminho completo do arquivo CSV.
    Returns:
        pd.DataFrame: DataFrame com colunas padronizadas ['date', 'symbol', 'open', 'high', 'low', 'close', 'Volume USD'].
                      Retorna DataFrame vazio em caso de erro.
    """
    try:
        df = pd.read_csv(filepath, skiprows=1)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        symbol = os.path.basename(filepath).split('_')[1].replace('USD', '')
        df['symbol'] = symbol.upper()
        logging.info(f"Arquivo carregado com sucesso: {filepath}")
        return df[['date', 'symbol', 'open', 'high', 'low', 'close', 'Volume USD']]
    except Exception as e:
        logging.error(f"Erro ao carregar {filepath}: {e}")
        return pd.DataFrame()

def load_all_cryptos(filepaths: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Carrega múltiplos arquivos CSV de criptomoedas e retorna um dicionário com seus DataFrames.
    Args:
        filepaths (List[str]): Lista de caminhos para os arquivos CSV.
    Returns:
        Dict[str, pd.DataFrame]: Dicionário com o nome do arquivo como chave e o DataFrame como valor.
    """
    return {os.path.basename(f): load_crypto_file(f) for f in filepaths}