# src/evaluation.py
from scipy import stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
from typing import Union
from models import split_features_labels, fit_and_predict_model
from utils import plot_dispersao, calcular_coeficientes_correlacao, obter_equacao_regressao, calcular_erro_padrao, erro_padrao_entre_modelos
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def simulate_investment(df: pd.DataFrame, preds: Union[np.ndarray, list], initial_capital: float = 1000.0) -> pd.DataFrame:
    """
    Simula uma estratégia simples de investimento: compra se previsão do próximo dia for maior que o preço atual.
    Args:
        df (pd.DataFrame): DataFrame com colunas 'close' e 'date'.
        preds (Union[np.ndarray, list]): Previsões geradas pelo modelo.
        initial_capital (float): Quantia inicial em dólares.
    Returns:
        pd.DataFrame: DataFrame com a evolução do saldo ao longo do tempo.
    """
    preds = np.asarray(preds)
    saldo = initial_capital
    saldo_evolucao = [saldo]

    for i in range(1, len(preds)):
        preco_atual = df['close'].iloc[i]
        preco_amanha_previsto = preds[i]

        if preco_amanha_previsto > preco_atual:
            quantidade = saldo / preco_atual
            saldo = quantidade * df['close'].iloc[i + 1] if i + 1 < len(df) else saldo

        saldo_evolucao.append(saldo)

    df_simulacao = df.iloc[1:len(saldo_evolucao)+1].copy()
    df_simulacao['saldo'] = saldo_evolucao[:len(df_simulacao)]

    return df_simulacao

def plot_lucro(df_sim: pd.DataFrame, nome_modelo: str, path: str = 'figures/') -> None:
    """
    Plota e salva a evolução do lucro para um modelo.
    Args:
        df_sim (pd.DataFrame): DataFrame com colunas 'date' e 'saldo'.
        nome_modelo (str): Nome do modelo (usado no gráfico e nome do arquivo).
        path (str): Diretório onde salvar o gráfico.
    """
    try:
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(df_sim['date'], df_sim['saldo'], label=f'{nome_modelo} - Lucro')
        plt.xlabel('Data')
        plt.ylabel('Saldo (USD)')
        plt.title(f'Evolução do Saldo - {nome_modelo}')
        plt.grid(True)
        plt.tight_layout()
        plt.legend()
        plt.savefig(f'{path}lucro_{nome_modelo}.png', dpi=150)
        plt.close()
        logging.info(f"Gráfico de lucro salvo: {path}lucro_{nome_modelo}.png")
    except Exception as e:
        logging.error(f"Erro ao salvar gráfico de lucro para {nome_modelo}: {e}")


def comparar_modelos(df_feat: pd.DataFrame, nome_cripto: str = 'ETH', grau_poli: int = 3, path: str = 'figures/') -> None:
    """
    Treina 3 modelos (MLP, Linear, Polinomial), simula investimentos e compara o saldo gerado.
    Args:
        df_feat (pd.DataFrame): DataFrame com as features da criptomoeda.
        nome_cripto (str): Nome da criptomoeda (usado no título).
        grau_poli (int): Grau da regressão polinomial.
        path (str): Caminho onde salvar o gráfico comparativo.
    """
    os.makedirs(path, exist_ok=True)
    X, y = split_features_labels(df_feat)

    modelos = {
        'MLP': 'mlp',
        'Linear': 'linear',
        f'Polinomial (grau {grau_poli})': 'poly'
    }

    preds_dict = {}
    saldos_dict = {}

    for nome_legivel, modelo_id in modelos.items():
        preds, _ = fit_and_predict_model(modelo_id, X, y, grau=grau_poli)
        preds_dict[nome_legivel] = preds
        df_sim = simulate_investment(df_feat, preds)
        saldos_dict[nome_legivel] = df_sim
        plot_lucro(df_sim, nome_legivel, path=path)

    # Gráfico comparativo
    plt.figure(figsize=(12, 5))
    for nome, df in saldos_dict.items():
        plt.plot(df['date'], df['saldo'], label=nome)
    plt.title(f'Evolução do Lucro - {nome_cripto}')
    plt.xlabel('Data')
    plt.ylabel('Saldo em U$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{path}comparacao_modelos_{nome_cripto}.png', dpi=150)
    plt.close()

    # Estatísticas
    plot_dispersao(y, preds_dict, path=path, nome_cripto=nome_cripto)
    corrs = calcular_coeficientes_correlacao(y, preds_dict)
    equacoes = {nome: obter_equacao_regressao(y, pred) for nome, pred in preds_dict.items()}
    erros_padrao = {nome: calcular_erro_padrao(y, pred) for nome, pred in preds_dict.items()}
    melhor = max(corrs, key=lambda k: corrs[k])
    erro_entre = erro_padrao_entre_modelos(preds_dict['MLP'], preds_dict[melhor])

    logging.info(f"Coeficientes de correlação: {corrs}")
    logging.info(f"Equações de regressão: {equacoes}")
    logging.info(f"Erro padrão dos modelos: {erros_padrao}")
    logging.info(f"Erro padrão entre MLP e {melhor}: {erro_entre:.4f}")

    # Salvar estatísticas em CSV
    estatisticas = pd.DataFrame({
        'Modelo': list(preds_dict.keys()),
        'Correlação': [corrs[nome] for nome in preds_dict],
        'Equação Regressão': [equacoes[nome] for nome in preds_dict],
        'Erro Padrão': [erros_padrao[nome] for nome in preds_dict]
    })

    estatisticas['Comparado com MLP'] = [
        erro_padrao_entre_modelos(preds_dict['MLP'], preds_dict[nome]) if nome != 'MLP' else None
        for nome in preds_dict
    ]

    estat_path = os.path.join(path, f'estatisticas_modelos_{nome_cripto}.csv')
    estatisticas.to_csv(estat_path, index=False, encoding='utf-8-sig')
    logging.info(f"Estatísticas salvas em: {estat_path}")