# src/utils.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import logging
import scipy.stats as stats
from typing import Dict
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def resumo_estatistico(df: pd.DataFrame) -> pd.Series:
    """
    Calcula estatísticas descritivas básicas da coluna 'close'.
    Args:
        df (pd.DataFrame): DataFrame com a coluna 'close'.
    Returns:
        pd.Series: Série com média, mediana, desvio padrão, variância, mínimo, máximo, amplitude e coeficiente de variação.
    """
    return pd.Series({
        'média': df['close'].mean(),
        'mediana': df['close'].median(),
        'desvio padrão': df['close'].std(),
        'variância': df['close'].var(),
        'mínimo': df['close'].min(),
        'máximo': df['close'].max(),
        'amplitude': df['close'].max() - df['close'].min(),
        'coef_variação': df['close'].std() / df['close'].mean() if df['close'].mean() != 0 else 0
    })

def plot_boxplot(df: pd.DataFrame, nome: str, path: str = 'figures/') -> None:
    """
    Gera e salva um gráfico boxplot da coluna 'close'.
    Args:
        df (pd.DataFrame): DataFrame com a coluna 'close'.
        nome (str): Nome para compor o nome do arquivo.
        path (str): Caminho da pasta onde o gráfico será salvo.
    """
    try:
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df['close'])
        plt.title(f"Boxplot - {nome}")
        plt.savefig(f"{path}boxplot_{nome}.png", dpi=150)
        plt.show()
        plt.close()
        logging.info(f"Boxplot salvo: {path}boxplot_{nome}.png")
    except Exception as e:
        logging.error(f"Erro ao salvar boxplot de {nome}: {e}")

def gerar_tabela_estat(dfs_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Gera uma tabela com estatísticas descritivas de vários DataFrames.
    Args:
        dfs_dict (Dict[str, pd.DataFrame]): Dicionário de DataFrames com nomes como chaves.
    Returns:
        pd.DataFrame: DataFrame com estatísticas de todos os arquivos não vazios.
    """
    estatisticas = []

    for nome, df in dfs_dict.items():
        if df.empty:
            logging.warning(f"DataFrame vazio ignorado: {nome}")
            continue
        resumo = resumo_estatistico(df)
        resumo.name = nome.replace('.csv', '').replace('Bitstamp_', '').replace('_d', '')
        estatisticas.append(resumo)

    logging.info("Tabela de estatísticas gerada com sucesso.")
    return pd.DataFrame(estatisticas)

def rolling_mode(series: pd.Series, window: int) -> pd.Series:
    """
    Calcula a moda móvel de uma série temporal.
    Args:
        series (pd.Series): Série de valores.
        window (int): Tamanho da janela.
    Returns:
        pd.Series: Moda móvel.
    """
    return series.rolling(window).apply(
        lambda x: stats.mode(x, keepdims=False).mode if len(x.dropna()) > 0 else np.nan
    )

def plot_completo(df: pd.DataFrame, nome: str, window: int = 7, path: str = 'figures/') -> None:
    """
    Plota e salva o gráfico de preço de fechamento com média, mediana e moda móveis.
    Args:
        df (pd.DataFrame): DataFrame com as colunas 'date' e 'close'.
        nome (str): Nome da criptomoeda.
        window (int): Tamanho da janela móvel (padrão = 7).
        path (str): Caminho da pasta onde o gráfico será salvo.
    """
    try:
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(12, 5))
        plt.plot(df['date'], df['close'], label='Fechamento', color='blue', alpha=0.6)
        plt.plot(df['date'], df['close'].rolling(window).mean(), label='Média 7d', color='orange')
        plt.plot(df['date'], df['close'].rolling(window).median(), label='Mediana 7d', color='green', linestyle='--')
        plt.plot(df['date'], rolling_mode(df['close'], window), label='Moda 7d', color='red', linestyle=':')

        plt.title(f'{nome} - Fechamento, Média, Mediana e Moda (janela={window} dias)')
        plt.xlabel('Data')
        plt.ylabel('Preço de Fechamento')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        file_path = os.path.join(path, f"linha_{nome}.png")
        plt.savefig(file_path, dpi=150)
        plt.show()
        plt.close()
        logging.info(f"Gráfico de linha salvo: {file_path}")

    except Exception as e:
        logging.error(f"Erro ao salvar gráfico de linha de {nome}: {e}")

def plot_dispersao(y_true: np.ndarray, preds_dict: Dict[str, np.ndarray], path: str = 'figures/', nome_cripto: str = 'BTC') -> None:
    """
    Gera e salva um gráfico de dispersão entre valores reais e previstos de múltiplos modelos.
    Args:
        y_true (np.ndarray): Valores reais.
        preds_dict (Dict[str, np.ndarray]): Dicionário com previsões dos modelos.
        path (str): Caminho para salvar o gráfico.
        nome_cripto (str): Nome da criptomoeda (para nome do arquivo).
    """
    try:
        os.makedirs(path, exist_ok=True)
        plt.figure(figsize=(8, 6))
        for nome, preds in preds_dict.items():
            plt.scatter(y_true, preds, label=nome, alpha=0.6)
        plt.xlabel('Valores reais')
        plt.ylabel('Valores previstos')
        plt.title(f'Diagrama de Dispersão - {nome_cripto}')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(path, f'dispersao_{nome_cripto}.png'), dpi=150)
        plt.close()
        logging.info(f"Gráfico de dispersão salvo: {path}dispersao_{nome_cripto}.png")
    except Exception as e:
        logging.error(f"Erro ao plotar dispersão: {e}")


def calcular_coeficientes_correlacao(y_true: np.ndarray, preds_dict: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Calcula os coeficientes de correlação entre os valores reais e as previsões de cada modelo.
    Args:
        y_true (np.ndarray): Valores reais.
        preds_dict (Dict[str, np.ndarray]): Previsões dos modelos.
    Returns:
        Dict[str, float]: Dicionário com nome do modelo e seu coeficiente de correlação.
    """
    corrs = {}
    for nome, preds in preds_dict.items():
        corrs[nome] = np.corrcoef(y_true, preds)[0, 1]
    return corrs


def obter_equacao_regressao(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """
    Calcula a equação da reta de regressão linear (y = ax + b) com base nos valores reais e previstos.
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
    Returns:
        str: Equação formatada como string. Em caso de erro, retorna 'Erro'.
    """
    try:
        coef = np.polyfit(y_true, y_pred, deg=1)
        return f'y = {coef[0]:.4f}x + {coef[1]:.4f}'
    except Exception as e:
        logging.warning(f"Erro ao calcular equação de regressão: {e}")
        return 'Erro'


def calcular_erro_padrao(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calcula o erro padrão entre os valores reais e previstos.
    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
    Returns:
        float: Valor do erro padrão.
    """
    return float(np.std(y_true - y_pred))


def erro_padrao_entre_modelos(pred1: np.ndarray, pred2: np.ndarray) -> float:
    """
    Calcula o erro padrão entre dois vetores de previsões de modelos diferentes.
    Args:
        pred1 (np.ndarray): Previsões do primeiro modelo.
        pred2 (np.ndarray): Previsões do segundo modelo.
    Returns:
        float: Erro padrão entre as duas previsões.
    """
    return float(np.std(pred1 - pred2))

def testar_retorno_medio(df: pd.DataFrame, retorno_minimo: float, nome_cripto: str) -> dict:
    """
    Realiza um teste de hipótese para verificar se o retorno médio diário é >= retorno_minimo.
    Retorno mínimo é fornecido em percentual (ex: 0.1 para 0.1%).

    Args:
        df (pd.DataFrame): DataFrame com a coluna 'Close'.
        retorno_minimo (float): Valor de referência (%) definido pelo usuário.
        nome_cripto (str): Nome da criptomoeda para exibir no log.

    Returns:
        dict: Resultado com média, p-valor, decisão e estatística t.
    """
    try:
        df = df.copy()
        df['retorno'] = df['close'].pct_change()
        df = df.dropna()
        amostra = df['retorno']

        # valor de referência convertido para taxa (ex: 0.1% -> 0.001)
        x = retorno_minimo / 100


        result = stats.ttest_1samp(amostra, x)
        t_stat = getattr(result, 'statistic')
        p_value = getattr(result, 'pvalue')

        p_value_unilateral = p_value / 2 if t_stat < 0 else 1.0

        media = amostra.mean()

        decisao = 'Rejeita H₀' if p_value_unilateral < 0.05 else 'Não rejeita H₀'

        logging.info(f"{nome_cripto} | Média: {media:.6f} | t: {t_stat:.4f} | p unilateral: {p_value_unilateral:.4f} | {decisao}")

        lista_resultados = {'cripto': nome_cripto,
            'media': media,
            't_stat': t_stat,
            'p_value_unilateral': p_value_unilateral,
            'decisao': decisao }
    

        return lista_resultados

    except Exception as e:
        logging.error(f"Erro ao testar retorno de {nome_cripto}: {e}")
        return {'cripto': nome_cripto, 'erro': str(e)}

def anova_simples(df: pd.DataFrame) -> None:
    """
    Realiza ANOVA para comparar o retorno médio entre diferentes criptomoedas.
    Salva os resultados em CSV e executa post hoc se aplicável.
    Args:
        df (pd.DataFrame): DataFrame com colunas 'retorno' e 'cripto'.
    """
    modelo = ols('retorno ~ C(cripto)', data=df).fit()
    anova_resultado = anova_lm(modelo, typ=2)
    print("\n--- ANOVA entre criptomoedas ---")
    print(anova_resultado)

    anova_resultado.to_csv('anova_entre_criptos.csv', index=True, encoding='utf-8-sig')


    if anova_resultado['PR(>F)'][0] < 0.05:
        print("\n>>> Resultado significativo! Executando teste post hoc (Tukey HSD):")
        tukey = pairwise_tukeyhsd(endog=df['retorno'], groups=df['cripto'], alpha=0.05)
        print(tukey)

def agrupar_por_volatilidade(df: pd.DataFrame, quantis: List[float] = [0.33, 0.66]) -> pd.DataFrame:
    """
    Agrupa criptomoedas em categorias de volatilidade (Baixa, Média, Alta) com base em quantis.
    Args:
        df (pd.DataFrame): DataFrame com colunas 'cripto' e 'retorno'.
        quantis (List[float]): Lista com os dois quantis de corte.
    Returns:
        pd.DataFrame: DataFrame original com coluna extra 'grupo_volatilidade'.
    """
    vol_por_cripto = df.groupby('cripto')['retorno'].std().reset_index()
    vol_por_cripto.columns = ['cripto', 'volatilidade']
    quantil_1, quantil_2 = vol_por_cripto['volatilidade'].quantile(quantis).values

    def categorizar(vol):
        if vol <= quantil_1:
            return 'Baixa'
        elif vol <= quantil_2:
            return 'Média'
        else:
            return 'Alta'

    vol_por_cripto['grupo_volatilidade'] = vol_por_cripto['volatilidade'].apply(categorizar)
    return df.merge(vol_por_cripto[['cripto', 'grupo_volatilidade']], on='cripto')

def anova_por_grupo(df_com_grupo: pd.DataFrame) -> None:
    """
    Realiza ANOVA entre grupos de volatilidade das criptomoedas.
    Args:
        df_com_grupo (pd.DataFrame): DataFrame com colunas 'retorno' e 'grupo_volatilidade'.
    """
    modelo = ols('retorno ~ C(grupo_volatilidade)', data=df_com_grupo).fit()
    anova_resultado = anova_lm(modelo, typ=2)
    print("\n--- ANOVA por grupo de volatilidade ---")
    print(anova_resultado)

    anova_resultado.to_csv('anova_por_grupo_volatilidade.csv', index=True, encoding='utf-8-sig')

    if anova_resultado['PR(>F)'][0] < 0.05:
        print("\n>>> Resultado significativo! Executando teste post hoc (Tukey HSD):")
        tukey = pairwise_tukeyhsd(endog=df_com_grupo['retorno'], groups=df_com_grupo['grupo_volatilidade'], alpha=0.05)
        print(tukey)