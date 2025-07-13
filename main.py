import sys
sys.path.append('src')

import argparse
import os
import logging
import pandas as pd
from data_load import load_crypto_file, load_all_cryptos
from features import engineer_features
from models import split_features_labels, train_model
from evaluation import comparar_modelos
from utils import testar_retorno_medio, anova_simples, agrupar_por_volatilidade, anova_por_grupo


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main() -> None:
    """
    Função principal para execução do pipeline de análise de criptomoedas.
    Permite rodar modelagem preditiva, testes de hipótese e ANOVA via argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(description="Treinamento de modelos para previsão de preços de criptomoedas")
    parser.add_argument('--crypto', type=str, help='Nome do arquivo da criptomoeda, ex: Bitstamp_BTCUSD_d.csv')
    parser.add_argument('--model', type=str, choices=['mlp', 'linear', 'poly', 'all'], default='all', help='Modelo a ser treinado')
    parser.add_argument('--kfolds', type=int, default=5, help='Número de folds para validação cruzada')
    parser.add_argument('--grau', type=int, default=3, help='Grau do polinômio para regressão polinomial')

    parser.add_argument('--testar-retorno', action='store_true', help='Executa o teste de hipótese de retorno médio')
    parser.add_argument('--retorno-minimo', type=float, default=0.05, help='Valor mínimo de retorno esperado (em percentual)')
    parser.add_argument('--anova', action='store_true', help='Executa a análise ANOVA dos retornos médios das criptomoedas')

    args = parser.parse_args()

    # Modelagem
    if args.crypto:
        logging.info(f"Carregando dados de: {args.crypto}")
        df = load_crypto_file(os.path.join('data', args.crypto))
        if df.empty:
            logging.error("Erro: arquivo não encontrado ou vazio.")
            return

        logging.info("Engenharia de features...")
        df_feat = engineer_features(df)
        X, y = split_features_labels(df_feat)

        if args.model in ['mlp', 'linear', 'poly']:
            logging.info(f"Treinando modelo {args.model.upper()}...")
            resultados = train_model(args.model, X, y, grau=args.grau, k=args.kfolds)
            logging.info(f"Resultados {args.model.upper()}: {resultados}")

        elif args.model == 'all':
            logging.info("Executando comparação entre todos os modelos...")
            nome = args.crypto.replace('.csv', '').replace('Bitstamp_', '').replace('_d', '')
            comparar_modelos(df_feat, nome_cripto=nome, grau_poli=args.grau)

    # Teste de hipótese para retorno médio
    if args.testar_retorno:
        logging.info("Executando teste de hipótese para retorno médio...")

        pasta_dados = 'data'
        resultados = []
        arquivos = [arq for arq in os.listdir(pasta_dados) if arq.endswith('.csv')]

        for arquivo in arquivos:
            caminho = os.path.join(pasta_dados, arquivo)
            df = load_crypto_file(caminho)
            if df.empty:
                logging.warning(f"Arquivo {arquivo} está vazio ou inválido. Pulando...")
                continue

            nome = arquivo.replace('.csv', '').replace('Bitstamp_', '').replace('_d', '')
            resultado = testar_retorno_medio(df, retorno_minimo=args.retorno_minimo, nome_cripto=nome)
            if 'erro' not in resultado:
                resultados.append(resultado)

        if resultados:
            pd.DataFrame(resultados).to_csv('teste_hipotese_retorno.csv', index=False, encoding='utf-8-sig')
            logging.info("Resultados do teste de hipótese salvos em teste_hipotese_retorno.csv")

        return

    # Análise ANOVA
    if args.anova:
        logging.info("Executando análise ANOVA dos retornos médios das criptomoedas...")

        pasta_dados = 'data'
        arquivos = [os.path.join(pasta_dados, f) for f in os.listdir(pasta_dados) if f.endswith('.csv')]
        dfs_dict = load_all_cryptos(arquivos)

        df_tudo = []
        for nome, df in dfs_dict.items():
            if df.empty:
                continue
            nome_cripto = nome.replace('.csv', '').replace('Bitstamp_', '').replace('_d', '')
            df['cripto'] = nome_cripto
            df['retorno'] = df['close'].pct_change()
            df_tudo.append(df[['cripto', 'retorno']].dropna())
        df_tudo = pd.concat(df_tudo, ignore_index=True)

        anova_simples(df_tudo)
        df_agrupado = agrupar_por_volatilidade(df_tudo)
        anova_por_grupo(df_agrupado)

        return

if __name__ == '__main__':  # type: ignore
    main()