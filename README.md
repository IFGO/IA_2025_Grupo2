# crypto_pred
Previsão do Preço de Fechamento de Criptomoeda

Este projeto tem como objetivo analisar, modelar e prever preços de criptomoedas usando técnicas de machine learning, estatística e testes de hipótese. Ele foi desenvolvido como trabalho final da disciplina de Inteligência Artificial Aplicada.

---

## 📁 Estrutura de Pastas

```shell
crypto_pred/
├── data/ # Arquivos CSV das criptomoedas
├── figuras/ # Gráficos e imagens geradas
├── estatisticas/ # Resultados salvos
├── src/ # Códigos-fonte modulares
│ ├── data_load.py
│ ├── features.py
│ ├── models.py
│ ├── evaluation.py
│ └── utils.py
├── tests/ # Testes unitários (pytest)
│ ├── test_data_load.py
│ ├── test_features.py
│ └── test_models.py
├── main.py # Script principal
├── notebook.ipynb # Análises exploratórias (EDA e testes)
├── requirements.txt # Dependências do projeto
└── README.md # Este arquivo
```

## Como Executar

### 1. Clone o projeto (ou baixe os arquivos)

```bash
git clone [https://github.com/rpa1tera/crypto_pred.git]
```

### 2. Crie um ambiente virtual
```bash
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows
```

### 3. Instale as dependências
```bash
pip install -r requirements.txt
```

### 4. Treinar modelo e validar

```bash
python main.py --crypto Bitstamp_ETHUSD_d.csv --model linear
```

ou

```bash
python main.py --crypto Bitstamp_ETHUSD_d.csv --model mpl
```

ou 

```bash
python main.py --crypto Bitstamp_ETHUSD_d.csv --model poly
```

### 5. Comparar todos os modelos

```bash
python main.py --crypto Bitstamp_ETHUSD_d.csv --model all
```

### 6. Teste de hipótese de retorno médio

```bash
python main.py --testar-retorno --retorno-minimo 0.05
```

### 7. Análise de variância (ANOVA)

```bash
python main.py --anova
```

### 8. Testes Automatizados

Para rodar os testes unitários: 

```bash
pytest --cov=src --cov-report=term-missing tests/
```

### 9. Resultados Gerados

*figures/: gráficos comparativos, boxplots, dispersão, etc.

*teste_hipotese_retorno.csv: resultado dos testes de hipótese para cada criptomoeda

*anova_entre_criptos.csv e anova_por_grupo_volatilidade.csv: resultados da ANOVA simples e agrupada

### 10. Modelos Implementados

*MLPRegressor
*Regressão Linear
*Regressão Polinomial (com Ridge)

### 11. Autor(a)

Desenvolvido por Raquel Pereira de Alcântara como parte do trabalho final da Pós em IA Aplicada – Instituto Federal de Goiás (IFG).
=======
Previsão do Preço de Fechamento de Criptomoeda
>>>>>>> 7147f353955cba7545736bb36408874e7f0ed48d
