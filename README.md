# Churn Prediction - Clientes

Este projeto utiliza Machine Learning para prever a probabilidade de churn (cancelamento/saída) de clientes com base em variáveis demográficas e comportamentais.

## Objetivo

O objetivo principal é:

- carregar e analisar os dados de clientes;
- treinar um modelo de classificação para prever churn;
- avaliar o desempenho do modelo usando métricas como acurácia, matriz de confusão e relatório de classificação;
- identificar quais variáveis têm maior influência na previsão.

## Tecnologias utilizadas

- Python
- Pandas
- Scikit-learn
- Plotly

## Estrutura do projeto

```text
churn/
├── churn_clientes.csv
├── main.py
├── requirements.txt
├── README.md
└── .venv/
```

## Dataset

O arquivo `churn_clientes.csv` contém dados de clientes com as seguintes colunas:

- `idade`
- `tempo_cliente_meses`
- `gasto_mensal`
- `reclamacoes`
- `uso_suporte`
- `churn` (variável alvo: 0 = não churn, 1 = churn)

## Requisitos

Antes de executar o projeto, certifique-se de ter instalado:

- Python 3.9+
- pip

## Instalação

1. Clone ou baixe este projeto.
2. Crie e ative um ambiente virtual:

```bash
python -m venv .venv
```

No Windows:

```bash
.venv\Scripts\activate
```

No Linux/macOS:

```bash
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

## Execução

Para rodar o projeto:

```bash
python main.py
```

O script executa as seguintes etapas:

1. Carrega os dados a partir de `churn_clientes.csv`.
2. Separa as variáveis explicativas (`X`) e a variável alvo (`y`).
3. Divide os dados em treino e teste com `train_test_split`.
4. Treina um modelo `RandomForestClassifier`.
5. Exibe:
   - acurácia no treino;
   - acurácia no teste;
   - matriz de confusão;
   - relatório de classificação;
6. Gera um gráfico de importância das variáveis com Plotly.

## Métricas e resultados

O modelo utiliza métricas padrão de classificação para avaliar desempenho, incluindo:

- `accuracy_score`
- `confusion_matrix`
- `classification_report`

Essas métricas ajudam a entender a qualidade da previsão e a proporção de clientes corretamente classificados.

## Observações

- O modelo foi configurado com `RandomForestClassifier` e `random_state=42` para reprodutibilidade.
- A visualização final mostra a importância relativa de cada variável para a previsão de churn.
- O arquivo `requirements.txt` contém as bibliotecas necessárias para a execução do projeto.

## Licença

Este projeto foi desenvolvido para fins acadêmicos e de estudo.
