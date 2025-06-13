# Analise_RF_MLP_UNSW_NB15
Repositório contendo o código-fonte utilizado no Trabalho de Conclusão de Curso (TCC) intitulado “Análise das features do dataset IDS UNSW-NB15 para avaliar a viabilidade de Federação em Sistemas IDS baseados em Machine Learning treinados em outros datasets”. (Em desenvolvimento)

# Análise do Dataset UNSW-NB15 para Federação em Sistemas IDS com Machine Learning

Este repositório contém os códigos utilizados no Trabalho de Conclusão de Curso (TCC) apresentado ao curso de Ciência da Computação, cujo tema é:

**“Análise das features do dataset IDS UNSW-NB15 para avaliar a viabilidade de Federação em Sistemas IDS baseados em Machine Learning treinados em outros datasets”**

## Objetivo

O objetivo principal do projeto é analisar as características do dataset UNSW-NB15 e avaliar sua compatibilidade e potencial de interoperabilidade com outros datasets de Intrusion Detection Systems (IDS), como o NSL-KDD, no contexto de **Aprendizado Federado** com algoritmos de **Machine Learning**.

## Estrutura do Projeto

- `analise_precisao2_UNSW15`
  - Limpeza e normalização dos dados (MinMax e Z-Score).
  - Avaliação de desempenho dos modelos MLP e Random Forest, com e sem normalização, de identificar a classe de ataque cada registros.
- `Analise_binaria_UNSW15`
  - Mesmos processos da analise multiclass, mas avaliando a capacidade dos modelos de apenas distinguir os registros como ataque ou comportamento normal
- `Analise_PCA_UNSW15`
  - Script de aplicação do PCA e análise de importância das features do UNSW-NB15.
- `resultados/`
  - Resultados quantitativos e gráficos gerados durante os experimentos.

## Técnicas Utilizadas

- **Pré-processamento**:
  - Remoção de colunas/linhas vazias
  - Codificação de atributos categóricos (`LabelEncoder`)
  - Normalização com `MinMaxScaler` e `StandardScaler`

- **Modelos Avaliados**:
  - Multi-Layer Perceptron (MLP)
  - Random Forest

- **Redução de Dimensionalidade**:
  - PCA (Principal Component Analysis)
    - Análise da variância explicada
    - Contribuição de cada atributo
  - MDI (Mean Decrease Impurity)
    - Importância dos atributos segundo a Random Forest

- **Métricas de Avaliação**:
  - Acurácia
  - Precisão
  - Recall
  - F1-Score
  - AUC-ROC Score
  - Tempo de treino

## Resultados Obtidos

- Comparação do desempenho dos classificadores com e sem normalização
- Identificação de atributos mais relevantes para classificação
- Determinação do número ideal de componentes principais (via PCA)
- Verificação da viabilidade de uso do UNSW-NB15 em federação com outros datasets

## Requisitos

- Python 3.10+
- Bibliotecas:
  - pandas
  - numpy
  - scikit-learn
  - matplotlib

Você pode instalar todas com:

```bash
pip install -r requirements.txt
