# Iris (dataset externo)

Conjunto de dados **externo** (não fornecido pela professora), para o quesito
opcional "Teste com outros conjuntos de dados". Mostra que a nossa MLP
generaliza para um problema de classificação fora do escopo do EP.

- **`iris.csv`**: 150 flores, 4 atributos contínuos (comprimento/largura de
  sépala e pétala) e 3 classes (setosa, versicolor, virginica). Domínio público
  (Fisher, 1936).
- **Fonte (baixado em 2026-06-07)**:
  https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
- **Pré-processamento no loader** (`src/mlp/datasets.py`): atributos
  normalizados por min-max para [0, 1] (faixa da sigmoide); classe vira one-hot
  de 3 posições. Split 80/10/10 com shuffle.

## Integrantes

- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)
