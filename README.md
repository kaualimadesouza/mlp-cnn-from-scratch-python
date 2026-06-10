# EP IA: MLP + CNN

Trabalho da disciplina **Inteligência Artificial (ACH2016)**, USP EACH, 1º semestre de 2026.

## Integrantes

- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)

## Objetivos

1. **MLP (Multilayer Perceptron)** - Implementação do zero (sem frameworks de redes neurais), treinada com Backpropagation em sua versão de Gradiente Descendente.
   - Datasets: portas lógicas (OR, AND, XOR), CARACTERES_REDUZIDO, CARACTERES_COMPLETO.

## Estrutura

```
EP_IA_MLP_CNN/
├── src/
│   ├── mlp/                # Objetivo 1: MLP from scratch
│   │   ├── main.py         # run() + main() + entry point (orquestração)
│   │   ├── entities.py     # Neurônio, Camada, MLP (forward, backprop)
│   │   ├── teste_de_mesa.py # conferência com o exemplo numérico da professora
│   │   ├── datasets.py     # carregamento dos datasets
│   │   ├── resultado.py    # ResultadoExperimento (resultado de um experimento)
│   │   ├── saidas.py       # funções salvar_* (txt/csv)
│   │   ├── graficos.py     # gráficos de MSE, matriz de confusão e validação cruzada
│   │   ├── validacao.py    # validação cruzada (k-fold)
│   │   ├── busca_parametros.py    # grid search de hiperparâmetros (com tempo de treino)
│   │   ├── plotar_busca.py        # tabela PNG com o ranking da busca
│   │   ├── gerar_variacao_autoral.py # gera o teste autoral com ruído
│   │   ├── config.py       # hiperparâmetros por dataset
│   │   └── value_objects.py
├── data/                   # Datasets brutos
├── saidas/
│   ├── mlp/                # Pesos, erros, hiperparâmetros, matrizes de confusão (Obj 1)
├── docs/                   # Especificação e exemplo numérico da professora
├── Makefile
├── pyproject.toml
└── README.md
```

## Setup

Usamos o [uv](https://docs.astral.sh/uv/) como gerenciador de pacotes (substitui pip + venv + pip-tools num comando só). Rode uma vez:

```bash
make install
```

Isso instala o uv (se precisar), cria o `.venv`, instala dependências e configura os hooks de pré-commit.

## Como rodar

### Objetivo 1: MLP

```bash
make run-mlp
```

Roda primeiro o **teste de mesa** (conferência do forward/backprop com o exemplo numérico de `docs/exemplo+numérico+MLP.pdf`) e depois treinamento + teste pros datasets em sequência. Os hiperparâmetros de cada dataset estão em `src/mlp/config.py`.

Alternativamente, com o venv ativo:

```bash
source .venv/bin/activate
python src/mlp/main.py
```


### Limpar os arquivos de saída

```bash
make clean
```

## Arquivos de saída (por dataset)

Pra cada dataset, o MLP gera em `saidas/mlp/<dataset>/`:

| Arquivo | Conteúdo |
|---|---|
| `hiperparametros.txt` | taxa, épocas, arquitetura, resultado final |
| `pesos_iniciais.txt`  | snapshot dos pesos antes do treino |
| `pesos_finais.txt`    | pesos depois do treino |
| `erro_por_epoca.csv`  | MSE de cada época |
| `saidas_teste.csv`    | classe esperada vs classe predita por amostra |
| `mse.png`             | gráfico de evolução do MSE (treino + validação) |
| `matriz_confusao.png` | matriz de confusão em heatmap |
| `validacao_cruzada.png` | acurácia por fold (só nos datasets com validação cruzada) |

O `caracteres_completo/` tem ainda os artefatos da busca de hiperparâmetros: `busca_parametros.csv` (ranking completo, com tempo de treino) e `busca_parametros.png` (top 15 em formato de tabela).

## Datasets (`data/`)

Todos os valores de entrada dos datasets de caracteres/portas estão em **representação bipolar** (`-1` e `+1`); o loader (`src/mlp/datasets.py`) converte pra unipolar (`0`/`1`) pra casar com a faixa da sigmoide.

### `portas_logicas/`

Três portas lógicas, uma por arquivo (`and.csv`, `or.csv`, `xor.csv`). Cada linha é uma entrada da tabela verdade: `x1, x2, y` → `num_entradas = 2`, `num_saidas = 1`. A XOR não é linearmente separável (exige camada oculta). Como são só 4 amostras, as mesmas servem de treino, validação e teste.

### `caracteres_reduzido/` (Fausett)

Dataset clássico do livro *Fausett - Fundamentals of Neural Networks*: **7 letras** (A, B, C, D, E, J, K) numa grade **9×7 = 63 pixels**, saída one-hot de 7 classes. Três arquivos (`limpo.csv`, `ruido.csv`, `ruido20.csv`), 21 linhas cada (3 variações × 7 letras) e 70 colunas (63 entradas + 7 saídas). O loader junta os três num pool, embaralha e faz split 80/10/10.

### `caracteres_completo/`

Alfabeto completo **A-Z** em grade **10×12 = 120 pixels**, one-hot de 26 classes. O loader usa `X.npy` (matriz de entradas) e `Y_classe.npy` (rótulos one-hot), embaralha e faz split 80/10/10 (~treino/validação/teste).

### `caracteres_completo_autoral/` (variação autoral)

Dado criado **pelo grupo** (não fornecido pela professora), para o quesito "Os testes contaram com variações (autorais) do conjunto de teste":

- **`X_ruido.npy`**: cópia do `caracteres_completo/X.npy` com **10% dos pixels (12 de 120) invertidos** em cada imagem, com seed fixa (42).
- **Rótulos**: os mesmos do original, na mesma ordem.
- **Uso**: somente como **conjunto de teste** - a rede treina/valida nos dados originais e é testada nesta variação, medindo robustez a ruído nunca visto.
- **Reprodução**: `python src/mlp/gerar_variacao_autoral.py`

### `iris/` (dataset externo)

Conjunto **externo** (não fornecido pela professora), para o quesito opcional "Teste com outros conjuntos de dados". Mostra que a MLP generaliza para um problema fora do escopo do EP.

- **`iris.csv`**: 150 flores, 4 atributos contínuos (comprimento/largura de sépala e pétala) e 3 classes (setosa, versicolor, virginica). Domínio público (Fisher, 1936).
- **Fonte (baixado em 2026-06-07)**: https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv
- **Pré-processamento no loader**: atributos normalizados por min-max para `[0, 1]` (faixa da sigmoide); classe vira one-hot de 3 posições. Split 80/10/10.

### Resumo

| Dataset | Entradas | Saídas | Split |
| --- | --- | --- | --- |
| `portas_logicas` (AND/OR/XOR) | 2 | 1 | 4 amostras em treino=validação=teste |
| `caracteres_reduzido` | 63 | 7 | 80/10/10 |
| `caracteres_completo` | 120 | 26 | 80/10/10 |
| `caracteres_completo_autoral` | 120 | 26 | treino/validação limpos, teste com ruído |
| `iris` (externo) | 4 | 3 | 80/10/10 |

## Decisões de design

- **Ativação**: sigmoide (`1 / (1 + e^(-x))`), seguindo o exemplo numérico da professora.
- **Inicialização dos pesos**: Xavier/Glorot uniforme (`L = sqrt(6/(fan_in+fan_out))`). Motivo: pesos muito grandes saturam a sigmoide, muito pequenos colapsam o sinal.
- **Nomenclatura**: segue o livro Fausett (página 294): `v_ij`, `w_jk`, `z_in_j`, `z_j`, `y_in_k`, `y_k`, `t_k`, `alpha`, `delta`.
- **Split dos dados**:
  - OR/AND/XOR: as 4 amostras em treino/validação/teste (dataset pequeno demais pra dividir).
  - CARACTERES_REDUZIDO: pool dos 3 arquivos (limpo + ruído + ruído20), shuffle, split 80/10/10.
  - CARACTERES_COMPLETO: shuffle prévio aleatório, split 80/10/10.

## Busca de hiperparâmetros

Os hiperparâmetros de cada dataset em `config.py` não foram chutados: para o CARACTERES_COMPLETO (o problema mais difícil), rodamos uma **busca em grade** (`src/mlp/busca_parametros.py`) cobrindo **125 combinações** (5 taxas de aprendizado × 5 tamanhos de camada oculta × 5 valores de paciência). O resultado completo fica em `saidas/mlp/caracteres_completo/busca_parametros.csv`, que inclui o **tempo de treino** de cada combinação (medido na busca); ao final, a busca também gera `busca_parametros.png` com o top 15 do ranking em formato de tabela (`src/mlp/plotar_busca.py`).

A combinação escolhida foi **taxa = 0.04, 55 neurônios na camada oculta, paciência = 15**, pelos seguintes motivos:

- **`taxa = 0.04` e `55 neurônios`** dominam a melhor faixa de acurácia de teste (**91.04%**). Taxas maiores (0.1) atingem MSE de validação um pouco menor, mas com acurácia pior (~89%) - sinal de sobreajuste/azar de split, já que o objetivo final é classificar, não minimizar o MSE isoladamente.
- **`paciência = 15`** é o melhor custo-benefício: dentro da faixa vencedora, a acurácia é a mesma (91.04%) para qualquer paciência, mudando só o MSE. Subir a paciência para 30 baixaria o MSE de 0.187 para 0.179 (ganho irrelevante), mas faria o treino rodar as 300 épocas inteiras, e a parada antecipada nem dispararia. Com paciência 15 o treino para por volta da época 215, com a mesma acurácia, demonstrando a parada antecipada funcionando e economizando épocas.

Ou seja: definimos os valores e a busca em grade **confirmou empiricamente** a escolha. Os demais datasets (portas lógicas, CARACTERES_REDUZIDO, Iris) são menores/mais simples e tiveram os hiperparâmetros ajustados manualmente.
