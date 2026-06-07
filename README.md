# EP IA: MLP + CNN

Trabalho da disciplina **Inteligencia Artificial (ACH2016)**, USP EACH, 1o semestre de 2026.

## Integrantes

- Isabelle da Silva Tobias - NUSP 15525991 (T04)
- Kevin Rodrigues Nunes    - NUSP 15676030 (T94)
- Kauã Lima de Souza       - NUSP 15674702 (T94)
- Victor Yodono            - NUSP 13829040 (T94)

## Objetivos

1. **MLP (Multilayer Perceptron)** - Implementacao do zero (sem frameworks de redes neurais), treinada com Backpropagation em sua versao de Gradiente Descendente.
   - Datasets: portas logicas (OR, AND, XOR), CARACTERES_REDUZIDO, CARACTERES_COMPLETO.

2. **CNN (Convolutional Neural Network)** - (Em andamento)

## Estrutura

```
EP_IA_MLP_CNN/
├── src/
│   ├── mlp/                # Objetivo 1: MLP from scratch
│   │   ├── main.py         # run() + main() + entry point (orquestracao)
│   │   ├── entities.py     # Neuronio, Camada, MLP (forward, backprop)
│   │   ├── teste_de_mesa.py # conferencia com o exemplo numerico da professora
│   │   ├── datasets.py     # carregamento dos datasets
│   │   ├── saidas.py       # ResultadoExperimento + funcoes salvar_* (txt/csv/png)
│   │   ├── config.py       # hiperparametros por dataset
│   │   └── value_objects.py
│   └── cnn/                # Objetivo 2: CNN
├── data/                   # Datasets brutos
├── saidas/
│   ├── mlp/                # Pesos, erros, hiperparametros, matrizes de confusao (Obj 1)
│   └── cnn/                # (Obj 2)
├── docs/                   # Especificacao e exemplo numerico da professora
├── Makefile
├── pyproject.toml
└── README.md
```

## Setup

Usamos o [uv](https://docs.astral.sh/uv/) como gerenciador de pacotes (substitui pip + venv + pip-tools num comando so). Rode uma vez:

```bash
make install
```

Isso instala o uv (se precisar), cria o `.venv`, instala dependencias e configura os hooks de pre-commit.

## Como rodar

### Objetivo 1: MLP

```bash
make run-mlp
```

Roda primeiro o **teste de mesa** (conferencia do forward/backprop com o exemplo numerico de `docs/exemplo+numérico+MLP.pdf`) e depois treinamento + teste pros 5 datasets em sequencia (OR, AND, XOR, CARACTERES_REDUZIDO, CARACTERES_COMPLETO). Os hiperparametros de cada dataset estao em `src/mlp/config.py`.

Alternativamente, com o venv ativo:

```bash
source .venv/bin/activate
python src/mlp/main.py
```

### Objetivo 2: CNN

```bash
make run-cnn
```

### Limpar os arquivos de saida

```bash
make clean
```

## Arquivos de saida (por dataset)

Pra cada dataset, o MLP gera em `saidas/mlp/<dataset>/`:

| Arquivo | Conteudo |
|---|---|
| `hiperparametros.txt` | taxa, epocas, arquitetura, resultado final |
| `pesos_iniciais.txt`  | snapshot dos pesos antes do treino |
| `pesos_finais.txt`    | pesos depois do treino |
| `erro_por_epoca.csv`  | MSE de cada epoca |
| `saidas_teste.csv`    | classe esperada vs classe predita por amostra |
| `mse.png`             | grafico de evolucao do MSE (treino + validacao) |
| `matriz_confusao.png` | matriz de confusao em heatmap |

## Decisoes de design

- **Ativacao**: sigmoide (`1 / (1 + e^(-x))`), seguindo o exemplo numerico da professora.
- **Inicializacao dos pesos**: Xavier/Glorot uniforme (`L = sqrt(6/(fan_in+fan_out))`). Motivo: pesos muito grandes saturam a sigmoide, muito pequenos colapsam o sinal.
- **Nomenclatura**: segue o livro Fausett (pagina 294): `v_ij`, `w_jk`, `z_in_j`, `z_j`, `y_in_k`, `y_k`, `t_k`, `alpha`, `delta`.
- **Split dos dados**:
  - OR/AND/XOR: as 4 amostras em treino/validacao/teste (dataset pequeno demais pra dividir).
  - CARACTERES_REDUZIDO: pool dos 3 arquivos (limpo + ruido + ruido20), shuffle, split 80/10/10.
  - CARACTERES_COMPLETO: shuffle previo aleatorio, split 80/10/10.
